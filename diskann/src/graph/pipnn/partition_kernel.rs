/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Nearest-leader selection for PiPNN partition assignment.
//!
//! The input contains a row-major point-to-leader dot matrix and
//! metric-specific [`PartitionScales`]. The output contains sorted leader-column
//! positions for each point. Its width sets the fanout and cannot exceed the
//! leader count.
//!
//! The caller supplies concrete architecture `A` and metric `M`. The function
//! checks row counts, scale units, scale lengths, fanout, and leader-ID range.
//! These checks occur before output changes or unchecked SIMD loads.
//!
//! L2 omits the point norm because it is constant for one point. Strict
//! comparisons keep leader scan order for equal scores. They do not rank NaN.
//! One runtime-sized workspace tracks the nearest leaders for each point.

use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;
use diskann_wide::{
    Architecture, Const, SIMDFloat, SIMDMask, SIMDPartialOrd, SIMDSelect, SIMDVector,
};

use super::kernel_metric::{KernelMetric, ScaleKind};

/// Reusable nearest-leader tracker for one partition worker.
#[derive(Debug, Default)]
pub struct PartitionKernelWorkspace {
    tracker: Vec<(u32, f32)>,
}

impl PartitionKernelWorkspace {
    fn prepare(&mut self, fanout: usize) -> Result<(), PartitionKernelError> {
        let additional = fanout.saturating_sub(self.tracker.len());
        self.tracker
            .try_reserve(additional)
            .map_err(|_| PartitionKernelError::Allocation { additional })?;
        self.tracker.resize(fanout, (u32::MAX, f32::INFINITY));
        Ok(())
    }
}

/// Metric-specific norm inputs for one partition tile.
///
/// The kernel checks each slice length before it changes output. Cosine point
/// values are squared norms from a matrix diagonal. Cosine leader values are
/// norms that partition setup computes once.
#[derive(Clone, Copy, Debug)]
pub enum PartitionScales<'a> {
    /// L2 needs only squared leader norms; the point norm cannot affect ranking.
    L2 {
        /// Squared norm for every leader column.
        leader_squared_norms: &'a [f32],
    },
    /// Unnormalized cosine needs squared point norms and leader norms.
    Cosine {
        /// Squared norm for every point.
        point_squared_norms: &'a [f32],
        /// Norm for every leader column.
        leader_norms: &'a [f32],
    },
    /// Normalized cosine and inner product need no normalization inputs.
    None,
}

/// One row-major point-to-leader dot-product tile.
///
/// Rows are points and columns are leaders. [`Self::scales`] must match concrete
/// metric `M`. This value borrows all input and stores no kernel state.
#[derive(Clone, Copy, Debug)]
pub struct PartitionInput<'a> {
    /// One point per matrix row and one leader per column.
    pub dots: MatrixView<'a, f32>,
    /// Normalization inputs matching concrete metric `M`.
    pub scales: PartitionScales<'a>,
}

/// Validation error returned by [`nearest_leaders`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PartitionKernelError {
    /// The output matrix does not match the input row count.
    #[error(
        "invalid output shape: expected {expected_rows} rows, got {actual_rows} rows and {actual_cols} columns"
    )]
    InvalidOutputShape {
        /// Required row count.
        expected_rows: usize,
        /// Supplied row count.
        actual_rows: usize,
        /// Supplied column count.
        actual_cols: usize,
    },
    /// A metric-specific scale slice has the wrong length.
    #[error("invalid {buffer} length: expected {expected}, got {actual}")]
    InvalidBufferLength {
        /// Name of the invalid scale buffer.
        buffer: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// Scale inputs do not match concrete metric `M`.
    #[error("partition scales do not match selected {expected} metric")]
    InvalidScales {
        /// Expected scale layout.
        expected: &'static str,
    },
    /// The requested fanout exceeds the available leader count.
    #[error("invalid fanout {fanout}: must not exceed {leader_count} leaders")]
    InvalidFanout {
        /// Requested number of leaders per point.
        fanout: usize,
        /// Available leader count.
        leader_count: usize,
    },
    /// Reusable tracker storage could not be reserved.
    #[error("failed to reserve {additional} partition tracker entries")]
    Allocation {
        /// Additional entries requested from the allocator.
        additional: usize,
    },
    /// Leader positions cannot be represented as `u32`.
    #[error("leader count {0} exceeds the u32 position limit")]
    TooManyLeaders(usize),
    /// A point did not contain enough rankable leaders to fill its output.
    #[error("point {point} has fewer than {fanout} rankable leaders")]
    InsufficientRankableLeaders {
        /// Zero-based point position in the input tile.
        point: usize,
        /// Requested number of leader positions.
        fanout: usize,
    },
}

/// Select the nearest leader positions for each input point.
///
/// `output.nrows()` must equal `input.dots.nrows()`. `output.ncols()` sets the
/// fanout and must not exceed the leader count.
pub(crate) fn nearest_leaders<A, M>(
    arch: A,
    input: PartitionInput<'_>,
    output: MutMatrixView<'_, u32>,
    workspace: &mut PartitionKernelWorkspace,
) -> Result<(), PartitionKernelError>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: KernelMetric,
{
    let scales = validate::<M>(input, &output)?;
    let fanout = output.ncols();
    if fanout == 0 || input.dots.nrows() == 0 {
        return Ok(());
    }

    workspace.prepare(fanout)?;
    process_points::<A::f32x16, M>(arch, input.dots, scales, output, &mut workspace.tracker)
}

/// Checked norm slices in the storage form that `M` requires.
///
/// A metric that does not use a norm receives an empty slice. Associated
/// `ScaleKind` constants remove these branches from concrete metric loops.
#[derive(Clone, Copy)]
struct ScaleSlices<'a> {
    point_scales: &'a [f32],
    leader_scales: &'a [f32],
}

/// Check the safety and metric conditions for partition selection.
///
/// The matrix views already prove their backing lengths. This function checks
/// row counts, leader-ID range, fanout, scale variant, and scale lengths. A
/// successful result contains norm slices in the units that `M` requires.
fn validate<'a, M: KernelMetric>(
    input: PartitionInput<'a>,
    output: &MutMatrixView<'_, u32>,
) -> Result<ScaleSlices<'a>, PartitionKernelError> {
    let point_count = input.dots.nrows();
    let leader_count = input.dots.ncols();
    let fanout = output.ncols();

    if output.nrows() != point_count {
        return Err(PartitionKernelError::InvalidOutputShape {
            expected_rows: point_count,
            actual_rows: output.nrows(),
            actual_cols: output.ncols(),
        });
    }
    if leader_count > u32::MAX as usize {
        return Err(PartitionKernelError::TooManyLeaders(leader_count));
    }
    if fanout > leader_count {
        return Err(PartitionKernelError::InvalidFanout {
            fanout,
            leader_count,
        });
    }

    // Match the scale variant to concrete metric `M` before extracting slices.
    // This prevents use of a squared point norm as a leader norm.
    let scales = match (M::METRIC, input.scales) {
        (
            Metric::L2,
            PartitionScales::L2 {
                leader_squared_norms,
            },
        ) => ScaleSlices {
            point_scales: &[],
            leader_scales: leader_squared_norms,
        },
        (
            Metric::Cosine,
            PartitionScales::Cosine {
                point_squared_norms,
                leader_norms,
            },
        ) => ScaleSlices {
            point_scales: point_squared_norms,
            leader_scales: leader_norms,
        },
        (Metric::CosineNormalized | Metric::InnerProduct, PartitionScales::None) => ScaleSlices {
            point_scales: &[],
            leader_scales: &[],
        },
        (Metric::L2, _) => return Err(PartitionKernelError::InvalidScales { expected: "L2" }),
        (Metric::Cosine, _) => {
            return Err(PartitionKernelError::InvalidScales { expected: "cosine" });
        }
        (Metric::CosineNormalized, _) => {
            return Err(PartitionKernelError::InvalidScales {
                expected: "normalized cosine",
            });
        }
        (Metric::InnerProduct, _) => {
            return Err(PartitionKernelError::InvalidScales {
                expected: "inner product",
            });
        }
    };

    // The associated scale kinds define the exact slice lengths. A metric that
    // does not use a scale must receive an empty slice.
    check_length(
        "point scales",
        scales.point_scales.len(),
        expected_scale_len(M::PARTITION_POINT_SCALE, point_count),
    )?;
    check_length(
        "leader scales",
        scales.leader_scales.len(),
        expected_scale_len(M::PARTITION_LEADER_SCALE, leader_count),
    )?;
    Ok(scales)
}

/// Return the required norm-slice length for one concrete metric.
const fn expected_scale_len(kind: ScaleKind, count: usize) -> usize {
    if kind.is_some() { count } else { 0 }
}

fn check_length(
    buffer: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), PartitionKernelError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PartitionKernelError::InvalidBufferLength {
            buffer,
            expected,
            actual,
        })
    }
}

/// Convert each point's leader scores into sorted leader IDs.
///
/// For each point, the function does these steps:
///
/// 1. Convert the point norm to the unit that `M` requires.
/// 2. Process complete SIMD groups.
/// 3. Process the scalar tail.
/// 4. Check that the tracker is full.
/// 5. Copy the sorted leader IDs to the output row.
///
/// `tracker` has `fanout` entries and stays sorted after each insertion. Strict
/// comparisons keep leader scan order for equal scores. They do not rank NaN.
///
/// The caller allocates the runtime-sized tracker once and reuses it for all
/// point rows.
fn process_points<F, M>(
    arch: F::Arch,
    dots: MatrixView<'_, f32>,
    scales: ScaleSlices<'_>,
    mut output: MutMatrixView<'_, u32>,
    tracker: &mut [(u32, f32)],
) -> Result<(), PartitionKernelError>
where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let leader_count = dots.ncols();
    let fanout = output.ncols();
    // Reset the tracker for each point. No assignment state can pass from one
    // output row to another.
    for (point, (point_dots, point_output)) in dots
        .row_iter()
        .zip(output.as_mut_slice().chunks_exact_mut(fanout))
        .enumerate()
    {
        tracker.fill((u32::MAX, f32::INFINITY));
        // Convert the point norm once for this row. Metrics without a point norm
        // use zero.
        let point_scale = if M::PARTITION_POINT_SCALE.is_some() {
            M::PARTITION_POINT_SCALE.transform(scales.point_scales[point])
        } else {
            0.0
        };
        let point_scale_vector = F::splat(arch, point_scale);
        // Process all complete SIMD groups first. The scalar tail uses the
        // metric's scalar operation order.
        let full = leader_count / F::LANES * F::LANES;

        for base in (0..full).step_by(F::LANES) {
            // SAFETY: `base + F::LANES <= full <= point_dots.len()`.
            let point_dots = unsafe { F::load_simd(arch, point_dots.as_ptr().add(base)) };
            let leader_scales = if M::PARTITION_LEADER_SCALE.is_some() {
                // SAFETY: `validate` established one scale per leader, and
                // `base + F::LANES <= full <= leader_count`.
                unsafe { F::load_simd(arch, scales.leader_scales.as_ptr().add(base)) }
            } else {
                F::default(arch)
            };
            insert_leader_lanes(
                M::partition_distance(arch, point_dots, point_scale_vector, leader_scales),
                base,
                tracker,
            );
        }

        // Use scalar formulas for the tail. A padded SIMD load can read past the
        // norm slice and can change L2 rounding.
        for (leader, &dot) in point_dots.iter().enumerate().skip(full) {
            let leader_scale = if M::PARTITION_LEADER_SCALE.is_some() {
                M::PARTITION_LEADER_SCALE.transform(scales.leader_scales[leader])
            } else {
                0.0
            };
            insert_leader(
                tracker,
                leader as u32,
                M::partition_distance_scalar(dot, point_scale, leader_scale),
            );
        }
        if tracker[fanout - 1].0 == u32::MAX {
            return Err(PartitionKernelError::InsufficientRankableLeaders { point, fanout });
        }
        // Distances stay in the workspace. Partition construction needs only the
        // leader-column positions in nearest-first order.
        copy_leader_ids(tracker, point_output);
    }
    Ok(())
}

/// Insert competitive SIMD lanes in increasing leader order.
///
/// One broadcast comparison rejects a group that cannot improve the last slot.
/// Bit iteration proceeds from low lane to high lane. This order matches scalar
/// tie behavior for all SIMD widths.
///
/// `distances` starts at `first_leader`. `tracker` is sorted.
fn insert_leader_lanes<F>(distances: F, first_leader: usize, tracker: &mut [(u32, f32)])
where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDPartialOrd,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let threshold = F::splat(distances.arch(), tracker[tracker.len() - 1].1);
    let eligible = distances.lt_simd(threshold);
    if eligible.none() {
        return;
    }

    let values: [f32; 16] = distances.to_array();
    let mut lanes = u64::from(eligible.bitmask().to_underlying());
    while lanes != 0 {
        let lane = lanes.trailing_zeros() as usize;
        lanes &= lanes - 1;
        insert_leader(tracker, (first_leader + lane) as u32, values[lane]);
    }
}

/// Insert one better candidate into a sorted tracker.
///
/// The function replaces the last slot and moves the new value to the left.
/// Equal scores and NaN do not enter. Thus, scan order resolves equal scores.
/// The last slot is also the rejection threshold and underfill sentinel.
///
/// `tracker` must be sorted and non-empty. `leader` is a local column position.
/// One insertion moves at most `fanout - 1` entries.
#[inline(always)]
fn insert_leader(tracker: &mut [(u32, f32)], leader: u32, distance: f32) {
    let threshold = tracker.len() - 1;
    if distance.partial_cmp(&tracker[threshold].1) != Some(std::cmp::Ordering::Less) {
        return;
    }

    tracker[threshold] = (leader, distance);
    let mut slot = threshold;
    while slot > 0 && tracker[slot].1 < tracker[slot - 1].1 {
        tracker.swap(slot, slot - 1);
        slot -= 1;
    }
}

/// Copy the retained leader IDs to one output row.
///
/// `assignments.len()` equals the checked fanout. The tracker keeps its distances
/// for the underfill check.
fn copy_leader_ids(tracker: &[(u32, f32)], assignments: &mut [u32]) {
    for (destination, &(leader, _)) in assignments.iter_mut().zip(tracker) {
        *destination = leader;
    }
}

#[cfg(test)]
struct DispatchedPartitionCall<'a> {
    input: PartitionInput<'a>,
    output: MutMatrixView<'a, u32>,
    workspace: &'a mut PartitionKernelWorkspace,
}

#[cfg(test)]
struct DispatchPartitionForTest(Metric);

#[cfg(test)]
impl<A>
    diskann_wide::arch::Target1<A, Result<(), PartitionKernelError>, DispatchedPartitionCall<'_>>
    for DispatchPartitionForTest
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn run(self, arch: A, call: DispatchedPartitionCall<'_>) -> Result<(), PartitionKernelError> {
        use super::kernel_metric::{Cosine, CosineNormalized, InnerProduct, L2};

        match self.0 {
            Metric::L2 => nearest_leaders::<A, L2>(arch, call.input, call.output, call.workspace),
            Metric::Cosine => {
                nearest_leaders::<A, Cosine>(arch, call.input, call.output, call.workspace)
            }
            Metric::CosineNormalized => nearest_leaders::<A, CosineNormalized>(
                arch,
                call.input,
                call.output,
                call.workspace,
            ),
            Metric::InnerProduct => {
                nearest_leaders::<A, InnerProduct>(arch, call.input, call.output, call.workspace)
            }
        }
    }
}

#[cfg(test)]
fn dispatch_nearest_leaders(
    metric: Metric,
    input: PartitionInput<'_>,
    output: MutMatrixView<'_, u32>,
    workspace: &mut PartitionKernelWorkspace,
) -> Result<(), PartitionKernelError> {
    diskann_wide::arch::dispatch1_no_features(
        DispatchPartitionForTest(metric),
        DispatchedPartitionCall {
            input,
            output,
            workspace,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::super::kernel_metric::{Cosine, CosineNormalized, InnerProduct, KernelMetric, L2};

    use super::*;

    fn test_input<'a>(
        metric: Metric,
        dots: &'a [f32],
        point_count: usize,
        leader_count: usize,
        point_scales: &'a [f32],
        leader_scales: &'a [f32],
    ) -> PartitionInput<'a> {
        let scales = match metric {
            Metric::L2 => PartitionScales::L2 {
                leader_squared_norms: leader_scales,
            },
            Metric::Cosine => PartitionScales::Cosine {
                point_squared_norms: point_scales,
                leader_norms: leader_scales,
            },
            Metric::CosineNormalized | Metric::InnerProduct => PartitionScales::None,
        };
        PartitionInput {
            dots: MatrixView::try_from(dots, point_count, leader_count).unwrap(),
            scales,
        }
    }

    // This oracle checks SIMD chunking, scalar tails, and tracker order. It uses
    // `M::partition_distance_scalar`. Separate tests define each formula directly.
    fn scalar_traversal_reference<M: KernelMetric>(
        input: PartitionInput<'_>,
        fanout: usize,
        output: &mut [u32],
    ) {
        let scales = match input.scales {
            PartitionScales::L2 {
                leader_squared_norms,
            } => ScaleSlices {
                point_scales: &[],
                leader_scales: leader_squared_norms,
            },
            PartitionScales::Cosine {
                point_squared_norms,
                leader_norms,
            } => ScaleSlices {
                point_scales: point_squared_norms,
                leader_scales: leader_norms,
            },
            PartitionScales::None => ScaleSlices {
                point_scales: &[],
                leader_scales: &[],
            },
        };
        for (point, (point_dots, point_output)) in input
            .dots
            .row_iter()
            .zip(output.chunks_exact_mut(fanout))
            .enumerate()
        {
            let point_scale = if M::PARTITION_POINT_SCALE.is_some() {
                M::PARTITION_POINT_SCALE.transform(scales.point_scales[point])
            } else {
                0.0
            };
            let mut tracker = vec![(u32::MAX, f32::INFINITY); fanout];
            for (leader, &dot) in point_dots.iter().enumerate() {
                let leader_scale = if M::PARTITION_LEADER_SCALE.is_some() {
                    M::PARTITION_LEADER_SCALE.transform(scales.leader_scales[leader])
                } else {
                    0.0
                };
                insert_leader(
                    &mut tracker,
                    leader as u32,
                    M::partition_distance_scalar(dot, point_scale, leader_scale),
                );
            }
            copy_leader_ids(&tracker, point_output);
        }
    }

    #[test]
    fn scalar_distance_matches_metric_contract() {
        assert_eq!(L2::partition_distance_scalar(2.0, 0.0, 9.0), 5.0);
        assert_eq!(
            CosineNormalized::partition_distance_scalar(0.25, 0.0, 0.0),
            0.75
        );
        assert_eq!(InnerProduct::partition_distance_scalar(3.0, 0.0, 0.0), -3.0);
        assert_eq!(Cosine::partition_distance_scalar(4.0, 2.0, 4.0), 0.5);
        assert_eq!(Cosine::partition_distance_scalar(4.0, 0.0, 4.0), 1.0);
        assert!(Cosine::partition_distance_scalar(1.0, f32::NAN, 1.0).is_nan());
    }

    #[test]
    fn cosine_special_norms_match_scalar_and_dispatched_kernel() {
        let leader_count = 17;
        let point_scales = [0.0, f32::MIN_POSITIVE / 2.0, f32::MIN_POSITIVE, f32::NAN];
        let dots = vec![1.0; point_scales.len() * leader_count];
        let mut leader_scales = vec![1.0; leader_count];
        leader_scales[..4].copy_from_slice(&[
            0.0,
            f32::MIN_POSITIVE.sqrt() / 2.0,
            f32::MIN_POSITIVE.sqrt(),
            f32::NAN,
        ]);
        let input = test_input(
            Metric::Cosine,
            &dots,
            point_scales.len(),
            leader_count,
            &point_scales,
            &leader_scales,
        );
        let mut expected = vec![u32::MAX; point_scales.len() * 2];
        scalar_traversal_reference::<Cosine>(input, 2, &mut expected);
        let mut actual = vec![u32::MAX; point_scales.len() * 2];
        dispatch_nearest_leaders(
            Metric::Cosine,
            input,
            MutMatrixView::try_from(actual.as_mut_slice(), point_scales.len(), 2).unwrap(),
            &mut PartitionKernelWorkspace::default(),
        )
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(&actual[..4], &[0, 1, 0, 1]);
        assert_eq!(&actual[6..], &[0, 1]);
    }

    #[test]
    fn scalar_topk_orders_candidates_and_preserves_ties() {
        let mut tracker = vec![(u32::MAX, f32::INFINITY); 4];
        for (leader, distance) in [(0, 4.0), (1, 1.0), (2, 3.0), (3, 2.0), (4, 1.0)] {
            insert_leader(&mut tracker, leader, distance);
        }
        insert_leader(&mut tracker, 5, f32::NAN);

        assert_eq!(tracker[..], [(1, 1.0), (4, 1.0), (3, 2.0), (2, 3.0)]);
    }

    #[test]
    fn workspace_reuses_runtime_fanout_capacity() {
        let mut workspace = PartitionKernelWorkspace::default();
        workspace.prepare(32).unwrap();
        let allocation = workspace.tracker.as_ptr();

        workspace.prepare(3).unwrap();

        assert_eq!(workspace.tracker.as_ptr(), allocation);
        assert_eq!(workspace.tracker.len(), 3);
    }
}
#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "deterministic test fixture construction must abort on invalid setup"
)]
mod integration_tests {
    use super::{
        PartitionInput, PartitionKernelError, PartitionKernelWorkspace, PartitionScales,
        dispatch_nearest_leaders,
    };
    use diskann_utils::views::{MatrixView, MutMatrixView};
    use diskann_vector::distance::Metric;

    fn test_input<'a>(
        metric: Metric,
        dots: &'a [f32],
        point_count: usize,
        leader_count: usize,
        point_scales: &'a [f32],
        leader_scales: &'a [f32],
    ) -> PartitionInput<'a> {
        let scales = match metric {
            Metric::L2 => PartitionScales::L2 {
                leader_squared_norms: leader_scales,
            },
            Metric::Cosine => PartitionScales::Cosine {
                point_squared_norms: point_scales,
                leader_norms: leader_scales,
            },
            Metric::CosineNormalized | Metric::InnerProduct => PartitionScales::None,
        };
        PartitionInput {
            dots: MatrixView::try_from(dots, point_count, leader_count).unwrap(),
            scales,
        }
    }

    fn brute_force_reference(input: PartitionInput<'_>, fanout: usize, metric: Metric) -> Vec<u32> {
        let point_count = input.dots.nrows();
        let leader_count = input.dots.ncols();
        let (point_scales, leader_scales) = match input.scales {
            PartitionScales::L2 {
                leader_squared_norms,
            } => (&[][..], leader_squared_norms),
            PartitionScales::Cosine {
                point_squared_norms,
                leader_norms,
            } => (point_squared_norms, leader_norms),
            PartitionScales::None => (&[][..], &[][..]),
        };
        let mut assignments = vec![u32::MAX; point_count * fanout];
        for (point, (point_dots, point_assignments)) in input
            .dots
            .as_slice()
            .chunks_exact(leader_count)
            .zip(assignments.chunks_exact_mut(fanout))
            .enumerate()
        {
            let point_scale = point_scales.get(point).copied().unwrap_or(0.0);
            let mut candidates: Vec<_> = point_dots
                .iter()
                .enumerate()
                .filter_map(|(leader, &dot)| {
                    let leader_scale = leader_scales.get(leader).copied().unwrap_or(0.0);
                    let distance = match metric {
                        Metric::L2 => leader_scale - 2.0 * dot,
                        Metric::CosineNormalized => 1.0 - dot,
                        Metric::InnerProduct => -dot,
                        Metric::Cosine => {
                            let point_norm = if point_scale < f32::MIN_POSITIVE {
                                0.0
                            } else {
                                point_scale.sqrt()
                            };
                            1.0 - if point_norm == 0.0 || leader_scale == 0.0 {
                                0.0
                            } else {
                                dot / (point_norm * leader_scale)
                            }
                        }
                    };
                    (distance.partial_cmp(&f32::INFINITY) == Some(std::cmp::Ordering::Less))
                        .then_some((leader as u32, distance))
                })
                .collect();
            candidates.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap());
            for (destination, (leader, _)) in point_assignments.iter_mut().zip(candidates) {
                *destination = leader;
            }
        }
        assignments
    }

    fn differential_data(metric: Metric, leader_count: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let dots = (0..2 * leader_count)
            .map(|index| {
                let leader = index % leader_count;
                let point = index / leader_count;
                let base = ((leader * 13 + point * 7) % 19) as f32 - 9.0;
                if leader == 2 || leader == 3 {
                    1.0
                } else if leader + 1 == leader_count {
                    f32::NAN
                } else {
                    base * 0.25
                }
            })
            .collect();
        let point_scales = if metric == Metric::Cosine {
            vec![0.0, 16.0]
        } else {
            Vec::new()
        };
        let leader_scales = match metric {
            Metric::Cosine => (0..leader_count)
                .map(|leader| {
                    if leader == 1 {
                        0.0
                    } else if leader == 2 || leader == 3 {
                        3.0
                    } else {
                        1.0 + leader as f32
                    }
                })
                .collect(),
            Metric::L2 => (0..leader_count)
                .map(|leader| {
                    let norm = if leader == 2 || leader == 3 {
                        3.0
                    } else {
                        leader as f32 + 1.0
                    };
                    norm * norm
                })
                .collect(),
            Metric::CosineNormalized | Metric::InnerProduct => Vec::new(),
        };
        (dots, point_scales, leader_scales)
    }

    fn run(
        metric: Metric,
        input: PartitionInput<'_>,
        fanout: usize,
    ) -> Result<Vec<u32>, PartitionKernelError> {
        let mut output = vec![u32::MAX; input.dots.nrows() * fanout];
        dispatch_nearest_leaders(
            metric,
            input,
            MutMatrixView::try_from(output.as_mut_slice(), input.dots.nrows(), fanout).unwrap(),
            &mut PartitionKernelWorkspace::default(),
        )?;
        Ok(output)
    }

    #[test]
    fn dispatched_kernel_matches_reference_across_simd_width_boundaries() {
        for metric in [
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct,
        ] {
            for leader_count in [2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
                let (dots, point_scales, leader_scales) = differential_data(metric, leader_count);
                let input = test_input(
                    metric,
                    &dots,
                    2,
                    leader_count,
                    &point_scales,
                    &leader_scales,
                );
                for fanout in [1, 2, 16, 17, 32] {
                    if fanout >= leader_count {
                        continue;
                    }
                    assert_eq!(
                        run(metric, input, fanout).unwrap(),
                        brute_force_reference(input, fanout, metric),
                        "{metric:?}, leaders={leader_count}, k={fanout}"
                    );
                }
            }
        }
    }

    #[test]
    fn l2_keeps_the_first_leader_when_boundary_distances_tie() {
        #[rustfmt::skip]
        let dots = [
            0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 4.0, 6.0,
        ];
        let norms = [0.0, 1.0, 4.0, 9.0];

        assert_eq!(
            run(
                Metric::L2,
                test_input(Metric::L2, &dots, 2, 4, &[], &norms),
                2
            )
            .unwrap(),
            [0, 1, 2, 1]
        );
    }

    #[test]
    fn supports_every_partition_metric() {
        #[rustfmt::skip]
        let dots = [
            1.0, 0.0, -1.0,
            2.0, 6.0, 0.0,
        ];
        for (metric, point_scales, leader_scales, expected) in [
            (Metric::L2, &[][..], &[1.0, 4.0, 9.0][..], [0, 1, 1, 0]),
            (
                Metric::Cosine,
                &[1.0, 4.0][..],
                &[1.0, 2.0, 3.0][..],
                [0, 1, 1, 0],
            ),
            (Metric::CosineNormalized, &[][..], &[][..], [0, 1, 1, 0]),
            (Metric::InnerProduct, &[][..], &[][..], [0, 1, 1, 0]),
        ] {
            assert_eq!(
                run(
                    metric,
                    test_input(metric, &dots, 2, 3, point_scales, leader_scales),
                    2,
                )
                .unwrap(),
                expected,
                "metric {metric:?}"
            );
        }
    }

    #[test]
    fn cosine_treats_a_zero_norm_as_zero_similarity() {
        assert_eq!(
            run(
                Metric::Cosine,
                test_input(Metric::Cosine, &[100.0, -100.0], 1, 2, &[0.0], &[1.0, 1.0]),
                2,
            )
            .unwrap(),
            [0, 1]
        );
    }

    #[test]
    fn finite_max_distance_fills_the_final_simd_slot() {
        let mut dots = [0.0; 8];
        dots[7] = -f32::MAX;
        assert_eq!(
            run(
                Metric::InnerProduct,
                test_input(Metric::InnerProduct, &dots, 1, 8, &[], &[]),
                8
            )
            .unwrap(),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn ignores_nan_distances_without_displacing_finite_leaders() {
        assert_eq!(
            run(
                Metric::InnerProduct,
                test_input(Metric::InnerProduct, &[f32::NAN, 3.0, 2.0], 1, 3, &[], &[]),
                2,
            )
            .unwrap(),
            [1, 2]
        );
    }

    #[test]
    fn rejects_points_with_too_few_rankable_leaders() {
        assert_eq!(
            run(
                Metric::InnerProduct,
                test_input(Metric::InnerProduct, &[f32::NAN, 3.0], 1, 2, &[], &[]),
                2,
            ),
            Err(PartitionKernelError::InsufficientRankableLeaders {
                point: 0,
                fanout: 2,
            })
        );
    }

    #[test]
    fn accepts_empty_points_zero_fanout_and_largest_leader_id() {
        run(
            Metric::InnerProduct,
            test_input(Metric::InnerProduct, &[], 0, 3, &[], &[]),
            2,
        )
        .unwrap();
        run(
            Metric::InnerProduct,
            test_input(Metric::InnerProduct, &[1.0, 2.0, 3.0], 1, 3, &[], &[]),
            0,
        )
        .unwrap();
        run(
            Metric::InnerProduct,
            test_input(Metric::InnerProduct, &[], 0, u32::MAX as usize, &[], &[]),
            0,
        )
        .unwrap();

        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            run(
                Metric::InnerProduct,
                test_input(
                    Metric::InnerProduct,
                    &[],
                    0,
                    u32::MAX as usize + 1,
                    &[],
                    &[],
                ),
                0,
            ),
            Err(PartitionKernelError::TooManyLeaders(u32::MAX as usize + 1))
        );
    }

    #[test]
    fn rejects_wrong_output_scales_and_fanout() {
        let dots = [0.0; 6];
        let valid_input = test_input(Metric::InnerProduct, &dots, 2, 3, &[], &[]);
        let mut wrong_output = [u32::MAX; 3];
        assert_eq!(
            dispatch_nearest_leaders(
                Metric::InnerProduct,
                valid_input,
                MutMatrixView::try_from(&mut wrong_output[..], 1, 3).unwrap(),
                &mut PartitionKernelWorkspace::default(),
            ),
            Err(PartitionKernelError::InvalidOutputShape {
                expected_rows: 2,
                actual_rows: 1,
                actual_cols: 3,
            })
        );

        let wrong_scales = PartitionInput {
            dots: MatrixView::try_from(&dots[..], 2, 3).unwrap(),
            scales: PartitionScales::None,
        };
        assert_eq!(
            run(Metric::L2, wrong_scales, 2),
            Err(PartitionKernelError::InvalidScales { expected: "L2" })
        );

        assert_eq!(
            run(Metric::InnerProduct, valid_input, 4),
            Err(PartitionKernelError::InvalidFanout {
                fanout: 4,
                leader_count: 3,
            })
        );

        let one = [0.0];
        assert_eq!(
            run(
                Metric::InnerProduct,
                test_input(Metric::InnerProduct, &one, 1, 1, &[], &[]),
                2,
            ),
            Err(PartitionKernelError::InvalidFanout {
                fanout: 2,
                leader_count: 1,
            })
        );
    }
}
