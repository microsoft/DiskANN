/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Select partition centers for PiPNN point assignment.
//!
//! A leader is a sampled dataset point that represents one child partition.
//! Each input row contains dot products from one assigned point to all sampled
//! leaders. Each output row contains the nearest leader-column IDs. The scatter
//! step uses each column ID as a child-partition ID.
//!
//! The caller supplies concrete architecture `A` and metric `M`. The function
//! checks row counts, norm units, norm lengths, fanout, and leader-ID range.
//! These checks occur before output changes or unchecked SIMD loads.
//!
//! L2 omits the assigned point's norm because it is constant across all sampled
//! leaders. Equal scores keep sampled-leader order. NaN is not rankable.

use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;
use diskann_wide::{
    Architecture, Const, SIMDFloat, SIMDMask, SIMDPartialOrd, SIMDSelect, SIMDVector,
};

use super::kernel_metric::{MetricTag, PartitionKernelMetric, norm_from_squared};

/// Reusable nearest-center state for one partition worker.
///
/// Each entry contains a sampled leader's matrix-column ID and its metric score.
#[derive(Debug, Default)]
pub(super) struct PartitionKernelWorkspace {
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

/// Norm values for one point-to-leader tile.
///
/// Cosine point values are squared norms. Cosine leader values are norms.
#[derive(Clone, Copy, Debug)]
pub(super) enum PartitionNorms<'a> {
    /// L2 uses the squared norm of each sampled partition center.
    L2 {
        /// Squared norm for each sampled leader.
        leader_squared_norms: &'a [f32],
    },
    /// Unnormalized cosine uses norms for assigned points and sampled leaders.
    Cosine {
        /// Squared norm for every point.
        point_squared_norms: &'a [f32],
        /// Norm for each sampled leader.
        leader_norms: &'a [f32],
    },
    /// Normalized cosine and inner product need no normalization inputs.
    None,
}

/// Dot products between assigned points and sampled partition centers.
///
/// Each row is one point being assigned. Each column is one sampled leader.
/// [`Self::norms`] supplies the norm layout for metric `M`.
#[derive(Clone, Copy, Debug)]
pub(super) struct PartitionInput<'a> {
    pub(super) dots: MatrixView<'a, f32>,
    pub(super) norms: PartitionNorms<'a>,
}

/// Validation error returned by [`nearest_leaders`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum PartitionKernelError {
    /// The output matrix does not match the input row count.
    #[error(
        "invalid output shape: expected {expected_rows} rows, got {actual_rows} rows and {actual_cols} columns"
    )]
    InvalidOutputShape {
        expected_rows: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    /// A metric-specific norm slice has the wrong length.
    #[error("invalid {buffer} length: expected {expected}, got {actual}")]
    InvalidBufferLength {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Norm inputs do not match concrete metric `M`.
    #[error("partition norms do not match selected {expected} metric")]
    InvalidNorms { expected: &'static str },
    /// The requested fanout exceeds the available leader count.
    #[error("invalid fanout {fanout}: must not exceed {leader_count} leaders")]
    InvalidFanout { fanout: usize, leader_count: usize },
    /// Reusable tracker storage could not be reserved.
    #[error("failed to reserve {additional} partition tracker entries")]
    Allocation { additional: usize },
    /// Leader positions cannot be represented as `u32`.
    #[error("leader count {0} exceeds the u32 position limit")]
    TooManyLeaders(usize),
    /// A point did not contain enough rankable leaders to fill its output.
    #[error("point {point} has fewer than {fanout} rankable leaders")]
    InsufficientRankableLeaders { point: usize, fanout: usize },
}

/// Select the nearest sampled partition centers for each input point.
///
/// The output width is the fanout. Each output value is a leader's column ID in
/// `input.dots`. Partition scatter uses that ID to select a child partition.
///
/// # Errors
///
/// Returns an error for an invalid shape, norm input, fanout, or allocation.
/// It also returns an error when fewer than `fanout` scores are rankable.
pub(super) fn nearest_leaders<A, M>(
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
    M: PartitionKernelMetric,
{
    let norms = validate::<M>(input, &output)?;
    let fanout = output.ncols();
    if fanout == 0 || input.dots.nrows() == 0 {
        return Ok(());
    }

    workspace.prepare(fanout)?;
    select_point_leaders::<A::f32x16, M>(arch, input.dots, norms, output, &mut workspace.tracker)
}

/// Checked norm slices for one concrete metric.
#[derive(Clone, Copy)]
struct PartitionNormSlices<'a> {
    point_squared_norms: &'a [f32],
    leader_norm_values: &'a [f32],
}

/// Check the safety and metric conditions for partition selection.
///
/// Matrix views prove their backing lengths. This function checks row counts,
/// leader-ID range, fanout, norm variant, and norm lengths.
fn validate<'a, M: PartitionKernelMetric>(
    input: PartitionInput<'a>,
    output: &MutMatrixView<'_, u32>,
) -> Result<PartitionNormSlices<'a>, PartitionKernelError> {
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

    match (<M as MetricTag>::METRIC, input.norms) {
        (
            Metric::L2,
            PartitionNorms::L2 {
                leader_squared_norms,
            },
        ) => {
            check_length("leader norms", leader_squared_norms.len(), leader_count)?;
            Ok(PartitionNormSlices {
                point_squared_norms: &[],
                leader_norm_values: leader_squared_norms,
            })
        }
        (
            Metric::Cosine,
            PartitionNorms::Cosine {
                point_squared_norms,
                leader_norms,
            },
        ) => {
            check_length("point norms", point_squared_norms.len(), point_count)?;
            check_length("leader norms", leader_norms.len(), leader_count)?;
            Ok(PartitionNormSlices {
                point_squared_norms,
                leader_norm_values: leader_norms,
            })
        }
        (Metric::CosineNormalized | Metric::InnerProduct, PartitionNorms::None) => {
            Ok(PartitionNormSlices {
                point_squared_norms: &[],
                leader_norm_values: &[],
            })
        }
        (Metric::L2, _) => Err(PartitionKernelError::InvalidNorms { expected: "L2" }),
        (Metric::Cosine, _) => Err(PartitionKernelError::InvalidNorms { expected: "cosine" }),
        (Metric::CosineNormalized, _) => Err(PartitionKernelError::InvalidNorms {
            expected: "normalized cosine",
        }),
        (Metric::InnerProduct, _) => Err(PartitionKernelError::InvalidNorms {
            expected: "inner product",
        }),
    }
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

/// Rank sampled partition centers for each assigned point.
///
/// The function converts point-to-leader dot products to metric `M` scores. It
/// keeps the nearest `output.ncols()` centers in sampled-leader order for ties.
/// NaN and positive infinity are not rankable.
///
/// `tracker` stores the retained center-column IDs and scores for the current
/// point. The function resets this state before it processes another point.
fn select_point_leaders<F, M>(
    arch: F::Arch,
    dots: MatrixView<'_, f32>,
    norms: PartitionNormSlices<'_>,
    mut output: MutMatrixView<'_, u32>,
    tracker: &mut [(u32, f32)],
) -> Result<(), PartitionKernelError>
where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: PartitionKernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let leader_count = dots.ncols();
    let fanout = output.ncols();
    let metric = <M as MetricTag>::METRIC;
    let uses_point_norm = metric == Metric::Cosine;
    let uses_leader_norm = matches!(metric, Metric::L2 | Metric::Cosine);
    // Reset the tracker for each point. No assignment state can pass from one
    // output row to another.
    for (point, (point_dots, point_output)) in dots
        .row_iter()
        .zip(output.as_mut_slice().chunks_exact_mut(fanout))
        .enumerate()
    {
        tracker.fill((u32::MAX, f32::INFINITY));
        let point_norm = if uses_point_norm {
            norm_from_squared(norms.point_squared_norms[point])
        } else {
            0.0
        };
        let point_norm_vector = F::splat(arch, point_norm);
        // Process all complete SIMD groups first. The scalar tail uses the
        // metric's scalar operation order.
        let full = leader_count / F::LANES * F::LANES;

        for base in (0..full).step_by(F::LANES) {
            // SAFETY: `base + F::LANES <= full <= point_dots.len()`.
            let point_dots = unsafe { F::load_simd(arch, point_dots.as_ptr().add(base)) };
            let leader_norms = if uses_leader_norm {
                // SAFETY: `validate` established one norm value per leader.
                // `base + F::LANES <= full <= leader_count`.
                unsafe { F::load_simd(arch, norms.leader_norm_values.as_ptr().add(base)) }
            } else {
                F::default(arch)
            };
            insert_leader_lanes(
                M::partition_ranking(arch, point_dots, point_norm_vector, leader_norms),
                base,
                tracker,
            );
        }

        // Use scalar formulas for the tail. A padded SIMD load can read past the
        // norm slice and can change L2 rounding.
        for (leader, &dot) in point_dots.iter().enumerate().skip(full) {
            let leader_norm = if uses_leader_norm {
                norms.leader_norm_values[leader]
            } else {
                0.0
            };
            insert_leader(
                tracker,
                leader as u32,
                M::partition_ranking_scalar(dot, point_norm, leader_norm),
            );
        }
        if tracker[fanout - 1].0 == u32::MAX {
            return Err(PartitionKernelError::InsufficientRankableLeaders { point, fanout });
        }
        // Scatter needs the sampled-center column IDs. Metric scores remain in
        // the worker workspace.
        for (destination, &(leader, _)) in point_output.iter_mut().zip(tracker.iter()) {
            *destination = leader;
        }
    }
    Ok(())
}

/// Offer one SIMD group of sampled centers to the current point's tracker.
///
/// `first_leader` is the matrix-column ID of the first lane. Lanes enter in
/// sampled-leader order, which preserves tie order.
fn insert_leader_lanes<F>(scores: F, first_leader: usize, tracker: &mut [(u32, f32)])
where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDPartialOrd,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let threshold = F::splat(scores.arch(), tracker[tracker.len() - 1].1);
    let eligible = scores.lt_simd(threshold);
    if eligible.none() {
        return;
    }

    let values: [f32; 16] = scores.to_array();
    let mut lanes = u64::from(eligible.bitmask().to_underlying());
    while lanes != 0 {
        let lane = lanes.trailing_zeros() as usize;
        lanes &= lanes - 1;
        insert_leader(tracker, (first_leader + lane) as u32, values[lane]);
    }
}

/// Insert one sampled partition center into the current point's retained set.
///
/// `leader` is the center's column ID in the point-to-leader matrix. `tracker`
/// stores retained centers in nearest-first order. Equal scores and NaN do not
/// enter, so sampled-leader order resolves ties.
#[inline(always)]
fn insert_leader(tracker: &mut [(u32, f32)], leader: u32, score: f32) {
    let threshold = tracker.len() - 1;
    if score.partial_cmp(&tracker[threshold].1) != Some(std::cmp::Ordering::Less) {
        return;
    }

    tracker[threshold] = (leader, score);
    let mut slot = threshold;
    while slot > 0 && tracker[slot].1 < tracker[slot - 1].1 {
        tracker.swap(slot, slot - 1);
        slot -= 1;
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
    use super::super::kernel_metric::{
        Cosine, CosineNormalized, InnerProduct, L2, PartitionKernelMetric,
    };

    use super::*;

    fn test_input<'a>(
        metric: Metric,
        dots: &'a [f32],
        point_count: usize,
        leader_count: usize,
        point_squared_norms: &'a [f32],
        leader_norm_values: &'a [f32],
    ) -> PartitionInput<'a> {
        let norms = match metric {
            Metric::L2 => PartitionNorms::L2 {
                leader_squared_norms: leader_norm_values,
            },
            Metric::Cosine => PartitionNorms::Cosine {
                point_squared_norms,
                leader_norms: leader_norm_values,
            },
            Metric::CosineNormalized | Metric::InnerProduct => PartitionNorms::None,
        };
        PartitionInput {
            dots: MatrixView::try_from(dots, point_count, leader_count).unwrap(),
            norms,
        }
    }

    // This oracle checks SIMD chunking, scalar tails, and tracker order. It uses
    // the scalar ranking formula for metric `M`.
    fn scalar_traversal_reference<M: PartitionKernelMetric>(
        input: PartitionInput<'_>,
        fanout: usize,
        output: &mut [u32],
    ) {
        let norms = match input.norms {
            PartitionNorms::L2 {
                leader_squared_norms,
            } => PartitionNormSlices {
                point_squared_norms: &[],
                leader_norm_values: leader_squared_norms,
            },
            PartitionNorms::Cosine {
                point_squared_norms,
                leader_norms,
            } => PartitionNormSlices {
                point_squared_norms,
                leader_norm_values: leader_norms,
            },
            PartitionNorms::None => PartitionNormSlices {
                point_squared_norms: &[],
                leader_norm_values: &[],
            },
        };
        let metric = <M as MetricTag>::METRIC;
        for (point, (point_dots, point_output)) in input
            .dots
            .row_iter()
            .zip(output.chunks_exact_mut(fanout))
            .enumerate()
        {
            let point_norm = if metric == Metric::Cosine {
                norm_from_squared(norms.point_squared_norms[point])
            } else {
                0.0
            };
            let mut tracker = vec![(u32::MAX, f32::INFINITY); fanout];
            for (leader, &dot) in point_dots.iter().enumerate() {
                let leader_norm = if matches!(metric, Metric::L2 | Metric::Cosine) {
                    norms.leader_norm_values[leader]
                } else {
                    0.0
                };
                insert_leader(
                    &mut tracker,
                    leader as u32,
                    M::partition_ranking_scalar(dot, point_norm, leader_norm),
                );
            }
            for (destination, &(leader, _)) in point_output.iter_mut().zip(&tracker) {
                *destination = leader;
            }
        }
    }

    #[test]
    fn scalar_ranking_matches_metric_contract() {
        assert_eq!(L2::partition_ranking_scalar(2.0, 0.0, 9.0), 5.0);
        assert_eq!(
            CosineNormalized::partition_ranking_scalar(0.25, 0.0, 0.0),
            0.75
        );
        assert_eq!(InnerProduct::partition_ranking_scalar(3.0, 0.0, 0.0), -3.0);
        assert_eq!(Cosine::partition_ranking_scalar(4.0, 2.0, 4.0), 0.5);
        assert_eq!(Cosine::partition_ranking_scalar(4.0, 0.0, 4.0), 1.0);
        assert!(Cosine::partition_ranking_scalar(1.0, f32::NAN, 1.0).is_nan());
    }

    #[test]
    fn cosine_special_norms_match_scalar_and_dispatched_kernel() {
        let leader_count = 17;
        let point_squared_norms = [0.0, f32::MIN_POSITIVE / 2.0, f32::MIN_POSITIVE, f32::NAN];
        let dots = vec![1.0; point_squared_norms.len() * leader_count];
        let mut leader_norms = vec![1.0; leader_count];
        leader_norms[..4].copy_from_slice(&[
            0.0,
            f32::MIN_POSITIVE.sqrt() / 2.0,
            f32::MIN_POSITIVE.sqrt(),
            f32::NAN,
        ]);
        let input = test_input(
            Metric::Cosine,
            &dots,
            point_squared_norms.len(),
            leader_count,
            &point_squared_norms,
            &leader_norms,
        );
        let mut expected = vec![u32::MAX; point_squared_norms.len() * 2];
        scalar_traversal_reference::<Cosine>(input, 2, &mut expected);
        let mut actual = vec![u32::MAX; point_squared_norms.len() * 2];
        dispatch_nearest_leaders(
            Metric::Cosine,
            input,
            MutMatrixView::try_from(actual.as_mut_slice(), point_squared_norms.len(), 2).unwrap(),
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
        PartitionInput, PartitionKernelError, PartitionKernelWorkspace, PartitionNorms,
        dispatch_nearest_leaders, norm_from_squared,
    };
    use diskann_utils::views::{MatrixView, MutMatrixView};
    use diskann_vector::distance::Metric;

    fn test_input<'a>(
        metric: Metric,
        dots: &'a [f32],
        point_count: usize,
        leader_count: usize,
        point_squared_norms: &'a [f32],
        leader_norm_values: &'a [f32],
    ) -> PartitionInput<'a> {
        let norms = match metric {
            Metric::L2 => PartitionNorms::L2 {
                leader_squared_norms: leader_norm_values,
            },
            Metric::Cosine => PartitionNorms::Cosine {
                point_squared_norms,
                leader_norms: leader_norm_values,
            },
            Metric::CosineNormalized | Metric::InnerProduct => PartitionNorms::None,
        };
        PartitionInput {
            dots: MatrixView::try_from(dots, point_count, leader_count).unwrap(),
            norms,
        }
    }

    fn brute_force_reference(input: PartitionInput<'_>, fanout: usize, metric: Metric) -> Vec<u32> {
        let point_count = input.dots.nrows();
        let leader_count = input.dots.ncols();
        let (point_squared_norms, leader_norm_values) = match input.norms {
            PartitionNorms::L2 {
                leader_squared_norms,
            } => (&[][..], leader_squared_norms),
            PartitionNorms::Cosine {
                point_squared_norms,
                leader_norms,
            } => (point_squared_norms, leader_norms),
            PartitionNorms::None => (&[][..], &[][..]),
        };
        let mut assignments = vec![u32::MAX; point_count * fanout];
        for (point, (point_dots, point_assignments)) in input
            .dots
            .as_slice()
            .chunks_exact(leader_count)
            .zip(assignments.chunks_exact_mut(fanout))
            .enumerate()
        {
            let point_squared_norm = point_squared_norms.get(point).copied().unwrap_or(0.0);
            let mut candidates: Vec<_> = point_dots
                .iter()
                .enumerate()
                .filter_map(|(leader, &dot)| {
                    let leader_norm = leader_norm_values.get(leader).copied().unwrap_or(0.0);
                    let score = match metric {
                        Metric::L2 => leader_norm - 2.0 * dot,
                        Metric::CosineNormalized => 1.0 - dot,
                        Metric::InnerProduct => -dot,
                        Metric::Cosine => {
                            let point_norm = norm_from_squared(point_squared_norm);
                            1.0 - if point_norm == 0.0 || leader_norm == 0.0 {
                                0.0
                            } else {
                                dot / (point_norm * leader_norm)
                            }
                        }
                    };
                    (score.partial_cmp(&f32::INFINITY) == Some(std::cmp::Ordering::Less))
                        .then_some((leader as u32, score))
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
        let point_squared_norms = if metric == Metric::Cosine {
            vec![0.0, 16.0]
        } else {
            Vec::new()
        };
        let leader_norm_values = match metric {
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
        (dots, point_squared_norms, leader_norm_values)
    }

    fn run_partition_kernel(
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
                let (dots, point_squared_norms, leader_norm_values) =
                    differential_data(metric, leader_count);
                let input = test_input(
                    metric,
                    &dots,
                    2,
                    leader_count,
                    &point_squared_norms,
                    &leader_norm_values,
                );
                for fanout in [1, 2, 16, 17, 32] {
                    if fanout >= leader_count {
                        continue;
                    }
                    assert_eq!(
                        run_partition_kernel(metric, input, fanout).unwrap(),
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
            run_partition_kernel(
                Metric::L2,
                test_input(Metric::L2, &dots, 2, 4, &[], &norms),
                2
            )
            .unwrap(),
            [0, 1, 2, 1]
        );
    }

    #[test]
    fn l2_scalar_tail_can_outrank_a_fused_simd_lane() {
        let mut dots = [0.0; 17];
        dots[0] = f32::MAX;
        dots[16] = f32::MAX;
        let leader_squared_norms = [f32::MAX; 17];

        assert_eq!(
            run_partition_kernel(
                Metric::L2,
                test_input(Metric::L2, &dots, 1, 17, &[], &leader_squared_norms,),
                1,
            )
            .unwrap(),
            [16]
        );
    }

    #[test]
    fn supports_every_partition_metric() {
        #[rustfmt::skip]
        let dots = [
            1.0, 0.0, -1.0,
            2.0, 6.0, 0.0,
        ];
        for (metric, point_squared_norms, leader_norm_values, expected) in [
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
                run_partition_kernel(
                    metric,
                    test_input(metric, &dots, 2, 3, point_squared_norms, leader_norm_values),
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
            run_partition_kernel(
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
            run_partition_kernel(
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
            run_partition_kernel(
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
            run_partition_kernel(
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
        run_partition_kernel(
            Metric::InnerProduct,
            test_input(Metric::InnerProduct, &[], 0, 3, &[], &[]),
            2,
        )
        .unwrap();
        run_partition_kernel(
            Metric::InnerProduct,
            test_input(Metric::InnerProduct, &[1.0, 2.0, 3.0], 1, 3, &[], &[]),
            0,
        )
        .unwrap();
        run_partition_kernel(
            Metric::InnerProduct,
            test_input(Metric::InnerProduct, &[], 0, u32::MAX as usize, &[], &[]),
            0,
        )
        .unwrap();

        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            run_partition_kernel(
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
    fn rejects_wrong_output_norms_and_fanout() {
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

        let wrong_norms = PartitionInput {
            dots: MatrixView::try_from(&dots[..], 2, 3).unwrap(),
            norms: PartitionNorms::None,
        };
        assert_eq!(
            run_partition_kernel(Metric::L2, wrong_norms, 2),
            Err(PartitionKernelError::InvalidNorms { expected: "L2" })
        );

        assert_eq!(
            run_partition_kernel(Metric::InnerProduct, valid_input, 4),
            Err(PartitionKernelError::InvalidFanout {
                fanout: 4,
                leader_count: 3,
            })
        );

        let one = [0.0];
        assert_eq!(
            run_partition_kernel(
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
