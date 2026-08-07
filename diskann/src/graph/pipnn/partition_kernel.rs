/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Prepared nearest-leader selection for PiPNN partition assignment.
//!
//! Caller supplies a row-major point-by-leader dot matrix plus metric-specific
//! [`PartitionScales`]. Output contains sorted leader-column positions for each
//! point; fanout is the output width and cannot exceed the leader count.
//!
//! Runtime callers may use [`PartitionKernel`]; production dispatches once at
//! the partition-stage boundary and calls the generic kernel directly. Calls
//! validate row counts, scale variants and lengths, fanout, and leader-ID
//! representation before output mutation or unchecked SIMD loads.
//!
//! L2 omits the point norm because it cannot change one point's leader order.
//! Strict comparisons preserve scan order for ties and leave NaN non-rankable.
//! Each point evaluates every leader; competitive scores move through a
//! caller-owned tracker reused across points.

use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;
use diskann_wide::{
    Architecture, Const, SIMDFloat, SIMDMask, SIMDPartialOrd, SIMDSelect, SIMDVector,
    arch::{self, Target1},
};

use super::kernel_metric::{KernelMetric, MetricVisitor, ScaleKind, visit_metric};

/// Reusable nearest-leader tracker for one partition worker.
#[derive(Debug, Default)]
pub struct PartitionKernelWorkspace {
    tracker: Vec<(u32, f32)>,
}

impl PartitionKernelWorkspace {
    /// Construct an empty allocation-free workspace.
    pub const fn new() -> Self {
        Self {
            tracker: Vec::new(),
        }
    }

    fn prepare(&mut self, fanout: usize) -> Result<(), PartitionKernelError> {
        let additional = fanout.saturating_sub(self.tracker.len());
        self.tracker
            .try_reserve(additional)
            .map_err(|_| PartitionKernelError::Allocation { additional })?;
        self.tracker.resize(fanout, (u32::MAX, f32::INFINITY));
        Ok(())
    }
}

/// Metric-specific normalization inputs for one partition tile.
///
/// Slice lengths are checked against dot-matrix dimensions before output
/// mutation. Names encode units: cosine points arrive as squared norms because
/// they come from the point matrix diagonal, while leaders are normalized once
/// by the partition caller and arrive as norms.
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

/// One row-major point-by-leader dot-product tile.
///
/// Matrix rows are points, columns are leaders, and [`Self::scales`] must match
/// the metric used to prepare [`PartitionKernel`]. This value only borrows input;
/// the prepared kernel stores no tile state.
#[derive(Clone, Copy, Debug)]
pub struct PartitionInput<'a> {
    /// One point per matrix row and one leader per column.
    pub dots: MatrixView<'a, f32>,
    /// Normalization inputs matching the prepared metric.
    pub scales: PartitionScales<'a>,
}

/// Validation error returned by [`PartitionKernel::nearest_leaders`].
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
    /// Scale inputs do not match the metric used to prepare the kernel.
    #[error("partition scales do not match prepared {expected} metric")]
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

/// Inputs for one immediate architecture/metric dispatch.
#[derive(Debug)]
struct PartitionCall<'a> {
    input: PartitionInput<'a>,
    output: MutMatrixView<'a, u32>,
    workspace: &'a mut PartitionKernelWorkspace,
}

/// Partition-kernel convenience API for callers with a runtime [`Metric`].
#[derive(Clone, Copy, Debug)]
pub struct PartitionKernel {
    metric: Metric,
}

impl PartitionKernel {
    /// Construct a kernel selector for `metric`.
    pub const fn new(metric: Metric) -> Self {
        Self { metric }
    }

    /// Select nearest leader positions for every input point.
    ///
    /// `output.nrows()` must equal `input.dots.nrows()` and fanout, represented
    /// by `output.ncols()`, must not exceed the leader count.
    pub fn nearest_leaders(
        &self,
        input: PartitionInput<'_>,
        output: MutMatrixView<'_, u32>,
        workspace: &mut PartitionKernelWorkspace,
    ) -> Result<(), PartitionKernelError> {
        arch::dispatch1_no_features(
            RunPartition {
                metric: self.metric,
            },
            PartitionCall {
                input,
                output,
                workspace,
            },
        )
    }
}

struct RunPartition {
    metric: Metric,
}

impl<A> Target1<A, Result<(), PartitionKernelError>, PartitionCall<'_>> for RunPartition
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn run(self, arch: A, call: PartitionCall<'_>) -> Result<(), PartitionKernelError> {
        visit_metric(self.metric, ExecutePartition { arch, call })
    }
}

struct ExecutePartition<'a, A> {
    arch: A,
    call: PartitionCall<'a>,
}

impl<A> MetricVisitor for ExecutePartition<'_, A>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    type Output = Result<(), PartitionKernelError>;

    fn visit<M: KernelMetric>(self) -> Self::Output {
        nearest_leaders_for::<A, M>(
            self.arch,
            self.call.input,
            self.call.output,
            self.call.workspace,
        )
    }
}

/// Architecture/metric-specialized partition kernel used by stage dispatch.
pub(crate) fn nearest_leaders_for<A, M>(
    arch: A,
    input: PartitionInput<'_>,
    mut output: MutMatrixView<'_, u32>,
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
    process_points::<A::f32x16, M>(
        arch,
        input.dots,
        scales,
        output.as_mut_slice(),
        &mut workspace.tracker,
    );
    if let Some(point) = output
        .as_slice()
        .chunks_exact(fanout)
        .position(|assignments| assignments[fanout - 1] == u32::MAX)
    {
        return Err(PartitionKernelError::InsufficientRankableLeaders { point, fanout });
    }
    Ok(())
}

/// Validated scale slices in the storage form required by `M`.
///
/// Empty slices are intentional for metrics that omit a scale; consumers branch
/// on associated `ScaleKind` constants that monomorphize out of hot loops.
#[derive(Clone, Copy)]
struct ScaleSlices<'a> {
    point_scales: &'a [f32],
    leader_scales: &'a [f32],
}

/// Validate the complete partition-kernel safety and metric contract.
///
/// `MatrixView` and `MutMatrixView` construction guarantee exact,
/// non-overflowing backing lengths. The `PartitionScales` variant must match
/// concrete metric `M`, preventing plausible but incorrect norm units from
/// crossing the interface. Success returns normalized scale slices and
/// establishes representable leader IDs plus bounded fanout.
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

    // Match the public enum against the concrete marker before erasing it to
    // slices. This prevents squared point norms from being mistaken for leader
    // norms even though both representations are `&[f32]`.
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

    // After variant validation, associated scale kinds define exact lengths.
    // Scale-free metrics must provide empty slices so stale data cannot be used.
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

/// Return required scale length after metric specialization.
///
/// Associated `ScaleKind` constants make this choice compile away.
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

/// Convert each point's leader scores into sorted top-fanout IDs.
///
/// Per-point flow:
///
/// 1. transform the point scale once according to concrete metric `M`;
/// 2. process full SIMD leader groups, rejecting lanes against the last slot;
/// 3. process the remaining leaders with the scalar metric operation;
/// 4. copy the sorted tracker prefix to that point's output.
///
/// `tracker[..fanout]` remains sorted after every accepted candidate. Strict `<`
/// preserves leader scan order for ties and makes NaNs non-rankable. L2 keeps
/// historical bulk-FMA/scalar-tail rounding because changing it can alter graph
/// assignment at near ties.
///
/// `dots` supplies `p × l` scores, `scales` contains validated metric inputs,
/// `fanout` is both tracker prefix length and output width, and `output` contains
/// `p * fanout` slots. The function writes leader IDs in place and returns no
/// value. It computes `p * l` scores; each competitive score may shift `O(fanout)`
/// tracker entries. Tracker memory is fixed on the stack and no allocation occurs.
fn process_points<F, M>(
    arch: F::Arch,
    dots: MatrixView<'_, f32>,
    scales: ScaleSlices<'_>,
    output: &mut [u32],
    tracker: &mut [(u32, f32)],
) where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: KernelMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let point_count = dots.nrows();
    let leader_count = dots.ncols();
    assert!(
        leader_count > 0,
        "validated partition input must contain leaders"
    );
    if M::PARTITION_POINT_SCALE.is_some() {
        assert_eq!(
            scales.point_scales.len(),
            point_count,
            "validated point scales must match point count"
        );
    }
    if M::PARTITION_LEADER_SCALE.is_some() {
        assert_eq!(
            scales.leader_scales.len(),
            leader_count,
            "validated leader scales must match leader count"
        );
    }
    let fanout = tracker.len();
    // Each point is independent. Reset the caller-owned tracker so no assignment
    // state or tie order leaks across rows.
    for (point, (point_dots, point_output)) in dots
        .row_iter()
        .zip(output.chunks_exact_mut(fanout))
        .enumerate()
    {
        tracker.fill((u32::MAX, f32::INFINITY));
        // Transform once per point rather than once per leader. For metrics
        // without a point scale, specialization removes this branch and load.
        let point_scale = if M::PARTITION_POINT_SCALE.is_some() {
            M::PARTITION_POINT_SCALE.transform(scales.point_scales[point])
        } else {
            0.0
        };
        let point_scale_vector = F::splat(arch, point_scale);
        // Split at the largest complete vector boundary. Scalar tail uses the
        // metric's explicit scalar operation order, not a padded SIMD load.
        let full = leader_count / F::LANES * F::LANES;

        for base in (0..full).step_by(F::LANES) {
            // SAFETY: `base + F::LANES <= full <= point_dots.len()`.
            let point_dots = unsafe { F::load_simd(arch, point_dots.as_ptr().add(base)) };
            let leader_scales = if M::PARTITION_LEADER_SCALE.is_some() {
                // SAFETY: the assertion above established one scale per leader, and
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

        // Tail values use scalar metric functions intentionally. Padding a SIMD
        // group would risk out-of-bounds scale loads and different L2 rounding.
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
        // Distances are only tracker state; child-group construction needs leader
        // column positions in deterministic nearest-first order.
        copy_leader_ids(tracker, point_output);
    }
}

/// Offer competitive SIMD lanes to a point tracker in increasing leader order.
///
/// The broadcast threshold avoids materializing lanes when none can improve the
/// last slot. Bit iteration follows low-to-high lane order, preserving scalar tie
/// behavior across SIMD widths.
///
/// `distances` contains consecutive leaders beginning at `first_leader`;
/// `tracker[..fanout]` is the point's sorted retained prefix. The function
/// mutates that tracker and returns no value. Rejected groups cost one comparison
/// and mask test; accepted lanes each pay `O(fanout)` worst-case insertion.
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

/// Insert one strictly better candidate while preserving sorted-prefix state.
///
/// The last slot is overwritten, then bubbled left. Equal and NaN distances do
/// not enter, so scan order is the deterministic tie breaker and the last slot
/// remains both rejection threshold and underfill sentinel.
///
/// `tracker[..fanout]` must already be sorted and `fanout` must be non-zero.
/// `leader` is a local column position. The function returns no value and shifts
/// at most `fanout - 1` entries without allocation.
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

/// Publish only leader IDs; distances stay private tracker state.
///
/// `assignments.len()` is validated fanout. Copying costs `O(fanout)` and leaves
/// tracker state available for the underfill sentinel check encoded in IDs.
fn copy_leader_ids(tracker: &[(u32, f32)], assignments: &mut [u32]) {
    for (destination, &(leader, _)) in assignments.iter_mut().zip(tracker) {
        *destination = leader;
    }
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

    // Differential oracle for SIMD chunking, scalar tails, and tracker order.
    // It intentionally shares `M::partition_distance_scalar`; public API tests
    // independently spell out ranking formulas and full sorting behavior.
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
    fn cosine_special_norms_match_scalar_and_prepared_dispatch() {
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
        PartitionKernel::new(Metric::Cosine)
            .nearest_leaders(
                input,
                MutMatrixView::try_from(actual.as_mut_slice(), point_scales.len(), 2).unwrap(),
                &mut PartitionKernelWorkspace::new(),
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
}
#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "deterministic test fixture construction must abort on invalid setup"
)]
mod integration_tests {
    use super::{
        PartitionInput, PartitionKernel, PartitionKernelError, PartitionKernelWorkspace,
        PartitionScales,
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
        PartitionKernel::new(metric).nearest_leaders(
            input,
            MutMatrixView::try_from(output.as_mut_slice(), input.dots.nrows(), fanout).unwrap(),
            &mut PartitionKernelWorkspace::new(),
        )?;
        Ok(output)
    }

    #[test]
    fn prepared_dispatch_matches_reference_across_simd_width_boundaries() {
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
                for fanout in [1, 2, 16] {
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
            PartitionKernel::new(Metric::InnerProduct).nearest_leaders(
                valid_input,
                MutMatrixView::try_from(&mut wrong_output[..], 1, 3).unwrap(),
                &mut PartitionKernelWorkspace::new(),
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
