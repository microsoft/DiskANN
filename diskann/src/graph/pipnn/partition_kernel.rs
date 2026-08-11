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
#[cfg(test)]
use diskann_vector::distance::Metric;
use diskann_wide::{
    Architecture, Const, SIMDFloat, SIMDMask, SIMDPartialOrd, SIMDSelect, SIMDVector,
};

use super::kernel_metric::{PartitionMetric, PartitionNorms};

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
    /// The requested fanout exceeds the available leader count.
    #[error("invalid fanout {fanout}: must not exceed {leader_count} leaders")]
    InvalidFanout { fanout: usize, leader_count: usize },
    /// Reusable ranked-leader storage could not be reserved.
    #[error("failed to reserve {additional} partition ranked-leader entries")]
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
/// The function returns an error for an invalid shape, norm input, fanout, or allocation.
/// It also returns an error when fewer than `fanout` scores are rankable.
pub(super) fn nearest_leaders<A, M>(
    arch: A,
    input: PartitionInput<'_>,
    output: MutMatrixView<'_, u32>,
    ranked_leaders: &mut Vec<(u32, f32)>,
) -> Result<(), PartitionKernelError>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: PartitionMetric,
{
    let norms = validate(input, &output)?;
    let fanout = output.ncols();
    if fanout == 0 || input.dots.nrows() == 0 {
        return Ok(());
    }

    let additional = fanout.saturating_sub(ranked_leaders.len());
    ranked_leaders
        .try_reserve(additional)
        .map_err(|_| PartitionKernelError::Allocation { additional })?;
    ranked_leaders.resize(fanout, (u32::MAX, f32::INFINITY));

    match (norms.point_norms.is_empty(), norms.leader_norms.is_empty()) {
        (false, false) => select_point_leaders::<A::f32x16, M, _, _>(
            arch,
            input.dots,
            PreparedNorms(norms.point_norms),
            PreparedNorms(norms.leader_norms),
            output,
            ranked_leaders,
        ),
        (false, true) => select_point_leaders::<A::f32x16, M, _, _>(
            arch,
            input.dots,
            PreparedNorms(norms.point_norms),
            EmptyNorms,
            output,
            ranked_leaders,
        ),
        (true, false) => select_point_leaders::<A::f32x16, M, _, _>(
            arch,
            input.dots,
            EmptyNorms,
            PreparedNorms(norms.leader_norms),
            output,
            ranked_leaders,
        ),
        (true, true) => select_point_leaders::<A::f32x16, M, _, _>(
            arch,
            input.dots,
            EmptyNorms,
            EmptyNorms,
            output,
            ranked_leaders,
        ),
    }
}

/// This function checks row counts, leader IDs, fanout, and norm lengths.
fn validate<'a>(
    input: PartitionInput<'a>,
    output: &MutMatrixView<'_, u32>,
) -> Result<PartitionNorms<'a>, PartitionKernelError> {
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

    check_norm_count("point norms", input.norms.point_norms, point_count)?;
    check_norm_count("leader norms", input.norms.leader_norms, leader_count)?;
    Ok(input.norms)
}

/// Check one optional norm buffer.
///
/// An empty slice means that the active metric does not use this norm.
fn check_norm_count(
    buffer: &'static str,
    norms: &[f32],
    expected: usize,
) -> Result<(), PartitionKernelError> {
    if norms.is_empty() || norms.len() == expected {
        Ok(())
    } else {
        Err(PartitionKernelError::InvalidBufferLength {
            buffer,
            expected,
            actual: norms.len(),
        })
    }
}

/// Provide norm values for partition ranking.
trait NormValues<F>
where
    F: SIMDVector<Scalar = f32>,
{
    /// Repeat one norm in all SIMD lanes.
    fn repeat_simd(self, arch: F::Arch, point: usize) -> F;

    /// Load one complete SIMD group of norms.
    fn load_simd(self, arch: F::Arch, first_point: usize) -> F;

    /// Read one norm.
    fn read(self, point: usize) -> f32;
}

/// Prepared norm values for points or sampled leaders.
#[derive(Clone, Copy)]
struct PreparedNorms<'a>(&'a [f32]);

impl<F> NormValues<F> for PreparedNorms<'_>
where
    F: SIMDVector<Scalar = f32>,
{
    #[inline(always)]
    fn repeat_simd(self, arch: F::Arch, point: usize) -> F {
        F::splat(arch, self.0[point])
    }

    #[inline(always)]
    fn load_simd(self, arch: F::Arch, first_point: usize) -> F {
        let last_point = first_point + F::LANES;
        let norm_group = &self.0[first_point..last_point];

        // SAFETY: `norm_group` contains one complete SIMD group.
        unsafe { F::load_simd(arch, norm_group.as_ptr()) }
    }

    #[inline(always)]
    fn read(self, point: usize) -> f32 {
        self.0[point]
    }
}

/// Zero norm values for a metric that does not use one norm type.
#[derive(Clone, Copy)]
struct EmptyNorms;

impl<F> NormValues<F> for EmptyNorms
where
    F: SIMDVector<Scalar = f32>,
{
    #[inline(always)]
    fn repeat_simd(self, arch: F::Arch, point: usize) -> F {
        let _ = point;
        F::default(arch)
    }

    #[inline(always)]
    fn load_simd(self, arch: F::Arch, first_point: usize) -> F {
        let _ = first_point;
        F::default(arch)
    }

    #[inline(always)]
    fn read(self, point: usize) -> f32 {
        let _ = point;
        0.0
    }
}

/// Rank sampled partition centers for each assigned point.
///
/// The function converts point-to-leader dot products to metric `M` scores. It
/// keeps the nearest `output.ncols()` centers in sampled-leader order for ties.
/// NaN and positive infinity are not rankable.
///
/// `ranked_leaders` stores the retained center-column IDs and scores for the current
/// point. The function resets this state before it processes another point.
fn select_point_leaders<F, M, P, L>(
    arch: F::Arch,
    dots: MatrixView<'_, f32>,
    point_norms: P,
    leader_norms: L,
    mut output: MutMatrixView<'_, u32>,
    ranked_leaders: &mut [(u32, f32)],
) -> Result<(), PartitionKernelError>
where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: PartitionMetric,
    P: NormValues<F> + Copy,
    L: NormValues<F> + Copy,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let leader_count = dots.ncols();
    let fanout = output.ncols();
    // Reset the retained leaders for each point. No assignment state can pass
    // from one output row to another.
    for (point, (point_dots, point_output)) in dots
        .row_iter()
        .zip(output.as_mut_slice().chunks_exact_mut(fanout))
        .enumerate()
    {
        ranked_leaders.fill((u32::MAX, f32::INFINITY));
        let point_norm_values = point_norms.repeat_simd(arch, point);
        let point_norm = point_norms.read(point);
        // Process all complete SIMD groups first. Single-value rankings use the
        // non-vector operation order.
        let full = leader_count / F::LANES * F::LANES;

        for first_leader in (0..full).step_by(F::LANES) {
            // SAFETY: `first_leader + F::LANES <= full <= point_dots.len()`.
            let dot = unsafe { F::load_simd(arch, point_dots.as_ptr().add(first_leader)) };
            let leader_norm_values = leader_norms.load_simd(arch, first_leader);
            let scores =
                M::partition_ranking_simd(arch, dot, point_norm_values, leader_norm_values);
            insert_leader_lanes(scores, first_leader, ranked_leaders);
        }

        // Use single-value formulas because SIMD padding can change L2 rounding.
        for (leader, &dot) in point_dots.iter().enumerate().skip(full) {
            let leader_norm = leader_norms.read(leader);
            insert_leader(
                ranked_leaders,
                leader as u32,
                M::partition_ranking_single(dot, point_norm, leader_norm),
            );
        }
        if ranked_leaders[fanout - 1].0 == u32::MAX {
            return Err(PartitionKernelError::InsufficientRankableLeaders { point, fanout });
        }
        // Scatter needs sampled-center column IDs. Scores remain in scratch.
        for (destination, &(leader, _)) in point_output.iter_mut().zip(ranked_leaders.iter()) {
            *destination = leader;
        }
    }
    Ok(())
}

/// Offer one SIMD group of sampled centers to the current point's ranked_leaders.
///
/// `first_leader` is the matrix-column ID of the first lane. Lanes enter in
/// sampled-leader order, which preserves tie order.
fn insert_leader_lanes<F>(scores: F, first_leader: usize, ranked_leaders: &mut [(u32, f32)])
where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDPartialOrd,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let threshold = F::splat(scores.arch(), ranked_leaders[ranked_leaders.len() - 1].1);
    let eligible = scores.lt_simd(threshold);
    if eligible.none() {
        return;
    }

    let values: [f32; 16] = scores.to_array();
    let mut lanes = u64::from(eligible.bitmask().to_underlying());
    while lanes != 0 {
        let lane = lanes.trailing_zeros() as usize;
        lanes &= lanes - 1;
        insert_leader(ranked_leaders, (first_leader + lane) as u32, values[lane]);
    }
}

/// Insert one sampled partition center into the current point's retained set.
///
/// `leader` is the center's column ID in the point-to-leader matrix. `ranked_leaders`
/// stores retained centers in nearest-first order. Equal scores and NaN do not
/// enter, so sampled-leader order resolves ties.
#[inline(always)]
fn insert_leader(ranked_leaders: &mut [(u32, f32)], leader: u32, score: f32) {
    let threshold = ranked_leaders.len() - 1;
    if score.partial_cmp(&ranked_leaders[threshold].1) != Some(std::cmp::Ordering::Less) {
        return;
    }

    ranked_leaders[threshold] = (leader, score);
    let mut slot = threshold;
    while slot > 0 && ranked_leaders[slot].1 < ranked_leaders[slot - 1].1 {
        ranked_leaders.swap(slot, slot - 1);
        slot -= 1;
    }
}

#[cfg(test)]
struct DispatchedPartitionCall<'a> {
    input: PartitionInput<'a>,
    output: MutMatrixView<'a, u32>,
    ranked_leaders: &'a mut Vec<(u32, f32)>,
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
            Metric::L2 => {
                nearest_leaders::<A, L2>(arch, call.input, call.output, call.ranked_leaders)
            }
            Metric::Cosine => {
                nearest_leaders::<A, Cosine>(arch, call.input, call.output, call.ranked_leaders)
            }
            Metric::CosineNormalized => nearest_leaders::<A, CosineNormalized>(
                arch,
                call.input,
                call.output,
                call.ranked_leaders,
            ),
            Metric::InnerProduct => nearest_leaders::<A, InnerProduct>(
                arch,
                call.input,
                call.output,
                call.ranked_leaders,
            ),
        }
    }
}

#[cfg(test)]
fn dispatch_nearest_leaders(
    metric: Metric,
    input: PartitionInput<'_>,
    output: MutMatrixView<'_, u32>,
    ranked_leaders: &mut Vec<(u32, f32)>,
) -> Result<(), PartitionKernelError> {
    diskann_wide::arch::dispatch1_no_features(
        DispatchPartitionForTest(metric),
        DispatchedPartitionCall {
            input,
            output,
            ranked_leaders,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::super::kernel_metric::{
        Cosine, CosineNormalized, InnerProduct, L2, PartitionMetric,
    };

    use super::*;
    use diskann_vector::distance::Metric;

    fn test_input<'a>(
        dots: &'a [f32],
        point_count: usize,
        leader_count: usize,
        point_norms: &'a [f32],
        leader_norms: &'a [f32],
    ) -> PartitionInput<'a> {
        PartitionInput {
            dots: MatrixView::try_from(dots, point_count, leader_count).unwrap(),
            norms: PartitionNorms {
                point_norms,
                leader_norms,
            },
        }
    }

    // This oracle checks SIMD groups, single values, and retained-leader order.
    // It uses the single-value ranking formula for metric `M`.
    fn ranking_reference<M: PartitionMetric>(
        input: PartitionInput<'_>,
        fanout: usize,
        output: &mut [u32],
    ) {
        for (point, (point_dots, point_output)) in input
            .dots
            .row_iter()
            .zip(output.chunks_exact_mut(fanout))
            .enumerate()
        {
            let point_norm = input.norms.point_norms.get(point).copied().unwrap_or(0.0);
            let mut ranked_leaders = vec![(u32::MAX, f32::INFINITY); fanout];
            for (leader, &dot) in point_dots.iter().enumerate() {
                insert_leader(
                    &mut ranked_leaders,
                    leader as u32,
                    M::partition_ranking_single(
                        dot,
                        point_norm,
                        input.norms.leader_norms.get(leader).copied().unwrap_or(0.0),
                    ),
                );
            }
            for (destination, &(leader, _)) in point_output.iter_mut().zip(&ranked_leaders) {
                *destination = leader;
            }
        }
    }

    fn single_ranking<M: PartitionMetric>(
        dot_product: f32,
        point_norms: &[f32],
        leader_norms: &[f32],
    ) -> f32 {
        M::partition_ranking_single(
            dot_product,
            point_norms.first().copied().unwrap_or(0.0),
            leader_norms.first().copied().unwrap_or(0.0),
        )
    }

    #[test]
    fn single_ranking_matches_metric_contract() {
        assert_eq!(single_ranking::<L2>(2.0, &[], &[9.0]), 5.0);
        assert_eq!(single_ranking::<CosineNormalized>(0.25, &[], &[]), 0.75);
        assert_eq!(single_ranking::<InnerProduct>(3.0, &[], &[]), -3.0);
        assert_eq!(single_ranking::<Cosine>(4.0, &[2.0], &[4.0]), 0.5);
        assert_eq!(single_ranking::<Cosine>(4.0, &[0.0], &[4.0]), 1.0);
        assert!(single_ranking::<Cosine>(1.0, &[f32::NAN], &[1.0]).is_nan());
    }

    #[test]
    fn cosine_special_norms_match_single_and_dispatched_kernel() {
        let leader_count = 17;
        let point_norms = [0.0, 0.0, f32::MIN_POSITIVE.sqrt(), f32::NAN];
        let dots = vec![1.0; point_norms.len() * leader_count];
        let mut leader_norms = vec![1.0; leader_count];
        leader_norms[..4].copy_from_slice(&[
            0.0,
            f32::MIN_POSITIVE.sqrt() / 2.0,
            f32::MIN_POSITIVE.sqrt(),
            f32::NAN,
        ]);
        let input = test_input(
            &dots,
            point_norms.len(),
            leader_count,
            &point_norms,
            &leader_norms,
        );
        let mut expected = vec![u32::MAX; point_norms.len() * 2];
        ranking_reference::<Cosine>(input, 2, &mut expected);
        let mut actual = vec![u32::MAX; point_norms.len() * 2];
        dispatch_nearest_leaders(
            Metric::Cosine,
            input,
            MutMatrixView::try_from(actual.as_mut_slice(), point_norms.len(), 2).unwrap(),
            &mut Vec::new(),
        )
        .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(&actual[..4], &[0, 1, 0, 1]);
        assert_eq!(&actual[6..], &[0, 1]);
    }

    #[test]
    fn topk_orders_candidates_and_preserves_ties() {
        let mut ranked_leaders = vec![(u32::MAX, f32::INFINITY); 4];
        for (leader, distance) in [(0, 4.0), (1, 1.0), (2, 3.0), (3, 2.0), (4, 1.0)] {
            insert_leader(&mut ranked_leaders, leader, distance);
        }
        insert_leader(&mut ranked_leaders, 5, f32::NAN);

        assert_eq!(ranked_leaders[..], [(1, 1.0), (4, 1.0), (3, 2.0), (2, 3.0)]);
    }

    #[test]
    fn ranked_leaders_reuses_runtime_fanout_capacity() {
        let dots = [0.0; 32];
        let input = test_input(&dots, 1, 32, &[], &[]);
        let mut ranked_leaders = Vec::new();
        let mut wide_output = [u32::MAX; 32];
        dispatch_nearest_leaders(
            Metric::InnerProduct,
            input,
            MutMatrixView::try_from(&mut wide_output[..], 1, 32).unwrap(),
            &mut ranked_leaders,
        )
        .unwrap();
        let allocation = ranked_leaders.as_ptr();

        let mut narrow_output = [u32::MAX; 3];
        dispatch_nearest_leaders(
            Metric::InnerProduct,
            input,
            MutMatrixView::try_from(&mut narrow_output[..], 1, 3).unwrap(),
            &mut ranked_leaders,
        )
        .unwrap();

        assert_eq!(ranked_leaders.as_ptr(), allocation);
        assert_eq!(ranked_leaders.len(), 3);
    }
}
#[cfg(test)]
#[allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "deterministic test fixture construction must abort on invalid setup"
)]
mod integration_tests {
    use super::{PartitionInput, PartitionKernelError, PartitionNorms, dispatch_nearest_leaders};
    use diskann_utils::views::{MatrixView, MutMatrixView};
    use diskann_vector::distance::Metric;

    fn test_input<'a>(
        dots: &'a [f32],
        point_count: usize,
        leader_count: usize,
        point_norms: &'a [f32],
        leader_norms: &'a [f32],
    ) -> PartitionInput<'a> {
        PartitionInput {
            dots: MatrixView::try_from(dots, point_count, leader_count).unwrap(),
            norms: PartitionNorms {
                point_norms,
                leader_norms,
            },
        }
    }

    fn brute_force_reference(input: PartitionInput<'_>, fanout: usize, metric: Metric) -> Vec<u32> {
        let point_count = input.dots.nrows();
        let leader_count = input.dots.ncols();
        let point_norms = input.norms.point_norms;
        let leader_norms = input.norms.leader_norms;
        let mut assignments = vec![u32::MAX; point_count * fanout];
        for (point, (point_dots, point_assignments)) in input
            .dots
            .as_slice()
            .chunks_exact(leader_count)
            .zip(assignments.chunks_exact_mut(fanout))
            .enumerate()
        {
            let point_norm = point_norms.get(point).copied().unwrap_or(0.0);
            let mut candidates: Vec<_> = point_dots
                .iter()
                .enumerate()
                .filter_map(|(leader, &dot)| {
                    let leader_norm = leader_norms.get(leader).copied().unwrap_or(0.0);
                    let score = match metric {
                        Metric::L2 => leader_norm - 2.0 * dot,
                        Metric::CosineNormalized => 1.0 - dot,
                        Metric::InnerProduct => -dot,
                        Metric::Cosine => {
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
        let point_norms = if metric == Metric::Cosine {
            vec![0.0, 4.0]
        } else {
            Vec::new()
        };
        let leader_norms = match metric {
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
        (dots, point_norms, leader_norms)
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
            &mut Vec::new(),
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
                let (dots, point_norms, leader_norms) = differential_data(metric, leader_count);
                let input = test_input(&dots, 2, leader_count, &point_norms, &leader_norms);
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
            run_partition_kernel(Metric::L2, test_input(&dots, 2, 4, &[], &norms), 2).unwrap(),
            [0, 1, 2, 1]
        );
    }

    #[test]
    fn l2_single_can_outrank_a_fused_simd_lane() {
        let mut dots = [0.0; 17];
        dots[0] = f32::MAX;
        dots[16] = f32::MAX;
        let leader_squared_norms = [f32::MAX; 17];

        assert_eq!(
            run_partition_kernel(
                Metric::L2,
                test_input(&dots, 1, 17, &[], &leader_squared_norms,),
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
        for (metric, point_norms, leader_norms, expected) in [
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
                    test_input(&dots, 2, 3, point_norms, leader_norms),
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
                test_input(&[100.0, -100.0], 1, 2, &[0.0], &[1.0, 1.0]),
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
            run_partition_kernel(Metric::InnerProduct, test_input(&dots, 1, 8, &[], &[]), 8)
                .unwrap(),
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn ignores_nan_distances_without_displacing_finite_leaders() {
        assert_eq!(
            run_partition_kernel(
                Metric::InnerProduct,
                test_input(&[f32::NAN, 3.0, 2.0], 1, 3, &[], &[]),
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
                test_input(&[f32::NAN, 3.0], 1, 2, &[], &[]),
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
        run_partition_kernel(Metric::InnerProduct, test_input(&[], 0, 3, &[], &[]), 2).unwrap();
        run_partition_kernel(
            Metric::InnerProduct,
            test_input(&[1.0, 2.0, 3.0], 1, 3, &[], &[]),
            0,
        )
        .unwrap();
        run_partition_kernel(
            Metric::InnerProduct,
            test_input(&[], 0, u32::MAX as usize, &[], &[]),
            0,
        )
        .unwrap();

        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            run_partition_kernel(
                Metric::InnerProduct,
                test_input(&[], 0, u32::MAX as usize + 1, &[], &[],),
                0,
            ),
            Err(PartitionKernelError::TooManyLeaders(u32::MAX as usize + 1))
        );
    }

    #[test]
    fn rejects_wrong_output_norms_and_fanout() {
        let dots = [0.0; 6];
        let valid_input = test_input(&dots, 2, 3, &[], &[]);
        let mut wrong_output = [u32::MAX; 3];
        assert_eq!(
            dispatch_nearest_leaders(
                Metric::InnerProduct,
                valid_input,
                MutMatrixView::try_from(&mut wrong_output[..], 1, 3).unwrap(),
                &mut Vec::new(),
            ),
            Err(PartitionKernelError::InvalidOutputShape {
                expected_rows: 2,
                actual_rows: 1,
                actual_cols: 3,
            })
        );

        let short_norms = [0.0; 2];
        let wrong_norms = PartitionInput {
            dots: MatrixView::try_from(&dots[..], 2, 3).unwrap(),
            norms: PartitionNorms {
                point_norms: &[],
                leader_norms: &short_norms,
            },
        };
        assert_eq!(
            run_partition_kernel(Metric::L2, wrong_norms, 2),
            Err(PartitionKernelError::InvalidBufferLength {
                buffer: "leader norms",
                expected: 3,
                actual: 2,
            })
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
            run_partition_kernel(Metric::InnerProduct, test_input(&one, 1, 1, &[], &[]), 2,),
            Err(PartitionKernelError::InvalidFanout {
                fanout: 2,
                leader_count: 1,
            })
        );
    }
}
