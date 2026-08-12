/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Select partition centers for PiPNN point assignment.
//!
//! A leader is a sampled dataset point that represents one child partition.
//! The kernel prepares reusable leader norms, computes point-to-leader dot
//! products, and returns nearest leader-column IDs for partition scatter.
//!
//! L2 omits the assigned point's norm because it is constant across all sampled
//! leaders. Equal scores keep sampled-leader order. NaN is not rankable.

use std::marker::PhantomData;

use crate::{ANNError, ANNResult};
use diskann_linalg::Transpose;
use diskann_utils::views::{MatrixView, MutMatrixView};
#[cfg(test)]
use diskann_vector::distance::Metric;
use diskann_wide::{
    Architecture, Const, SIMDFloat, SIMDMask, SIMDPartialOrd, SIMDSelect, SIMDVector,
};

use super::kernel_metric::{PartitionMetric, PartitionNorms};

/// Sampled leader vectors with metric-specific reusable norms.
pub(super) struct PreparedLeaders<'a, M> {
    leader_values: MatrixView<'a, f32>,
    leader_norms: Vec<f32>,
    metric: PhantomData<M>,
}

impl<'a, M> PreparedLeaders<'a, M>
where
    M: PartitionMetric,
{
    /// Prepare leader state for all point stripes in one partition split.
    pub(super) fn new(leader_values: MatrixView<'a, f32>) -> Self {
        let mut leader_norms = Vec::new();
        M::prepare_leader_norms(leader_values, &mut leader_norms);
        Self {
            leader_values,
            leader_norms,
            metric: PhantomData,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.leader_values.nrows()
    }
}

/// Reusable storage for one point-stripe numerical pipeline.
#[derive(Default)]
pub(super) struct PartitionKernelWorkspace {
    dot_scratch: Vec<f32>,
    point_norm_scratch: Vec<f32>,
    ranked_leader_scratch: Vec<(u32, f32)>,
}

/// Dot products between assigned points and sampled partition centers.
///
/// Each row is one point being assigned. Each column is one sampled leader.
/// [`Self::norms`] supplies the norm layout for metric `M`.
#[derive(Clone, Copy, Debug)]
struct PartitionInput<'a> {
    dots: MatrixView<'a, f32>,
    norms: PartitionNorms<'a>,
}

/// Assign one packed point stripe to prepared partition leaders.
///
/// # Errors
///
/// Returns an error for invalid GEMM input.
pub(super) fn assign_leaders<A, M>(
    arch: A,
    points: MatrixView<'_, f32>,
    leaders: &PreparedLeaders<'_, M>,
    output: MutMatrixView<'_, u32>,
    workspace: &mut PartitionKernelWorkspace,
) -> ANNResult<()>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: PartitionMetric,
{
    let point_count = points.nrows();
    let leader_count = leaders.len();
    let dot_count = point_count * leader_count;
    let PartitionKernelWorkspace {
        dot_scratch,
        point_norm_scratch,
        ranked_leader_scratch,
    } = workspace;
    if dot_scratch.len() < dot_count {
        dot_scratch.resize(dot_count, 0.0);
    }
    diskann_linalg::sgemm(
        Transpose::None,
        Transpose::Ordinary,
        point_count,
        leader_count,
        points.ncols(),
        1.0,
        points.as_slice(),
        leaders.leader_values.as_slice(),
        None,
        &mut dot_scratch[..dot_count],
    )
    .map_err(ANNError::new)?;
    M::prepare_point_norms(points, point_norm_scratch);
    let dots = MatrixView::try_from(&dot_scratch[..dot_count], point_count, leader_count)
        .map_err(|error| ANNError::new(error.as_static()))?;
    rank_leader_dots::<A, M>(
        arch,
        PartitionInput {
            dots,
            norms: PartitionNorms {
                point_norms: point_norm_scratch,
                leader_norms: &leaders.leader_norms,
            },
        },
        output,
        ranked_leader_scratch,
    );
    Ok(())
}

/// Rank prepared point-to-leader dot products.
fn rank_leader_dots<A, M>(
    arch: A,
    input: PartitionInput<'_>,
    output: MutMatrixView<'_, u32>,
    ranked_leaders: &mut Vec<(u32, f32)>,
) where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: PartitionMetric,
{
    let fanout = output.ncols();
    if fanout == 0 || input.dots.nrows() == 0 {
        return;
    }

    ranked_leaders.resize(fanout, (u32::MAX, f32::INFINITY));
    select_point_leaders::<A::f32x16, M>(arch, input.dots, input.norms, output, ranked_leaders);
}

/// Rank sampled partition centers for each assigned point.
///
/// The function keeps nearest-first order for every point. Full SIMD groups use
/// metric-specific formulas. Remaining leaders use the matching single formula.
fn select_point_leaders<F, M>(
    arch: F::Arch,
    dots: MatrixView<'_, f32>,
    norms: PartitionNorms<'_>,
    mut output: MutMatrixView<'_, u32>,
    ranked_leaders: &mut [(u32, f32)],
) where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<16>> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    M: PartitionMetric,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    let leader_count = dots.ncols();
    let fanout = output.ncols();

    for (point, (point_dots, point_output)) in dots
        .row_iter()
        .zip(output.as_mut_slice().chunks_exact_mut(fanout))
        .enumerate()
    {
        ranked_leaders.fill((u32::MAX, f32::INFINITY));
        let point_simd = M::point_simd::<F>(arch, norms, point);
        let point_single = M::point_single(norms, point);
        let simd_prefix = leader_count - leader_count % F::LANES;

        for first_leader in (0..simd_prefix).step_by(F::LANES) {
            // SAFETY: This group is inside the point's leader row.
            let dot_products = unsafe { F::load_simd(arch, point_dots.as_ptr().add(first_leader)) };
            let rankings =
                M::rankings_simd::<F>(arch, norms, point_simd, dot_products, first_leader);
            insert_leader_lanes(rankings, first_leader, ranked_leaders);
        }

        for (leader, &dot_product) in point_dots.iter().enumerate().skip(simd_prefix) {
            let ranking = M::ranking_single(norms, point_single, dot_product, leader);
            insert_leader(ranked_leaders, leader as u32, ranking);
        }
        for (destination, &(leader, _)) in point_output.iter_mut().zip(ranked_leaders.iter()) {
            *destination = leader;
        }
    }
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
impl<A> diskann_wide::arch::Target1<A, (), DispatchedPartitionCall<'_>> for DispatchPartitionForTest
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn run(self, arch: A, call: DispatchedPartitionCall<'_>) {
        use super::kernel_metric::{Cosine, CosineNormalized, InnerProduct, L2};

        match self.0 {
            Metric::L2 => {
                rank_leader_dots::<A, L2>(arch, call.input, call.output, call.ranked_leaders)
            }
            Metric::Cosine => {
                rank_leader_dots::<A, Cosine>(arch, call.input, call.output, call.ranked_leaders)
            }
            Metric::CosineNormalized => rank_leader_dots::<A, CosineNormalized>(
                arch,
                call.input,
                call.output,
                call.ranked_leaders,
            ),
            Metric::InnerProduct => rank_leader_dots::<A, InnerProduct>(
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
) {
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
            let point_norm = M::point_single(input.norms, point);
            let mut ranked_leaders = vec![(u32::MAX, f32::INFINITY); fanout];
            for (leader, &dot) in point_dots.iter().enumerate() {
                insert_leader(
                    &mut ranked_leaders,
                    leader as u32,
                    M::ranking_single(input.norms, point_norm, dot, leader),
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
        let norms = PartitionNorms {
            point_norms,
            leader_norms,
        };
        M::ranking_single(norms, M::point_single(norms, 0), dot_product, 0)
    }

    #[test]
    fn single_ranking_matches_metric_contract() {
        assert_eq!(single_ranking::<L2>(2.0, &[], &[9.0]), 5.0);
        assert_eq!(single_ranking::<CosineNormalized>(0.25, &[], &[]), 0.75);
        assert_eq!(single_ranking::<InnerProduct>(3.0, &[], &[]), -3.0);
        assert_eq!(single_ranking::<Cosine>(4.0, &[2.0], &[4.0]), 0.5);
        assert_eq!(single_ranking::<Cosine>(5.0, &[2.0], &[2.0]), 0.0);
        assert_eq!(single_ranking::<Cosine>(-5.0, &[2.0], &[2.0]), 2.0);
        assert_eq!(single_ranking::<Cosine>(4.0, &[0.0], &[4.0]), 1.0);
        assert_eq!(single_ranking::<Cosine>(1.0, &[f32::NAN], &[1.0]), 0.0);
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
        );

        assert_eq!(actual, expected);
        assert_eq!(&actual[..4], &[0, 1, 0, 1]);
        assert_eq!(&actual[6..], &[2, 3]);
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
    fn vector_pipeline_assigns_cosine_leaders_and_reuses_workspace() {
        let leader_values = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let leaders =
            PreparedLeaders::<Cosine>::new(MatrixView::try_from(&leader_values[..], 3, 2).unwrap());
        let point_values = [0.9, 0.1, -0.8, 0.2];
        let points = MatrixView::try_from(&point_values[..], 2, 2).unwrap();
        let mut output = [u32::MAX; 4];
        let mut workspace = PartitionKernelWorkspace::default();
        assign_leaders::<_, Cosine>(
            diskann_wide::ARCH,
            points,
            &leaders,
            MutMatrixView::try_from(&mut output[..], 2, 2).unwrap(),
            &mut workspace,
        )
        .unwrap();

        assert_eq!(output, [0, 1, 2, 1]);
        let dot_scratch = workspace.dot_scratch.as_ptr();
        let point_norm_scratch = workspace.point_norm_scratch.as_ptr();
        let ranked_leader_scratch = workspace.ranked_leader_scratch.as_ptr();

        let mut smaller_output = [u32::MAX; 2];
        assign_leaders::<_, Cosine>(
            diskann_wide::ARCH,
            MatrixView::try_from(&point_values[..2], 1, 2).unwrap(),
            &leaders,
            MutMatrixView::try_from(&mut smaller_output[..], 1, 2).unwrap(),
            &mut workspace,
        )
        .unwrap();

        assert_eq!(smaller_output, [0, 1]);
        assert_eq!(workspace.dot_scratch.as_ptr(), dot_scratch);
        assert_eq!(workspace.point_norm_scratch.as_ptr(), point_norm_scratch);
        assert_eq!(
            workspace.ranked_leader_scratch.as_ptr(),
            ranked_leader_scratch
        );
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
        );
        let allocation = ranked_leaders.as_ptr();

        let mut narrow_output = [u32::MAX; 3];
        dispatch_nearest_leaders(
            Metric::InnerProduct,
            input,
            MutMatrixView::try_from(&mut narrow_output[..], 1, 3).unwrap(),
            &mut ranked_leaders,
        );

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
    use super::{PartitionInput, PartitionNorms, dispatch_nearest_leaders};
    use diskann_utils::views::{Matrix, MatrixView};
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
                        Metric::L2 => (-2.0_f32).mul_add(dot, leader_norm),
                        Metric::CosineNormalized => 1.0 - dot,
                        Metric::InnerProduct => -dot,
                        Metric::Cosine => {
                            1.0 - if point_norm == 0.0 || leader_norm == 0.0 {
                                0.0
                            } else {
                                let cosine = dot / (point_norm * leader_norm);
                                (-1.0_f32).max(1.0_f32.min(cosine))
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

    fn run_partition_kernel(metric: Metric, input: PartitionInput<'_>, fanout: usize) -> Vec<u32> {
        let mut output = Matrix::new(u32::MAX, input.dots.nrows(), fanout);
        dispatch_nearest_leaders(metric, input, output.as_mut_view(), &mut Vec::new());
        output.into_inner().into_vec()
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
                        run_partition_kernel(metric, input, fanout),
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
            run_partition_kernel(Metric::L2, test_input(&dots, 2, 4, &[], &norms), 2),
            [0, 1, 2, 1]
        );
    }

    #[test]
    fn l2_single_matches_fused_simd_ranking() {
        let mut dots = [0.0; 17];
        dots[0] = f32::MAX;
        dots[16] = f32::MAX;
        let leader_squared_norms = [f32::MAX; 17];

        assert_eq!(
            run_partition_kernel(
                Metric::L2,
                test_input(&dots, 1, 17, &[], &leader_squared_norms,),
                1,
            ),
            [0]
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
                ),
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
            ),
            [0, 1]
        );
    }

    #[test]
    fn finite_max_distance_fills_the_final_simd_slot() {
        let mut dots = [0.0; 8];
        dots[7] = -f32::MAX;
        assert_eq!(
            run_partition_kernel(Metric::InnerProduct, test_input(&dots, 1, 8, &[], &[]), 8),
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
            ),
            [1, 2]
        );
    }

    #[test]
    fn accepts_empty_points_and_zero_fanout() {
        assert!(
            run_partition_kernel(Metric::InnerProduct, test_input(&[], 0, 3, &[], &[]), 2)
                .is_empty()
        );
        assert!(
            run_partition_kernel(
                Metric::InnerProduct,
                test_input(&[1.0, 2.0, 3.0], 1, 3, &[], &[]),
                0,
            )
            .is_empty()
        );
    }
}
