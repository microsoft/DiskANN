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
//! leaders. Equal scores keep sampled-leader order. NaN is not rankable. An
//! unfilled output slot contains [`UNASSIGNED_LEADER`].

use std::marker::PhantomData;

use crate::{ANNError, ANNResult};
use diskann_linalg::Transpose;
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_wide::{SIMDMask, SIMDVector};

use super::{
    kernel_metric::{PartitionMetric, PartitionNorms},
    simd::{PiPNNSIMDSchema, PiPNNSIMDVector},
};

/// No sampled partition center was rankable for this output slot.
pub(super) const UNASSIGNED_LEADER: u32 = u32::MAX;

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
/// A point can have fewer assignments than the output width. Each remaining
/// slot contains [`UNASSIGNED_LEADER`].
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
    A: PiPNNSIMDSchema,
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
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
{
    let fanout = output.ncols();
    if fanout == 0 || input.dots.nrows() == 0 {
        return;
    }

    ranked_leaders.resize(fanout, (UNASSIGNED_LEADER, f32::INFINITY));
    select_point_leaders::<A, M>(arch, input.dots, input.norms, output, ranked_leaders);
}

/// Rank sampled partition centers for each assigned point.
///
/// The function keeps nearest-first order for every point. Full SIMD groups use
/// metric-specific formulas. Remaining leaders use the matching single formula.
fn select_point_leaders<A, M>(
    arch: A,
    dots: MatrixView<'_, f32>,
    norms: PartitionNorms<'_>,
    mut output: MutMatrixView<'_, u32>,
    ranked_leaders: &mut [(u32, f32)],
) where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
{
    let leader_count = dots.ncols();
    let fanout = output.ncols();

    for (point, (point_dots, point_output)) in dots
        .row_iter()
        .zip(output.as_mut_slice().chunks_exact_mut(fanout))
        .enumerate()
    {
        ranked_leaders.fill((UNASSIGNED_LEADER, f32::INFINITY));
        let point_simd = M::point_simd(arch, norms, point);
        let point_single = M::point_single(norms, point);
        let simd_prefix = leader_count - leader_count % M::Simd::<A>::LANES;

        for first_leader in (0..simd_prefix).step_by(M::Simd::<A>::LANES) {
            // SAFETY: This group is inside the point's leader row.
            let dot_products =
                unsafe { M::Simd::<A>::load_simd(arch, point_dots.as_ptr().add(first_leader)) };
            let rankings = M::rankings_simd(arch, norms, point_simd, dot_products, first_leader);
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
    F: PiPNNSIMDVector,
{
    let threshold = F::splat(scores.arch(), ranked_leaders[ranked_leaders.len() - 1].1);
    let eligible = scores.lt_simd(threshold);
    if eligible.none() {
        return;
    }

    let values = scores.to_array();
    let values = values.as_ref();
    let mut lanes = F::active_lanes(eligible);
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
mod tests {
    use super::*;
    use crate::graph::pipnn::kernel_metric::{Cosine, CosineNormalized, InnerProduct, L2};
    use diskann_utils::views::{Matrix, MatrixView, MutMatrixView};
    use diskann_vector::distance::Metric;
    use diskann_wide::arch::{self, Target1};

    struct KernelCall<'a> {
        input: PartitionInput<'a>,
        output: MutMatrixView<'a, u32>,
        ranked_leaders: &'a mut Vec<(u32, f32)>,
    }

    struct DispatchMetric(Metric);

    impl<A> Target1<A, (), KernelCall<'_>> for DispatchMetric
    where
        A: PiPNNSIMDSchema,
    {
        fn run(self, arch: A, call: KernelCall<'_>) {
            match self.0 {
                Metric::L2 => {
                    rank_leader_dots::<A, L2>(arch, call.input, call.output, call.ranked_leaders)
                }
                Metric::Cosine => rank_leader_dots::<A, Cosine>(
                    arch,
                    call.input,
                    call.output,
                    call.ranked_leaders,
                ),
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

    fn partition_input<'a>(
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

    fn rank_partition_leaders(
        metric: Metric,
        input: PartitionInput<'_>,
        fanout: usize,
    ) -> Vec<u32> {
        let mut output = Matrix::new(u32::MAX, input.dots.nrows(), fanout);
        arch::dispatch1_no_features(
            DispatchMetric(metric),
            KernelCall {
                input,
                output: output.as_mut_view(),
                ranked_leaders: &mut Vec::new(),
            },
        );
        output.into_inner().into_vec()
    }

    fn reference_score(metric: Metric, dot: f32, point_norm: f32, leader_norm: f32) -> f32 {
        match metric {
            Metric::L2 => (-2.0_f32).mul_add(dot, leader_norm),
            Metric::CosineNormalized => 1.0 - dot,
            Metric::InnerProduct => -dot,
            Metric::Cosine => {
                if point_norm < f32::MIN_POSITIVE.sqrt() || leader_norm < f32::MIN_POSITIVE.sqrt() {
                    1.0
                } else {
                    1.0 - (dot / (point_norm * leader_norm)).clamp(-1.0, 1.0)
                }
            }
        }
    }

    fn reference_assignments(metric: Metric, input: PartitionInput<'_>, fanout: usize) -> Vec<u32> {
        let mut output = vec![UNASSIGNED_LEADER; input.dots.nrows() * fanout];
        for (point, (dots, assignments)) in input
            .dots
            .row_iter()
            .zip(output.chunks_exact_mut(fanout))
            .enumerate()
        {
            let point_norm = input.norms.point_norms.get(point).copied().unwrap_or(0.0);
            let mut candidates: Vec<_> = dots
                .iter()
                .enumerate()
                .filter_map(|(leader, &dot)| {
                    let leader_norm = input.norms.leader_norms.get(leader).copied().unwrap_or(0.0);
                    let score = reference_score(metric, dot, point_norm, leader_norm);
                    (score.partial_cmp(&f32::INFINITY) == Some(std::cmp::Ordering::Less))
                        .then_some((leader as u32, score))
                })
                .collect();
            candidates.sort_by(|left, right| left.1.total_cmp(&right.1));
            for (destination, (leader, _)) in assignments.iter_mut().zip(candidates) {
                *destination = leader;
            }
        }
        output
    }

    /// Build ranking input from two axis points and leaders between those axes.
    ///
    /// The first point prefers early leaders. The second point prefers late leaders.
    fn lane_boundary_input_from_point_and_leader_vectors(
        metric: Metric,
        leader_count: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let points = [[1.0_f32, 0.0], [0.0, 1.0]];
        let denominator = (leader_count + 1) as f32;
        let leaders: Vec<_> = (0..leader_count)
            .map(|leader| {
                let second_component = (leader + 1) as f32 / denominator;
                let vector = [1.0 - second_component, second_component];
                if metric == Metric::CosineNormalized {
                    let norm = vector[0].hypot(vector[1]);
                    [vector[0] / norm, vector[1] / norm]
                } else {
                    vector
                }
            })
            .collect();
        let dots = points
            .iter()
            .flat_map(|point| {
                leaders
                    .iter()
                    .map(|leader| point[0] * leader[0] + point[1] * leader[1])
            })
            .collect();
        let point_norms = if metric == Metric::Cosine {
            points
                .iter()
                .map(|point| point[0].hypot(point[1]))
                .collect()
        } else {
            Vec::new()
        };
        let leader_norms = match metric {
            Metric::L2 => leaders
                .iter()
                .map(|leader| leader[0] * leader[0] + leader[1] * leader[1])
                .collect(),
            Metric::Cosine => leaders
                .iter()
                .map(|leader| leader[0].hypot(leader[1]))
                .collect(),
            Metric::CosineNormalized | Metric::InnerProduct => Vec::new(),
        };
        (dots, point_norms, leader_norms)
    }

    mod insert_leader_tests {
        use super::*;

        #[test]
        fn topk_keeps_nearest_first_order_and_scan_order_ties() {
            // Given
            let expected_ranked_leaders = [(1, 1.0), (4, 1.0), (3, 2.0), (2, 3.0)];
            let mut ranked_leaders = vec![(UNASSIGNED_LEADER, f32::INFINITY); 4];

            // When
            insert_leader(&mut ranked_leaders, 0, 4.0);
            insert_leader(&mut ranked_leaders, 1, 1.0);
            insert_leader(&mut ranked_leaders, 2, 3.0);
            insert_leader(&mut ranked_leaders, 3, 2.0);
            insert_leader(&mut ranked_leaders, 4, 1.0);

            // Then
            assert_eq!(ranked_leaders, expected_ranked_leaders);
        }

        #[test]
        fn nan_score_does_not_enter_the_topk() {
            // Given
            let expected_ranked_leaders = [(0, 0.25), (UNASSIGNED_LEADER, f32::INFINITY)];
            let mut ranked_leaders = vec![(UNASSIGNED_LEADER, f32::INFINITY); 2];

            // When
            insert_leader(&mut ranked_leaders, 0, 0.25);
            insert_leader(&mut ranked_leaders, 1, f32::NAN);

            // Then
            assert_eq!(ranked_leaders, expected_ranked_leaders);
        }
    }

    mod assign_leaders_tests {
        use super::*;

        #[test]
        fn assigns_each_point_to_highest_similarity_leaders_with_cosine() {
            // Given
            let leader_values = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
            let leaders = PreparedLeaders::<Cosine>::new(
                MatrixView::try_from(&leader_values[..], 3, 2).unwrap(),
            );
            let point_values = [0.9, 0.1, -0.8, 0.2];
            let points = MatrixView::try_from(&point_values[..], 2, 2).unwrap();
            let expected_leaders_by_descending_cosine_similarity = [0, 1, 2, 1];
            let mut actual_assignments = [UNASSIGNED_LEADER; 4];

            // When
            assign_leaders::<_, Cosine>(
                diskann_wide::ARCH,
                points,
                &leaders,
                MutMatrixView::try_from(&mut actual_assignments[..], 2, 2).unwrap(),
                &mut PartitionKernelWorkspace::default(),
            )
            .unwrap();

            // Then
            assert_eq!(
                actual_assignments,
                expected_leaders_by_descending_cosine_similarity
            );
        }

        #[test]
        fn reused_workspace_matches_fresh_leader_assignment() {
            // Given
            let leader_values = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
            let leaders = PreparedLeaders::<Cosine>::new(
                MatrixView::try_from(&leader_values[..], 3, 2).unwrap(),
            );
            let point_values = [0.9, 0.1, -0.8, 0.2];
            let smaller_points = MatrixView::try_from(&point_values[..2], 1, 2).unwrap();
            let mut reused_workspace = PartitionKernelWorkspace::default();
            let mut discarded_large_output = [UNASSIGNED_LEADER; 4];
            assign_leaders::<_, Cosine>(
                diskann_wide::ARCH,
                MatrixView::try_from(&point_values[..], 2, 2).unwrap(),
                &leaders,
                MutMatrixView::try_from(&mut discarded_large_output[..], 2, 2).unwrap(),
                &mut reused_workspace,
            )
            .unwrap();
            let mut expected_assignments_from_fresh_workspace = [UNASSIGNED_LEADER; 2];
            assign_leaders::<_, Cosine>(
                diskann_wide::ARCH,
                smaller_points,
                &leaders,
                MutMatrixView::try_from(&mut expected_assignments_from_fresh_workspace[..], 1, 2)
                    .unwrap(),
                &mut PartitionKernelWorkspace::default(),
            )
            .unwrap();

            // When
            let mut actual_assignments_from_reused_workspace = [UNASSIGNED_LEADER; 2];
            assign_leaders::<_, Cosine>(
                diskann_wide::ARCH,
                smaller_points,
                &leaders,
                MutMatrixView::try_from(&mut actual_assignments_from_reused_workspace[..], 1, 2)
                    .unwrap(),
                &mut reused_workspace,
            )
            .unwrap();

            // Then
            assert_eq!(
                actual_assignments_from_reused_workspace,
                expected_assignments_from_fresh_workspace
            );
        }
    }

    mod rank_leader_dots_tests {
        use super::*;
        use rstest::rstest;

        #[rstest]
        #[case::two_leaders_fanout_one(2, 1)]
        #[case::scalar_fanout_two(7, 2)]
        #[case::lane_minus_one(15, 3)]
        #[case::one_complete_lane(16, 3)]
        #[case::lane_plus_one(17, 4)]
        #[case::two_lanes_minus_one(31, 7)]
        #[case::two_complete_lanes(32, 7)]
        #[case::two_lanes_plus_one(33, 7)]
        #[trace]
        fn dispatched_partition_ranking_matches_scalar_reference_across_lane_boundaries(
            #[values(
                Metric::L2,
                Metric::Cosine,
                Metric::CosineNormalized,
                Metric::InnerProduct
            )]
            metric: Metric,
            #[case] leader_count: usize,
            #[case] fanout: usize,
        ) {
            // Given
            let (dots, point_norms, leader_norms) =
                lane_boundary_input_from_point_and_leader_vectors(metric, leader_count);
            let input = partition_input(&dots, 2, leader_count, &point_norms, &leader_norms);
            let expected_assignments = reference_assignments(metric, input, fanout);

            // When
            let actual_assignments = rank_partition_leaders(metric, input, fanout);

            // Then
            assert_eq!(actual_assignments, expected_assignments);
        }

        #[test]
        fn equal_scores_keep_sampled_leader_order_with_l2() {
            // Given
            let dots = [0.0, 0.0, 0.0, 0.0];
            let leader_squared_norms = [1.0, 1.0, 1.0, 1.0];
            let expected_sampled_leader_order = [0, 1];

            // When
            let actual_assignments = rank_partition_leaders(
                Metric::L2,
                partition_input(&dots, 1, 4, &[], &leader_squared_norms),
                2,
            );

            // Then
            assert_eq!(actual_assignments, expected_sampled_leader_order);
        }

        #[test]
        fn zero_norm_keeps_sampled_leader_order_with_cosine() {
            // Given
            let dots = [0.0, 0.0];
            let point_norms = [0.0];
            let leader_norms = [1.0, 1.0];
            let expected_sampled_leader_order = [0, 1];

            // When
            let actual_assignments = rank_partition_leaders(
                Metric::Cosine,
                partition_input(&dots, 1, 2, &point_norms, &leader_norms),
                2,
            );

            // Then
            assert_eq!(actual_assignments, expected_sampled_leader_order);
        }

        #[test]
        fn f32_max_score_is_still_a_rankable_leader() {
            // Given
            let maximum_rankable_score = f32::MAX;
            let dot_product_that_produces_it = -maximum_rankable_score;
            let mut dots = [0.0; 8];
            dots[7] = dot_product_that_produces_it;
            let expected_all_leaders_in_scan_order = [0, 1, 2, 3, 4, 5, 6, 7];

            // When
            let actual_assignments = rank_partition_leaders(
                Metric::InnerProduct,
                partition_input(&dots, 1, 8, &[], &[]),
                8,
            );

            // Then
            assert_eq!(actual_assignments, expected_all_leaders_in_scan_order);
        }

        #[test]
        fn nan_leader_does_not_displace_finite_leaders() {
            // Given
            let dots = [f32::NAN, 3.0, 2.0];
            let expected_finite_leaders = [1, 2];

            // When
            let actual_assignments = rank_partition_leaders(
                Metric::InnerProduct,
                partition_input(&dots, 1, 3, &[], &[]),
                2,
            );

            // Then
            assert_eq!(actual_assignments, expected_finite_leaders);
        }

        #[test]
        fn empty_point_matrix_produces_no_assignments() {
            // Given
            let dots = [];
            let expected_no_assignments: [u32; 0] = [];

            // When
            let actual_assignments = rank_partition_leaders(
                Metric::InnerProduct,
                partition_input(&dots, 0, 3, &[], &[]),
                2,
            );

            // Then
            assert_eq!(actual_assignments, expected_no_assignments);
        }

        #[test]
        fn zero_fanout_produces_no_assignments() {
            // Given
            let dots = [1.0, 2.0, 3.0];
            let expected_no_assignments: [u32; 0] = [];

            // When
            let actual_assignments = rank_partition_leaders(
                Metric::InnerProduct,
                partition_input(&dots, 1, 3, &[], &[]),
                0,
            );

            // Then
            assert_eq!(actual_assignments, expected_no_assignments);
        }
    }
}
