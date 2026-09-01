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
//!
//! The GEMM stores one leader per row. Each complete SIMD group assigns one point
//! to each lane and keeps runtime-width score and ID rank vectors. A scalar tail
//! handles the remaining points.

use std::marker::PhantomData;

use crate::{ANNError, ANNResult};
use diskann_linalg::Transpose;
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_wide::{SIMDMask, SIMDPartialOrd, SIMDVector};

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
///
/// The architecture type keeps score and exact ID vectors in reusable storage.
pub(super) struct PartitionKernelWorkspace<A>
where
    A: PiPNNSIMDSchema,
{
    dot_scratch: Vec<f32>,
    point_norm_scratch: Vec<f32>,
    ranking_scratch: PartitionRankScratch<A>,
}

impl<A> Default for PartitionKernelWorkspace<A>
where
    A: PiPNNSIMDSchema,
{
    fn default() -> Self {
        Self {
            dot_scratch: Vec::new(),
            point_norm_scratch: Vec::new(),
            ranking_scratch: PartitionRankScratch::default(),
        }
    }
}

struct PartitionRankScratch<A>
where
    A: PiPNNSIMDSchema,
{
    scores: Vec<A::Vector>,
    ids: Vec<A::IdVector>,
    id_lanes: Vec<u32>,
    scalar: Vec<(u32, f32)>,
}

impl<A> Default for PartitionRankScratch<A>
where
    A: PiPNNSIMDSchema,
{
    fn default() -> Self {
        Self {
            scores: Vec::new(),
            ids: Vec::new(),
            id_lanes: Vec::new(),
            scalar: Vec::new(),
        }
    }
}

/// Dot products between assigned points and sampled partition centers.
///
/// Each row is one sampled leader. Each column is one assigned point.
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
    workspace: &mut PartitionKernelWorkspace<A>,
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
        ranking_scratch,
    } = workspace;
    if dot_scratch.len() < dot_count {
        dot_scratch.resize(dot_count, 0.0);
    }
    diskann_linalg::sgemm(
        Transpose::None,
        Transpose::Ordinary,
        leader_count,
        point_count,
        points.ncols(),
        1.0,
        leaders.leader_values.as_slice(),
        points.as_slice(),
        None,
        &mut dot_scratch[..dot_count],
    )
    .map_err(ANNError::new)?;
    M::prepare_point_norms(points, point_norm_scratch);
    let dots = MatrixView::try_from(&dot_scratch[..dot_count], leader_count, point_count)
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
        ranking_scratch,
    );
    Ok(())
}

/// Rank prepared leader-to-point dot products.
fn rank_leader_dots<A, M>(
    arch: A,
    input: PartitionInput<'_>,
    output: MutMatrixView<'_, u32>,
    scratch: &mut PartitionRankScratch<A>,
) where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
{
    let fanout = output.ncols();
    if fanout == 0 || input.dots.ncols() == 0 {
        return;
    }

    arch.run(move || {
        rank_leader_dots_target::<A, M>(arch, input, output, scratch);
    });
}

/// Keep one independent Top-K in each SIMD point lane.
#[inline(always)]
fn rank_leader_dots_target<A, M>(
    arch: A,
    input: PartitionInput<'_>,
    mut output: MutMatrixView<'_, u32>,
    scratch: &mut PartitionRankScratch<A>,
) where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
{
    let point_count = input.dots.ncols();
    let fanout = output.ncols();
    let lanes = A::Vector::LANES;
    let simd_points = point_count - point_count % lanes;
    let infinity = A::Vector::splat(arch, f32::INFINITY);
    let unassigned_ids = A::IdVector::splat(arch, UNASSIGNED_LEADER);

    scratch.scores.resize(fanout, infinity);
    scratch.ids.resize(fanout, unassigned_ids);
    scratch.id_lanes.resize(fanout * lanes, UNASSIGNED_LEADER);

    for first_point in (0..simd_points).step_by(lanes) {
        scratch.scores.fill(infinity);
        scratch.ids.fill(unassigned_ids);
        let point_norms = M::point_group_simd(arch, input.norms, first_point);

        for (leader, leader_dots) in input.dots.row_iter().enumerate() {
            // SAFETY: The row has `point_count` values. This group ends at or before
            // `simd_points`, so the complete load stays in the row.
            let dot_products =
                unsafe { A::Vector::load_simd(arch, leader_dots.as_ptr().add(first_point)) };
            let scores =
                M::rankings_transposed_simd(arch, input.norms, point_norms, dot_products, leader);
            insert_leader_group::<A>(
                arch,
                scores,
                leader as u32,
                &mut scratch.scores,
                &mut scratch.ids,
            );
        }

        for (rank, ids) in scratch.ids.iter().copied().enumerate() {
            // SAFETY: The scratch has `fanout * lanes` elements. Each rank starts
            // one complete vector inside that range.
            unsafe { ids.store_simd(scratch.id_lanes.as_mut_ptr().add(rank * lanes)) };
        }
        for lane in 0..lanes {
            let point_output = &mut output.as_mut_slice()
                [(first_point + lane) * fanout..(first_point + lane + 1) * fanout];
            for (rank, destination) in point_output.iter_mut().enumerate() {
                *destination = scratch.id_lanes[rank * lanes + lane];
            }
        }
    }

    scratch
        .scalar
        .resize(fanout, (UNASSIGNED_LEADER, f32::INFINITY));
    for point in simd_points..point_count {
        scratch.scalar.fill((UNASSIGNED_LEADER, f32::INFINITY));
        let point_norm = M::point_single(input.norms, point);
        for (leader, leader_dots) in input.dots.row_iter().enumerate() {
            let score = M::ranking_single(input.norms, point_norm, leader_dots[point], leader);
            insert_leader(&mut scratch.scalar, leader as u32, score);
        }
        let point_output = &mut output.as_mut_slice()[point * fanout..(point + 1) * fanout];
        for (destination, &(leader, _)) in point_output.iter_mut().zip(scratch.scalar.iter()) {
            *destination = leader;
        }
    }
}

/// Insert one leader into every eligible SIMD point lane.
#[inline(always)]
fn insert_leader_group<A>(
    arch: A,
    scores: A::Vector,
    leader: u32,
    ranked_scores: &mut [A::Vector],
    ranked_ids: &mut [A::IdVector],
) where
    A: PiPNNSIMDSchema,
{
    let eligible = scores.lt_simd(ranked_scores[ranked_scores.len() - 1]);
    if eligible.none() {
        return;
    }

    let mut candidate_score = scores;
    let mut candidate_id = A::IdVector::splat(arch, leader);
    for (best_score, best_id) in ranked_scores.iter_mut().zip(ranked_ids) {
        let better = candidate_score.lt_simd(*best_score);
        let previous_score = *best_score;
        let previous_id = *best_id;
        *best_score = A::Vector::select(better, candidate_score, previous_score);
        *best_id = A::select_ids(better, candidate_id, previous_id);
        candidate_score = A::Vector::select(better, previous_score, candidate_score);
        candidate_id = A::select_ids(better, previous_id, candidate_id);
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

    mod test_support {
        use super::*;
        use diskann_wide::arch::{self, Target1};

        struct KernelCall<'a> {
            input: PartitionInput<'a>,
            output: MutMatrixView<'a, u32>,
        }

        struct DispatchMetric(Metric);

        impl<A> Target1<A, (), KernelCall<'_>> for DispatchMetric
        where
            A: PiPNNSIMDSchema,
        {
            fn run(self, arch: A, call: KernelCall<'_>) {
                let mut scratch = PartitionRankScratch::default();
                match self.0 {
                    Metric::L2 => {
                        rank_leader_dots::<A, L2>(arch, call.input, call.output, &mut scratch)
                    }
                    Metric::Cosine => {
                        rank_leader_dots::<A, Cosine>(arch, call.input, call.output, &mut scratch)
                    }
                    Metric::CosineNormalized => rank_leader_dots::<A, CosineNormalized>(
                        arch,
                        call.input,
                        call.output,
                        &mut scratch,
                    ),
                    Metric::InnerProduct => rank_leader_dots::<A, InnerProduct>(
                        arch,
                        call.input,
                        call.output,
                        &mut scratch,
                    ),
                }
            }
        }

        pub(super) fn partition_input<'a>(
            dots: &'a [f32],
            point_count: usize,
            leader_count: usize,
            point_norms: &'a [f32],
            leader_norms: &'a [f32],
        ) -> PartitionInput<'a> {
            PartitionInput {
                dots: MatrixView::try_from(dots, leader_count, point_count).unwrap(),
                norms: PartitionNorms {
                    point_norms,
                    leader_norms,
                },
            }
        }

        /// Run the production ranker through runtime architecture and metric dispatch.
        pub(super) fn run_rank_leader_dots(
            metric: Metric,
            input: PartitionInput<'_>,
            nearest_leader_count: usize,
        ) -> Vec<u32> {
            let mut output = Matrix::new(u32::MAX, input.dots.ncols(), nearest_leader_count);
            arch::dispatch1_no_features(
                DispatchMetric(metric),
                KernelCall {
                    input,
                    output: output.as_mut_view(),
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
                    if point_norm < f32::MIN_POSITIVE.sqrt()
                        || leader_norm < f32::MIN_POSITIVE.sqrt()
                    {
                        1.0
                    } else {
                        1.0 - (dot / (point_norm * leader_norm)).clamp(-1.0, 1.0)
                    }
                }
            }
        }

        pub(super) fn reference_assignments(
            metric: Metric,
            input: PartitionInput<'_>,
            nearest_leader_count: usize,
        ) -> Vec<u32> {
            let point_count = input.dots.ncols();
            let mut output = vec![UNASSIGNED_LEADER; point_count * nearest_leader_count];
            if nearest_leader_count == 0 {
                return output;
            }
            for (point, assignments) in output.chunks_exact_mut(nearest_leader_count).enumerate() {
                let point_norm = input.norms.point_norms.get(point).copied().unwrap_or(0.0);
                let mut candidates: Vec<_> = input
                    .dots
                    .row_iter()
                    .enumerate()
                    .filter_map(|(leader, leader_dots)| {
                        let leader_norm =
                            input.norms.leader_norms.get(leader).copied().unwrap_or(0.0);
                        let score =
                            reference_score(metric, leader_dots[point], point_norm, leader_norm);
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

        /// Build deterministic point and leader vectors for vertical lane boundaries.
        pub(super) fn lane_boundary_input_from_point_and_leader_vectors(
            metric: Metric,
            point_count: usize,
            leader_count: usize,
        ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let point_denominator = (point_count + 1) as f32;
            let points: Vec<_> = (0..point_count)
                .map(|point| {
                    let second_component = (point + 1) as f32 / point_denominator;
                    let vector = [1.0 - second_component, second_component];
                    if metric == Metric::CosineNormalized {
                        let norm = vector[0].hypot(vector[1]);
                        [vector[0] / norm, vector[1] / norm]
                    } else {
                        vector
                    }
                })
                .collect();
            let leader_denominator = (leader_count + 1) as f32;
            let leaders: Vec<_> = (0..leader_count)
                .map(|leader| {
                    let second_component = (leader + 1) as f32 / leader_denominator;
                    let vector = [1.0 - second_component, second_component];
                    if metric == Metric::CosineNormalized {
                        let norm = vector[0].hypot(vector[1]);
                        [vector[0] / norm, vector[1] / norm]
                    } else {
                        vector
                    }
                })
                .collect();
            let dots = leaders
                .iter()
                .flat_map(|leader| {
                    points
                        .iter()
                        .map(|point| point[0] * leader[0] + point[1] * leader[1])
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
            let leader_values: Vec<_> = (0..13).map(|leader| leader as f32).collect();
            let leaders = PreparedLeaders::<L2>::new(
                MatrixView::try_from(leader_values.as_slice(), 13, 1).unwrap(),
            );
            let point_values: Vec<_> = (0..17).map(|point| point as f32 + 0.5).collect();
            let smaller_points = MatrixView::try_from(&point_values[..16], 16, 1).unwrap();
            let mut reused_workspace = PartitionKernelWorkspace::default();
            let mut discarded_large_output = vec![UNASSIGNED_LEADER; 17 * 11];
            assign_leaders::<_, L2>(
                diskann_wide::ARCH,
                MatrixView::try_from(point_values.as_slice(), 17, 1).unwrap(),
                &leaders,
                MutMatrixView::try_from(discarded_large_output.as_mut_slice(), 17, 11).unwrap(),
                &mut reused_workspace,
            )
            .unwrap();
            let mut expected_assignments_from_fresh_workspace = [UNASSIGNED_LEADER; 16 * 3];
            assign_leaders::<_, L2>(
                diskann_wide::ARCH,
                smaller_points,
                &leaders,
                MutMatrixView::try_from(&mut expected_assignments_from_fresh_workspace[..], 16, 3)
                    .unwrap(),
                &mut PartitionKernelWorkspace::default(),
            )
            .unwrap();

            // When
            let mut actual_assignments_from_reused_workspace = [UNASSIGNED_LEADER; 16 * 3];
            assign_leaders::<_, L2>(
                diskann_wide::ARCH,
                smaller_points,
                &leaders,
                MutMatrixView::try_from(&mut actual_assignments_from_reused_workspace[..], 16, 3)
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
        use super::test_support::*;
        use super::*;
        use rstest::rstest;

        #[rstest]
        #[case::zero_width_before_one_lane(15, 17, 0)]
        #[case::k1_before_one_lane(15, 15, 1)]
        #[case::k2_at_one_lane(16, 16, 2)]
        #[case::k3_after_one_lane(17, 17, 3)]
        #[case::k10_before_two_lanes(31, 31, 10)]
        #[case::k11_at_two_lanes(32, 17, 11)]
        #[case::exact_capacity_after_two_lanes(33, 12, 12)]
        #[trace]
        fn dispatched_partition_ranking_matches_scalar_reference_across_lane_boundaries(
            #[values(
                Metric::L2,
                Metric::Cosine,
                Metric::CosineNormalized,
                Metric::InnerProduct
            )]
            metric: Metric,
            #[case] point_count: usize,
            #[case] leader_count: usize,
            #[case] nearest_leader_count: usize,
        ) {
            // Given
            let (dots, point_norms, leader_norms) =
                lane_boundary_input_from_point_and_leader_vectors(
                    metric,
                    point_count,
                    leader_count,
                );
            let input = partition_input(
                &dots,
                point_count,
                leader_count,
                &point_norms,
                &leader_norms,
            );
            let expected_assignments = reference_assignments(metric, input, nearest_leader_count);

            // When
            let actual_assignments = run_rank_leader_dots(metric, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_assignments);
        }

        #[test]
        fn equal_scores_keep_sampled_leader_order_with_l2() {
            // Given
            let point_count = 16;
            let leader_count = 4;
            let nearest_leader_count = 2;
            let dots = [0.0; 16 * 4];
            let leader_squared_norms = [1.0, 1.0, 1.0, 1.0];
            let expected_sampled_leader_order: Vec<_> =
                (0..point_count).flat_map(|_| [0, 1]).collect();

            let input =
                partition_input(&dots, point_count, leader_count, &[], &leader_squared_norms);

            // When
            let actual_assignments = run_rank_leader_dots(Metric::L2, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_sampled_leader_order);
        }

        #[test]
        fn zero_norm_keeps_sampled_leader_order_with_cosine() {
            // Given
            let point_count = 16;
            let leader_count = 2;
            let nearest_leader_count = 2;
            let dots = [0.0; 16 * 2];
            let point_norms = [0.0; 16];
            let leader_norms = [1.0, 1.0];
            let expected_sampled_leader_order: Vec<_> =
                (0..point_count).flat_map(|_| [0, 1]).collect();

            let input = partition_input(
                &dots,
                point_count,
                leader_count,
                &point_norms,
                &leader_norms,
            );

            // When
            let actual_assignments =
                run_rank_leader_dots(Metric::Cosine, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_sampled_leader_order);
        }

        #[test]
        fn zero_point_norm_keeps_first_leader_in_a_complete_simd_group_with_cosine() {
            // Given
            let point_count = 16;
            let leader_count = 17;
            let nearest_leader_count = 1;
            let dots = [0.0; 16 * 17];
            let point_norms = [0.0; 16];
            let leader_norms = [1.0; 17];
            let expected_first_leader = [0; 16];

            // When
            let actual_assignment = run_rank_leader_dots(
                Metric::Cosine,
                partition_input(
                    &dots,
                    point_count,
                    leader_count,
                    &point_norms,
                    &leader_norms,
                ),
                nearest_leader_count,
            );

            // Then
            assert_eq!(actual_assignment, expected_first_leader);
        }

        #[test]
        fn f32_max_score_is_rankable_while_positive_infinity_is_not() {
            // Given
            let point_count = 16;
            let leader_count = 9;
            let nearest_leader_count = leader_count;
            let maximum_rankable_score = f32::MAX;
            let mut dots = [0.0; 16 * 9];
            dots[7 * point_count..8 * point_count].fill(-maximum_rankable_score);
            dots[8 * point_count..].fill(f32::NEG_INFINITY);
            let expected_all_rankable_leaders: Vec<_> = (0..point_count)
                .flat_map(|_| [0, 1, 2, 3, 4, 5, 6, 7, UNASSIGNED_LEADER])
                .collect();

            let input = partition_input(&dots, point_count, leader_count, &[], &[]);

            // When
            let actual_assignments =
                run_rank_leader_dots(Metric::InnerProduct, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_all_rankable_leaders);
        }

        #[test]
        fn nan_leader_does_not_displace_finite_leaders_with_inner_product() {
            // Given
            let point_count = 16;
            let leader_count = 3;
            let nearest_leader_count = 2;
            let dots: Vec<_> = [f32::NAN, 3.0, 2.0]
                .into_iter()
                .flat_map(|dot| [dot; 16])
                .collect();
            let expected_finite_leaders: Vec<_> = (0..point_count).flat_map(|_| [1, 2]).collect();

            let input = partition_input(&dots, point_count, leader_count, &[], &[]);

            // When
            let actual_assignments =
                run_rank_leader_dots(Metric::InnerProduct, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_finite_leaders);
        }

        #[test]
        fn nan_leader_does_not_displace_finite_leaders_with_cosine() {
            // Given
            let point_count = 16;
            let leader_count = 3;
            let nearest_leader_count = 2;
            let dots: Vec<_> = [f32::NAN, 0.75, 0.5]
                .into_iter()
                .flat_map(|dot| [dot; 16])
                .collect();
            let point_norms = [1.0; 16];
            let leader_norms = [1.0; 3];
            let expected_finite_leaders: Vec<_> = (0..point_count).flat_map(|_| [1, 2]).collect();

            let input = partition_input(
                &dots,
                point_count,
                leader_count,
                &point_norms,
                &leader_norms,
            );

            // When
            let actual_assignments =
                run_rank_leader_dots(Metric::Cosine, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_finite_leaders);
        }

        #[test]
        fn zero_output_width_leaves_ranked_leaders_unchanged() {
            // Given
            let dot_products = [0.0];
            let leader_squared_norms = [1.0];
            let input = partition_input(&dot_products, 1, 1, &[], &leader_squared_norms);
            let mut no_assignments = [];
            let output = MutMatrixView::try_from(&mut no_assignments[..], 1, 0).unwrap();
            let expected_ranked_leaders = vec![(7, 0.25)];
            let mut scratch = PartitionRankScratch {
                scalar: expected_ranked_leaders.clone(),
                ..PartitionRankScratch::default()
            };

            // When
            rank_leader_dots::<_, L2>(diskann_wide::ARCH, input, output, &mut scratch);

            // Then
            assert_eq!(scratch.scalar, expected_ranked_leaders);
        }

        #[test]
        fn empty_point_matrix_produces_no_assignments() {
            // Given
            let empty_point_count = 0;
            let leader_count = 3;
            let nearest_leader_count = 2;
            let no_dot_products = [];
            let expected_no_assignments: [u32; 0] = [];

            let input =
                partition_input(&no_dot_products, empty_point_count, leader_count, &[], &[]);

            // When
            let actual_assignments =
                run_rank_leader_dots(Metric::InnerProduct, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_no_assignments);
        }
    }
}
