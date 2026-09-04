/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Select partition centers for PiPNN point assignment.
//!
//! A leader is a sampled dataset point that represents one child partition.
//! The metric builds final point-to-leader distances. The kernel returns nearest
//! leader-column IDs for partition scatter.
//!
//! L2 omits the assigned point's norm because it is constant across all sampled
//! leaders. Equal distances keep sampled-leader order. NaN is not rankable. An
//! unfilled output slot contains [`UNASSIGNED_LEADER`].

use crate::ANNResult;
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_wide::{SIMDMask, SIMDVector};

use super::{
    partition_metric::PartitionMetric,
    simd::{PiPNNSIMDSchema, PiPNNSIMDVector},
};

/// No sampled partition center was rankable for this output slot.
pub(super) const UNASSIGNED_LEADER: u32 = u32::MAX;

/// Reusable storage for one point-stripe numerical pipeline.
#[derive(Default)]
pub(super) struct PartitionKernelWorkspace {
    distance_scratch: Vec<f32>,
    ranked_leader_scratch: Vec<(u32, f32)>,
}

/// Assign one packed point stripe to metric-owned partition leaders.
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
    leaders: &M::Leaders<'_>,
    output: MutMatrixView<'_, u32>,
    workspace: &mut PartitionKernelWorkspace,
) -> ANNResult<()>
where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
{
    let point_count = points.nrows();
    let leader_count = M::leader_count(leaders);
    let distance_count = point_count * leader_count;
    let PartitionKernelWorkspace {
        distance_scratch,
        ranked_leader_scratch,
    } = workspace;
    if distance_scratch.len() < distance_count {
        distance_scratch.resize(distance_count, 0.0);
    }
    M::compute_distances(points, leaders, &mut distance_scratch[..distance_count])?;
    rank_leader_distances(
        arch,
        &distance_scratch[..distance_count],
        point_count,
        leader_count,
        output,
        ranked_leader_scratch,
    );
    Ok(())
}

/// Rank final point-to-leader distances.
///
/// `distance_flatten` contains `point_count * leader_count` elements. Each row
/// stores all leader distances for one point. The caller supplies at least one
/// point and one output column.
fn rank_leader_distances<A>(
    arch: A,
    distance_flatten: &[f32],
    point_count: usize,
    leader_count: usize,
    output: MutMatrixView<'_, u32>,
    ranked_leaders: &mut Vec<(u32, f32)>,
) where
    A: PiPNNSIMDSchema,
{
    let fanout = output.ncols();
    ranked_leaders.resize(fanout, (UNASSIGNED_LEADER, f32::INFINITY));
    // Rayon outlines stripe workers. Reapply target features before ranking leaders.
    arch.run(move || {
        select_point_leaders(
            arch,
            distance_flatten,
            point_count,
            leader_count,
            output,
            ranked_leaders,
        );
    });
}

/// Rank sampled partition centers for each assigned point.
#[inline(always)]
fn select_point_leaders<A>(
    arch: A,
    distance_flatten: &[f32],
    point_count: usize,
    leader_count: usize,
    mut output: MutMatrixView<'_, u32>,
    ranked_leaders: &mut [(u32, f32)],
) where
    A: PiPNNSIMDSchema,
{
    let fanout = output.ncols();

    for (point, point_output) in output
        .as_mut_slice()
        .chunks_exact_mut(fanout)
        .take(point_count)
        .enumerate()
    {
        let row_start = point * leader_count;
        let point_distances = &distance_flatten[row_start..row_start + leader_count];
        ranked_leaders.fill((UNASSIGNED_LEADER, f32::INFINITY));
        let simd_prefix = leader_count - leader_count % A::Vector::LANES;

        for first_leader in (0..simd_prefix).step_by(A::Vector::LANES) {
            // SAFETY: This group is inside the point's leader row.
            let distance_group =
                unsafe { A::Vector::load_simd(arch, point_distances.as_ptr().add(first_leader)) };
            insert_leader_lanes(distance_group, first_leader, ranked_leaders);
        }

        for (leader, &distance) in point_distances.iter().enumerate().skip(simd_prefix) {
            insert_leader(ranked_leaders, leader as u32, distance);
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
fn insert_leader_lanes<F>(distances: F, first_leader: usize, ranked_leaders: &mut [(u32, f32)])
where
    F: PiPNNSIMDVector,
{
    let threshold = F::splat(distances.arch(), ranked_leaders[ranked_leaders.len() - 1].1);
    let eligible = distances.lt_simd(threshold);
    if eligible.none() {
        return;
    }

    let distance_lanes = distances.to_lane_array();
    let distance_lanes = distance_lanes.as_ref();
    let mut lanes = F::active_lanes(eligible);
    while lanes != 0 {
        let lane = lanes.trailing_zeros() as usize;
        lanes &= lanes - 1;
        insert_leader(
            ranked_leaders,
            (first_leader + lane) as u32,
            distance_lanes[lane],
        );
    }
}

/// Insert one sampled partition center into the current point's retained set.
///
/// `leader` is the center's column ID in the point-to-leader matrix. `ranked_leaders`
/// stores retained centers in nearest-first order. Equal distances and NaN do not
/// enter, so sampled-leader order resolves ties.
#[inline(always)]
fn insert_leader(ranked_leaders: &mut [(u32, f32)], leader: u32, distance: f32) {
    let threshold = ranked_leaders.len() - 1;
    if distance.partial_cmp(&ranked_leaders[threshold].1) != Some(std::cmp::Ordering::Less) {
        return;
    }

    ranked_leaders[threshold] = (leader, distance);
    let mut slot = threshold;
    while slot > 0 && ranked_leaders[slot].1 < ranked_leaders[slot - 1].1 {
        ranked_leaders.swap(slot, slot - 1);
        slot -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::Cosine;
    use diskann_utils::views::{Matrix, MatrixView, MutMatrixView};
    use diskann_vector::distance::Metric;

    mod test_support {
        use super::*;
        use diskann_wide::arch::{self, Target1};

        #[derive(Clone, Copy)]
        pub(super) struct DotFixture<'a> {
            dots: MatrixView<'a, f32>,
            point_norms: &'a [f32],
            leader_norms: &'a [f32],
        }

        struct KernelCall<'a> {
            distance_flatten: &'a [f32],
            point_count: usize,
            leader_count: usize,
            output: MutMatrixView<'a, u32>,
            ranked_leaders: &'a mut Vec<(u32, f32)>,
        }

        struct RankDistances;

        impl<A> Target1<A, (), KernelCall<'_>> for RankDistances
        where
            A: PiPNNSIMDSchema,
        {
            fn run(self, arch: A, call: KernelCall<'_>) {
                rank_leader_distances(
                    arch,
                    call.distance_flatten,
                    call.point_count,
                    call.leader_count,
                    call.output,
                    call.ranked_leaders,
                );
            }
        }

        pub(super) fn partition_input<'a>(
            dots: &'a [f32],
            point_count: usize,
            leader_count: usize,
            point_norms: &'a [f32],
            leader_norms: &'a [f32],
        ) -> DotFixture<'a> {
            DotFixture {
                dots: MatrixView::try_from(dots, point_count, leader_count).unwrap(),
                point_norms,
                leader_norms,
            }
        }

        fn reference_distance(metric: Metric, dot: f32, point_norm: f32, leader_norm: f32) -> f32 {
            match metric {
                Metric::L2 => (-2.0_f32).mul_add(dot, leader_norm),
                Metric::CosineNormalized => -dot,
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

        fn distance_flatten(metric: Metric, input: DotFixture<'_>) -> Vec<f32> {
            input
                .dots
                .row_iter()
                .enumerate()
                .flat_map(|(point, dots)| {
                    dots.iter().enumerate().map(move |(leader, &dot)| {
                        reference_distance(
                            metric,
                            dot,
                            input.point_norms.get(point).copied().unwrap_or(0.0),
                            input.leader_norms.get(leader).copied().unwrap_or(0.0),
                        )
                    })
                })
                .collect()
        }

        pub(super) fn rank_distance_fixture(
            metric: Metric,
            input: DotFixture<'_>,
            nearest_leader_count: usize,
        ) -> Vec<u32> {
            let distance_flatten = distance_flatten(metric, input);
            let mut output = Matrix::new(u32::MAX, input.dots.nrows(), nearest_leader_count);
            arch::dispatch1_no_features(
                RankDistances,
                KernelCall {
                    distance_flatten: &distance_flatten,
                    point_count: input.dots.nrows(),
                    leader_count: input.dots.ncols(),
                    output: output.as_mut_view(),
                    ranked_leaders: &mut Vec::new(),
                },
            );
            output.into_inner().into_vec()
        }

        pub(super) fn reference_assignments(
            metric: Metric,
            input: DotFixture<'_>,
            nearest_leader_count: usize,
        ) -> Vec<u32> {
            let mut output = vec![UNASSIGNED_LEADER; input.dots.nrows() * nearest_leader_count];
            for (point, (dots, assignments)) in input
                .dots
                .row_iter()
                .zip(output.chunks_exact_mut(nearest_leader_count))
                .enumerate()
            {
                let point_norm = input.point_norms.get(point).copied().unwrap_or(0.0);
                let mut candidates: Vec<_> = dots
                    .iter()
                    .enumerate()
                    .filter_map(|(leader, &dot)| {
                        let leader_norm = input.leader_norms.get(leader).copied().unwrap_or(0.0);
                        let distance = reference_distance(metric, dot, point_norm, leader_norm);
                        (distance.partial_cmp(&f32::INFINITY) == Some(std::cmp::Ordering::Less))
                            .then_some((leader as u32, distance))
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
        pub(super) fn lane_boundary_input_from_point_and_leader_vectors(
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
        fn nan_distance_does_not_enter_the_topk() {
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
            let leaders =
                Cosine::create_leaders(MatrixView::try_from(&leader_values[..], 3, 2).unwrap());
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
            let leaders =
                Cosine::create_leaders(MatrixView::try_from(&leader_values[..], 3, 2).unwrap());
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

    mod rank_leader_distances_tests {
        use super::test_support::*;
        use super::*;
        use rstest::rstest;

        #[rstest]
        #[case::two_leaders_select_one(2, 1)]
        #[case::scalar_select_two(7, 2)]
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
            #[case] nearest_leader_count: usize,
        ) {
            // Given
            let (dots, point_norms, leader_norms) =
                lane_boundary_input_from_point_and_leader_vectors(metric, leader_count);
            let input = partition_input(&dots, 2, leader_count, &point_norms, &leader_norms);
            let expected_assignments = reference_assignments(metric, input, nearest_leader_count);

            // When
            let actual_assignments = rank_distance_fixture(metric, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_assignments);
        }

        #[test]
        fn equal_distances_keep_sampled_leader_order_with_l2() {
            // Given
            let point_count = 1;
            let leader_count = 4;
            let nearest_leader_count = 2;
            let dots = [0.0, 0.0, 0.0, 0.0];
            let leader_squared_norms = [1.0, 1.0, 1.0, 1.0];
            let expected_sampled_leader_order = [0, 1];

            let input =
                partition_input(&dots, point_count, leader_count, &[], &leader_squared_norms);

            // When
            let actual_assignments = rank_distance_fixture(Metric::L2, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_sampled_leader_order);
        }

        #[test]
        fn zero_norm_keeps_sampled_leader_order_with_cosine() {
            // Given
            let point_count = 1;
            let leader_count = 2;
            let nearest_leader_count = 2;
            let dots = [0.0, 0.0];
            let point_norms = [0.0];
            let leader_norms = [1.0, 1.0];
            let expected_sampled_leader_order = [0, 1];

            let input = partition_input(
                &dots,
                point_count,
                leader_count,
                &point_norms,
                &leader_norms,
            );

            // When
            let actual_assignments =
                rank_distance_fixture(Metric::Cosine, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_sampled_leader_order);
        }

        #[test]
        fn zero_point_norm_keeps_first_leader_in_a_complete_simd_group_with_cosine() {
            // Given
            let point_count = 1;
            let leader_count = 17;
            let nearest_leader_count = 1;
            let dots = [0.0; 17];
            let point_norms = [0.0];
            let leader_norms = [1.0; 17];
            let expected_first_leader = [0];

            // When
            let actual_assignment = rank_distance_fixture(
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
        fn f32_max_distance_is_still_rankable() {
            // Given
            let point_count = 1;
            let leader_count = 8;
            let nearest_leader_count = leader_count;
            let maximum_rankable_distance = f32::MAX;
            let dot_product_that_produces_it = -maximum_rankable_distance;
            let mut dots = [0.0; 8];
            dots[7] = dot_product_that_produces_it;
            let expected_all_leaders_in_scan_order = [0, 1, 2, 3, 4, 5, 6, 7];

            let input = partition_input(&dots, point_count, leader_count, &[], &[]);

            // When
            let actual_assignments =
                rank_distance_fixture(Metric::InnerProduct, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_all_leaders_in_scan_order);
        }

        #[test]
        fn nan_leader_does_not_displace_finite_leaders_with_inner_product() {
            // Given
            let point_count = 1;
            let leader_count = 3;
            let nearest_leader_count = 2;
            let dots = [f32::NAN, 3.0, 2.0];
            let expected_finite_leaders = [1, 2];

            let input = partition_input(&dots, point_count, leader_count, &[], &[]);

            // When
            let actual_assignments =
                rank_distance_fixture(Metric::InnerProduct, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_finite_leaders);
        }

        #[test]
        fn nan_leader_does_not_displace_finite_leaders_with_cosine() {
            // Given
            let point_count = 1;
            let leader_count = 3;
            let nearest_leader_count = 2;
            let dots = [f32::NAN, 0.75, 0.5];
            let point_norms = [1.0];
            let leader_norms = [1.0; 3];
            let expected_finite_leaders = [1, 2];

            let input = partition_input(
                &dots,
                point_count,
                leader_count,
                &point_norms,
                &leader_norms,
            );

            // When
            let actual_assignments =
                rank_distance_fixture(Metric::Cosine, input, nearest_leader_count);

            // Then
            assert_eq!(actual_assignments, expected_finite_leaders);
        }
    }
}
