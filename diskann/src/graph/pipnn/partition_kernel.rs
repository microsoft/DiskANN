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
//! leaders. Equal distances can select either leader. NaN is not rankable. An
//! unfilled output slot contains [`UNASSIGNED_LEADER`].

use crate::{ANNError, ANNResult};
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
    let distances = MatrixView::try_from(
        &distance_scratch[..distance_count],
        point_count,
        leader_count,
    )
    .map_err(|error| ANNError::new(error.as_static()))?;
    rank_leader_distances(arch, distances, output, ranked_leader_scratch);
    Ok(())
}

/// Rank final point-to-leader distances.
///
/// Each distance row stores all leader distances for one point. The caller
/// supplies at least one point, one leader, and one output column.
fn rank_leader_distances<A>(
    arch: A,
    distances: MatrixView<'_, f32>,
    output: MutMatrixView<'_, u32>,
    ranked_leaders: &mut Vec<(u32, f32)>,
) where
    A: PiPNNSIMDSchema,
{
    let fanout = output.ncols();
    ranked_leaders.resize(fanout, (UNASSIGNED_LEADER, f32::INFINITY));
    // Rayon outlines stripe workers. Reapply target features before ranking leaders.
    arch.run(move || {
        select_point_leaders(arch, distances, output, ranked_leaders);
    });
}

/// Rank sampled partition centers for each assigned point.
#[inline(always)]
fn select_point_leaders<A>(
    arch: A,
    distances: MatrixView<'_, f32>,
    mut output: MutMatrixView<'_, u32>,
    ranked_leaders: &mut [(u32, f32)],
) where
    A: PiPNNSIMDSchema,
{
    let leader_count = distances.ncols();
    let simd_end = leader_count - leader_count % A::Vector::LANES;

    for (point_distances, point_output) in distances.row_iter().zip(output.row_iter_mut()) {
        ranked_leaders.fill((UNASSIGNED_LEADER, f32::INFINITY));

        for leader_base_idx in (0..simd_end).step_by(A::Vector::LANES) {
            // SAFETY: This group is inside the point's leader row.
            let distance_group = unsafe {
                A::Vector::load_simd(arch, point_distances.as_ptr().add(leader_base_idx))
            };
            insert_leader_lanes(distance_group, leader_base_idx, ranked_leaders);
        }

        for (leader, &distance) in point_distances.iter().enumerate().skip(simd_end) {
            insert_leader(ranked_leaders, leader as u32, distance);
        }
        for (destination, &(leader, _)) in point_output.iter_mut().zip(ranked_leaders.iter()) {
            *destination = leader;
        }
    }
}

/// Offer one SIMD group of sampled centers to the current point's ranked_leaders.
///
/// `leader_base_idx` is the matrix-column index of the first lane. The function reads
/// selected lanes from low to high.
fn insert_leader_lanes<F>(distances: F, leader_base_idx: usize, ranked_leaders: &mut [(u32, f32)])
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
            (leader_base_idx + lane) as u32,
            distance_lanes[lane],
        );
    }
}

/// Insert one sampled partition center into the current point's retained set.
///
/// `leader` is the center's column ID in the point-to-leader matrix. `ranked_leaders`
/// stores retained centers in nearest-first order. A candidate enters only when
/// its distance is less than the current farthest distance. NaN does not enter.
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

    mod test_support {
        use super::*;
        use diskann_wide::arch::{self, Target1};

        struct KernelCall<'a> {
            distances: MatrixView<'a, f32>,
            output: MutMatrixView<'a, u32>,
            ranked_leaders: &'a mut Vec<(u32, f32)>,
        }

        struct RankDistances;

        impl<A> Target1<A, (), KernelCall<'_>> for RankDistances
        where
            A: PiPNNSIMDSchema,
        {
            fn run(self, arch: A, call: KernelCall<'_>) {
                rank_leader_distances(arch, call.distances, call.output, call.ranked_leaders);
            }
        }

        pub(super) fn rank_distance_fixture(
            distances: MatrixView<'_, f32>,
            nearest_leader_count: usize,
        ) -> Vec<u32> {
            let mut output =
                Matrix::new(UNASSIGNED_LEADER, distances.nrows(), nearest_leader_count);
            arch::dispatch1_no_features(
                RankDistances,
                KernelCall {
                    distances,
                    output: output.as_mut_view(),
                    ranked_leaders: &mut Vec::new(),
                },
            );
            output.into_inner().into_vec()
        }

        pub(super) fn reference_assignments(
            distances: MatrixView<'_, f32>,
            nearest_leader_count: usize,
        ) -> Vec<u32> {
            let mut output = vec![UNASSIGNED_LEADER; distances.nrows() * nearest_leader_count];
            for (row, assignments) in distances
                .row_iter()
                .zip(output.chunks_exact_mut(nearest_leader_count))
            {
                let mut candidates: Vec<_> = row
                    .iter()
                    .enumerate()
                    .filter_map(|(leader, &distance)| {
                        (distance < f32::INFINITY).then_some((leader as u32, distance))
                    })
                    .collect();
                candidates.sort_unstable_by(|left, right| {
                    left.1
                        .partial_cmp(&right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for (destination, (leader, _)) in assignments.iter_mut().zip(candidates) {
                    *destination = leader;
                }
            }
            output
        }
    }

    mod insert_leader_tests {
        use super::*;

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
        #[case::all_leaders(17, 17)]
        #[case::two_lanes_minus_one(31, 7)]
        #[case::two_complete_lanes(32, 7)]
        #[case::two_lanes_plus_one(33, 7)]
        #[case::root_partition(1000, 64)]
        #[trace]
        fn dispatched_partition_ranking_matches_scalar_reference_across_lane_boundaries(
            #[case] leader_count: usize,
            #[case] nearest_leader_count: usize,
        ) {
            // Given: unique distances increase in one row and decrease in the other.
            let values: Vec<_> = (0..leader_count)
                .chain((0..leader_count).rev())
                .map(|leader| leader as f32 - leader_count as f32 / 2.0)
                .collect();
            let distances = MatrixView::try_from(values.as_slice(), 2, leader_count).unwrap();
            let expected = reference_assignments(distances, nearest_leader_count);

            // When
            let actual = rank_distance_fixture(distances, nearest_leader_count);

            // Then
            assert_eq!(actual, expected);
        }

        #[rstest]
        #[case::scalar(4)]
        #[case::simd_with_tail(17)]
        fn nearer_leader_displaces_one_of_the_tied_candidates(#[case] leader_count: usize) {
            // Given
            let nearest_leader = leader_count - 1;
            let mut values = vec![1.0; leader_count];
            values[nearest_leader] = 0.0;
            let distances = MatrixView::try_from(values.as_slice(), 1, leader_count).unwrap();

            // When
            let actual = rank_distance_fixture(distances, 3);

            // Then: the remaining two slots can contain any distinct tied leaders.
            assert_eq!(actual[0], nearest_leader as u32);
            assert!(actual[1] < nearest_leader as u32);
            assert!(actual[2] < nearest_leader as u32);
            assert_ne!(actual[1], actual[2]);
        }

        #[rstest]
        #[case::nan(f32::NAN)]
        #[case::positive_infinity(f32::INFINITY)]
        fn non_rankable_distances_leave_only_unfilled_ranks_unassigned(
            #[case] unrankable: f32,
            #[values(3, 17, 33)] leader_count: usize,
        ) {
            // Given: only the first row has rankable candidates.
            let mut values = vec![unrankable; 2 * leader_count];
            values[leader_count - 2] = -1.0;
            values[leader_count - 1] = 2.0;
            let distances = MatrixView::try_from(values.as_slice(), 2, leader_count).unwrap();
            let expected = [
                (leader_count - 2) as u32,
                (leader_count - 1) as u32,
                UNASSIGNED_LEADER,
                UNASSIGNED_LEADER,
                UNASSIGNED_LEADER,
                UNASSIGNED_LEADER,
            ];

            // When
            let actual = rank_distance_fixture(distances, 3);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn negative_infinity_and_f32_max_remain_rankable() {
            // Given: the smallest distance is in the scalar tail.
            let mut values = [f32::NAN; 17];
            values[0] = f32::MAX;
            values[1] = 0.0;
            values[2] = -f32::EPSILON;
            values[16] = f32::NEG_INFINITY;
            let distances = MatrixView::try_from(&values[..], 1, values.len()).unwrap();
            let expected = [16, 2, 1, 0];

            // When
            let actual = rank_distance_fixture(distances, 4);

            // Then
            assert_eq!(actual, expected);
        }
    }
}
