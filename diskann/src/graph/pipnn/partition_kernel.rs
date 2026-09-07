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

use super::{
    partition_metric::PartitionMetric,
    simd::{PiPNNSIMDSchema, distance_blocks},
    topk::{Candidate, UNASSIGNED, with_topk_rows},
};

/// No sampled partition center was rankable for this output slot.
pub(super) const UNASSIGNED_LEADER: u32 = UNASSIGNED;

/// Reusable storage for one point-stripe numerical pipeline.
#[derive(Default)]
pub(super) struct PartitionKernelWorkspace {
    distance_scratch: Vec<f32>,
    ranked_leader_scratch: Vec<Candidate>,
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
    mut output: MutMatrixView<'_, u32>,
    ranked_leaders: &mut Vec<Candidate>,
) where
    A: PiPNNSIMDSchema,
{
    let fanout = output.ncols();
    ranked_leaders.resize(fanout, Candidate::default());
    // Rayon outlines stripe workers. Reapply target features before ranking leaders.
    arch.run(move || {
        let mut worst = [f32::INFINITY];
        with_topk_rows!(
            MutMatrixView::row_vector(ranked_leaders.as_mut_slice()),
            &mut worst[..],
            |topks| {
                for (point_distances, point_output) in
                    distances.row_iter().zip(output.row_iter_mut())
                {
                    topks.reset();
                    for block in distance_blocks(arch, point_distances) {
                        topks.update_one(0, &block);
                    }
                    for (destination, leader) in point_output.iter_mut().zip(topks.row(0)) {
                        *destination = leader.local_idx;
                    }
                }
            }
        );
    });
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
            ranked_leaders: &'a mut Vec<Candidate>,
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
            // A previous valid leader must be overwritten even when no candidate is rankable.
            let mut output = Matrix::new(0, distances.nrows(), nearest_leader_count);
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
    }

    mod assign_leaders_tests {
        use super::*;

        #[test]
        fn cosine_assignment_reuses_workspace_and_output_for_a_different_point() {
            // Given
            let leader_values = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
            let leaders =
                Cosine::create_leaders(MatrixView::try_from(&leader_values[..], 3, 2).unwrap());
            let point_values = [0.9, 0.1, -0.8, 0.2];
            let points = MatrixView::try_from(&point_values[..], 2, 2).unwrap();
            let expected_leaders_by_descending_cosine_similarity = [0, 1, 2, 1];
            let expected_reassigned_leaders = [2, 1];
            let mut actual_assignments = [UNASSIGNED_LEADER; 4];
            let mut workspace = PartitionKernelWorkspace::default();

            // When
            assign_leaders::<_, Cosine>(
                diskann_wide::ARCH,
                points,
                &leaders,
                MutMatrixView::try_from(&mut actual_assignments[..], 2, 2).unwrap(),
                &mut workspace,
            )
            .unwrap();

            // Then
            assert_eq!(
                actual_assignments,
                expected_leaders_by_descending_cosine_similarity
            );

            // When: shrink to the second point, whose nearest leader differs from the old prefix.
            assign_leaders::<_, Cosine>(
                diskann_wide::ARCH,
                MatrixView::try_from(&point_values[2..], 1, 2).unwrap(),
                &leaders,
                MutMatrixView::try_from(&mut actual_assignments[..2], 1, 2).unwrap(),
                &mut workspace,
            )
            .unwrap();

            // Then
            assert_eq!(actual_assignments[..2], expected_reassigned_leaders);
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
        fn dispatched_partition_ranking_selects_nearest_columns_across_lane_boundaries(
            #[case] leader_count: usize,
            #[case] nearest_leader_count: usize,
        ) {
            // Given: unique distances increase in one row and decrease in the other.
            let values: Vec<_> = (0..leader_count)
                .chain((0..leader_count).rev())
                .map(|leader| leader as f32 - leader_count as f32 / 2.0)
                .collect();
            let distances = MatrixView::try_from(values.as_slice(), 2, leader_count).unwrap();
            let expected: Vec<_> = (0..nearest_leader_count)
                .chain((leader_count - nearest_leader_count..leader_count).rev())
                .map(|leader| leader as u32)
                .collect();

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
        fn non_rankable_distances_leave_only_unfilled_ranks_unassigned(#[case] unrankable: f32) {
            // Given: only the first row has rankable candidates, spanning SIMD and scalar tail.
            let leader_count = 17;
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
