/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf-local top-k selection from packed `f32` point vectors.
//!
//! The metric fills the lower triangle of a distance matrix. The kernel
//! reads each strict-lower point pair once and updates both points.
//!
//! The output is an `n × k` matrix of sorted [`Candidate`] values. Each target
//! is a position in the leaf.
//!
//! Equal distances can select either candidate. The kernel does not rank NaN.
//! An unfilled output slot contains [`Candidate::default`]. All supported
//! metrics use the same SIMD-group and single-value traversal.
//!
//! The caller supplies concrete architecture `A` and metric `M`.
//! [`LeafKernelWorkspace`] stores reusable numerical scratch.

use crate::{ANNError, ANNResult};
use diskann_utils::views::{MatrixView, MutMatrixView};

use super::{
    leaf_metric::LeafMetric,
    simd::{PiPNNSIMDSchema, distance_blocks},
    topk::{Candidate, with_topk_rows},
};

/// Reusable storage for one leaf numerical pipeline.
#[derive(Debug, Default)]
pub(super) struct LeafKernelWorkspace {
    distance_scratch: Vec<f32>,
    worst: Vec<f32>,
}

/// Invalid output width for leaf-neighbor selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum LeafKernelError {
    /// A source requests more neighbors than the leaf has other points.
    #[error("invalid leaf neighbor count {neighbors} for {points} points; maximum is {maximum}")]
    InvalidNeighborCount {
        points: usize,
        neighbors: usize,
        maximum: usize,
    },
}

/// Return the non-self neighbor count for one leaf.
///
/// `points` is the number of points in the leaf. `requested_k` is the configured
/// neighbor count. The result is `min(requested_k, points - 1)`.
///
pub(super) fn leaf_neighbor_count(points: usize, requested_k: usize) -> usize {
    requested_k.min(points.saturating_sub(1))
}

/// Compute local nearest neighbors for one packed leaf matrix.
///
/// # Errors
///
/// Returns an error for invalid linear-algebra input or output width.
/// Invalid output widths leave the output and workspace unchanged.
pub(super) fn select_leaf_neighbors<A, M>(
    arch: A,
    points: MatrixView<'_, f32>,
    output: MutMatrixView<'_, Candidate>,
    workspace: &mut LeafKernelWorkspace,
) -> ANNResult<()>
where
    A: PiPNNSIMDSchema,
    M: LeafMetric,
{
    let point_count = points.nrows();
    validate_neighbor_count(point_count, &output).map_err(ANNError::new)?;
    let distance_count = point_count * point_count;
    let LeafKernelWorkspace {
        distance_scratch,
        worst,
    } = workspace;
    if distance_scratch.len() < distance_count {
        distance_scratch.resize(distance_count, 0.0);
    }
    let mut distances = MutMatrixView::try_from(
        &mut distance_scratch[..distance_count],
        point_count,
        point_count,
    )
    .map_err(|error| ANNError::new(error.as_static()))?;
    M::compute_distances(points, distances.as_mut_slice())?;
    rank_leaf_distances(arch, distances.as_view(), output, worst);
    Ok(())
}

/// Traverse the strict lower triangle and update each pair's two candidate rows.
fn rank_leaf_distances<A>(
    arch: A,
    distances: MatrixView<'_, f32>,
    output: MutMatrixView<'_, Candidate>,
    worst: &mut Vec<f32>,
) where
    A: PiPNNSIMDSchema,
{
    if output.ncols() == 0 {
        return;
    }
    worst.resize(distances.nrows(), f32::INFINITY);
    with_topk_rows!(output, worst, |topks| {
        // Rayon outlines leaf workers. Reapply target features before the SIMD scan.
        arch.run(move || {
            for source_idx in 1..distances.nrows() {
                let source_distances = &distances.row(source_idx)[..source_idx];
                for block in distance_blocks(arch, source_distances) {
                    topks.update_one(source_idx, &block);
                    topks.update_many(source_idx as u32, &block);
                }
            }
        });
    });
}

/// Check the output width against the number of non-self points.
fn validate_neighbor_count(
    point_count: usize,
    output: &MutMatrixView<'_, Candidate>,
) -> Result<(), LeafKernelError> {
    let maximum_neighbors = point_count.saturating_sub(1);
    let neighbor_count = output.ncols();
    if neighbor_count > maximum_neighbors {
        return Err(LeafKernelError::InvalidNeighborCount {
            points: point_count,
            neighbors: neighbor_count,
            maximum: maximum_neighbors,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::L2;
    use diskann_utils::views::{MatrixView, MutMatrixView};

    mod test_support {
        use std::cmp::Ordering;

        use super::*;
        use diskann_wide::arch::{self, Target1};

        struct KernelCall<'a> {
            distances: MatrixView<'a, f32>,
            output: MutMatrixView<'a, Candidate>,
            worst: &'a mut Vec<f32>,
        }

        struct RankDistances;

        impl<A> Target1<A, (), KernelCall<'_>> for RankDistances
        where
            A: PiPNNSIMDSchema,
        {
            fn run(self, arch: A, call: KernelCall<'_>) {
                rank_leaf_distances(arch, call.distances, call.output, call.worst);
            }
        }

        pub(super) fn rank_distance_fixture(
            distances: &[f32],
            points: usize,
            output_width: usize,
        ) -> Vec<Candidate> {
            let mut output = vec![Candidate::default(); points * output_width];
            arch::dispatch1_no_features(
                RankDistances,
                KernelCall {
                    distances: MatrixView::try_from(distances, points, points).unwrap(),
                    output: MutMatrixView::try_from(output.as_mut_slice(), points, output_width)
                        .unwrap(),
                    worst: &mut Vec::new(),
                },
            );
            output
        }

        pub(super) fn reference_neighbors(
            distances: &[f32],
            points: usize,
            width: usize,
        ) -> Vec<Candidate> {
            let mut output = vec![Candidate::default(); points * width];
            for source in 0..points {
                let mut candidates = Vec::with_capacity(points.saturating_sub(1));
                for target in 0..points {
                    if source == target {
                        continue;
                    }
                    let row = source.max(target);
                    let column = source.min(target);
                    let distance = distances[row * points + column];
                    if distance < f32::INFINITY {
                        candidates.push(Candidate::new(target as u32, distance));
                    }
                }
                candidates.sort_unstable_by(|left, right| {
                    left.distance
                        .partial_cmp(&right.distance)
                        .unwrap_or(Ordering::Equal)
                });
                let retained = candidates.len().min(width);
                output[source * width..source * width + retained]
                    .copy_from_slice(&candidates[..retained]);
            }
            output
        }

        /// Give each pair a unique distance, with alternating signs in each row.
        pub(super) fn lower_triangle_distances(points: usize) -> Vec<f32> {
            // NaN in unused entries detects reads of the diagonal or upper triangle.
            let mut distances = vec![f32::NAN; points * points];
            for source in 1..points {
                for target in 0..source {
                    let distance = (source * points + target + 1) as f32;
                    distances[source * points + target] =
                        if target % 2 == 0 { -distance } else { distance };
                }
            }
            distances
        }
    }

    mod leaf_neighbor_count_tests {
        use super::leaf_neighbor_count;
        use rstest::rstest;

        #[rstest]
        #[case::singleton(1, 3, 0)]
        #[case::clamped(4, 4, 3)]
        #[case::requested(8, 5, 5)]
        fn count_is_bounded_by_non_self_points(
            #[case] points: usize,
            #[case] requested: usize,
            #[case] expected: usize,
        ) {
            assert_eq!(leaf_neighbor_count(points, requested), expected);
        }
    }

    mod select_leaf_neighbors_tests {
        use super::*;

        #[test]
        fn invalid_neighbor_width_leaves_buffers_unchanged() {
            // Given
            let values = [0.0_f32, 1.0, 3.0];
            let point_count = values.len();
            let invalid_width = point_count;
            let points = MatrixView::try_from(&values[..], point_count, 1).unwrap();
            let expected_output = [Candidate::default(); 9];
            // A smaller prior leaf forces scratch growth if validation runs too late.
            let expected_distances = [99.0; 4];
            let expected_thresholds = [7.0; 2];
            let expected_error = LeafKernelError::InvalidNeighborCount {
                points: point_count,
                neighbors: invalid_width,
                maximum: point_count - 1,
            };
            let mut output = expected_output;
            let mut workspace = LeafKernelWorkspace {
                distance_scratch: expected_distances.to_vec(),
                worst: expected_thresholds.to_vec(),
            };

            // When
            let error = select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                points,
                MutMatrixView::try_from(&mut output[..], point_count, invalid_width).unwrap(),
                &mut workspace,
            )
            .unwrap_err();

            // Then
            assert_eq!(
                error.downcast_ref::<LeafKernelError>(),
                Some(&expected_error)
            );
            assert_eq!(output, expected_output);
            assert_eq!(workspace.distance_scratch, expected_distances);
            assert_eq!(workspace.worst, expected_thresholds);
        }

        #[test]
        fn l2_selection_reuses_output_and_scratch_for_a_smaller_leaf() {
            // Given
            let values = [0.0_f32, 1.0, 3.0, 10.0];
            let points = MatrixView::try_from(&values[..], 4, 1).unwrap();
            let expected_neighbors = [
                Candidate::new(1, (values[0] - values[1]).powi(2)),
                Candidate::new(2, (values[0] - values[2]).powi(2)),
                Candidate::new(0, (values[1] - values[0]).powi(2)),
                Candidate::new(2, (values[1] - values[2]).powi(2)),
                Candidate::new(1, (values[2] - values[1]).powi(2)),
                Candidate::new(0, (values[2] - values[0]).powi(2)),
                Candidate::new(2, (values[3] - values[2]).powi(2)),
                Candidate::new(1, (values[3] - values[1]).powi(2)),
            ];
            let mut actual_neighbors = [Candidate::default(); 8];
            let mut workspace = LeafKernelWorkspace::default();

            // When
            select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                points,
                MutMatrixView::try_from(&mut actual_neighbors[..], 4, 2).unwrap(),
                &mut workspace,
            )
            .unwrap();
            assert_eq!(actual_neighbors, expected_neighbors);

            // Reuse both buffers with a smaller, different dataset and unfilled slots.
            let smaller = [0.0, 2.0, f32::NAN];
            select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                MatrixView::try_from(&smaller[..], 3, 1).unwrap(),
                MutMatrixView::try_from(&mut actual_neighbors[..6], 3, 2).unwrap(),
                &mut workspace,
            )
            .unwrap();
            assert_eq!(
                actual_neighbors[..6],
                [
                    Candidate::new(1, 4.0),
                    Candidate::default(),
                    Candidate::new(0, 4.0),
                    Candidate::default(),
                    Candidate::default(),
                    Candidate::default(),
                ]
            );
        }
    }

    mod rank_leaf_distances_tests {
        use super::test_support::*;
        use super::*;
        use rstest::rstest;

        #[rstest]
        #[case::two_points_fixed_one(2, 1)]
        #[case::scalar_fixed_two(7, 2)]
        #[case::lane_minus_one_fixed_three(16, 3)]
        #[case::one_complete_lane_fixed_one(17, 1)]
        #[case::one_complete_lane_fixed_two(17, 2)]
        #[case::one_complete_lane_fixed_three(17, 3)]
        #[case::lane_plus_one_runtime_width(18, 4)]
        #[case::all_non_self_neighbors(17, 16)]
        #[case::two_lanes_minus_one_runtime_width(32, 7)]
        #[case::two_complete_lanes_runtime_width(33, 7)]
        #[case::two_lanes_plus_one_runtime_width(34, 7)]
        #[case::maximum_leaf_size_runtime_width(512, 7)]
        #[trace]
        fn dispatched_leaf_ranking_matches_scalar_reference_across_lane_boundaries(
            #[case] point_count: usize,
            #[case] requested_k: usize,
        ) {
            // Miri covers the pointer boundaries in the smaller lane cases.
            if cfg!(miri) && point_count > 64 {
                return;
            }

            // Given: the last source row has `point_count - 1` distances.
            let distances = lower_triangle_distances(point_count);
            let expected_neighbors = reference_neighbors(&distances, point_count, requested_k);

            // When
            let actual_neighbors = rank_distance_fixture(&distances, point_count, requested_k);

            // Then
            assert_eq!(actual_neighbors, expected_neighbors);
        }

        #[rstest]
        #[case::finite_negative(-f32::EPSILON)]
        #[case::negative_infinity(f32::NEG_INFINITY)]
        fn negative_distances_remain_rankable(#[case] distance: f32) {
            // Given
            let distances = [f32::NAN, f32::NAN, distance, f32::NAN];
            let expected = [Candidate::new(1, distance), Candidate::new(0, distance)];

            // When
            let actual = rank_distance_fixture(&distances, 2, 1);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn tied_distances_fill_capacity_with_distinct_non_self_neighbors() {
            // Given: each point has three equally distant candidates for two slots.
            let point_count = 4;
            let width = 2;
            let distances = [1.0; 16];

            // When
            let actual = rank_distance_fixture(&distances, point_count, width);

            // Then: any two candidates are valid; no tie order is required.
            for (source, neighbors) in actual.chunks_exact(width).enumerate() {
                assert_eq!(neighbors[0].distance, 1.0);
                assert_eq!(neighbors[1].distance, 1.0);
                assert_ne!(neighbors[0].local_idx, neighbors[1].local_idx);
                for neighbor in neighbors {
                    assert!(neighbor.local_idx < point_count as u32);
                    assert_ne!(neighbor.local_idx, source as u32);
                }
            }
        }

        #[test]
        fn f32_max_distance_is_still_a_rankable_neighbor() {
            // Given
            let point_count = 4;
            let width = 3;
            let source = 3;
            let mut distances = [1.0; 16];
            distances[source * point_count] = f32::MAX;
            let expected_last_neighbor = Candidate::new(0, f32::MAX);

            // When
            let actual = rank_distance_fixture(&distances, point_count, width);

            // Then
            assert_eq!(actual[source * width + width - 1], expected_last_neighbor);
        }

        #[rstest]
        #[case::nan(f32::NAN)]
        #[case::positive_infinity(f32::INFINITY)]
        fn non_rankable_distances_leave_neighbor_slots_unassigned(#[case] distance: f32) {
            // One leaf exercises scalar rows, a full SIMD group, and its tail.
            let point_count = 18;
            let distances = vec![distance; point_count * point_count];
            let expected = vec![Candidate::default(); point_count];

            // When
            let actual = rank_distance_fixture(&distances, point_count, 1);

            // Then
            assert_eq!(actual, expected);
        }

        #[rstest]
        #[case::nan(f32::NAN)]
        #[case::positive_infinity(f32::INFINITY)]
        fn non_rankable_candidates_cannot_replace_finite_neighbors(
            #[case] distance: f32,
            #[values(17, 18)] point_count: usize,
        ) {
            // Given: the last candidate is in a full SIMD group or its scalar tail.
            let source = point_count - 1;
            let invalid_target = source - 1;
            let mut distances = vec![1.0; point_count * point_count];
            distances[source * point_count + invalid_target] = distance;

            // When
            let actual = rank_distance_fixture(&distances, point_count, 1);

            // Then
            assert!(actual[source].local_idx < invalid_target as u32);
            assert_eq!(actual[source].distance, 1.0);
        }

        #[test]
        fn scalar_tail_updates_both_endpoints_after_a_rejected_simd_group() {
            // Given: the final pair is the only rankable pair in the leaf.
            let point_count = 18;
            let source = 17;
            let target = 16;
            let mut distances = vec![f32::INFINITY; point_count * point_count];
            distances[source * point_count + target] = -1.0;
            let mut expected = vec![Candidate::default(); point_count];
            expected[source] = Candidate::new(target as u32, -1.0);
            expected[target] = Candidate::new(source as u32, -1.0);

            // When
            let actual = rank_distance_fixture(&distances, point_count, 1);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn singleton_leaf_has_no_neighbors() {
            // Given
            let distances = [f32::NAN];
            let expected: [Candidate; 0] = [];

            // When
            let actual = rank_distance_fixture(&distances, 1, 0);

            // Then
            assert_eq!(actual, expected);
        }
    }
}
