/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf-local top-k selection from packed `f32` point vectors.
//!
//! [`select_leaf_neighbors`] asks the metric to fill a lower-triangle ranking
//! buffer. It scans each point pair once and updates both points' neighbor lists.

use crate::{ANNError, ANNResult};
use diskann_utils::views::{MatrixView, MutMatrixView};

use super::{
    leaf_metric::LeafMetric,
    simd::PiPNNSIMDSchema,
    topk::{Candidate, with_topk},
};

/// Reusable storage for one leaf numerical pipeline.
#[derive(Debug, Default)]
pub(super) struct LeafKernelWorkspace {
    distance_scratch: Vec<f32>,
    worst: Vec<f32>,
}

/// Invalid output shape for leaf-neighbor selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum LeafKernelError {
    /// The output must contain one row per input point.
    #[error("invalid leaf output row count {rows} for {points} points")]
    InvalidOutputRows { points: usize, rows: usize },
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
/// `output` contains one row per point, ordered by increasing ranking distance.
/// Candidate IDs are positions in the leaf. Equal distances can select either
/// candidate. NaN and positive infinity are not retained; unfilled slots contain
/// [`Candidate::default`].
///
/// # Errors
///
/// Returns an error for invalid linear-algebra input or output shape.
/// Invalid output shapes leave the output and workspace unchanged.
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
    validate_output(point_count, &output).map_err(ANNError::new)?;
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
    )?;
    M::compute_distances(points, distances.as_mut_slice())?;
    rank_leaf_distances(arch, distances.as_view(), output, worst);
    Ok(())
}

/// Offer each lower-triangle row to the two endpoint neighbor lists.
fn rank_leaf_distances<A: PiPNNSIMDSchema>(
    arch: A,
    distances: MatrixView<'_, f32>,
    mut output: MutMatrixView<'_, Candidate>,
    worst: &mut Vec<f32>,
) {
    arch.run(move || {
        with_topk!(output.ncols(), |topk| {
            topk.initialize(output.as_mut_view(), worst);
            for point_idx in 1..distances.nrows() {
                topk.update_dual_topk(
                    arch,
                    point_idx,
                    &distances.row(point_idx)[..point_idx],
                    output.as_mut_view(),
                    worst.as_mut_slice(),
                );
            }
        });
    });
}

/// Check for one output row per point and a valid non-self neighbor count.
fn validate_output(
    point_count: usize,
    output: &MutMatrixView<'_, Candidate>,
) -> Result<(), LeafKernelError> {
    if output.nrows() != point_count {
        return Err(LeafKernelError::InvalidOutputRows {
            points: point_count,
            rows: output.nrows(),
        });
    }
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

    mod test_support {
        use super::*;
        use diskann_wide::arch::{self, Target1};

        struct KernelCall<'a> {
            distances: MatrixView<'a, f32>,
            output: MutMatrixView<'a, Candidate>,
        }

        struct RankDistances;

        impl<A: PiPNNSIMDSchema> Target1<A, (), KernelCall<'_>> for RankDistances {
            fn run(self, arch: A, call: KernelCall<'_>) {
                rank_leaf_distances(arch, call.distances, call.output, &mut Vec::new());
            }
        }

        pub(super) fn rank_distance_fixture(
            distances: &[f32],
            points: usize,
            width: usize,
        ) -> Vec<Candidate> {
            let mut output = vec![Candidate::default(); points * width];
            arch::dispatch1_no_features(
                RankDistances,
                KernelCall {
                    distances: MatrixView::try_from(distances, points, points).unwrap(),
                    output: MutMatrixView::try_from(output.as_mut_slice(), points, width).unwrap(),
                },
            );
            output
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

        #[rstest::rstest]
        #[case::missing_row(2)]
        #[case::extra_row(4)]
        fn invalid_output_rows_return_an_error(#[case] rows: usize) {
            let values = [0.0_f32, 1.0, 3.0];
            let mut output = vec![Candidate::default(); rows];
            let mut workspace = LeafKernelWorkspace::default();

            let error = select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                MatrixView::try_from(&values[..], 3, 1).unwrap(),
                MutMatrixView::try_from(output.as_mut_slice(), rows, 1).unwrap(),
                &mut workspace,
            )
            .unwrap_err();

            assert_eq!(
                error.downcast_ref::<LeafKernelError>(),
                Some(&LeafKernelError::InvalidOutputRows { points: 3, rows })
            );
        }

        #[test]
        fn requesting_self_as_an_extra_neighbor_returns_an_error() {
            let values = [0.0_f32, 1.0, 3.0];
            let mut output = [Candidate::default(); 9];
            let mut workspace = LeafKernelWorkspace::default();
            let expected = LeafKernelError::InvalidNeighborCount {
                points: 3,
                neighbors: 3,
                maximum: 2,
            };

            let error = select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                MatrixView::try_from(&values[..], 3, 1).unwrap(),
                MutMatrixView::try_from(&mut output[..], 3, 3).unwrap(),
                &mut workspace,
            )
            .unwrap_err();

            assert_eq!(error.downcast_ref::<LeafKernelError>(), Some(&expected));
        }

        #[test]
        fn l2_neighbors_follow_new_geometry_when_workspace_is_reused() {
            // Given: squared distances on the line determine each point's two neighbors.
            let values = [0.0_f32, 1.0, 3.0, 10.0];
            let expected = [
                Candidate::new(1, (values[0] - values[1]).powi(2)),
                Candidate::new(2, (values[0] - values[2]).powi(2)),
                Candidate::new(0, (values[1] - values[0]).powi(2)),
                Candidate::new(2, (values[1] - values[2]).powi(2)),
                Candidate::new(1, (values[2] - values[1]).powi(2)),
                Candidate::new(0, (values[2] - values[0]).powi(2)),
                Candidate::new(2, (values[3] - values[2]).powi(2)),
                Candidate::new(1, (values[3] - values[1]).powi(2)),
            ];
            let mut output = [Candidate::default(); 8];
            let mut workspace = LeafKernelWorkspace::default();

            // When: first populate output and scratch with a valid larger leaf.
            select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                MatrixView::try_from(&values[..], 4, 1).unwrap(),
                MutMatrixView::try_from(&mut output[..], 4, 2).unwrap(),
                &mut workspace,
            )
            .unwrap();
            assert_eq!(output, expected);

            // A smaller leaf has one finite pair and leaves the other slots unassigned.
            let smaller = [0.0, 2.0, f32::NAN];
            select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                MatrixView::try_from(&smaller[..], 3, 1).unwrap(),
                MutMatrixView::try_from(&mut output[..6], 3, 2).unwrap(),
                &mut workspace,
            )
            .unwrap();
            assert_eq!(
                output[..6],
                [
                    Candidate::new(1, 4.0),
                    Candidate::default(),
                    Candidate::new(0, 4.0),
                    Candidate::default(),
                    Candidate::default(),
                    Candidate::default(),
                ]
            );

            // Growing again must replace the NaN scratch and stale rows. Doubling
            // every coordinate preserves IDs and multiplies squared distances by four.
            let scaled = values.map(|value| 2.0 * value);
            let expected = expected
                .map(|candidate| Candidate::new(candidate.local_idx, 4.0 * candidate.distance));
            select_leaf_neighbors::<_, L2>(
                diskann_wide::ARCH,
                MatrixView::try_from(&scaled[..], 4, 1).unwrap(),
                MutMatrixView::try_from(&mut output[..], 4, 2).unwrap(),
                &mut workspace,
            )
            .unwrap();
            assert_eq!(output, expected);
        }
    }

    mod rank_leaf_distances_tests {
        use super::test_support::rank_distance_fixture;
        use super::*;

        #[test]
        fn lower_triangle_pairs_update_both_endpoints_across_a_simd_group_and_tail() {
            // Given: only three lower-triangle pairs rank. The diagonal and upper
            // triangle contain better distances, so reading either changes the answer.
            let mut distances = [f32::NEG_INFINITY; 18 * 18];
            let mut matrix = MutMatrixView::try_from(&mut distances[..], 18, 18).unwrap();
            for source in 1..18 {
                matrix.row_mut(source)[..source].fill(f32::INFINITY);
            }
            matrix.row_mut(16)[0] = 3.0;
            matrix.row_mut(17)[1] = 2.0;
            matrix.row_mut(17)[16] = 1.0;
            let mut expected = [[Candidate::default(); 2]; 18];
            expected[0][0] = Candidate::new(16, 3.0);
            expected[1][0] = Candidate::new(17, 2.0);
            expected[16] = [Candidate::new(17, 1.0), Candidate::new(0, 3.0)];
            expected[17] = [Candidate::new(16, 1.0), Candidate::new(1, 2.0)];

            // When: row 16 supplies a full SIMD group; row 17 also supplies a tail.
            let actual = rank_distance_fixture(matrix.as_slice(), 18, 2);

            // Then: every reciprocal ID is local to its output row; other rows stay empty.
            assert_eq!(actual.as_slice(), expected.as_flattened());
        }
    }
}
