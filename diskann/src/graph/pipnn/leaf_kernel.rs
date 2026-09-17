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
    if workspace.distance_scratch.len() < distance_count {
        workspace.distance_scratch.resize(distance_count, 0.0);
    }
    let mut distances = MutMatrixView::try_from(
        &mut workspace.distance_scratch[..distance_count],
        point_count,
        point_count,
    )?;
    M::compute_distances(points, distances.as_mut_slice())?;
    rank_leaf_distances(arch, distances.as_view(), output, &mut workspace.worst);
    Ok(())
}

/// Offer each lower-triangle row to the two endpoint neighbor lists.
fn rank_leaf_distances<A: PiPNNSIMDSchema>(
    arch: A,
    distances: MatrixView<'_, f32>,
    mut output: MutMatrixView<'_, Candidate>,
    worst: &mut Vec<f32>,
) {
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
    #[cfg(not(miri))]
    use crate::graph::pipnn::test_support;
    use crate::graph::pipnn::{Cosine, CosineNormalized, InnerProduct, L2};
    #[cfg(not(miri))]
    use diskann_vector::distance::Metric;
    use diskann_wide::ARCH;
    use rstest::rstest;

    const EMPTY: Candidate = Candidate::new(super::super::topk::UNASSIGNED, f32::INFINITY);

    #[rstest]
    #[case::empty(0, 8, 0)]
    #[case::singleton(1, 8, 0)]
    #[case::zero_requested(5, 0, 0)]
    #[case::below_available(5, 2, 2)]
    #[case::all_available(5, 4, 4)]
    #[case::above_available(5, 9, 4)]
    fn neighbor_count_is_limited_to_other_points(
        #[case] points: usize,
        #[case] requested: usize,
        #[case] expected: usize,
    ) {
        assert_eq!(leaf_neighbor_count(points, requested), expected);
    }

    #[cfg(not(miri))]
    #[rstest]
    #[case::l2(L2, Metric::L2, [[1,3,2,4], [0,2,3,4], [4,0,1,3], [0,2,1,4], [2,1,0,3]])]
    #[case::cosine(Cosine, Metric::Cosine, [[3,1,2,4], [0,3,4,2], [4,3,0,1], [0,2,1,4], [2,1,3,0]])]
    #[case::normalized_cosine(CosineNormalized, Metric::CosineNormalized, [[3,1,2,4], [0,3,4,2], [4,3,0,1], [0,2,1,4], [2,1,3,0]])]
    #[case::inner_product(InnerProduct, Metric::InnerProduct, [[3,1,2,4], [0,3,2,4], [4,3,0,1], [0,2,1,4], [2,1,0,3]])]
    fn neighbors_are_local_non_self_ids_in_metric_order<M: LeafMetric>(
        #[case] _metric: M,
        #[case] scalar_metric: Metric,
        #[case] expected_ids: [[u32; 4]; 5],
        #[values(0, 1, 2, 3, 4)] neighbors: usize,
        #[values(2, 7, 8, 9, 15, 16, 17, 128, 129)] dimensions: usize,
    ) {
        // These five points have distinct scores in each row for every metric.
        let coordinates = [
            [2.0, 2.0],
            [-1.0, 3.0],
            [0.0, -2.0],
            [5.0, 0.0],
            [-3.0, -4.0],
        ];
        let values = test_support::packed_points(
            &coordinates,
            dimensions,
            scalar_metric == Metric::CosineNormalized,
        );
        let points = MatrixView::try_from(values.as_slice(), 5, dimensions).unwrap();
        let mut output = vec![Candidate::new(0, -100.0); 5 * neighbors];
        let mut workspace = LeafKernelWorkspace::default();

        select_leaf_neighbors::<_, M>(
            ARCH,
            points,
            MutMatrixView::try_from(output.as_mut_slice(), 5, neighbors).unwrap(),
            &mut workspace,
        )
        .unwrap();

        for point in 0..5 {
            let actual = &output[point * neighbors..(point + 1) * neighbors];
            assert_eq!(
                actual.iter().map(|c| c.local_idx).collect::<Vec<_>>(),
                expected_ids[point][..neighbors],
                "point={point}"
            );
            for candidate in actual {
                let expected = test_support::distance(
                    scalar_metric,
                    points.row(point),
                    points.row(candidate.local_idx as usize),
                );
                // Only two coordinates are nonzero, so eight f32 ulps at the score scale
                // cover the dot/norm rounding without admitting another neighbor.
                let tolerance = 8.0 * f64::from(f32::EPSILON) * expected.abs().max(1.0);
                assert!(
                    (f64::from(candidate.distance) - expected).abs() <= tolerance,
                    "point={point}, candidate={candidate:?}, expected={expected}"
                );
            }
        }
    }

    #[cfg(not(miri))]
    #[rstest]
    fn neighbors_match_scalar_ranking_across_leaf_sizes_and_counts(
        #[values(1, 2, 16, 17, 18, 33, 34)] point_count: usize,
        #[values(1, 2, 3, 10, 11, 17)] requested: usize,
    ) {
        // Slightly increasing gaps avoid ties between the two sides of each point.
        // Center the coordinates so intermediate norm sums stay exactly representable.
        let mut values: Vec<_> = (0..point_count).map(|i| (64 * i + i * i) as f32).collect();
        let center = values[point_count - 1] / 2.0;
        for value in &mut values {
            *value -= center;
        }
        let neighbors = requested.min(point_count - 1);
        let mut output = vec![EMPTY; point_count * neighbors];

        select_leaf_neighbors::<_, L2>(
            ARCH,
            MatrixView::try_from(values.as_slice(), point_count, 1).unwrap(),
            MutMatrixView::try_from(output.as_mut_slice(), point_count, neighbors).unwrap(),
            &mut LeafKernelWorkspace::default(),
        )
        .unwrap();

        for point in 0..point_count {
            let mut expected: Vec<_> = values
                .iter()
                .enumerate()
                .filter(|&(other, _)| other != point)
                .map(|(other, &value)| {
                    Candidate::new(other as u32, (values[point] - value).powi(2))
                })
                .collect();
            expected.sort_by(|left, right| left.distance.total_cmp(&right.distance));
            expected.truncate(neighbors);
            assert_eq!(
                &output[point * neighbors..(point + 1) * neighbors],
                expected,
                "point={point}, point_count={point_count}, neighbors={neighbors}"
            );
        }
    }

    #[cfg(not(miri))]
    #[rstest]
    #[case::l2(L2, Metric::L2)]
    #[case::cosine(Cosine, Metric::Cosine)]
    #[case::normalized_cosine(CosineNormalized, Metric::CosineNormalized)]
    #[case::inner_product(InnerProduct, Metric::InnerProduct)]
    fn large_dense_leaves_select_nearest_non_self_neighbors<M: LeafMetric>(
        #[case] _metric: M,
        #[case] scalar_metric: Metric,
        #[values((33, 384), (65, 768), (129, 1536), (35, 1537))] shape: (usize, usize),
        #[values(3, 11)] neighbors: usize,
    ) {
        let (point_count, dimensions) = shape;
        let mut values = test_support::dense_points(point_count, dimensions, 1287);
        if scalar_metric == Metric::CosineNormalized {
            test_support::normalize(&mut values, dimensions);
        }
        let points = MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
        let mut output = vec![EMPTY; point_count * neighbors];

        select_leaf_neighbors::<_, M>(
            ARCH,
            points,
            MutMatrixView::try_from(output.as_mut_slice(), point_count, neighbors).unwrap(),
            &mut LeafKernelWorkspace::default(),
        )
        .unwrap();

        // Unnormalized dyadic sums are exact; cosine rounds sqrt/division, while
        // normalized dot products also accumulate coordinate rounding.
        let tolerance = match scalar_metric {
            Metric::L2 | Metric::InnerProduct => 0.0,
            Metric::Cosine => 16.0 * f64::from(f32::EPSILON),
            Metric::CosineNormalized => {
                let roundoff = dimensions as f64 * f64::from(f32::EPSILON);
                roundoff / (1.0 - roundoff)
            }
        };
        for point in 0..point_count {
            let mut expected: Vec<_> = (0..point_count)
                .filter(|&other| other != point)
                .map(|other| {
                    (
                        other as u32,
                        test_support::distance(scalar_metric, points.row(point), points.row(other)),
                    )
                })
                .collect();
            expected.sort_by(|left, right| left.1.total_cmp(&right.1));
            let actual = &output[point * neighbors..(point + 1) * neighbors];
            for (rank, candidate) in actual.iter().enumerate() {
                assert!(
                    !actual[..rank]
                        .iter()
                        .any(|previous| previous.local_idx == candidate.local_idx),
                    "duplicate neighbor for point={point}: {candidate:?}"
                );
                let own_score = expected
                    .iter()
                    .find(|&&(id, _)| id == candidate.local_idx)
                    .unwrap_or_else(|| {
                        panic!("invalid or self neighbor for point={point}: {candidate:?}")
                    })
                    .1;
                assert!(
                    (f64::from(candidate.distance) - own_score).abs() <= tolerance,
                    "shape={shape:?}, point={point}, candidate={candidate:?}, score={own_score}"
                );
                // Equal scores may use either ID, but every rank must be nearest-first.
                assert!(
                    (own_score - expected[rank].1).abs() <= tolerance,
                    "shape={shape:?}, point={point}, rank={rank}, score={own_score}, expected={:?}",
                    expected[rank]
                );
            }
        }
    }

    #[cfg(not(miri))]
    #[test]
    fn workspace_reuse_does_not_mix_results_from_different_leaves() {
        let values = [0.0, 1.0, 4.0, 10.0, 21.0];
        let mut workspace = LeafKernelWorkspace::default();
        let mut output = Vec::new();

        // Grow, shrink, change K, and finish with a singleton using the same buffers.
        for (count, neighbors) in [(4, 1), (2, 1), (5, 3), (1, 0)] {
            output.resize(count * neighbors, Candidate::new(4, -100.0));
            let points = MatrixView::try_from(&values[..count], count, 1).unwrap();
            select_leaf_neighbors::<_, L2>(
                ARCH,
                points,
                MutMatrixView::try_from(output.as_mut_slice(), count, neighbors).unwrap(),
                &mut workspace,
            )
            .unwrap();

            for point in 0..count {
                let mut expected: Vec<_> = (0..count)
                    .filter(|&other| other != point)
                    .map(|other| {
                        Candidate::new(other as u32, (values[point] - values[other]).powi(2))
                    })
                    .collect();
                expected.sort_by(|a, b| a.distance.total_cmp(&b.distance));
                expected.truncate(neighbors);
                assert_eq!(
                    &output[point * neighbors..(point + 1) * neighbors],
                    expected,
                    "count={count}, neighbors={neighbors}, point={point}"
                );
            }
        }
    }

    #[rstest]
    #[case::wrong_rows((2, 1), LeafKernelError::InvalidOutputRows { points: 3, rows: 2 })]
    #[case::too_many_neighbors((3, 3), LeafKernelError::InvalidNeighborCount { points: 3, neighbors: 3, maximum: 2 })]
    fn invalid_output_shape_preserves_output_and_workspace(
        #[case] shape: (usize, usize),
        #[case] expected: LeafKernelError,
    ) {
        let values = [1.0, 2.0, 4.0];
        let mut output = vec![Candidate::new(2, 9.0); shape.0 * shape.1];
        let previous_output = output.clone();
        let mut workspace = LeafKernelWorkspace {
            distance_scratch: vec![11.0, 13.0],
            worst: vec![3.0, 7.0],
        };

        let error = select_leaf_neighbors::<_, L2>(
            ARCH,
            MatrixView::try_from(&values[..], 3, 1).unwrap(),
            MutMatrixView::try_from(output.as_mut_slice(), shape.0, shape.1).unwrap(),
            &mut workspace,
        )
        .unwrap_err();

        assert_eq!(error.downcast_ref::<LeafKernelError>(), Some(&expected));
        assert_eq!(output, previous_output);
        assert_eq!(workspace.distance_scratch, [11.0, 13.0]);
        assert_eq!(workspace.worst, [3.0, 7.0]);
    }

    #[test]
    fn metric_failure_is_returned_without_publishing_neighbors() {
        #[derive(Debug, thiserror::Error)]
        #[error("distance computation failed")]
        struct MetricFailure;
        struct FailingMetric;
        impl LeafMetric for FailingMetric {
            fn compute_distances(_: MatrixView<'_, f32>, _: &mut [f32]) -> ANNResult<()> {
                Err(ANNError::new(MetricFailure))
            }
        }
        let values = [1.0, 2.0];
        let mut output = [Candidate::new(1, 7.0), Candidate::new(0, 7.0)];

        let error = select_leaf_neighbors::<_, FailingMetric>(
            ARCH,
            MatrixView::try_from(&values[..], 2, 1).unwrap(),
            MutMatrixView::try_from(&mut output[..], 2, 1).unwrap(),
            &mut LeafKernelWorkspace::default(),
        )
        .unwrap_err();

        assert!(error.is::<MetricFailure>());
        assert_eq!(output, [Candidate::new(1, 7.0), Candidate::new(0, 7.0)]);
    }

    // Keep this module path available to the nightly Miri selector.
    mod rank_leaf_distances_tests {
        use super::*;

        #[test]
        fn ranking_reads_only_pairs_in_the_strict_lower_triangle() {
            // Diagonal and upper entries are deliberately better than every real pair.
            let distances = [
                -100.0, -100.0, -100.0, -100.0, 7.0, -100.0, -100.0, -100.0, 3.0, 8.0, -100.0,
                -100.0, 5.0, 2.0, 6.0, -100.0,
            ];
            let mut output = [Candidate::new(0, -200.0); 8];
            let mut limits = vec![-200.0; 4];

            rank_leaf_distances(
                ARCH,
                MatrixView::try_from(&distances[..], 4, 4).unwrap(),
                MutMatrixView::try_from(&mut output[..], 4, 2).unwrap(),
                &mut limits,
            );

            let expected = [
                [Candidate::new(2, 3.0), Candidate::new(3, 5.0)],
                [Candidate::new(3, 2.0), Candidate::new(0, 7.0)],
                [Candidate::new(0, 3.0), Candidate::new(3, 6.0)],
                [Candidate::new(1, 2.0), Candidate::new(0, 5.0)],
            ];
            assert_eq!(output, expected.as_flattened());
            assert_eq!(limits, [5.0, 7.0, 6.0, 5.0]);
        }

        #[rstest]
        #[case::nan(f32::NAN)]
        #[case::infinity(f32::INFINITY)]
        fn unrankable_pairs_leave_unassigned_slots(#[case] unrankable: f32) {
            let mut distances = [unrankable; 9];
            distances[6] = 4.0;
            let mut output = [Candidate::new(1, -100.0); 6];
            let mut limits = vec![-100.0; 5];

            rank_leaf_distances(
                ARCH,
                MatrixView::try_from(&distances[..], 3, 3).unwrap(),
                MutMatrixView::try_from(&mut output[..], 3, 2).unwrap(),
                &mut limits,
            );

            assert_eq!(
                output,
                [
                    Candidate::new(2, 4.0),
                    EMPTY,
                    EMPTY,
                    EMPTY,
                    Candidate::new(0, 4.0),
                    EMPTY
                ]
            );
            assert_eq!(limits, [f32::INFINITY; 3]);
        }
    }
}
