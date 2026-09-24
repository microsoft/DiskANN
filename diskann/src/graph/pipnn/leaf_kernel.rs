/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Nearest-neighbor selection inside one PiPNN leaf.
//!
//! [`select_leaf_neighbors`] asks the metric for the distances between all point
//! pairs of the leaf. It then runs a pair scan, which reads each pair once and
//! offers it to the nearest sets of both points.

use crate::ANNResult;
use diskann_utils::views::{MatrixView, MutMatrixView};

use super::{
    check_output_rows, distance_scratch,
    leaf_metric::LeafMetric,
    simd::Simd,
    topk::{Candidate, select_top_k_symmetric},
};

/// Reusable storage for [`select_leaf_neighbors`]. Reuse one workspace across
/// leaves to avoid an allocation for each leaf.
#[derive(Debug, Default)]
pub(super) struct LeafKernelWorkspace {
    distance_scratch: Vec<f32>,
    kth_distances: Vec<f32>,
}

/// Find the k nearest other points of each point in one leaf.
///
/// `points` holds one leaf vector in each row and must not be empty. k is the
/// column count of `output`, which has one row per point. Each row receives the
/// nearest set of one point: its nearest other points, nearest first, as
/// positions in the leaf. Equal distances can select either point. NaN and
/// positive infinity never enter a row. A row can be wider than the number of
/// other points. Unfilled slots hold [`Candidate::EMPTY`].
///
/// # Errors
///
/// Returns an error if `output` does not have one row per point, or if the
/// distance matrix size overflows `usize`.
pub(super) fn select_leaf_neighbors<A, M>(
    arch: A,
    points: MatrixView<'_, f32>,
    output: MutMatrixView<'_, Candidate>,
    workspace: &mut LeafKernelWorkspace,
) -> ANNResult<()>
where
    A: Simd,
    M: LeafMetric,
{
    let point_count = points.nrows();
    check_output_rows(point_count, output.nrows())?;
    let mut distances =
        distance_scratch(&mut workspace.distance_scratch, point_count, point_count)?;
    M::compute_distances(points, distances.as_mut_view())?;
    select_top_k_symmetric(
        arch,
        distances.as_view(),
        output,
        &mut workspace.kth_distances,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::test_support::{self, ArchCheck, for_each_arch};
    use crate::graph::pipnn::{Cosine, CosineNormalized, InnerProduct, L2};
    use diskann_vector::distance::Metric;
    use diskann_wide::ARCH;
    use rstest::rstest;
    use std::marker::PhantomData;

    // The metric tests sweep dimensions and the TopK tests sweep widths and lengths.
    // This test checks only that the kernel composes them with leaf-local IDs.
    #[rstest]
    #[case::l2(L2, Metric::L2, [[1,3,2,4], [0,2,3,4], [4,0,1,3], [0,2,1,4], [2,1,0,3]])]
    #[case::cosine(Cosine, Metric::Cosine, [[3,1,2,4], [0,3,4,2], [4,3,0,1], [0,2,1,4], [2,1,3,0]])]
    #[case::normalized_cosine(CosineNormalized, Metric::CosineNormalized, [[3,1,2,4], [0,3,4,2], [4,3,0,1], [0,2,1,4], [2,1,3,0]])]
    #[case::inner_product(InnerProduct, Metric::InnerProduct, [[3,1,2,4], [0,3,2,4], [4,3,0,1], [0,2,1,4], [2,1,0,3]])]
    fn neighbors_are_other_points_of_the_leaf_in_metric_order<M: LeafMetric>(
        #[case] _metric: M,
        #[case] scalar_metric: Metric,
        #[case] expected_ids: [[u32; 4]; 5],
    ) {
        // These five points have distinct distances in each row for every metric.
        let coordinates = [
            [2.0, 2.0],
            [-1.0, 3.0],
            [0.0, -2.0],
            [5.0, 0.0],
            [-3.0, -4.0],
        ];
        let values =
            test_support::packed_points(&coordinates, 2, scalar_metric == Metric::CosineNormalized);
        let points = MatrixView::try_from(values.as_slice(), 5, 2).unwrap();
        // Widths above four exceed the other points in the leaf.
        for neighbors in [0, 1, 2, 3, 4, 6] {
            let mut output = vec![Candidate::new(0, -100.0); 5 * neighbors];

            select_leaf_neighbors::<_, M>(
                ARCH,
                points,
                MutMatrixView::try_from(output.as_mut_slice(), 5, neighbors).unwrap(),
                &mut LeafKernelWorkspace::default(),
            )
            .unwrap_or_else(|error| panic!("neighbors={neighbors}: {error}"));

            let filled = neighbors.min(4);
            for point in 0..5 {
                let actual = &output[point * neighbors..(point + 1) * neighbors];
                assert_eq!(
                    actual[..filled]
                        .iter()
                        .map(|c| c.local_idx)
                        .collect::<Vec<_>>(),
                    expected_ids[point][..filled],
                    "neighbors={neighbors}, point={point}"
                );
                assert!(
                    actual[filled..].iter().all(|&c| c == Candidate::EMPTY),
                    "neighbors={neighbors}, point={point}: {actual:?}"
                );
                for candidate in &actual[..filled] {
                    let expected = test_support::distance(
                        scalar_metric,
                        points.row(point),
                        points.row(candidate.local_idx as usize),
                    );
                    // Only two coordinates are nonzero, so eight f32 ulps at the distance scale
                    // cover the dot/norm rounding without admitting another neighbor.
                    let tolerance = 8.0 * f64::from(f32::EPSILON) * expected.abs().max(1.0);
                    assert!(
                        (f64::from(candidate.distance) - expected).abs() <= tolerance,
                        "neighbors={neighbors}, point={point}, candidate={candidate:?}, expected={expected}"
                    );
                }
            }
        }
    }

    // The composition test above uses five points, fewer than one SIMD group. This test
    // runs leaves whose distance rows cross SIMD groups, with fixed-size nearest sets
    // and slices, on each architecture that production can select.
    #[rstest]
    #[case::l2(L2, Metric::L2)]
    #[case::cosine(Cosine, Metric::Cosine)]
    #[case::normalized_cosine(CosineNormalized, Metric::CosineNormalized)]
    #[case::inner_product(InnerProduct, Metric::InnerProduct)]
    fn large_leaves_select_the_nearest_neighbors_on_every_architecture<M: LeafMetric>(
        #[case] _metric: M,
        #[case] metric: Metric,
    ) {
        struct LargeLeaves<M> {
            metric: Metric,
            leaf_metric: PhantomData<M>,
        }

        impl<M: LeafMetric> ArchCheck for LargeLeaves<M> {
            fn check<A: Simd>(&self, arch: A) {
                let metric = self.metric;
                let arch_name = std::any::type_name::<A>();
                // 35 points give distance rows with two SIMD groups and a tail. 129 points
                // with 1536 dimensions give many groups and a typical embedding width.
                for (point_count, dimensions) in [(35, 129), (129, 1536)] {
                    let mut values = test_support::dense_points(point_count, dimensions, 1287);
                    if metric == Metric::CosineNormalized {
                        test_support::normalize(&mut values, dimensions);
                    }
                    let points =
                        MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
                    let tolerance = test_support::dense_tolerance(metric, dimensions);
                    // The oracle sorts every other point by its scalar distance.
                    let oracle: Vec<Vec<(u32, f64)>> = (0..point_count)
                        .map(|point| {
                            let mut others: Vec<_> = (0..point_count)
                                .filter(|&other| other != point)
                                .map(|other| {
                                    let distance = test_support::distance(
                                        metric,
                                        points.row(point),
                                        points.row(other),
                                    );
                                    (other as u32, distance)
                                })
                                .collect();
                            others.sort_by(|left, right| left.1.total_cmp(&right.1));
                            others
                        })
                        .collect();

                    // k = 3 uses fixed-size nearest sets. k = 11 uses slices.
                    for neighbors in [3, 11] {
                        let context = format!(
                            "{arch_name}, {metric:?}, shape=({point_count}, {dimensions}), neighbors={neighbors}"
                        );
                        let mut output = vec![Candidate::EMPTY; point_count * neighbors];

                        select_leaf_neighbors::<A, M>(
                            arch,
                            points,
                            MutMatrixView::try_from(output.as_mut_slice(), point_count, neighbors)
                                .unwrap(),
                            &mut LeafKernelWorkspace::default(),
                        )
                        .unwrap_or_else(|error| panic!("{context}: {error}"));

                        for (point, expected) in oracle.iter().enumerate() {
                            let actual = &output[point * neighbors..(point + 1) * neighbors];
                            for (rank, candidate) in actual.iter().enumerate() {
                                assert!(
                                    !actual[..rank]
                                        .iter()
                                        .any(|previous| previous.local_idx == candidate.local_idx),
                                    "{context}, point={point}: duplicate {candidate:?}"
                                );
                                let own_distance = expected
                                    .iter()
                                    .find(|&&(id, _)| id == candidate.local_idx)
                                    .unwrap_or_else(|| {
                                        panic!("{context}, point={point}: invalid or self {candidate:?}")
                                    })
                                    .1;
                                assert!(
                                    (f64::from(candidate.distance) - own_distance).abs()
                                        <= tolerance,
                                    "{context}, point={point}: {candidate:?} != {own_distance}"
                                );
                                // Equal distances can select either ID, but each rank must be
                                // nearest-first.
                                assert!(
                                    (own_distance - expected[rank].1).abs() <= tolerance,
                                    "{context}, point={point}, rank={rank}: {own_distance} != {:?}",
                                    expected[rank]
                                );
                            }
                        }
                    }
                }
            }
        }

        for_each_arch(&LargeLeaves::<M> {
            metric,
            leaf_metric: PhantomData,
        });
    }

    #[test]
    fn workspace_reuse_does_not_mix_results_from_different_leaves() {
        let values = [0.0, 1.0, 4.0, 10.0, 21.0];
        let mut workspace = LeafKernelWorkspace::default();
        let mut output = Vec::new();

        // Grow, shrink, change K, and finish with a singleton wider than its leaf.
        for (count, neighbors) in [(4, 1), (2, 1), (5, 3), (1, 2)] {
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
                expected.resize(neighbors, Candidate::EMPTY);
                assert_eq!(
                    &output[point * neighbors..(point + 1) * neighbors],
                    expected,
                    "count={count}, neighbors={neighbors}, point={point}"
                );
            }
        }
    }

    #[test]
    fn output_without_one_row_per_point_is_rejected() {
        let values = [1.0, 2.0, 4.0];
        let mut output = [Candidate::EMPTY; 2];

        let error = select_leaf_neighbors::<_, L2>(
            ARCH,
            MatrixView::try_from(&values[..], 3, 1).unwrap(),
            MutMatrixView::try_from(&mut output[..], 2, 1).unwrap(),
            &mut LeafKernelWorkspace::default(),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("invalid kernel output row count 2 for 3 points"),
            "{error}"
        );
    }

    #[test]
    fn ranking_reads_only_pairs_in_the_strict_lower_triangle() {
        // Diagonal and upper entries are deliberately better than every real pair.
        let distances = [
            -100.0, -100.0, -100.0, -100.0, 7.0, -100.0, -100.0, -100.0, 3.0, 8.0, -100.0, -100.0,
            5.0, 2.0, 6.0, -100.0,
        ];
        let mut output = [Candidate::new(0, -200.0); 8];
        let mut kth_distances = vec![-200.0; 4];

        select_top_k_symmetric(
            ARCH,
            MatrixView::try_from(&distances[..], 4, 4).unwrap(),
            MutMatrixView::try_from(&mut output[..], 4, 2).unwrap(),
            &mut kth_distances,
        );

        let expected = [
            [Candidate::new(2, 3.0), Candidate::new(3, 5.0)],
            [Candidate::new(3, 2.0), Candidate::new(0, 7.0)],
            [Candidate::new(0, 3.0), Candidate::new(3, 6.0)],
            [Candidate::new(1, 2.0), Candidate::new(0, 5.0)],
        ];
        assert_eq!(output, expected.as_flattened());
        assert_eq!(kth_distances, [5.0, 7.0, 6.0, 5.0]);
    }
}
