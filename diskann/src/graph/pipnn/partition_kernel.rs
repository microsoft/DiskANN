/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Partition assignment for PiPNN.
//!
//! [`assign_leaders`] computes the ranking distance from each point of a stripe to
//! each partition leader, then selects the IDs of the nearest leaders. A ranking
//! distance orders the leaders of a point in the same way as the metric distance.
//! Each leader stands for one child partition, so the IDs tell the caller where to
//! send each point.

use crate::ANNResult;
use diskann_utils::views::{MatrixView, MutMatrixView};

use super::{
    check_output_rows, distance_scratch,
    partition_metric::PartitionMetric,
    simd::Simd,
    topk::{Candidate, select_top_k_ids},
};

/// Reusable storage for [`assign_leaders`]. Reuse one workspace across stripes to
/// avoid an allocation for each stripe.
#[derive(Debug, Default)]
pub(super) struct PartitionKernelWorkspace {
    distance_scratch: Vec<f32>,
    nearest_leaders: Vec<Candidate>,
}

/// Assign each point of one stripe to its k nearest partition leaders.
///
/// `points` holds one vector in each row. `leaders` must not be empty. k is the
/// column count of `output`, which has one row per point. Each row receives
/// leader IDs, nearest first. A leader ID is the row of the leader in the matrix
/// that created `leaders`. Equal ranking distances can select either leader.
///
/// NaN and positive infinity never enter a row, so a point can have fewer than k
/// leaders. Each remaining slot holds [`UNASSIGNED`](super::topk::UNASSIGNED).
///
/// # Errors
///
/// Returns an error if `output` does not have one row per point, if the
/// distance matrix size overflows `usize`, or if points and leaders have
/// different dimensions.
pub(super) fn assign_leaders<A, M>(
    arch: A,
    points: MatrixView<'_, f32>,
    leaders: &M::Leaders<'_>,
    output: MutMatrixView<'_, u32>,
    workspace: &mut PartitionKernelWorkspace,
) -> ANNResult<()>
where
    A: Simd,
    M: PartitionMetric,
{
    let point_count = points.nrows();
    check_output_rows(point_count, output.nrows())?;
    let mut distances = distance_scratch(
        &mut workspace.distance_scratch,
        point_count,
        M::leader_count(leaders),
    )?;
    M::compute_distances(points, leaders, distances.as_mut_view())?;
    select_top_k_ids(
        arch,
        distances.as_view(),
        output,
        &mut workspace.nearest_leaders,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::test_support::{self, ArchCheck, for_each_arch};
    use crate::graph::pipnn::topk::UNASSIGNED;
    use crate::graph::pipnn::{Cosine, CosineNormalized, InnerProduct, L2};
    use diskann_vector::distance::Metric;
    use diskann_wide::ARCH;
    use rstest::rstest;
    use std::marker::PhantomData;

    // The metric tests sweep dimensions and the TopK tests sweep widths and lengths.
    // This test checks only that the kernel composes them with leader-column IDs.
    #[rstest]
    #[case::l2(L2, false, [[0,1,3,2], [1,2,3,0], [3,0,2,1]])]
    #[case::cosine(Cosine, false, [[0,1,3,2], [1,2,0,3], [3,0,2,1]])]
    #[case::normalized_cosine(CosineNormalized, true, [[0,1,3,2], [1,2,0,3], [3,0,2,1]])]
    #[case::inner_product(InnerProduct, false, [[0,1,3,2], [1,2,3,0], [0,3,2,1]])]
    fn assignments_are_leader_ids_in_metric_order<M: PartitionMetric>(
        #[case] _metric: M,
        #[case] unit_norm: bool,
        #[case] expected_ids: [[u32; 4]; 3],
    ) {
        let point_values =
            test_support::packed_points(&[[3.0, 2.0], [-1.0, 3.0], [2.0, -4.0]], 2, unit_norm);
        let leader_values = test_support::packed_points(
            &[[4.0, 0.0], [0.0, 3.0], [-2.0, 0.0], [0.0, -1.0]],
            2,
            unit_norm,
        );
        let leaders =
            M::create_leaders(MatrixView::try_from(leader_values.as_slice(), 4, 2).unwrap());
        // A width of six exceeds the four leaders.
        for assignments in [1, 2, 3, 4, 6] {
            let mut output = vec![0; 3 * assignments];

            assign_leaders::<_, M>(
                ARCH,
                MatrixView::try_from(point_values.as_slice(), 3, 2).unwrap(),
                &leaders,
                MutMatrixView::try_from(output.as_mut_slice(), 3, assignments).unwrap(),
                &mut PartitionKernelWorkspace::default(),
            )
            .unwrap_or_else(|error| panic!("assignments={assignments}: {error}"));

            for point in 0..3 {
                let mut expected = expected_ids[point].to_vec();
                expected.truncate(assignments);
                expected.resize(assignments, UNASSIGNED);
                assert_eq!(
                    &output[point * assignments..(point + 1) * assignments],
                    expected,
                    "assignments={assignments}, point={point}"
                );
            }
        }
    }

    // The composition test above uses four leaders, fewer than one SIMD group. This test
    // runs stripes whose leader rows cross SIMD groups, with a fixed-size nearest set
    // and a slice, on each architecture that production can select.
    #[rstest]
    #[case::l2(L2, Metric::L2)]
    #[case::cosine(Cosine, Metric::Cosine)]
    #[case::normalized_cosine(CosineNormalized, Metric::CosineNormalized)]
    #[case::inner_product(InnerProduct, Metric::InnerProduct)]
    fn large_stripes_select_the_nearest_leaders_on_every_architecture<M: PartitionMetric>(
        #[case] _metric: M,
        #[case] metric: Metric,
    ) {
        struct LargeStripes<M> {
            metric: Metric,
            partition_metric: PhantomData<M>,
        }

        impl<M: PartitionMetric> ArchCheck for LargeStripes<M> {
            fn check<A: Simd>(&self, arch: A) {
                let metric = self.metric;
                let arch_name = std::any::type_name::<A>();
                // 35 leaders give distance rows with two SIMD groups and a tail. 129 leaders
                // with 1536 dimensions give many groups and a typical embedding width.
                for (point_count, leader_count, dimensions) in [(33, 35, 129), (65, 129, 1536)] {
                    let mut point_values =
                        test_support::dense_points(point_count, dimensions, 1287);
                    let mut leader_values =
                        test_support::dense_points(leader_count, dimensions, 2026);
                    if metric == Metric::CosineNormalized {
                        test_support::normalize(&mut point_values, dimensions);
                        test_support::normalize(&mut leader_values, dimensions);
                    }
                    let points =
                        MatrixView::try_from(point_values.as_slice(), point_count, dimensions)
                            .unwrap();
                    let leader_matrix =
                        MatrixView::try_from(leader_values.as_slice(), leader_count, dimensions)
                            .unwrap();
                    let leaders = M::create_leaders(leader_matrix);
                    let tolerance = test_support::dense_tolerance(metric, dimensions);
                    // The oracle sorts every leader by its scalar distance to the point.
                    let oracle: Vec<Vec<(u32, f64)>> = (0..point_count)
                        .map(|point| {
                            let mut ranked: Vec<_> = (0..leader_count)
                                .map(|leader| {
                                    let distance = test_support::distance(
                                        metric,
                                        points.row(point),
                                        leader_matrix.row(leader),
                                    );
                                    (leader as u32, distance)
                                })
                                .collect();
                            ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
                            ranked
                        })
                        .collect();

                    // k = 3 uses a fixed-size nearest set. k = 11 uses a slice.
                    for assignments in [3, 11] {
                        let context = format!(
                            "{arch_name}, {metric:?}, shape=({point_count}, {leader_count}, {dimensions}), assignments={assignments}"
                        );
                        let mut output = vec![UNASSIGNED; point_count * assignments];

                        assign_leaders::<A, M>(
                            arch,
                            points,
                            &leaders,
                            MutMatrixView::try_from(
                                output.as_mut_slice(),
                                point_count,
                                assignments,
                            )
                            .unwrap(),
                            &mut PartitionKernelWorkspace::default(),
                        )
                        .unwrap_or_else(|error| panic!("{context}: {error}"));

                        for (point, expected) in oracle.iter().enumerate() {
                            let actual = &output[point * assignments..(point + 1) * assignments];
                            for (rank, &id) in actual.iter().enumerate() {
                                assert!(
                                    !actual[..rank].contains(&id),
                                    "{context}, point={point}: duplicate leader {id}"
                                );
                                let distance = expected
                                    .iter()
                                    .find(|&&(leader, _)| leader == id)
                                    .unwrap_or_else(|| {
                                        panic!("{context}, point={point}: invalid leader {id}")
                                    })
                                    .1;
                                // Equal distances can select either leader, but each rank must
                                // be nearest-first.
                                assert!(
                                    (distance - expected[rank].1).abs() <= tolerance,
                                    "{context}, point={point}, rank={rank}: {distance} != {:?}",
                                    expected[rank]
                                );
                            }
                        }
                    }
                }
            }
        }

        for_each_arch(&LargeStripes::<M> {
            metric,
            partition_metric: PhantomData,
        });
    }

    #[test]
    fn workspace_reuse_does_not_mix_results_from_different_stripes() {
        let leader_values = [0.0, 5.0, 12.0];
        let leaders = L2::create_leaders(MatrixView::try_from(&leader_values[..], 3, 1).unwrap());
        let point_values = [1.0, 7.0, 11.0];
        let expected_ids = [[0, 1, 2], [1, 2, 0], [2, 1, 0]];
        let mut output = Vec::new();
        let mut workspace = PartitionKernelWorkspace::default();

        for (point_count, assignments) in [(3, 1), (1, 3), (2, 2), (3, 1)] {
            output.resize(point_count * assignments, 0);
            assign_leaders::<_, L2>(
                ARCH,
                MatrixView::try_from(&point_values[..point_count], point_count, 1).unwrap(),
                &leaders,
                MutMatrixView::try_from(output.as_mut_slice(), point_count, assignments).unwrap(),
                &mut workspace,
            )
            .unwrap();

            for point in 0..point_count {
                assert_eq!(
                    &output[point * assignments..(point + 1) * assignments],
                    &expected_ids[point][..assignments]
                );
            }
        }
    }

    #[test]
    fn stale_output_and_scratch_do_not_leak_into_new_assignments() {
        let distances = [
            3.0,
            1.0,
            2.0,
            4.0,
            f32::NAN,
            f32::INFINITY,
            f32::NAN,
            f32::INFINITY,
            f32::NAN,
            5.0,
            f32::INFINITY,
            f32::NAN,
        ];
        let mut output = [2; 9];
        let mut candidates = vec![Candidate::new(3, -10.0); 4];

        select_top_k_ids(
            ARCH,
            MatrixView::try_from(&distances[..], 3, 4).unwrap(),
            MutMatrixView::try_from(&mut output[..], 3, 3).unwrap(),
            &mut candidates,
        );

        assert_eq!(
            output,
            [
                1, 2, 0, UNASSIGNED, UNASSIGNED, UNASSIGNED, 1, UNASSIGNED, UNASSIGNED
            ]
        );
    }

    #[test]
    fn output_without_one_row_per_point_is_rejected() {
        let point_values = [1.0, 6.0];
        let leader_values = [0.0, 5.0, 12.0];
        let leaders = L2::create_leaders(MatrixView::try_from(&leader_values[..], 3, 1).unwrap());
        let mut output = [2];

        let error = assign_leaders::<_, L2>(
            ARCH,
            MatrixView::try_from(&point_values[..], 2, 1).unwrap(),
            &leaders,
            MutMatrixView::try_from(&mut output[..], 1, 1).unwrap(),
            &mut PartitionKernelWorkspace::default(),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("invalid kernel output row count 1 for 2 points"),
            "{error}"
        );
    }
}
