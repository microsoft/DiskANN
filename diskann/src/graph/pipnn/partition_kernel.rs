/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Select partition centers for PiPNN point assignment.
//!
//! [`assign_leaders`] computes point-to-leader ranking values and selects the
//! nearest leader-column IDs for partition scatter. Each sampled leader
//! represents one child partition.

use crate::{ANNError, ANNResult};
use diskann_utils::views::{MatrixView, MutMatrixView};

use super::{
    partition_metric::PartitionMetric,
    simd::Simd,
    topk::{Candidate, Ranker, TopKVisitor, UNASSIGNED, with_topk},
};

/// No sampled partition center was rankable for this output slot.
pub(super) const UNASSIGNED_LEADER: u32 = UNASSIGNED;

/// Reusable storage for one point-stripe numerical pipeline.
#[derive(Default)]
pub(super) struct PartitionKernelWorkspace {
    distance_scratch: Vec<f32>,
    ranked_leaders: Vec<Candidate>,
}

/// Assign one packed point stripe to metric-owned partition leaders.
///
/// Each output row contains leader-column IDs ordered by increasing ranking
/// value. Equal values can select either leader. NaN and positive infinity are
/// not retained.
///
/// A point can have fewer assignments than the output width. Each remaining
/// slot contains [`UNASSIGNED_LEADER`].
///
/// # Errors
///
/// Returns an error for invalid GEMM input or an output row count different from
/// the point count. A row-count mismatch leaves output and workspace unchanged.
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
    if output.nrows() != point_count {
        return Err(ANNError::message(format!(
            "invalid partition output row count {} for {point_count} points",
            output.nrows()
        )));
    }
    let leader_count = M::leader_count(leaders);
    let distance_count = point_count * leader_count;
    if workspace.distance_scratch.len() < distance_count {
        workspace.distance_scratch.resize(distance_count, 0.0);
    }
    let mut distances = MutMatrixView::try_from(
        &mut workspace.distance_scratch[..distance_count],
        point_count,
        leader_count,
    )?;
    M::compute_distances(points, leaders, distances.as_mut_view())?;
    rank_leader_distances(
        arch,
        distances.as_view(),
        output,
        &mut workspace.ranked_leaders,
    );
    Ok(())
}

/// Select leader columns from each point's distance row.
fn rank_leader_distances<A: Simd>(
    arch: A,
    distances: MatrixView<'_, f32>,
    output: MutMatrixView<'_, u32>,
    candidates: &mut Vec<Candidate>,
) {
    candidates.resize(output.ncols(), Candidate::default());
    with_topk(
        candidates.as_mut_slice(),
        RankLeaders {
            arch,
            distances,
            output,
        },
    );
}

struct RankLeaders<'a, A> {
    arch: A,
    distances: MatrixView<'a, f32>,
    output: MutMatrixView<'a, u32>,
}

impl<A: Simd> TopKVisitor for RankLeaders<'_, A> {
    #[inline]
    fn visit<R: Ranker>(mut self, mut ranker: R) {
        for (distances, output) in self.distances.row_iter().zip(self.output.row_iter_mut()) {
            ranker.select_topk(self.arch, distances);
            for (destination, candidate) in output.iter_mut().zip(ranker.as_ref().iter()) {
                *destination = candidate.local_idx;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::test_support;
    use crate::graph::pipnn::{Cosine, CosineNormalized, InnerProduct, L2};
    use diskann_vector::distance::Metric;
    use diskann_wide::ARCH;
    use rstest::rstest;

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
        for assignments in [1, 2, 3, 4, 6] {
            for dimensions in [2, 7, 8, 9, 15, 16, 17, 128, 129] {
                let point_values = test_support::packed_points(
                    &[[3.0, 2.0], [-1.0, 3.0], [2.0, -4.0]],
                    dimensions,
                    unit_norm,
                );
                let leader_values = test_support::packed_points(
                    &[[4.0, 0.0], [0.0, 3.0], [-2.0, 0.0], [0.0, -1.0]],
                    dimensions,
                    unit_norm,
                );
                let leaders = M::create_leaders(
                    MatrixView::try_from(leader_values.as_slice(), 4, dimensions).unwrap(),
                );
                let mut output = vec![0; 3 * assignments];

                assign_leaders::<_, M>(
                    ARCH,
                    MatrixView::try_from(point_values.as_slice(), 3, dimensions).unwrap(),
                    &leaders,
                    MutMatrixView::try_from(output.as_mut_slice(), 3, assignments).unwrap(),
                    &mut PartitionKernelWorkspace::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("assignments={assignments}, dimensions={dimensions}: {error}")
                });

                for point in 0..3 {
                    let mut expected = expected_ids[point].to_vec();
                    expected.truncate(assignments);
                    expected.resize(assignments, UNASSIGNED_LEADER);
                    assert_eq!(
                        &output[point * assignments..(point + 1) * assignments],
                        expected,
                        "assignments={assignments}, dimensions={dimensions}, point={point}"
                    );
                }
            }
        }
    }

    #[test]
    fn assignments_match_nearest_leaders_across_counts_and_widths() {
        for leader_count in [1, 5, 16, 17, 33] {
            for assignments in [1, 3, 10, 11, 17] {
                // Queries avoid midpoints, so each leader has a distinct distance.
                let leader_values: Vec<_> = (0..leader_count).map(|i| 2.0 * i as f32).collect();
                let point_values = [-1.0, 1.5, 2.0 * (leader_count - 1) as f32 + 0.25];
                let leaders = L2::create_leaders(
                    MatrixView::try_from(leader_values.as_slice(), leader_count, 1).unwrap(),
                );
                let mut output = vec![0; 3 * assignments];

                assign_leaders::<_, L2>(
                    ARCH,
                    MatrixView::try_from(&point_values[..], 3, 1).unwrap(),
                    &leaders,
                    MutMatrixView::try_from(output.as_mut_slice(), 3, assignments).unwrap(),
                    &mut PartitionKernelWorkspace::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("leader_count={leader_count}, assignments={assignments}: {error}")
                });

                for (point, &coordinate) in point_values.iter().enumerate() {
                    let mut ordered: Vec<_> = leader_values
                        .iter()
                        .enumerate()
                        .map(|(id, &value)| (id as u32, (coordinate - value).powi(2)))
                        .collect();
                    ordered.sort_by(|left, right| left.1.total_cmp(&right.1));
                    let mut expected: Vec<_> = ordered
                        .iter()
                        .take(assignments)
                        .map(|&(id, _)| id)
                        .collect();
                    expected.resize(assignments, UNASSIGNED_LEADER);
                    assert_eq!(
                        &output[point * assignments..(point + 1) * assignments],
                        expected,
                        "point={point}, leader_count={leader_count}, assignments={assignments}"
                    );
                }
            }
        }
    }

    #[rstest]
    #[case::l2(L2, Metric::L2)]
    #[case::cosine(Cosine, Metric::Cosine)]
    #[case::normalized_cosine(CosineNormalized, Metric::CosineNormalized)]
    #[case::inner_product(InnerProduct, Metric::InnerProduct)]
    fn large_dense_stripes_select_nearest_leaders<M: PartitionMetric>(
        #[case] _metric: M,
        #[case] scalar_metric: Metric,
    ) {
        for shape in [(9, 33, 384), (17, 65, 768), (33, 129, 1536), (5, 35, 1537)] {
            for assignments in [3, 11] {
                let (point_count, leader_count, dimensions) = shape;
                let mut point_values = test_support::dense_points(point_count, dimensions, 1287);
                let mut leader_values = test_support::dense_points(leader_count, dimensions, 2026);
                if scalar_metric == Metric::CosineNormalized {
                    test_support::normalize(&mut point_values, dimensions);
                    test_support::normalize(&mut leader_values, dimensions);
                }
                let points =
                    MatrixView::try_from(point_values.as_slice(), point_count, dimensions).unwrap();
                let leader_matrix =
                    MatrixView::try_from(leader_values.as_slice(), leader_count, dimensions)
                        .unwrap();
                let leaders = M::create_leaders(leader_matrix);
                let mut output = vec![UNASSIGNED_LEADER; point_count * assignments];

                assign_leaders::<_, M>(
                    ARCH,
                    points,
                    &leaders,
                    MutMatrixView::try_from(output.as_mut_slice(), point_count, assignments)
                        .unwrap(),
                    &mut PartitionKernelWorkspace::default(),
                )
                .unwrap_or_else(|error| {
                    panic!("shape={shape:?}, assignments={assignments}: {error}")
                });

                let tolerance = match scalar_metric {
                    Metric::L2 | Metric::InnerProduct => 0.0,
                    // Dyadic inputs make the dot/norm sums exact before cosine conversion.
                    Metric::Cosine => 16.0 * f64::from(f32::EPSILON),
                    Metric::CosineNormalized => {
                        let roundoff = dimensions as f64 * f64::from(f32::EPSILON);
                        roundoff / (1.0 - roundoff)
                    }
                };
                for point in 0..point_count {
                    let mut expected: Vec<_> = (0..leader_count)
                        .map(|leader| {
                            (
                                leader as u32,
                                test_support::distance(
                                    scalar_metric,
                                    points.row(point),
                                    leader_matrix.row(leader),
                                ),
                            )
                        })
                        .collect();
                    expected.sort_by(|left, right| left.1.total_cmp(&right.1));
                    let actual = &output[point * assignments..(point + 1) * assignments];
                    for (rank, &id) in actual.iter().enumerate() {
                        assert!(
                            !actual[..rank].contains(&id),
                            "shape={shape:?}, assignments={assignments}, duplicate leader {id} for point={point}"
                        );
                        let score = expected
                            .iter()
                            .find(|&&(leader, _)| leader == id)
                            .unwrap_or_else(|| panic!("shape={shape:?}, assignments={assignments}, invalid leader {id} for point={point}"))
                            .1;
                        assert!(
                            (score - expected[rank].1).abs() <= tolerance,
                            "shape={shape:?}, assignments={assignments}, point={point}, rank={rank}, score={score}, expected={:?}",
                            expected[rank]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn workspace_reuse_replaces_assignments_for_each_stripe() {
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
    fn each_distance_row_replaces_previous_assignments_and_unused_slots() {
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

        rank_leader_distances(
            ARCH,
            MatrixView::try_from(&distances[..], 3, 4).unwrap(),
            MutMatrixView::try_from(&mut output[..], 3, 3).unwrap(),
            &mut candidates,
        );

        assert_eq!(
            output,
            [
                1,
                2,
                0,
                UNASSIGNED_LEADER,
                UNASSIGNED_LEADER,
                UNASSIGNED_LEADER,
                1,
                UNASSIGNED_LEADER,
                UNASSIGNED_LEADER
            ]
        );
    }

    #[test]
    fn invalid_output_rows_preserve_assignments_and_workspace() {
        let point_values = [1.0, 6.0];
        let leader_values = [0.0, 5.0, 12.0];
        let leaders = L2::create_leaders(MatrixView::try_from(&leader_values[..], 3, 1).unwrap());
        let mut output = [2];
        let previous_candidates = vec![Candidate::new(1, 7.0)];
        let mut workspace = PartitionKernelWorkspace {
            distance_scratch: vec![9.0, 11.0],
            ranked_leaders: previous_candidates.clone(),
        };

        let error = assign_leaders::<_, L2>(
            ARCH,
            MatrixView::try_from(&point_values[..], 2, 1).unwrap(),
            &leaders,
            MutMatrixView::try_from(&mut output[..], 1, 1).unwrap(),
            &mut workspace,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("invalid partition output row count 1 for 2 points")
        );
        assert_eq!(output, [2]);
        assert_eq!(workspace.distance_scratch, [9.0, 11.0]);
        assert_eq!(workspace.ranked_leaders, previous_candidates);
    }

    #[test]
    fn metric_dimension_errors_are_returned_without_publishing_assignments() {
        let point_values = [1.0, 2.0, 3.0];
        let leader_values = [0.0, 1.0, 2.0, 3.0];
        let leaders = L2::create_leaders(MatrixView::try_from(&leader_values[..], 2, 2).unwrap());
        let mut output = [1];

        let error = assign_leaders::<_, L2>(
            ARCH,
            MatrixView::try_from(&point_values[..], 1, 3).unwrap(),
            &leaders,
            MutMatrixView::try_from(&mut output[..], 1, 1).unwrap(),
            &mut PartitionKernelWorkspace::default(),
        )
        .unwrap_err();

        assert_eq!(
            error.downcast_ref::<diskann_linalg::SgemmError>(),
            Some(&diskann_linalg::SgemmError::InvalidMatrixDimensions {
                matrix_name: diskann_linalg::MatrixName::B,
                expected_rows: 3,
                expected_cols: 2,
                actual_len: 4,
            })
        );
        assert_eq!(output, [1]);
    }
}
