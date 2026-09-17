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
    simd::PiPNNSIMDSchema,
    topk::{Candidate, UNASSIGNED, with_topk},
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
    A: PiPNNSIMDSchema,
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
    let PartitionKernelWorkspace {
        distance_scratch,
        ranked_leaders,
    } = workspace;
    if distance_scratch.len() < distance_count {
        distance_scratch.resize(distance_count, 0.0);
    }
    let mut distances = MutMatrixView::try_from(
        &mut distance_scratch[..distance_count],
        point_count,
        leader_count,
    )?;
    M::compute_distances(points, leaders, distances.as_mut_view())?;
    rank_leader_distances(arch, distances.as_view(), output, ranked_leaders);
    Ok(())
}

/// Select leader columns from each point's distance row.
fn rank_leader_distances<A: PiPNNSIMDSchema>(
    arch: A,
    distances: MatrixView<'_, f32>,
    mut output: MutMatrixView<'_, u32>,
    candidates: &mut Vec<Candidate>,
) {
    candidates.resize(output.ncols(), Candidate::default());
    with_topk!(output.ncols(), |topk| {
        for (distances, output) in distances.row_iter().zip(output.row_iter_mut()) {
            topk.select_topk(arch, distances, candidates.as_mut_slice());
            for (destination, candidate) in output.iter_mut().zip(candidates.iter()) {
                *destination = candidate.local_idx;
            }
        }
    });
}

// These entry-point tests call GEMM. Miri checks ranking separately.
#[cfg(test)]
#[cfg(not(miri))]
mod tests {
    use super::*;
    use crate::graph::pipnn::{Cosine, CosineNormalized, InnerProduct, L2, scalar_ranking};
    use diskann_utils::views::{MatrixView, MutMatrixView};
    use diskann_vector::distance::Metric;
    use diskann_wide::arch::{self, Target3};

    struct AssignLeaders<M>(M);

    impl<A: PiPNNSIMDSchema, M: PartitionMetric>
        Target3<A, ANNResult<()>, MatrixView<'_, f32>, MatrixView<'_, f32>, MutMatrixView<'_, u32>>
        for AssignLeaders<M>
    {
        fn run(
            self,
            arch: A,
            points: MatrixView<'_, f32>,
            leader_values: MatrixView<'_, f32>,
            output: MutMatrixView<'_, u32>,
        ) -> ANNResult<()> {
            assign_leaders::<_, M>(
                arch,
                points,
                &M::create_leaders(leader_values),
                output,
                &mut PartitionKernelWorkspace::default(),
            )
        }
    }

    mod assign_leaders_tests {
        use super::*;
        use rstest::rstest;

        #[rstest]
        #[case::l2(L2, Metric::L2)]
        #[case::cosine(Cosine, Metric::Cosine)]
        #[case::normalized_cosine(CosineNormalized, Metric::CosineNormalized)]
        #[case::inner_product(InnerProduct, Metric::InnerProduct)]
        fn leader_ids_match_scalar_ranking<M: PartitionMetric>(
            #[case] metric: M,
            #[case] scalar_metric: Metric,
            #[values(2, 7, 8, 9, 15, 16, 17, 127, 128, 129, 384, 768)] dimensions: usize,
            #[values(1, 3, 10, 11)] width: usize,
        ) {
            // Given: 33 leaders supply two SIMD groups and a scalar tail.
            // Four points lie between different leaders on the same irregular arc.
            let normalize = scalar_metric == Metric::CosineNormalized;
            let leader_values =
                scalar_ranking::arc_vectors((0..33).map(f64::from), dimensions, normalize);
            let point_values =
                scalar_ranking::arc_vectors([1.25, 10.5, 21.75, 31.25], dimensions, normalize);
            let leaders = MatrixView::try_from(leader_values.as_slice(), 33, dimensions).unwrap();
            let points = MatrixView::try_from(point_values.as_slice(), 4, dimensions).unwrap();
            let expected: Vec<Vec<_>> = points
                .row_iter()
                .map(|point| {
                    let mut row: Vec<_> = leaders
                        .row_iter()
                        .enumerate()
                        .map(|(leader, vector)| {
                            (
                                leader as u32,
                                scalar_ranking::distance(scalar_metric, point, vector),
                            )
                        })
                        .collect();
                    row.sort_by(|left, right| left.1.total_cmp(&right.1));
                    row
                })
                .collect();
            let tolerance = scalar_ranking::ranking_tolerance(scalar_metric, dimensions);
            let mut output = vec![UNASSIGNED_LEADER; 4 * width];

            // When: create metric-owned leaders and dispatch the complete assignment.
            arch::dispatch3_no_features(
                AssignLeaders(metric),
                points,
                leaders,
                MutMatrixView::try_from(output.as_mut_slice(), 4, width).unwrap(),
            )
            .unwrap();

            // Then: the full L2 distance has the same order as the omitted-norm score.
            for (row, expected) in output.chunks_exact(width).zip(&expected) {
                scalar_ranking::assert_ranked_ids(row, expected, tolerance);
            }
        }

        #[rstest::rstest]
        #[case::missing_row(2)]
        #[case::extra_row(4)]
        fn invalid_output_rows_return_error(#[case] rows: usize) {
            let values = [1.0_f32, 2.0, 3.0];
            let points = MatrixView::try_from(&values[..], 3, 1).unwrap();
            let leaders = Cosine::create_leaders(points);
            let mut output = vec![UNASSIGNED_LEADER; rows];
            let mut workspace = PartitionKernelWorkspace::default();

            let result = assign_leaders::<_, Cosine>(
                diskann_wide::ARCH,
                points,
                &leaders,
                MutMatrixView::try_from(output.as_mut_slice(), rows, 1).unwrap(),
                &mut workspace,
            );

            assert!(result.is_err());
        }

        #[test]
        fn l2_points_select_nearest_leaders() {
            // Given: leader j lies at x=j. Seventeen leaders enter one SIMD block and a tail.
            // The nearest two integer positions to 15.75, -0.25, and 7.25 are respectively
            // (16, 15), (0, 1), and (7, 8); their squared distances are (0.0625, 0.5625),
            // (0.0625, 1.5625), and (0.0625, 0.5625). The last point is unrankable.
            let leader_values: [f32; 17] = std::array::from_fn(|leader| leader as f32);
            let point_values = [15.75, -0.25, 7.25, f32::NAN];
            let expected = [16, 15, 0, 1, 7, 8, UNASSIGNED_LEADER, UNASSIGNED_LEADER];
            let mut output = [0; 8];

            // When: dispatch reaches the real metric and kernel on the available architecture.
            arch::dispatch3_no_features(
                AssignLeaders(L2),
                MatrixView::try_from(&point_values[..], 4, 1).unwrap(),
                MatrixView::try_from(&leader_values[..], 17, 1).unwrap(),
                MutMatrixView::try_from(&mut output[..], 4, 2).unwrap(),
            )
            .unwrap();

            // Then: each point replaces the previous row, including an unrankable final row.
            assert_eq!(output, expected);
        }

        #[test]
        fn cosine_assignments_update_after_workspace_reuse() {
            // Given: unit leaders point right, up, and left. The first point's cosine
            // similarities have order right > up > left; the second reverses right and left.
            let leader_values = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
            let leaders =
                Cosine::create_leaders(MatrixView::try_from(&leader_values[..], 3, 2).unwrap());
            let point_values = [0.9, 0.1, -0.8, 0.2];
            let points = MatrixView::try_from(&point_values[..], 2, 2).unwrap();
            let expected = [0, 1, 2, 1];
            let expected_smaller_stripe = [2, 1];
            let regrown_point_values = [-0.8, 0.2, 0.9, 0.1];
            let expected_regrown_stripe = [2, 1, 0, 1];
            let mut output = [UNASSIGNED_LEADER; 4];
            let mut workspace = PartitionKernelWorkspace::default();

            // When
            assign_leaders::<_, Cosine>(
                diskann_wide::ARCH,
                points,
                &leaders,
                MutMatrixView::try_from(&mut output[..], 2, 2).unwrap(),
                &mut workspace,
            )
            .unwrap();

            // Then
            assert_eq!(output, expected);

            // When: shrink to the second point, whose nearest leader differs from the old prefix.
            assign_leaders::<_, Cosine>(
                diskann_wide::ARCH,
                MatrixView::try_from(&point_values[2..], 1, 2).unwrap(),
                &leaders,
                MutMatrixView::try_from(&mut output[..2], 1, 2).unwrap(),
                &mut workspace,
            )
            .unwrap();

            // Then
            assert_eq!(output[..2], expected_smaller_stripe);

            // When: regrow with reversed rows, so the old tail cannot pass unchanged.
            assign_leaders::<_, Cosine>(
                diskann_wide::ARCH,
                MatrixView::try_from(&regrown_point_values[..], 2, 2).unwrap(),
                &leaders,
                MutMatrixView::try_from(&mut output[..], 2, 2).unwrap(),
                &mut workspace,
            )
            .unwrap();

            // Then
            assert_eq!(output, expected_regrown_stripe);
        }
    }
}
