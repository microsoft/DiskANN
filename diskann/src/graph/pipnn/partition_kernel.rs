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
    arch.run(move || {
        let mut worst = [f32::INFINITY];
        with_topk!(
            MutMatrixView::row_vector(candidates.as_mut_slice()),
            &mut worst,
            |topk| {
                for (distances, output) in distances.row_iter().zip(output.row_iter_mut()) {
                    topk.replace_topk(arch, 0, distances);
                    for (destination, candidate) in output.iter_mut().zip(topk.candidates(0)) {
                        *destination = candidate.local_idx;
                    }
                }
            }
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::{Cosine, L2};
    use diskann_utils::views::{MatrixView, MutMatrixView};
    use diskann_wide::arch::{self, Target3};

    struct AssignL2;

    impl<A: PiPNNSIMDSchema>
        Target3<A, ANNResult<()>, MatrixView<'_, f32>, MatrixView<'_, f32>, MutMatrixView<'_, u32>>
        for AssignL2
    {
        fn run(
            self,
            arch: A,
            points: MatrixView<'_, f32>,
            leader_values: MatrixView<'_, f32>,
            output: MutMatrixView<'_, u32>,
        ) -> ANNResult<()> {
            assign_leaders::<_, L2>(
                arch,
                points,
                &L2::create_leaders(leader_values),
                output,
                &mut PartitionKernelWorkspace::default(),
            )
        }
    }

    mod assign_leaders_tests {
        use super::*;

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
        fn assignment_projects_each_points_l2_leader_columns_with_simd_and_tail() {
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
                AssignL2,
                MatrixView::try_from(&point_values[..], 4, 1).unwrap(),
                MatrixView::try_from(&leader_values[..], 17, 1).unwrap(),
                MutMatrixView::try_from(&mut output[..], 4, 2).unwrap(),
            )
            .unwrap();

            // Then: each point replaces the previous row, including an unrankable final row.
            assert_eq!(output, expected);
        }

        #[test]
        fn assignment_reuses_workspace_after_shrinking_and_regrowing_a_cosine_stripe() {
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
