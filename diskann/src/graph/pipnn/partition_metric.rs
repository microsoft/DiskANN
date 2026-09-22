/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Build ranking distances from point stripes to partition leaders.
//!
//! A ranking distance preserves nearest-first order. One leader set serves all
//! point stripes in a partition split.

use crate::{ANNError, ANNResult};
use diskann_linalg::Transpose;
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::{
    Norm,
    norm::{FastL2Norm, FastL2NormSquared},
};

use super::{Cosine, CosineNormalized, InnerProduct, L2, cosine_distance};

/// Store leader values with immutable metric data.
///
/// L2 stores squared norms. Cosine stores norms. Construction computes them once
/// before point stripes share the leader set. Other metrics need no norms.
pub(super) struct PartitionLeaders<'a, Norms> {
    values: MatrixView<'a, f32>,
    norms: Norms,
}

/// Fill one point-to-leader ranking matrix.
///
/// The associated leader type hides metric data from the caller. The caller
/// creates one value and shares it across all point stripes.
pub(super) trait PartitionMetric: Send + Sync + 'static {
    /// Leader values and immutable metric data for one partition split.
    type Leaders<'a>: Sync;

    /// Bind one non-empty leader matrix to this metric.
    ///
    /// Partitioning creates at least one leader before it calls this function.
    fn create_leaders<'a>(values: MatrixView<'a, f32>) -> Self::Leaders<'a>;

    /// Return the number of leaders in a metric-owned leader set.
    fn leader_count(leaders: &Self::Leaders<'_>) -> usize;

    /// Compute one row-major point-to-leader ranking buffer.
    ///
    /// Values preserve nearest-first order. L2 omits the point's squared norm,
    /// which is constant across its row. Normalized cosine and inner product
    /// return `-dot`; cosine returns `1 - similarity`.
    ///
    /// `storage` has `points.nrows()` rows and `leader_count` columns.
    /// A zero distance can have either sign. Equal distances can select either leader.
    fn compute_distances(
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        storage: MutMatrixView<'_, f32>,
    ) -> ANNResult<()>;
}

/// Compute squared norms with DiskANN's vector implementation.
fn l2_squared_norms(vectors: MatrixView<'_, f32>) -> Vec<f32> {
    vectors
        .row_iter()
        .map(|vector| FastL2NormSquared.evaluate(vector))
        .collect()
}

/// Compute cosine norms with the same reduction for points and leaders.
fn cosine_norms(vectors: MatrixView<'_, f32>) -> Vec<f32> {
    vectors
        .row_iter()
        .map(|vector| FastL2Norm.evaluate(vector))
        .collect()
}

impl PartitionMetric for L2 {
    type Leaders<'a> = PartitionLeaders<'a, Vec<f32>>;

    fn create_leaders<'a>(values: MatrixView<'a, f32>) -> Self::Leaders<'a> {
        PartitionLeaders {
            values,
            norms: l2_squared_norms(values),
        }
    }

    fn leader_count(leaders: &Self::Leaders<'_>) -> usize {
        leaders.values.nrows()
    }

    fn compute_distances(
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        mut storage: MutMatrixView<'_, f32>,
    ) -> ANNResult<()> {
        if storage.nrows() != points.nrows() || storage.ncols() != leaders.values.nrows() {
            return Err(ANNError::message("point-to-leader output shape mismatch"));
        }
        // The point norm is constant across a point row. It cannot change ranking.
        // Initialize each point row before GEMM adds the dot-product term.
        for row in storage.row_iter_mut() {
            row.copy_from_slice(&leaders.norms);
        }
        diskann_linalg::sgemm(
            Transpose::None,
            Transpose::Ordinary,
            points.nrows(),
            leaders.values.nrows(),
            points.ncols(),
            -2.0,
            points.as_slice(),
            leaders.values.as_slice(),
            Some(1.0),
            storage.as_mut_slice(),
        )
        .map_err(ANNError::new)?;
        Ok(())
    }
}

impl PartitionMetric for Cosine {
    type Leaders<'a> = PartitionLeaders<'a, Vec<f32>>;

    fn create_leaders<'a>(values: MatrixView<'a, f32>) -> Self::Leaders<'a> {
        PartitionLeaders {
            values,
            norms: cosine_norms(values),
        }
    }

    fn leader_count(leaders: &Self::Leaders<'_>) -> usize {
        leaders.values.nrows()
    }

    fn compute_distances(
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        mut storage: MutMatrixView<'_, f32>,
    ) -> ANNResult<()> {
        if storage.nrows() != points.nrows() || storage.ncols() != leaders.values.nrows() {
            return Err(ANNError::message("point-to-leader output shape mismatch"));
        }
        diskann_linalg::sgemm(
            Transpose::None,
            Transpose::Ordinary,
            points.nrows(),
            leaders.values.nrows(),
            points.ncols(),
            1.0,
            points.as_slice(),
            leaders.values.as_slice(),
            None,
            storage.as_mut_slice(),
        )
        .map_err(ANNError::new)?;
        let point_norms = cosine_norms(points);
        let leader_norms = &leaders.norms;
        // Convert each dot to cosine distance. Reuse leader norms across stripes.
        for (row, &point_norm) in storage.row_iter_mut().zip(point_norms.iter()) {
            for (distance, &leader_norm) in row.iter_mut().zip(leader_norms.iter()) {
                *distance = cosine_distance(*distance, point_norm, leader_norm);
            }
        }
        Ok(())
    }
}

impl PartitionMetric for InnerProduct {
    type Leaders<'a> = PartitionLeaders<'a, ()>;

    fn create_leaders<'a>(values: MatrixView<'a, f32>) -> Self::Leaders<'a> {
        PartitionLeaders { values, norms: () }
    }

    fn leader_count(leaders: &Self::Leaders<'_>) -> usize {
        leaders.values.nrows()
    }

    fn compute_distances(
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        mut storage: MutMatrixView<'_, f32>,
    ) -> ANNResult<()> {
        if storage.nrows() != points.nrows() || storage.ncols() != leaders.values.nrows() {
            return Err(ANNError::message("point-to-leader output shape mismatch"));
        }
        diskann_linalg::sgemm(
            Transpose::None,
            Transpose::Ordinary,
            points.nrows(),
            leaders.values.nrows(),
            points.ncols(),
            -1.0,
            points.as_slice(),
            leaders.values.as_slice(),
            None,
            storage.as_mut_slice(),
        )
        .map_err(ANNError::new)?;
        Ok(())
    }
}

impl PartitionMetric for CosineNormalized {
    type Leaders<'a> = <InnerProduct as PartitionMetric>::Leaders<'a>;

    fn create_leaders<'a>(values: MatrixView<'a, f32>) -> Self::Leaders<'a> {
        InnerProduct::create_leaders(values)
    }

    fn leader_count(leaders: &Self::Leaders<'_>) -> usize {
        InnerProduct::leader_count(leaders)
    }

    fn compute_distances(
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        storage: MutMatrixView<'_, f32>,
    ) -> ANNResult<()> {
        // The constant in `1 - dot` does not change nearest-first order.
        InnerProduct::compute_distances(points, leaders, storage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(miri))]
    use crate::graph::pipnn::test_support;
    #[cfg(not(miri))]
    use diskann_vector::distance::Metric;
    use rstest::rstest;

    #[cfg(not(miri))]
    #[test]
    fn l2_ranking_retains_small_coordinate_contributions() {
        // Given: the origin ranks leaders solely by their squared norms.
        // 4096^2 + 127 is larger than 4096^2 + 16, regardless of coordinate order.
        let points = [0.0; 129];
        let mut values = vec![1.0; 2 * 129];
        values[0] = 4096.0;
        values[128] = 0.0;
        values[129..].fill(0.0);
        values[129] = 4096.0;
        values[130] = 4.0;
        let leaders = L2::create_leaders(MatrixView::try_from(values.as_slice(), 2, 129).unwrap());
        let mut output = [f32::NAN; 2];

        L2::compute_distances(
            MatrixView::try_from(&points[..], 1, 129).unwrap(),
            &leaders,
            MutMatrixView::try_from(&mut output[..], 1, 2).unwrap(),
        )
        .unwrap();

        assert!(
            output[1] < output[0],
            "nearer leader was ranked behind farther leader: {output:?}"
        );
        assert!(
            (output[0] - 16_777_344.0).abs() <= 16.0,
            "lost small squared coordinates: {output:?}"
        );
        assert_eq!(output[1], 16_777_232.0);
    }

    #[cfg(not(miri))]
    #[rstest]
    #[case::l2(L2, Metric::L2)]
    #[case::cosine(Cosine, Metric::Cosine)]
    #[case::normalized_cosine(CosineNormalized, Metric::CosineNormalized)]
    #[case::inner_product(InnerProduct, Metric::InnerProduct)]
    fn point_to_leader_scores_match_scalar_distances<M: PartitionMetric>(
        #[case] _metric: M,
        #[case] scalar_metric: Metric,
    ) {
        for point_count in [1, 3] {
            for leader_count in [1, 4, 17] {
                for dimensions in [1, 2, 7, 8, 9, 15, 16, 17, 127, 128, 129] {
                    // Small integer coordinates make unnormalized dot products exact. Points
                    // and leaders use different values, so swapped rows or columns change scores.
                    let mut point_values: Vec<_> = (0..point_count * dimensions)
                        .map(|index| (index % 7) as f32 - 3.0)
                        .collect();
                    let mut leader_values: Vec<_> = (0..leader_count * dimensions)
                        .map(|index| (index % 11) as f32 - 5.0)
                        .collect();
                    for (point, row) in point_values.chunks_exact_mut(dimensions).enumerate() {
                        row[0] = point as f32 + 1.0;
                    }
                    for (leader, row) in leader_values.chunks_exact_mut(dimensions).enumerate() {
                        row[0] = leader as f32 + 2.0;
                    }
                    if scalar_metric == Metric::CosineNormalized {
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
                    let mut output = vec![f32::NAN; point_count * leader_count];

                    M::compute_distances(
                        points,
                        &leaders,
                        MutMatrixView::try_from(output.as_mut_slice(), point_count, leader_count)
                            .unwrap(),
                    )
                    .unwrap_or_else(|error| {
                        panic!("point_count={point_count}, leader_count={leader_count}, dimensions={dimensions}: {error}")
                    });

                    assert_eq!(
                        M::leader_count(&leaders),
                        leader_count,
                        "point_count={point_count}, leader_count={leader_count}, dimensions={dimensions}"
                    );
                    for point in 0..point_count {
                        for leader in 0..leader_count {
                            let mut expected = test_support::distance(
                                scalar_metric,
                                points.row(point),
                                leader_matrix.row(leader),
                            );
                            if scalar_metric == Metric::L2 {
                                // Partition L2 omits exactly this constant from every column of the row.
                                expected -= points
                                    .row(point)
                                    .iter()
                                    .map(|&x| f64::from(x).powi(2))
                                    .sum::<f64>();
                            }
                            let tolerance = match scalar_metric {
                                Metric::L2 | Metric::InnerProduct => 0.0,
                                // Bound f32 norm/dot reductions at unit scale, as in the leaf metric.
                                Metric::Cosine | Metric::CosineNormalized => {
                                    8.0 * f64::from(f32::EPSILON) * dimensions as f64
                                }
                            };
                            let actual = f64::from(output[point * leader_count + leader]);
                            assert!(
                                (actual - expected).abs() <= tolerance,
                                "point_count={point_count}, leader_count={leader_count}, dimensions={dimensions}, point={point}, leader={leader}: {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(not(miri))]
    #[rstest]
    #[case::l2(L2, Metric::L2)]
    #[case::cosine(Cosine, Metric::Cosine)]
    #[case::normalized_cosine(CosineNormalized, Metric::CosineNormalized)]
    #[case::inner_product(InnerProduct, Metric::InnerProduct)]
    fn large_dense_inputs_match_scalar_distances<M: PartitionMetric>(
        #[case] _metric: M,
        #[case] scalar_metric: Metric,
    ) {
        for shape in [
            (17, 33, 384),
            (33, 65, 768),
            (65, 129, 1536),
            (9, 17, 1537),
            (257, 513, 129),
            (5, 17, 4097),
        ] {
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
                MatrixView::try_from(leader_values.as_slice(), leader_count, dimensions).unwrap();
            let leaders = M::create_leaders(leader_matrix);
            let mut output = vec![f32::NAN; point_count * leader_count];

            M::compute_distances(
                points,
                &leaders,
                MutMatrixView::try_from(output.as_mut_slice(), point_count, leader_count).unwrap(),
            )
            .unwrap_or_else(|error| panic!("shape={shape:?}: {error}"));

            let tolerance = match scalar_metric {
                Metric::L2 | Metric::InnerProduct => 0.0,
                // Dyadic dot/norm sums are exact; square roots and division round.
                Metric::Cosine => 16.0 * f64::from(f32::EPSILON),
                Metric::CosineNormalized => {
                    // Bound product and sum rounding for normalized f32 coordinates.
                    let roundoff = dimensions as f64 * f64::from(f32::EPSILON);
                    roundoff / (1.0 - roundoff)
                }
            };
            for point in 0..point_count {
                let point_norm: f64 = points
                    .row(point)
                    .iter()
                    .map(|&x| f64::from(x).powi(2))
                    .sum();
                for leader in 0..leader_count {
                    let mut expected = test_support::distance(
                        scalar_metric,
                        points.row(point),
                        leader_matrix.row(leader),
                    );
                    if scalar_metric == Metric::L2 {
                        expected -= point_norm;
                    }
                    let actual = f64::from(output[point * leader_count + leader]);
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "shape={shape:?}, point={point}, leader={leader}: {actual} != {expected}, tolerance={tolerance}"
                    );
                }
            }
        }
    }

    #[cfg(not(miri))]
    #[rstest]
    #[case::l2(L2, &[2.0, 0.0, 0.0, 3.0], &[1.0, 0.0, 0.0, -2.0, -3.0, 0.0], [-3.0, 4.0, 21.0, 1.0, 16.0, 9.0])]
    #[case::cosine(Cosine, &[2.0, 0.0, 0.0, 3.0], &[1.0, 0.0, 0.0, -2.0, -3.0, 0.0], [0.0, 1.0, 2.0, 1.0, 2.0, 1.0])]
    #[case::inner_product(InnerProduct, &[2.0, 0.0, 0.0, 3.0], &[1.0, 0.0, 0.0, -2.0, -3.0, 0.0], [-2.0, 0.0, 6.0, 0.0, 6.0, 0.0])]
    #[case::normalized_cosine(CosineNormalized, &[1.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0, -1.0, -1.0, 0.0], [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0])]
    fn leader_sets_can_be_reused_across_point_stripes<M: PartitionMetric>(
        #[case] _metric: M,
        #[case] point_values: &[f32],
        #[case] leader_values: &[f32],
        #[case] expected: [f32; 6],
    ) {
        let leaders = M::create_leaders(MatrixView::try_from(leader_values, 3, 2).unwrap());
        let mut output = [-100.0; 3];

        for (point, expected) in point_values.chunks_exact(2).zip(expected.chunks_exact(3)) {
            M::compute_distances(
                MatrixView::try_from(point, 1, 2).unwrap(),
                &leaders,
                MutMatrixView::try_from(&mut output[..], 1, 3).unwrap(),
            )
            .unwrap();

            assert_eq!(output, expected);
        }
    }

    #[cfg(not(miri))]
    #[test]
    fn cosine_assigns_unit_distance_when_either_vector_has_zero_norm() {
        let point_values = [0.0, 0.0, 0.0, 2.0];
        let leader_values = [3.0, 0.0, 0.0, 0.0, 0.0, -4.0];
        let leaders =
            Cosine::create_leaders(MatrixView::try_from(&leader_values[..], 3, 2).unwrap());
        let mut output = [42.0; 6];

        Cosine::compute_distances(
            MatrixView::try_from(&point_values[..], 2, 2).unwrap(),
            &leaders,
            MutMatrixView::try_from(&mut output[..], 2, 3).unwrap(),
        )
        .unwrap();

        assert_eq!(output, [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]);
    }

    #[rstest]
    #[case::l2(L2)]
    #[case::cosine(Cosine)]
    #[case::normalized_cosine(CosineNormalized)]
    #[case::inner_product(InnerProduct)]
    fn output_shape_mismatch_is_rejected_before_writing<M: PartitionMetric>(
        #[case] _metric: M,
        #[values((1, 3), (2, 2), (3, 3), (2, 4))] shape: (usize, usize),
    ) {
        let point_values = [1.0, 2.0, 3.0, 4.0];
        let leader_values = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let leaders = M::create_leaders(MatrixView::try_from(&leader_values[..], 3, 2).unwrap());
        let mut output = vec![17.0; shape.0 * shape.1];

        let error = M::compute_distances(
            MatrixView::try_from(&point_values[..], 2, 2).unwrap(),
            &leaders,
            MutMatrixView::try_from(output.as_mut_slice(), shape.0, shape.1).unwrap(),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("point-to-leader output shape mismatch")
        );
        assert_eq!(output, vec![17.0; shape.0 * shape.1]);
    }

    #[rstest]
    #[case::l2(L2)]
    #[case::cosine(Cosine)]
    #[case::normalized_cosine(CosineNormalized)]
    #[case::inner_product(InnerProduct)]
    fn mismatched_vector_dimensions_report_the_leader_matrix_error<M: PartitionMetric>(
        #[case] _metric: M,
    ) {
        let point_values = [1.0, 2.0, 3.0];
        let leader_values = [1.0, 2.0, 3.0, 4.0];
        let leaders = M::create_leaders(MatrixView::try_from(&leader_values[..], 2, 2).unwrap());
        let mut output = [17.0; 2];

        let error = M::compute_distances(
            MatrixView::try_from(&point_values[..], 1, 3).unwrap(),
            &leaders,
            MutMatrixView::try_from(&mut output[..], 1, 2).unwrap(),
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
    }
}
