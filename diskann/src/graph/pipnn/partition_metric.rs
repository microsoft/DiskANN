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
use diskann_vector::{Norm, norm::FastL2NormSquared};

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

/// Compute L2 squared norms with sequential accumulation.
///
/// This order fixes rounding of the leader term before GEMM adds the dot term.
fn l2_squared_norms(vectors: MatrixView<'_, f32>) -> Vec<f32> {
    vectors
        .row_iter()
        .map(|vector| vector.iter().map(|value| value * value).sum())
        .collect()
}

/// Compute cosine norms with the same reduction for points and leaders.
fn cosine_norms(vectors: MatrixView<'_, f32>) -> Vec<f32> {
    vectors
        .row_iter()
        .map(|vector| FastL2NormSquared.evaluate(vector).sqrt())
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

    mod create_leaders_tests {
        use super::*;

        #[test]
        fn l2_norms_follow_sequential_sum_order() {
            // Given: each small square is half an ULP at the first squared norm.
            // Each sequential addition rounds back to that norm.
            let mut values = [1.0_f32; 129];
            values[0] = 4096.0;
            let expected_squared_norm = 16_777_216.0_f32;
            let matrix = MatrixView::try_from(&values[..], 1, values.len()).unwrap();

            // When
            let leaders = L2::create_leaders(matrix);

            // Then
            assert_eq!(leaders.norms, [expected_squared_norm]);
        }
    }

    // GEMM-backed tests run natively, not under Miri.
    #[cfg(not(miri))]
    mod compute_distances_tests {
        use super::*;
        use crate::graph::pipnn::scalar_ranking;
        use diskann_vector::distance::Metric;
        use rstest::rstest;

        const DIMENSION_COUNT: usize = 2;
        const STALE_DISTANCE: f32 = 99.0;
        const FLOAT_TOLERANCE: f32 = 1.0e-6;

        fn matrix(values: &[f32], rows: usize) -> MatrixView<'_, f32> {
            MatrixView::try_from(values, rows, DIMENSION_COUNT).unwrap()
        }

        fn compute_one_ranking<M: PartitionMetric>(
            point: [f32; DIMENSION_COUNT],
            leader: [f32; DIMENSION_COUNT],
        ) -> f32 {
            let leaders = M::create_leaders(matrix(&leader, 1));
            let mut storage = [STALE_DISTANCE];

            M::compute_distances(
                matrix(&point, 1),
                &leaders,
                MutMatrixView::row_vector(&mut storage[..]),
            )
            .unwrap();

            storage[0]
        }

        #[rstest]
        #[case::l2(L2, Metric::L2)]
        #[case::cosine(Cosine, Metric::Cosine)]
        #[case::normalized_cosine(CosineNormalized, Metric::CosineNormalized)]
        #[case::inner_product(InnerProduct, Metric::InnerProduct)]
        fn point_leader_matrix_matches_scalar_ranking<M: PartitionMetric>(
            #[case] _metric: M,
            #[case] scalar_metric: Metric,
            #[values(32, 128)] point_count: usize,
            #[values(64, 256)] leader_count: usize,
            #[values(128, 384, 768)] dimensions: usize,
        ) {
            // Given: points and leaders sample the same bounded arc on different grids.
            // NaN marks unwritten entries in every point-to-leader row.
            let normalize = scalar_metric == Metric::CosineNormalized;
            let point_values = scalar_ranking::arc_vectors(
                (0..point_count).map(|point| 32.0 * (point as f64 + 0.5) / point_count as f64),
                dimensions,
                normalize,
            );
            let leader_values = scalar_ranking::arc_vectors(
                (0..leader_count).map(|leader| 32.0 * leader as f64 / leader_count as f64),
                dimensions,
                normalize,
            );
            let points =
                MatrixView::try_from(point_values.as_slice(), point_count, dimensions).unwrap();
            let leader_vectors =
                MatrixView::try_from(leader_values.as_slice(), leader_count, dimensions).unwrap();
            let expected: Vec<Vec<_>> = points
                .row_iter()
                .map(|point| {
                    // Derive the omitted L2 term from input coordinates, not cached norms.
                    let point_term = if scalar_metric == Metric::L2 {
                        point.iter().map(|&value| f64::from(value).powi(2)).sum()
                    } else {
                        0.0
                    };
                    leader_vectors
                        .row_iter()
                        .map(|leader| {
                            (
                                scalar_ranking::distance(scalar_metric, point, leader) - point_term,
                                scalar_ranking::distance_tolerance(scalar_metric, point, leader),
                            )
                        })
                        .collect()
                })
                .collect();
            let mut distances = vec![f32::NAN; point_count * leader_count];

            // When: prepare leaders and call the metric without partition selection.
            let leaders = M::create_leaders(leader_vectors);
            M::compute_distances(
                points,
                &leaders,
                MutMatrixView::try_from(distances.as_mut_slice(), point_count, leader_count)
                    .unwrap(),
            )
            .unwrap();

            // Then: check every leader column for every point, not only selected leaders.
            let actual =
                MatrixView::try_from(distances.as_slice(), point_count, leader_count).unwrap();
            for (point, row) in expected.iter().enumerate() {
                for (leader, &(expected, tolerance)) in row.iter().enumerate() {
                    let actual = f64::from(actual.row(point)[leader]);
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{scalar_metric:?}, {point_count} points, {leader_count} leaders, D={dimensions}, pair ({point}, {leader}): actual {actual}, expected {expected}, tolerance {tolerance}"
                    );
                }
            }
        }

        #[rstest::rstest]
        #[case::wrong_rows_with_equal_area(1, 4)]
        #[case::wrong_columns(2, 1)]
        fn output_shape_mismatch_returns_error<M: PartitionMetric>(
            #[values(L2, Cosine, InnerProduct)] _metric: M,
            #[case] rows: usize,
            #[case] columns: usize,
        ) {
            // Given: two points and two leaders require a 2-by-2 output.
            let leader_values = [1.0, 0.0, 0.0, 2.0];
            let leaders = M::create_leaders(matrix(&leader_values, 2));
            let points = matrix(&[0.0, 4.0, -2.0, 0.0], 2);
            let mut storage = vec![STALE_DISTANCE; rows * columns];

            // When
            let result = M::compute_distances(
                points,
                &leaders,
                MutMatrixView::try_from(storage.as_mut_slice(), rows, columns).unwrap(),
            );

            // Then: invalid output shapes stop the build.
            assert!(result.is_err());
        }

        #[test]
        fn l2_omits_squared_point_norm() {
            // Given: leader norms are 25 and 4; each entry is norm - 2 * dot.
            let leader_values = [3.0_f32, 4.0, 0.0, 2.0];
            let leaders = L2::create_leaders(matrix(&leader_values, 2));
            let points = matrix(&[2.0, 1.0, -1.0, 2.0], 2);
            let expected = [5.0, 0.0, 15.0, -4.0];
            let mut storage = [STALE_DISTANCE; 4];

            // When
            L2::compute_distances(
                points,
                &leaders,
                MutMatrixView::try_from(&mut storage[..], 2, 2).unwrap(),
            )
            .unwrap();

            // Then: every row replaces stale distances before GEMM accumulates.
            assert_eq!(storage, expected);
        }

        #[rstest::rstest]
        #[case::zero_point([0.0, 0.0], [1.0, 0.0])]
        #[case::zero_leader([1.0, 0.0], [0.0, 0.0])]
        #[case::subnormal_point([f32::MIN_POSITIVE.sqrt() / 2.0, 0.0], [1.0, 0.0])]
        #[case::subnormal_leader([1.0, 0.0], [f32::MIN_POSITIVE.sqrt() / 2.0, 0.0])]
        fn cosine_writes_one_for_small_norms(
            #[case] point: [f32; DIMENSION_COUNT],
            #[case] leader: [f32; DIMENSION_COUNT],
        ) {
            // Given: a zero or subnormal norm represents zero similarity.
            let expected = 1.0;

            // When
            let actual = compute_one_ranking::<Cosine>(point, leader);

            // Then
            assert_eq!(actual, expected);
        }

        #[rstest::rstest]
        #[case::l2(compute_one_ranking::<L2>)]
        #[case::cosine(compute_one_ranking::<Cosine>)]
        #[case::inner_product(compute_one_ranking::<InnerProduct>)]
        fn nan_coordinate_produces_nan_ranking(
            #[case] compute: fn([f32; DIMENSION_COUNT], [f32; DIMENSION_COUNT]) -> f32,
        ) {
            // Given: neither vector has zero norm.
            let point = [f32::NAN, 1.0];
            let leader = [1.0, 0.0];

            // When
            let actual = compute(point, leader);

            // Then
            assert!(actual.is_nan());
        }

        #[test]
        fn normalized_cosine_writes_negative_dot_products() {
            // Given
            let point = [1.0_f32, 0.0];
            let leader = [0.6_f32, 0.8];
            let dot = point[0].mul_add(leader[0], point[1] * leader[1]);
            let expected = -dot;

            // When
            let actual = compute_one_ranking::<CosineNormalized>(point, leader);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn inner_product_writes_negative_dot_products() {
            // Given
            let point = [2.0_f32, -1.0];
            let leader = [3.0_f32, 4.0];
            let dot = point[0].mul_add(leader[0], point[1] * leader[1]);
            let expected = -dot;

            // When
            let actual = compute_one_ranking::<InnerProduct>(point, leader);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn cosine_replaces_all_distance_rows() {
            // Given: non-unit points point right, up, and left. Leaders point
            // diagonally up-right and up, so similarities include ±1/sqrt(2).
            let leader_values = [1.0_f32, 1.0, 0.0, 2.0];
            let leaders = Cosine::create_leaders(matrix(&leader_values, 2));
            let points = matrix(&[2.0, 0.0, 0.0, 4.0, -2.0, 0.0], 3);
            let diagonal_similarity = std::f32::consts::FRAC_1_SQRT_2;
            let expected = [
                1.0 - diagonal_similarity,
                1.0,
                1.0 - diagonal_similarity,
                0.0,
                1.0 + diagonal_similarity,
                1.0,
            ];
            let mut storage = [STALE_DISTANCE; 6];

            // When
            Cosine::compute_distances(
                points,
                &leaders,
                MutMatrixView::try_from(&mut storage[..], 3, 2).unwrap(),
            )
            .unwrap();

            // Then
            for (actual, expected) in storage.into_iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= FLOAT_TOLERANCE,
                    "actual {actual} differs from expected {expected}"
                );
            }
        }
    }
}
