/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Build ranking distances for one PiPNN leaf.
//!
//! A ranking distance preserves nearest-first order. It does not need to equal
//! the mathematical metric distance. The leaf kernel reads only the lower triangle.

use crate::{ANNError, ANNResult};
use diskann_utils::views::MatrixView;

use super::{Cosine, CosineNormalized, InnerProduct, L2, cosine_distance};

/// Fill one flattened lower-triangular ranking buffer.
///
/// An implementation initializes the diagonal and lower triangle. The upper
/// triangle stays unspecified. The input matrix has one point in each row.
/// A zero distance can have either sign. Equal distances can select either candidate.
pub(super) trait LeafMetric: Send + Sync + 'static {
    /// Compute ranking distances for all unordered point pairs.
    ///
    /// Values preserve nearest-first order, but need not equal metric distances.
    /// L2 returns squared distances. Normalized cosine and inner product return
    /// `-dot`, omitting the constant in normalized cosine's `1 - dot`.
    /// Cosine returns `1 - similarity`.
    ///
    /// `storage` has `points.nrows() * points.nrows()` elements.
    fn compute_distances(points: MatrixView<'_, f32>, storage: &mut [f32]) -> ANNResult<()>;
}

impl LeafMetric for L2 {
    fn compute_distances(points: MatrixView<'_, f32>, storage: &mut [f32]) -> ANNResult<()> {
        let point_count = points.nrows();
        // The expanded L2 formula is `||x||² + ||y||² - 2(x·y)`.
        let squared_norms: Vec<f32> = points
            .row_iter()
            .map(|point| point.iter().map(|value| value * value).sum())
            .collect();
        // Initialize the first two terms before GEMM adds the dot-product term.
        for source in 0..point_count {
            let row = &mut storage[source * point_count..source * point_count + source + 1];
            let source_norm = squared_norms[source];
            let mut target = 0;
            while target < row.len() {
                row[target] = source_norm + squared_norms[target];
                target += 1;
            }
        }
        diskann_linalg::sgemm_aat_lower_add(
            point_count,
            points.ncols(),
            -2.0,
            points.as_slice(),
            storage,
        )
        .map_err(ANNError::new)?;
        Ok(())
    }
}

impl LeafMetric for Cosine {
    fn compute_distances(points: MatrixView<'_, f32>, storage: &mut [f32]) -> ANNResult<()> {
        let point_count = points.nrows();
        // The diagonal supplies each point norm after GEMM computes all dots.
        diskann_linalg::sgemm_aat_lower(
            point_count,
            points.ncols(),
            1.0,
            points.as_slice(),
            storage,
        )
        .map_err(ANNError::new)?;
        let norms: Vec<f32> = (0..point_count)
            .map(|point| storage[point * point_count + point].sqrt())
            .collect();
        // Convert each lower-triangle dot to the bounded cosine distance.
        for source in 0..point_count {
            for target in 0..=source {
                let index = source * point_count + target;
                storage[index] = cosine_distance(storage[index], norms[source], norms[target]);
            }
        }
        Ok(())
    }
}

impl LeafMetric for CosineNormalized {
    fn compute_distances(points: MatrixView<'_, f32>, storage: &mut [f32]) -> ANNResult<()> {
        let point_count = points.nrows();
        // The constant in `1 - dot` does not change nearest-first order.
        diskann_linalg::sgemm_aat_lower(
            point_count,
            points.ncols(),
            -1.0,
            points.as_slice(),
            storage,
        )
        .map_err(ANNError::new)?;
        Ok(())
    }
}

impl LeafMetric for InnerProduct {
    fn compute_distances(points: MatrixView<'_, f32>, storage: &mut [f32]) -> ANNResult<()> {
        // Both metrics rank with `-dot`. Their graph-pruning policies stay separate.
        CosineNormalized::compute_distances(points, storage)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test matrices have fixed valid shapes")]
mod tests {
    use super::*;
    use rstest::rstest;

    const POINT_COUNT: usize = 2;
    const DIMENSION_COUNT: usize = 2;
    const FIRST_POINT: usize = 0;
    const SECOND_POINT: usize = 1;
    const STALE_DISTANCE: f32 = 99.0;
    const FLOAT_TOLERANCE: f32 = 1.0e-6;

    fn compute_pair_ranking<M: LeafMetric>(
        first_point: [f32; DIMENSION_COUNT],
        second_point: [f32; DIMENSION_COUNT],
    ) -> f32 {
        let point_values = [
            first_point[0],
            first_point[1],
            second_point[0],
            second_point[1],
        ];
        let points = MatrixView::try_from(&point_values[..], POINT_COUNT, DIMENSION_COUNT).unwrap();
        let mut storage = [STALE_DISTANCE; POINT_COUNT * POINT_COUNT];

        M::compute_distances(points, &mut storage).unwrap();

        storage[SECOND_POINT * POINT_COUNT + FIRST_POINT]
    }

    mod compute_distances_tests {
        use super::*;

        #[test]
        fn squared_l2_ranking_equals_the_sum_of_squared_coordinate_differences() {
            // Given
            let first_point = [3.0_f32, 4.0];
            let second_point = [0.0_f32, 4.0];
            let x_difference = first_point[0] - second_point[0];
            let y_difference = first_point[1] - second_point[1];
            let expected = x_difference.mul_add(x_difference, y_difference * y_difference);

            // When
            let actual = compute_pair_ranking::<L2>(first_point, second_point);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn cosine_ranking_equals_one_minus_normalized_similarity() {
            // Given
            let first_point = [2.0_f32, 0.0];
            let second_point = [1.0_f32, 1.0];
            let dot = first_point[0].mul_add(second_point[0], first_point[1] * second_point[1]);
            let first_norm = first_point[0].hypot(first_point[1]);
            let second_norm = second_point[0].hypot(second_point[1]);
            let expected = 1.0 - dot / (first_norm * second_norm);

            // When
            let actual = compute_pair_ranking::<Cosine>(first_point, second_point);

            // Then
            assert!(
                (actual - expected).abs() <= FLOAT_TOLERANCE,
                "actual {actual} differs from expected {expected}"
            );
        }

        #[rstest]
        #[case::zero([0.0, 0.0])]
        #[case::subnormal([f32::MIN_POSITIVE.sqrt() / 2.0, 0.0])]
        fn small_norm_produces_unit_cosine_ranking(#[case] small_point: [f32; DIMENSION_COUNT]) {
            // Given: a zero or subnormal norm represents zero similarity.
            let unit_point = [1.0_f32, 0.0];
            let expected = 1.0;

            // When
            let actual = compute_pair_ranking::<Cosine>(small_point, unit_point);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn normalized_cosine_ranking_equals_the_negative_dot_product() {
            // Given
            let first_point = [1.0_f32, 0.0];
            let second_point = [0.6_f32, 0.8];
            let dot = first_point[0].mul_add(second_point[0], first_point[1] * second_point[1]);
            let expected = -dot;

            // When
            let actual = compute_pair_ranking::<CosineNormalized>(first_point, second_point);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn inner_product_ranking_equals_the_negative_dot_product() {
            // Given
            let first_point = [2.0_f32, -1.0];
            let second_point = [3.0_f32, 4.0];
            let dot = first_point[0].mul_add(second_point[0], first_point[1] * second_point[1]);
            let expected = -dot;

            // When
            let actual = compute_pair_ranking::<InnerProduct>(first_point, second_point);

            // Then
            assert_eq!(actual, expected);
        }

        #[rstest]
        #[case::l2(compute_pair_ranking::<L2>)]
        #[case::cosine(compute_pair_ranking::<Cosine>)]
        #[case::normalized_cosine(compute_pair_ranking::<CosineNormalized>)]
        #[case::inner_product(compute_pair_ranking::<InnerProduct>)]
        fn nan_coordinate_produces_nan_ranking(
            #[case] compute: fn([f32; DIMENSION_COUNT], [f32; DIMENSION_COUNT]) -> f32,
        ) {
            // Given: neither vector has zero norm.
            let first_point = [f32::NAN, 1.0];
            let second_point = [1.0, 0.0];

            // When
            let actual = compute(first_point, second_point);

            // Then
            assert!(actual.is_nan());
        }

        #[rstest]
        #[case::inner_product(compute_pair_ranking::<InnerProduct>)]
        #[case::normalized_cosine(compute_pair_ranking::<CosineNormalized>)]
        fn orthogonal_vectors_have_zero_ranking(
            #[case] compute: fn([f32; DIMENSION_COUNT], [f32; DIMENSION_COUNT]) -> f32,
        ) {
            // The contract permits either zero sign at both metric entry points.
            assert_eq!(compute([1.0, 0.0], [0.0, 1.0]), 0.0);
        }

        #[test]
        fn inner_product_uses_the_last_coordinate_beyond_a_complete_dimension_block() {
            let dimensions = 129;
            let mut values = vec![0.0; 2 * dimensions];
            values[dimensions - 1] = 2.0;
            values[dimensions] = -1.0;
            values[2 * dimensions - 1] = 3.0;
            let points = MatrixView::try_from(values.as_slice(), 2, dimensions).unwrap();
            let mut distances = [STALE_DISTANCE; 4];

            InnerProduct::compute_distances(points, &mut distances).unwrap();

            // Dot products are 4, 6 and 10; all nonzero pair contribution is in the tail.
            assert_eq!(
                [distances[0], distances[2], distances[3]],
                [-4.0, -6.0, -10.0]
            );
        }
    }
}
