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
use diskann_wide::SIMDVector;

use super::{Cosine, CosineNormalized, InnerProduct, L2, cosine_distance, simd::PiPNNSIMDSchema};

/// Fill one flattened lower-triangular ranking buffer.
///
/// An implementation initializes the diagonal and lower triangle. The upper
/// triangle stays unspecified. The input matrix has one point in each row.
pub(super) trait LeafMetric: Send + Sync + 'static {
    /// Compute ranking distances for all unordered point pairs.
    ///
    /// `storage` has `points.nrows() * points.nrows()` elements.
    fn compute_distances<A>(
        arch: A,
        points: MatrixView<'_, f32>,
        storage: &mut [f32],
    ) -> ANNResult<()>
    where
        A: PiPNNSIMDSchema;
}

/// Fill each lower-triangle entry with the two endpoint squared norms.
///
/// Each SIMD operation copies one full group of norm sums. The scalar loop
/// copies the remaining entries. The GEMM operation adds the dot-product term.
fn fill_lower_norm_sums<A>(arch: A, storage: &mut [f32], points: usize, norms: &[f32])
where
    A: PiPNNSIMDSchema,
{
    for source in 0..points {
        let row = &mut storage[source * points..source * points + source + 1];
        let simd_end = row.len() - row.len() % A::Vector::LANES;
        let source_norm = A::Vector::splat(arch, norms[source]);
        for target in (0..simd_end).step_by(A::Vector::LANES) {
            // SAFETY: both offsets start complete SIMD groups in live slices.
            unsafe {
                let target_norms = A::Vector::load_simd(arch, norms.as_ptr().add(target));
                (source_norm + target_norms).store_simd(row.as_mut_ptr().add(target));
            }
        }
        for target in simd_end..row.len() {
            row[target] = norms[source] + norms[target];
        }
    }
}

impl LeafMetric for L2 {
    fn compute_distances<A>(
        arch: A,
        points: MatrixView<'_, f32>,
        storage: &mut [f32],
    ) -> ANNResult<()>
    where
        A: PiPNNSIMDSchema,
    {
        let point_count = points.nrows();
        // The expanded L2 formula is `||x||² + ||y||² - 2(x·y)`.
        let squared_norms: Vec<f32> = points
            .row_iter()
            .map(|point| point.iter().map(|value| value * value).sum())
            .collect();
        arch.run(|| fill_lower_norm_sums(arch, storage, point_count, &squared_norms));
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
    fn compute_distances<A>(
        _arch: A,
        points: MatrixView<'_, f32>,
        storage: &mut [f32],
    ) -> ANNResult<()>
    where
        A: PiPNNSIMDSchema,
    {
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
    fn compute_distances<A>(
        _arch: A,
        points: MatrixView<'_, f32>,
        storage: &mut [f32],
    ) -> ANNResult<()>
    where
        A: PiPNNSIMDSchema,
    {
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
    fn compute_distances<A>(
        _arch: A,
        points: MatrixView<'_, f32>,
        storage: &mut [f32],
    ) -> ANNResult<()>
    where
        A: PiPNNSIMDSchema,
    {
        let point_count = points.nrows();
        // DiskANN ranks inner products in descending order through `-dot`.
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

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test matrices have fixed valid shapes")]
mod tests {
    use super::*;
    use diskann_wide::ARCH;
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

        M::compute_distances(ARCH, points, &mut storage).unwrap();

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

        #[test]
        fn cosine_ranking_is_one_when_one_point_has_zero_norm() {
            // Given
            let zero_point = [0.0_f32, 0.0];
            let unit_point = [1.0_f32, 0.0];
            let zero_similarity = 0.0_f32;
            let expected = 1.0 - zero_similarity;

            // When
            let actual = compute_pair_ranking::<Cosine>(zero_point, unit_point);

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
        #[case::positive_zero(0.0)]
        #[case::negative_zero(-0.0)]
        #[trace]
        fn inner_product_ranking_matches_negated_scalar_dot_bits(#[case] zero_coordinate: f32) {
            // Given
            let first_point = [zero_coordinate, 0.0];
            let second_point = [1.0_f32, 0.0];
            let dot = first_point[0].mul_add(second_point[0], first_point[1] * second_point[1]);
            let expected = -dot;

            // When
            let actual = compute_pair_ranking::<InnerProduct>(first_point, second_point);

            // Then
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}
