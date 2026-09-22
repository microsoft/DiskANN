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
use diskann_vector::{
    Norm,
    norm::{FastL2Norm, FastL2NormSquared},
};

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
            .map(|point| FastL2NormSquared.evaluate(point))
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
        diskann_linalg::sgemm_aat_lower(
            point_count,
            points.ncols(),
            1.0,
            points.as_slice(),
            storage,
        )
        .map_err(ANNError::new)?;
        let norms: Vec<f32> = points
            .row_iter()
            .map(|point| FastL2Norm.evaluate(point))
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

impl LeafMetric for InnerProduct {
    fn compute_distances(points: MatrixView<'_, f32>, storage: &mut [f32]) -> ANNResult<()> {
        diskann_linalg::sgemm_aat_lower(
            points.nrows(),
            points.ncols(),
            -1.0,
            points.as_slice(),
            storage,
        )
        .map_err(ANNError::new)?;
        Ok(())
    }
}

impl LeafMetric for CosineNormalized {
    fn compute_distances(points: MatrixView<'_, f32>, storage: &mut [f32]) -> ANNResult<()> {
        // The constant in `1 - dot` does not change nearest-first order.
        InnerProduct::compute_distances(points, storage)
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
    fn l2_distance_retains_small_coordinate_contributions() {
        // Given: 4096^2 + 127 unit coordinates, orthogonal to a unit vector.
        // Scalar left-to-right f32 summation loses every unit after 4096^2.
        let mut values = vec![1.0; 2 * 129];
        values[0] = 4096.0;
        values[128] = 0.0;
        values[129..].fill(0.0);
        values[257] = 1.0;
        let points = MatrixView::try_from(values.as_slice(), 2, 129).unwrap();
        let mut output = [f32::NAN; 4];
        let expected = 16_777_344.0; // 4096^2 + 128.

        L2::compute_distances(points, &mut output).unwrap();

        // Allow eight f32 ULPs at this scale, far less than the lost 128.
        let tolerance = 16.0;
        assert!(
            (output[2] - expected).abs() <= tolerance,
            "{} != {expected}",
            output[2]
        );
    }

    #[cfg(not(miri))]
    #[rstest]
    #[case::l2(L2, Metric::L2)]
    #[case::cosine(Cosine, Metric::Cosine)]
    #[case::normalized_cosine(CosineNormalized, Metric::CosineNormalized)]
    #[case::inner_product(InnerProduct, Metric::InnerProduct)]
    fn lower_triangle_matches_scalar_distances<M: LeafMetric>(
        #[case] _metric: M,
        #[case] scalar_metric: Metric,
    ) {
        for point_count in [1, 4, 17] {
            for dimensions in [1, 2, 7, 8, 9, 15, 16, 17, 127, 128, 129] {
                // Small integers keep L2 and dot products exact; the first coordinate gives
                // every vector a nonzero norm. Every dimension contributes to the result.
                let mut values: Vec<_> = (0..point_count * dimensions)
                    .map(|index| (index % 11) as f32 - 5.0)
                    .collect();
                for (point, row) in values.chunks_exact_mut(dimensions).enumerate() {
                    row[0] = point as f32 + 1.0;
                }
                if scalar_metric == Metric::CosineNormalized {
                    test_support::normalize(&mut values, dimensions);
                }
                let points =
                    MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
                let mut output = vec![f32::NAN; point_count * point_count];

                M::compute_distances(points, &mut output).unwrap_or_else(|error| {
                    panic!("point_count={point_count}, dimensions={dimensions}: {error}")
                });

                for source in 0..point_count {
                    for target in 0..=source {
                        let expected = test_support::distance(
                            scalar_metric,
                            points.row(source),
                            points.row(target),
                        );
                        let actual = f64::from(output[source * point_count + target]);
                        // Cosine reductions and normalization round in f32; allow eight ulps
                        // per dimension at unit scale. Integer L2 and dot products are exact.
                        let tolerance = match scalar_metric {
                            Metric::L2 | Metric::InnerProduct => 0.0,
                            Metric::Cosine | Metric::CosineNormalized => {
                                8.0 * f64::from(f32::EPSILON) * dimensions as f64
                            }
                        };
                        assert!(
                            (actual - expected).abs() <= tolerance,
                            "point_count={point_count}, dimensions={dimensions}, pair=({source},{target}): {actual} != {expected}"
                        );
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
    fn large_dense_inputs_match_scalar_distances<M: LeafMetric>(
        #[case] _metric: M,
        #[case] scalar_metric: Metric,
    ) {
        for shape in [
            (33, 384),
            (65, 768),
            (129, 1536),
            (17, 1537),
            (513, 129),
            (17, 4097),
        ] {
            let (point_count, dimensions) = shape;
            let mut values = test_support::dense_points(point_count, dimensions, 1287);
            if scalar_metric == Metric::CosineNormalized {
                test_support::normalize(&mut values, dimensions);
            }
            let points = MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
            let mut output = vec![f32::NAN; point_count * point_count];

            M::compute_distances(points, &mut output)
                .unwrap_or_else(|error| panic!("shape={shape:?}: {error}"));

            let tolerance = match scalar_metric {
                Metric::L2 | Metric::InnerProduct => 0.0,
                // Dyadic dot/norm sums are exact; only square roots and division round.
                Metric::Cosine => 16.0 * f64::from(f32::EPSILON),
                Metric::CosineNormalized => {
                    // Normalized f32 coordinates need a dot-product rounding bound.
                    // Sum |x*y| is at most approximately one; EPSILON allows both
                    // product and reduction rounding in gamma = n*u/(1-n*u).
                    let roundoff = dimensions as f64 * f64::from(f32::EPSILON);
                    roundoff / (1.0 - roundoff)
                }
            };
            for source in 0..point_count {
                for target in 0..=source {
                    let expected = test_support::distance(
                        scalar_metric,
                        points.row(source),
                        points.row(target),
                    );
                    let actual = f64::from(output[source * point_count + target]);
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "shape={shape:?}, pair=({source},{target}): {actual} != {expected}, tolerance={tolerance}"
                    );
                }
            }
        }
    }

    #[cfg(not(miri))]
    #[rstest]
    #[case::squared_l2(L2, &[2.0, 0.0, 0.0, 3.0, -4.0, 0.0], [0.0, 13.0, 0.0, 36.0, 25.0, 0.0])]
    #[case::negative_dot(InnerProduct, &[2.0, 0.0, 0.0, 3.0, -4.0, 0.0], [-4.0, 0.0, -9.0, 8.0, 0.0, -16.0])]
    #[case::cosine(Cosine, &[2.0, 0.0, 0.0, 3.0, -4.0, 0.0], [0.0, 1.0, 0.0, 2.0, 1.0, 0.0])]
    #[case::normalized_cosine(CosineNormalized, &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0], [-1.0, 0.0, -1.0, 1.0, 0.0, -1.0])]
    fn ranking_values_follow_the_metric_definition<M: LeafMetric>(
        #[case] _metric: M,
        #[case] values: &[f32],
        #[case] expected: [f32; 6],
    ) {
        let points = MatrixView::try_from(values, 3, 2).unwrap();
        let mut output = [42.0; 9];

        M::compute_distances(points, &mut output).unwrap();

        assert_eq!(
            [
                output[0], output[3], output[4], output[6], output[7], output[8]
            ],
            expected
        );
    }

    #[cfg(not(miri))]
    #[rstest]
    #[case::zero(0.0)]
    #[case::squared_norm_underflows(f32::MIN_POSITIVE)]
    fn cosine_gives_unit_distance_to_points_with_small_norms(#[case] coordinate: f32) {
        let values = [coordinate, 0.0, 0.0, 2.0];
        let points = MatrixView::try_from(&values[..], 2, 2).unwrap();
        let mut output = [42.0; 4];

        Cosine::compute_distances(points, &mut output).unwrap();

        assert_eq!([output[0], output[2], output[3]], [1.0, 1.0, 0.0]);
    }

    #[cfg(not(miri))]
    #[rstest]
    #[case::l2(L2, 2.0)]
    #[case::cosine(Cosine, 1.0)]
    #[case::normalized_cosine(CosineNormalized, 0.0)]
    #[case::inner_product(InnerProduct, 0.0)]
    fn a_nan_point_does_not_change_other_pair_distances<M: LeafMetric>(
        #[case] _metric: M,
        #[case] expected_finite_pair: f32,
    ) {
        let values = [1.0, 0.0, 0.0, -1.0, f32::NAN, f32::NAN];
        let mut output = [42.0; 9];

        M::compute_distances(
            MatrixView::try_from(&values[..], 3, 2).unwrap(),
            &mut output,
        )
        .unwrap();

        assert_eq!(output[3], expected_finite_pair);
        assert!(output[6..=8].iter().all(|distance| distance.is_nan()));
    }

    #[rstest]
    #[case::l2(L2)]
    #[case::cosine(Cosine)]
    #[case::normalized_cosine(CosineNormalized)]
    #[case::inner_product(InnerProduct)]
    fn storage_length_mismatch_reports_the_output_matrix_error<M: LeafMetric>(#[case] _metric: M) {
        let values = [1.0, 0.0, 0.0, 2.0];
        let mut output = [42.0; 5];

        let error = M::compute_distances(
            MatrixView::try_from(&values[..], 2, 2).unwrap(),
            &mut output,
        )
        .unwrap_err();

        assert_eq!(
            error.downcast_ref::<diskann_linalg::SgemmError>(),
            Some(&diskann_linalg::SgemmError::InvalidMatrixDimensions {
                matrix_name: diskann_linalg::MatrixName::C,
                expected_rows: 2,
                expected_cols: 2,
                actual_len: 5,
            })
        );
    }
}
