/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Pairwise distances inside one PiPNN leaf.
//!
//! Leaf distances equal the DiskANN metric distances, because later graph stages
//! compare and quantize them. The leaf kernel reads only the strict lower
//! triangle, which holds each pair below the diagonal.

use crate::{ANNError, ANNResult};
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::{
    Norm,
    norm::{FastL2Norm, FastL2NormSquared},
};

use super::{Cosine, CosineNormalized, InnerProduct, L2, cosine_distance};

/// Compute the distances between all point pairs of one leaf.
///
/// `points` has one point in each row. An implementation writes the diagonal and
/// the lower triangle of `storage`. The upper triangle can hold any value. A zero
/// distance can have either sign.
pub(super) trait LeafMetric: Send + Sync + 'static {
    /// Compute the metric distance for all unordered point pairs.
    ///
    /// L2 returns squared distances. Cosine and normalized cosine return
    /// `1 - similarity`. Inner product returns `-dot`.
    ///
    /// `storage` has one row and one column per point.
    fn compute_distances(
        points: MatrixView<'_, f32>,
        storage: MutMatrixView<'_, f32>,
    ) -> ANNResult<()>;
}

impl LeafMetric for L2 {
    fn compute_distances(
        points: MatrixView<'_, f32>,
        mut storage: MutMatrixView<'_, f32>,
    ) -> ANNResult<()> {
        // The expanded L2 formula is `||x||² + ||y||² - 2(x·y)`.
        let squared_norms: Vec<f32> = points
            .row_iter()
            .map(|point| FastL2NormSquared.evaluate(point))
            .collect();
        // Initialize the norm terms before GEMM adds the dot-product term.
        for (source, row) in storage.row_iter_mut().enumerate() {
            let source_norm = squared_norms[source];
            for (distance, &target_norm) in row[..=source].iter_mut().zip(&squared_norms) {
                *distance = source_norm + target_norm;
            }
        }
        diskann_linalg::sgemm_aat_lower_add(
            points.nrows(),
            points.ncols(),
            -2.0,
            points.as_slice(),
            storage.as_mut_slice(),
        )
        .map_err(ANNError::new)
    }
}

impl LeafMetric for Cosine {
    fn compute_distances(
        points: MatrixView<'_, f32>,
        mut storage: MutMatrixView<'_, f32>,
    ) -> ANNResult<()> {
        diskann_linalg::sgemm_aat_lower(
            points.nrows(),
            points.ncols(),
            1.0,
            points.as_slice(),
            storage.as_mut_slice(),
        )
        .map_err(ANNError::new)?;
        let norms: Vec<f32> = points
            .row_iter()
            .map(|point| FastL2Norm.evaluate(point))
            .collect();
        // Convert each lower-triangle dot to the bounded cosine distance.
        for (source, row) in storage.row_iter_mut().enumerate() {
            let source_norm = norms[source];
            for (distance, &target_norm) in row[..=source].iter_mut().zip(&norms) {
                *distance = cosine_distance(*distance, source_norm, target_norm);
            }
        }
        Ok(())
    }
}

impl LeafMetric for InnerProduct {
    fn compute_distances(
        points: MatrixView<'_, f32>,
        mut storage: MutMatrixView<'_, f32>,
    ) -> ANNResult<()> {
        diskann_linalg::sgemm_aat_lower(
            points.nrows(),
            points.ncols(),
            -1.0,
            points.as_slice(),
            storage.as_mut_slice(),
        )
        .map_err(ANNError::new)
    }
}

impl LeafMetric for CosineNormalized {
    fn compute_distances(
        points: MatrixView<'_, f32>,
        mut storage: MutMatrixView<'_, f32>,
    ) -> ANNResult<()> {
        InnerProduct::compute_distances(points, storage.as_mut_view())?;
        // Keep the constant of `1 - dot`. Near neighbors then have distances near
        // zero, where floating-point spacing is finest. Later stages quantize these
        // distances, so the constant keeps more near neighbors distinguishable.
        for (source, row) in storage.row_iter_mut().enumerate() {
            row[..=source]
                .iter_mut()
                .for_each(|distance| *distance += 1.0);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::test_support;
    use diskann_vector::distance::Metric;
    use rstest::rstest;

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

        L2::compute_distances(
            points,
            MutMatrixView::try_from(&mut output[..], 2, 2).unwrap(),
        )
        .unwrap();

        // Allow eight f32 ULPs at this scale, far less than the lost 128.
        let tolerance = 16.0;
        assert!(
            (output[2] - expected).abs() <= tolerance,
            "{} != {expected}",
            output[2]
        );
    }

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

                M::compute_distances(
                    points,
                    MutMatrixView::try_from(output.as_mut_slice(), point_count, point_count)
                        .unwrap(),
                )
                .unwrap_or_else(|error| {
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

            M::compute_distances(
                points,
                MutMatrixView::try_from(output.as_mut_slice(), point_count, point_count).unwrap(),
            )
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

    #[rstest]
    #[case::squared_l2(L2, &[2.0, 0.0, 0.0, 3.0, -4.0, 0.0], [0.0, 13.0, 0.0, 36.0, 25.0, 0.0])]
    #[case::negative_dot(InnerProduct, &[2.0, 0.0, 0.0, 3.0, -4.0, 0.0], [-4.0, 0.0, -9.0, 8.0, 0.0, -16.0])]
    #[case::cosine(Cosine, &[2.0, 0.0, 0.0, 3.0, -4.0, 0.0], [0.0, 1.0, 0.0, 2.0, 1.0, 0.0])]
    #[case::normalized_cosine(CosineNormalized, &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0], [0.0, 1.0, 0.0, 2.0, 1.0, 0.0])]
    fn distances_follow_the_metric_definition<M: LeafMetric>(
        #[case] _metric: M,
        #[case] values: &[f32],
        #[case] expected: [f32; 6],
    ) {
        let points = MatrixView::try_from(values, 3, 2).unwrap();
        let mut output = [42.0; 9];

        M::compute_distances(
            points,
            MutMatrixView::try_from(&mut output[..], 3, 3).unwrap(),
        )
        .unwrap();

        assert_eq!(
            [
                output[0], output[3], output[4], output[6], output[7], output[8]
            ],
            expected
        );
    }

    #[rstest]
    #[case::zero(0.0)]
    #[case::squared_norm_underflows(f32::MIN_POSITIVE)]
    fn cosine_gives_unit_distance_to_points_with_small_norms(#[case] coordinate: f32) {
        let values = [coordinate, 0.0, 0.0, 2.0];
        let points = MatrixView::try_from(&values[..], 2, 2).unwrap();
        let mut output = [42.0; 4];

        Cosine::compute_distances(
            points,
            MutMatrixView::try_from(&mut output[..], 2, 2).unwrap(),
        )
        .unwrap();

        assert_eq!([output[0], output[2], output[3]], [1.0, 1.0, 0.0]);
    }

    #[rstest]
    #[case::l2(L2, 2.0)]
    #[case::cosine(Cosine, 1.0)]
    #[case::normalized_cosine(CosineNormalized, 1.0)]
    #[case::inner_product(InnerProduct, 0.0)]
    fn a_nan_point_does_not_change_other_pair_distances<M: LeafMetric>(
        #[case] _metric: M,
        #[case] expected_finite_pair: f32,
    ) {
        let values = [1.0, 0.0, 0.0, -1.0, f32::NAN, f32::NAN];
        let mut output = [42.0; 9];

        M::compute_distances(
            MatrixView::try_from(&values[..], 3, 2).unwrap(),
            MutMatrixView::try_from(&mut output[..], 3, 3).unwrap(),
        )
        .unwrap();

        assert_eq!(output[3], expected_finite_pair);
        assert!(output[6..=8].iter().all(|distance| distance.is_nan()));
    }
}
