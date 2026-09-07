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
use diskann_utils::views::MatrixView;
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

/// Fill one flattened point-to-leader ranking buffer.
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
    /// `storage` has `points.nrows() * leader_count` elements.
    /// A zero distance can have either sign. Equal distances can select either leader.
    fn compute_distances(
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        storage: &mut [f32],
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
        storage: &mut [f32],
    ) -> ANNResult<()> {
        // The point norm is constant across a point row. It cannot change ranking.
        let leader_norms = &leaders.norms;
        // Initialize each point row before GEMM adds the dot-product term.
        let leader_count = leader_norms.len();
        for row in storage.chunks_exact_mut(leader_count) {
            let mut leader = 0;
            while leader < leader_count {
                row[leader] = leader_norms[leader];
                leader += 1;
            }
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
            storage,
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
        storage: &mut [f32],
    ) -> ANNResult<()> {
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
            storage,
        )
        .map_err(ANNError::new)?;
        let point_norms = cosine_norms(points);
        let leader_norms = &leaders.norms;
        let leader_count = leaders.values.nrows();
        // Convert each dot to cosine distance. Reuse leader norms across stripes.
        for (row, &point_norm) in storage
            .chunks_exact_mut(leader_count)
            .zip(point_norms.iter())
        {
            for (distance, &leader_norm) in row.iter_mut().zip(leader_norms.iter()) {
                *distance = cosine_distance(*distance, point_norm, leader_norm);
            }
        }
        Ok(())
    }
}

// Normalized cosine and inner product have the same ranking expression.
// Both metrics rank candidates with `-dot`.
impl PartitionMetric for CosineNormalized {
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
        storage: &mut [f32],
    ) -> ANNResult<()> {
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
            storage,
        )
        .map_err(ANNError::new)?;
        Ok(())
    }
}

impl PartitionMetric for InnerProduct {
    type Leaders<'a> = <CosineNormalized as PartitionMetric>::Leaders<'a>;

    fn create_leaders<'a>(values: MatrixView<'a, f32>) -> Self::Leaders<'a> {
        CosineNormalized::create_leaders(values)
    }

    fn leader_count(leaders: &Self::Leaders<'_>) -> usize {
        CosineNormalized::leader_count(leaders)
    }

    fn compute_distances(
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        storage: &mut [f32],
    ) -> ANNResult<()> {
        CosineNormalized::compute_distances(points, leaders, storage)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test matrices have fixed valid shapes")]
mod tests {
    use super::*;

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

        M::compute_distances(matrix(&point, 1), &leaders, &mut storage).unwrap();

        storage[0]
    }

    mod create_leaders_tests {
        use super::*;

        #[test]
        fn l2_leader_norms_preserve_the_sequential_reduction_order() {
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

    mod compute_distances_tests {
        use super::*;

        #[test]
        fn squared_l2_ranking_omits_the_point_norm() {
            // Given
            let point = [0.0_f32, 4.0];
            let leader = [3.0_f32, 4.0];
            let leader_squared_norm = leader[0].mul_add(leader[0], leader[1] * leader[1]);
            let dot = point[0].mul_add(leader[0], point[1] * leader[1]);
            let expected = (-2.0_f32).mul_add(dot, leader_squared_norm);

            // When
            let actual = compute_one_ranking::<L2>(point, leader);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn cosine_ranking_equals_one_minus_normalized_similarity() {
            // Given
            let point = [2.0_f32, 0.0];
            let leader = [1.0_f32, 1.0];
            let dot = point[0].mul_add(leader[0], point[1] * leader[1]);
            let point_norm = point[0].hypot(point[1]);
            let leader_norm = leader[0].hypot(leader[1]);
            let expected = 1.0 - dot / (point_norm * leader_norm);

            // When
            let actual = compute_one_ranking::<Cosine>(point, leader);

            // Then
            assert!(
                (actual - expected).abs() <= FLOAT_TOLERANCE,
                "actual {actual} differs from expected {expected}"
            );
        }

        #[rstest::rstest]
        #[case::zero_point([0.0, 0.0], [1.0, 0.0])]
        #[case::zero_leader([1.0, 0.0], [0.0, 0.0])]
        #[case::subnormal_point([f32::MIN_POSITIVE.sqrt() / 2.0, 0.0], [1.0, 0.0])]
        #[case::subnormal_leader([1.0, 0.0], [f32::MIN_POSITIVE.sqrt() / 2.0, 0.0])]
        fn small_norm_produces_unit_cosine_ranking(
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
        #[case::normalized_cosine(compute_one_ranking::<CosineNormalized>)]
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
        fn normalized_cosine_ranking_equals_the_negative_dot_product() {
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
        fn inner_product_ranking_equals_the_negative_dot_product() {
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
        fn l2_reinitializes_every_output_row_for_a_new_point_stripe() {
            let leader_values = [3.0_f32, 4.0, 0.0, 2.0];
            let leaders = L2::create_leaders(matrix(&leader_values, 2));
            let mut output = [STALE_DISTANCE; 4];

            // Leader norms are 25 and 4; each entry is norm - 2 * dot.
            L2::compute_distances(matrix(&[1.0, 0.0, 0.0, 1.0], 2), &leaders, &mut output).unwrap();
            assert_eq!(output, [19.0, 4.0, 17.0, 0.0]);

            L2::compute_distances(matrix(&[2.0, 1.0, -1.0, 2.0], 2), &leaders, &mut output)
                .unwrap();
            assert_eq!(output, [5.0, 0.0, 15.0, -4.0]);
        }

        #[test]
        fn cosine_overwrites_output_for_a_new_point_stripe() {
            let leader_values = [1.0_f32, 0.0, 0.0, 2.0];
            let leaders = Cosine::create_leaders(matrix(&leader_values, 2));
            let mut output = [STALE_DISTANCE; 4];

            Cosine::compute_distances(matrix(&[1.0, 0.0, 0.0, 3.0], 2), &leaders, &mut output)
                .unwrap();
            assert_eq!(output, [0.0, 1.0, 1.0, 0.0]);

            Cosine::compute_distances(matrix(&[0.0, 4.0, -2.0, 0.0], 2), &leaders, &mut output)
                .unwrap();
            assert_eq!(output, [1.0, 0.0, 2.0, 1.0]);
        }
    }
}
