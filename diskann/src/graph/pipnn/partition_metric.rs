/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Build ranking distances from point stripes to partition leaders.
//!
//! A ranking distance preserves nearest-first order. One leader set serves all
//! point stripes in a partition split.

use std::sync::OnceLock;

use crate::{ANNError, ANNResult};
use diskann_linalg::Transpose;
use diskann_utils::views::MatrixView;
use diskann_vector::{Norm, norm::FastL2NormSquared};
use diskann_wide::SIMDVector;

use super::{Cosine, CosineNormalized, InnerProduct, L2, cosine_distance, simd::PiPNNSIMDSchema};

/// Store leader values with immutable metric data.
///
/// The metric selects `Cache`. An L2 cache stores squared norms. A cosine cache
/// stores norms. `OnceLock` lets concurrent point stripes initialize data once.
pub(super) struct PartitionLeaders<'a, Cache> {
    values: MatrixView<'a, f32>,
    cache: Cache,
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
    fn compute_distances<A>(
        arch: A,
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        storage: &mut [f32],
    ) -> ANNResult<()>
    where
        A: PiPNNSIMDSchema;
}

/// Compute L2 squared norms with the established sequential reduction order.
///
/// Small rounding differences can change a leader tie. Keep this order equal to
/// the pre-GEMM implementation.
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

/// Copy one leader value into the same column of each point row.
///
/// L2 uses this operation to initialize every row with leader squared norms.
/// Full SIMD groups use dispatched vector stores. The scalar tail copies the rest.
fn fill_rows<A>(arch: A, rows: &mut [f32], row_values: &[f32])
where
    A: PiPNNSIMDSchema,
{
    let row_length = row_values.len();
    let simd_end = row_length - row_length % A::Vector::LANES;
    for row in rows.chunks_exact_mut(row_length) {
        for start in (0..simd_end).step_by(A::Vector::LANES) {
            // SAFETY: both offsets start complete SIMD groups in live slices.
            unsafe {
                A::Vector::load_simd(arch, row_values.as_ptr().add(start))
                    .store_simd(row.as_mut_ptr().add(start));
            }
        }
        row[simd_end..].copy_from_slice(&row_values[simd_end..]);
    }
}

impl PartitionMetric for L2 {
    type Leaders<'a> = PartitionLeaders<'a, OnceLock<Vec<f32>>>;

    fn create_leaders<'a>(values: MatrixView<'a, f32>) -> Self::Leaders<'a> {
        PartitionLeaders {
            values,
            cache: OnceLock::new(),
        }
    }

    fn leader_count(leaders: &Self::Leaders<'_>) -> usize {
        leaders.values.nrows()
    }

    fn compute_distances<A>(
        arch: A,
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        storage: &mut [f32],
    ) -> ANNResult<()>
    where
        A: PiPNNSIMDSchema,
    {
        // The point norm is constant across a point row. It cannot change ranking.
        let leader_norms = leaders
            .cache
            .get_or_init(|| l2_squared_norms(leaders.values));
        arch.run(|| fill_rows(arch, storage, leader_norms));
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
    type Leaders<'a> = PartitionLeaders<'a, OnceLock<Vec<f32>>>;

    fn create_leaders<'a>(values: MatrixView<'a, f32>) -> Self::Leaders<'a> {
        PartitionLeaders {
            values,
            cache: OnceLock::new(),
        }
    }

    fn leader_count(leaders: &Self::Leaders<'_>) -> usize {
        leaders.values.nrows()
    }

    fn compute_distances<A>(
        _arch: A,
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        storage: &mut [f32],
    ) -> ANNResult<()>
    where
        A: PiPNNSIMDSchema,
    {
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
        let leader_norms = leaders.cache.get_or_init(|| cosine_norms(leaders.values));
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
        PartitionLeaders { values, cache: () }
    }

    fn leader_count(leaders: &Self::Leaders<'_>) -> usize {
        leaders.values.nrows()
    }

    fn compute_distances<A>(
        _arch: A,
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        storage: &mut [f32],
    ) -> ANNResult<()>
    where
        A: PiPNNSIMDSchema,
    {
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
    type Leaders<'a> = PartitionLeaders<'a, ()>;

    fn create_leaders<'a>(values: MatrixView<'a, f32>) -> Self::Leaders<'a> {
        PartitionLeaders { values, cache: () }
    }

    fn leader_count(leaders: &Self::Leaders<'_>) -> usize {
        leaders.values.nrows()
    }

    fn compute_distances<A>(
        _arch: A,
        points: MatrixView<'_, f32>,
        leaders: &Self::Leaders<'_>,
        storage: &mut [f32],
    ) -> ANNResult<()>
    where
        A: PiPNNSIMDSchema,
    {
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

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "test matrices have fixed valid shapes")]
mod tests {
    use super::*;
    use diskann_wide::ARCH;

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

        M::compute_distances(ARCH, matrix(&point, 1), &leaders, &mut storage).unwrap();

        storage[0]
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
        fn l2_leader_cache_preserves_the_sequential_reduction_order() {
            fn next_regression_value(state: &mut u64) -> f32 {
                *state ^= *state << 13;
                *state ^= *state >> 7;
                *state ^= *state << 17;
                (((*state >> 40) as f32 / 8_388_608.0) - 1.0) * 1_000.0
            }

            // Given
            const DIMENSIONS: usize = 129;
            const REGRESSION_SEED: u64 = 0x3a85_f952_c718_6e49;
            let mut state = REGRESSION_SEED;
            let _point_values: Vec<f32> = (0..DIMENSIONS)
                .map(|_| next_regression_value(&mut state))
                .collect();
            let leader_values: Vec<f32> = (0..DIMENSIONS)
                .map(|_| next_regression_value(&mut state))
                .collect();
            let expected: f32 = leader_values.iter().map(|value| value * value).sum();
            let reassociated = FastL2NormSquared.evaluate(leader_values.as_slice());
            let leader_matrix =
                MatrixView::try_from(leader_values.as_slice(), 1, DIMENSIONS).unwrap();
            let leaders = L2::create_leaders(leader_matrix);
            let zero_point = vec![0.0_f32; DIMENSIONS];
            let point_matrix = MatrixView::try_from(zero_point.as_slice(), 1, DIMENSIONS).unwrap();
            let mut distance = [STALE_DISTANCE];

            // When
            L2::compute_distances(ARCH, point_matrix, &leaders, &mut distance).unwrap();
            let actual = leaders.cache.get().unwrap()[0];

            // Then
            assert_ne!(expected.to_bits(), reassociated.to_bits());
            assert_eq!(actual.to_bits(), expected.to_bits());
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
        fn reused_l2_leaders_match_fresh_leaders_for_a_new_point_stripe() {
            // Given
            let leader_values = [3.0_f32, 4.0, 0.0, 2.0];
            let leader_count = leader_values.len() / DIMENSION_COUNT;
            let first_point = [1.0_f32, 0.0];
            let second_point = [0.0_f32, 1.0];
            let first_leader_squared_norm =
                leader_values[0].mul_add(leader_values[0], leader_values[1] * leader_values[1]);
            let second_leader_squared_norm =
                leader_values[2].mul_add(leader_values[2], leader_values[3] * leader_values[3]);
            let first_dot =
                second_point[0].mul_add(leader_values[0], second_point[1] * leader_values[1]);
            let second_dot =
                second_point[0].mul_add(leader_values[2], second_point[1] * leader_values[3]);
            let expected = [
                (-2.0_f32).mul_add(first_dot, first_leader_squared_norm),
                (-2.0_f32).mul_add(second_dot, second_leader_squared_norm),
            ];
            let reused_leaders = L2::create_leaders(matrix(&leader_values, leader_count));
            let fresh_leaders = L2::create_leaders(matrix(&leader_values, leader_count));
            let mut discarded_first_output = [STALE_DISTANCE; 2];
            L2::compute_distances(
                ARCH,
                matrix(&first_point, 1),
                &reused_leaders,
                &mut discarded_first_output,
            )
            .unwrap();
            let mut reused_output = [STALE_DISTANCE; 2];
            let mut fresh_output = [STALE_DISTANCE; 2];

            // When
            L2::compute_distances(
                ARCH,
                matrix(&second_point, 1),
                &reused_leaders,
                &mut reused_output,
            )
            .unwrap();
            L2::compute_distances(
                ARCH,
                matrix(&second_point, 1),
                &fresh_leaders,
                &mut fresh_output,
            )
            .unwrap();

            // Then
            assert_eq!(reused_output, expected);
            assert_eq!(fresh_output, expected);
        }

        #[test]
        fn reused_cosine_leaders_match_fresh_leaders_for_a_new_point_stripe() {
            // Given
            let leader_values = [1.0_f32, 0.0, 0.0, 2.0];
            let leader_count = leader_values.len() / DIMENSION_COUNT;
            let first_point = [1.0_f32, 0.0];
            let second_point = [0.0_f32, 1.0];
            let orthogonal_similarity = 0.0_f32;
            let equal_direction_similarity = 1.0_f32;
            let expected = [
                1.0 - orthogonal_similarity,
                1.0 - equal_direction_similarity,
            ];
            let reused_leaders = Cosine::create_leaders(matrix(&leader_values, leader_count));
            let fresh_leaders = Cosine::create_leaders(matrix(&leader_values, leader_count));
            let mut discarded_first_output = [STALE_DISTANCE; 2];
            Cosine::compute_distances(
                ARCH,
                matrix(&first_point, 1),
                &reused_leaders,
                &mut discarded_first_output,
            )
            .unwrap();
            let mut reused_output = [STALE_DISTANCE; 2];
            let mut fresh_output = [STALE_DISTANCE; 2];

            // When
            Cosine::compute_distances(
                ARCH,
                matrix(&second_point, 1),
                &reused_leaders,
                &mut reused_output,
            )
            .unwrap();
            Cosine::compute_distances(
                ARCH,
                matrix(&second_point, 1),
                &fresh_leaders,
                &mut fresh_output,
            )
            .unwrap();

            // Then
            assert_eq!(reused_output, expected);
            assert_eq!(fresh_output, expected);
        }
    }
}
