/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Numerical kernels for PiPNN graph construction.
//!
//! Metric modules fill portable distance buffers. Kernel modules use architecture
//! `A` to traverse those buffers. Callers select metric `M` once per graph build.

#![expect(
    dead_code,
    reason = "graph construction integrates these kernels in the next PR"
)]

mod leaf_kernel;
mod leaf_metric;
mod partition_kernel;
mod partition_metric;
mod simd;
mod topk;

pub(super) struct L2;
pub(super) struct Cosine;
pub(super) struct CosineNormalized;
pub(super) struct InnerProduct;

/// Convert one dot product and two norms to cosine distance.
///
/// Treat a zero or subnormal norm as zero similarity. This rule takes precedence
/// over the dot value. Clamp finite similarity to the cosine range. Otherwise,
/// a NaN input produces a NaN distance.
#[inline(always)]
fn cosine_distance(dot: f32, source_norm: f32, target_norm: f32) -> f32 {
    if source_norm < f32::MIN_POSITIVE.sqrt() || target_norm < f32::MIN_POSITIVE.sqrt() {
        1.0
    } else {
        1.0 - (dot / (source_norm * target_norm)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod cosine_distance_tests {
    use super::cosine_distance;
    use rstest::rstest;

    #[rstest]
    #[case::same_direction(6.0, 2.0, 3.0, 0.0)]
    #[case::opposite_directions(-6.0, 2.0, 3.0, 2.0)]
    #[case::orthogonal(0.0, 2.0, 3.0, 1.0)]
    #[case::positive_similarity(3.0, 2.0, 3.0, 0.5)]
    #[case::negative_similarity(-3.0, 2.0, 3.0, 1.5)]
    fn distance_uses_both_vector_norms(
        #[case] dot: f32,
        #[case] source_norm: f32,
        #[case] target_norm: f32,
        #[case] expected: f32,
    ) {
        assert_eq!(cosine_distance(dot, source_norm, target_norm), expected);
    }

    #[rstest]
    #[case::above_one(1.0 + f32::EPSILON, 0.0)]
    #[case::below_minus_one(-1.0 - f32::EPSILON, 2.0)]
    fn rounding_outside_the_similarity_range_is_clamped(
        #[case] similarity: f32,
        #[case] expected: f32,
    ) {
        assert_eq!(cosine_distance(similarity, 1.0, 1.0), expected);
    }

    #[rstest]
    #[case::zero(0.0)]
    #[case::below_cutoff(f32::from_bits(f32::MIN_POSITIVE.sqrt().to_bits() - 1))]
    fn a_small_norm_takes_precedence_over_the_dot_product(
        #[case] small_norm: f32,
        #[values(0.75, f32::NAN)] dot: f32,
        #[values(false, true)] small_source: bool,
    ) {
        let (source_norm, target_norm) = if small_source {
            (small_norm, 1.0)
        } else {
            (1.0, small_norm)
        };

        assert_eq!(cosine_distance(dot, source_norm, target_norm), 1.0);
    }

    #[rstest]
    #[case::source(true)]
    #[case::target(false)]
    fn a_norm_at_the_cutoff_still_contributes_similarity(#[case] source_at_cutoff: bool) {
        let norm = f32::MIN_POSITIVE.sqrt();
        let (source_norm, target_norm) = if source_at_cutoff {
            (norm, 1.0)
        } else {
            (1.0, norm)
        };

        assert_eq!(cosine_distance(norm / 4.0, source_norm, target_norm), 0.75);
    }

    #[rstest]
    #[case::dot(f32::NAN, 2.0, 3.0)]
    #[case::source_norm(1.0, f32::NAN, 3.0)]
    #[case::target_norm(1.0, 2.0, f32::NAN)]
    fn nan_propagates_when_neither_norm_is_small(
        #[case] dot: f32,
        #[case] source_norm: f32,
        #[case] target_norm: f32,
    ) {
        assert!(cosine_distance(dot, source_norm, target_norm).is_nan());
    }
}

#[cfg(all(test, not(miri)))]
mod test_support {
    use diskann_vector::distance::Metric;

    pub(super) fn dense_points(rows: usize, dimensions: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng, rngs::StdRng};

        // Multiples of 1/8 keep unnormalized products exact at the tested sizes.
        let mut rng = StdRng::seed_from_u64(seed);
        let mut values: Vec<_> = (0..rows * dimensions)
            .map(|_| rng.random_range(-16..=16) as f32 / 8.0)
            .collect();
        for (point, row) in values.chunks_exact_mut(dimensions).enumerate() {
            // A substantial final coordinate makes an omitted dimension visible
            // even after normalization. It also guarantees nonzero norms.
            row[dimensions - 1] = 8.0 + (point % 7) as f32;
        }
        values
    }

    // These scalar definitions use the actual f32 inputs, with f64 arithmetic.
    // Callers supply finite vectors with nonzero norms. L2 includes the point norm.
    pub(super) fn distance(metric: Metric, point: &[f32], target: &[f32]) -> f64 {
        match metric {
            Metric::L2 => point
                .iter()
                .zip(target)
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                .sum(),
            Metric::InnerProduct | Metric::CosineNormalized => -point
                .iter()
                .zip(target)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>(),
            Metric::Cosine => {
                let dot: f64 = point
                    .iter()
                    .zip(target)
                    .map(|(&x, &y)| f64::from(x) * f64::from(y))
                    .sum();
                let norm = |row: &[f32]| {
                    row.iter()
                        .map(|&x| f64::from(x).powi(2))
                        .sum::<f64>()
                        .sqrt()
                };
                1.0 - (dot / (norm(point) * norm(target))).clamp(-1.0, 1.0)
            }
        }
    }

    pub(super) fn normalize(values: &mut [f32], dimensions: usize) {
        for row in values.chunks_exact_mut(dimensions) {
            let norm = row
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt();
            for value in row {
                *value = (f64::from(*value) / norm) as f32;
            }
        }
    }

    // Put the second coordinate in the last dimension so omitting a tail changes ranking.
    pub(super) fn packed_points(
        coordinates: &[[f32; 2]],
        dimensions: usize,
        unit_norm: bool,
    ) -> Vec<f32> {
        let mut values = vec![0.0; coordinates.len() * dimensions];
        for (row, &[x, y]) in values.chunks_exact_mut(dimensions).zip(coordinates) {
            row[0] = x;
            row[dimensions - 1] = y;
        }
        if unit_norm {
            normalize(&mut values, dimensions);
        }
        values
    }

    #[test]
    fn dense_fixtures_are_replayable_and_exercise_the_last_dimension() {
        let values = dense_points(3, 17, 1287);

        assert_eq!(values, dense_points(3, 17, 1287));
        assert_eq!(values.len(), 3 * 17);
        assert_ne!(&values[..17], &values[17..34]);
        assert_eq!([values[16], values[33], values[50]], [8.0, 9.0, 10.0]);
        assert!(values[..16].iter().filter(|&&x| x != 0.0).count() > 8);
    }

    #[rstest::rstest]
    #[case::squared_l2(Metric::L2, 18.0)]
    #[case::negative_dot(Metric::InnerProduct, -2.0)]
    #[case::cosine(Metric::Cosine, 0.7830695421813438)]
    fn scalar_reference_matches_hand_calculated_distances(
        #[case] metric: Metric,
        #[case] expected: f64,
    ) {
        // [1, 2] and [4, -1]: squared difference 18, dot 2, squared norms 5 and 17.
        assert!((distance(metric, &[1.0, 2.0], &[4.0, -1.0]) - expected).abs() < 1.0e-14);
    }

    #[test]
    fn normalized_points_keep_their_direction_and_last_coordinate() {
        let actual = packed_points(&[[3.0, 4.0], [0.0, -2.0]], 3, true);
        assert_eq!(actual, [0.6, 0.0, 0.8, 0.0, 0.0, -1.0]);
        assert_eq!(
            distance(Metric::CosineNormalized, &actual[..3], &actual[3..]),
            f64::from(0.8_f32)
        );
    }
}
