/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! This module provides metric markers and shared numerical functions.

mod leaf;
mod partition;

pub(super) use leaf::LeafMetric;
pub(super) use partition::PartitionMetric;

use super::simd::PiPNNSIMDVector;

pub(super) struct L2;
pub(super) struct Cosine;
pub(super) struct CosineNormalized;
pub(super) struct InnerProduct;

/// Prepared norms for one point stripe and its sampled leaders.
#[derive(Clone, Copy, Debug)]
pub(super) struct PartitionNorms<'a> {
    pub(super) point_norms: &'a [f32],
    pub(super) leader_norms: &'a [f32],
}

/// Compute SIMD cosine distance with the DiskANN zero-norm and NaN rules.
///
/// Each lane contains one point pair. A zero norm produces zero similarity.
/// Finite similarity is clamped to the cosine range before distance conversion.
#[inline(always)]
pub(super) fn cosine_distance_simd<F>(arch: F::Arch, dot: F, source_norm: F, target_norm: F) -> F
where
    F: PiPNNSIMDVector,
{
    let zero = F::default(arch);
    let one = F::splat(arch, 1.0);
    let minimum_norm = F::splat(arch, f32::MIN_POSITIVE.sqrt());
    let source_zero = source_norm.lt_simd(minimum_norm);
    let target_zero = target_norm.lt_simd(minimum_norm);
    let denominator = source_norm * target_norm;
    let safe_denominator = F::select(source_zero, one, F::select(target_zero, one, denominator));
    let cosine = F::select(
        source_zero,
        zero,
        F::select(target_zero, zero, dot / safe_denominator),
    );
    let negative_one = F::splat(arch, -1.0);
    let distance = one - negative_one.max_simd(cosine.min_simd(one));
    F::select(cosine.ne_simd(cosine), cosine, distance)
}

/// Compute one cosine distance with the DiskANN zero-norm and NaN rules.
#[inline(always)]
pub(super) fn cosine_distance_single(dot: f32, source_norm: f32, target_norm: f32) -> f32 {
    if source_norm < f32::MIN_POSITIVE.sqrt() || target_norm < f32::MIN_POSITIVE.sqrt() {
        1.0
    } else {
        let cosine = dot / (source_norm * target_norm);
        1.0 - cosine.clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::cosine_distance_single;

    mod test_support {
        use super::super::super::simd::PiPNNSIMDSchema;
        use super::super::cosine_distance_simd;
        use diskann_wide::{ARCH, SIMDVector, arch::Current};

        pub(super) fn run_cosine_distance_simd(
            dot_products: [f32; 16],
            source_norms: [f32; 16],
            target_norms: [f32; 16],
        ) -> [f32; 16] {
            type TestVector = <Current as PiPNNSIMDSchema>::Vector;
            assert_eq!(16 % TestVector::LANES, 0);
            let mut output = [0.0; 16];
            for first in (0..16).step_by(TestVector::LANES) {
                // SAFETY: each offset starts one complete SIMD group in every array.
                unsafe {
                    let dots = TestVector::load_simd(ARCH, dot_products.as_ptr().add(first));
                    let sources = TestVector::load_simd(ARCH, source_norms.as_ptr().add(first));
                    let targets = TestVector::load_simd(ARCH, target_norms.as_ptr().add(first));
                    cosine_distance_simd(ARCH, dots, sources, targets)
                        .store_simd(output.as_mut_ptr().add(first));
                }
            }
            output
        }
    }

    mod cosine_distance_single_tests {
        use super::cosine_distance_single;

        #[test]
        fn zero_source_norm_takes_precedence_over_nan_dot_product() {
            // Given
            let source_norm = 0.0;
            let zero_norm_similarity = 0.0;
            let expected_one_minus_zero_similarity = 1.0 - zero_norm_similarity;

            // When
            let actual_distance = cosine_distance_single(f32::NAN, source_norm, 2.0);

            // Then
            assert_eq!(actual_distance, expected_one_minus_zero_similarity);
        }

        #[test]
        fn zero_target_norm_produces_unit_distance() {
            // Given
            let target_norm = 0.0;
            let zero_norm_similarity = 0.0;
            let expected_one_minus_zero_similarity = 1.0 - zero_norm_similarity;

            // When
            let actual_distance = cosine_distance_single(0.0, 2.0, target_norm);

            // Then
            assert_eq!(actual_distance, expected_one_minus_zero_similarity);
        }

        #[test]
        fn minimum_normal_norm_uses_normalized_similarity() {
            // Given
            let source_norm = f32::MIN_POSITIVE.sqrt();
            let target_norm = 1.0;
            let dot_product = source_norm / 2.0;
            let expected_distance = 1.0 - dot_product / (source_norm * target_norm);

            // When
            let actual_distance = cosine_distance_single(dot_product, source_norm, target_norm);

            // Then
            assert_eq!(actual_distance, expected_distance);
        }

        #[test]
        fn similarity_above_one_clamps_to_zero_distance() {
            // Given
            let dot_product_just_above_norm_product = 4.000_001;
            let maximum_cosine_similarity = 1.0;
            let expected_one_minus_maximum_similarity = 1.0 - maximum_cosine_similarity;

            // When
            let actual_distance =
                cosine_distance_single(dot_product_just_above_norm_product, 2.0, 2.0);

            // Then
            assert_eq!(actual_distance, expected_one_minus_maximum_similarity);
        }

        #[test]
        fn similarity_below_negative_one_clamps_to_distance_two() {
            // Given
            let dot_product_just_below_negative_norm_product = -4.000_001;
            let minimum_cosine_similarity = -1.0;
            let expected_one_minus_minimum_similarity = 1.0 - minimum_cosine_similarity;

            // When
            let actual_distance =
                cosine_distance_single(dot_product_just_below_negative_norm_product, 2.0, 2.0);

            // Then
            assert_eq!(actual_distance, expected_one_minus_minimum_similarity);
        }

        #[test]
        fn nan_similarity_remains_nan() {
            let actual_distance = cosine_distance_single(f32::NAN, 1.0, 1.0);
            assert!(actual_distance.is_nan());
        }

        #[test]
        fn nan_source_norm_produces_nan_distance() {
            let actual_distance = cosine_distance_single(0.0, f32::NAN, 1.0);
            assert!(actual_distance.is_nan());
        }

        #[test]
        fn nan_target_norm_produces_nan_distance() {
            let actual_distance = cosine_distance_single(0.0, 1.0, f32::NAN);
            assert!(actual_distance.is_nan());
        }
    }

    mod cosine_distance_simd_tests {
        use super::test_support::run_cosine_distance_simd;

        #[test]
        fn zero_source_norm_takes_precedence_over_nan_dot_product_in_every_lane() {
            let actual_distances = run_cosine_distance_simd([f32::NAN; 16], [0.0; 16], [2.0; 16]);
            assert_eq!(actual_distances, [1.0; 16]);
        }

        #[test]
        fn zero_target_norm_produces_unit_distance_in_every_lane() {
            let actual_distances = run_cosine_distance_simd([0.0; 16], [2.0; 16], [0.0; 16]);
            assert_eq!(actual_distances, [1.0; 16]);
        }

        #[test]
        fn subnormal_norm_produces_unit_distance_in_every_lane() {
            let subnormal_norm = f32::MIN_POSITIVE.sqrt() / 2.0;
            let actual_distances =
                run_cosine_distance_simd([0.0; 16], [2.0; 16], [subnormal_norm; 16]);
            assert_eq!(actual_distances, [1.0; 16]);
        }

        #[test]
        fn minimum_normal_norm_uses_normalized_similarity_in_every_lane() {
            let source_norm = f32::MIN_POSITIVE.sqrt();
            let target_norm = 1.0;
            let dot_product = source_norm / 2.0;
            let expected_distance = 1.0 - dot_product / (source_norm * target_norm);

            let actual_distances =
                run_cosine_distance_simd([dot_product; 16], [source_norm; 16], [target_norm; 16]);

            assert_eq!(actual_distances, [expected_distance; 16]);
        }

        #[test]
        fn similarity_outside_bounds_is_clamped_in_every_lane() {
            let dot_above_norm_product = 4.000_001;
            let dot_below_negative_norm_product = -4.000_001;
            let dot_products = [
                f32::INFINITY,
                dot_above_norm_product,
                dot_above_norm_product,
                dot_above_norm_product,
                dot_above_norm_product,
                dot_above_norm_product,
                dot_above_norm_product,
                dot_above_norm_product,
                f32::NEG_INFINITY,
                dot_below_negative_norm_product,
                dot_below_negative_norm_product,
                dot_below_negative_norm_product,
                dot_below_negative_norm_product,
                dot_below_negative_norm_product,
                dot_below_negative_norm_product,
                dot_below_negative_norm_product,
            ];
            let expected_distances = [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            ];

            let actual_distances = run_cosine_distance_simd(dot_products, [2.0; 16], [2.0; 16]);

            assert_eq!(actual_distances, expected_distances);
        }

        #[test]
        fn nan_similarity_remains_nan_in_every_lane() {
            let actual_distances = run_cosine_distance_simd([f32::NAN; 16], [1.0; 16], [1.0; 16]);
            assert!(actual_distances.into_iter().all(f32::is_nan));
        }

        #[test]
        fn nan_source_norm_produces_nan_distance_in_every_lane() {
            let actual_distances = run_cosine_distance_simd([0.0; 16], [f32::NAN; 16], [1.0; 16]);
            assert!(actual_distances.into_iter().all(f32::is_nan));
        }

        #[test]
        fn nan_target_norm_produces_nan_distance_in_every_lane() {
            let actual_distances = run_cosine_distance_simd([0.0; 16], [1.0; 16], [f32::NAN; 16]);
            assert!(actual_distances.into_iter().all(f32::is_nan));
        }
    }
}
