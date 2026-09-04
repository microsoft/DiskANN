/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Numerical kernels for PiPNN graph construction.
//!
//! Metric modules fill portable distance buffers. Kernel modules use architecture
//! `A` to traverse those buffers. Callers select metric `M` once per graph build.

#[allow(dead_code)]
mod leaf_kernel;
#[allow(dead_code)]
mod leaf_metric;
#[allow(dead_code)]
mod partition_kernel;
#[allow(dead_code)]
mod partition_metric;
#[allow(dead_code)]
mod simd;

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
mod cosine_distance_contract_tests {
    use super::cosine_distance;
    use rstest::rstest;

    mod cosine_distance_tests {
        use super::*;

        #[test]
        fn zero_norm_takes_precedence_over_a_nan_dot_product() {
            // Given
            let dot = f32::NAN;
            let source_norm = 0.0_f32;
            let target_norm = 1.0_f32;
            let zero_similarity = 0.0_f32;
            let expected = 1.0 - zero_similarity;

            // When
            let actual = cosine_distance(dot, source_norm, target_norm);

            // Then
            assert_eq!(actual, expected);
        }

        #[rstest]
        #[case::zero_source(0.0, 1.0)]
        #[case::zero_target(1.0, 0.0)]
        #[case::subnormal_source(f32::MIN_POSITIVE.sqrt() / 2.0, 1.0)]
        #[case::subnormal_target(1.0, f32::MIN_POSITIVE.sqrt() / 2.0)]
        #[trace]
        fn zero_or_subnormal_norm_produces_unit_distance(
            #[case] source_norm: f32,
            #[case] target_norm: f32,
        ) {
            // Given
            let dot = 0.0_f32;
            let zero_similarity = 0.0_f32;
            let expected = 1.0 - zero_similarity;

            // When
            let actual = cosine_distance(dot, source_norm, target_norm);

            // Then
            assert_eq!(actual, expected);
        }

        #[test]
        fn minimum_normal_norm_uses_normalized_similarity() {
            // Given
            let source_norm = f32::MIN_POSITIVE.sqrt();
            let target_norm = 1.0_f32;
            let expected_similarity = 0.5_f32;
            let dot = expected_similarity * source_norm * target_norm;
            let expected = 1.0 - expected_similarity;

            // When
            let actual = cosine_distance(dot, source_norm, target_norm);

            // Then
            assert_eq!(actual, expected);
        }

        #[rstest]
        #[case::above_one(1.0)]
        #[case::below_negative_one(-1.0)]
        #[trace]
        fn finite_similarity_outside_the_cosine_range_is_clamped(#[case] bounded_similarity: f32) {
            // Given
            let source_norm = 2.0_f32;
            let target_norm = 2.0_f32;
            let norm_product = source_norm * target_norm;
            let rounding_excess = f32::EPSILON * norm_product;
            let dot = bounded_similarity * (norm_product + rounding_excess);
            let expected = 1.0 - bounded_similarity;

            // When
            let actual = cosine_distance(dot, source_norm, target_norm);

            // Then
            assert_eq!(actual, expected);
        }

        #[rstest]
        #[case::nan_dot(f32::NAN, 1.0, 1.0)]
        #[case::nan_source_norm(0.0, f32::NAN, 1.0)]
        #[case::nan_target_norm(0.0, 1.0, f32::NAN)]
        #[trace]
        fn nan_without_a_zero_norm_produces_nan_distance(
            #[case] dot: f32,
            #[case] source_norm: f32,
            #[case] target_norm: f32,
        ) {
            // Given: the case supplies one NaN. The other values do not select the zero-norm rule.

            // When
            let actual = cosine_distance(dot, source_norm, target_norm);

            // Then
            assert!(actual.is_nan());
        }
    }
}
