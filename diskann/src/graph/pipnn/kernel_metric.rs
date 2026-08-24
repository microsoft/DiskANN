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
    one - negative_one.max_simd(cosine.min_simd(one))
}

/// Compute one cosine distance with the DiskANN zero-norm and NaN rules.
#[inline(always)]
pub(super) fn cosine_distance_single(dot: f32, source_norm: f32, target_norm: f32) -> f32 {
    if source_norm < f32::MIN_POSITIVE.sqrt() || target_norm < f32::MIN_POSITIVE.sqrt() {
        1.0
    } else {
        let cosine = dot / (source_norm * target_norm);
        1.0 - (-1.0_f32).max(1.0_f32.min(cosine))
    }
}

#[cfg(test)]
mod tests {
    use super::cosine_distance_single;

    mod cosine_distance_single_tests {
        use super::cosine_distance_single;

        #[test]
        fn zero_source_norm_produces_unit_distance() {
            // Given
            let source_norm = 0.0;
            let zero_norm_similarity = 0.0;
            let expected_one_minus_zero_similarity = 1.0 - zero_norm_similarity;

            // When
            let actual_distance = cosine_distance_single(0.0, source_norm, 2.0);

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
        fn nan_similarity_clamps_to_zero_distance() {
            // Given
            let nan_dot_product = f32::NAN;
            // `f32::min` keeps its finite operand when the other operand is NaN.
            let finite_operand_selected_by_min = 1.0;
            let expected_one_minus_selected_operand = 1.0 - finite_operand_selected_by_min;

            // When
            let actual_distance = cosine_distance_single(nan_dot_product, 1.0, 1.0);

            // Then
            assert_eq!(actual_distance, expected_one_minus_selected_operand);
        }
    }
}
