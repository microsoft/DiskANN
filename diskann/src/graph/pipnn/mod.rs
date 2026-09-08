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
#[allow(dead_code)]
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
    #[case::zero_source(0.0, 0.0, 1.0)]
    #[case::zero_target(0.0, 1.0, 0.0)]
    #[case::subnormal_source(f32::MIN_POSITIVE.sqrt() / 2.0, f32::MIN_POSITIVE.sqrt() / 2.0, 1.0)]
    #[case::subnormal_target(f32::MIN_POSITIVE.sqrt() / 2.0, 1.0, f32::MIN_POSITIVE.sqrt() / 2.0)]
    #[case::zero_norm_before_nan_dot(f32::NAN, 0.0, 1.0)]
    fn small_norm_produces_unit_distance(
        #[case] dot: f32,
        #[case] source_norm: f32,
        #[case] target_norm: f32,
    ) {
        assert_eq!(cosine_distance(dot, source_norm, target_norm), 1.0);
    }

    #[test]
    fn minimum_normal_norm_uses_normalized_similarity() {
        let norm = f32::MIN_POSITIVE.sqrt();
        assert_eq!(cosine_distance(0.5 * norm, norm, 1.0), 0.5);
    }

    #[rstest]
    #[case::above_one(1.0 + f32::EPSILON, 0.0)]
    #[case::below_negative_one(-1.0 - f32::EPSILON, 2.0)]
    fn finite_similarity_outside_the_cosine_range_is_clamped(
        #[case] similarity: f32,
        #[case] expected: f32,
    ) {
        assert_eq!(cosine_distance(similarity, 1.0, 1.0), expected);
    }

    #[rstest]
    #[case::nan_dot(f32::NAN, 1.0, 1.0)]
    #[case::nan_source_norm(0.0, f32::NAN, 1.0)]
    #[case::nan_target_norm(0.0, 1.0, f32::NAN)]
    fn nan_without_a_zero_norm_produces_nan_distance(
        #[case] dot: f32,
        #[case] source_norm: f32,
        #[case] target_norm: f32,
    ) {
        assert!(cosine_distance(dot, source_norm, target_norm).is_nan());
    }
}
