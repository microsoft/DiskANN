/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Metric formulas for leaf-local neighbor selection.

use diskann_wide::{SIMDFloat, SIMDSelect, SIMDVector};

use super::{
    Cosine, CosineNormalized, InnerProduct, L2, MetricTag, cosine_distance, cosine_distance_scalar,
};

/// Metric contract for leaf-local neighbor selection.
///
/// Each function returns an ascending distance. The leaf kernel supplies squared
/// norms to L2 and norms to cosine. Dot-only metrics receive zero norm values.
pub(in super::super) trait LeafKernelMetric: MetricTag {
    /// Compute SIMD distances from one source to earlier leaf targets.
    fn leaf_distance<F>(arch: F::Arch, dot: F, source_norm: F, target_norm: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// Compute one scalar-tail leaf distance.
    fn leaf_distance_scalar(dot: f32, source_norm: f32, target_norm: f32) -> f32;
}

/// Clamp negative SIMD roundoff to zero and preserve NaN lanes.
#[inline(always)]
fn clamp_nonnegative<F>(arch: F::Arch, distance: F) -> F
where
    F: SIMDVector<Scalar = f32> + SIMDFloat,
    F::Mask: SIMDSelect<F>,
{
    let zero = F::default(arch);
    distance
        .eq_simd(distance)
        .select(zero.max_simd(distance), distance)
}

/// Clamp negative scalar roundoff to zero and preserve NaN.
#[inline(always)]
fn clamp_nonnegative_scalar(distance: f32) -> f32 {
    if distance < 0.0 { 0.0 } else { distance }
}

impl LeafKernelMetric for L2 {
    #[inline(always)]
    fn leaf_distance<F>(arch: F::Arch, dot: F, source_norm: F, target_norm: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        clamp_nonnegative(arch, source_norm + target_norm - F::splat(arch, 2.0) * dot)
    }

    #[inline(always)]
    fn leaf_distance_scalar(dot: f32, source_norm: f32, target_norm: f32) -> f32 {
        clamp_nonnegative_scalar(source_norm + target_norm - 2.0 * dot)
    }
}

impl LeafKernelMetric for Cosine {
    #[inline(always)]
    fn leaf_distance<F>(arch: F::Arch, dot: F, source_norm: F, target_norm: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        clamp_nonnegative(arch, cosine_distance(arch, dot, source_norm, target_norm))
    }

    #[inline(always)]
    fn leaf_distance_scalar(dot: f32, source_norm: f32, target_norm: f32) -> f32 {
        clamp_nonnegative_scalar(cosine_distance_scalar(dot, source_norm, target_norm))
    }
}

impl LeafKernelMetric for CosineNormalized {
    #[inline(always)]
    fn leaf_distance<F>(arch: F::Arch, dot: F, _: F, _: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        clamp_nonnegative(arch, F::splat(arch, 1.0) - dot)
    }

    #[inline(always)]
    fn leaf_distance_scalar(dot: f32, _: f32, _: f32) -> f32 {
        clamp_nonnegative_scalar(1.0 - dot)
    }
}

impl LeafKernelMetric for InnerProduct {
    #[inline(always)]
    fn leaf_distance<F>(arch: F::Arch, dot: F, _: F, _: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        F::default(arch) - dot
    }

    #[inline(always)]
    fn leaf_distance_scalar(dot: f32, _: f32, _: f32) -> f32 {
        -dot
    }
}
