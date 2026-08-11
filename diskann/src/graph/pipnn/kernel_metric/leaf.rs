/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::TryReserveError;

use diskann_utils::views::MatrixView;
use diskann_wide::{SIMDFloat, SIMDSelect, SIMDVector};

use super::{
    Cosine, CosineNormalized, InnerProduct, L2, cosine_distance, cosine_distance_scalar,
    norm_from_squared, resize_norms,
};

/// Leaf formulas return ascending distances.
/// L2 uses squared norms. Cosine uses norms. Other metrics ignore norms.
pub(in super::super) trait LeafMetric: Send + Sync + 'static {
    fn prepare_norms(_: MatrixView<'_, f32>, norms: &mut Vec<f32>) -> Result<(), TryReserveError> {
        norms.clear();
        Ok(())
    }

    fn leaf_distance<F>(arch: F::Arch, dot: F, source_norm: F, target_norm: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    fn leaf_distance_scalar(dot: f32, source_norm: f32, target_norm: f32) -> f32;
}

/// This function clamps negative SIMD roundoff to zero and preserves NaN lanes.
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

/// This function clamps negative scalar roundoff to zero and preserves NaN.
#[inline(always)]
fn clamp_nonnegative_scalar(distance: f32) -> f32 {
    if distance < 0.0 { 0.0 } else { distance }
}

impl LeafMetric for L2 {
    fn prepare_norms(
        dots: MatrixView<'_, f32>,
        norms: &mut Vec<f32>,
    ) -> Result<(), TryReserveError> {
        resize_norms(norms, dots.nrows())?;
        for (point, norm) in norms.iter_mut().enumerate() {
            *norm = dots[(point, point)];
        }
        Ok(())
    }

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

impl LeafMetric for Cosine {
    fn prepare_norms(
        dots: MatrixView<'_, f32>,
        norms: &mut Vec<f32>,
    ) -> Result<(), TryReserveError> {
        resize_norms(norms, dots.nrows())?;
        for (point, norm) in norms.iter_mut().enumerate() {
            *norm = norm_from_squared(dots[(point, point)]);
        }
        Ok(())
    }

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

impl LeafMetric for CosineNormalized {
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

impl LeafMetric for InnerProduct {
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
