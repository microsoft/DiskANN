/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::TryReserveError;

use diskann_wide::{SIMDFloat, SIMDSelect, SIMDVector};

use super::{
    Cosine, CosineNormalized, InnerProduct, L2, NormPreparation, cosine_distance_simd,
    cosine_distance_single, norm_from_squared, resize_norms,
};

/// Leaf formulas return ascending distances.
/// L2 uses squared norms. Cosine uses norms. Other metrics ignore norms.
pub(in super::super) trait LeafMetric: Send + Sync + 'static {
    /// Prepare one contiguous metric-specific norm for each leaf-local point.
    fn prepare_leaf_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError>;

    /// Compute distances for one complete SIMD group.
    fn leaf_distance_simd<F>(arch: F::Arch, dot_products: F, source_norms: F, target_norms: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// Compute one distance outside the complete SIMD prefix.
    fn leaf_distance_single(dot_product: f32, source_norm: f32, target_norm: f32) -> f32;
}

/// Clamp negative SIMD roundoff to zero and preserve NaN lanes.
#[inline(always)]
fn clamp_nonnegative_simd<F>(arch: F::Arch, distance: F) -> F
where
    F: SIMDVector<Scalar = f32> + SIMDFloat,
    F::Mask: SIMDSelect<F>,
{
    let zero = F::default(arch);
    distance
        .eq_simd(distance)
        .select(zero.max_simd(distance), distance)
}

/// Clamp negative roundoff to zero and preserve NaN.
#[inline(always)]
fn clamp_nonnegative_single(distance: f32) -> f32 {
    if distance < 0.0 { 0.0 } else { distance }
}

impl LeafMetric for L2 {
    fn prepare_leaf_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        resize_norms(preparation.norms, preparation.values.nrows())?;
        for (point, norm) in preparation.norms.iter_mut().enumerate() {
            *norm = preparation.values[(point, point)];
        }
        Ok(())
    }

    #[inline(always)]
    fn leaf_distance_simd<F>(arch: F::Arch, dot_products: F, source_norms: F, target_norms: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        clamp_nonnegative_simd(
            arch,
            source_norms + target_norms - F::splat(arch, 2.0) * dot_products,
        )
    }

    #[inline(always)]
    fn leaf_distance_single(dot_product: f32, source_norm: f32, target_norm: f32) -> f32 {
        clamp_nonnegative_single(source_norm + target_norm - 2.0 * dot_product)
    }
}

impl LeafMetric for Cosine {
    fn prepare_leaf_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        resize_norms(preparation.norms, preparation.values.nrows())?;
        for (point, norm) in preparation.norms.iter_mut().enumerate() {
            *norm = norm_from_squared(preparation.values[(point, point)]);
        }
        Ok(())
    }

    #[inline(always)]
    fn leaf_distance_simd<F>(arch: F::Arch, dot_products: F, source_norms: F, target_norms: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        clamp_nonnegative_simd(
            arch,
            cosine_distance_simd(arch, dot_products, source_norms, target_norms),
        )
    }

    #[inline(always)]
    fn leaf_distance_single(dot_product: f32, source_norm: f32, target_norm: f32) -> f32 {
        clamp_nonnegative_single(cosine_distance_single(
            dot_product,
            source_norm,
            target_norm,
        ))
    }
}

impl LeafMetric for CosineNormalized {
    fn prepare_leaf_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        preparation.norms.clear();
        Ok(())
    }

    #[inline(always)]
    fn leaf_distance_simd<F>(arch: F::Arch, dot_products: F, source_norms: F, target_norms: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        let _ = (source_norms, target_norms);
        clamp_nonnegative_simd(arch, F::splat(arch, 1.0) - dot_products)
    }

    #[inline(always)]
    fn leaf_distance_single(dot_product: f32, source_norm: f32, target_norm: f32) -> f32 {
        let _ = (source_norm, target_norm);
        clamp_nonnegative_single(1.0 - dot_product)
    }
}

impl LeafMetric for InnerProduct {
    fn prepare_leaf_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        preparation.norms.clear();
        Ok(())
    }

    #[inline(always)]
    fn leaf_distance_simd<F>(arch: F::Arch, dot_products: F, source_norms: F, target_norms: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        let _ = (source_norms, target_norms);
        F::default(arch) - dot_products
    }

    #[inline(always)]
    fn leaf_distance_single(dot_product: f32, source_norm: f32, target_norm: f32) -> f32 {
        let _ = (source_norm, target_norm);
        -dot_product
    }
}
