/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::MatrixView;
use diskann_wide::{SIMDFloat, SIMDSelect, SIMDVector};

use super::{
    Cosine, CosineNormalized, InnerProduct, L2, cosine_distance_simd, cosine_distance_single,
};

/// Compute leaf distances for one concrete metric.
pub(in super::super) trait LeafMetric: Send + Sync + 'static {
    /// Prepare one contiguous metric-specific norm for each leaf-local point.
    fn prepare_leaf_norms(_dots: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.clear();
    }

    /// Prepare one source norm for reuse across SIMD target groups.
    #[inline(always)]
    fn source_simd<F>(arch: F::Arch, _norms: &[f32], _source: usize) -> F
    where
        F: SIMDVector<Scalar = f32>,
    {
        F::default(arch)
    }

    /// Prepare one source norm for reuse across single target values.
    #[inline(always)]
    fn source_single(_norms: &[f32], _source: usize) -> f32 {
        0.0
    }

    /// Compute distances for one complete SIMD group.
    fn distances_simd<F>(
        arch: F::Arch,
        norms: &[f32],
        source_norms: F,
        dot_products: F,
        first_target: usize,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// Compute one distance outside the complete SIMD prefix.
    fn distance_single(norms: &[f32], source_norm: f32, dot_product: f32, target: usize) -> f32;
}

/// Load one complete SIMD group of prepared norms.
#[inline(always)]
fn load_norms_simd<F>(arch: F::Arch, norms: &[f32], first_norm: usize) -> F
where
    F: SIMDVector<Scalar = f32>,
{
    let last_norm = first_norm + F::LANES;
    let norm_group = &norms[first_norm..last_norm];

    // SAFETY: `norm_group` contains one complete SIMD group.
    unsafe { F::load_simd(arch, norm_group.as_ptr()) }
}

impl LeafMetric for L2 {
    fn prepare_leaf_norms(dots: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(dots.nrows(), 0.0);
        for (point, norm) in norms.iter_mut().enumerate() {
            *norm = dots[(point, point)];
        }
    }

    #[inline(always)]
    fn source_simd<F>(arch: F::Arch, norms: &[f32], source: usize) -> F
    where
        F: SIMDVector<Scalar = f32>,
    {
        F::splat(arch, norms[source])
    }

    #[inline(always)]
    fn source_single(norms: &[f32], source: usize) -> f32 {
        norms[source]
    }

    #[inline(always)]
    fn distances_simd<F>(
        arch: F::Arch,
        norms: &[f32],
        source_norms: F,
        dot_products: F,
        first_target: usize,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        let target_norms = load_norms_simd::<F>(arch, norms, first_target);
        (F::splat(arch, -2.0).mul_add_simd(dot_products, source_norms) + target_norms)
            .max_simd(F::default(arch))
    }

    #[inline(always)]
    fn distance_single(norms: &[f32], source_norm: f32, dot_product: f32, target: usize) -> f32 {
        ((-2.0_f32).mul_add(dot_product, source_norm) + norms[target]).max(0.0)
    }
}

impl LeafMetric for Cosine {
    fn prepare_leaf_norms(dots: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(dots.nrows(), 0.0);
        for (point, norm) in norms.iter_mut().enumerate() {
            *norm = dots[(point, point)].sqrt();
        }
    }

    #[inline(always)]
    fn source_simd<F>(arch: F::Arch, norms: &[f32], source: usize) -> F
    where
        F: SIMDVector<Scalar = f32>,
    {
        F::splat(arch, norms[source])
    }

    #[inline(always)]
    fn source_single(norms: &[f32], source: usize) -> f32 {
        norms[source]
    }

    #[inline(always)]
    fn distances_simd<F>(
        arch: F::Arch,
        norms: &[f32],
        source_norms: F,
        dot_products: F,
        first_target: usize,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        let target_norms = load_norms_simd::<F>(arch, norms, first_target);
        cosine_distance_simd(arch, dot_products, source_norms, target_norms)
            .max_simd(F::default(arch))
    }

    #[inline(always)]
    fn distance_single(norms: &[f32], source_norm: f32, dot_product: f32, target: usize) -> f32 {
        cosine_distance_single(dot_product, source_norm, norms[target]).max(0.0)
    }
}

impl LeafMetric for CosineNormalized {
    #[inline(always)]
    fn distances_simd<F>(
        arch: F::Arch,
        _norms: &[f32],
        _source_norms: F,
        dot_products: F,
        _first_target: usize,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        F::splat(arch, 1.0) - dot_products
    }

    #[inline(always)]
    fn distance_single(_norms: &[f32], _source_norm: f32, dot_product: f32, _target: usize) -> f32 {
        1.0 - dot_product
    }
}

impl LeafMetric for InnerProduct {
    #[inline(always)]
    fn distances_simd<F>(
        arch: F::Arch,
        _norms: &[f32],
        _source_norms: F,
        dot_products: F,
        _first_target: usize,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        F::default(arch) - dot_products
    }

    #[inline(always)]
    fn distance_single(_norms: &[f32], _source_norm: f32, dot_product: f32, _target: usize) -> f32 {
        -dot_product
    }
}
