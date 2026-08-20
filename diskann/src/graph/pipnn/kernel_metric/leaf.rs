/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::MatrixView;
use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector};

use super::super::simd::{PiPNNSIMDSchema, PiPNNSIMDVector};
use super::{
    Cosine, CosineNormalized, InnerProduct, L2, cosine_distance_simd, cosine_distance_single,
};

/// Compute leaf distances for one concrete metric.
pub(in super::super) trait LeafMetric: Send + Sync + 'static {
    /// SIMD representation for leaf distance scores.
    type Simd<A>: PiPNNSIMDVector<Arch = A>
    where
        A: PiPNNSIMDSchema;

    /// Prepare one contiguous metric-specific norm for each leaf-local point.
    fn prepare_leaf_norms(_dots: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.clear();
    }

    /// Prepare one source norm for reuse across SIMD target groups.
    #[inline(always)]
    fn source_simd<A>(arch: A, _norms: &[f32], _source: usize) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        Self::Simd::<A>::default(arch)
    }

    /// Prepare one source norm for reuse across single target values.
    #[inline(always)]
    fn source_single(_norms: &[f32], _source: usize) -> f32 {
        0.0
    }

    /// Compute distances for one complete SIMD group.
    fn distances_simd<A>(
        arch: A,
        norms: &[f32],
        source_norms: Self::Simd<A>,
        dot_products: Self::Simd<A>,
        first_target: usize,
    ) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema;

    /// Compute one distance outside the complete SIMD prefix.
    fn distance_single(norms: &[f32], source_norm: f32, dot_product: f32, target: usize) -> f32;
}

/// Load one complete SIMD group of prepared norms.
#[inline(always)]
fn load_norms_simd<F>(arch: F::Arch, norms: &[f32], first_norm: usize) -> F
where
    F: PiPNNSIMDVector,
{
    let last_norm = first_norm + F::LANES;
    let norm_group = &norms[first_norm..last_norm];

    // SAFETY: `norm_group` contains one complete SIMD group.
    unsafe { F::load_simd(arch, norm_group.as_ptr()) }
}

impl LeafMetric for L2 {
    type Simd<A>
        = A::LeafScore
    where
        A: PiPNNSIMDSchema;

    fn prepare_leaf_norms(dots: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(dots.nrows(), 0.0);
        for (point, norm) in norms.iter_mut().enumerate() {
            *norm = dots[(point, point)];
        }
    }

    #[inline(always)]
    fn source_simd<A>(arch: A, norms: &[f32], source: usize) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        Self::Simd::<A>::splat(arch, norms[source])
    }

    #[inline(always)]
    fn source_single(norms: &[f32], source: usize) -> f32 {
        norms[source]
    }

    #[inline(always)]
    fn distances_simd<A>(
        arch: A,
        norms: &[f32],
        source_norms: Self::Simd<A>,
        dot_products: Self::Simd<A>,
        first_target: usize,
    ) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        let target_norms = load_norms_simd::<Self::Simd<A>>(arch, norms, first_target);
        (Self::Simd::<A>::splat(arch, -2.0).mul_add_simd(dot_products, source_norms) + target_norms)
            .max_simd(Self::Simd::<A>::default(arch))
    }

    #[inline(always)]
    fn distance_single(norms: &[f32], source_norm: f32, dot_product: f32, target: usize) -> f32 {
        ((-2.0_f32).mul_add(dot_product, source_norm) + norms[target]).max(0.0)
    }
}

impl LeafMetric for Cosine {
    type Simd<A>
        = A::LeafScore
    where
        A: PiPNNSIMDSchema;

    fn prepare_leaf_norms(dots: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(dots.nrows(), 0.0);
        for (point, norm) in norms.iter_mut().enumerate() {
            *norm = dots[(point, point)].sqrt();
        }
    }

    #[inline(always)]
    fn source_simd<A>(arch: A, norms: &[f32], source: usize) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        Self::Simd::<A>::splat(arch, norms[source])
    }

    #[inline(always)]
    fn source_single(norms: &[f32], source: usize) -> f32 {
        norms[source]
    }

    #[inline(always)]
    fn distances_simd<A>(
        arch: A,
        norms: &[f32],
        source_norms: Self::Simd<A>,
        dot_products: Self::Simd<A>,
        first_target: usize,
    ) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        let target_norms = load_norms_simd::<Self::Simd<A>>(arch, norms, first_target);
        cosine_distance_simd(arch, dot_products, source_norms, target_norms)
            .max_simd(Self::Simd::<A>::default(arch))
    }

    #[inline(always)]
    fn distance_single(norms: &[f32], source_norm: f32, dot_product: f32, target: usize) -> f32 {
        cosine_distance_single(dot_product, source_norm, norms[target]).max(0.0)
    }
}

impl LeafMetric for CosineNormalized {
    type Simd<A>
        = A::LeafScore
    where
        A: PiPNNSIMDSchema;

    #[inline(always)]
    fn distances_simd<A>(
        arch: A,
        _norms: &[f32],
        _source_norms: Self::Simd<A>,
        dot_products: Self::Simd<A>,
        _first_target: usize,
    ) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        Self::Simd::<A>::splat(arch, 1.0) - dot_products
    }

    #[inline(always)]
    fn distance_single(_norms: &[f32], _source_norm: f32, dot_product: f32, _target: usize) -> f32 {
        1.0 - dot_product
    }
}

impl LeafMetric for InnerProduct {
    type Simd<A>
        = A::LeafScore
    where
        A: PiPNNSIMDSchema;

    #[inline(always)]
    fn distances_simd<A>(
        arch: A,
        _norms: &[f32],
        _source_norms: Self::Simd<A>,
        dot_products: Self::Simd<A>,
        _first_target: usize,
    ) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        Self::Simd::<A>::default(arch) - dot_products
    }

    #[inline(always)]
    fn distance_single(_norms: &[f32], _source_norm: f32, dot_product: f32, _target: usize) -> f32 {
        -dot_product
    }
}
