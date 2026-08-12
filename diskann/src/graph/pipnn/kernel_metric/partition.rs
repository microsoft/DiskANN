/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::MatrixView;
use diskann_vector::{Norm, norm::FastL2NormSquared};
use diskann_wide::{SIMDFloat, SIMDSelect, SIMDVector};

use super::{
    Cosine, CosineNormalized, InnerProduct, L2, PartitionNorms, cosine_distance_simd,
    cosine_distance_single,
};

/// Compute partition rankings for one concrete metric.
pub(in super::super) trait PartitionMetric: Send + Sync + 'static {
    /// Prepare one norm value for each point in the active stripe.
    fn prepare_point_norms(_points: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.clear();
    }

    /// Prepare one norm value for each sampled leader.
    fn prepare_leader_norms(_leaders: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.clear();
    }

    /// Prepare one point norm for reuse across SIMD leader groups.
    #[inline(always)]
    fn point_simd<F>(arch: F::Arch, _norms: PartitionNorms<'_>, _point: usize) -> F
    where
        F: SIMDVector<Scalar = f32>,
    {
        F::default(arch)
    }

    /// Prepare one point norm for reuse across single leader values.
    #[inline(always)]
    fn point_single(_norms: PartitionNorms<'_>, _point: usize) -> f32 {
        0.0
    }

    /// Compute rankings for one complete SIMD group.
    fn rankings_simd<F>(
        arch: F::Arch,
        norms: PartitionNorms<'_>,
        point_norms: F,
        dot_products: F,
        first_leader: usize,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// Compute one ranking outside the complete SIMD prefix.
    fn ranking_single(
        norms: PartitionNorms<'_>,
        point_norm: f32,
        dot_product: f32,
        leader: usize,
    ) -> f32;
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

impl PartitionMetric for L2 {
    fn prepare_leader_norms(leaders: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(leaders.nrows(), 0.0);
        for (norm, leader) in norms.iter_mut().zip(leaders.row_iter()) {
            *norm = leader.iter().map(|value| value * value).sum();
        }
    }

    #[inline(always)]
    fn rankings_simd<F>(
        arch: F::Arch,
        norms: PartitionNorms<'_>,
        _point_norms: F,
        dot_products: F,
        first_leader: usize,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        let leader_norms = load_norms_simd::<F>(arch, norms.leader_norms, first_leader);
        F::splat(arch, -2.0).mul_add_simd(dot_products, leader_norms)
    }

    #[inline(always)]
    fn ranking_single(
        norms: PartitionNorms<'_>,
        _point_norm: f32,
        dot_product: f32,
        leader: usize,
    ) -> f32 {
        (-2.0_f32).mul_add(dot_product, norms.leader_norms[leader])
    }
}

impl PartitionMetric for Cosine {
    fn prepare_point_norms(points: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(points.nrows(), 0.0);
        for (norm, point) in norms.iter_mut().zip(points.row_iter()) {
            *norm = FastL2NormSquared.evaluate(point).sqrt();
        }
    }

    fn prepare_leader_norms(leaders: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(leaders.nrows(), 0.0);
        for (norm, leader) in norms.iter_mut().zip(leaders.row_iter()) {
            *norm = leader.iter().map(|value| value * value).sum::<f32>().sqrt();
        }
    }

    #[inline(always)]
    fn point_simd<F>(arch: F::Arch, norms: PartitionNorms<'_>, point: usize) -> F
    where
        F: SIMDVector<Scalar = f32>,
    {
        F::splat(arch, norms.point_norms[point])
    }

    #[inline(always)]
    fn point_single(norms: PartitionNorms<'_>, point: usize) -> f32 {
        norms.point_norms[point]
    }

    #[inline(always)]
    fn rankings_simd<F>(
        arch: F::Arch,
        norms: PartitionNorms<'_>,
        point_norms: F,
        dot_products: F,
        first_leader: usize,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        let leader_norms = load_norms_simd::<F>(arch, norms.leader_norms, first_leader);
        cosine_distance_simd(arch, dot_products, point_norms, leader_norms)
    }

    #[inline(always)]
    fn ranking_single(
        norms: PartitionNorms<'_>,
        point_norm: f32,
        dot_product: f32,
        leader: usize,
    ) -> f32 {
        cosine_distance_single(dot_product, point_norm, norms.leader_norms[leader])
    }
}

impl PartitionMetric for CosineNormalized {
    #[inline(always)]
    fn rankings_simd<F>(
        arch: F::Arch,
        _norms: PartitionNorms<'_>,
        _point_norms: F,
        dot_products: F,
        _first_leader: usize,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        F::splat(arch, 1.0) - dot_products
    }

    #[inline(always)]
    fn ranking_single(
        _norms: PartitionNorms<'_>,
        _point_norm: f32,
        dot_product: f32,
        _leader: usize,
    ) -> f32 {
        1.0 - dot_product
    }
}

impl PartitionMetric for InnerProduct {
    #[inline(always)]
    fn rankings_simd<F>(
        arch: F::Arch,
        _norms: PartitionNorms<'_>,
        _point_norms: F,
        dot_products: F,
        _first_leader: usize,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        F::default(arch) - dot_products
    }

    #[inline(always)]
    fn ranking_single(
        _norms: PartitionNorms<'_>,
        _point_norm: f32,
        dot_product: f32,
        _leader: usize,
    ) -> f32 {
        -dot_product
    }
}
