/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::TryReserveError;

use diskann_vector::{Norm, norm::FastL2NormSquared};
use diskann_wide::{SIMDFloat, SIMDSelect, SIMDVector};

use super::{
    Cosine, CosineNormalized, InnerProduct, L2, NormPreparation, cosine_distance_simd,
    cosine_distance_single, norm_from_squared, resize_norms,
};

/// Partition formulas return ascending rankings.
/// L2 uses squared leader norms. Cosine uses point and leader norms.
pub(in super::super) trait PartitionMetric: Send + Sync + 'static {
    /// Prepare one norm value for each point in the active stripe.
    fn prepare_point_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError>;

    /// Prepare one norm value for each sampled leader.
    fn prepare_leader_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError>;

    /// Compute rankings for one complete SIMD group.
    fn partition_ranking_simd<F>(
        arch: F::Arch,
        dot_products: F,
        point_norms: F,
        leader_norms: F,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// Compute one ranking outside the complete SIMD prefix.
    fn partition_ranking_single(dot_product: f32, point_norm: f32, leader_norm: f32) -> f32;
}

impl PartitionMetric for L2 {
    fn prepare_point_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        preparation.norms.clear();
        Ok(())
    }

    fn prepare_leader_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        resize_norms(preparation.norms, preparation.values.nrows())?;
        for (norm, leader) in preparation
            .norms
            .iter_mut()
            .zip(preparation.values.row_iter())
        {
            *norm = leader.iter().map(|value| value * value).sum();
        }
        Ok(())
    }

    #[inline(always)]
    fn partition_ranking_simd<F>(
        arch: F::Arch,
        dot_products: F,
        point_norms: F,
        leader_norms: F,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        let _ = point_norms;
        // Fused arithmetic defines the ranking order for complete SIMD groups.
        F::splat(arch, -2.0).mul_add_simd(dot_products, leader_norms)
    }

    #[inline(always)]
    fn partition_ranking_single(dot_product: f32, point_norm: f32, leader_norm: f32) -> f32 {
        let _ = point_norm;
        // Non-fused arithmetic defines the ranking order outside the SIMD prefix.
        leader_norm - 2.0 * dot_product
    }
}

impl PartitionMetric for Cosine {
    fn prepare_point_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        resize_norms(preparation.norms, preparation.values.nrows())?;
        for (norm, point) in preparation
            .norms
            .iter_mut()
            .zip(preparation.values.row_iter())
        {
            *norm = norm_from_squared(FastL2NormSquared.evaluate(point));
        }
        Ok(())
    }

    fn prepare_leader_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        resize_norms(preparation.norms, preparation.values.nrows())?;
        for (norm, leader) in preparation
            .norms
            .iter_mut()
            .zip(preparation.values.row_iter())
        {
            let squared_norm = leader.iter().map(|value| value * value).sum();
            *norm = norm_from_squared(squared_norm);
        }
        Ok(())
    }

    #[inline(always)]
    fn partition_ranking_simd<F>(
        arch: F::Arch,
        dot_products: F,
        point_norms: F,
        leader_norms: F,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        cosine_distance_simd(arch, dot_products, point_norms, leader_norms)
    }

    #[inline(always)]
    fn partition_ranking_single(dot_product: f32, point_norm: f32, leader_norm: f32) -> f32 {
        cosine_distance_single(dot_product, point_norm, leader_norm)
    }
}

impl PartitionMetric for CosineNormalized {
    fn prepare_point_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        preparation.norms.clear();
        Ok(())
    }

    fn prepare_leader_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        preparation.norms.clear();
        Ok(())
    }

    #[inline(always)]
    fn partition_ranking_simd<F>(
        arch: F::Arch,
        dot_products: F,
        point_norms: F,
        leader_norms: F,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        let _ = (point_norms, leader_norms);
        F::splat(arch, 1.0) - dot_products
    }

    #[inline(always)]
    fn partition_ranking_single(dot_product: f32, point_norm: f32, leader_norm: f32) -> f32 {
        let _ = (point_norm, leader_norm);
        1.0 - dot_product
    }
}

impl PartitionMetric for InnerProduct {
    fn prepare_point_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        preparation.norms.clear();
        Ok(())
    }

    fn prepare_leader_norms(preparation: NormPreparation<'_, '_>) -> Result<(), TryReserveError> {
        preparation.norms.clear();
        Ok(())
    }

    #[inline(always)]
    fn partition_ranking_simd<F>(
        arch: F::Arch,
        dot_products: F,
        point_norms: F,
        leader_norms: F,
    ) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        let _ = (point_norms, leader_norms);
        F::default(arch) - dot_products
    }

    #[inline(always)]
    fn partition_ranking_single(dot_product: f32, point_norm: f32, leader_norm: f32) -> f32 {
        let _ = (point_norm, leader_norm);
        -dot_product
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_ranking_preserves_non_fused_arithmetic() {
        let ranking = L2::partition_ranking_single(f32::MAX, 0.0, f32::MAX);
        let fused = (-2.0f32).mul_add(f32::MAX, f32::MAX);

        assert_eq!(ranking, f32::NEG_INFINITY);
        assert_eq!(fused, -f32::MAX);
    }
}
