/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! This module defines metric formulas for partition-leader ranking.

use diskann_wide::{SIMDFloat, SIMDSelect, SIMDVector};

use super::{
    Cosine, CosineNormalized, InnerProduct, L2, MetricTag, cosine_distance, cosine_distance_scalar,
};

/// This trait defines metric operations for partition-leader ranking.
///
/// Each function returns an ascending score. L2 receives a squared leader norm.
/// Cosine receives point and leader norms. Dot-only metrics receive zero norms.
pub(in super::super) trait PartitionKernelMetric: MetricTag {
    /// This value is true when the metric reads a point norm.
    const USES_POINT_NORM: bool;

    /// This value is true when the metric reads leader norm values.
    const USES_LEADER_NORMS: bool;

    /// This function converts one squared point norm to the required unit.
    fn prepare_point_norm(squared_norm: f32) -> f32;

    /// This function computes SIMD ranking scores for one point and multiple leaders.
    fn partition_ranking<F>(arch: F::Arch, dot: F, point_norm: F, leader_norm: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// This function computes one scalar-tail partition ranking score.
    fn partition_ranking_scalar(dot: f32, point_norm: f32, leader_norm: f32) -> f32;
}

impl PartitionKernelMetric for L2 {
    const USES_POINT_NORM: bool = false;
    const USES_LEADER_NORMS: bool = true;

    #[inline(always)]
    fn prepare_point_norm(_: f32) -> f32 {
        0.0
    }

    #[inline(always)]
    fn partition_ranking<F>(arch: F::Arch, dot: F, _: F, leader_norm: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        // The point norm is constant for all leaders. Fused arithmetic defines
        // the ranking order for complete SIMD groups.
        F::splat(arch, -2.0).mul_add_simd(dot, leader_norm)
    }

    #[inline(always)]
    fn partition_ranking_scalar(dot: f32, _: f32, leader_norm: f32) -> f32 {
        // Non-fused arithmetic defines the ranking order for the scalar tail.
        leader_norm - 2.0 * dot
    }
}

impl PartitionKernelMetric for Cosine {
    const USES_POINT_NORM: bool = true;
    const USES_LEADER_NORMS: bool = true;

    #[inline(always)]
    fn prepare_point_norm(squared_norm: f32) -> f32 {
        super::norm_from_squared(squared_norm)
    }

    #[inline(always)]
    fn partition_ranking<F>(arch: F::Arch, dot: F, point_norm: F, leader_norm: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        cosine_distance(arch, dot, point_norm, leader_norm)
    }

    #[inline(always)]
    fn partition_ranking_scalar(dot: f32, point_norm: f32, leader_norm: f32) -> f32 {
        cosine_distance_scalar(dot, point_norm, leader_norm)
    }
}

impl PartitionKernelMetric for CosineNormalized {
    const USES_POINT_NORM: bool = false;
    const USES_LEADER_NORMS: bool = false;

    #[inline(always)]
    fn prepare_point_norm(_: f32) -> f32 {
        0.0
    }

    #[inline(always)]
    fn partition_ranking<F>(arch: F::Arch, dot: F, _: F, _: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        F::splat(arch, 1.0) - dot
    }

    #[inline(always)]
    fn partition_ranking_scalar(dot: f32, _: f32, _: f32) -> f32 {
        1.0 - dot
    }
}

impl PartitionKernelMetric for InnerProduct {
    const USES_POINT_NORM: bool = false;
    const USES_LEADER_NORMS: bool = false;

    #[inline(always)]
    fn prepare_point_norm(_: f32) -> f32 {
        0.0
    }

    #[inline(always)]
    fn partition_ranking<F>(arch: F::Arch, dot: F, _: F, _: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        F::default(arch) - dot
    }

    #[inline(always)]
    fn partition_ranking_scalar(dot: f32, _: f32, _: f32) -> f32 {
        -dot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_scalar_ranking_preserves_non_fused_rounding() {
        let scalar = L2::partition_ranking_scalar(f32::MAX, 0.0, f32::MAX);
        let fused = (-2.0f32).mul_add(f32::MAX, f32::MAX);

        assert_eq!(scalar, f32::NEG_INFINITY);
        assert_eq!(fused, -f32::MAX);
    }
}
