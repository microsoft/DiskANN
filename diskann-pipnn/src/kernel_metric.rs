/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Metric marker types shared by the partition and leaf kernels.
//!
//! Runtime metric selection happens only while preparing a dispatched kernel.
//! The hot loops receive a concrete marker type, allowing metric arithmetic and
//! scale handling to inline without a per-row or per-chunk enum match.

use diskann_vector::distance::Metric;
use diskann_wide::{SIMDFloat, SIMDSelect, SIMDVector};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScaleKind {
    None,
    SquaredNorm,
    NormFromSquared,
    Norm,
}

impl ScaleKind {
    #[inline(always)]
    pub(crate) fn transform(self, stored: f32) -> f32 {
        match self {
            Self::None => 0.0,
            Self::SquaredNorm => stored,
            Self::Norm => {
                if stored < f32::MIN_POSITIVE.sqrt() {
                    0.0
                } else {
                    stored
                }
            }
            Self::NormFromSquared => {
                if stored < f32::MIN_POSITIVE {
                    0.0
                } else {
                    stored.sqrt()
                }
            }
        }
    }

    pub(crate) const fn is_some(self) -> bool {
        !matches!(self, Self::None)
    }
}

pub(crate) trait KernelMetric: Send + Sync + 'static {
    const METRIC: Metric;
    const LEAF_SCALE: ScaleKind;
    const PARTITION_ROW_SCALE: ScaleKind;
    const PARTITION_LEADER_SCALE: ScaleKind;

    fn leaf_distance<F>(arch: F::Arch, dot: F, row_scale: F, column_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    fn leaf_distance_scalar(dot: f32, row_scale: f32, column_scale: f32) -> f32;

    fn partition_distance<F>(arch: F::Arch, dot: F, row_scale: F, leader_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    fn partition_distance_scalar(dot: f32, row_scale: f32, leader_scale: f32) -> f32;
}

pub(crate) struct L2;
pub(crate) struct Cosine;
pub(crate) struct CosineNormalized;
pub(crate) struct InnerProduct;

#[inline(always)]
fn clamp_nonnegative<F>(arch: F::Arch, distance: F) -> F
where
    F: SIMDVector<Scalar = f32> + SIMDFloat,
    F::Mask: SIMDSelect<F>,
{
    let zero = F::default(arch);
    // SIMD max has ISA-specific NaN behavior. Select the original NaN so it
    // remains non-rankable on every backend.
    distance
        .eq_simd(distance)
        .select(zero.max_simd(distance), distance)
}

#[inline(always)]
fn clamp_nonnegative_scalar(distance: f32) -> f32 {
    if distance < 0.0 {
        0.0
    } else {
        distance
    }
}

#[inline(always)]
fn cosine_distance<F>(arch: F::Arch, dot: F, row_norm: F, column_norm: F) -> F
where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
{
    let zero = F::default(arch);
    let one = F::splat(arch, 1.0);
    let minimum_norm = F::splat(arch, f32::MIN_POSITIVE.sqrt());
    let row_zero = row_norm.lt_simd(minimum_norm);
    let column_zero = column_norm.lt_simd(minimum_norm);
    let denominator = row_norm * column_norm;
    let safe_denominator = row_zero.select(one, column_zero.select(one, denominator));
    let cosine = row_zero.select(zero, column_zero.select(zero, dot / safe_denominator));
    one - cosine
}

#[inline(always)]
fn cosine_distance_scalar(dot: f32, row_norm: f32, column_norm: f32) -> f32 {
    if row_norm < f32::MIN_POSITIVE.sqrt() || column_norm < f32::MIN_POSITIVE.sqrt() {
        1.0
    } else {
        1.0 - dot / (row_norm * column_norm)
    }
}

impl KernelMetric for L2 {
    const METRIC: Metric = Metric::L2;
    const LEAF_SCALE: ScaleKind = ScaleKind::SquaredNorm;
    const PARTITION_ROW_SCALE: ScaleKind = ScaleKind::None;
    const PARTITION_LEADER_SCALE: ScaleKind = ScaleKind::SquaredNorm;

    #[inline(always)]
    fn leaf_distance<F>(arch: F::Arch, dot: F, row_scale: F, column_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        clamp_nonnegative(arch, row_scale + column_scale - F::splat(arch, 2.0) * dot)
    }

    #[inline(always)]
    fn leaf_distance_scalar(dot: f32, row_scale: f32, column_scale: f32) -> f32 {
        clamp_nonnegative_scalar(row_scale + column_scale - 2.0 * dot)
    }

    #[inline(always)]
    fn partition_distance<F>(arch: F::Arch, dot: F, _: F, leader_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        F::splat(arch, -2.0).mul_add_simd(dot, leader_scale)
    }

    #[inline(always)]
    fn partition_distance_scalar(dot: f32, _: f32, leader_scale: f32) -> f32 {
        // Preserve the scalar reduction shape used by the original partition
        // kernel; changing this rounding can change leader tie order.
        leader_scale - 2.0 * dot
    }
}

impl KernelMetric for Cosine {
    const METRIC: Metric = Metric::Cosine;
    const LEAF_SCALE: ScaleKind = ScaleKind::NormFromSquared;
    const PARTITION_ROW_SCALE: ScaleKind = ScaleKind::NormFromSquared;
    const PARTITION_LEADER_SCALE: ScaleKind = ScaleKind::Norm;

    #[inline(always)]
    fn leaf_distance<F>(arch: F::Arch, dot: F, row_scale: F, column_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        clamp_nonnegative(arch, cosine_distance(arch, dot, row_scale, column_scale))
    }

    #[inline(always)]
    fn leaf_distance_scalar(dot: f32, row_scale: f32, column_scale: f32) -> f32 {
        clamp_nonnegative_scalar(cosine_distance_scalar(dot, row_scale, column_scale))
    }

    #[inline(always)]
    fn partition_distance<F>(arch: F::Arch, dot: F, row_scale: F, leader_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        cosine_distance(arch, dot, row_scale, leader_scale)
    }

    #[inline(always)]
    fn partition_distance_scalar(dot: f32, row_scale: f32, leader_scale: f32) -> f32 {
        cosine_distance_scalar(dot, row_scale, leader_scale)
    }
}

impl KernelMetric for CosineNormalized {
    const METRIC: Metric = Metric::CosineNormalized;
    const LEAF_SCALE: ScaleKind = ScaleKind::None;
    const PARTITION_ROW_SCALE: ScaleKind = ScaleKind::None;
    const PARTITION_LEADER_SCALE: ScaleKind = ScaleKind::None;

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

    #[inline(always)]
    fn partition_distance<F>(arch: F::Arch, dot: F, _: F, _: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        F::splat(arch, 1.0) - dot
    }

    #[inline(always)]
    fn partition_distance_scalar(dot: f32, _: f32, _: f32) -> f32 {
        1.0 - dot
    }
}

impl KernelMetric for InnerProduct {
    const METRIC: Metric = Metric::InnerProduct;
    const LEAF_SCALE: ScaleKind = ScaleKind::None;
    const PARTITION_ROW_SCALE: ScaleKind = ScaleKind::None;
    const PARTITION_LEADER_SCALE: ScaleKind = ScaleKind::None;

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

    #[inline(always)]
    fn partition_distance<F>(arch: F::Arch, dot: F, _: F, _: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        F::default(arch) - dot
    }

    #[inline(always)]
    fn partition_distance_scalar(dot: f32, _: f32, _: f32) -> f32 {
        -dot
    }
}

pub(crate) trait EraseMetric {
    type Output;

    fn erase<M: KernelMetric>(self) -> Self::Output;
}

pub(crate) fn erase_metric<E: EraseMetric>(metric: Metric, erase: E) -> E::Output {
    match metric {
        Metric::L2 => erase.erase::<L2>(),
        Metric::Cosine => erase.erase::<Cosine>(),
        Metric::CosineNormalized => erase.erase::<CosineNormalized>(),
        Metric::InnerProduct => erase.erase::<InnerProduct>(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_scales_apply_zero_threshold_without_erasing_nan() {
        assert_eq!(ScaleKind::Norm.transform(-0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(
            ScaleKind::Norm.transform(f32::MIN_POSITIVE.sqrt() / 2.0),
            0.0
        );
        assert_eq!(
            ScaleKind::NormFromSquared.transform(f32::MIN_POSITIVE / 2.0),
            0.0
        );
        assert_eq!(
            ScaleKind::NormFromSquared.transform(f32::MIN_POSITIVE),
            f32::MIN_POSITIVE.sqrt()
        );
        assert!(ScaleKind::Norm.transform(f32::NAN).is_nan());
        assert!(ScaleKind::NormFromSquared.transform(f32::NAN).is_nan());
    }

    #[test]
    fn l2_partition_scalar_tail_preserves_non_fused_rounding() {
        let scalar = L2::partition_distance_scalar(f32::MAX, 0.0, f32::MAX);
        let fused = (-2.0f32).mul_add(f32::MAX, f32::MAX);

        assert_eq!(scalar, f32::NEG_INFINITY);
        assert_eq!(fused, -f32::MAX);
    }
}
