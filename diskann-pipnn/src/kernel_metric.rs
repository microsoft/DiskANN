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

/// Stored scale representation consumed by one kernel position.
///
/// Associated constants on `KernelMetric` let the compiler remove unused scale
/// loads and allocations after metric selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScaleKind {
    /// Metric does not read this scale position.
    None,
    /// Stored value is already a squared norm.
    SquaredNorm,
    /// Stored value is a squared norm that must become a norm.
    NormFromSquared,
    /// Stored value is already a norm.
    Norm,
}

impl ScaleKind {
    /// Convert stored scale to the arithmetic form required by a kernel.
    ///
    /// DiskANN treats subnormal squared norms, and corresponding subnormal
    /// norms, as zero before division. Ordered comparisons intentionally leave
    /// NaN unchanged so later distance comparisons keep it non-rankable.
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

/// Concrete metric contract shared by leaf and partition hot loops.
///
/// Runtime `Metric` is converted to one implementor before final type erasure.
/// Generic methods then inline metric arithmetic into the architecture-specific
/// function pointer. Leaf and partition operations remain separate because L2
/// partition ranking deliberately omits the row norm.
pub(crate) trait KernelMetric: Send + Sync + 'static {
    /// Runtime tag represented by this marker.
    const METRIC: Metric;
    /// Diagonal scale representation used by the leaf kernel.
    const LEAF_SCALE: ScaleKind;
    /// Point-row scale representation used by partition assignment.
    const PARTITION_ROW_SCALE: ScaleKind;
    /// Leader-column scale representation used by partition assignment.
    const PARTITION_LEADER_SCALE: ScaleKind;

    /// SIMD distance for one leaf row against a lane group of earlier points.
    fn leaf_distance<F>(arch: F::Arch, dot: F, row_scale: F, column_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// Scalar-tail equivalent of `leaf_distance`.
    fn leaf_distance_scalar(dot: f32, row_scale: f32, column_scale: f32) -> f32;

    /// SIMD ranking score for one point row against a lane group of leaders.
    fn partition_distance<F>(arch: F::Arch, dot: F, row_scale: F, leader_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// Scalar-tail equivalent of `partition_distance`.
    fn partition_distance_scalar(dot: f32, row_scale: f32, leader_scale: f32) -> f32;
}

/// Zero-sized metric markers used only for monomorphization.
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

/// Compute cosine distance while preserving DiskANN zero/NaN semantics.
///
/// Zero lanes divide by one only to keep the operation defined, then explicitly
/// select zero similarity. NaN norms fail the zero comparison and propagate
/// through division, leaving the final distance non-rankable.
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

/// BYO-type-erasure visitor for runtime metric selection.
///
/// The visitor receives concrete `M`, allowing architecture and width wrappers
/// to compose with metric arithmetic before producing the final function pointer.
/// This avoids a nested metric trait object inside architecture dispatch.
pub(crate) trait MetricVisitor {
    /// Final caller-selected erased representation.
    type Output;

    /// Consume the visitor with one concrete metric marker.
    fn visit<M: KernelMetric>(self) -> Self::Output;
}

/// Visit the concrete marker represented by a runtime metric tag.
pub(crate) fn visit_metric<V: MetricVisitor>(metric: Metric, visitor: V) -> V::Output {
    match metric {
        Metric::L2 => visitor.visit::<L2>(),
        Metric::Cosine => visitor.visit::<Cosine>(),
        Metric::CosineNormalized => visitor.visit::<CosineNormalized>(),
        Metric::InnerProduct => visitor.visit::<InnerProduct>(),
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
