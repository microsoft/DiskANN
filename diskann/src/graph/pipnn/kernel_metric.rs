/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Metric formulas for the PiPNN partition and leaf kernels.
//!
//! `build_graph` maps each runtime [`Metric`] to one zero-sized marker type. The
//! partition and leaf functions receive that concrete type.
//!
//! Each formula returns an ascending score. L2 uses squared norms. Cosine uses
//! norms and maps a zero norm to zero similarity. Normalized cosine and inner
//! product do not use norms. Ordered comparisons do not rank NaN.
//!
//! The L2 partition SIMD path uses fused arithmetic. Its scalar tail uses
//! non-fused arithmetic. This operation order is part of the tie-order contract.
//!
//! [`ScaleKind`] defines the stored norm unit. [`KernelMetric`] defines the leaf
//! and partition formulas.

use diskann_vector::distance::Metric;
use diskann_wide::{SIMDFloat, SIMDSelect, SIMDVector};

/// Stored norm representation for one kernel input.
///
/// `KernelMetric` supplies this value as an associated constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScaleKind {
    /// Metric does not read this scale position.
    None,
    /// Stored value is already a squared norm.
    SquaredNorm,
    /// Stored value is a squared norm; the kernel takes its square root.
    NormFromSquared,
    /// Stored value is already a norm.
    Norm,
}

impl ScaleKind {
    /// Convert a stored norm to the unit that the kernel requires.
    ///
    /// A squared norm below `f32::MIN_POSITIVE` becomes zero. A norm below
    /// `sqrt(f32::MIN_POSITIVE)` also becomes zero. The function does not change
    /// NaN, so the kernel does not rank it.
    ///
    /// The concrete [`KernelMetric`] supplies `self` as an associated constant.
    /// The compiler selects one match arm for each kernel instance.
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

    /// Return `true` when the metric requires this norm input.
    pub(crate) const fn is_some(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Metric contract for leaf and partition selection.
///
/// `build_graph` selects one implementation. Generic calls inline its arithmetic
/// through the complete build. Leaf and partition formulas are separate because
/// L2 partition ranking does not need the point norm.
///
/// Each method returns an ascending score. Strict comparisons do not rank NaN.
/// Marker types contain no data. Associated constants remove unused norm work.
pub(crate) trait KernelMetric: Send + Sync + 'static {
    /// Runtime tag represented by this marker.
    const METRIC: Metric;
    /// Diagonal scale representation used by the leaf kernel.
    const LEAF_SCALE: ScaleKind;
    /// Point scale representation used by partition assignment.
    const PARTITION_POINT_SCALE: ScaleKind;
    /// Leader-column scale representation used by partition assignment.
    const PARTITION_LEADER_SCALE: ScaleKind;

    /// Compute SIMD distances from one leaf source to earlier targets.
    ///
    /// `dot` and `target_scale` contain one target per lane. `source_scale`
    /// contains the source norm in each lane. A metric without norms receives
    /// zero for both norm arguments. The result contains one distance per lane.
    fn leaf_distance<F>(arch: F::Arch, dot: F, source_scale: F, target_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// Compute the scalar-tail equivalent of `leaf_distance`.
    ///
    /// The inputs and result represent one SIMD lane. Each implementation
    /// documents any required operation order.
    fn leaf_distance_scalar(dot: f32, source_scale: f32, target_scale: f32) -> f32;

    /// Compute SIMD scores from one point to a group of leaders.
    ///
    /// `dot` and `leader_scale` contain one leader per lane. `point_scale`
    /// contains the point norm in each lane. A metric without a norm receives
    /// zero for that argument. The formula can omit terms that are constant for
    /// all leaders.
    fn partition_distance<F>(arch: F::Arch, dot: F, point_scale: F, leader_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// Compute the scalar-tail equivalent of `partition_distance`.
    ///
    /// The inputs and result represent one SIMD lane. Each implementation uses
    /// its documented operation order.
    fn partition_distance_scalar(dot: f32, point_scale: f32, leader_scale: f32) -> f32;
}

/// Zero-sized metric markers used only for monomorphization.
pub(crate) struct L2;
/// Unnormalized-cosine marker.
pub(crate) struct Cosine;
/// Unit-normalized-cosine marker.
pub(crate) struct CosineNormalized;
/// Negative-inner-product marker.
pub(crate) struct InnerProduct;

/// Clamp negative SIMD roundoff to zero and keep NaN lanes unchanged.
///
/// SIMD `max` has architecture-specific NaN behavior. The ordered self-test
/// selects the original value for each NaN lane.
#[inline(always)]
fn clamp_nonnegative<F>(arch: F::Arch, distance: F) -> F
where
    F: SIMDVector<Scalar = f32> + SIMDFloat,
    F::Mask: SIMDSelect<F>,
{
    let zero = F::default(arch);
    // Select the original value for NaN lanes. This gives all architectures the
    // same non-rankable NaN result.
    distance
        .eq_simd(distance)
        .select(zero.max_simd(distance), distance)
}

/// Scalar equivalent of [`clamp_nonnegative`].
#[inline(always)]
fn clamp_nonnegative_scalar(distance: f32) -> f32 {
    if distance < 0.0 { 0.0 } else { distance }
}

/// Compute cosine distance with the DiskANN zero-norm and NaN rules.
///
/// A zero-norm lane divides by one and then selects zero similarity. A NaN norm
/// propagates through division. If the other norm is zero, zero similarity takes
/// precedence.
///
/// Each input contains one point pair per lane. The result is
/// `1 - cosine_similarity`. The function uses no lane branch.
#[inline(always)]
fn cosine_distance<F>(arch: F::Arch, dot: F, source_norm: F, target_norm: F) -> F
where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
{
    let zero = F::default(arch);
    let one = F::splat(arch, 1.0);
    let minimum_norm = F::splat(arch, f32::MIN_POSITIVE.sqrt());
    let source_zero = source_norm.lt_simd(minimum_norm);
    let target_zero = target_norm.lt_simd(minimum_norm);
    let denominator = source_norm * target_norm;
    let safe_denominator = source_zero.select(one, target_zero.select(one, denominator));
    let cosine = source_zero.select(zero, target_zero.select(zero, dot / safe_denominator));
    one - cosine
}

#[inline(always)]
fn cosine_distance_scalar(dot: f32, source_norm: f32, target_norm: f32) -> f32 {
    if source_norm < f32::MIN_POSITIVE.sqrt() || target_norm < f32::MIN_POSITIVE.sqrt() {
        1.0
    } else {
        1.0 - dot / (source_norm * target_norm)
    }
}

impl KernelMetric for L2 {
    const METRIC: Metric = Metric::L2;
    const LEAF_SCALE: ScaleKind = ScaleKind::SquaredNorm;
    const PARTITION_POINT_SCALE: ScaleKind = ScaleKind::None;
    const PARTITION_LEADER_SCALE: ScaleKind = ScaleKind::SquaredNorm;

    #[inline(always)]
    fn leaf_distance<F>(arch: F::Arch, dot: F, source_scale: F, target_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        // Reconstruct squared L2 from Gram-matrix entries. Negative values can
        // arise only from floating-point roundoff, so clamp without hiding NaN.
        clamp_nonnegative(
            arch,
            source_scale + target_scale - F::splat(arch, 2.0) * dot,
        )
    }

    #[inline(always)]
    fn leaf_distance_scalar(dot: f32, source_scale: f32, target_scale: f32) -> f32 {
        // Keep scalar tail arithmetic in the same left-to-right shape.
        clamp_nonnegative_scalar(source_scale + target_scale - 2.0 * dot)
    }

    #[inline(always)]
    fn partition_distance<F>(arch: F::Arch, dot: F, _: F, leader_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        // The point norm is constant for this ranking. The SIMD path uses the
        // fused multiply-add operation that defines its tie order.
        F::splat(arch, -2.0).mul_add_simd(dot, leader_scale)
    }

    #[inline(always)]
    fn partition_distance_scalar(dot: f32, _: f32, leader_scale: f32) -> f32 {
        // The scalar tail uses non-fused subtraction. A fused operation can
        // change rounding and select a different leader at a tie.
        leader_scale - 2.0 * dot
    }
}

impl KernelMetric for Cosine {
    const METRIC: Metric = Metric::Cosine;
    const LEAF_SCALE: ScaleKind = ScaleKind::NormFromSquared;
    const PARTITION_POINT_SCALE: ScaleKind = ScaleKind::NormFromSquared;
    const PARTITION_LEADER_SCALE: ScaleKind = ScaleKind::Norm;

    #[inline(always)]
    fn leaf_distance<F>(arch: F::Arch, dot: F, source_scale: F, target_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        // Leaf output stores metric distances, so clamp negative roundoff after
        // applying zero-norm and NaN handling in `cosine_distance`.
        clamp_nonnegative(arch, cosine_distance(arch, dot, source_scale, target_scale))
    }

    #[inline(always)]
    fn leaf_distance_scalar(dot: f32, source_scale: f32, target_scale: f32) -> f32 {
        // Match the bulk path's distance clamp for the scalar tail.
        clamp_nonnegative_scalar(cosine_distance_scalar(dot, source_scale, target_scale))
    }

    #[inline(always)]
    fn partition_distance<F>(arch: F::Arch, dot: F, point_scale: F, leader_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        // Partitioning uses only score order. Do not clamp the score because a
        // clamp can change the order of near ties.
        cosine_distance(arch, dot, point_scale, leader_scale)
    }

    #[inline(always)]
    fn partition_distance_scalar(dot: f32, point_scale: f32, leader_scale: f32) -> f32 {
        // Use the same unclamped score in the scalar tail.
        cosine_distance_scalar(dot, point_scale, leader_scale)
    }
}

impl KernelMetric for CosineNormalized {
    const METRIC: Metric = Metric::CosineNormalized;
    const LEAF_SCALE: ScaleKind = ScaleKind::None;
    const PARTITION_POINT_SCALE: ScaleKind = ScaleKind::None;
    const PARTITION_LEADER_SCALE: ScaleKind = ScaleKind::None;

    #[inline(always)]
    fn leaf_distance<F>(arch: F::Arch, dot: F, _: F, _: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        // Unit-normalized inputs need no scale loads; only roundoff below zero
        // is clamped in stored leaf distances.
        clamp_nonnegative(arch, F::splat(arch, 1.0) - dot)
    }

    #[inline(always)]
    fn leaf_distance_scalar(dot: f32, _: f32, _: f32) -> f32 {
        // Scalar tail mirrors the normalized-cosine bulk formula.
        clamp_nonnegative_scalar(1.0 - dot)
    }

    #[inline(always)]
    fn partition_distance<F>(arch: F::Arch, dot: F, _: F, _: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        // Ranking needs only `1 - dot`; no norm memory is touched.
        F::splat(arch, 1.0) - dot
    }

    #[inline(always)]
    fn partition_distance_scalar(dot: f32, _: f32, _: f32) -> f32 {
        // Use the same unclamped score as the SIMD path.
        1.0 - dot
    }
}

impl KernelMetric for InnerProduct {
    const METRIC: Metric = Metric::InnerProduct;
    const LEAF_SCALE: ScaleKind = ScaleKind::None;
    const PARTITION_POINT_SCALE: ScaleKind = ScaleKind::None;
    const PARTITION_LEADER_SCALE: ScaleKind = ScaleKind::None;

    #[inline(always)]
    fn leaf_distance<F>(arch: F::Arch, dot: F, _: F, _: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        // Negation converts maximum inner product into the common ascending
        // distance order without scale loads.
        F::default(arch) - dot
    }

    #[inline(always)]
    fn leaf_distance_scalar(dot: f32, _: f32, _: f32) -> f32 {
        // Scalar tail uses the same ascending score.
        -dot
    }

    #[inline(always)]
    fn partition_distance<F>(arch: F::Arch, dot: F, _: F, _: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>,
    {
        // Partition leader ranking shares the negative-inner-product score.
        F::default(arch) - dot
    }

    #[inline(always)]
    fn partition_distance_scalar(dot: f32, _: f32, _: f32) -> f32 {
        // Scalar tail uses the same ascending score.
        -dot
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
