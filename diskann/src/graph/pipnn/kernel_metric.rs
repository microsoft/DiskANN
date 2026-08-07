/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Metric formulas shared by PiPNN partition and leaf kernels.
//!
//! The build boundary converts runtime [`Metric`] into one zero-sized marker
//! type. That concrete type is carried through partition and leaf construction,
//! so scalar and SIMD hot loops contain neither metric matches nor trait objects.
//!
//! All formulas produce ascending scores. L2 uses squared norms; unnormalized
//! cosine uses norms with zero/subnormal inputs mapped to zero similarity;
//! normalized cosine and inner product need no scales. Ordered comparisons leave
//! NaN non-rankable. The L2 partition scalar tail deliberately keeps its
//! non-fused operation order because rounding can change leader ties.
//!
//! [`ScaleKind`] records required scale representation and [`KernelMetric`]
//! owns the leaf and partition formulas. Runtime selection belongs to the build
//! entry point; this module contains no dispatch or type erasure.

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
    /// DiskANN treats squared norms below `f32::MIN_POSITIVE`, and norms below
    /// `sqrt(f32::MIN_POSITIVE)`, as zero before division. Ordered comparisons
    /// intentionally leave NaN unchanged so later distance comparisons keep it
    /// non-rankable.
    ///
    /// `stored` is interpreted according to `self`. The return value is zero,
    /// the original norm, the original squared norm, or its square root. This
    /// operation is constant-time and normally specializes to one match arm
    /// because `ScaleKind` comes from a [`KernelMetric`] associated constant.
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

    /// Return whether callers must supply this scale position.
    ///
    /// Calls use an associated constant, so this test compiles out of hot loops.
    pub(crate) const fn is_some(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Concrete metric contract shared by leaf and partition hot loops.
///
/// Runtime `Metric` is converted to one implementor at the build boundary.
/// Generic methods then inline metric arithmetic through the complete partition
/// and leaf stages. Those operations remain separate because L2 partition
/// ranking deliberately omits the point norm.
///
/// All methods return scores ordered from nearest to farthest. Implementations
/// follow the module-level zero/NaN contract; caller-side strict comparisons
/// leave scores that remain NaN non-rankable. Marker types carry no data;
/// associated scale constants and forced inlining
/// remove metric branches from dispatched loops.
pub(crate) trait KernelMetric: Send + Sync + 'static {
    /// Runtime tag represented by this marker.
    const METRIC: Metric;
    /// Diagonal scale representation used by the leaf kernel.
    const LEAF_SCALE: ScaleKind;
    /// Point scale representation used by partition assignment.
    const PARTITION_POINT_SCALE: ScaleKind;
    /// Leader-column scale representation used by partition assignment.
    const PARTITION_LEADER_SCALE: ScaleKind;

    /// SIMD distance for one leaf source against a lane group of earlier targets.
    ///
    /// `arch` is the selected architecture token. `dot` and `target_scale` hold
    /// one target per lane; `source_scale` broadcasts the source scale. Scale
    /// arguments are zero when [`Self::LEAF_SCALE`] is [`ScaleKind::None`]. The
    /// return value contains one ascending-order distance per lane.
    fn leaf_distance<F>(arch: F::Arch, dot: F, source_scale: F, target_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// Scalar-tail equivalent of `leaf_distance`.
    ///
    /// Inputs and return value represent one SIMD lane. Operation order is part
    /// of graph determinism where an implementation documents it.
    fn leaf_distance_scalar(dot: f32, source_scale: f32, target_scale: f32) -> f32;

    /// SIMD ranking score for one point against a lane group of leaders.
    ///
    /// `arch` is the selected architecture token. `dot` and `leader_scale` hold
    /// one leader per lane; `point_scale` broadcasts one point scale. Scale
    /// arguments are zero when the corresponding associated kind is
    /// [`ScaleKind::None`]. The return value contains one ascending-order score
    /// per lane; point-constant terms may be omitted.
    fn partition_distance<F>(arch: F::Arch, dot: F, point_scale: F, leader_scale: F) -> F
    where
        F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
        F::Mask: SIMDSelect<F>;

    /// Scalar-tail equivalent of `partition_distance`.
    ///
    /// Inputs and return value represent one SIMD lane. Implementations preserve
    /// any documented non-fused order used by existing graph builds.
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

/// Clamp negative SIMD roundoff to zero while preserving NaN lanes.
///
/// One ordered self-comparison normalizes backend-specific SIMD `max` NaN
/// behavior; no lane branches or allocations are introduced.
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

/// Scalar equivalent of [`clamp_nonnegative`].
#[inline(always)]
fn clamp_nonnegative_scalar(distance: f32) -> f32 {
    if distance < 0.0 { 0.0 } else { distance }
}

/// Compute cosine distance while preserving DiskANN zero/NaN semantics.
///
/// Zero lanes divide by one only to keep the operation defined, then explicitly
/// select zero similarity. A NaN norm fails its own zero comparison and
/// propagates through division unless the other endpoint takes the zero-norm
/// path; in that case zero similarity takes precedence.
///
/// `dot`, `source_norm`, and `target_norm` each contain one pair per lane. The
/// return value is `1 - cosine_similarity`. All lane handling is branchless.
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
        // Point norm is constant for this ranking. Bulk lanes retain the
        // historical fused multiply-add used by partition assignment.
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
        // Partitioning consumes only score order, so no post-formula clamp is
        // needed; omitting it preserves existing near-tie behavior.
        cosine_distance(arch, dot, point_scale, leader_scale)
    }

    #[inline(always)]
    fn partition_distance_scalar(dot: f32, point_scale: f32, leader_scale: f32) -> f32 {
        // Preserve the same unclamped ranking score in the scalar tail.
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
        // Preserve the unclamped ranking score used by full SIMD groups.
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
