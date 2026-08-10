/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared metric definitions for the PiPNN numerical kernels.
//!
//! `build_graph` maps each runtime [`Metric`] to one marker type. Leaf and
//! partition kernels use separate traits for that marker. Both traits use the
//! common cosine and norm functions in this module.

mod leaf;
mod partition;

pub(super) use leaf::LeafKernelMetric;
pub(super) use partition::PartitionKernelMetric;

use diskann_vector::distance::Metric;
use diskann_wide::{SIMDFloat, SIMDSelect, SIMDVector};

/// Identify one metric across all PiPNN build stages.
pub(super) trait MetricTag: Send + Sync + 'static {
    /// Runtime metric represented by this marker.
    const METRIC: Metric;
}

/// Squared-L2 marker.
pub(super) struct L2;
/// Unnormalized-cosine marker.
pub(super) struct Cosine;
/// Unit-normalized-cosine marker.
pub(super) struct CosineNormalized;
/// Negative-inner-product marker.
pub(super) struct InnerProduct;

impl MetricTag for L2 {
    const METRIC: Metric = Metric::L2;
}

impl MetricTag for Cosine {
    const METRIC: Metric = Metric::Cosine;
}

impl MetricTag for CosineNormalized {
    const METRIC: Metric = Metric::CosineNormalized;
}

impl MetricTag for InnerProduct {
    const METRIC: Metric = Metric::InnerProduct;
}

/// Convert a squared norm to a norm.
///
/// The function maps subnormal squared norms to zero. It preserves NaN so that
/// kernel comparisons do not rank an invalid value.
#[inline(always)]
pub(super) fn norm_from_squared(squared_norm: f32) -> f32 {
    if squared_norm < f32::MIN_POSITIVE {
        0.0
    } else {
        squared_norm.sqrt()
    }
}

/// Compute cosine distance with the DiskANN zero-norm and NaN rules.
///
/// Each lane contains one point pair. A zero norm produces zero similarity. A
/// NaN norm remains NaN unless the other norm is zero.
#[inline(always)]
pub(super) fn cosine_distance<F>(arch: F::Arch, dot: F, source_norm: F, target_norm: F) -> F
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

/// Compute scalar cosine distance with the DiskANN zero-norm rules.
#[inline(always)]
pub(super) fn cosine_distance_scalar(dot: f32, source_norm: f32, target_norm: f32) -> f32 {
    if source_norm < f32::MIN_POSITIVE.sqrt() || target_norm < f32::MIN_POSITIVE.sqrt() {
        1.0
    } else {
        1.0 - dot / (source_norm * target_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_from_squared_applies_zero_threshold_without_erasing_nan() {
        assert_eq!(norm_from_squared(-0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(norm_from_squared(f32::MIN_POSITIVE / 2.0), 0.0);
        assert_eq!(
            norm_from_squared(f32::MIN_POSITIVE),
            f32::MIN_POSITIVE.sqrt()
        );
        assert!(norm_from_squared(f32::NAN).is_nan());
    }
}
