/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! This module provides metric markers and shared numerical functions.

mod leaf;
mod partition;

pub(super) use leaf::LeafMetric;
pub(super) use partition::PartitionMetric;

use super::simd::PiPNNSIMDVector;

pub(super) struct L2;
pub(super) struct Cosine;
pub(super) struct CosineNormalized;
pub(super) struct InnerProduct;

/// Prepared norms for one point stripe and its sampled leaders.
#[derive(Clone, Copy, Debug)]
pub(super) struct PartitionNorms<'a> {
    pub(super) point_norms: &'a [f32],
    pub(super) leader_norms: &'a [f32],
}

/// Compute SIMD cosine distance with the DiskANN zero-norm and NaN rules.
///
/// Each lane contains one point pair. A zero norm produces zero similarity.
/// Finite similarity is clamped to the cosine range before distance conversion.
#[inline(always)]
pub(super) fn cosine_distance_simd<F>(arch: F::Arch, dot: F, source_norm: F, target_norm: F) -> F
where
    F: PiPNNSIMDVector,
{
    let zero = F::default(arch);
    let one = F::splat(arch, 1.0);
    let minimum_norm = F::splat(arch, f32::MIN_POSITIVE.sqrt());
    let source_zero = source_norm.lt_simd(minimum_norm);
    let target_zero = target_norm.lt_simd(minimum_norm);
    let denominator = source_norm * target_norm;
    let safe_denominator = F::select(source_zero, one, F::select(target_zero, one, denominator));
    let cosine = F::select(
        source_zero,
        zero,
        F::select(target_zero, zero, dot / safe_denominator),
    );
    let negative_one = F::splat(arch, -1.0);
    one - negative_one.max_simd(cosine.min_simd(one))
}

/// Compute one cosine distance with the DiskANN zero-norm and NaN rules.
#[inline(always)]
pub(super) fn cosine_distance_single(dot: f32, source_norm: f32, target_norm: f32) -> f32 {
    if source_norm < f32::MIN_POSITIVE.sqrt() || target_norm < f32::MIN_POSITIVE.sqrt() {
        1.0
    } else {
        let cosine = dot / (source_norm * target_norm);
        1.0 - (-1.0_f32).max(1.0_f32.min(cosine))
    }
}
