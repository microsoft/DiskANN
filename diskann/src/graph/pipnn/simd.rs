/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! SIMD schema for PiPNN numerical kernels.

use diskann_wide::{Architecture, SIMDFloat, SIMDMask, SIMDSelect, SIMDVector};

/// Default SIMD representation used by every PiPNN numerical stage.
///
/// This alias is the single build-time width selection.
type DefaultVector<A> = <A as Architecture>::f32x16;

/// Operations required by PiPNN SIMD vectors.
pub(super) trait PiPNNSIMDVector:
    SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = Self>
{
    /// Return one bit for each selected lane.
    fn active_lanes(mask: Self::Mask) -> u64;

    /// Select one value from each pair of lanes.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;
}

impl<F> PiPNNSIMDVector for F
where
    F: SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = F>,
    F::Mask: SIMDSelect<F>,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    #[inline(always)]
    fn active_lanes(mask: Self::Mask) -> u64 {
        u64::from(mask.bitmask().to_underlying())
    }

    #[inline(always)]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        mask.select(if_true, if_false)
    }
}

/// Stage-specific SIMD representations for one architecture.
pub(super) trait PiPNNSIMDSchema: Architecture {
    /// SIMD representation for leaf distance scores.
    type LeafScore: PiPNNSIMDVector<Arch = Self>;
    /// SIMD representation for partition ranking scores.
    type PartitionScore: PiPNNSIMDVector<Arch = Self>;
    /// SIMD representation for relative-hash sketch comparisons.
    type HashScore: PiPNNSIMDVector<Arch = Self>;
}

impl<A> PiPNNSIMDSchema for A
where
    A: Architecture,
    DefaultVector<A>: PiPNNSIMDVector<Arch = A>,
{
    type LeafScore = DefaultVector<A>;
    type PartitionScore = DefaultVector<A>;
    type HashScore = DefaultVector<A>;
}
