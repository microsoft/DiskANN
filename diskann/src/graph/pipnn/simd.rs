/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! SIMD schema for PiPNN numerical kernels.

use diskann_wide::{Architecture, SIMDFloat, SIMDMask, SIMDVector};

/// Default SIMD representation used by both PiPNN ranking kernels.
///
/// This alias is the single build-time width selection.
type DefaultVector<A> = <A as Architecture>::f32x16;

/// PiPNN SIMD representation for one architecture.
pub(super) trait PiPNNSIMDSchema: Architecture {
    /// SIMD vector used by both ranking kernels.
    type Vector: SIMDVector<Arch = Self, Scalar = f32> + SIMDFloat;

    /// Return one bit for each selected lane.
    fn active_lanes(mask: <Self::Vector as SIMDVector>::Mask) -> u64;
}

impl<A> PiPNNSIMDSchema for A
where
    A: Architecture,
    DefaultVector<A>: SIMDVector<Arch = A, Scalar = f32> + SIMDFloat,
    u64: From<
        <<<DefaultVector<A> as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying,
    >,
{
    type Vector = DefaultVector<A>;

    #[inline(always)]
    fn active_lanes(mask: <Self::Vector as SIMDVector>::Mask) -> u64 {
        u64::from(mask.bitmask().to_underlying())
    }
}
