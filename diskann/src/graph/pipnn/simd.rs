/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The SIMD vector type of the PiPNN ranking kernels.

use diskann_wide::{Architecture, Const, SIMDFloat, SIMDMask, SIMDVector};

/// The number of distances in one SIMD group of both ranking kernels.
///
/// Change the SIMD width here only. 16 lanes fill one AVX-512 register. AVX2 uses
/// two registers for a group, and Neon uses four.
pub(super) const LANES: usize = 16;

/// Map a lane count `Const<N>` to the `f32` vector of architecture `A` with `N`
/// lanes.
pub(super) trait F32Vector<A: Architecture> {
    type Vector;
}

impl<A: Architecture> F32Vector<A> for Const<4> {
    type Vector = A::f32x4;
}

impl<A: Architecture> F32Vector<A> for Const<8> {
    type Vector = A::f32x8;
}

impl<A: Architecture> F32Vector<A> for Const<16> {
    type Vector = A::f32x16;
}

/// The `f32` vector of architecture `A` with [`LANES`] lanes.
type DefaultVector<A> = <Const<LANES> as F32Vector<A>>::Vector;

/// An architecture with an `f32` vector of [`LANES`] lanes.
///
/// The blanket implementation below covers `Scalar`, `V3`, `V4`, and `Neon`.
pub(super) trait Simd: Architecture {
    /// The vector that holds one group of [`LANES`] distances.
    type Vector: SIMDVector<Arch = Self, Scalar = f32, ConstLanes = Const<LANES>> + SIMDFloat;

    /// Convert a lane mask to an integer with bit `i` set when lane `i` is set.
    fn active_lanes(mask: <Self::Vector as SIMDVector>::Mask) -> u64;
}

// The mask must convert to `u64`, so the kernels can visit the set lanes with
// integer bit operations.
impl<A> Simd for A
where
    A: Architecture,
    DefaultVector<A>: SIMDVector<Arch = A, Scalar = f32, ConstLanes = Const<LANES>> + SIMDFloat,
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
