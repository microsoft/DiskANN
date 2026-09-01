/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! SIMD schema for PiPNN numerical kernels.

use diskann_wide::{
    Architecture, Const, SIMDFloat, SIMDMask, SIMDSelect, SIMDVector, SupportedLaneCount,
};

/// Default SIMD representation used by every PiPNN numerical stage.
///
/// This alias is the single build-time width selection.
type DefaultVector<A> = <A as Architecture>::f32x16;

type DefaultIdVector<A> = <<DefaultVector<A> as SIMDVector>::ConstLanes as IdVectorFor<A>>::Vector;

pub(super) trait IdVectorFor<A>
where
    A: Architecture,
{
    type Vector;
}

impl<A> IdVectorFor<A> for Const<8>
where
    A: Architecture,
{
    type Vector = A::u32x8;
}

impl<A> IdVectorFor<A> for Const<16>
where
    A: Architecture,
{
    type Vector = A::u32x16;
}

/// Operations required by PiPNN SIMD vectors.
pub(super) trait PiPNNSIMDVector:
    SIMDVector<Scalar = f32> + SIMDFloat + std::ops::Div<Output = Self>
{
    /// Convert the SIMD value to a readable lane array.
    fn to_lane_array(self) -> impl AsRef<[f32]>;

    /// Return one bit for each selected lane.
    fn active_lanes(mask: Self::Mask) -> u64;

    /// Select one value from each pair of lanes.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;
}

impl<F, const N: usize> PiPNNSIMDVector for F
where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<N>> + SIMDFloat + std::ops::Div<Output = F>,
    Const<N>: SupportedLaneCount,
    F::Mask: SIMDSelect<F>,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    #[inline(always)]
    fn to_lane_array(self) -> impl AsRef<[f32]> {
        let values: [f32; N] = self.to_array();
        values
    }

    #[inline(always)]
    fn active_lanes(mask: Self::Mask) -> u64 {
        u64::from(mask.bitmask().to_underlying())
    }

    #[inline(always)]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self {
        mask.select(if_true, if_false)
    }
}

/// PiPNN SIMD representation for one architecture.
pub(super) trait PiPNNSIMDSchema: Architecture {
    /// SIMD score vector used by every numerical stage.
    type Vector: PiPNNSIMDVector<Arch = Self> + Send;
    /// SIMD ID vector paired lane-for-lane with [`Self::Vector`].
    type IdVector: SIMDVector<Arch = Self, Scalar = u32> + Send;

    /// Select exact leader IDs with a score comparison mask.
    fn select_ids(
        mask: <Self::Vector as SIMDVector>::Mask,
        if_true: Self::IdVector,
        if_false: Self::IdVector,
    ) -> Self::IdVector;
}

impl<A> PiPNNSIMDSchema for A
where
    A: Architecture,
    DefaultVector<A>: PiPNNSIMDVector<Arch = A> + Send,
    DefaultIdVector<A>: SIMDVector<
            Arch = A,
            Scalar = u32,
            ConstLanes = <DefaultVector<A> as SIMDVector>::ConstLanes,
        > + Send,
    <DefaultIdVector<A> as SIMDVector>::Mask: SIMDSelect<DefaultIdVector<A>>
        + From<<<DefaultVector<A> as SIMDVector>::Mask as SIMDMask>::BitMask>,
{
    type Vector = DefaultVector<A>;
    type IdVector = DefaultIdVector<A>;

    #[inline(always)]
    fn select_ids(
        mask: <Self::Vector as SIMDVector>::Mask,
        if_true: Self::IdVector,
        if_false: Self::IdVector,
    ) -> Self::IdVector {
        <Self::IdVector as SIMDVector>::Mask::from(mask.bitmask()).select(if_true, if_false)
    }
}
