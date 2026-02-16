/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

//! # Masks
//!
//! Neon masks are bit-width specific, but type agnostic (meaning the mask representation
//! for `u32x4` and `f32x4` are the same).
//!
//! The representation for a type with bitwidth `B` and `L` lanes is a SIMD register
//! containing `L` lanes of `B`-bit unsigned integers.
//!
//! Within each lane, a mask is "set" is all `B` bits of the corresponding integer are 1,
//! and unset if all `B` bits are 0. These masks are automatically generated in this form
//! by the various compare intrinsics.
//!
//! Setting all bits is important because the `select` operations in Neon are bit-wise
//! selects, unlike AVX2 where only the most-significant bit is important.
//!
//! The conversion implementation in this file still refer to the uppermost bit when
//! implementing `move_mask`-like functionality.

use crate::{BitMask, FromInt, SIMDMask};

use super::Neon;

use std::arch::aarch64::*;

macro_rules! define_mask {
    ($mask:ident, $repr:ident, $lanes:literal, $arch:ty) => {
        #[derive(Debug, Clone, Copy)]
        #[allow(non_camel_case_types)]
        #[repr(transparent)]
        pub struct $mask(pub(crate) $repr);

        impl SIMDMask for $mask {
            type Arch = $arch;
            type Underlying = $repr;
            type BitMask = BitMask<$lanes, $arch>;
            const ISBITS: bool = false;
            const LANES: usize = $lanes;

            #[inline(always)]
            fn arch(self) -> Self::Arch {
                // SAFETY: Since `self` cannot be safely constructed without its `Arch`,
                // it's safe to construct the arch.
                unsafe { <$arch>::new() }
            }

            #[inline(always)]
            fn to_underlying(self) -> Self::Underlying {
                self.0
            }

            #[inline(always)]
            fn from_underlying(_arch: $arch, value: Self::Underlying) -> Self {
                Self(value)
            }

            #[inline(always)]
            fn keep_first(arch: $arch, lanes: usize) -> Self {
                Self(<$repr as MaskOps>::keep_first(arch, lanes))
            }

            #[inline(always)]
            fn get_unchecked(&self, i: usize) -> bool {
                <$repr as MaskOps>::move_mask(self.0, self.arch()).get_unchecked(i)
            }
        }

        impl From<BitMask<$lanes, $arch>> for $mask {
            #[inline(always)]
            fn from(mask: BitMask<$lanes, $arch>) -> Self {
                Self(<$repr as MaskOps>::from_mask(mask))
            }
        }

        impl From<$mask> for BitMask<$lanes, $arch> {
            #[inline(always)]
            fn from(mask: $mask) -> BitMask<$lanes, $arch> {
                <$repr as MaskOps>::move_mask(mask.0, mask.arch())
            }
        }
    };
}

define_mask!(mask8x8, uint8x8_t, 8, Neon);
define_mask!(mask8x16, uint8x16_t, 16, Neon);
define_mask!(mask16x4, uint16x4_t, 4, Neon);
define_mask!(mask16x8, uint16x8_t, 8, Neon);
define_mask!(mask32x2, uint32x2_t, 2, Neon);
define_mask!(mask32x4, uint32x4_t, 4, Neon);
define_mask!(mask64x1, uint64x1_t, 1, Neon);
define_mask!(mask64x2, uint64x2_t, 2, Neon);

/////////////
// MaskOps //
/////////////

trait MaskOps: Sized {
    type BitMask: SIMDMask<Arch = Neon>;
    type Array;

    /// Convert `self` into a BitMask.
    fn move_mask(self, arch: Neon) -> Self::BitMask;

    /// Construct `Self` from a BitMask.
    fn from_mask(mask: Self::BitMask) -> Self;

    /// Convert `self` to an array.
    #[cfg(any(test, miri))]
    fn to_array(self) -> Self::Array;

    /// Construct `Self` by only keeping up to the first `lanes` lanes.
    #[inline(always)]
    fn keep_first(arch: Neon, lanes: usize) -> Self {
        Self::from_mask(Self::BitMask::keep_first(arch, lanes))
    }
}

// Two approaches are used for `move_mask` depending on lane count:
//
// * For types with few lanes (≤4), we use a shift-right-accumulate (USRA) chain:
//   normalize each lane to 0/1 via a right-shift, then progressively fold adjacent bits
//   together by reinterpreting at wider lane widths and using USRA. This requires zero
//   constants and no horizontal reductions.
//
// * For types with many lanes (8+), we use the MSB-isolate + variable-shift + horizontal-add
//   approach: mask the upper-most bit of each lane, perform a variable shift to place each
//   retained bit in a unique position, then finish with a horizontal sum to concatenate
//   all bits together.

impl MaskOps for uint8x8_t {
    type BitMask = BitMask<8, Neon>;
    type Array = [u8; 8];

    #[inline(always)]
    fn move_mask(self, arch: Neon) -> Self::BitMask {
        cfg_if::cfg_if! {
            if #[cfg(miri)] {
                let array = self.to_array();
                BitMask::from_fn(arch, |i| array[i] == u8::MAX)
            } else {
                // SAFETY: Inclusion of this function is dependent on the "neon" target
                // feature. This function does not access memory directly.
                let value = unsafe {
                    let mask =  vmov_n_u8(0x80);
                    // Effectively creates [-7, -6, -5, -4, -3, -2, -1, 0]
                    let shifts = vcreate_s8(0x00FF_FEFD_FCFB_FAF9);
                    vaddlv_u8(vshl_u8(vand_u8(self, mask), shifts))
                };
                BitMask::from_int(arch, value as u8)
            }
        }
    }

    #[inline(always)]
    fn from_mask(mask: Self::BitMask) -> Self {
        const BIT_SELECTOR: u64 = 0x8040201008040201;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe { vtst_u8(vmov_n_u8(mask.0), vcreate_u8(BIT_SELECTOR)) }
    }

    #[inline(always)]
    fn keep_first(_arch: Neon, lanes: usize) -> Self {
        const INDICES: u64 = 0x0706050403020100;
        let n = lanes.min(8) as u8;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe { vclt_u8(vcreate_u8(INDICES), vmov_n_u8(n)) }
    }

    #[cfg(any(test, miri))]
    fn to_array(self) -> Self::Array {
        // SAFETY: Both the source and destination types are trivially destructible and are
        // valid for all possible bit-patterns.
        unsafe { std::mem::transmute::<Self, Self::Array>(self) }
    }
}

impl MaskOps for uint8x16_t {
    type BitMask = BitMask<16, Neon>;
    type Array = [u8; 16];

    #[inline(always)]
    fn move_mask(self, arch: Neon) -> Self::BitMask {
        cfg_if::cfg_if! {
            if #[cfg(miri)] {
                let array = self.to_array();
                BitMask::from_fn(arch, |i| array[i] == u8::MAX)
            } else {
                // SAFETY: Inclusion of this function is dependent on the "neon" target
                // feature. This function does not access memory directly.
                let value = unsafe {
                    let mask = vmovq_n_u8(0x80);
                    let masked = vandq_u8(self, mask);
                    // Effectively creates [-7, -6, -5, -4, -3, -2, -1, 0]
                    let shifts = vcreate_s8(0x00FF_FEFD_FCFB_FAF9);

                    let low = vaddlv_u8(vshl_u8(vget_low_u8(masked), shifts));
                    let high = vaddlv_u8(vshl_u8(vget_high_u8(masked), shifts));

                    low | (high << 8)
                };
                BitMask::from_int(arch, value)
            }
        }
    }

    #[inline(always)]
    fn from_mask(mask: Self::BitMask) -> Self {
        let mask: u16 = mask.0;
        const BIT_SELECTOR: u64 = 0x8040201008040201;

        let low = mask as u8;
        let high = (mask >> 8) as u8;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe {
            vtstq_u8(
                vcombine_u8(vmov_n_u8(low), vmov_n_u8(high)),
                vcombine_u8(vcreate_u8(BIT_SELECTOR), vcreate_u8(BIT_SELECTOR)),
            )
        }
    }

    #[inline(always)]
    fn keep_first(_arch: Neon, lanes: usize) -> Self {
        const LO: u64 = 0x0706050403020100;
        const HI: u64 = 0x0F0E0D0C0B0A0908;
        let n = lanes.min(16) as u8;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe { vcltq_u8(vcombine_u8(vcreate_u8(LO), vcreate_u8(HI)), vmovq_n_u8(n)) }
    }

    #[cfg(any(test, miri))]
    fn to_array(self) -> Self::Array {
        // SAFETY: Both the source and destination types are trivially destructible and are
        // valid for all possible bit-patterns.
        unsafe { std::mem::transmute::<Self, Self::Array>(self) }
    }
}

impl MaskOps for uint16x4_t {
    type BitMask = BitMask<4, Neon>;
    type Array = [u16; 4];

    #[inline(always)]
    fn move_mask(self, arch: Neon) -> Self::BitMask {
        cfg_if::cfg_if! {
            if #[cfg(miri)] {
                let array = self.to_array();
                BitMask::from_fn(arch, |i| array[i] == u16::MAX)
            } else {
                // Step 1: Isolate single bits in each lane and compact to bytes:
                //
                // |    Lane 0   |    Lane 1   |    Lane 2   |    Lane 3   |
                // | 0b0000'000a | 0b0000'000b | 0b0000'000c | 0b0000'000d |
                //
                // Step 2: Shift the even lanes and then add with the odd lanes:
                //
                // |          Lane 0       |          Lane 1       |
                // | 0b0000'0000'0000'00ab | 0b0000'0000'0000'00cd |
                //
                // Step 3: Shift the even lane and add with the odd lane.
                //
                // | 0b0000'0000'0000'0000'b0000'0000'0000'abcd |
                //
                // Thus, everything gets compressed down to 4-bits.
                //
                // SAFETY: Inclusion of this function is dependent on the "neon" target
                // feature. This function does not access memory directly.
                let value = unsafe {
                    let bits = vshr_n_u16(self, 15);
                    let paired = vsra_n_u32(
                        vreinterpret_u32_u16(bits),
                        vreinterpret_u32_u16(bits),
                        15,
                    );
                    let packed = vsra_n_u64(
                        vreinterpret_u64_u32(paired),
                        vreinterpret_u64_u32(paired),
                        30,
                    );
                    vget_lane_u8(vreinterpret_u8_u64(packed), 0)
                };
                BitMask::from_int(arch, value)
            }
        }
    }

    #[inline(always)]
    fn from_mask(mask: Self::BitMask) -> Self {
        const BIT_SELECTOR: u64 = 0x0008_0004_0002_0001;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe { vtst_u16(vmov_n_u16(mask.0 as u16), vcreate_u16(BIT_SELECTOR)) }
    }

    #[inline(always)]
    fn keep_first(_arch: Neon, lanes: usize) -> Self {
        const INDICES: u64 = 0x0003_0002_0001_0000;
        let n = lanes.min(4) as u16;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe { vclt_u16(vcreate_u16(INDICES), vmov_n_u16(n)) }
    }

    #[cfg(any(test, miri))]
    fn to_array(self) -> Self::Array {
        // SAFETY: Both the source and destination types are trivially destructible and are
        // valid for all possible bit-patterns.
        unsafe { std::mem::transmute::<Self, Self::Array>(self) }
    }
}

impl MaskOps for uint16x8_t {
    type BitMask = BitMask<8, Neon>;
    type Array = [u16; 8];

    #[inline(always)]
    fn move_mask(self, arch: Neon) -> Self::BitMask {
        cfg_if::cfg_if! {
            if #[cfg(miri)] {
                let array = self.to_array();
                BitMask::from_fn(arch, |i| array[i] == u16::MAX)
            } else {
                // SAFETY: Inclusion of this function is dependent on the "neon" target
                // feature. This function does not access memory directly.
                let value = unsafe {
                    // Effectively creates [-15, -14, -13, -12, -11, -10, -9, -8]
                    let shifts = vcombine_s16(
                        vcreate_s16(0xFFF4_FFF3_FFF2_FFF1),
                        vcreate_s16(0xFFF8_FFF7_FFF6_FFF5),
                    );
                    let mask = vmovq_n_u16(0x8000);
                    vaddlvq_u16(vshlq_u16(vandq_u16(self, mask), shifts))
                };
                BitMask::from_int(arch, value as u8)
            }
        }
    }

    #[inline(always)]
    fn from_mask(mask: Self::BitMask) -> Self {
        const BIT_SELECTOR_LOW: u64 = 0x0008_0004_0002_0001;
        const BIT_SELECTOR_HIGH: u64 = 0x0080_0040_0020_0010;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe {
            vtstq_u16(
                vmovq_n_u16(mask.0 as u16),
                vcombine_u16(
                    vcreate_u16(BIT_SELECTOR_LOW),
                    vcreate_u16(BIT_SELECTOR_HIGH),
                ),
            )
        }
    }

    #[inline(always)]
    fn keep_first(_arch: Neon, lanes: usize) -> Self {
        const LO: u64 = 0x0003_0002_0001_0000;
        const HI: u64 = 0x0007_0006_0005_0004;
        let n = lanes.min(8) as u16;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe {
            vcltq_u16(
                vcombine_u16(vcreate_u16(LO), vcreate_u16(HI)),
                vmovq_n_u16(n),
            )
        }
    }

    #[cfg(any(test, miri))]
    fn to_array(self) -> Self::Array {
        // SAFETY: Both the source and destination types are trivially destructible and are
        // valid for all possible bit-patterns.
        unsafe { std::mem::transmute::<Self, Self::Array>(self) }
    }
}

impl MaskOps for uint32x2_t {
    type BitMask = BitMask<2, Neon>;
    type Array = [u32; 2];

    #[inline(always)]
    fn move_mask(self, arch: Neon) -> Self::BitMask {
        cfg_if::cfg_if! {
            if #[cfg(miri)] {
                let array = self.to_array();
                BitMask::from_fn(arch, |i| array[i] == u32::MAX)
            } else {
                // Normalize each lane to 0 or 1, then use shift-right-accumulate to pack
                // bits into position.
                //
                // SAFETY: Inclusion of this function is dependent on the "neon" target
                // feature. This function does not access memory directly.
                let value = unsafe {
                    let bits = vshr_n_u32(self, 31);
                    let packed = vsra_n_u64(
                        vreinterpret_u64_u32(bits),
                        vreinterpret_u64_u32(bits),
                        31,
                    );
                    vget_lane_u8(vreinterpret_u8_u64(packed), 0)
                };
                BitMask::from_int(arch, value)
            }
        }
    }

    #[inline(always)]
    fn from_mask(mask: Self::BitMask) -> Self {
        const BIT_SELECTOR: u64 = 0x0000_0002_0000_0001;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe { vtst_u32(vmov_n_u32(mask.0 as u32), vcreate_u32(BIT_SELECTOR)) }
    }

    #[inline(always)]
    fn keep_first(_arch: Neon, lanes: usize) -> Self {
        const INDICES: u64 = 0x0000_0001_0000_0000;
        let n = lanes.min(2) as u32;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe { vclt_u32(vcreate_u32(INDICES), vmov_n_u32(n)) }
    }

    #[cfg(any(test, miri))]
    fn to_array(self) -> Self::Array {
        // SAFETY: Both the source and destination types are trivially destructible and are
        // valid for all possible bit-patterns.
        unsafe { std::mem::transmute::<Self, Self::Array>(self) }
    }
}

impl MaskOps for uint32x4_t {
    type BitMask = BitMask<4, Neon>;
    type Array = [u32; 4];

    #[inline(always)]
    fn move_mask(self, arch: Neon) -> Self::BitMask {
        cfg_if::cfg_if! {
            if #[cfg(miri)] {
                let array = self.to_array();
                BitMask::from_fn(arch, |i| array[i] == u32::MAX)
            } else {
                // Refer to the implementation for `uint16x4_t`. The approach here is
                // identical, just twice as wide.
                //
                // SAFETY: Inclusion of this function is dependent on the "neon" target
                // feature. This function does not access memory directly.
                let value = unsafe {
                    let bits = vshrq_n_u32(self, 31);
                    let paired = vsraq_n_u64(
                        vreinterpretq_u64_u32(bits),
                        vreinterpretq_u64_u32(bits),
                        31,
                    );
                    // Narrow the two u64 lanes to two u32 lanes in a 64-bit register.
                    let narrowed = vmovn_u64(paired);
                    let packed = vsra_n_u64(
                        vreinterpret_u64_u32(narrowed),
                        vreinterpret_u64_u32(narrowed),
                        30,
                    );
                    vget_lane_u8(vreinterpret_u8_u64(packed), 0)
                };
                BitMask::from_int(arch, value)
            }
        }
    }

    #[inline(always)]
    fn from_mask(mask: Self::BitMask) -> Self {
        const BIT_SELECTOR_LOW: u64 = 0x0000_0002_0000_0001;
        const BIT_SELECTOR_HIGH: u64 = 0x0000_0008_0000_0004;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe {
            vtstq_u32(
                vmovq_n_u32(mask.0 as u32),
                vcombine_u32(
                    vcreate_u32(BIT_SELECTOR_LOW),
                    vcreate_u32(BIT_SELECTOR_HIGH),
                ),
            )
        }
    }

    #[inline(always)]
    fn keep_first(_arch: Neon, lanes: usize) -> Self {
        const LO: u64 = 0x0000_0001_0000_0000;
        const HI: u64 = 0x0000_0003_0000_0002;
        let n = lanes.min(4) as u32;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe {
            vcltq_u32(
                vcombine_u32(vcreate_u32(LO), vcreate_u32(HI)),
                vmovq_n_u32(n),
            )
        }
    }

    #[cfg(any(test, miri))]
    fn to_array(self) -> Self::Array {
        // SAFETY: Both the source and destination types are trivially destructible and are
        // valid for all possible bit-patterns.
        unsafe { std::mem::transmute::<Self, Self::Array>(self) }
    }
}

impl MaskOps for uint64x1_t {
    type BitMask = BitMask<1, Neon>;
    type Array = [u64; 1];

    #[inline(always)]
    fn move_mask(self, arch: Neon) -> Self::BitMask {
        cfg_if::cfg_if! {
            if #[cfg(miri)] {
                let array = self.to_array();
                BitMask::from_fn(arch, |i| array[i] == u64::MAX)
            } else {
                // Single lane: just shift the MSB down to bit 0 and extract.
                //
                // SAFETY: Inclusion of this function is dependent on the "neon" target
                // feature. This function does not access memory directly.
                let value = unsafe {
                    vget_lane_u8(vreinterpret_u8_u64(vshr_n_u64(self, 63)), 0)
                };
                BitMask::from_int(arch, value)
            }
        }
    }

    #[inline(always)]
    fn from_mask(mask: Self::BitMask) -> Self {
        // Single lane: negation maps 0→0 and 1→0xFFFF_FFFF_FFFF_FFFF.
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe { vcreate_u64((mask.0 as u64).wrapping_neg()) }
    }

    #[inline(always)]
    fn keep_first(_arch: Neon, lanes: usize) -> Self {
        // Single lane: negation maps 0→0 and 1→0xFFFF_FFFF_FFFF_FFFF.
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe { vcreate_u64((lanes.min(1) as u64).wrapping_neg()) }
    }

    #[cfg(any(test, miri))]
    fn to_array(self) -> Self::Array {
        // SAFETY: Both the source and destination types are trivially destructible and are
        // valid for all possible bit-patterns.
        unsafe { std::mem::transmute::<Self, Self::Array>(self) }
    }
}

impl MaskOps for uint64x2_t {
    type BitMask = BitMask<2, Neon>;
    type Array = [u64; 2];

    #[inline(always)]
    fn move_mask(self, arch: Neon) -> Self::BitMask {
        cfg_if::cfg_if! {
            if #[cfg(miri)] {
                let array = self.to_array();
                BitMask::from_fn(arch, |i| array[i] == u64::MAX)
            } else {
                // Normalize each lane to 0 or 1, then narrow to a 64-bit register and
                // use shift-right-accumulate to combine the two bits.
                //
                // SAFETY: Inclusion of this function is dependent on the "neon" target
                // feature. This function does not access memory directly.
                let value = unsafe {
                    let bits = vshrq_n_u64(self, 63);
                    let narrowed = vmovn_u64(bits);
                    let packed = vsra_n_u64(
                        vreinterpret_u64_u32(narrowed),
                        vreinterpret_u64_u32(narrowed),
                        31,
                    );
                    vget_lane_u8(vreinterpret_u8_u64(packed), 0)
                };
                BitMask::from_int(arch, value)
            }
        }
    }

    #[inline(always)]
    fn from_mask(mask: Self::BitMask) -> Self {
        const BIT_SELECTOR_LOW: u64 = 0x0000_0000_0000_0001;
        const BIT_SELECTOR_HIGH: u64 = 0x0000_0000_0000_0002;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe {
            vtstq_u64(
                vmovq_n_u64(mask.0 as u64),
                vcombine_u64(
                    vcreate_u64(BIT_SELECTOR_LOW),
                    vcreate_u64(BIT_SELECTOR_HIGH),
                ),
            )
        }
    }

    #[inline(always)]
    fn keep_first(_arch: Neon, lanes: usize) -> Self {
        const LO: u64 = 0;
        const HI: u64 = 1;
        let n = lanes.min(2) as u64;
        // SAFETY: Inclusion of this function is dependent on the "neon" target
        // feature. This function does not access memory directly.
        unsafe {
            vcltq_u64(
                vcombine_u64(vcreate_u64(LO), vcreate_u64(HI)),
                vmovq_n_u64(n),
            )
        }
    }

    #[cfg(any(test, miri))]
    fn to_array(self) -> Self::Array {
        // SAFETY: Both the source and destination types are trivially destructible and are
        // valid for all possible bit-patterns.
        unsafe { std::mem::transmute::<Self, Self::Array>(self) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Const, SupportedLaneCount};

    trait MaskTraits: std::fmt::Debug {
        const SET: Self;
        const UNSET: Self;
    }

    impl MaskTraits for u8 {
        const SET: u8 = u8::MAX;
        const UNSET: u8 = 0;
    }

    impl MaskTraits for u16 {
        const SET: u16 = u16::MAX;
        const UNSET: u16 = 0;
    }

    impl MaskTraits for u32 {
        const SET: u32 = u32::MAX;
        const UNSET: u32 = 0;
    }

    impl MaskTraits for u64 {
        const SET: u64 = u64::MAX;
        const UNSET: u64 = 0;
    }

    trait AllValues: SIMDMask {
        fn all_values() -> impl Iterator<Item = <Self as SIMDMask>::Underlying>;
    }

    impl AllValues for BitMask<1, Neon> {
        fn all_values() -> impl Iterator<Item = <Self as SIMDMask>::Underlying> {
            0..2
        }
    }

    impl AllValues for BitMask<2, Neon> {
        fn all_values() -> impl Iterator<Item = <Self as SIMDMask>::Underlying> {
            0..4
        }
    }

    impl AllValues for BitMask<4, Neon> {
        fn all_values() -> impl Iterator<Item = <Self as SIMDMask>::Underlying> {
            0..16
        }
    }

    impl AllValues for BitMask<8, Neon> {
        fn all_values() -> impl Iterator<Item = <Self as SIMDMask>::Underlying> {
            0..=u8::MAX
        }
    }

    impl AllValues for BitMask<16, Neon> {
        fn all_values() -> impl Iterator<Item = <Self as SIMDMask>::Underlying> {
            0..=u16::MAX
        }
    }

    fn test_mask<M, T, const N: usize>()
    where
        Const<N>: SupportedLaneCount,
        BitMask<N, Neon>: SIMDMask<Arch = Neon> + AllValues + From<M>,
        T: MaskTraits + PartialEq + Copy,
        M: SIMDMask<Arch = Neon, BitMask = BitMask<N, Neon>> + From<BitMask<N, Neon>>,
        <M as SIMDMask>::Underlying: MaskOps<BitMask = BitMask<N, Neon>, Array = [T; N]>,
    {
        let arch = Neon::new_checked().unwrap();

        // Test keep-first.
        for i in 0..N + 5 {
            let m = M::keep_first(arch, i);

            // Inspect the underlying mask.
            let a = m.to_underlying().to_array();
            assert_eq!(a.len(), N);
            for (j, v) in a.into_iter().enumerate() {
                if j < i {
                    assert_eq!(
                        v,
                        T::SET,
                        "expected lane {} of keep_first({}) to be {:?}. Instead, it is {:?}",
                        j,
                        i,
                        T::SET,
                        v
                    );
                } else {
                    assert_eq!(
                        v,
                        T::UNSET,
                        "expected lane {} of keep_first({}) to be {:?}. Instead, it is {:?}",
                        j,
                        i,
                        T::UNSET,
                        v
                    );
                }
            }

            // Inspect the bitmask.
            assert_eq!(m.bitmask(), BitMask::<N, Neon>::keep_first(arch, i));
        }

        // Test all bitmask precursors.
        for v in BitMask::<N, Neon>::all_values() {
            let bitmask = BitMask::<N, Neon>::from_underlying(arch, v);
            let mask = <M as From<BitMask<N, Neon>>>::from(bitmask);

            assert_eq!(BitMask::<N, _>::from(mask), bitmask);
            let a = mask.to_underlying().to_array();
            assert_eq!(a.len(), N);
            for (j, v) in a.into_iter().enumerate() {
                if bitmask.get_unchecked(j) {
                    assert_eq!(
                        v,
                        T::SET,
                        "expected lane {} to be {:?}. Instead, it is {:?}",
                        j,
                        T::SET,
                        v
                    );
                } else {
                    assert_eq!(
                        v,
                        T::UNSET,
                        "expected lane {} to be {:?}. Instead, it is {:?}",
                        j,
                        T::UNSET,
                        v
                    );
                }
            }
        }
    }

    #[test]
    fn test_mask8x8() {
        test_mask::<mask8x8, u8, 8>();
    }

    #[cfg(not(miri))]
    #[test]
    fn test_mask8x16() {
        test_mask::<mask8x16, u8, 16>();
    }

    #[test]
    fn test_mask16x4() {
        test_mask::<mask16x4, u16, 4>();
    }

    #[test]
    fn test_mask16x8() {
        test_mask::<mask16x8, u16, 8>();
    }

    #[test]
    fn test_mask32x2() {
        test_mask::<mask32x2, u32, 2>();
    }

    #[test]
    fn test_mask32x4() {
        test_mask::<mask32x4, u32, 4>();
    }

    #[test]
    fn test_mask64x1() {
        test_mask::<mask64x1, u64, 1>();
    }

    #[test]
    fn test_mask64x2() {
        test_mask::<mask64x2, u64, 2>();
    }
}
