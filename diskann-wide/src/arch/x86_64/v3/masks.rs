/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::arch::x86_64::*;

use super::V3;
use crate::{
    bitmask::{BitMask, FromInt},
    doubled,
    traits::SIMDMask,
};

///////////////////////
// AVX2 32-bit masks //
///////////////////////

// mask8x16
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct mask8x16(pub(crate) __m128i);

impl SIMDMask for mask8x16 {
    type Arch = V3;
    type Underlying = __m128i;
    type BitMask = BitMask<16, V3>;
    const ISBITS: bool = false;
    const LANES: usize = 16;

    #[inline(always)]
    fn arch(self) -> V3 {
        // SAFETY: The existence of `self` is proof that we are V3 compatible.
        unsafe { V3::new() }
    }

    #[inline(always)]
    fn to_underlying(self) -> Self::Underlying {
        self.0
    }

    #[inline(always)]
    fn from_underlying(_: V3, value: Self::Underlying) -> Self {
        Self(value)
    }

    #[inline(always)]
    fn keep_first(_: V3, i: usize) -> Self {
        let i = i.min(Self::LANES);
        const CMP: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        // SAFETY: The `V3` architecture instance is proof that we can use the V3 compatible
        // intrinsics invoked used here.
        //
        // Unaligned load is valid because the array is the correct size.
        //
        // This constant local variable is hoisted to a constant in the final binary.
        // Codegen emits a load to this value so it is relatively cheap to use.
        unsafe {
            let c = _mm_loadu_si128(CMP.as_ptr() as *const __m128i);

            // Broadcast the argument across all SIMD lanes, and compare the broadcasted register
            // with the incremental value in `CMP`.
            Self(_mm_cmpgt_epi8(_mm_set1_epi8(i as i8), c))
        }
    }

    fn get_unchecked(&self, i: usize) -> bool {
        // This is not particularly efficient.
        // For bulk checking, users should first convert to a bit-mask and then check.
        //
        // Essentially, what this is doing is an entire conversion to a bit-mask to check
        // a single lane.
        Into::<Self::BitMask>::into(*self).get_unchecked(i)
    }
}

// Conversion back-and-forth.
// Credit to https://stackoverflow.com/a/72899629 for this algorithm.
//
// The gist here is that we load `selector` with
// ```ignore
// Lane  |  15   14   13   12   11   10   09   08      07  06   05   04   03   02   01   00
//       |
// Value | 0x80 0x40 0x20 0x10 0x08 0x04 0x02 0x01 | 0x80 0x40 0x20 0x10 0x08 0x04 0x02 0x01
// ```
// Then, we put the 2-bytes of mask into the lower lanes of a `_mm128i`.
//
// Using a shuffle, we move the lower byte of the mask into lanes 00 to 07 and the upper
// byte to lanes 08 to 15 using a shuffle.
//
// We then use bit-wise "and" and a comparison to re-create the SIMD mask.
impl From<BitMask<16, V3>> for mask8x16 {
    #[inline(always)]
    fn from(mask: BitMask<16, V3>) -> Self {
        // Extract the underlying integer.
        let mask: u16 = mask.0;

        // Masks used for the bit-twiddling.
        // Select
        // - bit 7 of byte 7
        // - bit 6 of byte 6
        // - bit 5 of byte 5
        // etc.
        const BIT_SELECTOR: i64 = 0x8040201008040201u64 as i64;

        // Select byte 0 and broadcast it across 8 bytes.
        const BROADCAST_BYTE_0: i64 = 0;
        // Select byte 1 and broadcast it across 8 bytes;
        const BROADCAST_BYTE_1: i64 = 0x0101010101010101;

        // SAFETY: The `V3` architecture instance in `mask` is proof that we can use the V3
        // compatible intrinsics invoked used here.
        unsafe {
            let selector = _mm_set1_epi64x(BIT_SELECTOR);
            Self(_mm_cmpeq_epi8(
                _mm_and_si128(
                    _mm_shuffle_epi8(
                        _mm_cvtsi32_si128(mask as i32),
                        _mm_set_epi64x(BROADCAST_BYTE_1, BROADCAST_BYTE_0),
                    ),
                    selector,
                ),
                selector,
            ))
        }
    }
}

impl From<mask8x16> for BitMask<16, V3> {
    #[inline(always)]
    fn from(mask: mask8x16) -> Self {
        let m = mask.to_underlying();
        // Use an intrinsics to convert the upper bits to an integer bit-mask.
        // SAFETY: Using intrinsics without touching memory. Invocation of the intrinsic
        // is gated on successful check of the `cfg` macro.
        let bitmask: i32 = unsafe { _mm_movemask_epi8(m) };
        // The intrinsic only sets the lower-bit bits of the returned integer.
        // We can safely truncate to an 8-bit integer.
        BitMask::from_int(mask.arch(), bitmask as u16)
    }
}

// mask8x32
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct mask8x32(pub(crate) __m256i);

impl SIMDMask for mask8x32 {
    type Arch = V3;
    type Underlying = __m256i;
    type BitMask = BitMask<32, V3>;
    const ISBITS: bool = false;
    const LANES: usize = 32;

    #[inline(always)]
    fn arch(self) -> V3 {
        // SAFETY: The existence of `self` is proof that we are V3 compatible.
        unsafe { V3::new() }
    }

    #[inline(always)]
    fn to_underlying(self) -> Self::Underlying {
        self.0
    }

    #[inline(always)]
    fn from_underlying(_: V3, value: Self::Underlying) -> Self {
        Self(value)
    }

    #[inline(always)]
    fn keep_first(_: V3, i: usize) -> Self {
        let i = i.min(Self::LANES);
        const CMP: [i8; 32] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ];

        // SAFETY: The `V3` architecture instance is proof that we can use the V3 compatible
        // intrinsics invoked used here.
        //
        // Unaligned load is valid because the array is the correct size.
        //
        // This constant local variable is hoisted to a constant in the final binary.
        // Codegen emits a load to this value so it is relatively cheap to use.
        unsafe {
            let c = _mm256_loadu_si256(CMP.as_ptr() as *const __m256i);

            // Broadcast the argument across all SIMD lanes, and compare the broadcasted register
            // with the incremental value in `CMP`.
            Self(_mm256_cmpgt_epi8(_mm256_set1_epi8(i as i8), c))
        }
    }

    fn get_unchecked(&self, i: usize) -> bool {
        // This is not particularly efficient.
        // For bulk checking, users should first convert to a bit-mask and then check.
        //
        // Essentially, what this is doing is an entire conversion to a bit-mask to check
        // a single lane.
        Into::<Self::BitMask>::into(*self).get_unchecked(i)
    }
}

// Conversion back-and-forth.
// Credit to https://stackoverflow.com/a/72899629 for this algorithm.
//
// This follows the same strategy as `From<BitMask<16, V3>> for `mask8x16` - just twice as
// wide.
//
// The 32-bit representation of `BitMask` is broadcast to all 8 lanes of a 256-bit wide
// register. If we represent the 32-bit mask in terms of bytes like `b3b2b1b0`, then
// following the broadcast, we get:
// ```text
// |  Lane 0  |  Lane 1  |  Lane 2  |  Lane 3  |  Lane 4  |  Lane 5  |  Lane 6  |  Lane 7  |
// | b3b2b1b0 | b3b2b1b0 | b3b2b1b0 | b3b2b1b0 | b3b2b1b0 | b3b2b1b0 | b3b2b1b0 | b3b2b1b0 |
// ```
// Then, we shuffle to get
// ```text
// |  Lane 0  |  Lane 1  |  Lane 2  |  Lane 3  |  Lane 4  |  Lane 5  |  Lane 6  |  Lane 7  |
// | b0b0b0b0 | b0b0b0b0 | b1b1b1b1 | b1b1b1b1 | b2b2b2b2 | b2b2b2b2 | b3b3b3b3 | b3b3b3b3 |
// ```
// From this position, we apply a bit mask to keep bit 0 of byte position 0 (`b0`) in lane 0,
// bit 1 of byte position 1 (still `b0`) in lane 0 etc. In this way, we can isolate all the
// bits in `mask` into bytes a `__m256i`. At which point, `_mm256_cmpeq_epi8` can be used
// to test whether the bit is set or not and thus create the full mask.
impl From<BitMask<32, V3>> for mask8x32 {
    #[inline(always)]
    fn from(mask: BitMask<32, V3>) -> Self {
        // Extract the underlying integer.
        let mask: u32 = mask.0;

        // Masks used for the bit-twiddling.
        // Select
        // - bit 7 of byte 7
        // - bit 6 of byte 6
        // - bit 5 of byte 5
        // etc.
        const BIT_SELECTOR: i64 = 0x8040201008040201u64 as i64;

        // Select byte 0 and broadcast it across 8 bytes.
        const BROADCAST_BYTE_0: i64 = 0;
        // Select byte 1 and broadcast it across 8 bytes;
        const BROADCAST_BYTE_1: i64 = 0x0101010101010101;
        // Select byte 2 and broadcast it across 8 bytes;
        const BROADCAST_BYTE_2: i64 = 0x0202020202020202;
        // Select byte 2 and broadcast it across 8 bytes;
        const BROADCAST_BYTE_3: i64 = 0x0303030303030303;

        // SAFETY: The `V3` architecture instance in `mask` is proof that we can use the V3
        // compatible intrinsics invoked used here.
        unsafe {
            let selector = _mm256_set1_epi64x(BIT_SELECTOR);
            Self(_mm256_cmpeq_epi8(
                _mm256_and_si256(
                    _mm256_shuffle_epi8(
                        _mm256_set1_epi32(mask as i32),
                        _mm256_set_epi64x(
                            BROADCAST_BYTE_3,
                            BROADCAST_BYTE_2,
                            BROADCAST_BYTE_1,
                            BROADCAST_BYTE_0,
                        ),
                    ),
                    selector,
                ),
                selector,
            ))
        }
    }
}

impl From<mask8x32> for BitMask<32, V3> {
    #[inline(always)]
    fn from(mask: mask8x32) -> Self {
        let m = mask.to_underlying();
        // Use an intrinsics to convert the upper bits to an integer bit-mask.
        //
        // SAFETY: `_mm256_movemask_epi8` requires AVX2 - which is implied by `V3`.
        let bitmask: i32 = unsafe { _mm256_movemask_epi8(m) };
        BitMask::from_int(mask.arch(), bitmask as u32)
    }
}

// mask32x4
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct mask32x4(pub(crate) __m128i);

impl SIMDMask for mask32x4 {
    type Arch = V3;
    type Underlying = __m128i;
    type BitMask = BitMask<4, V3>;
    const ISBITS: bool = false;
    const LANES: usize = 4;

    #[inline(always)]
    fn arch(self) -> V3 {
        // SAFETY: The existence of `Self` proves its architecture is safe.
        unsafe { V3::new() }
    }

    #[inline(always)]
    fn to_underlying(self) -> Self::Underlying {
        self.0
    }

    #[inline(always)]
    fn from_underlying(_: V3, value: Self::Underlying) -> Self {
        Self(value)
    }

    #[inline(always)]
    fn keep_first(_: V3, i: usize) -> Self {
        let i = i.min(Self::LANES);
        const CMP: [i32; 4] = [0, 1, 2, 3];

        // SAFETY: This function is conditionally compiled only if the target platform
        // contains the instruction set necessary for the intrinsics used here.
        //
        // Unaligned load is valid because the array is the correct size.
        //
        // This constant local variable is hoisted to a constant in the final binary.
        // Codegen emits a load to this value so it is relatively cheap to use.
        unsafe {
            let c = _mm_loadu_si128(CMP.as_ptr() as *const __m128i);

            // Broadcast the argument across all SIMD lanes, and compare the broadcasted register
            // with the incremental value in `CMP`.
            Self(_mm_cmpgt_epi32(_mm_set1_epi32(i as i32), c))
        }
    }

    fn get_unchecked(&self, i: usize) -> bool {
        // This is not particularly efficient.
        // For bulk checking, users should first convert to a bit-mask and then check.
        //
        // Essentially, what this is doing is an entire conversion to a bit-mask to check
        // a single lane.
        Into::<Self::BitMask>::into(*self).get_unchecked(i)
    }
}

// Conversion back-and-forth.
impl From<BitMask<4, V3>> for mask32x4 {
    #[inline(always)]
    fn from(mask: BitMask<4, V3>) -> Self {
        // Extract the underlying integer.
        let mask: u8 = mask.0;
        // SAFETY: Using intrinsics without touching memory.
        // The trait implementation is conditional compiled on the intrinsics being
        // available for the target platform.
        unsafe {
            let b = _mm_set1_epi32(mask as i32);
            let cmp = _mm_set_epi32(8, 4, 2, 1);
            let x = _mm_and_si128(b, cmp);
            Self(_mm_cmpgt_epi32(x, _mm_setzero_si128()))
        }
    }
}

impl From<mask32x4> for BitMask<4, V3> {
    #[inline(always)]
    fn from(mask: mask32x4) -> Self {
        let m = mask.to_underlying();
        // Use an intrinsics to convert the upper bits to an integer bit-mask.
        // SAFETY: Using intrinsics without touching memory. Invocation of the intrinsic
        // is gated on successful check of the `cfg` macro.
        let bitmask: i32 = unsafe { _mm_movemask_ps(_mm_castsi128_ps(m)) };
        // The intrinsic only sets the lower-bit bits of the returned integer.
        // We can safely truncate to an 8-bit integer.
        BitMask::from_int(mask.arch(), bitmask as u8)
    }
}

// mask32x8
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct mask32x8(pub(crate) __m256i);

impl SIMDMask for mask32x8 {
    type Arch = V3;
    type Underlying = __m256i;
    type BitMask = BitMask<8, V3>;
    const ISBITS: bool = false;
    const LANES: usize = 8;

    #[inline(always)]
    fn arch(self) -> V3 {
        // SAFETY: The existence of `Self` proves its architecture is safe.
        unsafe { V3::new() }
    }

    #[inline(always)]
    fn to_underlying(self) -> Self::Underlying {
        self.0
    }

    #[inline(always)]
    fn from_underlying(_: V3, value: Self::Underlying) -> Self {
        Self(value)
    }

    #[inline(always)]
    fn keep_first(_: V3, i: usize) -> Self {
        let i = i.min(Self::LANES);
        // This kind of hurts my brain to look at.
        const MASKS: [[u32; 8]; 9] = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [!0, 0, 0, 0, 0, 0, 0, 0],
            [!0, !0, 0, 0, 0, 0, 0, 0],
            [!0, !0, !0, 0, 0, 0, 0, 0],
            [!0, !0, !0, !0, 0, 0, 0, 0],
            [!0, !0, !0, !0, !0, 0, 0, 0],
            [!0, !0, !0, !0, !0, !0, 0, 0],
            [!0, !0, !0, !0, !0, !0, !0, 0],
            [!0, !0, !0, !0, !0, !0, !0, !0],
        ];

        // const CMP: [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

        // SAFETY: This function is conditionally compiled only if the target platform
        // contains the instruction set necessary for the intrinsics used here.
        //
        // Unaligned load is valid because the array is the correct size.
        //
        // This constant local variable is hoisted to a constant in the final binary.
        // Codegen emits a load to this value so it is relatively cheap to use.

        Self(unsafe { std::mem::transmute::<[u32; 8], __m256i>(MASKS[i]) })
    }

    fn get_unchecked(&self, i: usize) -> bool {
        // This is not particularly efficient.
        // For bulk checking, users should first convert to a bit-mask and then check.
        //
        // Essentially, what this is doing is an entire conversion to a bit-mask to check
        // a single lane.
        Into::<Self::BitMask>::into(*self).get_unchecked(i)
    }
}

// Conversion back-and-forth.
impl From<BitMask<8, V3>> for mask32x8 {
    #[inline(always)]
    fn from(mask: BitMask<8, V3>) -> Self {
        // Extract the underlying integer.
        let mask: u8 = mask.0;
        // SAFETY: Using intrinsics without touching memory.
        // Trait implementation gated on the intrinsic being available for the target
        // platform.
        unsafe {
            let b = _mm256_set1_epi32(mask as i32);
            let cmp = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
            let x = _mm256_and_si256(b, cmp);
            Self(_mm256_cmpgt_epi32(x, _mm256_setzero_si256()))
        }
    }
}

impl From<mask32x8> for BitMask<8, V3> {
    #[inline(always)]
    fn from(mask: mask32x8) -> Self {
        let m = mask.to_underlying();
        // SAFETY: Using intrinsics without touching memory.
        // Use an intrinsics to convert the upper bits to an integer bit-mask.
        let bitmask: i32 = unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(m)) };
        // The intrinsic only sets the lower-bit bits of the returned integer.
        // We can safely truncate to an 8-bit integer.
        BitMask::from_int(mask.arch(), bitmask as u8)
    }
}

///////////////////////
// AVX2 64-bit masks //
///////////////////////

// mask64x2
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct mask64x2(pub(crate) __m128i);

impl SIMDMask for mask64x2 {
    type Arch = V3;
    type Underlying = __m128i;
    type BitMask = BitMask<2, V3>;
    const ISBITS: bool = false;
    const LANES: usize = 2;

    #[inline(always)]
    fn arch(self) -> V3 {
        // SAFETY: The existence of `Self` proves its architecture is safe.
        unsafe { V3::new() }
    }

    #[inline(always)]
    fn to_underlying(self) -> Self::Underlying {
        self.0
    }

    #[inline(always)]
    fn from_underlying(_: V3, value: Self::Underlying) -> Self {
        Self(value)
    }

    #[inline(always)]
    fn keep_first(_: V3, i: usize) -> Self {
        let i = i.min(Self::LANES);
        // SAFETY: This function is conditionally compiled only if the target platform
        // contains the instruction set necessary for the intrinsics used here.
        //
        // Unaligned load is valid because the array is the correct size.
        //
        // This constant local variable is hoisted to a constant in the final binary.
        // Codegen emits a load to this value so it is relatively cheap to use.
        unsafe {
            const CMP: [i64; 2] = [0, 1];
            let c = _mm_loadu_si128(CMP.as_ptr() as *const __m128i);

            // Broadcast the argument across all SIMD lanes, and compare the broadcasted register
            // with the incremental value in `CMP`.
            Self(_mm_cmpgt_epi64(_mm_set1_epi64x(i as i64), c))
        }
    }

    fn get_unchecked(&self, i: usize) -> bool {
        // This is not particularly efficient.
        // For bulk checking, users should first convert to a bit-mask and then check.
        //
        // Essentially, what this is doing is an entire conversion to a bit-mask to check
        // a single lane.
        Into::<Self::BitMask>::into(*self).get_unchecked(i)
    }
}

// Conversion back-and-forth.
impl From<BitMask<2, V3>> for mask64x2 {
    #[inline(always)]
    fn from(mask: BitMask<2, V3>) -> Self {
        // Extract the underlying integer.
        let mask: u8 = mask.0;
        // SAFETY: Using intrinsics without touching memory.
        // The trait implementation is conditional compiled on the intrinsics being
        // available for the target platform.
        unsafe {
            let b = _mm_set1_epi64x(mask as i64);
            let cmp = _mm_set_epi64x(2, 1);
            let x = _mm_and_si128(b, cmp);
            Self(_mm_cmpgt_epi64(x, _mm_setzero_si128()))
        }
    }
}

impl From<mask64x2> for BitMask<2, V3> {
    #[inline(always)]
    fn from(mask: mask64x2) -> Self {
        let m = mask.to_underlying();
        // Use an intrinsics to convert the upper bits to an integer bit-mask.
        // SAFETY: Using intrinsics without touching memory. Invocation of the intrinsic
        // is gated on successful check of the `cfg` macro.
        let bitmask: i32 = unsafe { _mm_movemask_pd(_mm_castsi128_pd(m)) };
        // The intrinsic only sets the lower-bit bits of the returned integer.
        // We can safely truncate to an 8-bit integer.
        BitMask::from_int(mask.arch(), bitmask as u8)
    }
}

// mask64x4
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct mask64x4(pub(crate) __m256i);

impl SIMDMask for mask64x4 {
    type Arch = V3;
    type Underlying = __m256i;
    type BitMask = BitMask<4, V3>;
    const ISBITS: bool = false;
    const LANES: usize = 4;

    #[inline(always)]
    fn arch(self) -> V3 {
        // SAFETY: The existence of `Self` proves its architecture is safe.
        unsafe { V3::new() }
    }

    #[inline(always)]
    fn to_underlying(self) -> Self::Underlying {
        self.0
    }

    #[inline(always)]
    fn from_underlying(_: V3, value: Self::Underlying) -> Self {
        Self(value)
    }

    #[inline(always)]
    fn keep_first(_: V3, i: usize) -> Self {
        let i = i.min(Self::LANES);
        // SAFETY: This function is conditionally compiled only if the target platform
        // contains the instruction set necessary for the intrinsics used here.
        //
        // Unaligned load is valid because the array is the correct size.
        //
        // This constant local variable is hoisted to a constant in the final binary.
        // Codegen emits a load to this value so it is relatively cheap to use.
        unsafe {
            const CMP: [i64; 4] = [0, 1, 2, 3];
            let c = _mm256_loadu_si256(CMP.as_ptr() as *const __m256i);

            // Broadcast the argument across all SIMD lanes, and compare the broadcasted register
            // with the incremental value in `CMP`.
            Self(_mm256_cmpgt_epi64(_mm256_set1_epi64x(i as i64), c))
        }
    }

    fn get_unchecked(&self, i: usize) -> bool {
        // This is not particularly efficient.
        // For bulk checking, users should first convert to a bit-mask and then check.
        //
        // Essentially, what this is doing is an entire conversion to a bit-mask to check
        // a single lane.
        Into::<Self::BitMask>::into(*self).get_unchecked(i)
    }
}

// Conversion back-and-forth.
impl From<BitMask<4, V3>> for mask64x4 {
    #[inline(always)]
    fn from(mask: BitMask<4, V3>) -> Self {
        // Extract the underlying integer.
        let mask: u8 = mask.0;
        // SAFETY: Using intrinsics without touching memory.
        // The trait implementation is conditional compiled on the intrinsics being
        // available for the target platform.
        unsafe {
            let b = _mm256_set1_epi64x(mask as i64);
            let cmp = _mm256_set_epi64x(8, 4, 2, 1);
            let x = _mm256_and_si256(b, cmp);
            Self(_mm256_cmpgt_epi64(x, _mm256_setzero_si256()))
        }
    }
}

impl From<mask64x4> for BitMask<4, V3> {
    #[inline(always)]
    fn from(mask: mask64x4) -> Self {
        let m = mask.to_underlying();
        // Use an intrinsics to convert the upper bits to an integer bit-mask.
        // SAFETY: Using intrinsics without touching memory. Invocation of the intrinsic
        // is gated on successful check of the `cfg` macro.
        let bitmask: i32 = unsafe { _mm256_movemask_pd(_mm256_castsi256_pd(m)) };
        // The intrinsic only sets the lower-bit bits of the returned integer.
        // We can safely truncate to an 8-bit integer.
        BitMask::from_int(mask.arch(), bitmask as u8)
    }
}

//////////////////
// Double Masks //
//////////////////

// These mask definitions are shared across the double-wide implementations.

// Native Masks
doubled::double_mask!(64, mask8x32);
doubled::double_mask!(16, mask32x8);

// Bit Mask
doubled::double_mask!(32, BitMask<16, V3>);

#[cfg(test)]
mod test_masks {
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::{
        Architecture, BitMask, Const, FromInt, SupportedLaneCount, doubled::Doubled, test_utils,
        traits::SIMDMask,
    };

    trait TypeRange:
        Copy + rand::distr::uniform::SampleUniform + std::cmp::PartialOrd + std::fmt::Display
    {
        fn make_range_() -> std::ops::RangeInclusive<Self>;
    }

    impl TypeRange for u8 {
        fn make_range_() -> std::ops::RangeInclusive<Self> {
            Self::MIN..=Self::MAX
        }
    }

    impl TypeRange for u16 {
        fn make_range_() -> std::ops::RangeInclusive<Self> {
            Self::MIN..=Self::MAX
        }
    }

    /// A trait to extract the top bit of an integer.
    trait TopBit {
        fn is_top_bit_set(&self) -> bool;
    }

    impl TopBit for u8 {
        fn is_top_bit_set(&self) -> bool {
            (self & 0x80) != 0
        }
    }

    impl TopBit for u32 {
        fn is_top_bit_set(&self) -> bool {
            (self & 0x8000_0000) != 0
        }
    }

    impl TopBit for u64 {
        fn is_top_bit_set(&self) -> bool {
            (self & 0x8000_0000_0000_0000) != 0
        }
    }

    /// Trait to compare AVX2 masks with a corresponding bit-mask.
    trait CheckWithBitmask {
        type BitMask: SIMDMask;
        /// Panics on a mismatch.
        fn check(self, bitmask: Self::BitMask);
    }

    impl CheckWithBitmask for mask8x16 {
        type BitMask = BitMask<16>;
        fn check(self, bitmask: Self::BitMask) {
            // Transmute the underlying register to the correct array.
            //
            // SAFETY: The two types are the same length, do not hold any resources, and
            // are valid for all possible bit patterns.
            let array = unsafe { std::mem::transmute::<__m128i, [u8; 16]>(self.to_underlying()) };
            for (i, v) in array.iter().enumerate() {
                assert_eq!(v.is_top_bit_set(), bitmask.get(i).unwrap());
            }
        }
    }

    impl CheckWithBitmask for mask8x32 {
        type BitMask = BitMask<32>;
        fn check(self, bitmask: Self::BitMask) {
            // Transmute the underlying register to the correct array.
            //
            // SAFETY: The two types are the same length, do not hold any resources, and
            // are valid for all possible bit patterns.
            let array = unsafe { std::mem::transmute::<__m256i, [u8; 32]>(self.to_underlying()) };
            for (i, v) in array.iter().enumerate() {
                assert_eq!(v.is_top_bit_set(), bitmask.get(i).unwrap());
            }
        }
    }

    impl CheckWithBitmask for mask32x4 {
        type BitMask = BitMask<4>;
        fn check(self, bitmask: Self::BitMask) {
            // Transmute the underlying register to the correct array.
            //
            // SAFETY: The two types are the same length, do not hold any resources, and
            // are valid for all possible bit patterns.
            let array = unsafe { std::mem::transmute::<__m128i, [u32; 4]>(self.to_underlying()) };
            for (i, v) in array.iter().enumerate() {
                assert_eq!(v.is_top_bit_set(), bitmask.get(i).unwrap());
            }
        }
    }

    impl CheckWithBitmask for mask32x8 {
        type BitMask = BitMask<8>;
        fn check(self, bitmask: Self::BitMask) {
            // Transmute the underlying register to the correct array.
            //
            // SAFETY: The two types are the same length, do not hold any resources, and
            // are valid for all possible bit patterns.
            let array = unsafe { std::mem::transmute::<__m256i, [u32; 8]>(self.to_underlying()) };
            for (i, v) in array.iter().enumerate() {
                assert_eq!(v.is_top_bit_set(), bitmask.get(i).unwrap());
            }
        }
    }

    impl CheckWithBitmask for mask64x2 {
        type BitMask = BitMask<2>;
        fn check(self, bitmask: Self::BitMask) {
            // Transmute the underlying register to the correct array.
            //
            // SAFETY: The two types are the same length, do not hold any resources, and
            // are valid for all possible bit patterns.
            let array = unsafe { std::mem::transmute::<__m128i, [u64; 2]>(self.to_underlying()) };
            for (i, v) in array.iter().enumerate() {
                assert_eq!(v.is_top_bit_set(), bitmask.get(i).unwrap());
            }
        }
    }

    impl CheckWithBitmask for mask64x4 {
        type BitMask = BitMask<4>;
        fn check(self, bitmask: Self::BitMask) {
            // Transmute the underlying register to the correct array.
            //
            // SAFETY: The two types are the same length, do not hold any resources, and
            // are valid for all possible bit patterns.
            let array = unsafe { std::mem::transmute::<__m256i, [u64; 4]>(self.to_underlying()) };
            for (i, v) in array.iter().enumerate() {
                assert_eq!(v.is_top_bit_set(), bitmask.get(i).unwrap());
            }
        }
    }

    fn check_avx2_mask<T, const N: usize, A>(mask: T, bitmask: BitMask<N, A>)
    where
        A: Architecture,
        Const<N>: SupportedLaneCount,
        BitMask<N, A>: SIMDMask,
        T: CheckWithBitmask<BitMask = BitMask<N>>,
    {
        mask.check(bitmask.as_current())
    }

    /// Test the conversion from bitmask to full-mask.
    ///
    /// Randomly generates bitmasks, constructs a full-mask, and ensures the `get` API
    /// yields the same results.
    ///
    /// Also checks that converting the full-mask back to a bitmask is lossless.
    fn test_mask_conversion_impl<T, const N: usize, A>(arch: A, num_trials: usize, seed: u64)
    where
        A: Architecture,
        Const<N>: SupportedLaneCount,
        BitMask<N, A>:
            SIMDMask<Arch = A> + From<T> + FromInt<<BitMask<N, A> as SIMDMask>::Underlying, A>,
        T: SIMDMask<Arch = A, BitMask = BitMask<N, A>> + From<BitMask<N, A>>,
        <BitMask<N, A> as SIMDMask>::Underlying: TypeRange,
    {
        const MAXLEN: usize = 64;
        assert_eq!(T::LANES, N);
        assert!(MAXLEN >= T::LANES);

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for _ in 0..num_trials {
            let u = rng.random_range(
                <<BitMask<N, A> as SIMDMask>::Underlying as TypeRange>::make_range_(),
            );

            let bit_mask = BitMask::<N, A>::from_int(arch, u);
            let full_mask: T = bit_mask.into();
            for i in 0..=MAXLEN {
                assert_eq!(bit_mask.get(i), full_mask.get(i));
                assert_eq!(bit_mask.get_unchecked(i), full_mask.get_unchecked(i));
            }

            let from_full: BitMask<N, A> = full_mask.into();
            assert_eq!(from_full, bit_mask);
        }
    }

    #[test]
    fn test_mask_conversion() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_mask_conversion_impl::<mask8x16, 16, _>(arch, 5000, 0x12345);

            test_mask_conversion_impl::<mask32x4, 4, _>(arch, 200, 0xc0ffee);
            test_mask_conversion_impl::<mask32x8, 8, _>(arch, 1000, 0x7a08f5);

            test_mask_conversion_impl::<mask64x4, 4, _>(arch, 32, 0x7a08f5);
            test_mask_conversion_impl::<mask64x2, 2, _>(arch, 32, 0xc59783c5d8c4b59b);
        }
    }

    #[test]
    #[should_panic]
    fn test_check_avx2_mask_panics_mask8x16() {
        if let Some(arch) = V3::new_checked_uncached() {
            let m = mask8x16::from_fn(arch, |i| i < 7);
            let bm = BitMask::<16, V3>::from_fn(arch, |i| i < 10);
            check_avx2_mask(m, bm);
        }
    }

    #[test]
    #[should_panic]
    fn test_check_avx2_mask_panics_mask8x32() {
        if let Some(arch) = V3::new_checked_uncached() {
            let m = mask8x32::from_fn(arch, |i| i < 7);
            let bm = BitMask::<32, V3>::from_fn(arch, |i| i < 10);
            check_avx2_mask(m, bm);
        }
    }

    #[test]
    #[should_panic]
    fn test_check_avx2_mask_panics_mask32x4() {
        if let Some(arch) = V3::new_checked_uncached() {
            let m = mask32x4::from_fn(arch, |i| i < 3);
            let bm = BitMask::<4, V3>::from_fn(arch, |i| i <= 3);
            check_avx2_mask(m, bm);
        }
    }

    #[test]
    #[should_panic]
    fn test_check_avx2_mask_panics_mask32x8() {
        if let Some(arch) = V3::new_checked_uncached() {
            let m = mask32x8::from_fn(arch, |i| i < 7);
            let bm = BitMask::<8, V3>::from_fn(arch, |i| i <= 7);
            check_avx2_mask(m, bm);
        }
    }

    #[test]
    #[should_panic]
    fn test_check_avx2_mask_panics_mask64x2() {
        if let Some(arch) = V3::new_checked_uncached() {
            let m = mask64x2::from_fn(arch, |i| i < 1);
            let bm = BitMask::<2, V3>::from_fn(arch, |i| i <= 1);
            check_avx2_mask(m, bm);
        }
    }

    #[test]
    #[should_panic]
    fn test_check_avx2_mask_panics_mask64x4() {
        if let Some(arch) = V3::new_checked_uncached() {
            let m = mask64x4::from_fn(arch, |i| i < 3);
            let bm = BitMask::<4, V3>::from_fn(arch, |i| i <= 3);
            check_avx2_mask(m, bm);
        }
    }

    // Helper macro to run the AVX2 masks through the SIMDMask test routines.
    macro_rules! test_simdmask {
        ($mask:ident $(< $($ps:tt),+ >)?, $N:literal, $checker:expr) => {
            paste::paste! {
                #[test]
                fn [<test_simd_mask_ $mask:lower $(_$($ps:lower )x+)? x $N>]() {
                    type T = $mask $(< $($ps),+>)?;
                    if let Some(arch) = V3::new_checked_uncached() {
                        test_utils::mask::test_keep_first::<T, $N, _, _>(arch, $checker);
                        test_utils::mask::test_from_fn::<T, $N, _, _>(arch, $checker);
                        test_utils::mask::test_reductions::<T, $N, _, _>(arch, $checker);
                        test_utils::mask::test_first::<T, $N, _, _>(arch, $checker);
                    }
                }
            }
        };
    }

    test_simdmask!(mask8x16, 16, check_avx2_mask);
    test_simdmask!(mask8x32, 32, check_avx2_mask);

    test_simdmask!(mask32x4, 4, check_avx2_mask);
    test_simdmask!(mask32x8, 8, check_avx2_mask);

    test_simdmask!(mask64x2, 2, check_avx2_mask);
    test_simdmask!(mask64x4, 4, check_avx2_mask);

    fn nop<T, const N: usize, A>(_: T, _: BitMask<N, A>)
    where
        A: crate::arch::Sealed,
        Const<N>: SupportedLaneCount,
    {
    }

    // Double
    test_simdmask!(Doubled<mask8x32>, 64, nop);
    test_simdmask!(Doubled<mask32x8>, 16, nop);

    // Type alias to work around limitations in `test_simdmask`.
    type BitMask16V3 = BitMask<16, V3>;
    test_simdmask!(Doubled<BitMask16V3>, 32, nop);
}
