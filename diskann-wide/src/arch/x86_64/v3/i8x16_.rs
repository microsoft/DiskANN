/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::{
    LoHi, SplitJoin,
    arch::{
        emulated::Scalar,
        x86_64::{
            V3, algorithms,
            common::AllOnes,
            macros::{self, X86Default, X86LoadStore, X86Splat},
            v3::{i16x16, masks::mask8x16},
        },
    },
    constant::Const,
    emulated::Emulated,
    helpers,
    traits::{AsSIMD, SIMDAbs, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector},
};

//////////////////
// 8-bit signed //
//////////////////

macros::x86_define_register!(i8x16, __m128i, mask8x16, i8, 16, V3);
macros::x86_define_splat!(i8x16, _mm_set1_epi8, "sse2");
macros::x86_define_default!(i8x16, _mm_setzero_si128, "sse2");

helpers::unsafe_map_binary_op!(i8x16, std::ops::Add, add, _mm_add_epi8, "sse2");
helpers::unsafe_map_binary_op!(i8x16, std::ops::Sub, sub, _mm_sub_epi8, "sse2");
helpers::unsafe_map_unary_op!(i8x16, SIMDAbs, abs_simd, _mm_abs_epi8, "sse3");

impl std::ops::Mul for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let arch = self.arch();
        let x: i16x16 = self.into();
        let y: i16x16 = rhs.into();
        let LoHi { lo, hi } = ((x * y) & i16x16::splat(arch, 0x00ff)).split();
        // `_mm_packus_epi16` performs saturating truncation and interleaving of the 16-bit
        // results. So if the result is `v`, then
        //
        // * `v[0] = lo[0]`
        // * `v[1] = hi[0]`
        // * `v[2] = lo[1]`
        // * `v[3] = hi[1]`
        // * ...
        //
        // SAFETY: `_mm_packus_epi16` requires SSE2 - implied by `V3`.
        Self::from_underlying(arch, unsafe { _mm_packus_epi16(lo.0, hi.0) })
    }
}

helpers::unsafe_map_binary_op!(i8x16, std::ops::BitAnd, bitand, _mm_and_si128, "sse2");
helpers::unsafe_map_binary_op!(i8x16, std::ops::BitOr, bitor, _mm_or_si128, "sse2");
helpers::unsafe_map_binary_op!(i8x16, std::ops::BitXor, bitxor, _mm_xor_si128, "sse2");

impl std::ops::Shr for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        self.emulated().shr(rhs.emulated()).as_simd(self.arch())
    }
}

impl std::ops::Shl for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        self.emulated().shl(rhs.emulated()).as_simd(self.arch())
    }
}

helpers::scalar_shift_by_splat!(i8x16, i8);

impl std::ops::Not for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for i8x16 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

impl X86LoadStore for i8x16 {
    #[inline(always)]
    unsafe fn load_simd(_: V3, ptr: *const i8) -> Self {
        // SAFETY: The caller asserts this pointer access is safe.
        Self(unsafe { _mm_loadu_si128(ptr as *const __m128i) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(arch: V3, ptr: *const i8, mask: Self::Mask) -> Self {
        // SAFETY: The caller asserts this pointer access is safe. The implementations of
        // `bitmask()` and `Emulated` are trusted.
        Self::from_array(arch, unsafe {
            Emulated::load_simd_masked_logical(Scalar, ptr, mask.bitmask().as_scalar()).to_array()
        })
    }

    #[inline(always)]
    unsafe fn load_simd_first(arch: V3, ptr: *const i8, first: usize) -> Self {
        // SAFETY: The caller asserts this pointer access is safe.
        let bytes = unsafe { algorithms::__load_first_of_16_bytes(arch, ptr as *const u8, first) };

        // SAFETY: Transmuting between fundamental types and similarly sized intrinsics
        // is safe.
        Self::from_underlying(arch, unsafe { std::mem::transmute::<u128, __m128i>(bytes) })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut i8) {
        // SAFETY: The caller asserts this pointer access is safe.
        unsafe { _mm_storeu_si128(ptr.cast::<__m128i>(), self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut i8, mask: Self::Mask) {
        // SAFETY: The caller asserts this pointer access is safe. The implementations of
        // `bitmask()` and `Emulated` are trusted.
        unsafe {
            self.emulated()
                .store_simd_masked_logical(ptr, mask.bitmask().as_scalar())
        }
    }

    #[inline(always)]
    unsafe fn store_simd_first(self, ptr: *mut i8, first: usize) {
        // SAFETY: The caller asserts this pointer access is safe. The implementations of
        // `bitmask()` and `Emulated` are trusted.
        unsafe { self.emulated().store_simd_first(ptr, first) }
    }
}

impl SIMDPartialEq for i8x16 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        // SAFETY: The presence of `Self` attest the intrinsic can be used.
        Self::Mask::from_underlying(self.arch(), unsafe { _mm_cmpeq_epi8(self.0, other.0) })
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        // SAFETY: The presence of `Self` attest the intrinsic can be used.
        let m = unsafe { _mm_xor_si128(_mm_cmpeq_epi8(self.0, other.0), __m128i::all_ones()) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDPartialOrd for i8x16 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        // SAFETY: The presence of `Self` attest the intrinsic can be used.
        Self::Mask::from_underlying(self.arch(), unsafe { _mm_cmpgt_epi8(other.0, self.0) })
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        // SAFETY: The presence of `Self` attest the intrinsic can be used.
        let m = unsafe { _mm_cmpeq_epi8(self.0, _mm_min_epi8(self.0, other.0)) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_i8 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    // Test loading logic - ensure that no out of bounds accesses are made.
    // In particular, this is meant to be run under `Miri` to ensure that our guarantees
    // regarding out-of-bounds accesses are honored.
    #[test]
    fn miri_test_load() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<i8, 16, i8x16>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<i8, 16, i8x16>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<i8, 16, i8x16>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i8x16, 0x54696b283652af78, V3::new_checked_uncached());
    test_utils::ops::test_sub!(i8x16, 0x6d0fdf5d8ce94e08, V3::new_checked_uncached());
    test_utils::ops::test_mul!(i8x16, 0x244f0650bef85ba1, V3::new_checked_uncached());
    test_utils::ops::test_fma!(i8x16, 0x9d1b834775effe58, V3::new_checked_uncached());
    test_utils::ops::test_abs!(i8x16, 0x40638a9d09522d1c, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(i8x16, 0x197695206c36808d, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i8x16, 0xeb7bb4da5b84ebbe, V3::new_checked_uncached());
}
