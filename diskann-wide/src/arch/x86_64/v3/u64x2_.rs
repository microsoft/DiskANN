/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::{
    arch::x86_64::{
        V3,
        common::AllOnes,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v3::masks::mask64x2,
    },
    constant::Const,
    helpers,
    traits::{AsSIMD, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector},
};

/////
///// 64-bit unsigned integer
/////

macros::x86_define_register!(u64x2, __m128i, mask64x2, u64, 2, V3);
macros::x86_define_splat!(u64x2 as i64, _mm_set1_epi64x, "sse2");
macros::x86_define_default!(u64x2, _mm_setzero_si128, "sse2");

helpers::unsafe_map_binary_op!(u64x2, std::ops::Add, add, _mm_add_epi64, "sse2");
helpers::unsafe_map_binary_op!(u64x2, std::ops::Sub, sub, _mm_sub_epi64, "sse2");

// Emulated multiplication.
// This actually generates not terrible code.
impl std::ops::Mul for u64x2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        (self.emulated() * rhs.emulated()).as_simd(self.arch())
    }
}

helpers::unsafe_map_binary_op!(u64x2, std::ops::BitAnd, bitand, _mm_and_si128, "sse2");
helpers::unsafe_map_binary_op!(u64x2, std::ops::BitOr, bitor, _mm_or_si128, "sse2");
helpers::unsafe_map_binary_op!(u64x2, std::ops::BitXor, bitxor, _mm_xor_si128, "sse2");
helpers::unsafe_map_binary_op!(u64x2, std::ops::Shr, shr, _mm_srlv_epi64, "avx2");
helpers::unsafe_map_binary_op!(u64x2, std::ops::Shl, shl, _mm_sllv_epi64, "avx2");

helpers::scalar_shift_by_splat!(u64x2, u64);

impl std::ops::Not for u64x2 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u64x2 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

impl X86LoadStore for u64x2 {
    #[inline(always)]
    unsafe fn load_simd(_: V3, ptr: *const u64) -> Self {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use this intrinsic.
        Self(unsafe { _mm_loadu_si128(ptr as *const __m128i) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(_: V3, ptr: *const u64, mask: Self::Mask) -> Self {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use this intrinsic.
        Self(unsafe { _mm_maskload_epi64(ptr as *const i64, mask.to_underlying()) })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut u64) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe { _mm_storeu_si128(ptr.cast::<__m128i>(), self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut u64, mask: Self::Mask) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe {
            _mm_maskstore_epi64(
                ptr.cast::<i64>(),
                mask.to_underlying(),
                self.to_underlying(),
            )
        };
    }
}

impl SIMDPartialEq for u64x2 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe { _mm_cmpeq_epi64(self.0, other.0) };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe { _mm_xor_si128(_mm_cmpeq_epi64(self.0, other.0), __m128i::all_ones()) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

// Notes: 64-bit `max` and `min` require AVX512-F and AVX512-VL
//
// We use an emulated implementation for AVX2.
impl SIMDPartialOrd for u64x2 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        self.emulated()
            .lt_simd(other.emulated())
            .as_arch(self.arch())
            .into()
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        self.emulated()
            .le_simd(other.emulated())
            .as_arch(self.arch())
            .into()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_u64 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<u64, 2, u64x2>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<u64, 2, u64x2>(arch);
        }
    }

    // constructors
    #[cfg(not(miri))]
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<u64, 2, u64x2>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(u64x2, 0xd1eb6bdca533ca00, V3::new_checked_uncached());
    test_utils::ops::test_sub!(u64x2, 0xde6675680ad0a65b, V3::new_checked_uncached());
    test_utils::ops::test_mul!(u64x2, 0xc78dafd8072a8868, V3::new_checked_uncached());
    test_utils::ops::test_fma!(u64x2, 0xcb45c9f29a44719f, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(u64x2, 0x92486698bb7603e7, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u64x2, 0xf9566b095125ca45, V3::new_checked_uncached());
}
