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
        v3::masks::mask32x4,
    },
    constant::Const,
    helpers,
    traits::{SIMDAbs, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector},
};

/////
///// 32-bit signed integer
/////

macros::x86_define_register!(i32x4, __m128i, mask32x4, i32, 4, V3);
macros::x86_define_splat!(i32x4, _mm_set1_epi32, "sse2");
macros::x86_define_default!(i32x4, _mm_setzero_si128, "sse2");

helpers::unsafe_map_binary_op!(i32x4, std::ops::Add, add, _mm_add_epi32, "sse2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::Sub, sub, _mm_sub_epi32, "sse2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::Mul, mul, _mm_mullo_epi32, "sse4.1");
helpers::unsafe_map_unary_op!(i32x4, SIMDAbs, abs_simd, _mm_abs_epi32, "ssse3");

helpers::unsafe_map_binary_op!(i32x4, std::ops::BitAnd, bitand, _mm_and_si128, "sse2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::BitOr, bitor, _mm_or_si128, "sse2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::BitXor, bitxor, _mm_xor_si128, "sse2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::Shr, shr, _mm_srav_epi32, "avx2");
helpers::unsafe_map_binary_op!(i32x4, std::ops::Shl, shl, _mm_sllv_epi32, "avx2");
helpers::scalar_shift_by_splat!(i32x4, i32);

impl std::ops::Not for i32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for i32x4 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

impl SIMDPartialEq for i32x4 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        Self::Mask::from_underlying(self.arch(), unsafe { _mm_cmpeq_epi32(self.0, other.0) })
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe { _mm_xor_si128(_mm_cmpeq_epi32(self.0, other.0), __m128i::all_ones()) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDPartialOrd for i32x4 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        Self::Mask::from_underlying(self.arch(), unsafe { _mm_cmpgt_epi32(other.0, self.0) })
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe { _mm_cmpeq_epi32(self.0, _mm_min_epi32(self.0, other.0)) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl X86LoadStore for i32x4 {
    #[inline(always)]
    unsafe fn load_simd(_: V3, ptr: *const i32) -> Self {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use this intrinsic.
        Self(unsafe { _mm_loadu_si128(ptr as *const __m128i) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(_: V3, ptr: *const i32, mask: Self::Mask) -> Self {
        // MIRI does not support `_mm_maskload_epi32`.
        // So we go through a kind of convoluted dance to let this be tested by miri.
        //
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use these intrinsics.
        Self(unsafe { _mm_castps_si128(_mm_maskload_ps(ptr as *const f32, mask.to_underlying())) })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut i32) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe { _mm_storeu_si128(ptr as *mut __m128i, self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut i32, mask: Self::Mask) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe {
            _mm_maskstore_ps(
                ptr.cast::<f32>(),
                mask.to_underlying(),
                _mm_castsi128_ps(self.to_underlying()),
            )
        };
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_i32 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<i32, 4, i32x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<i32, 4, i32x4>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<i32, 4, i32x4>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(i32x4, 0xf04d77df1da6aee6, V3::new_checked_uncached());
    test_utils::ops::test_sub!(i32x4, 0xd1e591845e17fc01, V3::new_checked_uncached());
    test_utils::ops::test_mul!(i32x4, 0xf0caa85d919a41a8, V3::new_checked_uncached());
    test_utils::ops::test_fma!(i32x4, 0x1f0340c2109aef6f, V3::new_checked_uncached());
    test_utils::ops::test_abs!(i32x4, 0x60710c0c88537c7d, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(i32x4, 0x9a2d73b7295214c6, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i32x4, 0x763fc44f8f7cd40c, V3::new_checked_uncached());
}
