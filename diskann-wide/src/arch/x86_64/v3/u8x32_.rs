/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::{
    SplitJoin,
    arch::{
        emulated::Scalar,
        x86_64::{
            V3,
            common::AllOnes,
            macros::{self, X86Default, X86LoadStore, X86Splat},
            v3::{masks::mask8x32, u8x16},
        },
    },
    constant::Const,
    emulated::Emulated,
    helpers,
    traits::{AsSIMD, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector},
};

////////////////////
// 8-bit unsigned //
////////////////////

macros::x86_define_register!(u8x32, __m256i, mask8x32, u8, 32, V3);
macros::x86_define_splat!(u8x32 as i8, _mm256_set1_epi8, "avx");
macros::x86_define_default!(u8x32, _mm256_setzero_si256, "avx");
macros::x86_splitjoin!(
    u8x32,
    u8x16,
    _mm256_extracti128_si256,
    _mm256_set_m128i,
    "avx2"
);

impl std::ops::Mul for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.split()
            .map_with(rhs.split(), std::ops::Mul::mul)
            .join()
    }
}

helpers::unsafe_map_binary_op!(u8x32, std::ops::Add, add, _mm256_add_epi8, "avx2");
helpers::unsafe_map_binary_op!(u8x32, std::ops::Sub, sub, _mm256_sub_epi8, "avx2");

helpers::unsafe_map_binary_op!(u8x32, std::ops::BitAnd, bitand, _mm256_and_si256, "avx2");
helpers::unsafe_map_binary_op!(u8x32, std::ops::BitOr, bitor, _mm256_or_si256, "avx2");
helpers::unsafe_map_binary_op!(u8x32, std::ops::BitXor, bitxor, _mm256_xor_si256, "avx2");

impl std::ops::Shr for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        self.emulated().shr(rhs.emulated()).as_simd(self.arch())
    }
}

impl std::ops::Shl for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        self.emulated().shl(rhs.emulated()).as_simd(self.arch())
    }
}

helpers::scalar_shift_by_splat!(u8x32, u8);

impl std::ops::Not for u8x32 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u8x32 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

impl X86LoadStore for u8x32 {
    #[inline(always)]
    unsafe fn load_simd(_: V3, ptr: *const u8) -> Self {
        // SAFETY: The pointer access is guaranteed by the caller.
        //
        // `_mm256_loadu_si256` requires AVX - implied by `V3`.
        Self(unsafe { _mm256_loadu_si256(ptr as *const __m256i) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(arch: V3, ptr: *const u8, mask: Self::Mask) -> Self {
        // SAFETY: The caller asserts this pointer access is safe. The implementations of
        // `bitmask()` and `Emulated` are trusted.
        Self::from_array(arch, unsafe {
            Emulated::load_simd_masked_logical(Scalar, ptr, mask.bitmask().as_scalar()).to_array()
        })
    }

    #[inline(always)]
    unsafe fn load_simd_first(arch: V3, ptr: *const u8, first: usize) -> Self {
        // SAFETY: The caller asserts this pointer access is safe. The implementations of
        // `bitmask()` and `Emulated` are trusted.
        Self::from_array(arch, unsafe {
            Emulated::load_simd_first(Scalar, ptr, first).to_array()
        })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut u8) {
        // SAFETY: The pointer access is guaranteed by the caller.
        //
        // `_mm256_storeu_si256` requires AVX - implied by `V3`.
        unsafe { _mm256_storeu_si256(ptr.cast::<__m256i>(), self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut u8, mask: Self::Mask) {
        // SAFETY: The caller asserts this pointer access is safe. The implementations of
        // `bitmask()` and `Emulated` are trusted.
        unsafe {
            self.emulated()
                .store_simd_masked_logical(ptr, mask.bitmask().as_scalar())
        }
    }

    #[inline(always)]
    unsafe fn store_simd_first(self, ptr: *mut u8, first: usize) {
        // SAFETY: The caller asserts this pointer access is safe. The implementations of
        // `bitmask()` and `Emulated` are trusted.
        unsafe { self.emulated().store_simd_first(ptr, first) }
    }
}

/// AVX2 does not have native support for comparing unsigned integers.
///
/// Instead, we emulate this by comparing against the element-wise max.
impl SIMDPartialOrd for u8x32 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        // Check that each lane in `self` is not equal to the element-wise maximum.
        //
        // SAFETY: The intrinsics `_mm256_max_epu8`, `_mm256_cmpeq_epi8`, and
        // `_mm256_xor_si256` require AVX2 - all of which are implied by `V3`.
        let m = unsafe {
            let max = _mm256_max_epu8(self.0, other.0);
            _mm256_xor_si256(_mm256_cmpeq_epi8(self.0, max), __m256i::all_ones())
        };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        // Check that each lane in `self` is not equal to the element-wise maximum.
        //
        // SAFETY: The intrinsics `_mm256_min_epu8` and `_mm256_cmpeq_epi8` require AVX2
        // - implied by `V3`.
        let m = unsafe { _mm256_cmpeq_epi8(self.0, _mm256_min_epu8(self.0, other.0)) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDPartialEq for u8x32 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        // SAFETY: The intrinsic `_mm256_cmpeq_epi8` requires AVX2 - implied by `V3`.
        Self::Mask::from_underlying(self.arch(), unsafe { _mm256_cmpeq_epi8(self.0, other.0) })
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        // SAFETY: The intrinsics `_mm256_cmpeq_epi8` and `_mm256_xor_si256` require AVX2
        // - all of which are implied by `V3`.
        let m =
            unsafe { _mm256_xor_si256(_mm256_cmpeq_epi8(self.0, other.0), __m256i::all_ones()) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_u8 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<u8, 32, u8x32>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<u8, 32, u8x32>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<u8, 32, u8x32>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(u8x32, 0x3017fd73c99cc633, V3::new_checked_uncached());
    test_utils::ops::test_sub!(u8x32, 0xfc627f10b5f8db8a, V3::new_checked_uncached());
    test_utils::ops::test_mul!(u8x32, 0x0f4caa80eceaa523, V3::new_checked_uncached());
    test_utils::ops::test_fma!(u8x32, 0xb8f702ba85375041, V3::new_checked_uncached());
    test_utils::ops::test_splitjoin!(u8x32 => u8x16, 0x475a19e80c2f3977, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(u8x32, 0x941757bd5cc641a1, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u8x32, 0xd62d8de09f82ed4e, V3::new_checked_uncached());
}
