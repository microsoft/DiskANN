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
        v3::{masks::mask64x4, u64x2},
    },
    constant::Const,
    helpers,
    traits::{AsSIMD, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector},
};

/////
///// 64-bit unsigned integer
/////

macros::x86_define_register!(u64x4, __m256i, mask64x4, u64, 4, V3);
macros::x86_define_splat!(u64x4 as i64, _mm256_set1_epi64x, "avx");
macros::x86_define_default!(u64x4, _mm256_setzero_si256, "avx");
macros::x86_splitjoin!(
    u64x4,
    u64x2,
    _mm256_extracti128_si256,
    _mm256_set_m128i,
    "avx2"
);

helpers::unsafe_map_binary_op!(u64x4, std::ops::Add, add, _mm256_add_epi64, "avx2");
helpers::unsafe_map_binary_op!(u64x4, std::ops::Sub, sub, _mm256_sub_epi64, "avx2");

// Emulated multiplication.
// This actually generates not terrible code.
impl std::ops::Mul for u64x4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        (self.emulated() * rhs.emulated()).as_simd(self.arch())
    }
}

helpers::unsafe_map_binary_op!(u64x4, std::ops::BitAnd, bitand, _mm256_and_si256, "avx2");
helpers::unsafe_map_binary_op!(u64x4, std::ops::BitOr, bitor, _mm256_or_si256, "avx2");
helpers::unsafe_map_binary_op!(u64x4, std::ops::BitXor, bitxor, _mm256_xor_si256, "avx2");
helpers::unsafe_map_binary_op!(u64x4, std::ops::Shr, shr, _mm256_srlv_epi64, "avx2");
helpers::unsafe_map_binary_op!(u64x4, std::ops::Shl, shl, _mm256_sllv_epi64, "avx2");

helpers::scalar_shift_by_splat!(u64x4, u64);

impl std::ops::Not for u64x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u64x4 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

impl X86LoadStore for u64x4 {
    #[inline(always)]
    unsafe fn load_simd(_: V3, ptr: *const u64) -> Self {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use this intrinsic.
        Self(unsafe { _mm256_loadu_si256(ptr as *const __m256i) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(_: V3, ptr: *const u64, mask: Self::Mask) -> Self {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use this intrinsic.
        Self(unsafe { _mm256_maskload_epi64(ptr as *const i64, mask.to_underlying()) })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut u64) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe { _mm256_storeu_si256(ptr.cast::<__m256i>(), self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut u64, mask: Self::Mask) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe {
            _mm256_maskstore_epi64(
                ptr.cast::<i64>(),
                mask.to_underlying(),
                self.to_underlying(),
            )
        };
    }
}

impl SIMDPartialEq for u64x4 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe { _mm256_cmpeq_epi64(self.0, other.0) };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m =
            unsafe { _mm256_xor_si256(_mm256_cmpeq_epi64(self.0, other.0), __m256i::all_ones()) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

// Notes: 64-bit `max` and `min` require AVX512-F and AVX512-VL
//
// We use an emulated implementation for AVX2.
impl SIMDPartialOrd for u64x4 {
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
            test_utils::test_load_simd::<u64, 4, u64x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<u64, 4, u64x4>(arch);
        }
    }

    // constructors
    #[cfg(not(miri))]
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<u64, 4, u64x4>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(u64x4, 0xeaee2fd0398fe357, V3::new_checked_uncached());
    test_utils::ops::test_sub!(u64x4, 0x40af040b0c2c1e28, V3::new_checked_uncached());
    test_utils::ops::test_mul!(u64x4, 0x68f68933a29c5ea9, V3::new_checked_uncached());
    test_utils::ops::test_fma!(u64x4, 0x31bc9d25e91e6744, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(u64x4, 0x0beda0dd5141ec40, V3::new_checked_uncached());
    test_utils::ops::test_splitjoin!(u64x4 => u64x2, 0xb151fcd6141b10c9, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u64x4, 0xb1ac2e16327a8d5e, V3::new_checked_uncached());
}
