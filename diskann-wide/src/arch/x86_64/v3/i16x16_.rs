/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::{
    arch::{
        emulated::Scalar,
        x86_64::{
            V3,
            common::AllOnes,
            macros::{self, X86Default, X86LoadStore, X86Splat},
            v3::i16x8,
        },
    },
    bitmask::BitMask,
    constant::Const,
    emulated::Emulated,
    helpers,
    traits::{AsSIMD, SIMDAbs, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector},
};

///////////////////
// 16-bit signed //
///////////////////

macros::x86_define_register!(i16x16, __m256i, BitMask<16, V3>, i16, 16, V3);
macros::x86_define_splat!(i16x16, _mm256_set1_epi16, "avx");
macros::x86_define_default!(i16x16, _mm256_setzero_si256, "avx");
macros::x86_splitjoin!(
    i16x16,
    i16x8,
    _mm256_extracti128_si256,
    _mm256_set_m128i,
    "avx2"
);

helpers::unsafe_map_binary_op!(i16x16, std::ops::Add, add, _mm256_add_epi16, "avx2");
helpers::unsafe_map_binary_op!(i16x16, std::ops::Sub, sub, _mm256_sub_epi16, "avx2");
helpers::unsafe_map_binary_op!(i16x16, std::ops::Mul, mul, _mm256_mullo_epi16, "avx2");
helpers::unsafe_map_unary_op!(i16x16, SIMDAbs, abs_simd, _mm256_abs_epi16, "avx2");

helpers::unsafe_map_binary_op!(i16x16, std::ops::BitAnd, bitand, _mm256_and_si256, "avx2");
helpers::unsafe_map_binary_op!(i16x16, std::ops::BitOr, bitor, _mm256_or_si256, "avx2");
helpers::unsafe_map_binary_op!(i16x16, std::ops::BitXor, bitxor, _mm256_xor_si256, "avx2");

impl std::ops::Shr for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        self.emulated().shr(rhs.emulated()).as_simd(self.arch())
    }
}

impl std::ops::Shl for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        self.emulated().shl(rhs.emulated()).as_simd(self.arch())
    }
}

helpers::scalar_shift_by_splat!(i16x16, i16);

impl std::ops::Not for i16x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for i16x16 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

impl SIMDPartialEq for i16x16 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        self.emulated()
            .eq_simd(other.emulated())
            .as_arch(self.arch())
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        self.emulated()
            .ne_simd(other.emulated())
            .as_arch(self.arch())
    }
}

impl SIMDPartialOrd for i16x16 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        self.emulated()
            .lt_simd(other.emulated())
            .as_arch(self.arch())
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        self.emulated()
            .le_simd(other.emulated())
            .as_arch(self.arch())
    }
}

impl X86LoadStore for i16x16 {
    #[inline(always)]
    unsafe fn load_simd(_: V3, ptr: *const i16) -> Self {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use this intrinsic.
        Self(unsafe { _mm256_loadu_si256(ptr as *const __m256i) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(arch: V3, ptr: *const i16, mask: Self::Mask) -> Self {
        // SAFETY: The caller asserts this pointer access is safe and the implementation of
        // `Emulated` is trusted.
        Self::from_array(arch, unsafe {
            Emulated::load_simd_masked_logical(Scalar, ptr, mask.as_scalar()).to_array()
        })
    }

    #[inline(always)]
    unsafe fn load_simd_first(arch: V3, ptr: *const i16, first: usize) -> Self {
        // SAFETY: The caller asserts this pointer access is safe.
        Self::from_array(arch, unsafe {
            Emulated::load_simd_first(Scalar, ptr, first).to_array()
        })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut i16) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe { _mm256_storeu_si256(ptr as *mut __m256i, self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut i16, mask: Self::Mask) {
        // SAFETY: The caller asserts this pointer access is safe and the implementation of
        // `Emulated` is trusted.
        unsafe {
            self.emulated()
                .store_simd_masked_logical(ptr, mask.as_scalar())
        }
    }

    #[inline(always)]
    unsafe fn store_simd_first(self, ptr: *mut i16, first: usize) {
        // SAFETY: The caller asserts this pointer access is safe and the implementation of
        // `Emulated` is trusted.
        unsafe { self.emulated().store_simd_first(ptr, first) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_i16 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    // Test loading logic - ensure that no out of bounds accesses are made.
    // In particular, this is meant to be run under `Miri` to ensure that our guarantees
    // regarding out-of-bounds accesses are honored.
    #[test]
    fn miri_test_load() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<i16, 16, i16x16>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<i16, 16, i16x16>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<i16, 16, i16x16>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i16x16, 0xcd7a8ad4a3b7760c, V3::new_checked_uncached());
    test_utils::ops::test_sub!(i16x16, 0x9e5abc35dda03aa5, V3::new_checked_uncached());
    test_utils::ops::test_mul!(i16x16, 0x47159f68b972ad07, V3::new_checked_uncached());
    test_utils::ops::test_fma!(i16x16, 0xed4244971a90e2f0, V3::new_checked_uncached());
    test_utils::ops::test_abs!(i16x16, 0x9aaa0504d1598348, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(i16x16, 0x242f7f6c3709920d, V3::new_checked_uncached());
    test_utils::ops::test_splitjoin!(i16x16 => i16x8, 0x05931ca2c2577fae, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i16x16, 0xba0be356b04d6427, V3::new_checked_uncached());
}
