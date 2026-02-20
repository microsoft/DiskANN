/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

// x86 specific
use crate::{
    arch::{
        emulated::Scalar,
        x86_64::{
            V3,
            algorithms::__load_first_u16_of_16_bytes,
            common::AllOnes,
            macros::{self, X86Default, X86LoadStore, X86Splat},
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

macros::x86_define_register!(i16x8, __m128i, BitMask<8, V3>, i16, 8, V3);
macros::x86_define_splat!(i16x8, _mm_set1_epi16, "sse2");
macros::x86_define_default!(i16x8, _mm_setzero_si128, "sse2");

helpers::unsafe_map_binary_op!(i16x8, std::ops::Add, add, _mm_add_epi16, "sse2");
helpers::unsafe_map_binary_op!(i16x8, std::ops::Sub, sub, _mm_sub_epi16, "sse2");
helpers::unsafe_map_binary_op!(i16x8, std::ops::Mul, mul, _mm_mullo_epi16, "sse2");
helpers::unsafe_map_unary_op!(i16x8, SIMDAbs, abs_simd, _mm_abs_epi16, "ssse3");

helpers::unsafe_map_binary_op!(i16x8, std::ops::BitAnd, bitand, _mm_and_si128, "sse2");
helpers::unsafe_map_binary_op!(i16x8, std::ops::BitOr, bitor, _mm_or_si128, "sse2");
helpers::unsafe_map_binary_op!(i16x8, std::ops::BitXor, bitxor, _mm_xor_si128, "sse2");

impl std::ops::Shr for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        self.emulated().shr(rhs.emulated()).as_simd(self.arch())
    }
}

impl std::ops::Shl for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        self.emulated().shl(rhs.emulated()).as_simd(self.arch())
    }
}

helpers::scalar_shift_by_splat!(i16x8, i16);

impl std::ops::Not for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for i16x8 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

impl SIMDPartialEq for i16x8 {
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

impl SIMDPartialOrd for i16x8 {
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

impl X86LoadStore for i16x8 {
    #[inline(always)]
    unsafe fn load_simd(_: V3, ptr: *const i16) -> Self {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use this intrinsic.
        Self(unsafe { _mm_loadu_si128(ptr as *const __m128i) })
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
        Self(unsafe { __load_first_u16_of_16_bytes(arch, ptr as *const u16, first) })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut i16) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe { _mm_storeu_si128(ptr as *mut __m128i, self.to_underlying()) }
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
            test_utils::test_load_simd::<i16, 8, i16x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<i16, 8, i16x8>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<i16, 8, i16x8>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i16x8, 0xe4a01a0464d8f0f5, V3::new_checked_uncached());
    test_utils::ops::test_sub!(i16x8, 0x22a4acd07cc44364, V3::new_checked_uncached());
    test_utils::ops::test_mul!(i16x8, 0x7414b12b626e37b0, V3::new_checked_uncached());
    test_utils::ops::test_fma!(i16x8, 0xfa36384b583b2ee7, V3::new_checked_uncached());
    test_utils::ops::test_abs!(i16x8, 0x79e6e4033868a5bc, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(i16x8, 0xe48cca9a63cb982f, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i16x8, 0x2dd8796bda1cb89d, V3::new_checked_uncached());
}
