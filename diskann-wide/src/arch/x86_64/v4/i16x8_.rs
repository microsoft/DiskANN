/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

// x86 specific
use crate::{
    SIMDMask,
    arch::x86_64::{
        V4,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v3,
    },
    bitmask::BitMask,
    constant::Const,
    helpers,
    traits::{SIMDAbs, SIMDMulAdd, SIMDVector},
};

///////////////////
// 16-bit signed //
///////////////////

macros::x86_define_register!(i16x8, __m128i, BitMask<8, V4>, i16, 8, V4);
macros::x86_define_splat!(i16x8, _mm_set1_epi16, "sse2");
macros::x86_define_default!(i16x8, _mm_setzero_si128, "sse2");
macros::x86_retarget!(i16x8 => v3::i16x8);

helpers::unsafe_map_binary_op!(i16x8, std::ops::Add, add, _mm_add_epi16, "sse2");
helpers::unsafe_map_binary_op!(i16x8, std::ops::Sub, sub, _mm_sub_epi16, "sse2");
helpers::unsafe_map_binary_op!(i16x8, std::ops::Mul, mul, _mm_mullo_epi16, "sse2");
helpers::unsafe_map_unary_op!(i16x8, SIMDAbs, abs_simd, _mm_abs_epi16, "ssse3");

helpers::unsafe_map_binary_op!(i16x8, std::ops::BitAnd, bitand, _mm_and_si128, "sse2");
helpers::unsafe_map_binary_op!(i16x8, std::ops::BitOr, bitor, _mm_or_si128, "sse2");
helpers::unsafe_map_binary_op!(i16x8, std::ops::BitXor, bitxor, _mm_xor_si128, "sse2");
helpers::unsafe_map_binary_op!(i16x8, std::ops::Shr, shr, _mm_srav_epi16, "avx512bw,avx512vl");
helpers::unsafe_map_binary_op!(i16x8, std::ops::Shl, shl, _mm_sllv_epi16, "avx512bw,avx512vl");
helpers::scalar_shift_by_splat!(i16x8, i16);

impl std::ops::Not for i16x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self.from(!self.retarget())
    }
}

impl SIMDMulAdd for i16x8 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_int_comparisons!(i16x8, _mm_cmp_epi16_mask, "avx512bw,avx512vl");

macros::x86_avx512_load_store!(
    i16x8,
    _mm_loadu_epi16,
    _mm_maskz_loadu_epi16,
    _mm_storeu_epi16,
    _mm_mask_storeu_epi16,
    i16,
    "avx512bw,avx512vl"
);

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
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<i16, 8, i16x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<i16, 8, i16x8>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<i16, 8, i16x8>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i16x8, 0xe4a01a0464d8f0f5, V4::new_checked_uncached());
    test_utils::ops::test_sub!(i16x8, 0x22a4acd07cc44364, V4::new_checked_uncached());
    test_utils::ops::test_mul!(i16x8, 0x7414b12b626e37b0, V4::new_checked_uncached());
    test_utils::ops::test_fma!(i16x8, 0xfa36384b583b2ee7, V4::new_checked_uncached());
    test_utils::ops::test_abs!(i16x8, 0x79e6e4033868a5bc, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(i16x8, 0xe48cca9a63cb982f, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i16x8, 0x2dd8796bda1cb89d, V4::new_checked_uncached());
}
