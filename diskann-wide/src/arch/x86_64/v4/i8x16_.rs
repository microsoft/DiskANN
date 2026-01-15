/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::{
    BitMask,
    arch::x86_64::{
        V4,
        common::AllOnes,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v3,
    },
    constant::Const,
    helpers,
    traits::{SIMDAbs, SIMDMask, SIMDMulAdd, SIMDVector},
};

//////////////////
// 8-bit signed //
//////////////////

macros::x86_define_register!(i8x16, __m128i, BitMask<16, V4>, i8, 16, V4);
macros::x86_define_splat!(i8x16, _mm_set1_epi8, "sse2");
macros::x86_define_default!(i8x16, _mm_setzero_si128, "sse2");
macros::x86_retarget!(i8x16 => v3::i8x16);

helpers::unsafe_map_binary_op!(i8x16, std::ops::Add, add, _mm_add_epi8, "sse2");
helpers::unsafe_map_binary_op!(i8x16, std::ops::Sub, sub, _mm_sub_epi8, "sse2");
helpers::unsafe_map_unary_op!(i8x16, SIMDAbs, abs_simd, _mm_abs_epi8, "sse3");

impl std::ops::Mul for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.from(self.retarget().mul(rhs.retarget()))
    }
}

helpers::unsafe_map_binary_op!(i8x16, std::ops::BitAnd, bitand, _mm_and_si128, "sse2");
helpers::unsafe_map_binary_op!(i8x16, std::ops::BitOr, bitor, _mm_or_si128, "sse2");
helpers::unsafe_map_binary_op!(i8x16, std::ops::BitXor, bitxor, _mm_xor_si128, "sse2");

impl std::ops::Shr for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        self.from(self.retarget().shr(rhs.retarget()))
    }
}

impl std::ops::Shl for i8x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        self.from(self.retarget().shl(rhs.retarget()))
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

macros::x86_avx512_load_store!(
    i8x16,
    _mm_loadu_epi8,
    _mm_maskz_loadu_epi8,
    _mm_storeu_epi8,
    _mm_mask_storeu_epi8,
    i8,
    "avx512bw,avx512vl"
);

macros::x86_avx512_int_comparisons!(i8x16, _mm_cmp_epi8_mask, "avx512bw,avx512vl");

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
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<i8, 16, i8x16>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<i8, 16, i8x16>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<i8, 16, i8x16>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i8x16, 0x54696b283652af78, V4::new_checked_uncached());
    test_utils::ops::test_sub!(i8x16, 0x6d0fdf5d8ce94e08, V4::new_checked_uncached());
    test_utils::ops::test_mul!(i8x16, 0x244f0650bef85ba1, V4::new_checked_uncached());
    test_utils::ops::test_fma!(i8x16, 0x9d1b834775effe58, V4::new_checked_uncached());
    test_utils::ops::test_abs!(i8x16, 0x40638a9d09522d1c, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(i8x16, 0x197695206c36808d, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i8x16, 0xeb7bb4da5b84ebbe, V4::new_checked_uncached());
}
