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
    traits::{SIMDMask, SIMDMulAdd, SIMDVector},
};

////////////////////
// 8-bit unsigned //
////////////////////

macros::x86_define_register!(u8x16, __m128i, BitMask<16, V4>, u8, 16, V4);
macros::x86_define_splat!(u8x16 as i8, _mm_set1_epi8, "sse2");
macros::x86_define_default!(u8x16, _mm_setzero_si128, "sse2");
macros::x86_retarget!(u8x16 => v3::u8x16);

impl std::ops::Mul for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        self.from(self.retarget().mul(rhs.retarget()))
    }
}

helpers::unsafe_map_binary_op!(
    u8x16,
    std::ops::Add,
    add,
    _mm_add_epi8, /* intentional: epu8 does not exist and this does the same thing */
    "sse2"
);
helpers::unsafe_map_binary_op!(
    u8x16,
    std::ops::Sub,
    sub,
    _mm_sub_epi8, /* intentional: epu8 does not exist and this does the same thing */
    "sse2"
);

helpers::unsafe_map_binary_op!(u8x16, std::ops::BitAnd, bitand, _mm_and_si128, "sse2");
helpers::unsafe_map_binary_op!(u8x16, std::ops::BitOr, bitor, _mm_or_si128, "sse2");
helpers::unsafe_map_binary_op!(u8x16, std::ops::BitXor, bitxor, _mm_xor_si128, "sse2");

impl std::ops::Shr for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        self.from(self.retarget().shr(rhs.retarget()))
    }
}

impl std::ops::Shl for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        self.from(self.retarget().shl(rhs.retarget()))
    }
}

helpers::scalar_shift_by_splat!(u8x16, u8);

impl std::ops::Not for u8x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u8x16 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_load_store!(
    u8x16,
    _mm_loadu_epi8,
    _mm_maskz_loadu_epi8,
    _mm_storeu_epi8,
    _mm_mask_storeu_epi8,
    i8,
    "avx512bw,avx512vl"
);

macros::x86_avx512_int_comparisons!(u8x16, _mm_cmp_epu8_mask, "avx512bw,avx512vl");

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_u8 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<u8, 16, u8x16>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<u8, 16, u8x16>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<u8, 16, u8x16>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(u8x16, 0x3017fd73c99cc633, V4::new_checked_uncached());
    test_utils::ops::test_sub!(u8x16, 0xfc627f10b5f8db8a, V4::new_checked_uncached());
    test_utils::ops::test_mul!(u8x16, 0x0f4caa80eceaa523, V4::new_checked_uncached());
    test_utils::ops::test_fma!(u8x16, 0xb8f702ba85375041, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(u8x16, 0x941757bd5cc641a1, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u8x16, 0xd62d8de09f82ed4e, V4::new_checked_uncached());
}
