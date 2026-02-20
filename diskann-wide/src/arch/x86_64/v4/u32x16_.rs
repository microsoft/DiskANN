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
        v4::u32x8_::u32x8, // direct import for Miri compat
    },
    constant::Const,
    helpers,
    traits::{SIMDMask, SIMDMulAdd, SIMDSelect, SIMDSumTree, SIMDVector},
};

/////
///// 32-bit unsigned integer
/////

macros::x86_define_register!(u32x16, __m512i, BitMask<16, V4>, u32, 16, V4);
macros::x86_define_splat!(u32x16 as i32, _mm512_set1_epi32, "avx512f");
macros::x86_define_default!(u32x16, _mm512_setzero_si512, "avx512f");
macros::x86_splitjoin!(__m512i, u32x16, u32x8);

helpers::unsafe_map_binary_op!(u32x16, std::ops::Add, add, _mm512_add_epi32, "avx512f");
helpers::unsafe_map_binary_op!(u32x16, std::ops::Sub, sub, _mm512_sub_epi32, "avx512f");
helpers::unsafe_map_binary_op!(u32x16, std::ops::Mul, mul, _mm512_mullo_epi32, "avx512f");

helpers::unsafe_map_binary_op!(
    u32x16,
    std::ops::BitAnd,
    bitand,
    _mm512_and_si512,
    "avx512f"
);
helpers::unsafe_map_binary_op!(u32x16, std::ops::BitOr, bitor, _mm512_or_si512, "avx512f");
helpers::unsafe_map_binary_op!(
    u32x16,
    std::ops::BitXor,
    bitxor,
    _mm512_xor_si512,
    "avx512f"
);
helpers::unsafe_map_binary_op!(u32x16, std::ops::Shr, shr, _mm512_srlv_epi32, "avx512f");
helpers::unsafe_map_binary_op!(u32x16, std::ops::Shl, shl, _mm512_sllv_epi32, "avx512f");
helpers::scalar_shift_by_splat!(u32x16, u32);

impl std::ops::Not for u32x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u32x16 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

macros::x86_avx512_load_store!(
    u32x16,
    _mm512_loadu_epi32,
    _mm512_maskz_loadu_epi32,
    _mm512_storeu_epi32,
    _mm512_mask_storeu_epi32,
    i32,
    "avx512f"
);

macros::x86_avx512_int_comparisons!(u32x16, _mm512_cmp_epu32_mask, "avx512f");

impl SIMDSumTree for u32x16 {
    #[inline(always)]
    fn sum_tree(self) -> u32 {
        // SAFETY: `_mm512_reduce_add_epi32` requires AVX512F - implied by V4
        (unsafe { _mm512_reduce_add_epi32(self.0) }) as u32
    }
}

impl SIMDSelect<u32x16> for BitMask<16, V4> {
    #[inline(always)]
    fn select(self, x: u32x16, y: u32x16) -> u32x16 {
        // SAFETY: `_mm512_mask_blend_epi32` requires AVX512F - implied by V4
        u32x16::from_underlying(self.arch(), unsafe {
            _mm512_mask_blend_epi32(self.to_underlying(), y.to_underlying(), x.to_underlying())
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_u32 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_load_simd::<u32, 16, u32x16>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::test_store_simd::<u32, 16, u32x16>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V4::new_checked_uncached() {
            test_utils::ops::test_splat::<u32, 16, u32x16>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(u32x16, 0xa93d54e3aab5d406, V4::new_checked_uncached());
    test_utils::ops::test_sub!(u32x16, 0x4b86c88f6958d930, V4::new_checked_uncached());
    test_utils::ops::test_mul!(u32x16, 0x0ad0524dc17b747a, V4::new_checked_uncached());
    test_utils::ops::test_fma!(u32x16, 0x277aca15e0552388, V4::new_checked_uncached());

    test_utils::ops::test_cmp!(u32x16, 0xdc88c2a44d17c78a, V4::new_checked_uncached());
    test_utils::ops::test_splitjoin!(u32x16 => u32x8, 0x475a19e80c2f3977, V4::new_checked_uncached());
    test_utils::ops::test_select!(u32x16, 0x568c7fcbf00fb03a, V4::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u32x16, 0xc5f7d8d8df0b7b6c, V4::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(u32x16, 0xe533708e69ca0117, V4::new_checked_uncached());
}
