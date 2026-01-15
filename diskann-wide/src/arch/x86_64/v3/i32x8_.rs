/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::{
    SplitJoin,
    arch::x86_64::{
        V3,
        common::AllOnes,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v3::{i8x32, i16x16, i32x4, masks::mask32x8, u8x32},
    },
    constant::Const,
    helpers,
    traits::{
        SIMDAbs, SIMDDotProduct, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDSelect,
        SIMDSumTree, SIMDVector,
    },
};

/////
///// 32-bit floating point
/////

macros::x86_define_register!(i32x8, __m256i, mask32x8, i32, 8, V3);
macros::x86_define_splat!(i32x8, _mm256_set1_epi32, "avx");
macros::x86_define_default!(i32x8, _mm256_setzero_si256, "avx");
macros::x86_splitjoin!(
    i32x8,
    i32x4,
    _mm256_extracti128_si256,
    _mm256_set_m128i,
    "avx2"
);

helpers::unsafe_map_binary_op!(i32x8, std::ops::Add, add, _mm256_add_epi32, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::Sub, sub, _mm256_sub_epi32, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::Mul, mul, _mm256_mullo_epi32, "avx2");
helpers::unsafe_map_unary_op!(i32x8, SIMDAbs, abs_simd, _mm256_abs_epi32, "sse3");

helpers::unsafe_map_binary_op!(i32x8, std::ops::BitAnd, bitand, _mm256_and_si256, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::BitOr, bitor, _mm256_or_si256, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::BitXor, bitxor, _mm256_xor_si256, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::Shr, shr, _mm256_srav_epi32, "avx2");
helpers::unsafe_map_binary_op!(i32x8, std::ops::Shl, shl, _mm256_sllv_epi32, "avx2");
helpers::scalar_shift_by_splat!(i32x8, i32);

impl std::ops::Not for i32x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for i32x8 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

impl X86LoadStore for i32x8 {
    #[inline(always)]
    unsafe fn load_simd(_: V3, ptr: *const i32) -> Self {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use this intrinsic.
        Self(unsafe { _mm256_loadu_si256(ptr as *const __m256i) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(_: V3, ptr: *const i32, mask: Self::Mask) -> Self {
        // MIRI does not support `_mm256_maskload_epi32`.
        // So we go through a kind of convoluted dance to let this be tested by miri.
        //
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use these intrinsics.
        Self(unsafe {
            _mm256_castps_si256(_mm256_maskload_ps(ptr as *const f32, mask.to_underlying()))
        })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut i32) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe { _mm256_storeu_si256(ptr.cast::<__m256i>(), self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut i32, mask: Self::Mask) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe {
            _mm256_maskstore_ps(
                ptr.cast::<f32>(),
                mask.to_underlying(),
                _mm256_castsi256_ps(self.to_underlying()),
            )
        };
    }
}

impl SIMDPartialEq for i32x8 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        Self::Mask::from_underlying(self.arch(), unsafe { _mm256_cmpeq_epi32(self.0, other.0) })
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m =
            unsafe { _mm256_xor_si256(_mm256_cmpeq_epi32(self.0, other.0), __m256i::all_ones()) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDPartialOrd for i32x8 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        Self::Mask::from_underlying(self.arch(), unsafe { _mm256_cmpgt_epi32(other.0, self.0) })
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe { _mm256_cmpeq_epi32(self.0, _mm256_min_epi32(self.0, other.0)) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDSumTree for i32x8 {
    #[inline(always)]
    fn sum_tree(self) -> i32 {
        let x = self.to_underlying();
        // SAFETY: Invoking intrinsics on value SIMD types without touching memory.
        unsafe {
            let hi_quad = _mm256_extracti128_si256(x, 0x1);
            let lo_quad = _mm256_castsi256_si128(x);
            let sum_quad = _mm_add_epi32(lo_quad, hi_quad);

            // Do a dance through the `ps` instructions.
            let lo_dual = sum_quad;
            let hi_dual = _mm_castps_si128(_mm_movehl_ps(
                _mm_castsi128_ps(sum_quad),
                _mm_castsi128_ps(sum_quad),
            ));
            let sum_dual = _mm_add_epi32(lo_dual, hi_dual);

            // Sum the last two elements.
            let lo = sum_dual;
            let hi = _mm_shuffle_epi32(sum_dual, 0x1);
            let sum = _mm_add_epi32(lo, hi);
            _mm_cvtsi128_si32(sum)
        }
    }
}

impl SIMDSelect<i32x8> for mask32x8 {
    #[inline(always)]
    fn select(self, x: i32x8, y: i32x8) -> i32x8 {
        // SAFETY: Compilation of this trait implementation is predicated on the invoked
        // intrinsics being available at compile time.
        i32x8::from_underlying(self.arch(), unsafe {
            _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(y.to_underlying()),
                _mm256_castsi256_ps(x.to_underlying()),
                _mm256_castsi256_ps(self.to_underlying()),
            ))
        })
    }
}

//--------------//
// Dot Products //
//--------------//

impl SIMDDotProduct<i16x16> for i32x8 {
    #[inline(always)]
    fn dot_simd(self, left: i16x16, right: i16x16) -> Self {
        self + Self::from_underlying(
            self.arch(),
            // SAFETY: Gated by CFG.
            unsafe { _mm256_madd_epi16(left.to_underlying(), right.to_underlying()) },
        )
    }
}

/// 8-bit dot products are implemented as two 16-bit dot products. However, this means that
/// need to shuffle the 8-bit vectors so that when we [`SplitJoin::split`] them, the resulting
/// `lo` and `hi` halves contain the appropriate values. In other words, we want the permutation:
/// ```text
/// |-- 32B  --|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|
///  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
///
/// To
///
/// |-- 32B  --|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|
///  0  1  4  5  8  9 12 13 16 17 20 21 24 25 28 29  2  3  6  7 10 11 14 15 18 19 22 23 26 27 30 31
/// ```
/// This way, when we split into two 16-byte registers for conversion to `i16x16`, two dot
/// product instructions will accumulate, for example, all of 0, 1, 2, and 3 into the same
/// destination location.
///
/// AVX2 does not provide us with a single instruction to do this.
///
/// First, we use `_mm256_shuffle_epi8` to shuffle around values within 128-bit lanes like
/// so:
/// ```text
/// |-- 32B  --|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|--- 32B ---|
///  0  1  4  5  8  9 12 13  2  3  6  7 10 11 14 15 16 17 20 21 24 25 28 29 18 19 22 23 26 27 30 31
/// ```
/// Then we use `_mm256_permute4x64_epi64` to shuffle around 64-bit blocks to achieve the
/// final permutation.
#[inline(always)]
fn deinterleave(_arch: V3, v: __m256i) -> __m256i {
    const SHUFFLE_LO: i64 = 0x0D0C0908_05040100;
    const SHUFFLE_HI: i64 = 0x0F0E0B0A_07060302;

    // SAFETY: The intrinsics used here have the following architectural requirements:
    //
    // * `_mm256_set_epi64x`: AVX
    // * `_mm256_shuffle_epi8`: AVX2
    // * `_mm256_permute4x64_epi64`: AVX2
    //
    // All of these are allowed by the `V3` architecture.
    unsafe {
        let mask = _mm256_set_epi64x(SHUFFLE_HI, SHUFFLE_LO, SHUFFLE_HI, SHUFFLE_LO);
        let shuffled = _mm256_shuffle_epi8(v, mask);
        _mm256_permute4x64_epi64(shuffled, 0b11_01_10_00)
    }
}

impl SIMDDotProduct<u8x32, i8x32> for i32x8 {
    #[inline(always)]
    fn dot_simd(self, left: u8x32, right: i8x32) -> Self {
        let arch = self.arch();

        let left = u8x32::from_underlying(arch, deinterleave(arch, left.0));
        let right = i8x32::from_underlying(arch, deinterleave(arch, right.0));

        let left = left.split().map(i16x16::from);
        let right = right.split().map(i16x16::from);
        self.dot_simd(left.lo, right.lo).dot_simd(left.hi, right.hi)
    }
}

impl SIMDDotProduct<i8x32, u8x32> for i32x8 {
    #[inline(always)]
    fn dot_simd(self, left: i8x32, right: u8x32) -> Self {
        self.dot_simd(right, left)
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
            test_utils::test_load_simd::<i32, 8, i32x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<i32, 8, i32x8>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<i32, 8, i32x8>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(i32x8, 0xa93d54e3aab5d406, V3::new_checked_uncached());
    test_utils::ops::test_sub!(i32x8, 0x4b86c88f6958d930, V3::new_checked_uncached());
    test_utils::ops::test_mul!(i32x8, 0x0ad0524dc17b747a, V3::new_checked_uncached());
    test_utils::ops::test_fma!(i32x8, 0x277aca15e0552388, V3::new_checked_uncached());
    test_utils::ops::test_abs!(i32x8, 0x62ca26a68c1a238d, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(i32x8, 0xdc88c2a44d17c78a, V3::new_checked_uncached());
    test_utils::ops::test_splitjoin!(i32x8 => i32x4, 0x475a19e80c2f3977, V3::new_checked_uncached());
    test_utils::ops::test_select!(i32x8, 0xf72531c40af38507, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i32x8, 0xc5f7d8d8df0b7b6c, V3::new_checked_uncached());

    // Dot Products
    test_utils::dot_product::test_dot_product!(
        (i16x16, i16x16) => i32x8, 0x145f89b446c03ff1, V3::new_checked_uncached()
    );
    test_utils::dot_product::test_dot_product!(
        (u8x32, i8x32) => i32x8, 0xa56e0de8ce99136c, V3::new_checked_uncached()
    );
    test_utils::dot_product::test_dot_product!(
        (i8x32, u8x32) => i32x8, 0xbcbcff932237df6d, V3::new_checked_uncached()
    );

    // Reductions
    test_utils::ops::test_sumtree!(i32x8, 0xe533708e69ca0117, V3::new_checked_uncached());
}
