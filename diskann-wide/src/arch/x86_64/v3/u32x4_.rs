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
    traits::{
        SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDSelect, SIMDSumTree, SIMDVector,
    },
};

/////
///// 32-bit floating point
/////

macros::x86_define_register!(u32x4, __m128i, mask32x4, u32, 4, V3);
macros::x86_define_splat!(u32x4 as i32, _mm_set1_epi32, "sse2");
macros::x86_define_default!(u32x4, _mm_setzero_si128, "sse2");

helpers::unsafe_map_binary_op!(u32x4, std::ops::Add, add, _mm_add_epi32, "sse2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::Sub, sub, _mm_sub_epi32, "sse2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::Mul, mul, _mm_mullo_epi32, "sse4.1");

helpers::unsafe_map_binary_op!(u32x4, std::ops::BitAnd, bitand, _mm_and_si128, "sse2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::BitOr, bitor, _mm_or_si128, "sse2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::BitXor, bitxor, _mm_xor_si128, "sse2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::Shr, shr, _mm_srlv_epi32, "avx2");
helpers::unsafe_map_binary_op!(u32x4, std::ops::Shl, shl, _mm_sllv_epi32, "avx2");
helpers::scalar_shift_by_splat!(u32x4, u32);

impl std::ops::Not for u32x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::from_underlying(self.arch(), <Self as SIMDVector>::Underlying::all_ones())
    }
}

impl SIMDMulAdd for u32x4 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        self * rhs + accumulator
    }
}

impl X86LoadStore for u32x4 {
    #[inline(always)]
    unsafe fn load_simd(_: V3, ptr: *const u32) -> Self {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use this intrinsic.
        Self(unsafe { _mm_loadu_si128(ptr as *const __m128i) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(_: V3, ptr: *const u32, mask: Self::Mask) -> Self {
        // MIRI does not support `_mm256_maskload_epi32`.
        // So we go through a kind of convoluted dance to let this be tested by miri.
        //
        // SAFETY: The caller asserts this pointer access is safe and the presence of `V3`
        // means we can use these intrinsics.
        Self(unsafe { _mm_castps_si128(_mm_maskload_ps(ptr as *const f32, mask.to_underlying())) })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut u32) {
        // SAFETY: The caller asserts this pointer access is safe and the presence of `Self`
        // (implicitly `V3`) means we can use this intrinsic.
        unsafe { _mm_storeu_si128(ptr.cast::<__m128i>(), self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut u32, mask: Self::Mask) {
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

impl SIMDPartialEq for u32x4 {
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

impl SIMDPartialOrd for u32x4 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe {
            let max = _mm_max_epu32(self.0, other.0);
            _mm_xor_si128(_mm_cmpeq_epi32(self.0, max), __m128i::all_ones())
        };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe { _mm_cmpeq_epi32(self.0, _mm_min_epu32(self.0, other.0)) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDSumTree for u32x4 {
    #[inline(always)]
    fn sum_tree(self) -> u32 {
        let x = self.to_underlying();
        // SAFETY: Invoking intrinsics on value SIMD types without touching memory.
        unsafe {
            // Do a dance through the `ps` instructions.
            let lo_dual = x;
            let hi_dual = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(x), _mm_castsi128_ps(x)));
            let sum_dual = _mm_add_epi32(lo_dual, hi_dual);

            // Sum the last two elements.
            let lo = sum_dual;
            let hi = _mm_shuffle_epi32(sum_dual, 0x1);
            let sum = _mm_add_epi32(lo, hi);
            _mm_cvtsi128_si32(sum) as u32
        }
    }
}

impl SIMDSelect<u32x4> for mask32x4 {
    #[inline(always)]
    fn select(self, x: u32x4, y: u32x4) -> u32x4 {
        // SAFETY: Compilation of this trait implementation is predicated on the invoked
        // intrinsics being available at compile time.
        u32x4::from_underlying(self.arch(), unsafe {
            _mm_castps_si128(_mm_blendv_ps(
                _mm_castsi128_ps(y.to_underlying()),
                _mm_castsi128_ps(x.to_underlying()),
                _mm_castsi128_ps(self.to_underlying()),
            ))
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
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<u32, 4, u32x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<u32, 4, u32x4>(arch);
        }
    }

    // constructors
    #[cfg(not(miri))]
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<u32, 4, u32x4>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(u32x4, 0x8d7bf28b1c6e2545, V3::new_checked_uncached());
    test_utils::ops::test_sub!(u32x4, 0x4a1c644a1a910bed, V3::new_checked_uncached());
    test_utils::ops::test_mul!(u32x4, 0xf42ee707a808fd10, V3::new_checked_uncached());
    test_utils::ops::test_fma!(u32x4, 0x28540d9936a9e803, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(u32x4, 0xfae27072c6b70885, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u32x4, 0xbe927713ea310164, V3::new_checked_uncached());
    test_utils::ops::test_select!(u32x4, 0xc27a2a7bca1e7877, V3::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(u32x4, 0xb9ac82ab23a855da, V3::new_checked_uncached());
}
