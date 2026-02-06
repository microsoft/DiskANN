/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use crate::{
    arch::x86_64::{
        V3,
        macros::{self, X86Default, X86LoadStore, X86Splat},
        v3::masks::mask32x4,
    },
    constant::Const,
    helpers,
    traits::{
        SIMDAbs, SIMDMask, SIMDMinMax, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDSelect,
        SIMDSumTree, SIMDVector,
    },
};

/////////////////////
// 32-bit floating //
/////////////////////

macros::x86_define_register!(f32x4, __m128, mask32x4, f32, 4, V3);
macros::x86_define_splat!(f32x4, _mm_set1_ps, "sse");
macros::x86_define_default!(f32x4, _mm_setzero_ps, "sse");

helpers::unsafe_map_binary_op!(f32x4, std::ops::Add, add, _mm_add_ps, "sse");
helpers::unsafe_map_binary_op!(f32x4, std::ops::Sub, sub, _mm_sub_ps, "sse");
helpers::unsafe_map_binary_op!(f32x4, std::ops::Mul, mul, _mm_mul_ps, "sse");

impl f32x4 {
    #[inline(always)]
    fn is_nan(self) -> mask32x4 {
        // NOTE: `_CMP_UNORD_Q` returns `true` if either argument is NaN. Since we compare
        // `self` with `self`, this returns `true` exactly when `self` is NaN.
        mask32x4::from_underlying(
            self.arch(),
            // SAFETY: `_mm_castps_si128` requires SSE2 and `_mm_cmp_ps` requires AVX,
            // both of which are implied by the implicit V3 architecture.
            unsafe { _mm_castps_si128(_mm_cmp_ps(self.0, self.0, _CMP_UNORD_Q)) },
        )
    }
}

impl SIMDMulAdd for f32x4 {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        // SAFETY: Invoking an intrinsic with valid intrinsic types.
        Self(unsafe { _mm_fmadd_ps(self.0, rhs.0, accumulator.0) })
    }
}

impl SIMDMinMax for f32x4 {
    #[inline(always)]
    fn min_simd(self, rhs: Self) -> Self {
        // SAFETY: `_mm_min_ps` requires SSE, which is implied by the V3 architecture.
        Self(unsafe { _mm_min_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn min_simd_standard(self, rhs: Self) -> Self {
        // NOTE: The behavior of the non-IEEE compliant min is order dependent and thus
        // it is important that the order of the arguments is swapped.
        let min = rhs.min_simd(self);
        self.is_nan().select(rhs, min)
    }

    #[inline(always)]
    fn max_simd(self, rhs: Self) -> Self {
        // SAFETY: `_mm_max_ps` requires SSE, which is implied by the V3 architecture.
        Self(unsafe { _mm_max_ps(self.0, rhs.0) })
    }

    #[inline(always)]
    fn max_simd_standard(self, rhs: Self) -> Self {
        // NOTE: The behavior of the non-IEEE compliant max is order dependent and thus
        // it is important that the order of the arguments is swapped.
        let max = rhs.max_simd(self);
        self.is_nan().select(rhs, max)
    }
}

impl SIMDAbs for f32x4 {
    #[inline(always)]
    fn abs_simd(self) -> Self {
        use super::u32x4;

        // There is not a hardware intrinsic for `f32` absolute value.
        // Instead, we just mask out the sign bits.
        let mask = <u32x4 as SIMDVector>::splat(self.arch(), 0x7FFF_FFFF).to_underlying();

        // SAFETY: The intrinsics used here are limited to those available in the V3
        // architecture.
        Self(unsafe {
            _mm_castsi128_ps(_mm_and_si128(_mm_castps_si128(self.to_underlying()), mask))
        })
    }
}

impl X86LoadStore for f32x4 {
    #[inline(always)]
    unsafe fn load_simd(_: V3, ptr: *const f32) -> Self {
        // SAFETY: This is valid provided the preconditions for `load_simd` hold.
        Self(unsafe { _mm_loadu_ps(ptr) })
    }

    #[inline(always)]
    unsafe fn load_simd_masked_logical(_: V3, ptr: *const f32, mask: Self::Mask) -> Self {
        // SAFETY: This is valid provided the preconditions for `load_simd` hold.
        //
        // Fault suppression will protect us from accessing invalid memory
        Self(unsafe { _mm_maskload_ps(ptr, mask.to_underlying()) })
    }

    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut f32) {
        // SAFETY: This is valid provided the preconditions for `store_simd` hold.
        unsafe { _mm_storeu_ps(ptr, self.to_underlying()) }
    }

    #[inline(always)]
    unsafe fn store_simd_masked_logical(self, ptr: *mut f32, mask: Self::Mask) {
        // SAFETY: This is valid provided the preconditions for `store_simd` hold.
        //
        // Fault suppression will protect us from accessing invalid memory
        unsafe { _mm_maskstore_ps(ptr, mask.to_underlying(), self.to_underlying()) }
    }
}

impl SIMDPartialEq for f32x4 {
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe { _mm_castps_si128(_mm_cmp_ps(self.0, other.0, _CMP_EQ_OQ)) };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG
        let m = unsafe { _mm_castps_si128(_mm_cmp_ps(self.0, other.0, _CMP_NEQ_UQ)) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDPartialOrd for f32x4 {
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG.
        let m = unsafe { _mm_castps_si128(_mm_cmp_ps(self.0, other.0, _CMP_LT_OQ)) };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG.
        let m = unsafe { _mm_castps_si128(_mm_cmp_ps(self.0, other.0, _CMP_LE_OQ)) };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn gt_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG.
        let m = unsafe { _mm_castps_si128(_mm_cmp_ps(self.0, other.0, _CMP_GT_OQ)) };
        Self::Mask::from_underlying(self.arch(), m)
    }

    #[inline(always)]
    fn ge_simd(self, other: Self) -> Self::Mask {
        // SAFETY: Gated by CFG.
        let m = unsafe { _mm_castps_si128(_mm_cmp_ps(self.0, other.0, _CMP_GE_OQ)) };
        Self::Mask::from_underlying(self.arch(), m)
    }
}

impl SIMDSumTree for f32x4 {
    #[inline(always)]
    fn sum_tree(self) -> f32 {
        let x = self.to_underlying();
        // SAFETY: Gated by CFG.
        unsafe {
            // loDual = ( -, -, x1, x0 )
            let lo_dual = x;
            // hiDual = ( -, -, x3, x2 )
            let hi_dual = _mm_movehl_ps(x, x);
            // sumDual = ( -, -, x1 + x3, x0 + x2 )
            let sum_dual = _mm_add_ps(lo_dual, hi_dual);
            // lo = ( -, -, -, x0 + x2 )
            let lo = sum_dual;
            // hi = ( -, -, -, x1 + x3 )
            let hi = _mm_shuffle_ps(sum_dual, sum_dual, 0x1);
            // sum = ( -, -, -, x0 + x1 + x2 + x3 )
            let sum = _mm_add_ss(lo, hi);
            _mm_cvtss_f32(sum)
        }
    }
}

impl SIMDSelect<f32x4> for mask32x4 {
    #[inline(always)]
    fn select(self, x: f32x4, y: f32x4) -> f32x4 {
        // SAFETY: Compilation of this trait implementation is predicated on the invoked
        // intrinsics being available at compile time.
        f32x4::from_underlying(self.arch(), unsafe {
            _mm_blendv_ps(
                y.to_underlying(),
                x.to_underlying(),
                _mm_castsi128_ps(self.to_underlying()),
            )
        })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_f32 {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    // Test loading logic - ensure that no out of bounds accesses are made.
    // In particular, this is meant to be run under `Miri` to ensure that our guarantees
    // regarding out-of-bounds accesses are honored.
    #[test]
    fn miri_test_load() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<f32, 4, f32x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<f32, 4, f32x4>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<f32, 4, f32x4>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(f32x4, 0xcd7a8fea9a3fb727, V3::new_checked_uncached());
    test_utils::ops::test_sub!(f32x4, 0x3f6562c94c923238, V3::new_checked_uncached());
    test_utils::ops::test_mul!(f32x4, 0x07e48666c0fc564c, V3::new_checked_uncached());
    test_utils::ops::test_fma!(f32x4, 0xcfde9d031302cf2c, V3::new_checked_uncached());
    test_utils::ops::test_minmax!(f32x4, 0x6d7fc8ed6d852187, V3::new_checked_uncached());
    test_utils::ops::test_abs!(f32x4, 0x8e6d9944c9c43a74, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(f32x4, 0xc4f468b224622326, V3::new_checked_uncached());
    test_utils::ops::test_select!(f32x4, 0xef24013b8578637c, V3::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(f32x4, 0x828bd890a470dc4d, V3::new_checked_uncached());
}
