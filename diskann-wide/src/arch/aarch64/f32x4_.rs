/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use half::f16;

use crate::{
    Emulated, SIMDAbs, SIMDMask, SIMDMinMax, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDSelect,
    SIMDSumTree, SIMDVector, constant::Const, helpers,
};

// AArch64 masks
use super::{
    Neon, algorithms, f16x4, f32x2,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask32x4,
};

// AArch64 intrinsics
use std::arch::{aarch64::*, asm};

/////////////////////
// 32-bit floating //
/////////////////////

macros::aarch64_define_register!(f32x4, float32x4_t, mask32x4, f32, 4, Neon);
macros::aarch64_define_splat!(f32x4, vmovq_n_f32);
macros::aarch64_define_loadstore!(
    f32x4,
    vld1q_f32,
    algorithms::load_first::f32x4,
    vst1q_f32,
    4
);
macros::aarch64_splitjoin!(f32x4, f32x2, vget_low_f32, vget_high_f32, vcombine_f32);

helpers::unsafe_map_binary_op!(f32x4, std::ops::Add, add, vaddq_f32, "neon");
helpers::unsafe_map_binary_op!(f32x4, std::ops::Sub, sub, vsubq_f32, "neon");
helpers::unsafe_map_binary_op!(f32x4, std::ops::Mul, mul, vmulq_f32, "neon");
helpers::unsafe_map_unary_op!(f32x4, SIMDAbs, abs_simd, vabsq_f32, "neon");
macros::aarch64_define_fma!(f32x4, vfmaq_f32);

impl SIMDMinMax for f32x4 {
    #[inline(always)]
    fn min_simd(self, rhs: Self) -> Self {
        // SAFETY: `vminnmq_f32` requires "neon", implied by the `Neon` architecture.
        Self(unsafe { vminnmq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn min_simd_standard(self, rhs: Self) -> Self {
        // SAFETY: `vminnmq_f32` requires "neon", implied by the `Neon` architecture.
        Self(unsafe { vminnmq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn max_simd(self, rhs: Self) -> Self {
        // SAFETY: `vminnmq_f32` requires "neon", implied by the `Neon` architecture.
        Self(unsafe { vmaxnmq_f32(self.0, rhs.0) })
    }

    #[inline(always)]
    fn max_simd_standard(self, rhs: Self) -> Self {
        // SAFETY: `vminnmq_f32` requires "neon", implied by the `Neon` architecture.
        Self(unsafe { vmaxnmq_f32(self.0, rhs.0) })
    }
}

macros::aarch64_define_cmp!(
    f32x4,
    vceqq_f32,
    (vmvnq_u32),
    vcltq_f32,
    vcleq_f32,
    vcgtq_f32,
    vcgeq_f32
);

impl SIMDSumTree for f32x4 {
    #[inline(always)]
    fn sum_tree(self) -> f32 {
        // Miri does not support `vaddv_f32`.
        if cfg!(miri) {
            self.emulated().sum_tree()
        } else {
            // NOTE: `vaddvq` does not do a tree reduction, so we need to do a bit of work
            // manually.
            let x = self.to_underlying();
            // SAFETY: Allowed by the implicit `Neon` architecture.
            unsafe {
                let low = vget_low_f32(x);
                let high = vget_high_f32(x);
                vaddv_f32(vadd_f32(low, high))
            }
        }
    }
}

impl SIMDSelect<f32x4> for mask32x4 {
    #[inline(always)]
    fn select(self, x: f32x4, y: f32x4) -> f32x4 {
        // SAFETY: Allowed by the implicit `Neon` architecture.
        f32x4(unsafe { vbslq_f32(self.0, x.0, y.0) })
    }
}

//------------//
// Conversion //
//------------//

// Rust does not expose any of the f16 style intrinsics, so we need to drop down straight
// into inline assembly.
impl From<f16x4> for f32x4 {
    #[inline(always)]
    fn from(value: f16x4) -> f32x4 {
        if cfg!(miri) {
            Self::from_array(value.arch(), value.to_array().map(crate::cast_f16_to_f32))
        } else {
            let raw = value.0;
            let result: float32x4_t;
            // SAFETY: The instruction we are running is available with the `neon` platform,
            // just not exposed by Rust's intrinsics.
            unsafe {
                asm!(
                    "fcvtl {0:v}.4s, {1:v}.4h",
                    out(vreg) result,
                    in(vreg) raw,
                    options(pure, nomem, nostack)
                );
            }
            Self(result)
        }
    }
}

impl crate::SIMDCast<f16> for f32x4 {
    type Cast = f16x4;
    #[inline(always)]
    fn simd_cast(self) -> f16x4 {
        if cfg!(miri) {
            f16x4::from_array(self.arch(), self.to_array().map(crate::cast_f32_to_f16))
        } else {
            let raw = self.0;
            let result: uint16x4_t;
            // SAFETY: The instruction we are running is available with the `neon` platform,
            // just not exposed by Rust's intrinsics.
            unsafe {
                asm!(
                    "fcvtn {0:v}.4h, {1:v}.4s",
                    out(vreg) result,
                    in(vreg) raw,
                    options(pure, nomem, nostack)
                );
            }
            f16x4::from_underlying(self.arch(), result)
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arch::aarch64::test_neon, reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = test_neon() {
            test_utils::test_load_simd::<f32, 4, f32x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<f32, 4, f32x4>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<f32, 4, f32x4>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(f32x4, 0xcd7a8fea9a3fb727, test_neon());
    test_utils::ops::test_sub!(f32x4, 0x3f6562c94c923238, test_neon());
    test_utils::ops::test_mul!(f32x4, 0x07e48666c0fc564c, test_neon());
    test_utils::ops::test_fma!(f32x4, 0xcfde9d031302cf2c, test_neon());
    test_utils::ops::test_abs!(f32x4, 0xb8f702ba85375041, test_neon());
    test_utils::ops::test_minmax!(f32x4, 0x6d7fc8ed6d852187, test_neon());
    test_utils::ops::test_splitjoin!(f32x4 => f32x2, 0xa4d00a4d04293967, test_neon());

    test_utils::ops::test_cmp!(f32x4, 0xc4f468b224622326, test_neon());
    test_utils::ops::test_select!(f32x4, 0xef24013b8578637c, test_neon());

    test_utils::ops::test_sumtree!(f32x4, 0x828bd890a470dc4d, test_neon());

    // Conversions
    test_utils::ops::test_lossless_convert!(f16x4 => f32x4, 0xecba3008eae54ce7, test_neon());

    test_utils::ops::test_cast!(f32x4 => f16x4, 0xba8fe343fc9dbeff, test_neon());
}
