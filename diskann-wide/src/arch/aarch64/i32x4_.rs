/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated, SIMDAbs, SIMDCast, SIMDDotProduct, SIMDMask, SIMDMulAdd, SIMDPartialEq,
    SIMDPartialOrd, SIMDSelect, SIMDSumTree, SIMDVector, constant::Const, helpers,
};

// AArch64 masks
use super::{
    Neon, algorithms, f32x4, i8x8, i8x16, i16x8,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask32x4,
    u8x8, u8x16,
};

// AArch64 intrinsics
use std::arch::{aarch64::*, asm};

///////////////////
// 32-bit signed //
///////////////////

macros::aarch64_define_register!(i32x4, int32x4_t, mask32x4, i32, 4, Neon);
macros::aarch64_define_splat!(i32x4, vmovq_n_s32);
macros::aarch64_define_loadstore!(
    i32x4,
    vld1q_s32,
    algorithms::load_first::i32x4,
    vst1q_s32,
    4
);

helpers::unsafe_map_binary_op!(i32x4, std::ops::Add, add, vaddq_s32, "neon");
helpers::unsafe_map_binary_op!(i32x4, std::ops::Sub, sub, vsubq_s32, "neon");
helpers::unsafe_map_binary_op!(i32x4, std::ops::Mul, mul, vmulq_s32, "neon");
helpers::unsafe_map_unary_op!(i32x4, SIMDAbs, abs_simd, vabsq_s32, "neon");
macros::aarch64_define_fma!(i32x4, vmlaq_s32);

macros::aarch64_define_cmp!(
    i32x4,
    vceqq_s32,
    (vmvnq_u32),
    vcltq_s32,
    vcleq_s32,
    vcgtq_s32,
    vcgeq_s32
);
macros::aarch64_define_bitops!(
    i32x4,
    vmvnq_s32,
    vandq_s32,
    vorrq_s32,
    veorq_s32,
    (
        vshlq_s32,
        32,
        vnegq_s32,
        vminq_u32,
        vreinterpretq_s32_u32,
        vreinterpretq_u32_s32
    ),
    (u32, i32, vmovq_n_s32),
);

impl SIMDSumTree for i32x4 {
    #[inline(always)]
    fn sum_tree(self) -> i32 {
        if cfg!(miri) {
            self.emulated().sum_tree()
        } else {
            // SAFETY: Allowed by the `Neon` architecture.
            unsafe { vaddvq_s32(self.0) }
        }
    }
}

impl SIMDSelect<i32x4> for mask32x4 {
    #[inline(always)]
    fn select(self, x: i32x4, y: i32x4) -> i32x4 {
        // SAFETY: Allowed by the `Neon` architecture.
        i32x4(unsafe { vbslq_s32(self.0, x.0, y.0) })
    }
}

impl SIMDDotProduct<i16x8> for i32x4 {
    #[inline(always)]
    fn dot_simd(self, left: i16x8, right: i16x8) -> Self {
        if cfg!(miri) {
            use crate::AsSIMD;
            self.emulated()
                .dot_simd(left.emulated(), right.emulated())
                .as_simd(self.arch())
        } else {
            let left = left.0;
            let right = right.0;
            // SAFETY: Allowed by the `Neon` architecture.
            unsafe {
                let lo: int32x4_t = vmull_s16(vget_low_s16(left), vget_low_s16(right));
                let hi: int32x4_t = vmull_high_s16(left, right);
                Self(vaddq_s32(self.0, vpaddq_s32(lo, hi)))
            }
        }
    }
}

impl SIMDDotProduct<u8x16, i8x16> for i32x4 {
    #[inline(always)]
    fn dot_simd(self, left: u8x16, right: i8x16) -> Self {
        if cfg!(miri) {
            use crate::AsSIMD;
            self.emulated()
                .dot_simd(left.emulated(), right.emulated())
                .as_simd(self.arch())
        } else {
            use crate::SplitJoin;

            // SAFETY: The intrinsics used here are allowed by the implicit `Neon` architecture.
            unsafe {
                let left = left.split();
                let right = right.split();

                let left_evens: i16x8 = u8x8(vuzp1_u8(left.lo.0, left.hi.0)).into();
                let left_odds: i16x8 = u8x8(vuzp2_u8(left.lo.0, left.hi.0)).into();

                let right_evens: i16x8 = i8x8(vuzp1_s8(right.lo.0, right.hi.0)).into();
                let right_odds: i16x8 = i8x8(vuzp2_s8(right.lo.0, right.hi.0)).into();

                self.dot_simd(left_evens, right_evens)
                    .dot_simd(left_odds, right_odds)
            }
        }
    }
}

impl SIMDDotProduct<i8x16, u8x16> for i32x4 {
    #[inline(always)]
    fn dot_simd(self, left: i8x16, right: u8x16) -> Self {
        self.dot_simd(right, left)
    }
}

impl SIMDDotProduct<i8x16, i8x16> for i32x4 {
    #[inline(always)]
    fn dot_simd(self, left: i8x16, right: i8x16) -> Self {
        if cfg!(miri) {
            use crate::AsSIMD;
            self.emulated()
                .dot_simd(left.emulated(), right.emulated())
                .as_simd(self.arch())
        } else {
            // SAFETY: Instantiating `Neon` implies `dotprod`.
            //
            // We need this wrapper to allow compilation of the underlying ASM when compiling
            // without the `dotprod` feature globally enabled.
            #[target_feature(enable = "dotprod")]
            unsafe fn sdot(mut s: int32x4_t, x: int8x16_t, y: int8x16_t) -> int32x4_t {
                // SAFETY: The `Neon` architecture implies `dotprod`, allowing us to use
                // this intrinsic.
                unsafe {
                    asm!(
                        "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
                        inout(vreg) s,
                        in(vreg) x,
                        in(vreg) y,
                        options(pure, nomem, nostack)
                    );
                }

                s
            }

            // SAFETY: The `Neon` architecture guarantees the `dotprod` feature.
            Self::from_underlying(self.arch(), unsafe { sdot(self.0, left.0, right.0) })
        }
    }
}

//-------------//
// Conversions //
//-------------//

helpers::unsafe_map_cast!(
    i32x4 => (f32, f32x4),
    vcvtq_f32_s32,
    "neon"
);

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
            test_utils::test_load_simd::<i32, 4, i32x4>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<i32, 4, i32x4>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<i32, 4, i32x4>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i32x4, 0x3017fd73c99cc633, test_neon());
    test_utils::ops::test_sub!(i32x4, 0xfc627f10b5f8db8a, test_neon());
    test_utils::ops::test_mul!(i32x4, 0x0f4caa80eceaa523, test_neon());
    test_utils::ops::test_fma!(i32x4, 0xb8f702ba85375041, test_neon());
    test_utils::ops::test_abs!(i32x4, 0xb8f702ba85375041, test_neon());

    test_utils::ops::test_cmp!(i32x4, 0x941757bd5cc641a1, test_neon());

    // Bit ops
    test_utils::ops::test_bitops!(i32x4, 0xd62d8de09f82ed4e, test_neon());
    test_utils::ops::test_select!(i32x4, 0xd62d8de09f82ed4e, test_neon());

    // Dot Products
    test_utils::dot_product::test_dot_product!(
        (i16x8, i16x8) => i32x4,
        0x145f89b446c03ff1,
        test_neon()
    );

    test_utils::dot_product::test_dot_product!(
        (u8x16, i8x16) => i32x4,
        0x145f89b446c03ff1,
        test_neon()
    );

    test_utils::dot_product::test_dot_product!(
        (i8x16, u8x16) => i32x4,
        0x145f89b446c03ff1,
        test_neon()
    );

    test_utils::dot_product::test_dot_product!(
        (i8x16, i8x16) => i32x4,
        0x145f89b446c03ff1,
        test_neon()
    );

    // Reductions
    test_utils::ops::test_sumtree!(i32x4, 0xb9ac82ab23a855da, test_neon());

    // Conversions
    test_utils::ops::test_cast!(i32x4 => f32x4, 0xba8fe343fc9dbeff, test_neon());
}
