/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated, SIMDDotProduct, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDSelect,
    SIMDSumTree, SIMDVector, constant::Const, helpers,
};

// AArch64 masks
use super::{
    Neon, algorithms,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask32x4,
    u8x16,
};

// AArch64 intrinsics
use std::arch::{aarch64::*, asm};

/////////////////////
// 32-bit unsigned //
/////////////////////

macros::aarch64_define_register!(u32x4, uint32x4_t, mask32x4, u32, 4, Neon);
macros::aarch64_define_splat!(u32x4, vmovq_n_u32);
macros::aarch64_define_loadstore!(
    u32x4,
    vld1q_u32,
    algorithms::load_first::u32x4,
    vst1q_u32,
    4
);

helpers::unsafe_map_binary_op!(u32x4, std::ops::Add, add, vaddq_u32, "neon");
helpers::unsafe_map_binary_op!(u32x4, std::ops::Sub, sub, vsubq_u32, "neon");
helpers::unsafe_map_binary_op!(u32x4, std::ops::Mul, mul, vmulq_u32, "neon");
macros::aarch64_define_fma!(u32x4, vmlaq_u32);

macros::aarch64_define_cmp!(
    u32x4,
    vceqq_u32,
    (vmvnq_u32),
    vcltq_u32,
    vcleq_u32,
    vcgtq_u32,
    vcgeq_u32
);
macros::aarch64_define_bitops!(
    u32x4,
    vmvnq_u32,
    vandq_u32,
    vorrq_u32,
    veorq_u32,
    (
        vshlq_u32,
        32,
        vnegq_s32,
        vminq_u32,
        vreinterpretq_s32_u32,
        std::convert::identity
    ),
    (u32, i32, vmovq_n_s32),
);

impl SIMDSumTree for u32x4 {
    #[inline(always)]
    fn sum_tree(self) -> u32 {
        if cfg!(miri) {
            self.emulated().sum_tree()
        } else {
            // SAFETY: Allowed by the `Neon` architecture.
            unsafe { vaddvq_u32(self.0) }
        }
    }
}

impl SIMDSelect<u32x4> for mask32x4 {
    #[inline(always)]
    fn select(self, x: u32x4, y: u32x4) -> u32x4 {
        // SAFETY: Allowed by the `Neon` architecture.
        u32x4(unsafe { vbslq_u32(self.0, x.0, y.0) })
    }
}

impl SIMDDotProduct<u8x16, u8x16> for u32x4 {
    #[inline(always)]
    fn dot_simd(self, left: u8x16, right: u8x16) -> Self {
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
            unsafe fn udot(mut s: uint32x4_t, x: uint8x16_t, y: uint8x16_t) -> uint32x4_t {
                // SAFETY: The `Neon` architecture implies `dotprod`, allowing us to use
                // this intrinsic.
                unsafe {
                    asm!(
                        "udot {0:v}.4s, {1:v}.16b, {2:v}.16b",
                        inout(vreg) s,
                        in(vreg) x,
                        in(vreg) y,
                        options(pure, nomem, nostack)
                    );
                }

                s
            }

            // SAFETY: The `Neon` architecture guarantees the `dotprod` feature.
            Self::from_underlying(self.arch(), unsafe { udot(self.0, left.0, right.0) })
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        test_utils::test_load_simd::<u32, 4, u32x4>(Neon::new_checked().unwrap());
    }

    #[test]
    fn miri_test_store() {
        test_utils::test_store_simd::<u32, 4, u32x4>(Neon::new_checked().unwrap());
    }

    // constructors
    #[test]
    fn test_constructors() {
        test_utils::ops::test_splat::<u32, 4, u32x4>(Neon::new_checked().unwrap());
    }

    // Ops
    test_utils::ops::test_add!(u32x4, 0x3017fd73c99cc633, Neon::new_checked());
    test_utils::ops::test_sub!(u32x4, 0xfc627f10b5f8db8a, Neon::new_checked());
    test_utils::ops::test_mul!(u32x4, 0x0f4caa80eceaa523, Neon::new_checked());
    test_utils::ops::test_fma!(u32x4, 0xb8f702ba85375041, Neon::new_checked());

    test_utils::ops::test_cmp!(u32x4, 0x941757bd5cc641a1, Neon::new_checked());

    // Dot Product
    test_utils::dot_product::test_dot_product!(
        (u8x16, u8x16) => u32x4,
        0x145f89b446c03ff1,
        Neon::new_checked()
    );

    // Bit ops
    test_utils::ops::test_bitops!(u32x4, 0xd62d8de09f82ed4e, Neon::new_checked());

    // Reductions
    test_utils::ops::test_sumtree!(u32x4, 0xb9ac82ab23a855da, Neon::new_checked());
}
