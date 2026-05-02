/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated, SIMDAbs, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector,
    constant::Const, helpers,
};

// AArch64 masks
use super::{
    Neon, i8x8, internal,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask8x16,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

//////////////////
// 8-bit signed //
//////////////////

macros::aarch64_define_register!(i8x16, int8x16_t, mask8x16, i8, 16, Neon);
macros::aarch64_define_splat!(i8x16, vmovq_n_s8);
macros::aarch64_define_loadstore!(i8x16, vld1q_s8, internal::load_first::i8x16, vst1q_s8, 16);
macros::aarch64_splitjoin!(i8x16, i8x8, vget_low_s8, vget_high_s8, vcombine_s8);

helpers::unsafe_map_binary_op!(i8x16, std::ops::Add, add, vaddq_s8, "neon");
helpers::unsafe_map_binary_op!(i8x16, std::ops::Sub, sub, vsubq_s8, "neon");
helpers::unsafe_map_binary_op!(i8x16, std::ops::Mul, mul, vmulq_s8, "neon");
helpers::unsafe_map_unary_op!(i8x16, SIMDAbs, abs_simd, vabsq_s8, "neon");
macros::aarch64_define_fma!(i8x16, vmlaq_s8);

macros::aarch64_define_cmp!(
    i8x16,
    vceqq_s8,
    (vmvnq_u8),
    vcltq_s8,
    vcleq_s8,
    vcgtq_s8,
    vcgeq_s8
);
macros::aarch64_define_bitops!(
    i8x16,
    vmvnq_s8,
    vandq_s8,
    vorrq_s8,
    veorq_s8,
    (
        vshlq_s8,
        8,
        vnegq_s8,
        vminq_u8,
        vreinterpretq_s8_u8,
        vreinterpretq_u8_s8
    ),
    (u8, i8, vmovq_n_s8),
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
            test_utils::test_load_simd::<i8, 16, i8x16>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<i8, 16, i8x16>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<i8, 16, i8x16>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i8x16, 0x3017fd73c99cc633, test_neon());
    test_utils::ops::test_sub!(i8x16, 0xfc627f10b5f8db8a, test_neon());
    test_utils::ops::test_mul!(i8x16, 0x0f4caa80eceaa523, test_neon());
    test_utils::ops::test_fma!(i8x16, 0xb8f702ba85375041, test_neon());
    test_utils::ops::test_abs!(i8x16, 0xb8f702ba85375041, test_neon());
    test_utils::ops::test_splitjoin!(i8x16 => i8x8, 0xa4d00a4d04293967, test_neon());

    test_utils::ops::test_cmp!(i8x16, 0x941757bd5cc641a1, test_neon());

    // Bit ops
    test_utils::ops::test_bitops!(i8x16, 0xd62d8de09f82ed4e, test_neon());
}
