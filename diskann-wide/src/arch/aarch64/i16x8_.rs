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
    masks::mask16x8,
    u8x8,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

///////////////////
// 16-bit signed //
///////////////////

macros::aarch64_define_register!(i16x8, int16x8_t, mask16x8, i16, 8, Neon);
macros::aarch64_define_splat!(i16x8, vmovq_n_s16);
macros::aarch64_define_loadstore!(i16x8, vld1q_s16, internal::load_first::i16x8, vst1q_s16, 8);

helpers::unsafe_map_binary_op!(i16x8, std::ops::Add, add, vaddq_s16, "neon");
helpers::unsafe_map_binary_op!(i16x8, std::ops::Sub, sub, vsubq_s16, "neon");
helpers::unsafe_map_binary_op!(i16x8, std::ops::Mul, mul, vmulq_s16, "neon");
helpers::unsafe_map_unary_op!(i16x8, SIMDAbs, abs_simd, vabsq_s16, "neon");
macros::aarch64_define_fma!(i16x8, vmlaq_s16);

macros::aarch64_define_cmp!(
    i16x8,
    vceqq_s16,
    (vmvnq_u16),
    vcltq_s16,
    vcleq_s16,
    vcgtq_s16,
    vcgeq_s16
);
macros::aarch64_define_bitops!(
    i16x8,
    vmvnq_s16,
    vandq_s16,
    vorrq_s16,
    veorq_s16,
    (
        vshlq_s16,
        16,
        vnegq_s16,
        vminq_u16,
        vreinterpretq_s16_u16,
        vreinterpretq_u16_s16
    ),
    (u16, i16, vmovq_n_s16),
);

// Conversion
helpers::unsafe_map_conversion!(i8x8, i16x8, vmovl_s8, "neon");
helpers::unsafe_map_conversion!(u8x8, i16x8, (vreinterpretq_s16_u16, vmovl_u8), "neon");

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
            test_utils::test_load_simd::<i16, 8, i16x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<i16, 8, i16x8>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<i16, 8, i16x8>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(i16x8, 0x3017fd73c99cc633, test_neon());
    test_utils::ops::test_sub!(i16x8, 0xfc627f10b5f8db8a, test_neon());
    test_utils::ops::test_mul!(i16x8, 0x0f4caa80eceaa523, test_neon());
    test_utils::ops::test_fma!(i16x8, 0xb8f702ba85375041, test_neon());
    test_utils::ops::test_abs!(i16x8, 0xb8f702ba85375041, test_neon());

    test_utils::ops::test_cmp!(i16x8, 0x941757bd5cc641a1, test_neon());

    // Bit ops
    test_utils::ops::test_bitops!(i16x8, 0xd62d8de09f82ed4e, test_neon());

    // Conversion
    test_utils::ops::test_lossless_convert!(i8x8 => i16x8, 0x79458ca52356242e, test_neon());
    test_utils::ops::test_lossless_convert!(u8x8 => i16x8, 0xa9a57c5c541ce360, test_neon());
}
