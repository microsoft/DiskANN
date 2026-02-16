/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated,
    constant::Const,
    helpers,
    traits::{SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector},
};

// AArch64 masks
use super::{
    Neon, algorithms,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask8x16,
    u8x8,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

////////////////////
// 8-bit unsigned //
////////////////////

macros::aarch64_define_register!(u8x16, uint8x16_t, mask8x16, u8, 16, Neon);
macros::aarch64_define_splat!(u8x16, vmovq_n_u8);
macros::aarch64_define_loadstore!(u8x16, vld1q_u8, algorithms::load_first::u8x16, vst1q_u8, 16);
macros::aarch64_splitjoin!(u8x16, u8x8, vget_low_u8, vget_high_u8, vcombine_u8);

helpers::unsafe_map_binary_op!(u8x16, std::ops::Add, add, vaddq_u8, "neon");
helpers::unsafe_map_binary_op!(u8x16, std::ops::Sub, sub, vsubq_u8, "neon");
helpers::unsafe_map_binary_op!(u8x16, std::ops::Mul, mul, vmulq_u8, "neon");
macros::aarch64_define_fma!(u8x16, vmlaq_u8);

macros::aarch64_define_cmp!(
    u8x16,
    vceqq_u8,
    (vmvnq_u8),
    vcltq_u8,
    vcleq_u8,
    vcgtq_u8,
    vcgeq_u8
);
macros::aarch64_define_bitops!(
    u8x16,
    vmvnq_u8,
    vandq_u8,
    vorrq_u8,
    veorq_u8,
    (
        vshlq_u8,
        8,
        vnegq_s8,
        vminq_u8,
        vreinterpretq_s8_u8,
        std::convert::identity
    ),
    (u8, i8, vmovq_n_s8),
);

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        test_utils::test_load_simd::<u8, 16, u8x16>(Neon::new_checked().unwrap());
    }

    #[test]
    fn miri_test_store() {
        test_utils::test_store_simd::<u8, 16, u8x16>(Neon::new_checked().unwrap());
    }

    // constructors
    #[test]
    fn test_constructors() {
        test_utils::ops::test_splat::<u8, 16, u8x16>(Neon::new_checked().unwrap());
    }

    // Ops
    test_utils::ops::test_add!(u8x16, 0x3017fd73c99cc633, Neon::new_checked());
    test_utils::ops::test_sub!(u8x16, 0xfc627f10b5f8db8a, Neon::new_checked());
    test_utils::ops::test_mul!(u8x16, 0x0f4caa80eceaa523, Neon::new_checked());
    test_utils::ops::test_fma!(u8x16, 0xb8f702ba85375041, Neon::new_checked());
    test_utils::ops::test_splitjoin!(u8x16 => u8x8, 0xa4d00a4d04293967, Neon::new_checked());

    test_utils::ops::test_cmp!(u8x16, 0x941757bd5cc641a1, Neon::new_checked());

    // Bit ops
    test_utils::ops::test_bitops!(u8x16, 0xd62d8de09f82ed4e, Neon::new_checked());
}
