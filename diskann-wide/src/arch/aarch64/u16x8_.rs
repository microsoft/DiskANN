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
    Neon, internal,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask16x8,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

/////////////////////
// 16-bit unsigned //
/////////////////////

macros::aarch64_define_register!(u16x8, uint16x8_t, mask16x8, u16, 8, Neon);
macros::aarch64_define_splat!(u16x8, vmovq_n_u16);
macros::aarch64_define_loadstore!(u16x8, vld1q_u16, internal::load_first::u16x8, vst1q_u16, 8);

helpers::unsafe_map_binary_op!(u16x8, std::ops::Add, add, vaddq_u16, "neon");
helpers::unsafe_map_binary_op!(u16x8, std::ops::Sub, sub, vsubq_u16, "neon");
helpers::unsafe_map_binary_op!(u16x8, std::ops::Mul, mul, vmulq_u16, "neon");
macros::aarch64_define_fma!(u16x8, vmlaq_u16);

macros::aarch64_define_cmp!(
    u16x8,
    vceqq_u16,
    (vmvnq_u16),
    vcltq_u16,
    vcleq_u16,
    vcgtq_u16,
    vcgeq_u16
);
macros::aarch64_define_bitops!(
    u16x8,
    vmvnq_u16,
    vandq_u16,
    vorrq_u16,
    veorq_u16,
    (
        vshlq_u16,
        16,
        vnegq_s16,
        vminq_u16,
        vreinterpretq_s16_u16,
        std::convert::identity
    ),
    (u16, i16, vmovq_n_s16),
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
            test_utils::test_load_simd::<u16, 8, u16x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<u16, 8, u16x8>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<u16, 8, u16x8>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(u16x8, 0x3017fd73c99cc633, test_neon());
    test_utils::ops::test_sub!(u16x8, 0xfc627f10b5f8db8a, test_neon());
    test_utils::ops::test_mul!(u16x8, 0x0f4caa80eceaa523, test_neon());
    test_utils::ops::test_fma!(u16x8, 0xb8f702ba85375041, test_neon());

    test_utils::ops::test_cmp!(u16x8, 0x941757bd5cc641a1, test_neon());

    // Bit ops
    test_utils::ops::test_bitops!(u16x8, 0xd62d8de09f82ed4e, test_neon());
}
