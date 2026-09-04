/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDSelect, SIMDSumTree,
    SIMDVector, constant::Const, helpers,
};

// AArch64 masks
use super::{
    Neon, internal,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask32x2,
};

// AArch64 intrinsics
use std::arch::aarch64::*;

/////////////////////
// 32-bit unsigned //
/////////////////////

macros::aarch64_define_register!(u32x2, uint32x2_t, mask32x2, u32, 2, Neon);
macros::aarch64_define_splat!(u32x2, vmov_n_u32);
macros::aarch64_define_loadstore!(u32x2, vld1_u32, internal::load_first::u32x2, vst1_u32, 2);

helpers::unsafe_map_binary_op!(u32x2, std::ops::Add, add, vadd_u32, "neon");
helpers::unsafe_map_binary_op!(u32x2, std::ops::Sub, sub, vsub_u32, "neon");
helpers::unsafe_map_binary_op!(u32x2, std::ops::Mul, mul, vmul_u32, "neon");
macros::aarch64_define_fma!(u32x2, vmla_u32);

macros::aarch64_define_cmp!(
    u32x2,
    vceq_u32,
    (vmvn_u32),
    vclt_u32,
    vcle_u32,
    vcgt_u32,
    vcge_u32
);
macros::aarch64_define_bitops!(
    u32x2,
    vmvn_u32,
    vand_u32,
    vorr_u32,
    veor_u32,
    (
        vshl_u32,
        32,
        vneg_s32,
        vmin_u32,
        vreinterpret_s32_u32,
        std::convert::identity
    ),
    (u32, i32, vmov_n_s32),
);

impl SIMDSumTree for u32x2 {
    #[inline(always)]
    fn sum_tree(self) -> u32 {
        if cfg!(miri) {
            self.emulated().sum_tree()
        } else {
            // SAFETY: Allowed by the `Neon` architecture.
            unsafe { vaddv_u32(self.0) }
        }
    }
}

impl SIMDSelect<u32x2> for mask32x2 {
    #[inline(always)]
    fn select(self, x: u32x2, y: u32x2) -> u32x2 {
        // SAFETY: Allowed by the `Neon` architecture.
        u32x2(unsafe { vbsl_u32(self.0, x.0, y.0) })
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
            test_utils::test_load_simd::<u32, 2, u32x2>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<u32, 2, u32x2>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<u32, 2, u32x2>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(u32x2, 0x3017fd73c99cc633, test_neon());
    test_utils::ops::test_sub!(u32x2, 0xfc627f10b5f8db8a, test_neon());
    test_utils::ops::test_mul!(u32x2, 0x0f4caa80eceaa523, test_neon());
    test_utils::ops::test_fma!(u32x2, 0xb8f702ba85375041, test_neon());

    test_utils::ops::test_cmp!(u32x2, 0x941757bd5cc641a1, test_neon());

    // Bit ops
    test_utils::ops::test_bitops!(u32x2, 0xd62d8de09f82ed4e, test_neon());
    test_utils::ops::test_select!(u32x2, 0xd62d8de09f82ed4e, test_neon());

    // Reductions
    test_utils::ops::test_sumtree!(u32x2, 0xb9ac82ab23a855da, test_neon());
}
