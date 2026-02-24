/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Emulated, SIMDAbs, SIMDMask, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd, SIMDVector,
    arch::Scalar, constant::Const, helpers,
};

// AArch64 masks
use super::{
    Neon, internal,
    macros::{self, AArchLoadStore, AArchSplat},
    masks::mask64x2,
    u64x2_::{emulated_vminq_u64, emulated_vmvnq_u64},
};

// AArch64 intrinsics
use std::arch::aarch64::*;

///////////////////
// 64-bit signed //
///////////////////

#[inline(always)]
pub(super) unsafe fn emulated_vmvnq_s64(x: int64x2_t) -> int64x2_t {
    let x: [i64; 2] = i64x2(x).to_array();
    let mapped: [i64; 2] = core::array::from_fn(|i| !x[i]);
    // SAFETY: This is only called in a context where the caller guarantees `Neon` is
    // available.
    i64x2::from_array(unsafe { Neon::new() }, mapped).0
}

macros::aarch64_define_register!(i64x2, int64x2_t, mask64x2, i64, 2, Neon);
macros::aarch64_define_splat!(i64x2, vmovq_n_s64);
macros::aarch64_define_loadstore!(i64x2, vld1q_s64, internal::load_first::i64x2, vst1q_s64, 2);

helpers::unsafe_map_binary_op!(i64x2, std::ops::Add, add, vaddq_s64, "neon");
helpers::unsafe_map_binary_op!(i64x2, std::ops::Sub, sub, vsubq_s64, "neon");
helpers::unsafe_map_unary_op!(i64x2, SIMDAbs, abs_simd, vabsq_s64, "neon");

impl std::ops::Mul for i64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let x = Emulated::<i64, 2>::from_array(Scalar, self.to_array());
        let y = Emulated::<i64, 2>::from_array(Scalar, rhs.to_array());
        Self::from_array(self.arch(), (x * y).to_array())
    }
}

macros::aarch64_define_fma!(i64x2, integer);

macros::aarch64_define_cmp!(
    i64x2,
    vceqq_s64,
    (emulated_vmvnq_u64),
    vcltq_s64,
    vcleq_s64,
    vcgtq_s64,
    vcgeq_s64
);
macros::aarch64_define_bitops!(
    i64x2,
    emulated_vmvnq_s64,
    vandq_s64,
    vorrq_s64,
    veorq_s64,
    (
        vshlq_s64,
        64,
        vnegq_s64,
        emulated_vminq_u64,
        vreinterpretq_s64_u64,
        vreinterpretq_u64_s64
    ),
    (u64, i64, vmovq_n_s64),
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
            test_utils::test_load_simd::<i64, 2, i64x2>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = test_neon() {
            test_utils::test_store_simd::<i64, 2, i64x2>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = test_neon() {
            test_utils::ops::test_splat::<i64, 2, i64x2>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(i64x2, 0x8d7bf28b1c6e2545, test_neon());
    test_utils::ops::test_sub!(i64x2, 0x4a1c644a1a910bed, test_neon());
    test_utils::ops::test_mul!(i64x2, 0xf42ee707a808fd10, test_neon());
    test_utils::ops::test_fma!(i64x2, 0x28540d9936a9e803, test_neon());
    test_utils::ops::test_abs!(i64x2, 0xb8f702ba85375041, test_neon());

    test_utils::ops::test_cmp!(i64x2, 0xfae27072c6b70885, test_neon());

    // Bit ops
    test_utils::ops::test_bitops!(i64x2, 0xbe927713ea310164, test_neon());
}
