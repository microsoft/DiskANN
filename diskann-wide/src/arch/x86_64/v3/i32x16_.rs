/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 masks
use super::i32x8;
use crate::doubled;

///////////////////
// 32-bit signed //
///////////////////

doubled::double_vector!(i32, 16, i32x8);
doubled::double_scalar_shift!(Doubled<i32x8>);

#[allow(non_camel_case_types)]
pub type i32x16 = doubled::Doubled<i32x8>;

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_f32 {
    use super::*;
    use crate::{
        arch::x86_64::{
            V3,
            v3::{i8x64, i16x32, u8x64},
        },
        reference::ReferenceScalarOps,
        test_utils,
    };

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<i32, 16, i32x16>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<i32, 16, i32x16>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<i32, 16, i32x16>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(i32x16, 0x9422b74058e8a6a3, V3::new_checked_uncached());
    test_utils::ops::test_sub!(i32x16, 0xeeff074e1ad2e720, V3::new_checked_uncached());
    test_utils::ops::test_mul!(i32x16, 0x57e2dbc100a1ed82, V3::new_checked_uncached());
    test_utils::ops::test_fma!(i32x16, 0x138fd8f4ec8e5530, V3::new_checked_uncached());
    test_utils::ops::test_abs!(i32x16, 0xd45290c6a8b26899, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(i32x16, 0x3b68b51ecb187598, V3::new_checked_uncached());
    test_utils::ops::test_splitjoin!(i32x16 => i32x8, 0x11f385aeffd15159, V3::new_checked_uncached());
    test_utils::ops::test_select!(i32x16, 0xfb5bdcfc653f7f63, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i32x16, 0xc5f7d8d8df0b7b6c, V3::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(i32x16, 0xe533708e69ca0117, V3::new_checked_uncached());

    // Dot Products
    test_utils::dot_product::test_dot_product!(
        (i16x32, i16x32) => i32x16, 0x145f89b446c03ff1, V3::new_checked_uncached()
    );
    test_utils::dot_product::test_dot_product!(
        (u8x64, i8x64) => i32x16, 0xa56e0de8ce99136c, V3::new_checked_uncached()
    );
    test_utils::dot_product::test_dot_product!(
        (i8x64, u8x64) => i32x16, 0xbcbcff932237df6d, V3::new_checked_uncached()
    );
}
