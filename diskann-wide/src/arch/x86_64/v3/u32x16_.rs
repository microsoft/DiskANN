/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 masks
use super::u32x8;
use crate::doubled;

/////////////////////
// 32-bit floating //
/////////////////////

doubled::double_vector!(u32, 16, u32x8);
doubled::double_scalar_shift!(Doubled<u32x8>);

#[allow(non_camel_case_types)]
pub type u32x16 = doubled::Doubled<u32x8>;

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::{arch::x86_64::V3, reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<u32, 16, u32x16>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<u32, 16, u32x16>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<u32, 16, u32x16>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(u32x16, 0x9422b74058e8a6a3, V3::new_checked_uncached());
    test_utils::ops::test_sub!(u32x16, 0xeeff074e1ad2e720, V3::new_checked_uncached());
    test_utils::ops::test_mul!(u32x16, 0x57e2dbc100a1ed82, V3::new_checked_uncached());
    test_utils::ops::test_fma!(u32x16, 0x138fd8f4ec8e5530, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(u32x16, 0x3b68b51ecb187598, V3::new_checked_uncached());
    test_utils::ops::test_splitjoin!(u32x16 => u32x8, 0x11f385aeffd15159, V3::new_checked_uncached());
    test_utils::ops::test_select!(u32x16, 0xfb5bdcfc653f7f63, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(u32x16, 0xc5f7d8d8df0b7b6c, V3::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(u32x16, 0xe533708e69ca0117, V3::new_checked_uncached());
}
