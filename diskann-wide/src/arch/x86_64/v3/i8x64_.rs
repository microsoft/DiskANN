/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 masks
use super::i8x32;
use crate::doubled;

//////////////////
// 8-bit signed //
//////////////////

doubled::double_vector!(i8, 64, i8x32);
doubled::double_scalar_shift!(Doubled<i8x32>);

#[allow(non_camel_case_types)]
pub type i8x64 = doubled::Doubled<i8x32>;

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arch::x86_64::V3, reference::ReferenceScalarOps, test_utils};

    #[test]
    fn miri_test_load() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<i8, 64, i8x64>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<i8, 64, i8x64>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<i8, 64, i8x64>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(i8x64, 0x9422b74058e8a6a3, V3::new_checked_uncached());
    test_utils::ops::test_sub!(i8x64, 0xeeff074e1ad2e720, V3::new_checked_uncached());
    test_utils::ops::test_mul!(i8x64, 0x57e2dbc100a1ed82, V3::new_checked_uncached());
    test_utils::ops::test_fma!(i8x64, 0x138fd8f4ec8e5530, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(i8x64, 0x3b68b51ecb187598, V3::new_checked_uncached());
    test_utils::ops::test_splitjoin!(i8x64 => i8x32, 0x11f385aeffd15159, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i8x64, 0xc5f7d8d8df0b7b6c, V3::new_checked_uncached());
}
