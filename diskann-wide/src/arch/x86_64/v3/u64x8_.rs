/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::u64x4;
use crate::doubled;

/////////////////////
// 64-bit unsigned //
/////////////////////

doubled::double_vector!(u64, 8, u64x4);
doubled::double_scalar_shift!(Doubled<u64x4>);

#[allow(non_camel_case_types)]
pub type u64x8 = doubled::Doubled<u64x4>;

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
            test_utils::test_load_simd::<u64, 8, u64x8>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<u64, 8, u64x8>(arch);
        }
    }

    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<u64, 8, u64x8>(arch);
        }
    }

    test_utils::ops::test_add!(u64x8, 0xeaee2fd0398fe357, V3::new_checked_uncached());
    test_utils::ops::test_sub!(u64x8, 0x40af040b0c2c1e28, V3::new_checked_uncached());
    test_utils::ops::test_mul!(u64x8, 0x68f68933a29c5ea9, V3::new_checked_uncached());
    test_utils::ops::test_fma!(u64x8, 0x31bc9d25e91e6744, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(u64x8, 0x0beda0dd5141ec40, V3::new_checked_uncached());
    test_utils::ops::test_splitjoin!(u64x8 => u64x4, 0xb151fcd6141b10c9, V3::new_checked_uncached());

    test_utils::ops::test_sumtree!(u64x8, 0x529c27f62ea171ec, V3::new_checked_uncached());

    test_utils::ops::test_bitops!(u64x8, 0xb1ac2e16327a8d5e, V3::new_checked_uncached());
}
