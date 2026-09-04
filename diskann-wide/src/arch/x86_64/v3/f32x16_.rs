/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 masks
use super::f32x8;
use crate::doubled;

/////////////////////
// 32-bit floating //
/////////////////////

doubled::double_vector!(f32, 16, f32x8);

#[allow(non_camel_case_types)]
pub type f32x16 = doubled::Doubled<f32x8>;

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_f32 {
    use super::*;
    use crate::{arch::x86_64::V3, reference::ReferenceScalarOps, test_utils};

    // Test loading logic - ensure that no out of bounds accesses are made.
    // In particular, this is meant to be run under `Miri` to ensure that our guarantees
    // regarding out-of-bounds accesses are honored.
    #[test]
    fn miri_test_load() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_load_simd::<f32, 16, f32x16>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<f32, 16, f32x16>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<f32, 16, f32x16>(arch);
        }
    }

    // Ops
    test_utils::ops::test_add!(f32x16, 0xa8989b97ca888d11, V3::new_checked_uncached());
    test_utils::ops::test_sub!(f32x16, 0xb2554fc13fdc1182, V3::new_checked_uncached());
    test_utils::ops::test_mul!(f32x16, 0x23becaa968b0cd71, V3::new_checked_uncached());
    test_utils::ops::test_fma!(f32x16, 0x32a814070a93df4e, V3::new_checked_uncached());
    test_utils::ops::test_minmax!(f32x16, 0x6d7fc8ed6d852187, V3::new_checked_uncached());
    test_utils::ops::test_abs!(f32x16, 0x6799e60873a2efe2, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(f32x16, 0x1246278e242caecc, V3::new_checked_uncached());
    test_utils::ops::test_splitjoin!(f32x16 => f32x8, 0xde4ff375903351e3, V3::new_checked_uncached());
    test_utils::ops::test_select!(f32x16, 0xa4a7950ec1dc3b22, V3::new_checked_uncached());

    // Reductions
    test_utils::ops::test_sumtree!(f32x16, 0x0180a265222e3fcf, V3::new_checked_uncached());
}
