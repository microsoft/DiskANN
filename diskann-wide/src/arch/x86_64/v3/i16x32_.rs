/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 masks
use super::{i8x32, i16x16, u8x32};
use crate::{LoHi, SplitJoin, doubled};

////////////////////
// 16-bit integer //
////////////////////

doubled::double_vector!(i16, 32, i16x16);
doubled::double_scalar_shift!(Doubled<i16x16>);

#[allow(non_camel_case_types)]
pub type i16x32 = doubled::Doubled<i16x16>;

impl From<u8x32> for i16x32 {
    #[inline(always)]
    fn from(x: u8x32) -> Self {
        let LoHi { lo, hi } = x.split();
        Self(lo.into(), hi.into())
    }
}

impl From<i8x32> for i16x32 {
    #[inline(always)]
    fn from(x: i8x32) -> Self {
        let LoHi { lo, hi } = x.split();
        Self(lo.into(), hi.into())
    }
}

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
            test_utils::test_load_simd::<i16, 32, i16x32>(arch);
        }
    }

    #[test]
    fn miri_test_store() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::test_store_simd::<i16, 32, i16x32>(arch);
        }
    }

    // constructors
    #[test]
    fn test_constructors() {
        if let Some(arch) = V3::new_checked_uncached() {
            test_utils::ops::test_splat::<i16, 32, i16x32>(arch);
        }
    }

    // Binary Ops
    test_utils::ops::test_add!(i16x32, 0x9422b74058e8a6a3, V3::new_checked_uncached());
    test_utils::ops::test_sub!(i16x32, 0xeeff074e1ad2e720, V3::new_checked_uncached());
    test_utils::ops::test_mul!(i16x32, 0x57e2dbc100a1ed82, V3::new_checked_uncached());
    test_utils::ops::test_fma!(i16x32, 0x138fd8f4ec8e5530, V3::new_checked_uncached());
    test_utils::ops::test_abs!(i16x32, 0xd45290c6a8b26899, V3::new_checked_uncached());

    test_utils::ops::test_cmp!(i16x32, 0x3b68b51ecb187598, V3::new_checked_uncached());
    test_utils::ops::test_splitjoin!(i16x32 => i16x16, 0x11f385aeffd15159, V3::new_checked_uncached());

    // Bit ops
    test_utils::ops::test_bitops!(i16x32, 0xc5f7d8d8df0b7b6c, V3::new_checked_uncached());

    // Conversions
    test_utils::ops::test_lossless_convert!(
        u8x32 => i16x32, 0x84602159fb122584, V3::new_checked_uncached()
    );
    test_utils::ops::test_lossless_convert!(
        i8x32 => i16x32, 0x84602159fb122584, V3::new_checked_uncached()
    );
}
