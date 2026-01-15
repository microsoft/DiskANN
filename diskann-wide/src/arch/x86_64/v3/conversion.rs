/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use super::{f16x8, f16x16, f32x8, f32x16, i8x16, i16x16, i32x8, u8x16, u32x8};
use crate::{LoHi, SIMDCast, SIMDReinterpret, SplitJoin, helpers};

/////////////////
// Conversions //
/////////////////

// f16 to f32
cfg_if::cfg_if! {
    // Miri does not have built-in support for these intrinsics.
    //
    // So when we run under Miri, we need to use emulated conversion.
    if #[cfg(not(miri))] {
        helpers::unsafe_map_conversion!(f16x8, f32x8, _mm256_cvtph_ps, "f16c");
        helpers::unsafe_map_cast!(
            f32x8 => (half::f16, f16x8),
            _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>,
            "f16c"
        );
        helpers::unsafe_map_cast!(f16x8 => (f32, f32x8), _mm256_cvtph_ps, "f16c");
    } else {
        use crate::{SIMDVector, reference::ReferenceCast};

        impl From<f16x8> for f32x8 {
            fn from(value: f16x8) -> f32x8 {
                let array = value.to_array();
                let converted = array.map(|i| i.reference_cast());
                f32x8::from_array(value.arch(), converted)
            }
        }

        impl SIMDCast<f32> for f16x8 {
            type Cast = f32x8;

            fn simd_cast(self) -> f32x8 {
                self.into()
            }
        }

        impl SIMDCast<half::f16> for f32x8 {
            type Cast = f16x8;

            fn simd_cast(self) -> f16x8 {
                let array = self.to_array();
                let converted = array.map(|i| i.reference_cast());
                f16x8::from_array(self.arch(), converted)
            }
        }
    }
}

impl From<f16x16> for f32x16 {
    #[inline(always)]
    fn from(value: f16x16) -> f32x16 {
        let LoHi { lo, hi } = value.split();
        f32x16::new(lo.into(), hi.into())
    }
}

impl SIMDCast<f32> for f16x16 {
    type Cast = f32x16;
    #[inline(always)]
    fn simd_cast(self) -> f32x16 {
        self.into()
    }
}

impl SIMDCast<half::f16> for f32x16 {
    type Cast = f16x16;
    #[inline(always)]
    fn simd_cast(self) -> f16x16 {
        self.split().map(|x| -> f16x8 { x.simd_cast() }).join()
    }
}

// i8 to i16
helpers::unsafe_map_conversion!(i8x16, i16x16, _mm256_cvtepi8_epi16, "avx2");

// u8 to i16
helpers::unsafe_map_conversion!(u8x16, i16x16, _mm256_cvtepu8_epi16, "avx2");

// i32 to f32
helpers::unsafe_map_cast!(i32x8 => (f32, f32x8), _mm256_cvtepi32_ps, "avx2");

//////////////////
// Reinterprets //
//////////////////

impl SIMDReinterpret<i16x16> for u32x8 {
    fn reinterpret_simd(self) -> i16x16 {
        i16x16(self.0)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_conversions {
    use super::*;
    use crate::{arch::x86_64::V3, test_utils};

    // Lossless Conversions
    #[cfg(not(miri))]
    test_utils::ops::test_lossless_convert!(
        f16x8 => f32x8, 0xa998182f02ff4d0d, V3::new_checked_uncached()
    );

    #[cfg(not(miri))]
    test_utils::ops::test_lossless_convert!(
        f16x16 => f32x16, 0xf63b356d1dfc1d52, V3::new_checked_uncached()
    );

    test_utils::ops::test_lossless_convert!(
        i8x16 => i16x16, 0x84602159fb122584, V3::new_checked_uncached()
    );
    test_utils::ops::test_lossless_convert!(
        u8x16 => i16x16, 0x5ba4b69df84ca568, V3::new_checked_uncached()
    );

    // Numeric Casts
    test_utils::ops::test_cast!(f16x8 => f32x8, 0x37314659b022466a, V3::new_checked_uncached());
    test_utils::ops::test_cast!(f16x16 => f32x16, 0xba8fe343fc9dbeff, V3::new_checked_uncached());

    test_utils::ops::test_cast!(f32x8 => f16x8, 0x8386cb0a7091cc3b, V3::new_checked_uncached());
    test_utils::ops::test_cast!(f32x16 => f16x16, 0x7b9c9afee7e6ac63, V3::new_checked_uncached());

    test_utils::ops::test_cast!(i32x8 => f32x8, 0xde4fbf25c554b29e, V3::new_checked_uncached());
}
