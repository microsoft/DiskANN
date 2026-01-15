/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// x86 intrinsics
use std::arch::x86_64::*;

use super::{
    f16x8_::f16x8, f16x16_::f16x16, f32x8_::f32x8, f32x16_::f32x16, i8x16_::i8x16, i8x32_::i8x32,
    i8x64_::i8x64, i16x16_::i16x16, i16x32_::i16x32, i32x8_::i32x8, u8x16_::u8x16, u8x32_::u8x32,
    u8x64_::u8x64, u32x8_::u32x8, u32x16_::u32x16,
};
use crate::{SIMDCast, SIMDReinterpret, SIMDVector, arch::x86_64::v3, helpers};

/////////////////
// Conversions //
/////////////////

impl From<f16x8> for f32x8 {
    #[inline(always)]
    fn from(value: f16x8) -> f32x8 {
        f32x8::from_underlying(value.arch(), v3::f32x8::from(value.retarget()).0)
    }
}

impl From<f16x16> for f32x16 {
    #[inline(always)]
    fn from(value: f16x16) -> f32x16 {
        // SAFETY: `_mm512_cvtph_ps` requires AVX512F - implied by `V4`.
        let cvt = unsafe { _mm512_cvtph_ps(value.to_underlying()) };
        f32x16::from_underlying(value.arch(), cvt)
    }
}

impl SIMDCast<f32> for f16x8 {
    type Cast = f32x8;
    fn simd_cast(self) -> f32x8 {
        self.into()
    }
}

impl SIMDCast<f32> for f16x16 {
    type Cast = f32x16;
    fn simd_cast(self) -> f32x16 {
        self.into()
    }
}

impl SIMDCast<half::f16> for f32x8 {
    type Cast = f16x8;
    fn simd_cast(self) -> f16x8 {
        f16x8::from_underlying(self.arch(), self.retarget().simd_cast().0)
    }
}

impl SIMDCast<half::f16> for f32x16 {
    type Cast = f16x16;
    fn simd_cast(self) -> f16x16 {
        // SAFETY: `_mm512_cvtps_ph` requires AVX512F - implied by `V4`.
        let cvt = unsafe {
            _mm512_cvtps_ph(
                self.to_underlying(),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
            )
        };
        f16x16::from_underlying(self.arch(), cvt)
    }
}

// i8 to i16
helpers::unsafe_map_conversion!(i8x16, i16x16, _mm256_cvtepi8_epi16, "avx2");
helpers::unsafe_map_conversion!(i8x32, i16x32, _mm512_cvtepi8_epi16, "avx512bw");

// u8 to i16
helpers::unsafe_map_conversion!(u8x16, i16x16, _mm256_cvtepu8_epi16, "avx2");
helpers::unsafe_map_conversion!(u8x32, i16x32, _mm512_cvtepu8_epi16, "avx512bw");

// i32 to f32
helpers::unsafe_map_cast!(i32x8 => (f32, f32x8), _mm256_cvtepi32_ps, "avx2");

helpers::unsafe_map_cast!(i16x16 => (u8, u8x16), _mm256_cvtepi16_epi8, "avx512bw,avx512vl");
helpers::unsafe_map_cast!(i16x16 => (i8, i8x16), _mm256_cvtepi16_epi8, "avx512bw,avx512vl");
helpers::unsafe_map_cast!(i16x32 => (u8, u8x32), _mm512_cvtepi16_epi8, "avx512bw");
helpers::unsafe_map_cast!(i16x32 => (i8, i8x32), _mm512_cvtepi16_epi8, "avx512bw");

//////////////////
// Reinterprets //
//////////////////

impl SIMDReinterpret<i16x16> for u32x8 {
    fn reinterpret_simd(self) -> i16x16 {
        i16x16(self.0)
    }
}

impl SIMDReinterpret<i16x32> for u32x16 {
    fn reinterpret_simd(self) -> i16x32 {
        i16x32(self.0)
    }
}

impl SIMDReinterpret<u8x64> for u32x16 {
    fn reinterpret_simd(self) -> u8x64 {
        u8x64(self.0)
    }
}

impl SIMDReinterpret<i8x64> for u32x16 {
    fn reinterpret_simd(self) -> i8x64 {
        i8x64(self.0)
    }
}

impl SIMDReinterpret<u32x16> for u8x64 {
    fn reinterpret_simd(self) -> u32x16 {
        u32x16(self.0)
    }
}

impl SIMDReinterpret<u32x16> for i8x64 {
    fn reinterpret_simd(self) -> u32x16 {
        u32x16(self.0)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test_x86_conversions {
    use super::*;
    use crate::{arch::x86_64::V4, test_utils};

    // Lossless Conversions
    #[cfg(not(miri))]
    test_utils::ops::test_lossless_convert!(
        f16x8 => f32x8, 0xa998182f02ff4d0d, V4::new_checked_uncached()
    );

    #[cfg(not(miri))]
    test_utils::ops::test_lossless_convert!(
        f16x16 => f32x16, 0xe6ab583cbb1b06e0, V4::new_checked_uncached()
    );

    test_utils::ops::test_lossless_convert!(
        i8x16 => i16x16, 0x84602159fb122584, V4::new_checked_uncached()
    );
    test_utils::ops::test_lossless_convert!(
        i8x32 => i16x32, 0xa9e19910dabee638, V4::new_checked_uncached()
    );
    test_utils::ops::test_lossless_convert!(
        u8x16 => i16x16, 0x5ba4b69df84ca568, V4::new_checked_uncached()
    );
    test_utils::ops::test_lossless_convert!(
        u8x32 => i16x32, 0xb42af810c6768193, V4::new_checked_uncached()
    );

    // Numeric Casts
    test_utils::ops::test_cast!(f16x8 => f32x8, 0x37314659b022466a, V4::new_checked_uncached());
    test_utils::ops::test_cast!(f16x16 => f32x16, 0x1aa5762d788d7749, V4::new_checked_uncached());

    test_utils::ops::test_cast!(f32x8 => f16x8, 0x8386cb0a7091cc3b, V4::new_checked_uncached());
    test_utils::ops::test_cast!(f32x16 => f16x16, 0xb3cbae34def475df, V4::new_checked_uncached());

    test_utils::ops::test_cast!(i32x8 => f32x8, 0xde4fbf25c554b29e, V4::new_checked_uncached());

    test_utils::ops::test_cast!(i16x16 => u8x16, 0x0f81df9e640b0269, V4::new_checked_uncached());
    test_utils::ops::test_cast!(i16x16 => i8x16, 0x4ab1546b9d0e4046, V4::new_checked_uncached());

    test_utils::ops::test_cast!(i16x32 => u8x32, 0xf2c00ea1a1b5c380, V4::new_checked_uncached());
    test_utils::ops::test_cast!(i16x32 => i8x32, 0x6090af7cb2847dd5, V4::new_checked_uncached());
}
