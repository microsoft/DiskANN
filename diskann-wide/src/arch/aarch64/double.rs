/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use half::f16;

use crate::{
    LoHi, SplitJoin,
    doubled::{self, Doubled},
};

use super::{
    f16x8, f32x4, i8x16, i16x8, i32x4,
    masks::{mask8x16, mask16x8, mask32x4, mask64x2},
    u8x16, u32x4, u64x2,
};

// Double Masks
doubled::double_mask!(32, mask8x16);
doubled::double_mask!(16, mask16x8);
doubled::double_mask!(8, mask32x4);
doubled::double_mask!(4, mask64x2);

// Double-Double Masks
doubled::double_mask!(64, Doubled<mask8x16>);
doubled::double_mask!(32, Doubled<mask16x8>);
doubled::double_mask!(16, Doubled<mask32x4>);

macro_rules! double_alias {
    ($type:ident, $scalar:ty, $lanes:literal, $subtype:ty) => {
        // Implement `SIMDVector` and friends for the `Double` type.
        doubled::double_vector!($scalar, $lanes, $subtype);

        #[allow(non_camel_case_types)]
        pub type $type = Doubled<$subtype>;
    };
}

// Double Wide
double_alias!(f32x8, f32, 8, f32x4);

double_alias!(u8x32, u8, 32, u8x16);
double_alias!(u32x8, u32, 8, u32x4);
double_alias!(u64x4, u64, 4, u64x2);

double_alias!(i8x32, i8, 32, i8x16);
double_alias!(i16x16, i16, 16, i16x8);
double_alias!(i32x8, i32, 8, i32x4);

doubled::double_scalar_shift!(Doubled<u8x16>);
doubled::double_scalar_shift!(Doubled<u32x4>);
doubled::double_scalar_shift!(Doubled<u64x2>);

doubled::double_scalar_shift!(Doubled<i8x16>);
doubled::double_scalar_shift!(Doubled<i16x8>);
doubled::double_scalar_shift!(Doubled<i32x4>);

// Double-Double Wide
double_alias!(f32x16, f32, 16, f32x8);
double_alias!(f16x16, f16, 16, f16x8);

double_alias!(u8x64, u8, 64, u8x32);
double_alias!(u32x16, u32, 16, u32x8);

double_alias!(i8x64, i8, 64, i8x32);
double_alias!(i16x32, i16, 32, i16x16);
double_alias!(i32x16, i32, 16, i32x8);

doubled::double_scalar_shift!(Doubled<Doubled<u8x16>>);
doubled::double_scalar_shift!(Doubled<Doubled<u32x4>>);
doubled::double_scalar_shift!(Doubled<Doubled<u64x2>>);

doubled::double_scalar_shift!(Doubled<Doubled<i8x16>>);
doubled::double_scalar_shift!(Doubled<Doubled<i16x8>>);
doubled::double_scalar_shift!(Doubled<Doubled<i32x4>>);

//-------------//
// Conversions //
//-------------//

// Lossless
impl From<f16x8> for f32x8 {
    #[inline(always)]
    fn from(value: f16x8) -> Self {
        let LoHi { lo, hi } = value.split();
        Self::new(lo.into(), hi.into())
    }
}

impl From<f16x16> for f32x16 {
    #[inline(always)]
    fn from(value: f16x16) -> Self {
        Self::new(value.0.into(), value.1.into())
    }
}

impl From<u8x16> for i16x16 {
    #[inline(always)]
    fn from(value: u8x16) -> Self {
        let LoHi { lo, hi } = value.split();
        Self::new(lo.into(), hi.into())
    }
}

impl From<u8x32> for i16x32 {
    #[inline(always)]
    fn from(value: u8x32) -> Self {
        Self::new(value.0.into(), value.1.into())
    }
}

impl From<i8x16> for i16x16 {
    #[inline(always)]
    fn from(value: i8x16) -> Self {
        let LoHi { lo, hi } = value.split();
        Self::new(lo.into(), hi.into())
    }
}

impl From<i8x32> for i16x32 {
    #[inline(always)]
    fn from(value: i8x32) -> Self {
        Self::new(value.0.into(), value.1.into())
    }
}

// (Potentially) Lossy
impl crate::SIMDCast<f32> for f16x16 {
    type Cast = f32x16;
    #[inline(always)]
    fn simd_cast(self) -> f32x16 {
        self.into()
    }
}

impl crate::SIMDCast<f16> for f32x8 {
    type Cast = f16x8;
    #[inline(always)]
    fn simd_cast(self) -> f16x8 {
        f16x8::join(LoHi::new(self.0.simd_cast(), self.1.simd_cast()))
    }
}

impl crate::SIMDCast<f16> for f32x16 {
    type Cast = f16x16;
    #[inline(always)]
    fn simd_cast(self) -> f16x16 {
        f16x16::new(self.0.simd_cast(), self.1.simd_cast())
    }
}

impl crate::SIMDCast<f32> for i32x8 {
    type Cast = f32x8;
    #[inline(always)]
    fn simd_cast(self) -> f32x8 {
        f32x8::new(self.0.simd_cast(), self.1.simd_cast())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arch::aarch64::test_neon, reference::ReferenceScalarOps, test_utils};

    // Run a standard set of:
    // - Load
    // - Store
    // - Add, Sub, Mul, FMA
    // - SIMDPartialEq, SIMDPartialCmp
    macro_rules! standard_tests {
        ($type:ident, $scalar:ty, $lanes:literal) => {
            #[test]
            fn miri_test_load() {
                if let Some(arch) = test_neon() {
                    test_utils::test_load_simd::<$scalar, $lanes, $type>(arch);
                }
            }

            #[test]
            fn miri_test_store() {
                if let Some(arch) = test_neon() {
                    test_utils::test_store_simd::<$scalar, $lanes, $type>(arch);
                }
            }

            #[test]
            fn test_constructors() {
                if let Some(arch) = test_neon() {
                    test_utils::ops::test_splat::<$scalar, $lanes, $type>(arch);
                }
            }

            test_utils::ops::test_add!($type, 0x1c08175714ae637e, test_neon());
            test_utils::ops::test_sub!($type, 0x3746ddcb006b7b4c, test_neon());
            test_utils::ops::test_mul!($type, 0xde99e62aaea3f38a, test_neon());
            test_utils::ops::test_fma!($type, 0x2e301b7e12090d5c, test_neon());

            test_utils::ops::test_cmp!($type, 0x90a59e23ad545de1, test_neon());
        };
    }

    // f32s
    mod test_f32x8 {
        use super::*;
        standard_tests!(f32x8, f32, 8);
        test_utils::ops::test_sumtree!(f32x8, 0x90a59e23ad545de1, test_neon());
        test_utils::ops::test_splitjoin!(f32x8 => f32x4, 0x2e301b7e12090d5c, test_neon());
    }

    mod test_f32x16 {
        use super::*;
        standard_tests!(f32x16, f32, 16);
        test_utils::ops::test_sumtree!(f32x16, 0x90a59e23ad545de1, test_neon());
        test_utils::ops::test_splitjoin!(f32x16 => f32x8, 0x2e301b7e12090d5c, test_neon());
    }

    // u8s
    mod test_u8x32 {
        use super::*;
        standard_tests!(u8x32, u8, 32);

        // Bit ops
        test_utils::ops::test_bitops!(u8x32, 0xd62d8de09f82ed4e, test_neon());
    }

    mod test_u8x64 {
        use super::*;
        standard_tests!(u8x64, u8, 64);

        // Bit ops
        test_utils::ops::test_bitops!(u8x64, 0xd62d8de09f82ed4e, test_neon());
    }

    // u32s
    mod test_u32x8 {
        use super::*;
        standard_tests!(u32x8, u32, 8);

        // Bit ops
        test_utils::ops::test_bitops!(u32x8, 0xd62d8de09f82ed4e, test_neon());

        // Reductions
        test_utils::ops::test_sumtree!(u32x8, 0x90a59e23ad545de1, test_neon());
    }

    mod test_u32x16 {
        use super::*;
        standard_tests!(u32x16, u32, 16);

        // Bit ops
        test_utils::ops::test_bitops!(u32x16, 0xd62d8de09f82ed4e, test_neon());

        // Reductions
        test_utils::ops::test_sumtree!(u32x16, 0x90a59e23ad545de1, test_neon());
    }

    // u64s
    mod test_u64x4 {
        use super::*;
        standard_tests!(u64x4, u64, 4);

        // Bit ops
        test_utils::ops::test_bitops!(u64x4, 0xc4491a44af4aa58e, test_neon());
    }

    // i8s
    mod test_i8x32 {
        use super::*;
        standard_tests!(i8x32, i8, 32);

        // Bit ops
        test_utils::ops::test_bitops!(i8x32, 0xd62d8de09f82ed4e, test_neon());
    }

    mod test_i8x64 {
        use super::*;
        standard_tests!(i8x64, i8, 64);

        // Bit ops
        test_utils::ops::test_bitops!(i8x64, 0xd62d8de09f82ed4e, test_neon());
    }

    // i16s
    mod test_i16x16 {
        use super::*;
        standard_tests!(i16x16, i16, 16);

        // Bit ops
        test_utils::ops::test_bitops!(i16x16, 0x9167644fc4ad5cfa, test_neon());
    }

    mod test_i16x32 {
        use super::*;
        standard_tests!(i16x32, i16, 32);

        // Bit ops
        test_utils::ops::test_bitops!(i16x32, 0x9167644fc4ad5cfa, test_neon());
    }

    // i32s
    mod test_i32x8 {
        use super::*;
        standard_tests!(i32x8, i32, 8);

        // Bit ops
        test_utils::ops::test_bitops!(i32x8, 0xc4491a44af4aa58e, test_neon());

        // Dot Products
        test_utils::dot_product::test_dot_product!(
            (i16x16, i16x16) => i32x8,
            0x145f89b446c03ff1,
            test_neon()
        );

        test_utils::dot_product::test_dot_product!(
            (u8x32, i8x32) => i32x8,
            0x145f89b446c03ff1,
            test_neon()
        );

        test_utils::dot_product::test_dot_product!(
            (i8x32, u8x32) => i32x8,
            0x145f89b446c03ff1,
            test_neon()
        );

        // Reductions
        test_utils::ops::test_sumtree!(i32x8, 0x90a59e23ad545de1, test_neon());
    }

    mod test_i32x16 {
        use super::*;
        standard_tests!(i32x16, i32, 16);

        // Bit ops
        test_utils::ops::test_bitops!(i32x16, 0xc4491a44af4aa58e, test_neon());

        // Dot Products
        test_utils::dot_product::test_dot_product!(
            (i16x32, i16x32) => i32x16,
            0x145f89b446c03ff1,
            test_neon()
        );

        test_utils::dot_product::test_dot_product!(
            (u8x64, i8x64) => i32x16,
            0x145f89b446c03ff1,
            test_neon()
        );

        test_utils::dot_product::test_dot_product!(
            (i8x64, u8x64) => i32x16,
            0x145f89b446c03ff1,
            test_neon()
        );

        // Reductions
        test_utils::ops::test_sumtree!(i32x16, 0x90a59e23ad545de1, test_neon());
    }

    // Conversions
    test_utils::ops::test_lossless_convert!(f16x8 => f32x8, 0x84c1c6f05b169a20, test_neon());
    test_utils::ops::test_lossless_convert!(f16x16 => f32x16, 0x84c1c6f05b169a20, test_neon());

    test_utils::ops::test_lossless_convert!(u8x16 => i16x16, 0x84c1c6f05b169a20, test_neon());
    test_utils::ops::test_lossless_convert!(i8x16 => i16x16, 0x84c1c6f05b169a20, test_neon());

    test_utils::ops::test_cast!(f16x8 => f32x8, 0xba8fe343fc9dbeff, test_neon());
    test_utils::ops::test_cast!(f16x16 => f32x16, 0xba8fe343fc9dbeff, test_neon());
    test_utils::ops::test_cast!(f32x8 => f16x8, 0xba8fe343fc9dbeff, test_neon());
    test_utils::ops::test_cast!(f32x16 => f16x16, 0xba8fe343fc9dbeff, test_neon());

    test_utils::ops::test_cast!(i32x8 => f32x8, 0xba8fe343fc9dbeff, test_neon());
}
