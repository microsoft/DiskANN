/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// #![cfg_attr(
//     not(all(target_arch = "x86_64", target_feature = "avx2")),
//     allow(unused_macros, unused_imports)
// )]

/// Utility macro for defining simple operations that lower to a single intrinsic.
///
/// SAFETY: It is the invoker's responsibility to ensure that the intrinsic is safe to call
macro_rules! unsafe_map_binary_op {
    // This variant maps the implementation to an intrinsic.
    ($type:ty, $trait:path, $func:ident, $intrinsic:expr, $requires:literal) => {
        impl $trait for $type {
            type Output = Self;
            #[inline(always)]
            fn $func(self, rhs: Self) -> Self::Output {
                // SAFETY: The invoker of this macro must pass the `target_feature`
                // requirement of the intrinsic.
                //
                // That way, if the intrinsic is not available, we get a compile-time error.
                Self(unsafe { $intrinsic(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! unsafe_map_unary_op {
    ($type:ty, $trait:path, $func:ident, $intrinsic:expr, $requires:literal) => {
        impl $trait for $type {
            #[inline(always)]
            fn $func(self) -> Self {
                // SAFETY: The invoker of this macro must pass the `target_feature`
                // requirement of the intrinsic.
                //
                // That way, if the intrinsic is not available, we get a compile-time error.
                Self(unsafe { $intrinsic(self.0) })
            }
        }
    };
}

/// A utility macro for mapping SIMD conversion to an intrinsic.
///
/// SAFETY: It is the invoker's responsibility to ensure that the intrinsic is safe to call.
macro_rules! unsafe_map_conversion {
    ($from:ty, $to:ty, $intrinsic:expr, $requires:literal) => {
        impl From<$from> for $to {
            #[inline(always)]
            fn from(value: $from) -> $to {
                // SAFETY: The invoker of this macro must pass the `target_feature`
                // requirement of the intrinsic.
                //
                // That way, if the intrinsic is not available, we get a compile-time error.
                Self(unsafe { $intrinsic(value.0) })
            }
        }
    };
}

/// A utility macro for mapping SIMD casting to an intrinsic.
///
/// SAFETY: It is the invoker's responsibility to ensure that the intrinsic is safe to call.
macro_rules! unsafe_map_cast {
    ($from:ty => ($scalar:ty, $to:ty), $intrinsic:expr, $requires:literal) => {
        impl SIMDCast<$scalar> for $from {
            type Cast = $to;

            #[inline(always)]
            fn simd_cast(self) -> Self::Cast {
                use crate::SIMDVector;

                // SAFETY: The invoker of this macro must pass the `target_feature`
                // requirement of the intrinsic.
                //
                // That way, if the intrinsic is not available, we get a compile-time error.
                Self::Cast::from_underlying(self.arch(), unsafe { $intrinsic(self.0) })
            }
        }
    };
}

/// Implement shifting by calling Splat.
macro_rules! scalar_shift_by_splat {
    ($T:ty, $scalar:ty) => {
        impl std::ops::Shr<$scalar> for $T {
            type Output = Self;
            #[inline(always)]
            fn shr(self, rhs: $scalar) -> Self {
                self.shr(<Self as SIMDVector>::splat(self.arch(), rhs))
            }
        }

        impl std::ops::Shl<$scalar> for $T {
            type Output = Self;
            #[inline(always)]
            fn shl(self, rhs: $scalar) -> Self {
                self.shl(<Self as SIMDVector>::splat(self.arch(), rhs))
            }
        }
    };
}

// Allow modules in this crate to use these macros.
pub(crate) use scalar_shift_by_splat;
pub(crate) use unsafe_map_binary_op;
pub(crate) use unsafe_map_cast;
pub(crate) use unsafe_map_conversion;
pub(crate) use unsafe_map_unary_op;
