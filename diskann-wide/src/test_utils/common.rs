/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::cmp::Ordering;

use half::f16;

use crate::{BitMask, Const, SIMDMask, SupportedLaneCount, arch, reference::ReferenceCast};

// Helper trait to enable setting of values from unsigned 64-bit integers.
pub(crate) trait USizeConvertTo<T> {
    fn test_convert(self) -> T;
}

// Macro to opt types into using `as` conversion from `usize`.
// This is used to initialize memory like `[0, 1, 2, 3, ... N]` for all the types that we
// wish to express as SIMD scalars.
macro_rules! use_as_conversion {
    ($type:ty) => {
        impl USizeConvertTo<$type> for usize {
            fn test_convert(self) -> $type {
                self as $type
            }
        }
    };
}

// Floats
impl USizeConvertTo<f16> for usize {
    fn test_convert(self) -> f16 {
        // Kind of a convoluted logic chain.
        crate::cast_f32_to_f16(self as f32)
    }
}

use_as_conversion!(f32);
use_as_conversion!(f64);

// Signed Integers
use_as_conversion!(i8);
use_as_conversion!(i16);
use_as_conversion!(i32);
use_as_conversion!(i64);

// Unsigned Integers
use_as_conversion!(u8);
use_as_conversion!(u16);
use_as_conversion!(u32);
use_as_conversion!(u64);

/// Initialize an iterator to contain the equivalent of `[0, 1, 2, 3,  ... x.len() - 1]`.
pub(crate) fn iota_iter<'a, T, I>(x: I)
where
    T: 'a,
    usize: USizeConvertTo<T>,
    I: Iterator<Item = &'a mut T>,
{
    x.enumerate().for_each(|(i, d)| {
        *d = i.test_convert();
    });
}

/// Initialize a slice to contain the equivalent of `[0, 1, 2, 3, ..., x.len() - 1]`.
pub(crate) fn iota_slice<T>(x: &mut [T])
where
    usize: USizeConvertTo<T>,
{
    iota_iter(x.iter_mut());
}

/// Convert a bit-mask to an array.
pub(crate) fn promote_to_array<const N: usize, A>(x: BitMask<N, A>) -> [bool; N]
where
    A: arch::Sealed,
    Const<N>: SupportedLaneCount,
    BitMask<N, A>: SIMDMask<Arch = A>,
{
    core::array::from_fn(|i| x.get_unchecked(i))
}

/// A trait for scalar types that participate in Wide
pub(crate) trait ScalarTraits: Copy + std::cmp::PartialEq + std::fmt::Debug {
    /// A collection of test values to provide when testing the `splat` constructor.
    fn splat_test_values() -> impl Iterator<Item = Self>;

    /// A supplied trait to check for exact equality.
    /// This is overridden for floating point numbers to use `total_cmp` for a strict
    /// ordering.
    fn exact_eq(self, other: Self) -> bool {
        self == other
    }
}

// floating point
impl ScalarTraits for f16 {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        let v: Vec<f32> = vec![-127.567, -1.0, -0.0, 0.0, 1.25, 10.5];
        v.into_iter().map(|x| x.reference_cast())
    }

    fn exact_eq(self, other: Self) -> bool {
        // Use `ReferenceCast` for Miri compatibility.
        let x: f32 = self.reference_cast();
        let y: f32 = other.reference_cast();
        x.total_cmp(&y) == Ordering::Equal
    }
}

impl ScalarTraits for f32 {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        [
            f32::MIN,
            f32::MAX,
            0.0,
            -0.0,
            12.0,
            f32::MIN_POSITIVE,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ]
        .into_iter()
    }

    fn exact_eq(self, other: Self) -> bool {
        // Miri does not seem to handle `total_cmp` correctly when it comes to `NAN`s - so
        // we special case this comparison when running with Miri.
        #[cfg(miri)]
        if self.is_nan() && other.is_nan() {
            return true;
        }

        self.total_cmp(&other) == Ordering::Equal
    }
}

impl ScalarTraits for f64 {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        [
            f64::MIN,
            f64::MAX,
            0.0,
            -0.0,
            12.0,
            f64::MIN_POSITIVE,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ]
        .into_iter()
    }

    fn exact_eq(self, other: Self) -> bool {
        // Miri does not seem to handle `total_cmp` correctly when it comes to `NAN`s - so
        // we special case this comparison when running with Miri.
        #[cfg(miri)]
        if self.is_nan() && other.is_nan() {
            return true;
        }

        self.total_cmp(&other) == Ordering::Equal
    }
}

// bool
impl ScalarTraits for bool {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        [false, true].into_iter()
    }
}

// signed integers
impl ScalarTraits for i8 {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        -128..=127
    }
}

impl ScalarTraits for i16 {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        [i16::MIN, -10, -1, 0, 1, 10, i16::MAX].into_iter()
    }
}

impl ScalarTraits for i32 {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        [i32::MIN, -10, -1, 0, 1, 10, i32::MAX].into_iter()
    }
}

impl ScalarTraits for i64 {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        [i64::MIN, -10, -1, 0, 1, 10, i64::MAX].into_iter()
    }
}

// unsigned integers
impl ScalarTraits for u8 {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        0..=255
    }
}

impl ScalarTraits for u16 {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        [0, 1, 10, 255, 9128, u16::MAX].into_iter()
    }
}

impl ScalarTraits for u32 {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        [0, 1, 10, 255, 9128, u32::MAX].into_iter()
    }
}

impl ScalarTraits for u64 {
    fn splat_test_values() -> impl Iterator<Item = Self> {
        [0, 1, 10, 255, 9128, u64::MAX].into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_eq_f16() {
        let zero: f16 = 0.0f32.reference_cast();
        let minus_zero: f16 = -0.0f32.reference_cast();

        assert!(zero.exact_eq(zero));
        assert!(!zero.exact_eq(minus_zero));
        assert!(!minus_zero.exact_eq(zero));
    }

    #[test]
    fn test_exact_eq_f32() {
        assert_eq!(0.0f32, -0.0f32);
        assert!(!(0.0f32.exact_eq(-0.0f32)));
        assert!(f32::NAN.exact_eq(f32::NAN));
    }
}
