/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Number types with limited dynamic range.

use std::{fmt::Debug, num::NonZeroUsize};

use thiserror::Error;

/// A number type that must be greater than zero.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Positive<T>(T)
where
    T: PartialOrd + Default + Debug;

#[derive(Debug, Clone, Copy, Error)]
#[error("value {:?} is not greater than {:?} (its default value)", .0, T::default())]
pub struct NotPositiveError<T: Debug + Default>(T);

impl<T> Positive<T>
where
    T: PartialOrd + Default + Debug,
{
    /// Create a new `Positive` if the given value is greater than 0 (`T::default()`);
    pub fn new(value: T) -> Result<Self, NotPositiveError<T>> {
        if value > T::default() {
            Ok(Self(value))
        } else {
            Err(NotPositiveError(value))
        }
    }

    /// Create a new `Positive` without checking whether the value is greater than 0.
    ///
    /// # Safety
    ///
    /// The value must be greater than `T::default()`.
    pub const unsafe fn new_unchecked(value: T) -> Self {
        Self(value)
    }

    /// Consume `self` and return the inner value.
    pub fn into_inner(self) -> T {
        self.0
    }
}

// SAFETY: 1.0 is positive.
pub(crate) const POSITIVE_ONE_F32: Positive<f32> = unsafe { Positive::new_unchecked(1.0) };

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct PowerOfTwo(NonZeroUsize);

#[derive(Debug, Clone, Copy, Error)]
#[error("value {0} must be a power of two")]
#[non_exhaustive]
pub struct NotPowerOfTwo(usize);

/// PowerOfTwo constants for all valid exponents (0..64)
///
/// # Safety
///
/// 1 << N is a power of two for all N in 0..64.
#[allow(clippy::undocumented_unsafe_blocks)]
impl PowerOfTwo {
    pub const P0: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 0).unwrap()) };
    pub const P1: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 1).unwrap()) };
    pub const P2: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 2).unwrap()) };
    pub const P3: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 3).unwrap()) };
    pub const P4: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 4).unwrap()) };
    pub const P5: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 5).unwrap()) };
    pub const P6: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 6).unwrap()) };
    pub const P7: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 7).unwrap()) };
    pub const P8: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 8).unwrap()) };
    pub const P9: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 9).unwrap()) };
    pub const P10: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 10).unwrap()) };
    pub const P11: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 11).unwrap()) };
    pub const P12: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 12).unwrap()) };
    pub const P13: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 13).unwrap()) };
    pub const P14: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 14).unwrap()) };
    pub const P15: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 15).unwrap()) };
    pub const P16: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 16).unwrap()) };
    pub const P17: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 17).unwrap()) };
    pub const P18: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 18).unwrap()) };
    pub const P19: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 19).unwrap()) };
    pub const P20: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 20).unwrap()) };
    pub const P21: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 21).unwrap()) };
    pub const P22: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 22).unwrap()) };
    pub const P23: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 23).unwrap()) };
    pub const P24: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 24).unwrap()) };
    pub const P25: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 25).unwrap()) };
    pub const P26: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 26).unwrap()) };
    pub const P27: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 27).unwrap()) };
    pub const P28: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 28).unwrap()) };
    pub const P29: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 29).unwrap()) };
    pub const P30: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 30).unwrap()) };
    pub const P31: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 31).unwrap()) };
    pub const P32: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 32).unwrap()) };
    pub const P33: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 33).unwrap()) };
    pub const P34: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 34).unwrap()) };
    pub const P35: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 35).unwrap()) };
    pub const P36: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 36).unwrap()) };
    pub const P37: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 37).unwrap()) };
    pub const P38: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 38).unwrap()) };
    pub const P39: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 39).unwrap()) };
    pub const P40: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 40).unwrap()) };
    pub const P41: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 41).unwrap()) };
    pub const P42: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 42).unwrap()) };
    pub const P43: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 43).unwrap()) };
    pub const P44: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 44).unwrap()) };
    pub const P45: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 45).unwrap()) };
    pub const P46: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 46).unwrap()) };
    pub const P47: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 47).unwrap()) };
    pub const P48: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 48).unwrap()) };
    pub const P49: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 49).unwrap()) };
    pub const P50: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 50).unwrap()) };
    pub const P51: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 51).unwrap()) };
    pub const P52: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 52).unwrap()) };
    pub const P53: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 53).unwrap()) };
    pub const P54: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 54).unwrap()) };
    pub const P55: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 55).unwrap()) };
    pub const P56: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 56).unwrap()) };
    pub const P57: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 57).unwrap()) };
    pub const P58: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 58).unwrap()) };
    pub const P59: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 59).unwrap()) };
    pub const P60: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 60).unwrap()) };
    pub const P61: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 61).unwrap()) };
    pub const P62: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 62).unwrap()) };
    pub const P63: Self = unsafe { Self::new_unchecked(NonZeroUsize::new(1 << 63).unwrap()) };
}

impl PowerOfTwo {
    /// V values for some of the equivalent P(owersOfTwo)'s
    pub const V1: Self = Self::P0;
    pub const V2: Self = Self::P1;
    pub const V4: Self = Self::P2;
    pub const V8: Self = Self::P3;
    pub const V16: Self = Self::P4;
    pub const V32: Self = Self::P5;
    pub const V64: Self = Self::P6;
    pub const V128: Self = Self::P7;
    pub const V256: Self = Self::P8;
    pub const V512: Self = Self::P9;
    pub const V1024: Self = Self::P10;
    pub const V2048: Self = Self::P11;
    pub const V4096: Self = Self::P12;
    pub const V8192: Self = Self::P13;
    pub const V16384: Self = Self::P14;
    pub const V32768: Self = Self::P15;
    pub const V65536: Self = Self::P16;
    pub const V131072: Self = Self::P17;
    pub const V262144: Self = Self::P18;
    pub const V524288: Self = Self::P19;
    pub const V1048576: Self = Self::P20;
    pub const V2097152: Self = Self::P21;
    pub const V4194304: Self = Self::P22;
    pub const V8388608: Self = Self::P23;
    pub const V16777216: Self = Self::P24;
    pub const V33554432: Self = Self::P25;
    pub const V67108864: Self = Self::P26;
    pub const V134217728: Self = Self::P27;
    pub const V268435456: Self = Self::P28;
    pub const V536870912: Self = Self::P29;
    pub const V1073741824: Self = Self::P30;
    pub const V2147483648: Self = Self::P31;
    pub const V4294967296: Self = Self::P32;
    pub const V8589934592: Self = Self::P33;
    pub const V17179869184: Self = Self::P34;
    pub const V34359738368: Self = Self::P35;
    pub const V68719476736: Self = Self::P36;
    pub const V137438953472: Self = Self::P37;
    pub const V274877906944: Self = Self::P38;
    pub const V549755813888: Self = Self::P39;
    pub const V1099511627776: Self = Self::P40;
    pub const V2199023255552: Self = Self::P41;
    pub const V4398046511104: Self = Self::P42;
    pub const V8796093022208: Self = Self::P43;
    pub const V17592186044416: Self = Self::P44;
    pub const V35184372088832: Self = Self::P45;
    pub const V70368744177664: Self = Self::P46;
    pub const V140737488355328: Self = Self::P47;
    pub const V281474976710656: Self = Self::P48;
    pub const V562949953421312: Self = Self::P49;
    pub const V1125899906842624: Self = Self::P50;
    pub const V2251799813685248: Self = Self::P51;
    pub const V4503599627370496: Self = Self::P52;
    pub const V9007199254740992: Self = Self::P53;
    pub const V18014398509481984: Self = Self::P54;
    pub const V36028797018963968: Self = Self::P55;
    pub const V72057594037927936: Self = Self::P56;
    pub const V144115188075855872: Self = Self::P57;
    pub const V288230376151711744: Self = Self::P58;
    pub const V576460752303423488: Self = Self::P59;
    pub const V1152921504606846976: Self = Self::P60;
    pub const V2305843009213693952: Self = Self::P61;
    pub const V4611686018427387904: Self = Self::P62;
    pub const V9223372036854775808: Self = Self::P63;

    /// Create a new `PowerOfTwo` if the given value is greater a power of two.
    pub const fn new(value: usize) -> Result<Self, NotPowerOfTwo> {
        let v = match NonZeroUsize::new(value) {
            Some(value) => value,
            None => return Err(NotPowerOfTwo(value)),
        };
        if v.is_power_of_two() {
            // Safety: We just checked.
            Ok(unsafe { Self::new_unchecked(v) })
        } else {
            Err(NotPowerOfTwo(value))
        }
    }

    /// Return the smallest power of two greater than or equal to `value`. If the next
    /// power of two is greater than `usize::MAX`, `None` is returned.
    pub const fn next(value: usize) -> Option<Self> {
        // Note: use `match` instead of `Option::map` for `const`-compatibility.
        match value.checked_next_power_of_two() {
            // SAFETY: We trust the implementation of `usize::checked_next_power_of_two` since:
            //
            // * 0 can never be a power of two and thus cannot be returned.
            // * If it succeeds, the returned value should be a power of two.
            Some(v) => Some(unsafe { Self::new_unchecked(NonZeroUsize::new_unchecked(v)) }),
            None => None,
        }
    }

    /// Create a new `PowerOfTwo` without checking whether the value is a power of two.
    ///
    /// # Safety
    ///
    /// The value must be a power of two.
    pub const unsafe fn new_unchecked(value: NonZeroUsize) -> Self {
        Self(value)
    }

    /// Consume `self` and return the inner value.
    pub const fn into_inner(self) -> NonZeroUsize {
        self.0
    }

    /// Consume `self` and return the inner value as a `usize`.
    pub const fn raw(self) -> usize {
        self.0.get()
    }

    /// Construct `self` from the alignment in `layout`.
    pub const fn from_align(layout: &std::alloc::Layout) -> Self {
        // SAFETY: Alignment is guaranteed to be a power of two:
        // - <https://doc.rust-lang.org/beta/std/alloc/struct.Layout.html#method.align>
        unsafe { Self::new_unchecked(NonZeroUsize::new_unchecked(layout.align())) }
    }

    /// Return the alignment of `T` as a power of two.
    pub const fn alignment_of<T>() -> Self {
        // SAFETY: Alignment is guaranteed to be a power of two:
        // - <https://doc.rust-lang.org/beta/std/alloc/struct.Layout.html#method.align>
        unsafe { Self::new_unchecked(NonZeroUsize::new_unchecked(std::mem::align_of::<T>())) }
    }

    /// Compute the operation `lhs % self`.
    ///
    /// # Note
    ///
    /// The argument order of this function is reversed from the typical `align_offset`
    /// method in the standard library.
    pub const fn arg_mod(self, lhs: usize) -> usize {
        lhs & (self.raw() - 1)
    }

    /// Compute the amount `x` that would have to be added to `lhs` so the quantity
    /// `lhs + x` is a multiple of `self`.
    ///
    /// # Note
    ///
    /// The argument order of this function is reversed from the typical `align_offset`
    /// method in the standard library.
    pub const fn arg_align_offset(self, lhs: usize) -> usize {
        let m = self.arg_mod(lhs);
        if m == 0 { 0 } else { self.raw() - m }
    }

    /// Calculate the smallest value greater than or equal to `lhs` that is a multiple of
    /// `self`. Return `None` if the operation would result in an overflow.
    pub const fn arg_checked_next_multiple_of(self, lhs: usize) -> Option<usize> {
        let offset = self.arg_align_offset(lhs);
        lhs.checked_add(offset)
    }
}

impl From<PowerOfTwo> for usize {
    #[inline(always)]
    fn from(value: PowerOfTwo) -> Self {
        value.raw()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format_not_positive_error<T>(value: T) -> String
    where
        T: Debug + Default,
    {
        format!(
            "value {:?} is not greater than {:?} (its default value)",
            value,
            T::default(),
        )
    }

    #[test]
    fn test_positive_f32() {
        let x = Positive::<f32>::new(1.0);
        assert!(x.is_ok());
        let x = x.unwrap();
        assert_eq!(x.into_inner(), 1.0);

        // Using 0 should return an error.
        let x = Positive::<f32>::new(0.0);
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            format_not_positive_error::<f32>(0.0)
        );

        // Using -1 should return an error.
        let x = Positive::<f32>::new(-1.0);
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            format_not_positive_error::<f32>(-1.0)
        );

        // SAFETY: 1.0 is greater than zero.
        let x = unsafe { Positive::<f32>::new_unchecked(1.0) };
        assert_eq!(x.into_inner(), 1.0);
    }

    #[test]
    fn test_positive_i64() {
        let x = Positive::<i64>::new(1);
        assert!(x.is_ok());
        let x = x.unwrap();
        assert_eq!(x.into_inner(), 1);

        // Using 0 should return an error.
        let x = Positive::<i64>::new(0);
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            format_not_positive_error::<i64>(0)
        );

        // Using -1 should return an error.
        let x = Positive::<i64>::new(-1);
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            format_not_positive_error::<i64>(-1)
        );

        // SAFETY: 1 is greater than zero.
        let x = unsafe { Positive::<i64>::new_unchecked(1) };
        assert_eq!(x.into_inner(), 1);
    }

    #[test]
    fn test_power_of_two() {
        assert!(PowerOfTwo::new(0).is_err());
        assert_eq!(PowerOfTwo::next(0).unwrap(), PowerOfTwo::new(1).unwrap());
        for i in 0..63 {
            let base = 2usize.pow(i);
            let p = PowerOfTwo::new(base).unwrap();
            assert_eq!(p.into_inner().get(), base);
            assert_eq!(p.raw(), base);
            assert_eq!(PowerOfTwo::new(base).unwrap().raw(), base);
            assert_eq!(<_ as Into<usize>>::into(p), base);

            if i != 1 {
                assert!(PowerOfTwo::new(base - 1).is_err(), "failed for i = {}", i);
                assert_eq!(PowerOfTwo::next(base - 1).unwrap().raw(), base);
            }

            if i != 0 {
                assert!(PowerOfTwo::new(base + 1).is_err(), "failed for i = {}", i);
            }

            assert_eq!(p.arg_mod(0), 0);
            assert_eq!(p.arg_mod(p.raw()), 0);

            assert_eq!(p.arg_align_offset(0), 0);
            assert_eq!(p.arg_align_offset(base), 0);

            assert_eq!(p.arg_checked_next_multiple_of(0), Some(0));
            assert_eq!(p.arg_checked_next_multiple_of(base), Some(base));

            assert_eq!(p.arg_checked_next_multiple_of(1), Some(base));

            if i > 1 {
                assert_eq!(p.arg_mod(base + 1), 1);
                assert_eq!(p.arg_mod(2 * base - 1), base - 1);

                assert_eq!(p.arg_align_offset(base + 1), base - 1);
                assert_eq!(p.arg_align_offset(2 * base - 1), 1);

                assert_eq!(p.arg_checked_next_multiple_of(base + 1), Some(2 * base));
                assert_eq!(p.arg_checked_next_multiple_of(2 * base - 1), Some(2 * base));
            }
        }

        assert!(PowerOfTwo::next(2usize.pow(63) + 1).is_none());
        assert!(PowerOfTwo::next(usize::MAX).is_none());
    }
}
