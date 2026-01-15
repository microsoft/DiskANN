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

impl PowerOfTwo {
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
        if m == 0 {
            0
        } else {
            self.raw() - m
        }
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
