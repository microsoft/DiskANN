/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bytes(usize);

impl Bytes {
    pub const CACHELINE: Self = Self::new(64);
    pub const ZERO: Self = Self::new(0);

    #[inline]
    pub const fn new(bytes: usize) -> Self {
        Self(bytes)
    }

    #[inline]
    pub const fn value(self) -> usize {
        self.0
    }

    #[inline]
    pub const fn checked_add(self, other: Bytes) -> Option<Bytes> {
        match self.value().checked_add(other.value()) {
            Some(v) => Some(Bytes::new(v)),
            None => None,
        }
    }

    #[inline]
    pub const fn checked_mul(self, other: usize) -> Option<Bytes> {
        match self.value().checked_mul(other) {
            Some(v) => Some(Bytes::new(v)),
            None => None,
        }
    }

    #[inline]
    pub(crate) const fn unchecked_mul(self, other: usize) -> Bytes {
        Bytes::new(self.value() * other)
    }

    #[inline]
    pub const fn checked_sub(self, other: Bytes) -> Option<Bytes> {
        match self.value().checked_sub(other.value()) {
            Some(v) => Some(Bytes::new(v)),
            None => None,
        }
    }

    #[inline]
    pub(crate) const fn unchecked_sub(self, other: Bytes) -> Bytes {
        Self::new(self.value() - other.value())
    }

    #[inline]
    pub const fn checked_next_multiple_of(self, other: Bytes) -> Option<Bytes> {
        match self.value().checked_next_multiple_of(other.value()) {
            Some(v) => Some(Bytes::new(v)),
            None => None,
        }
    }

    #[inline]
    pub const fn size_of<T>() -> Self {
        Self::new(std::mem::size_of::<T>())
    }

    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Display for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} bytes", self.value())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Align(NonZeroUsize);

impl Align {
    pub const fn new(value: usize) -> Option<Self> {
        match NonZeroUsize::new(value) {
            Some(value) => {
                if value.is_power_of_two() {
                    Some(Self(value))
                } else {
                    None
                }
            }
            None => None,
        }
    }

    pub const fn value(self) -> usize {
        self.0.get()
    }

    pub const unsafe fn new_unchecked(value: usize) -> Self {
        debug_assert!(value.is_power_of_two());
        Self(unsafe { NonZeroUsize::new_unchecked(value) })
    }

    pub const fn of<T>() -> Self {
        // SAFETY: `std::mem::align_of` is guaranteed to return a power of 2.
        unsafe { Self::new_unchecked(std::mem::align_of::<T>()) }
    }

    pub const fn from_layout(layout: std::alloc::Layout) -> Self {
        // SAFETY: `Layout::align` is guaranteed to be a power of 2.
        unsafe { Self::new_unchecked(layout.align()) }
    }

    // Constants.
    pub const _1: Self = Self::new(1).unwrap();
    pub const _2: Self = Self::new(2).unwrap();
    pub const _4: Self = Self::new(4).unwrap();
    pub const _8: Self = Self::new(8).unwrap();
    pub const _16: Self = Self::new(16).unwrap();
    pub const _32: Self = Self::new(32).unwrap();
    pub const _64: Self = Self::new(64).unwrap();
    pub const _128: Self = Self::new(128).unwrap();
    pub const _256: Self = Self::new(256).unwrap();
    pub const _512: Self = Self::new(512).unwrap();
    pub const _1024: Self = Self::new(1024).unwrap();
    pub const _2048: Self = Self::new(2048).unwrap();
    pub const _4096: Self = Self::new(4096).unwrap();
}

impl std::fmt::Display for Align {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_value_roundtrip() {
        assert_eq!(Bytes::new(42).value(), 42);
        assert_eq!(Bytes::new(0).value(), 0);
    }

    #[test]
    fn cacheline_constant() {
        assert_eq!(Bytes::CACHELINE, Bytes::new(64));
    }

    #[test]
    fn size_of_returns_correct_size() {
        assert_eq!(Bytes::size_of::<u8>(), Bytes::new(1));
        assert_eq!(Bytes::size_of::<u64>(), Bytes::new(8));
        assert_eq!(Bytes::size_of::<[u8; 128]>(), Bytes::new(128));
    }

    #[test]
    fn checked_add_success() {
        assert_eq!(
            Bytes::new(10).checked_add(Bytes::new(20)),
            Some(Bytes::new(30))
        );
    }

    #[test]
    fn checked_add_overflow() {
        assert_eq!(Bytes::new(usize::MAX).checked_add(Bytes::new(1)), None);
    }

    #[test]
    fn checked_sub_success() {
        assert_eq!(
            Bytes::new(30).checked_sub(Bytes::new(10)),
            Some(Bytes::new(20))
        );
    }

    #[test]
    fn checked_sub_underflow() {
        assert_eq!(Bytes::new(5).checked_sub(Bytes::new(10)), None);
    }

    #[test]
    fn checked_mul_success() {
        assert_eq!(Bytes::new(64).checked_mul(4), Some(Bytes::new(256)));
    }

    #[test]
    fn checked_mul_overflow() {
        assert_eq!(Bytes::new(usize::MAX).checked_mul(2), None);
    }

    #[test]
    fn checked_mul_by_zero() {
        assert_eq!(Bytes::new(100).checked_mul(0), Some(Bytes::new(0)));
    }

    #[test]
    fn unchecked_mul() {
        assert_eq!(Bytes::new(64).unchecked_mul(3), Bytes::new(192));
    }

    #[test]
    fn unchecked_sub() {
        assert_eq!(
            Bytes::new(100).unchecked_sub(Bytes::new(30)),
            Bytes::new(70)
        );
    }

    #[test]
    fn checked_next_multiple_of_already_aligned() {
        assert_eq!(
            Bytes::new(128).checked_next_multiple_of(Bytes::new(64)),
            Some(Bytes::new(128))
        );
    }

    #[test]
    fn checked_next_multiple_of_rounds_up() {
        assert_eq!(
            Bytes::new(100).checked_next_multiple_of(Bytes::new(64)),
            Some(Bytes::new(128))
        );
    }

    #[test]
    fn checked_next_multiple_of_overflow() {
        assert_eq!(
            Bytes::new(usize::MAX).checked_next_multiple_of(Bytes::new(2)),
            None
        );
    }

    #[test]
    fn ordering() {
        assert!(Bytes::new(10) < Bytes::new(20));
        assert!(Bytes::new(20) > Bytes::new(10));
        assert_eq!(Bytes::new(5), Bytes::new(5));
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", Bytes::new(256)), "256 bytes");
    }

    // Align tests

    #[test]
    fn align_new_power_of_two() {
        assert_eq!(Align::new(1).unwrap().value(), 1);
        assert_eq!(Align::new(2).unwrap().value(), 2);
        assert_eq!(Align::new(64).unwrap().value(), 64);
        assert_eq!(Align::new(4096).unwrap().value(), 4096);
    }

    #[test]
    fn align_new_rejects_zero() {
        assert!(Align::new(0).is_none());
    }

    #[test]
    fn align_new_rejects_non_power_of_two() {
        assert!(Align::new(3).is_none());
        assert!(Align::new(5).is_none());
        assert!(Align::new(6).is_none());
        assert!(Align::new(100).is_none());
    }

    #[test]
    fn align_of_matches_std() {
        assert_eq!(Align::of::<()>().value(), 1);
        assert_eq!(Align::of::<u8>().value(), std::mem::align_of::<u8>());
        assert_eq!(Align::of::<u64>().value(), std::mem::align_of::<u64>());
        assert_eq!(Align::of::<u128>().value(), std::mem::align_of::<u128>());
    }

    #[test]
    fn align_from_layout() {
        let layout = std::alloc::Layout::from_size_align(256, 128).unwrap();
        assert_eq!(Align::from_layout(layout).value(), 128);
    }

    #[test]
    fn align_constants() {
        assert_eq!(Align::_1.value(), 1);
        assert_eq!(Align::_2.value(), 2);
        assert_eq!(Align::_4.value(), 4);
        assert_eq!(Align::_8.value(), 8);
        assert_eq!(Align::_16.value(), 16);
        assert_eq!(Align::_32.value(), 32);
        assert_eq!(Align::_64.value(), 64);
        assert_eq!(Align::_128.value(), 128);
        assert_eq!(Align::_256.value(), 256);
        assert_eq!(Align::_512.value(), 512);
        assert_eq!(Align::_1024.value(), 1024);
        assert_eq!(Align::_2048.value(), 2048);
        assert_eq!(Align::_4096.value(), 4096);
    }

    #[test]
    fn align_ordering() {
        assert!(Align::_1 < Align::_64);
        assert!(Align::_128 > Align::_64);
        assert_eq!(Align::_32, Align::new(32).unwrap());
    }

    #[test]
    fn align_display() {
        assert_eq!(format!("{}", Align::_64), "64");
    }
}
