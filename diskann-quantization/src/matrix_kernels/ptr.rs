/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, ptr::NonNull};

use super::{
    bounds::{self, Bound},
    num::Elements,
};

//-------//
// Slice //
//-------//

/// An immutable slice with length tracking when debug-assertions are enabled.
#[derive(Debug)]
pub(crate) struct Slice<'a, T> {
    ptr: NonNull<T>,
    len: Bound,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T> Slice<'a, T> {
    /// Construct a new [`Slice`] with the same length and base pointer as `slice`.
    pub(crate) const fn new(slice: &'a [T]) -> Self {
        unsafe { Self::from_raw(slice_to_nonnull(slice), Bound::new(slice.len())) }
    }

    /// Construct a new [`Slice`] from raw parts.
    ///
    /// # Safety
    ///
    /// This function has the same requirements as [`std::slice::from_raw_parts`].
    pub(super) const unsafe fn from_raw(ptr: NonNull<T>, len: Bound) -> Self {
        Self {
            ptr,
            len,
            _lifetime: PhantomData,
        }
    }

    /// Return the base pointer for `self`.
    pub(super) const fn as_ptr(&self) -> *const T {
        self.as_nonnull().as_ptr().cast_const()
    }

    /// Return the base pointer for `self`.
    pub(super) const fn as_nonnull(&self) -> NonNull<T> {
        self.ptr
    }

    pub(super) unsafe fn as_std_slice(self, len: usize) -> &'a [T] {
        bounds::check_eq!(self.len, len);
        unsafe { std::slice::from_raw_parts(self.as_ptr(), len) }
    }

    /// Return a reference to the only element.
    ///
    /// # Safety
    ///
    /// This function has the same safety reqauirements as [`std::ptr::as_ref_unchecked`].
    /// Additionally, this function will panic if the true length of `self` is not equal
    /// to 1.
    pub(super) unsafe fn as_ref(self) -> &'a T {
        bounds::check_eq!(self.len, 1, "slice must have a length of exactly one",);

        unsafe { self.ptr.as_ref() }
    }

    /// Add an unsigned `offset` to `self.
    ///
    /// When debug assertions are enabled, invalid `offsets` will panic.
    ///
    /// # Safety
    ///
    /// This function has the same safety requirements as [`std::ptr::add`].
    pub(super) unsafe fn add(self, offset: Elements<T>) -> Slice<'a, T> {
        let offset = offset.value();

        // Debug check that there is room.
        bounds::check_ge!(self.len, offset, "offset would go out-of-bounds");

        unsafe { Self::from_raw(self.ptr.add(offset), self.len - Bound::new(offset)) }
    }

    /// Shorten the tracked length to `length`.
    ///
    /// When debug assertions are enabled, this function will panic if `length` is less than
    /// the tracked length.
    ///
    /// # Safety
    ///
    /// `length` must be less than or equal to the length provenance of self.
    pub(super) unsafe fn truncate(self, length: Elements<T>) -> Slice<'a, T> {
        let length = length.value();
        bounds::check_ge!(self.len, length, "truncation would make the slice longer");

        unsafe { Self::from_raw(self.ptr, Bound::new(length)) }
    }

    /// Return a new [`Slice`] with the same base pointer as `self` and length 1.
    ///
    /// # Safety
    ///
    /// The true length of `self` must be at least one.
    pub(super) unsafe fn as_unit(self) -> Slice<'a, T> {
        unsafe { self.truncate(Elements::new(1)) }
    }

    /// Return the tracked [`Bound`].
    pub(super) fn len(&self) -> Bound {
        self.len
    }
}

impl<T> Clone for Slice<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Slice<'_, T> {}

#[cfg(test)]
impl<'a, T> Slice<'a, T> {
    fn checked_as_std_slice(self, len: usize) -> &'a [T] {
        // SAFETY: This operation is check under `test/debug_assertions`.
        unsafe { self.as_std_slice(len) }
    }

    fn checked_as_ref(self) -> &'a T {
        // SAFETY: This operation is check under `test/debug_assertions`.
        unsafe { self.as_ref() }
    }

    fn checked_add(self, offset: Elements<T>) -> Slice<'a, T> {
        // SAFETY: This operation is check under `test/debug_assertions`.
        unsafe { self.add(offset) }
    }

    fn checked_truncate(self, length: Elements<T>) -> Slice<'a, T> {
        // SAFETY: This operation is check under `test/debug_assertions`.
        unsafe { self.truncate(length) }
    }

    fn checked_as_unit(self) -> Slice<'a, T> {
        // SAFETY: This operation is check under `test/debug_assertions`.
        unsafe { self.as_unit() }
    }
}

//----------//
// MutSlice //
//----------//

/// An mutable slice with length tracking when debug-assertions are enabled.
#[derive(Debug)]
pub(crate) struct MutSlice<'a, T> {
    ptr: NonNull<T>,
    len: Bound,
    _lifetime: PhantomData<&'a mut T>,
}

impl<'a, T> MutSlice<'a, T> {
    /// Construct a new [`MutSlice`] with the same length and base pointer as `slice`.
    pub(crate) const fn new(slice: &'a mut [T]) -> Self {
        unsafe { Self::from_raw(mut_slice_to_nonnull(slice), Bound::new(slice.len())) }
    }

    /// Construct a new [`MutSlice`] from raw parts.
    ///
    /// # Safety
    ///
    /// This function has the same requirements as [`std::slice::from_raw_parts_mut`].
    pub(super) const unsafe fn from_raw(ptr: NonNull<T>, len: Bound) -> Self {
        Self {
            ptr,
            len,
            _lifetime: PhantomData,
        }
    }

    /// Materialize `self` as a slice.
    ///
    /// # Safety
    ///
    /// This function has the same requirements as [`std::slice::from_raw_parts`] with
    /// the addition that `len` **must** be the true length of `self`.
    pub(super) unsafe fn as_std_slice(&self, len: usize) -> &[T] {
        bounds::check_eq!(self.len(), len);
        unsafe { std::slice::from_raw_parts(self.as_ptr(), len) }
    }

    /// Materialize `self` as a mutable slice.
    ///
    /// # Safety
    ///
    /// This function has the same requirements as [`std::slice::from_raw_parts_mut`] with
    /// the addition that `len` **must** be the true length of `self`.
    pub(super) unsafe fn as_std_mut_slice(&mut self, len: usize) -> &mut [T] {
        bounds::check_eq!(self.len(), len);
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), len) }
    }

    /// Return the base pointer for `self`.
    pub(super) const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr().cast_const()
    }

    /// Return the base pointer for `self`.
    pub(super) const fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Borrow the region of memory in `[start, start + length)`.
    ///
    /// # Safety
    ///
    /// The entire region `[start, start + length)` must be within `self`. In debug builds,
    /// out-of-bounds accesses will panic.
    #[must_use]
    pub(super) unsafe fn subslice(&mut self, start: usize, length: Bound) -> MutSlice<'_, T> {
        bounds::check_ge!(self.len, start, "start is out-of-bounds");
        bounds::check_ge!(self.len, Bound::new(start) + length, "end is out-of-bounds");

        unsafe { Self::from_raw(self.ptr.add(start), length) }
    }

    /// Borrow `self` as a fixed-size slice.
    ///
    /// # Safety
    ///
    /// The length of `self` must be exactly `N`. This is checked in debug builds.
    pub(super) unsafe fn as_array<const N: usize>(&mut self) -> &mut [T; N] {
        bounds::check_eq!(self.len, N, "invalid materialization of size {N}");
        unsafe { &mut *self.as_mut_ptr().cast::<[T; N]>() }
    }

    /// Return the [`Bound`] representin the slice's length.
    pub(super) fn len(&self) -> Bound {
        self.len
    }
}

#[cfg(test)]
impl<'a, T> MutSlice<'a, T> {
    fn checked_as_std_mut_slice(&mut self, len: usize) -> &mut [T] {
        // SAFETY: This operation is check under `test/debug_assertions`.
        unsafe { self.as_std_mut_slice(len) }
    }

    fn checked_subslice(&mut self, start: usize, length: Bound) -> MutSlice<'_, T> {
        // SAFETY: This operation is check under `test/debug_assertions`.
        unsafe { self.subslice(start, length) }
    }

    fn checked_as_array<const N: usize>(&mut self) -> &mut [T; N] {
        // SAFETY: This operation is check under `test/debug_assertions`.
        unsafe { self.as_array::<N>() }
    }
}

const fn slice_to_nonnull<T>(x: &[T]) -> NonNull<T> {
    // SAFETY: Slices always have non-null base pointers.
    unsafe { NonNull::new_unchecked(x.as_ptr().cast::<T>().cast_mut()) }
}

const fn mut_slice_to_nonnull<T>(x: &mut [T]) -> NonNull<T> {
    // SAFETY: Slices always have non-null base pointers.
    unsafe { NonNull::new_unchecked(x.as_mut_ptr().cast::<T>()) }
}

// Verify that length-tracking disappears in release builds.
#[cfg(not(any(test, debug_assertions)))]
const _: () = assert!(
    std::mem::size_of::<Slice<'static, f32>>() == std::mem::size_of::<NonNull<f32>>(),
    "non-debug `Slice` does not have the expected size"
);

#[cfg(not(any(test, debug_assertions)))]
const _: () = assert!(
    std::mem::size_of::<MutSlice<'static, f32>>() == std::mem::size_of::<NonNull<f32>>(),
    "non-debug `MutSlice` does not have the expected size"
);

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::matrix_kernels::test_util::{assert_contains, panic_message_for};

    #[test]
    fn test_slice() {
        let bases: [&[usize]; _] = [&[], &[1], &[1, 2, 3], &[1, 2, 3, 4]];

        for base in bases {
            let ptr = base.as_ptr();
            let s = Slice::new(base);

            assert_eq!(s.len().value(), base.len());
            assert_eq!(s.as_ptr(), ptr);
            assert_eq!(s.as_nonnull().as_ptr().cast_const(), ptr);

            let reconstructed = s.checked_as_std_slice(base.len());
            assert_eq!(reconstructed, base);

            for invalid_len in [base.len().checked_sub(1), base.len().checked_add(1)]
                .into_iter()
                .flatten()
            {
                let _ = panic_message_for(|| {
                    let _ = s.checked_as_std_slice(invalid_len);
                });
            }

            if base.len() != 1 {
                let message = panic_message_for(|| {
                    let _ = s.checked_as_ref();
                });

                assert_contains!(message, "slice must have a length of exactly one");
            }

            if base.is_empty() {
                let message = panic_message_for(|| {
                    let _ = s.checked_as_unit();
                });

                assert_contains!(message, "truncation would make the slice longer");
            } else {
                for i in 0..base.len() {
                    let offset = s.checked_add(Elements::new(i));
                    assert_eq!(offset.len().value(), base.len() - i);
                    let v = *offset.checked_as_unit().checked_as_ref();
                    assert_eq!(v, base[i]);
                }
            }

            // Adds equal to the length are allowed - but the resulting size is zero.
            {
                let empty = s.checked_add(Elements::new(base.len()));
                assert_eq!(empty.len().value(), 0);
            }

            // Out-of-bounds "adds" should panic.
            for i in 1..3 {
                let message = panic_message_for(|| {
                    // SAEFTY: The provided `offset` exceeds the true length of `s`, but
                    // this is caught in debug builds.
                    let _ = unsafe { s.add(Elements::new(base.len() + i)) };
                });

                assert_contains!(message, "offset would go out-of-bounds");
            }

            // Truncate
            for i in 0..=base.len() {
                let truncated = s.checked_truncate(Elements::new(i));

                assert_eq!(truncated.len().value(), i);
                assert_eq!(truncated.as_ptr(), ptr, "base pointer should be unchanged");
            }

            // Truncate - out-of-bounds
            for i in 1..3 {
                let message = panic_message_for(|| {
                    // SAEFTY: The provided `length` exceeds the true length of `s`, but
                    // this is caught in debug builds.
                    let _ = unsafe { s.truncate(Elements::new(base.len() + i)) };
                });

                assert_contains!(message, "truncation would make the slice longer");
            }
        }
    }

    #[test]
    fn test_mut_slice_subslice() {
        let mut base = [0u32; 8];
        let base_len = base.len();
        for start in 0..=base.len() {
            for len in 0..=(base.len() - start) {
                base.fill(0);
                let ptr = base.as_mut_ptr();

                {
                    let mut s = MutSlice::new(&mut base);
                    assert_eq!(s.len().value(), base_len);
                    assert_eq!(s.as_mut_ptr(), ptr);

                    let mut sub = s.checked_subslice(start, Bound::new(len));
                    let as_mut = sub.checked_as_std_mut_slice(len);
                    as_mut.fill(1);
                }

                let expected = core::array::from_fn(|i| {
                    if i >= start && i < start + len {
                        1u32
                    } else {
                        0
                    }
                });

                assert_eq!(expected, base, "start = {start}, len = {len}");
            }

            // Out-of-bounds length
            for i in 1..3 {
                let len = (base.len() - start) + i;
                let message = panic_message_for(|| {
                    // Copy `base` since `&mut` is not unwide-safe.
                    let mut copy = base;
                    let mut s = MutSlice::new(&mut copy);
                    // SAFETY: We're indexing out of bounds - but in debug builds this should
                    // be caught.
                    let _ = unsafe { s.subslice(start, Bound::new(len)) };
                });

                assert_contains!(message, "end is out-of-bounds");
            }
        }

        let message = panic_message_for(|| {
            // Copy `base` since `&mut` is not unwide-safe.
            let mut copy = base;
            let mut s = MutSlice::new(&mut copy);

            // SAFETY: This access is unsafe - but caught in debug builds.
            let _ = unsafe { s.subslice(base.len() + 1, Bound::new(0)) };
        });

        assert_contains!(message, "start is out-of-bounds");
    }

    #[test]
    fn test_mut_slice_as_array() {
        let mut parent = [1, 2];

        let mut s = MutSlice::<usize>::new(&mut []);
        let _ = s.checked_as_array::<0>();

        let mut s = MutSlice::new(&mut parent[..1]);
        let v = s.checked_as_array::<1>();
        assert_eq!(*v, [1]);

        let mut s = MutSlice::new(&mut parent);
        let v = s.checked_as_array::<2>();
        assert_eq!(*v, [1, 2]);

        let mut mutated = [1, 2];
        let mut s = MutSlice::new(&mut mutated);
        s.checked_as_array::<2>()[1] = 3;
        assert_eq!(mutated, [1, 3]);

        let message = panic_message_for(|| {
            let mut copy = parent;
            let mut s = MutSlice::new(&mut copy);
            let _ = s.checked_as_array::<1>();
        });

        assert_contains!(message, "invalid materialization of size 1");

        let message = panic_message_for(|| {
            let mut copy = parent;
            let mut s = MutSlice::new(&mut copy);
            let _ = s.checked_as_array::<4>();
        });

        assert_contains!(message, "invalid materialization of size 4");
    }

    #[test]
    fn test_mut_slice_as_std_mut_slice_rejects_invalid_lengths() {
        for invalid_len in [1, 3] {
            let _ = panic_message_for(|| {
                let mut base = [1, 2];
                let mut s = MutSlice::new(&mut base);
                let _ = s.checked_as_std_mut_slice(invalid_len);
            });
        }
    }

    #[test]
    fn test_zero_sized_elements() {
        let base = [(); 4];
        let ptr = base.as_ptr();
        let s = Slice::new(&base);

        for offset in 0..=base.len() {
            let remaining = s.checked_add(Elements::new(offset));
            assert_eq!(remaining.as_ptr(), ptr);
            assert_eq!(remaining.len().value(), base.len() - offset);
            assert_eq!(
                remaining.checked_as_std_slice(base.len() - offset).len(),
                base.len() - offset
            );
        }

        let mut base = [(); 4];
        let ptr = base.as_mut_ptr();
        let mut s = MutSlice::new(&mut base);
        let mut sub = s.checked_subslice(2, Bound::new(2));

        assert_eq!(sub.as_mut_ptr(), ptr);
        assert_eq!(sub.len().value(), 2);
        let _ = sub.checked_as_array::<2>();
    }
}
