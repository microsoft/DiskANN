/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, ptr::NonNull};

use super::{bounds::{self, Bound}, num::Elements};

const fn slice_to_nonnull<T>(x: &[T]) -> NonNull<T> {
    unsafe { NonNull::new_unchecked(x.as_ptr().cast::<T>().cast_mut()) }
}

const fn mut_slice_to_nonnull<T>(x: &mut [T]) -> NonNull<T> {
    unsafe { NonNull::new_unchecked(x.as_mut_ptr().cast::<T>()) }
}

//-------//
// Slice //
//-------//

/// An immutable slice with length tracking when debug-assertions are enabled.
#[derive(Debug)]
pub(super) struct Slice<'a, T> {
    ptr: NonNull<T>,
    len: Bound,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T> Slice<'a, T> {
    /// Construct a new [`Slice`] with the same length and base pointer as `slice`.
    pub(super) const fn new(slice: &'a [T]) -> Self {
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

    /// Read a value from `self` without moving it.
    ///
    /// # Safety
    ///
    /// THis function has the same requirements as [`std::ptr::read`].
    pub(super) unsafe fn read(self) -> T {
        bounds::check_ge!(
            self.len,
            1,
            "slices must have a length of at least 1 to read the first element"
        );

        unsafe { self.ptr.read() }
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
        bounds::check_ge!(
            self.len,
            length,
            "truncation would make the slice longer"
        );

        unsafe { Self::from_raw(self.ptr, Bound::new(length)) }
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

//----------//
// MutSlice //
//----------//

/// An mutable slice with length tracking when debug-assertions are enabled.
#[derive(Debug, Clone, Copy)]
pub(super) struct MutSlice<'a, T> {
    ptr: NonNull<T>,
    len: Bound,
    _lifetime: PhantomData<&'a mut T>,
}

impl<'a, T> MutSlice<'a, T> {
    /// Construct a new [`MutSlice`] with the same length and base pointer as `slice`.
    pub(super) const fn new(slice: &'a mut [T]) -> Self {
        unsafe { Self::from_raw(mut_slice_to_nonnull(slice), Bound::new(slice.len())) }
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

    /// Borrow `self` as a [`Slice`].
    pub(super) const fn as_slice(&self) -> Slice<'_, T> {
        unsafe { Slice::from_raw(self.ptr, self.len) }
    }

    /// Return the base pointer for `self`.
    pub(super) const fn as_ptr(&self) -> *const T {
        self.as_mut_ptr().cast_const()
    }

    /// Return the base pointer for `self`.
    pub(super) const fn as_mut_ptr(&self) -> *mut T {
        self.as_nonnull().as_ptr()
    }

    /// Return the base pointer for `self`.
    pub(super) const fn as_nonnull(&self) -> NonNull<T> {
        self.ptr
    }

    pub(super) fn subslice(&mut self, start: usize, length: Bound) -> MutSlice<'_, T> {
        bounds::check_gt!(self.len, start, "start is out-of-bounds");
        bounds::check_ge!(
            self.len,
            Bound::new(start) + length,
            "end is out-of-bounds"
        );

        unsafe { Self::from_raw(self.ptr.add(start), length) }
    }

    pub(super) unsafe fn materialize<const N: usize>(&mut self) -> &mut [T; N] {
        bounds::check_eq!(self.len, N, "invalid materialization of size {N}");
        unsafe { &mut *self.as_mut_ptr().cast::<[T; N]>() }
    }

    pub(super) fn reborrow(&mut self) -> MutSlice<'_, T> {
        unsafe { Self::from_raw(self.ptr, self.len) }
    }

    pub(super) fn len(&self) -> Bound {
        self.len
    }
}
