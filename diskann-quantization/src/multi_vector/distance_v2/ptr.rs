/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, ptr::NonNull};

use super::{Check, Length, bounds, num::Elements};

const fn slice_to_nonnull<T>(x: &[T]) -> NonNull<T> {
    unsafe { NonNull::new_unchecked(x.as_ptr().cast::<T>().cast_mut()) }
}

const fn mut_slice_to_nonnull<T>(x: &mut [T]) -> NonNull<T> {
    unsafe { NonNull::new_unchecked(x.as_mut_ptr().cast::<T>()) }
}

//-------//
// Slice //
//-------//

#[derive(Debug)]
pub(super) struct Slice<'a, T> {
    ptr: NonNull<T>,
    length: Length,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T> Slice<'a, T> {
    pub(super) const fn new(slice: &'a [T]) -> Self {
        unsafe { Self::from_raw(slice_to_nonnull(slice), Length::new(slice.len())) }
    }

    pub(super) const fn as_ptr(&self) -> *const T {
        self.as_nonnull().as_ptr().cast_const()
    }

    pub(super) const fn as_nonnull(&self) -> NonNull<T> {
        self.ptr
    }

    pub(super) const unsafe fn from_raw(ptr: NonNull<T>, length: Length) -> Self {
        Self {
            ptr,
            length,
            _lifetime: PhantomData,
        }
    }

    pub(super) unsafe fn read(self) -> T {
        bounds::check_ge!(
            self.length,
            1,
            "slices must have a length of at least 1 to read the first element"
        );

        unsafe { self.ptr.read() }
    }

    pub(super) unsafe fn add(self, offset: Elements<T>) -> Slice<'a, T> {
        let offset = offset.value();

        // Debug check that there is room.
        bounds::check_ge!(
            self.length,
            offset,
            "offset would go out-of-bounds"
        );

        unsafe { Self::from_raw(self.ptr.add(offset), self.length - Length::new(offset)) }
    }

    pub(super) unsafe fn truncate(self, length: Elements<T>) -> Slice<'a, T> {
        let length = length.value();
        bounds::check_ge!(self.length, length, "truncation would make the slice longer");

        unsafe { Self::from_raw(self.ptr, Length::new(length)) }
    }

    pub(super) fn length(&self) -> Length {
        self.length
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

#[derive(Debug, Clone, Copy)]
pub(super) struct MutSlice<'a, T> {
    ptr: NonNull<T>,
    length: Length,
    _lifetime: PhantomData<&'a mut T>,
}

impl<'a, T> MutSlice<'a, T> {
    pub(super) const fn new(slice: &'a mut [T]) -> Self {
        unsafe { Self::from_raw(mut_slice_to_nonnull(slice), Length::new(slice.len())) }
    }

    pub(super) const fn as_slice(&self) -> Slice<'_, T> {
        unsafe { Slice::from_raw(self.ptr, self.length) }
    }

    pub(super) const fn as_ptr(&self) -> *const T {
        self.as_mut_ptr().cast_const()
    }

    pub(super) const fn as_mut_ptr(&self) -> *mut T {
        self.as_nonnull().as_ptr()
    }

    pub(super) const fn as_nonnull(&self) -> NonNull<T> {
        self.ptr
    }

    pub(super) const unsafe fn from_raw(ptr: NonNull<T>, length: Length) -> Self {
        Self {
            ptr,
            length,
            _lifetime: PhantomData,
        }
    }

    pub(super) fn subslice(&mut self, start: usize, length: Length) -> MutSlice<'_, T> {
        bounds::check_gt!(self.length, start, "start is out-of-bounds");
        bounds::check_ge!(self.length, Length::new(start) + length, "end is out-of-bounds");

        unsafe { Self::from_raw(self.ptr.add(start), length) }
    }

    pub(super) fn reborrow(&mut self) -> MutSlice<'_, T> {
        unsafe { Self::from_raw(self.ptr, self.length) }
    }

    pub(super) fn length(&self) -> Length {
        self.length
    }
}
