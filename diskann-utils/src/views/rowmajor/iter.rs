/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, num::NonZeroUsize, ptr::NonNull};

use crate::views::rowmajor::{Layout, Matrix, Mut, Ref};

//------//
// Rows //
//------//

/// An iterator over rows in a matrix. See: [`Matrix::row_iter`].
#[derive(Debug)]
pub struct Rows<'a, T> {
    ptr: NonNull<T>,
    remaining: usize,
    ncols: usize,
    _lifetime: PhantomData<&'a [T]>,
}

impl<'a, T> Rows<'a, T> {
    pub(super) fn new(m: Ref<'a, T>) -> Self {
        let layout = m.layout();
        Self {
            ptr: m.as_nonnull(),
            remaining: layout.nrows(),
            ncols: layout.ncols(),
            _lifetime: PhantomData,
        }
    }
}

// SAFETY: `Rows<'_, T>` owns a shared slice borrow, so sending it requires `T: Sync`.
unsafe impl<T> Send for Rows<'_, T> where T: Sync {}
// SAFETY: Shared access to `Rows<'_, T>` exposes only shared access to `T`.
unsafe impl<T> Sync for Rows<'_, T> where T: Sync {}

impl<'a, T> Iterator for Rows<'a, T> {
    type Item = &'a [T];
    fn next(&mut self) -> Option<&'a [T]> {
        self.remaining.checked_sub(1).map(|remaining| {
            // SAFETY: Construction from a valid `Ref` guarantees that each remaining row
            // contains `ncols` initialized elements beginning at `self.ptr`.
            let item =
                unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast_const(), self.ncols) };
            self.remaining = remaining;

            // SAFETY: Advancing by one row remains within or one past the original matrix
            // span. The validated parent layout guarantees that the offset is representable.
            self.ptr = unsafe { self.ptr.add(self.ncols) };
            item
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for Rows<'_, T> {}
impl<T> std::iter::FusedIterator for Rows<'_, T> {}

//---------//
// RowsMut //
//---------//

/// An iterator over mutable rows in a matrix. See: [`Matrix::row_iter_mut`].
#[derive(Debug)]
pub struct RowsMut<'a, T> {
    ptr: NonNull<T>,
    remaining: usize,
    ncols: usize,
    _lifetime: PhantomData<&'a mut [T]>,
}

impl<'a, T> RowsMut<'a, T> {
    pub(super) fn new(m: Mut<'a, T>) -> Self {
        let layout = m.layout();
        Self {
            ptr: m.as_nonnull(),
            remaining: layout.nrows(),
            ncols: layout.ncols(),
            _lifetime: PhantomData,
        }
    }
}

// SAFETY: `RowsMut<'_, T>` owns an exclusive slice borrow, so sending it requires `T: Send`.
unsafe impl<T> Send for RowsMut<'_, T> where T: Send {}
// SAFETY: Shared access to `RowsMut<'_, T>` exposes only shared access to `T`.
unsafe impl<T> Sync for RowsMut<'_, T> where T: Sync {}

impl<'a, T> Iterator for RowsMut<'a, T> {
    type Item = &'a mut [T];
    fn next(&mut self) -> Option<&'a mut [T]> {
        self.remaining.checked_sub(1).map(|remaining| {
            // SAFETY: Construction from a valid `Mut` guarantees that each remaining row
            // contains `ncols` initialized elements beginning at `self.ptr`. Advancing the
            // pointer after every yield makes nonempty returned rows disjoint; zero-length
            // rows do not access memory and may share an address.
            let item = unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.ncols) };
            self.remaining = remaining;

            // SAFETY: Advancing by one row remains within or one past the original matrix
            // span. The validated parent layout guarantees that the offset is representable.
            self.ptr = unsafe { self.ptr.add(self.ncols) };
            item
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for RowsMut<'_, T> {}
impl<T> std::iter::FusedIterator for RowsMut<'_, T> {}

//---------//
// Windows //
//---------//

/// An iterator over rows in a matrix. See: [`Matrix::window_iter`].
#[derive(Debug)]
pub struct Windows<'a, T> {
    ptr: NonNull<T>,
    remaining: usize,
    batchsize: NonZeroUsize,
    ncols: usize,
    _lifetime: PhantomData<&'a [T]>,
}

impl<'a, T> Windows<'a, T> {
    pub(super) fn new(m: Ref<'a, T>, batchsize: NonZeroUsize) -> Self {
        let layout = m.layout();
        Self {
            ptr: m.as_nonnull(),
            remaining: layout.nrows(),
            batchsize,
            ncols: layout.ncols(),
            _lifetime: PhantomData,
        }
    }
}

// SAFETY: `Windows<'_, T>` owns a shared slice borrow, so sending it requires `T: Sync`.
unsafe impl<T> Send for Windows<'_, T> where T: Sync {}
// SAFETY: Shared access to `Windows<'_, T>` exposes only shared access to `T`.
unsafe impl<T> Sync for Windows<'_, T> where T: Sync {}

impl<'a, T> Iterator for Windows<'a, T> {
    type Item = Ref<'a, T>;
    fn next(&mut self) -> Option<Ref<'a, T>> {
        if self.remaining == 0 {
            None
        } else {
            let next_remaining = self.remaining.saturating_sub(self.batchsize.get());
            let nrows = self.remaining - next_remaining;

            // SAFETY: `self.ptr` starts the remaining suffix of a valid `Ref`, and `nrows`
            // does not exceed that suffix. Keeping the parent's column count therefore
            // produces a valid subview and a layout no larger than the parent layout.
            let window = unsafe {
                Ref {
                    ptr: self.ptr,
                    layout: Layout::new_unchecked(nrows, self.ncols),
                    _lifetime: PhantomData,
                }
            };

            // SAFETY: Advancing by the yielded window remains within or one past the
            // original matrix span. The validated parent layout guarantees that the
            // multiplication and pointer offset are representable.
            self.ptr = unsafe { self.ptr.add(nrows * self.ncols) };
            self.remaining = next_remaining;
            Some(window)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining.div_ceil(self.batchsize.get());
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for Windows<'_, T> {}
impl<T> std::iter::FusedIterator for Windows<'_, T> {}
