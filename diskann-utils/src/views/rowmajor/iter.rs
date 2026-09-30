/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, num::NonZeroUsize, ptr::NonNull};

use crate::views::rowmajor::{Layout, Matrix, Mut, Ref};

//------//
// Rows //
//------//

// An iterator over rows in a matrix. See: [`Matrix::row_iter`].
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

unsafe impl<T> Send for Rows<'_, T> where T: Sync {}
unsafe impl<T> Sync for Rows<'_, T> where T: Sync {}

impl<'a, T> Iterator for Rows<'a, T> {
    type Item = &'a [T];
    fn next(&mut self) -> Option<&'a [T]> {
        self.remaining.checked_sub(1).map(|remaining| {
            let item =
                unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast_const(), self.ncols) };
            self.remaining = remaining;
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

// An iterator over mutable rows in a matrix. See: [`Matrix::row_iter_mut`].
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

unsafe impl<T> Send for RowsMut<'_, T> where T: Send {}
unsafe impl<T> Sync for RowsMut<'_, T> where T: Sync {}

impl<'a, T> Iterator for RowsMut<'a, T> {
    type Item = &'a mut [T];
    fn next(&mut self) -> Option<&'a mut [T]> {
        self.remaining.checked_sub(1).map(|remaining| {
            let item = unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.ncols) };
            self.remaining = remaining;
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

// An iterator over rows in a matrix. See: [`Matrix::window_iter`].
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

unsafe impl<T> Send for Windows<'_, T> where T: Sync {}
unsafe impl<T> Sync for Windows<'_, T> where T: Sync {}

impl<'a, T> Iterator for Windows<'a, T> {
    type Item = Ref<'a, T>;
    fn next(&mut self) -> Option<Ref<'a, T>> {
        if self.remaining == 0 {
            None
        } else {
            let next_remaining = self.remaining.saturating_sub(self.batchsize.get());
            let nrows = self.remaining - next_remaining;

            let window = unsafe {
                Ref {
                    ptr: self.ptr,
                    layout: Layout::new_unchecked(nrows, self.ncols),
                    _lifetime: PhantomData,
                }
            };

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
