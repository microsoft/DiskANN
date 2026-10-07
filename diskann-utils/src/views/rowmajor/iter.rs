/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, num::NonZeroUsize, ptr::NonNull};

#[cfg(feature = "rayon")]
use crate::views::rowmajor::MatrixMut;
use crate::views::rowmajor::{Layout, Matrix, Mut, Ref};

//------//
// Rows //
//------//

/// An iterator over rows in a matrix. See: [`Matrix::rows`].
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

/// An iterator over mutable rows in a matrix. See: [`crate::views::rowmajor::MatrixMut::rows_mut`].
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

//--------//
// ParMut //
//--------//

#[cfg(feature = "rayon")]
/// Carries an exclusive matrix borrow across Rayon workers.
pub(super) struct ParMut<'a, T> {
    ptr: NonNull<T>,
    layout: Layout<T>,
    _lifetime: PhantomData<&'a mut [T]>,
}

// SAFETY: `ParMut` owns an exclusive slice borrow, so sending it requires `T: Send`.
#[cfg(feature = "rayon")]
unsafe impl<T: Send> Send for ParMut<'_, T> {}
// SAFETY: The only methods that produce mutable views are unsafe and require callers to
// ensure disjointness. Sending those views between workers requires `T: Send`.
#[cfg(feature = "rayon")]
unsafe impl<T: Send> Sync for ParMut<'_, T> {}

#[cfg(feature = "rayon")]
impl<'a, T> ParMut<'a, T> {
    pub(super) fn new<M>(matrix: &'a mut M) -> Self
    where
        M: MatrixMut<Element = T> + ?Sized,
    {
        let layout = matrix.layout();
        let ptr = matrix.as_nonnull_mut();
        Self {
            ptr,
            layout,
            _lifetime: PhantomData,
        }
    }

    pub(super) fn nrows(&self) -> usize {
        self.layout.nrows()
    }

    /// # Safety
    ///
    /// * `row < self.nrows()`.
    /// * No other live reference derived from this `ParMut` may overlap any element of
    ///   row `row`. Zero-column rows never overlap, even when their addresses match.
    pub(super) unsafe fn row_disjoint_unchecked(&self, row: usize) -> &'a mut [T] {
        debug_assert!(row < self.layout.nrows());
        let ncols = self.layout.ncols();

        // SAFETY: The caller guarantees that `row` is in-bounds and does not overlap any
        // other live view. The validated parent layout makes the offset representable and
        // places the row within the initialized matrix span.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(row * ncols), ncols) }
    }

    /// # Safety
    ///
    /// * `rows.start <= rows.end <= self.nrows()`.
    /// * No other live reference derived from this `ParMut` may overlap any element in
    ///   `rows`. Zero-column windows never overlap.
    pub(super) unsafe fn window_disjoint_unchecked(
        &self,
        rows: std::ops::Range<usize>,
    ) -> Mut<'a, T> {
        debug_assert!(rows.start <= rows.end);
        debug_assert!(rows.end <= self.layout.nrows());

        let ncols = self.layout.ncols();
        let nrows = rows.end - rows.start;

        // SAFETY: The caller guarantees an ordered, in-bounds range. The validated parent
        // layout makes the offset representable and places it within or one past the matrix.
        let ptr = unsafe { self.ptr.add(rows.start * ncols) };

        Mut {
            ptr,
            // SAFETY: This window has no more rows than the validated parent and keeps its
            // column count, so its element count and byte span cannot exceed the parent.
            layout: unsafe { Layout::new_unchecked(nrows, ncols) },
            _lifetime: PhantomData,
        }
    }
}

#[cfg(all(test, feature = "rayon"))]
mod tests {
    use super::ParMut;
    use crate::views::rowmajor::{Matrix, MatrixMut, Owned};

    #[test]
    fn par_mut_zero_column_views_can_coexist() {
        let mut matrix = Owned::from_element(usize::MAX, 0, 0);
        let ptr = matrix.as_ptr();
        let matrix = ParMut::new(&mut matrix);

        // SAFETY: Empty views do not overlap any elements, even when their addresses match.
        let rows = unsafe {
            [
                matrix.row_disjoint_unchecked(0),
                matrix.row_disjoint_unchecked(1),
                matrix.row_disjoint_unchecked(usize::MAX - 1),
            ]
        };
        // SAFETY: Empty windows do not overlap any elements, including the live row views
        // whose logical rows fall within these windows.
        let windows = unsafe {
            [
                matrix.window_disjoint_unchecked(0..2),
                matrix.window_disjoint_unchecked(2..usize::MAX),
            ]
        };

        assert!(rows.iter().all(|row| row.is_empty() && row.as_ptr() == ptr));
        assert!(windows.iter().all(|window| {
            window.ncols() == 0 && window.as_slice().is_empty() && window.as_ptr() == ptr
        }));
    }

    #[test]
    fn par_mut_disjoint_nonempty_views_can_coexist() {
        let mut matrix = Owned::from_fn(6, 2, |rc| rc.row * 100 + rc.col);
        {
            let matrix = ParMut::new(&mut matrix);

            // SAFETY: These rows and windows are in-bounds and pairwise disjoint.
            let (row0, row1, mut window2, mut window4) = unsafe {
                (
                    matrix.row_disjoint_unchecked(0),
                    matrix.row_disjoint_unchecked(1),
                    matrix.window_disjoint_unchecked(2..4),
                    matrix.window_disjoint_unchecked(4..6),
                )
            };
            row0[0] = 10;
            row1[1] = 11;
            *window2.element_mut(0, 0) = 20;
            *window2.element_mut(1, 1) = 31;
            *window4.element_mut(0, 0) = 40;
            *window4.element_mut(1, 1) = 51;
        }

        assert_eq!(
            matrix.as_slice(),
            &[10, 1, 100, 11, 20, 201, 300, 31, 40, 401, 500, 51]
        );
    }
}
