/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, mem::ManuallyDrop, num::NonZeroUsize, ptr::NonNull};

#[cfg(feature = "rayon")]
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut};
use thiserror::Error;

pub mod iter;

use crate::{internal, Reborrow, ReborrowMut};

////////////
// Matrix //
////////////

/// Tools for treating contiguous regions of memory as a dense, row-major matrix.
///
/// See also: [`MatrixMut`], [`Owned`], [`Ref`], [`Mut`].
///
/// # Safety
///
/// For the duration of each borrow of `self`, implementations must ensure that:
///
/// * Repeated calls to [`Matrix::as_nonnull`] and [`Matrix::layout`] return the same pointer
///   and layout.
///
/// * Given:
///
///   ```text
///   let data = self.as_nonnull();
///   let layout = self.layout();
///   ```
///
///   constructing the following slice is valid:
///
///   ```text
///   unsafe { std::slice::from_raw_parts(self.as_ptr().cast_const(), layout.num_elements()) };
///   ```
///
///   In particular:
///
///   - `data` must be properly aligned and the span `[data, data + layout.num_elements())`
///     must be within a single allocation.
///
///   - `data` must point to `layout.num_elements()` consecutive, properly initialized values
///     of type `Self::Element`.
///
///   - The memory referenced must not be mutated for the duration of the borrow, except
///     inside an `UnsafeCell`.
pub unsafe trait Matrix {
    /// The type of the element stored in the matrix.
    type Element;

    /// Return the base pointer for the matrix.
    fn as_nonnull(&self) -> NonNull<Self::Element>;

    /// Return the [`Layout`] for the matrix.
    fn layout(&self) -> Layout<Self::Element>;

    //----------//
    // Provided //
    //----------//

    /// Return the number of rows in the matrix.
    fn nrows(&self) -> usize {
        self.layout().nrows()
    }

    /// Return the number of columns in the matrix.
    fn ncols(&self) -> usize {
        self.layout().ncols()
    }

    /// Returns the requested row without boundschecking.
    ///
    /// # Safety
    ///
    /// The following conditions must hold to avoid undefined behavior:
    ///
    /// * `row < self.nrows()`.
    unsafe fn row_unchecked(&self, row: usize) -> &[Self::Element] {
        let layout = self.layout();
        debug_assert!(row < layout.nrows());

        // SAFETY: We're allowed to assume the entire span described the the pointer and
        // layout is valid. This extracts a portion of that span.
        //
        // The lifetime is tied to the borrow `self`, so prevents modifications to `self`.
        unsafe {
            std::slice::from_raw_parts(self.as_ptr().add(layout.ncols() * row), layout.ncols())
        }
    }

    /// Return a pointer to the base of the matrix.
    fn as_ptr(&self) -> *const Self::Element {
        self.as_nonnull().as_ptr().cast_const()
    }

    /// Return the underlying data as a slice.
    fn as_slice(&self) -> &[Self::Element] {
        // SAFETY: Required by the implementors of `Matrix`.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.layout().num_elements()) }
    }

    /// Return row `row` as a slice.
    ///
    /// # Panic
    ///
    /// Panics if `row >= self.nrows()`.
    fn row(&self, row: usize) -> &[Self::Element] {
        assert!(
            row < self.nrows(),
            "tried to access row {row} of a matrix with {} rows",
            self.nrows()
        );

        // SAFETY: `row` is in-bounds.
        unsafe { self.row_unchecked(row) }
    }

    /// Return row `row` if `row < self.nrows()`. Otherwise, return `None`.
    fn get_row(&self, row: usize) -> Option<&[Self::Element]> {
        if row < self.nrows() {
            // SAFETY: `row` is in-bounds.
            Some(unsafe { self.row_unchecked(row) })
        } else {
            None
        }
    }

    /// Return a iterator over all rows in the matrix.
    ///
    /// Rows are yielded sequentially beginning with row 0.
    fn rows(&self) -> iter::Rows<'_, Self::Element> {
        iter::Rows::new(self.as_view())
    }

    /// Returns a reference to an element without boundschecking.
    ///
    /// # Safety
    ///
    /// The following conditions must hold to avoid undefined behavior:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    unsafe fn element_unchecked(&self, row: usize, col: usize) -> &Self::Element {
        let layout = self.layout();
        debug_assert!(row < layout.nrows());
        debug_assert!(col < layout.ncols());

        // SAFETY: We're allowed to assume the entire span described the the pointer and
        // layout is valid. This extracts one element of that span.
        //
        // The lifetime is tied to the borrow `self`, so prevents modifications to `self`.
        unsafe { &*self.as_ptr().add(row * layout.ncols() + col) }
    }

    /// Return the value at the specified `row` and `col`.
    ///
    /// If either index is out-of-bounds, return `None`.
    fn get_element(&self, row: usize, col: usize) -> Option<&Self::Element> {
        if row >= self.nrows() || col >= self.ncols() {
            None
        } else {
            // SAFETY: We just verified that `row` and `col` are in-bounds.
            Some(unsafe { self.element_unchecked(row, col) })
        }
    }

    /// Return the value at the specified `row` and `col`.
    ///
    /// # Panics
    ///
    /// Panics if either `row` or `col` is out-of-bounds.
    fn element(&self, row: usize, col: usize) -> &Self::Element {
        assert!(
            row < self.nrows(),
            "row {row} is out of bounds (max: {})",
            self.nrows()
        );
        assert!(
            col < self.ncols(),
            "col {col} is out of bounds (max: {})",
            self.ncols()
        );

        // SAFETY: We just verified that `row` and `col` are in-bounds.
        unsafe { self.element_unchecked(row, col) }
    }

    /// Return a view over the matrix.
    fn as_view(&self) -> Ref<'_, Self::Element> {
        Ref {
            ptr: self.as_nonnull(),
            layout: self.layout(),
            _lifetime: PhantomData,
        }
    }

    /// Return a view over the rows in `rows`, or `None` when the range is invalid.
    fn subview(&self, rows: std::ops::Range<usize>) -> Option<Ref<'_, Self::Element>> {
        if rows.start > rows.end || rows.end > self.nrows() {
            return None;
        }

        let ncols = self.ncols();
        // SAFETY: Both bounds are within the matrix, so the offset is within or one past
        // the allocation. The parent layout guarantees that the multiplication fits.
        let ptr =
            unsafe { NonNull::new_unchecked(self.as_ptr().add(rows.start * ncols).cast_mut()) };
        // SAFETY: The selected row count cannot exceed the validated parent layout.
        let layout = unsafe { Layout::new_unchecked(rows.end - rows.start, ncols) };
        Some(Ref {
            ptr,
            layout,
            _lifetime: PhantomData,
        })
    }

    /// Return an iterator that divides the matrix into sub-matrices with (up to)
    /// `batchsize` rows with `self.ncols()` columns.
    ///
    /// It is possible for yielded sub-matrices to have fewer than `batchsize` rows if the
    /// number of rows in the parent matrix is not evenly divisible by `batchsize`.
    fn window_iter(&self, batchsize: NonZeroUsize) -> iter::Windows<'_, Self::Element> {
        iter::Windows::new(self.as_view(), batchsize)
    }

    /// Return an [`Owned`] with the same shape as `self` and cloned contents.
    fn to_rowmajor_owned(&self) -> Owned<Self::Element>
    where
        Self::Element: Clone,
    {
        // Safety: `self.layout()` is already validated and by the trait requirements,
        // `self.as_slice()` is required to be exactly `self.layout().len()`.
        unsafe { Owned::from_data_unchecked(self.as_slice().into(), self.layout()) }
    }

    /// Create a new [`Matrix`] by applying the closure `f` to each element.
    ///
    /// The returned matrix has the same shape as `self`.
    fn try_map<F, R>(&self, f: F) -> Result<Owned<R>, LayoutError>
    where
        F: FnMut(&Self::Element) -> R,
    {
        let layout = self.layout().rebind::<R>()?;
        let data: Box<[_]> = self.as_slice().iter().map(f).collect();

        // SAFETY: The trait requirements require `self.as_slice().len()` to be equal
        // to `self.layout.len()` and `layout` haws been validated for the destination
        // type.
        Ok(unsafe { Owned::from_data_unchecked(data, layout) })
    }

    /// Create a new [`Matrix`] by applying the closure `f` to each element.
    ///
    /// The returned matrix has the same shape as `self`.
    ///
    /// # Panics
    ///
    /// Panics if allocating space for [`Owned`] would overflow `isize::MAX`.
    #[track_caller]
    fn map<F, R>(&self, f: F) -> Owned<R>
    where
        F: FnMut(&Self::Element) -> R,
    {
        match self.try_map(f) {
            Ok(owned) => owned,
            Err(error) => panic!("`Matrix::map` failed: {error}"),
        }
    }

    /// Transpose the elements in `self`.
    fn transpose(&self) -> Owned<Self::Element>
    where
        Self::Element: Clone,
    {
        Owned::from_fn_with_layout(self.layout().transpose(), |RowCol { row, col }| {
            // SAFETY: By contruction, `col < self.nrows()` and `row < self.ncols()`.
            unsafe { self.element_unchecked(col, row).clone() }
        })
    }

    //-------//
    // Rayon //
    //-------//

    /// Return a parallel iterator over the rows of the matrix.
    #[cfg(feature = "rayon")]
    fn par_row_iter(&self) -> impl IndexedParallelIterator<Item = &[Self::Element]>
    where
        Self::Element: Sync,
    {
        self.as_slice().par_chunks_exact(self.ncols())
    }

    /// Return a parallel iterator that divides the matrix into sub-matrices with (up to)
    /// `batchsize` rows with `self.ncols()` columns.
    ///
    /// This allows workers in parallel algorithms to work on dense subsets of the whole
    /// matrix for better locality.
    ///
    /// It is possible for yielded sub-matrices to have fewer than `batchsize` rows if the
    /// number of rows in the parent matrix is not evenly divisible by `batchsize`.
    ///
    /// # Panics
    ///
    /// Panics if `batchsize = 0`.
    #[cfg(feature = "rayon")]
    fn par_window_iter(
        &self,
        batchsize: usize,
    ) -> impl IndexedParallelIterator<Item = Ref<'_, Self::Element>>
    where
        Self::Element: Sync,
    {
        assert!(batchsize != 0, "par_window_iter batchsize cannot be zero");
        let ncols = self.ncols();
        self.as_slice()
            .par_chunks(ncols * batchsize)
            .map(move |data| {
                let blobsize = data.len();
                let nrows = blobsize / ncols;
                assert_eq!(blobsize % ncols, 0);

                unsafe { Ref::from_data_unchecked(data, Layout::new_unchecked(nrows, ncols)) }
            })
    }
}

///////////////
// MatrixMut //
///////////////

/// Tools for treating contiguous regions of mutable memory as a dense, row-major matrix.
///
/// See also: [`Owned`], [`Ref`], [`Mut`].
///
/// # Safety
///
/// In addition to the requirements of [`Matrix`], implementations must ensure that for
/// the duration of each **mutable** borrow of `self`, the entire span described by
/// [`Matrix::as_nonnull`] and [`Matrix::layout`] may be accessed exclusively through a
/// mutable reference.
pub unsafe trait MatrixMut: Matrix {
    //----------//
    // Provided //
    //----------//

    /// Returns the requested row without boundschecking.
    ///
    /// # Safety
    ///
    /// The following conditions must hold to avoid undefined behavior:
    ///
    /// * `row < self.nrows()`.
    unsafe fn row_unchecked_mut(&mut self, row: usize) -> &mut [Self::Element] {
        let layout = self.layout();

        debug_assert!(row < layout.nrows());
        unsafe {
            std::slice::from_raw_parts_mut(
                self.as_mut_ptr().add(layout.ncols() * row),
                layout.ncols(),
            )
        }
    }

    /// Return a pointer to the base of the matrix.
    fn as_mut_ptr(&mut self) -> *mut Self::Element {
        self.as_nonnull().as_ptr()
    }

    /// Return the underlying data as a mutable slice.
    fn as_mut_slice(&mut self) -> &mut [Self::Element] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.layout().num_elements()) }
    }

    /// Return row `row` as a mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.nrows()`.
    fn row_mut(&mut self, row: usize) -> &mut [Self::Element] {
        assert!(
            row < self.nrows(),
            "tried to access row {row} of a matrix with {} rows",
            self.nrows()
        );

        // SAFETY: `row` is in-bounds.
        unsafe { self.row_unchecked_mut(row) }
    }

    /// Return row `row` if `row < self.nrows()`. Otherwise, return `None`.
    fn get_row_mut(&mut self, row: usize) -> Option<&mut [Self::Element]> {
        if row < self.nrows() {
            // SAFETY: `row` is in-bounds.
            Some(unsafe { self.row_unchecked_mut(row) })
        } else {
            None
        }
    }

    /// Return a mutable iterator over all rows in the matrix.
    ///
    /// Rows are yielded sequentially beginning with row 0.
    fn rows_mut(&mut self) -> iter::RowsMut<'_, Self::Element> {
        iter::RowsMut::new(self.as_view_mut())
    }

    /// Returns a mutable reference to an element without boundschecking.
    ///
    /// # Safety
    ///
    /// The following conditions must hold to avoid undefined behavior:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    unsafe fn element_unchecked_mut(&mut self, row: usize, col: usize) -> &mut Self::Element {
        let layout = self.layout();
        debug_assert!(row < layout.nrows());
        debug_assert!(col < layout.ncols());

        unsafe { &mut *self.as_mut_ptr().add(row * layout.ncols() + col) }
    }

    /// Return the value at the specified `row` and `col`.
    ///
    /// If either index is out-of-bounds, return `None`.
    fn get_element_mut(&mut self, row: usize, col: usize) -> Option<&mut Self::Element> {
        if row >= self.nrows() || col >= self.ncols() {
            None
        } else {
            // SAFETY: We just verified that `row` and `col` are in-bounds.
            Some(unsafe { self.element_unchecked_mut(row, col) })
        }
    }

    /// Return the value at the specified `row` and `col`.
    ///
    /// # Panics
    ///
    /// Panics if either `row` or `col` is out-of-bounds.
    fn element_mut(&mut self, row: usize, col: usize) -> &mut Self::Element {
        assert!(
            row < self.nrows(),
            "row {row} is out of bounds (max: {})",
            self.nrows()
        );
        assert!(
            col < self.ncols(),
            "col {col} is out of bounds (max: {})",
            self.ncols()
        );

        // SAFETY: We just verified that `row` and `col` are in-bounds.
        unsafe { self.element_unchecked_mut(row, col) }
    }

    /// Return a view over the matrix.
    fn as_view_mut(&mut self) -> Mut<'_, Self::Element> {
        Mut {
            ptr: self.as_nonnull(),
            layout: self.layout(),
            _lifetime: PhantomData,
        }
    }

    //-------//
    // Rayon //
    //-------//

    /// Return a parallel iterator over the rows of the matrix.
    #[cfg(feature = "rayon")]
    fn par_row_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [Self::Element]>
    where
        Self::Element: Send,
    {
        let ncols = self.ncols();
        self.as_mut_slice().par_chunks_exact_mut(ncols)
    }

    /// Return a parallel iterator that divides the matrix into mutable sub-matrices with
    /// (up to) `batchsize` rows with `self.ncols()` columns.
    ///
    /// This allows workers in parallel algorithms to work on dense subsets of the whole
    /// matrix for better locality.
    ///
    /// It is possible for yielded sub-matrices to have fewer than `batchsize` rows if the
    /// number of rows in the parent matrix is not evenly divisible by `batchsize`.
    ///
    /// # Panics
    ///
    /// Panics if `batchsize = 0`.
    #[cfg(feature = "rayon")]
    fn par_window_iter_mut(
        &mut self,
        batchsize: usize,
    ) -> impl IndexedParallelIterator<Item = Mut<'_, Self::Element>>
    where
        Self::Element: Send,
    {
        assert!(
            batchsize != 0,
            "par_window_iter_mut batchsize cannot be zero"
        );
        let ncols = self.ncols();
        self.as_mut_slice()
            .par_chunks_mut(ncols * batchsize)
            .map(move |data| {
                let blobsize = data.len();
                let nrows = blobsize / ncols;
                assert_eq!(blobsize % ncols, 0);

                unsafe { Mut::from_data_unchecked(data, Layout::new_unchecked(nrows, ncols)) }
            })
    }
}

///////////////////
// Matrix Layout //
///////////////////

/// A validated layout for [`MatrixBase`].
///
/// This type guarantees the following invariants:
///
/// * `self.nrows() * self.ncols()` does not exceed `usize::MAX`.
/// * `self.nrows() * self.ncols() * std::mem::size_of::<T>()` does not exceed `isize::MAX`.
pub struct Layout<T> {
    nrows: usize,
    ncols: usize,
    _type: PhantomData<fn() -> T>,
}

impl<T> Layout<T> {
    /// Construct a new [`Layout`], validating the following:
    ///
    /// * `nrows * ncols` does not exceed `usize::MAX`.
    /// * `nrows * ncols * std::mem::size_of::<T>()` does not exceed `isize::MAX` (the maximum
    ///   addressable byte span).
    pub const fn new(nrows: usize, ncols: usize) -> Result<Self, LayoutError> {
        match LayoutError::check::<T>(nrows, ncols) {
            Ok(()) => Ok(Self {
                nrows,
                ncols,
                _type: PhantomData,
            }),
            Err(err) => Err(err),
        }
    }

    /// Construct a layout without validating its dimensions.
    ///
    /// # Safety
    ///
    /// `LayoutError::check::<T>(nrows, ncols)` must succeed.
    unsafe fn new_unchecked(nrows: usize, ncols: usize) -> Self {
        debug_assert!(LayoutError::check::<T>(nrows, ncols).is_ok());
        Self {
            nrows,
            ncols,
            _type: PhantomData,
        }
    }

    /// Return the product `self.nrows() * self.ncols()`.
    pub fn num_elements(&self) -> usize {
        self.nrows() * self.ncols()
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Return the number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Rebind the element type to `U`.
    ///
    /// # Errors
    ///
    /// Returns an error if the rebound layout's byte size would exceed `isize::MAX`.
    pub fn rebind<U>(&self) -> Result<Layout<U>, LayoutError> {
        if std::mem::size_of::<U>() <= std::mem::size_of::<T>() {
            // This branch is mainly to communicate to the compiler situations where an
            // erroring branch can be avoided.
            //
            // SAFETY: `self` already has a validated layout. Since we know
            // `self.nrows() * self.ncols()` cannot overflow, the only danger is allocation
            // overflow. If we are staying or decreasing size, no need to revalidate.
            Ok(unsafe { Layout::new_unchecked(self.nrows(), self.ncols()) })
        } else {
            Layout::new(self.nrows(), self.ncols())
        }
    }

    /// Swap the rows and columns.
    pub fn transpose(&self) -> Layout<T> {
        // SAFETY: We've already validated the relationship between `self.nrows` and
        // `self.ncols`. Since multiplication is commutative, swapping rows and cols does
        // not invalidate the relationship.
        unsafe { Layout::new_unchecked(self.ncols, self.nrows) }
    }
}

impl<T> Clone for Layout<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Layout<T> {}

impl<T> std::fmt::Debug for Layout<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layout")
            .field("nrows", &self.nrows)
            .field("ncols", &self.ncols)
            .field("elsize", &std::mem::size_of::<T>())
            .finish()
    }
}

impl<T> PartialEq for Layout<T> {
    fn eq(&self, other: &Self) -> bool {
        self.nrows == other.nrows && self.ncols == other.ncols
    }
}

impl<T> Eq for Layout<T> {}

/// Errors in the invariants guaranteed by [`Layout`].
#[derive(Debug, Clone, Copy)]
pub struct LayoutError {
    nrows: usize,
    ncols: usize,
    elsize: Option<NonZeroUsize>,
}

impl LayoutError {
    pub(crate) const fn check<T>(nrows: usize, ncols: usize) -> Result<(), Self> {
        // Guard the element count itself so that `num_elements()` can never overflow.
        let elsize = std::mem::size_of::<T>();
        let num_elements = match nrows.checked_mul(ncols) {
            Some(num_elements) => num_elements,
            None => {
                return Err(Self {
                    nrows,
                    ncols,
                    elsize: None,
                })
            }
        };

        if let Some(len) = num_elements.checked_mul(std::mem::size_of::<T>()) {
            if len <= isize::MAX as usize {
                return Ok(());
            }
        }

        Err(Self {
            nrows,
            ncols,
            elsize: NonZeroUsize::new(elsize),
        })
    }
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.elsize {
            Some(elsize) => {
                write!(
                    f,
                    "a matrix of size {}x{} with elements of size {} exceeds `isize::MAX` bytes",
                    self.nrows, self.ncols, elsize
                )
            }
            None => {
                write!(
                    f,
                    "a matrix of size {}x{} has a length exceeding `usize::MAX`",
                    self.nrows, self.ncols,
                )
            }
        }
    }
}

impl std::error::Error for LayoutError {}

//---------------//
// Helper Macros //
//---------------//

macro_rules! constructors {
    ($element:ident, $data:ty) => {
        /// Try to construct directly from `data`.
        ///
        /// Returns an error if [`Layout::new`] fails for `nrows` and `ncols` or `data.len()`
        /// is not equal to `nrows * ncols`.
        pub fn try_from_data(
            data: $data,
            nrows: usize,
            ncols: usize,
        ) -> Result<Self, TryFromError<$data>> {
            let layout = match Layout::<$element>::new(nrows, ncols) {
                Ok(layout) => layout,
                Err(err) => return Err(TryFromError::layout(data, err)),
            };

            let len = data.len();
            if len == layout.num_elements() {
                Ok(unsafe { Self::from_data_unchecked(data, layout) })
            } else {
                Err(TryFromError::mismatch(
                    data,
                    layout.nrows(),
                    layout.ncols(),
                    len,
                ))
            }
        }

        /// Construct a row vector directly from `data`.
        pub fn row_vector(data: $data) -> Self {
            let layout = unsafe { Layout::new_unchecked(1, data.len()) };
            unsafe { Self::from_data_unchecked(data, layout) }
        }

        /// Construct a column vector directly from `data`.
        pub fn column_vector(data: $data) -> Self {
            let layout = unsafe { Layout::new_unchecked(data.len(), 1) };
            unsafe { Self::from_data_unchecked(data, layout) }
        }
    };
}

///////////
// Owned //
///////////

/// An initializer argument for the closure provided to [`Owned::from_fn`] and
/// [`Owned::try_from_fn`] to remove ambiguity of the row and column being initialiazed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowCol {
    pub row: usize,
    pub col: usize,
}

/// A [`Matrix`]/[`MatrixMut`] that owns its data.
#[derive(Debug)]
pub struct Owned<T> {
    ptr: NonNull<T>,
    layout: Layout<T>,
}

impl<T> Owned<T> {
    constructors!(T, Box<[T]>);

    // NOTE: For constructors, keep `from_fn` and `from_element` first.
    //
    // Rust suggests methods in their declaration order, so this keeps the most common
    // methods as top suggestions.

    /// Construct a new matrix using `init`.
    ///
    /// Elements are initialized in memory order.
    ///
    /// ```
    /// use diskann_utils::views::rowmajor::{self, Matrix};
    ///
    /// let mut i = 0;
    /// let mat = rowmajor::Owned::from_fn(2, 3, |_| {
    ///     let value = i;
    ///     i += 1;
    ///     value
    /// });
    ///
    /// assert_eq!(mat.row(0), &[0, 1, 2]);
    /// assert_eq!(mat.row(1), &[3, 4, 5]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `nrows * ncols` overflows `usize::MAX`, or if the allocation size exceeds
    /// `isize::MAX`.
    #[track_caller]
    pub fn from_fn<F>(nrows: usize, ncols: usize, init: F) -> Self
    where
        F: FnMut(RowCol) -> T,
    {
        match Self::try_from_fn(nrows, ncols, init) {
            Ok(matrix) => matrix,
            Err(error) => panic!("Owned::from_fn failed with: {error}"),
        }
    }

    /// Construct a new matrix using `init`.
    ///
    /// Elements are initialized in memory order.
    ///
    /// ```
    /// use diskann_utils::views::rowmajor::{self, Matrix};
    ///
    /// let mut i = 0;
    /// let mat = rowmajor::Owned::try_from_fn(2, 3, |_| {
    ///     let value = i;
    ///     i += 1;
    ///     value
    /// }).unwrap();
    ///
    /// assert_eq!(mat.row(0), &[0, 1, 2]);
    /// assert_eq!(mat.row(1), &[3, 4, 5]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if `nrows * ncols` overflows `usize::MAX`, or if the allocation size
    /// exceeds `isize::MAX`.
    pub fn try_from_fn<F>(nrows: usize, ncols: usize, init: F) -> Result<Self, LayoutError>
    where
        F: FnMut(RowCol) -> T,
    {
        let layout = Layout::new(nrows, ncols)?;
        Ok(Self::from_fn_with_layout(layout, init))
    }

    /// Construct a new matrix by cloning `element`.
    ///
    /// Elements are initialized in memory order.
    ///
    /// ```
    /// use diskann_utils::views::rowmajor::{self, Matrix};
    ///
    /// let mat = rowmajor::Owned::from_element(2, 3, 0u32);
    ///
    /// assert_eq!(mat.row(0), &[0, 0, 0]);
    /// assert_eq!(mat.row(1), &[0, 0, 0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `nrows * ncols` overflows `usize::MAX`, or if the allocation size exceeds
    /// `isize::MAX`.
    #[track_caller]
    pub fn from_element(nrows: usize, ncols: usize, element: T) -> Self
    where
        T: Clone,
    {
        match Self::try_from_element(nrows, ncols, element) {
            Ok(matrix) => matrix,
            Err(error) => panic!("Owned::from_element failed with: {error}"),
        }
    }

    /// Construct a new matrix by cloning `element`.
    ///
    /// Elements are initialized in memory order.
    ///
    /// ```
    /// use diskann_utils::views::rowmajor::{self, Matrix};
    ///
    /// let mat = rowmajor::Owned::try_from_element(2, 3, 0u32).unwrap();
    ///
    /// assert_eq!(mat.row(0), &[0, 0, 0]);
    /// assert_eq!(mat.row(1), &[0, 0, 0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if `nrows * ncols` overflows `usize::MAX`, or if the allocation size
    /// exceeds `isize::MAX`.
    pub fn try_from_element(nrows: usize, ncols: usize, element: T) -> Result<Self, LayoutError>
    where
        T: Clone,
    {
        let layout = Layout::new(nrows, ncols)?;
        Ok(Self::from_element_with_layout(layout, element))
    }

    // Less common constructors.

    /// Construct a new matrix using `init`.
    ///
    /// Elements are initialized in memory order.
    pub fn from_fn_with_layout<F>(layout: Layout<T>, mut init: F) -> Self
    where
        F: FnMut(RowCol) -> T,
    {
        let mut row = 0;
        let mut col = 0;

        let data: Box<[T]> = (0..layout.num_elements())
            .map(|_| {
                let v = (init)(RowCol { row, col });
                col += 1;
                if col == layout.ncols() {
                    col = 0;
                    row += 1;
                }
                v
            })
            .collect();

        unsafe { Self::from_data_unchecked(data, layout) }
    }

    /// Construct a new matrix by cloning `element`.
    ///
    /// Elements are initialized in memory order.
    pub fn from_element_with_layout(layout: Layout<T>, element: T) -> Self
    where
        T: Clone,
    {
        let data: Box<[T]> = std::iter::repeat_n(element, layout.num_elements()).collect();
        unsafe { Self::from_data_unchecked(data, layout) }
    }

    /// # Safety
    ///
    /// `b.len()` must equal `layout.num_elements()`.
    unsafe fn from_data_unchecked(b: Box<[T]>, layout: Layout<T>) -> Self {
        debug_assert_eq!(b.len(), layout.num_elements());
        Self {
            ptr: internal::box_to_nonnull(b),
            layout,
        }
    }

    /// Consume `self`, returning the unmodified contents as a boxed slice.
    ///
    /// ```
    /// use diskann_utils::views::rowmajor::{Matrix, Owned};
    ///
    /// let mat = Owned::from_fn(2, 3, |rc| rc.col);
    /// assert_eq!(mat.row(0), &[0, 1, 2]);
    /// assert_eq!(mat.row(1), &[0, 1, 2]);
    ///
    /// let b: Box<[usize]> = mat.into_inner();
    /// assert_eq!(&*b, &[0, 1, 2, 0, 1, 2]);
    /// ```
    pub fn into_inner(self) -> Box<[T]> {
        let me = ManuallyDrop::new(self);
        unsafe { internal::nonnull_to_box(me.ptr, me.layout.num_elements()) }
    }
}

unsafe impl<T> Send for Owned<T> where T: Send {}
unsafe impl<T> Sync for Owned<T> where T: Sync {}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        let _ = unsafe { internal::nonnull_to_box(self.ptr, self.layout.num_elements()) };
    }
}

impl<T> Clone for Owned<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            ptr: unsafe {
                NonNull::new_unchecked(Box::<[T]>::into_raw(self.as_slice().into()).cast())
            },
            layout: self.layout,
        }
    }
}

unsafe impl<T> Matrix for Owned<T> {
    type Element = T;

    fn as_nonnull(&self) -> NonNull<T> {
        self.ptr
    }

    fn layout(&self) -> Layout<T> {
        self.layout
    }
}

unsafe impl<T> MatrixMut for Owned<T> {}

impl<T> PartialEq for Owned<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        Matrix::as_view(self).eq(&Matrix::as_view(other))
    }
}

impl<'a, T> Reborrow<'a> for Owned<T> {
    type Target = Ref<'a, T>;
    fn reborrow(&'a self) -> Self::Target {
        Matrix::as_view(self)
    }
}

impl<'a, T> ReborrowMut<'a> for Owned<T> {
    type Target = Mut<'a, T>;
    fn reborrow_mut(&'a mut self) -> Self::Target {
        MatrixMut::as_view_mut(self)
    }
}

//-----//
// Ref //
//-----//

/// A [`Matrix`] implementation that references its data.
#[derive(Debug)]
pub struct Ref<'a, T> {
    ptr: NonNull<T>,
    layout: Layout<T>,
    _lifetime: PhantomData<&'a [T]>,
}

unsafe impl<T> Send for Ref<'_, T> where T: Sync {}
unsafe impl<T> Sync for Ref<'_, T> where T: Sync {}

impl<'a, T> Ref<'a, T> {
    constructors!(T, &'a [T]);

    /// # Safety
    ///
    /// `b.len()` must equal `layout.num_elements()`.
    unsafe fn from_data_unchecked(b: &'a [T], layout: Layout<T>) -> Self {
        debug_assert_eq!(b.len(), layout.num_elements());
        Self {
            ptr: internal::slice_to_nonnull(b),
            layout,
            _lifetime: PhantomData,
        }
    }

    /// Return the contents of `self` as a slice.
    ///
    /// Unlike [`Matrix::as_slice`], the returned slices inherits the lifetime of the [`Ref`].
    pub fn into_slice(self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.layout().num_elements()) }
    }
}

impl<T> Clone for Ref<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Ref<'_, T> {}

unsafe impl<T> Matrix for Ref<'_, T> {
    type Element = T;

    fn as_nonnull(&self) -> NonNull<T> {
        self.ptr
    }

    fn layout(&self) -> Layout<T> {
        self.layout
    }
}

impl<'a, T> Reborrow<'a> for Ref<'_, T> {
    type Target = Ref<'a, T>;
    fn reborrow(&'a self) -> Self::Target {
        Matrix::as_view(self)
    }
}

impl<T> PartialEq for Ref<'_, T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.layout() == other.layout() && self.as_slice() == other.as_slice()
    }
}

//-----//
// Mut //
//-----//

/// A [`Matrix`]/[`MatrixMut`] implementation that mutably references its data.
#[derive(Debug)]
pub struct Mut<'a, T> {
    ptr: NonNull<T>,
    layout: Layout<T>,
    _lifetime: PhantomData<&'a mut [T]>,
}

unsafe impl<T> Send for Mut<'_, T> where T: Send {}
unsafe impl<T> Sync for Mut<'_, T> where T: Sync {}

impl<'a, T> Mut<'a, T> {
    constructors!(T, &'a mut [T]);

    /// # Safety
    ///
    /// `b.len()` must equal `layout.num_elements()`.
    unsafe fn from_data_unchecked(b: &'a mut [T], layout: Layout<T>) -> Self {
        debug_assert_eq!(b.len(), layout.num_elements());
        Self {
            ptr: internal::mut_slice_to_nonnull(b),
            layout,
            _lifetime: PhantomData,
        }
    }

    /// Consume `self` and return the underlying data as a mutable slice.
    pub fn into_mut_slice(self) -> &'a mut [T] {
        // SAFETY: `self.ptr` and `self.layout` together describe a valid `&'a mut [T]` of
        // length `self.layout.num_elements()`, per the invariants of `Mut`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.num_elements()) }
    }
}

unsafe impl<T> Matrix for Mut<'_, T> {
    type Element = T;

    fn as_nonnull(&self) -> NonNull<T> {
        self.ptr
    }

    fn layout(&self) -> Layout<T> {
        self.layout
    }
}

unsafe impl<T> MatrixMut for Mut<'_, T> {}

impl<T> PartialEq for Mut<'_, T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        Matrix::as_view(self).eq(&Matrix::as_view(other))
    }
}

impl<'a, T> Reborrow<'a> for Mut<'_, T> {
    type Target = Ref<'a, T>;
    fn reborrow(&'a self) -> Self::Target {
        Matrix::as_view(self)
    }
}

impl<'a, T> ReborrowMut<'a> for Mut<'_, T> {
    type Target = Mut<'a, T>;
    fn reborrow_mut(&'a mut self) -> Self::Target {
        MatrixMut::as_view_mut(self)
    }
}

//--------//
// Errors //
//--------//

/// Errors from [`Owned::try_from_data`], [`Ref::try_from_data`], and [`Mut::try_from_data`].
pub struct TryFromError<T> {
    data: T,
    inner: TryFromErrorInner,
}

impl<T> TryFromError<T> {
    /// Consume the error and return the base data.
    pub fn into_inner(self) -> T {
        self.data
    }

    /// Return a variation of `Self` that is guaranteed to be `'static` by removing the
    /// data that was passed to the original constructor.
    pub fn as_static(&self) -> TryFromErrorLight {
        TryFromErrorLight(self.inner)
    }

    //--------------//
    // Constructors //
    //--------------//

    fn layout(data: T, error: LayoutError) -> Self {
        Self {
            data,
            inner: TryFromErrorInner::Layout(error),
        }
    }

    fn mismatch(data: T, nrows: usize, ncols: usize, len: usize) -> Self {
        Self {
            data,
            inner: TryFromErrorInner::Mismatch { nrows, ncols, len },
        }
    }
}

impl<T> std::fmt::Debug for TryFromError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TryFromError")
            .field("data", &"<hidden>")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<T> std::fmt::Display for TryFromError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

impl<T> std::error::Error for TryFromError<T> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.inner {
            TryFromErrorInner::Layout(error) => Some(error),
            TryFromErrorInner::Mismatch { .. } => None,
        }
    }
}

/// A guaranteed `'static` version of [`TryFromError`].
#[derive(Debug, Error)]
#[error(transparent)]
pub struct TryFromErrorLight(TryFromErrorInner);

#[derive(Debug, Error, Clone, Copy)]
enum TryFromErrorInner {
    #[error(transparent)]
    Layout(LayoutError),
    #[error(
        "tried to construct a {}x{} matrix over a span of length {}",
        nrows,
        ncols,
        len
    )]
    Mismatch {
        nrows: usize,
        ncols: usize,
        len: usize,
    },
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_contains, lazy_format};

    /// This function is only callable with copyable types.
    ///
    /// This lets us test for types we expect to be `Copy`.
    fn is_copyable<T: Copy>(_x: T) -> bool {
        true
    }

    /// This function attests that `Ref` is covariant in the view lifetime.
    fn _matrix_view_is_covariant<'a, 'b>(m: Ref<'a, f32>) -> Ref<'b, f32>
    where
        'a: 'b,
    {
        m
    }

    fn _matrix_view_is_covariant_in_t<'a, 'b, 'm>(m: Ref<'m, &'a f32>) -> Ref<'m, &'b f32>
    where
        'a: 'b,
    {
        m
    }

    fn _matrix_is_covariant_in_t<'a, 'b, 'm>(m: &'m Owned<&'a f32>) -> &'m Owned<&'b f32>
    where
        'a: 'b,
    {
        m
    }

    //--------//
    // Layout //
    //--------//

    #[test]
    fn test_layout() {
        // Happy path
        for rows in 0..5 {
            for cols in 0..5 {
                let layout = Layout::<String>::new(rows, cols).unwrap();
                assert_eq!(layout.nrows(), rows);
                assert_eq!(layout.ncols(), cols);
                assert_eq!(layout.num_elements(), rows * cols);

                let transpose = layout.transpose();
                assert_eq!(transpose.nrows(), cols);
                assert_eq!(transpose.ncols(), rows);
                assert_eq!(transpose.num_elements(), rows * cols);

                let rebind = layout.rebind::<u32>().unwrap();
                assert_eq!(rebind.nrows(), rows);
                assert_eq!(rebind.ncols(), cols);
                assert_eq!(rebind.num_elements(), rows * cols);

                is_copyable(layout);
            }
        }

        #[expect(unused, reason = "we need this so the size is non-zero")]
        struct NotDebugOrEq(u32);

        assert_eq!(
            Layout::<NotDebugOrEq>::new(10, 20).unwrap(),
            Layout::<NotDebugOrEq>::new(10, 20).unwrap(),
        );

        assert_eq!(
            Layout::<NotDebugOrEq>::new(20, 0).unwrap(),
            Layout::<NotDebugOrEq>::new(20, 0).unwrap(),
        );

        assert_ne!(
            Layout::<NotDebugOrEq>::new(10, 20).unwrap(),
            Layout::<NotDebugOrEq>::new(20, 0).unwrap(),
        );

        let fmt = format!("{:?}", Layout::<NotDebugOrEq>::new(5, 6).unwrap());
        assert_eq!(fmt, "Layout { nrows: 5, ncols: 6, elsize: 4 }");

        // Overflowing the element count returns an error.
        let error = Layout::<u8>::new(usize::MAX, 2).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "a matrix of size {}x2 has a length exceeding `usize::MAX`",
                usize::MAX
            )
        );

        // The largest possible byte span is valid without allocating it.
        let layout = Layout::<u8>::new(isize::MAX as usize, 1).unwrap();
        assert_eq!(layout.num_elements(), isize::MAX as usize);

        let transpose = layout.transpose();
        assert_eq!(transpose.nrows(), 1);
        assert_eq!(transpose.ncols(), isize::MAX as usize);
        assert_eq!(transpose.num_elements(), layout.num_elements());

        // One byte beyond the maximum span returns an error.
        let error = Layout::<u8>::new(isize::MAX as usize + 1, 1).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "a matrix of size {}x1 with elements of size 1 exceeds `isize::MAX` bytes",
                isize::MAX as usize + 1
            )
        );

        // Rebinding to a larger element type revalidates the byte span.
        let rebound = Layout::<u8>::new(3, 4).unwrap().rebind::<u16>().unwrap();
        assert_eq!(rebound.nrows(), 3);
        assert_eq!(rebound.ncols(), 4);
        assert_eq!(rebound.num_elements(), 12);

        let error = layout.rebind::<u16>().unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "a matrix of size {}x1 with elements of size 2 exceeds `isize::MAX` bytes",
                isize::MAX
            )
        );
    }

    /////////////////
    // Matrix View //
    /////////////////

    #[test]
    fn fallible_matrix_constructors() {
        let err = Owned::try_from_element(usize::MAX, usize::MAX, 0u32).unwrap_err();
        let msg = err.to_string();
        assert_contains!(msg, "exceeding `usize::MAX`");

        let err = Owned::try_from_element(isize::MAX as usize, 1, 0u32).unwrap_err();
        let msg = err.to_string();
        assert_contains!(msg, "exceeds `isize::MAX` bytes");

        // Panicking
        let err = std::panic::catch_unwind(|| {
            Owned::from_element(usize::MAX, usize::MAX, 0u32);
        })
        .unwrap_err()
        .downcast::<String>()
        .unwrap();

        let msg = err.to_string();
        assert_contains!(msg, "exceeding `usize::MAX`");

        let err = std::panic::catch_unwind(|| {
            Owned::from_element(isize::MAX as usize, 1, 0u32);
        })
        .unwrap_err()
        .downcast::<String>()
        .unwrap();
        let msg = err.to_string();
        assert_contains!(msg, "exceeds `isize::MAX` bytes");

        // Construction fails without invoking the generator.
        let err = Owned::try_from_fn(usize::MAX, usize::MAX, |_| panic!("boom")).unwrap_err();
        let msg = err.to_string();
        assert_contains!(msg, "exceeding `usize::MAX`");
    }

    fn make_test_matrix() -> Vec<usize> {
        // Construct a matrix with 4 rows of length 3.
        // The expected layout is as follows:
        //
        // 0, 1, 2,
        // 1, 2, 3,
        // 2, 3, 4,
        // 3, 4, 5
        //
        vec![0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5]
    }

    #[cfg(feature = "rayon")]
    fn test_basic_indexing_parallel(m: Ref<'_, usize>) {
        // Par window iters.
        let batchsize = 2;
        m.par_window_iter(batchsize)
            .enumerate()
            .for_each(|(i, submatrix)| {
                assert_eq!(submatrix.nrows(), batchsize);
                assert_eq!(submatrix.ncols(), m.ncols());

                // Make sure we are in the correct window of the original matrix.
                let base = i * batchsize;
                assert_eq!(*submatrix.element(0, 0), base);
                assert_eq!(*submatrix.element(0, 1), base + 1);
                assert_eq!(*submatrix.element(0, 2), base + 2);

                assert_eq!(*submatrix.element(1, 0), base + 1);
                assert_eq!(*submatrix.element(1, 1), base + 2);
                assert_eq!(*submatrix.element(1, 2), base + 3);
            });

        // Try again, but with a batch size of 3 to ensure that we correctly handle cases
        // where the last block is under-sized.
        let batchsize = 3;
        m.par_window_iter(batchsize)
            .enumerate()
            .for_each(|(i, submatrix)| {
                if i == 0 {
                    assert_eq!(submatrix.nrows(), batchsize);
                    assert_eq!(submatrix.ncols(), m.ncols());

                    // Check indexing
                    assert_eq!(*submatrix.element(0, 0), 0);
                    assert_eq!(*submatrix.element(0, 1), 1);
                    assert_eq!(*submatrix.element(0, 2), 2);

                    assert_eq!(*submatrix.element(1, 0), 1);
                    assert_eq!(*submatrix.element(1, 1), 2);
                    assert_eq!(*submatrix.element(1, 2), 3);

                    assert_eq!(*submatrix.element(2, 0), 2);
                    assert_eq!(*submatrix.element(2, 1), 3);
                    assert_eq!(*submatrix.element(2, 2), 4);
                } else {
                    assert_eq!(submatrix.nrows(), 1);
                    assert_eq!(submatrix.ncols(), m.ncols());

                    // Check indexing
                    assert_eq!(*submatrix.element(0, 0), 3);
                    assert_eq!(*submatrix.element(0, 1), 4);
                    assert_eq!(*submatrix.element(0, 2), 5);
                }
            });

        // par-row-iter
        let seen_rows: Box<[usize]> = m
            .par_row_iter()
            .enumerate()
            .map(|(i, row)| {
                let expected: Box<[usize]> = (0..m.ncols()).map(|j| j + i).collect();
                assert_eq!(row, &*expected);
                i
            })
            .collect();

        let expected: Box<[usize]> = (0..m.nrows()).collect();
        assert_eq!(seen_rows, expected);
    }

    fn test_basic_indexing<T>(m: &T)
    where
        T: Matrix<Element = usize> + Sync,
    {
        assert_eq!(m.nrows(), 4);
        assert_eq!(m.ncols(), 3);

        // Basic indexing
        assert_eq!(*m.element(0, 0), 0);
        assert_eq!(*m.element(0, 1), 1);
        assert_eq!(*m.element(0, 2), 2);

        assert_eq!(*m.element(1, 0), 1);
        assert_eq!(*m.element(1, 1), 2);
        assert_eq!(*m.element(1, 2), 3);

        assert_eq!(*m.element(2, 0), 2);
        assert_eq!(*m.element(2, 1), 3);
        assert_eq!(*m.element(2, 2), 4);

        assert_eq!(*m.element(3, 0), 3);
        assert_eq!(*m.element(3, 1), 4);
        assert_eq!(*m.element(3, 2), 5);

        assert_eq!(*m.get_element(0, 0).unwrap(), 0);
        assert_eq!(*m.get_element(0, 1).unwrap(), 1);
        assert_eq!(*m.get_element(0, 2).unwrap(), 2);

        assert_eq!(*m.get_element(1, 0).unwrap(), 1);
        assert_eq!(*m.get_element(1, 1).unwrap(), 2);
        assert_eq!(*m.get_element(1, 2).unwrap(), 3);

        assert_eq!(*m.get_element(2, 0).unwrap(), 2);
        assert_eq!(*m.get_element(2, 1).unwrap(), 3);
        assert_eq!(*m.get_element(2, 2).unwrap(), 4);

        assert_eq!(*m.get_element(3, 0).unwrap(), 3);
        assert_eq!(*m.get_element(3, 1).unwrap(), 4);
        assert_eq!(*m.get_element(3, 2).unwrap(), 5);

        // Row indexing.
        assert_eq!(m.row(0), &[0, 1, 2]);
        assert_eq!(m.row(1), &[1, 2, 3]);
        assert_eq!(m.row(2), &[2, 3, 4]);
        assert_eq!(m.row(3), &[3, 4, 5]);

        let rows: Vec<Vec<usize>> = m.rows().map(|x| x.to_vec()).collect();
        assert_eq!(m.row(0), &rows[0]);
        assert_eq!(m.row(1), &rows[1]);
        assert_eq!(m.row(2), &rows[2]);
        assert_eq!(m.row(3), &rows[3]);

        // Window Iters.
        let batchsize = 2;
        m.window_iter(NonZeroUsize::new(batchsize).unwrap())
            .enumerate()
            .for_each(|(i, submatrix)| {
                assert_eq!(submatrix.nrows(), batchsize);
                assert_eq!(submatrix.ncols(), m.ncols());

                // Make sure we are in the correct window of the original matrix.
                let base = i * batchsize;
                assert_eq!(*submatrix.element(0, 0), base);
                assert_eq!(*submatrix.element(0, 1), base + 1);
                assert_eq!(*submatrix.element(0, 2), base + 2);

                assert_eq!(*submatrix.element(1, 0), base + 1);
                assert_eq!(*submatrix.element(1, 1), base + 2);
                assert_eq!(*submatrix.element(1, 2), base + 3);
            });

        // Try again, but with a batch size of 3 to ensure that we correctly handle cases
        // where the last block is under-sized.
        let batchsize = 3;
        m.window_iter(NonZeroUsize::new(batchsize).unwrap())
            .enumerate()
            .for_each(|(i, submatrix)| {
                if i == 0 {
                    assert_eq!(submatrix.nrows(), batchsize);
                    assert_eq!(submatrix.ncols(), m.ncols());

                    // Check indexing
                    assert_eq!(*submatrix.element(0, 0), 0);
                    assert_eq!(*submatrix.element(0, 1), 1);
                    assert_eq!(*submatrix.element(0, 2), 2);

                    assert_eq!(*submatrix.element(1, 0), 1);
                    assert_eq!(*submatrix.element(1, 1), 2);
                    assert_eq!(*submatrix.element(1, 2), 3);

                    assert_eq!(*submatrix.element(2, 0), 2);
                    assert_eq!(*submatrix.element(2, 1), 3);
                    assert_eq!(*submatrix.element(2, 2), 4);
                } else {
                    assert_eq!(submatrix.nrows(), 1);
                    assert_eq!(submatrix.ncols(), m.ncols());

                    // Check indexing
                    assert_eq!(*submatrix.element(0, 0), 3);
                    assert_eq!(*submatrix.element(0, 1), 4);
                    assert_eq!(*submatrix.element(0, 2), 5);
                }
            });

        #[cfg(all(not(miri), feature = "rayon"))]
        test_basic_indexing_parallel(m.as_view());
    }

    #[test]
    fn matrix_happy_path() {
        let data = make_test_matrix();
        let m = Owned::try_from_data(data.into(), 4, 3).unwrap();
        test_basic_indexing(&m);

        // Get the base pointer of the matrix and make sure view-conversion preserves this
        // value.
        let ptr = m.as_ptr();
        let view = m.as_view();
        assert!(is_copyable(view));
        assert_eq!(view.as_ptr(), ptr);
        assert_eq!(view.nrows(), m.nrows());
        assert_eq!(view.ncols(), m.ncols());
        test_basic_indexing(&view);
    }

    #[test]
    fn matrix_try_from_construction_error() {
        let data = make_test_matrix();
        let ptr = data.as_ptr();
        let len = data.len();

        let m = Owned::try_from_data(data.into(), 5, 4);
        assert!(m.is_err());
        let err = m.unwrap_err();
        assert_eq!(
            err.to_string(),
            "tried to construct a 5x4 matrix over a span of length 12"
        );

        // Make sure that we can retrieve the original allocation from the interior.
        let data = err.into_inner();
        assert_eq!(data.as_ptr(), ptr);
        assert_eq!(data.len(), len);

        let m = Ref::try_from_data(&data, 5, 4);
        assert!(m.is_err());
        assert_eq!(
            m.unwrap_err().to_string(),
            "tried to construct a 5x4 matrix over a span of length 12"
        );
    }

    #[test]
    fn matrix_mut_view() {
        let mut m = Owned::<usize>::from_element(4, 3, 0);
        assert_eq!(m.nrows(), 4);
        assert_eq!(m.ncols(), 3);
        assert!(m.as_slice().iter().all(|&i| i == 0));
        let ptr = m.as_ptr();
        let mut_ptr = m.as_mut_ptr();
        assert_eq!(ptr, mut_ptr);

        let mut view = m.as_view_mut();
        assert_eq!(view.nrows(), 4);
        assert_eq!(view.ncols(), 3);
        assert_eq!(view.as_ptr(), ptr);
        assert_eq!(view.as_mut_ptr(), mut_ptr);

        // Construct the test matrix manually.
        for i in 0..view.nrows() {
            for j in 0..view.ncols() {
                *view.element_mut(i, j) = i + j;
            }
        }

        // Drop the view and test the original matrix.
        test_basic_indexing(&m);

        let inner = m.into_inner();
        assert_eq!(inner.as_ptr(), ptr);
        assert_eq!(inner.len(), 4 * 3);
    }

    #[test]
    fn matrix_view_zero_sizes() {
        let data: Vec<usize> = vec![];
        // Zero rows, but non-zero columns.
        let m = Ref::try_from_data(data.as_slice(), 0, 10).unwrap();
        assert_eq!(m.nrows(), 0);
        assert_eq!(m.ncols(), 10);

        // Non-zero rows, but zero columns.
        let m = Ref::try_from_data(data.as_slice(), 3, 0).unwrap();
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 0);
        let empty: &[usize] = &[];
        assert_eq!(m.row(0), empty);
        assert_eq!(m.row(1), empty);
        assert_eq!(m.row(2), empty);

        // Zero rows and columns.
        let m = Ref::try_from_data(data.as_slice(), 0, 0).unwrap();
        assert_eq!(m.nrows(), 0);
        assert_eq!(m.ncols(), 0);
    }

    #[test]
    fn matrix_view_construction_elementwise() {
        let mut m = Owned::<usize>::from_element(4, 3, 0);

        // Construct the test matrix manually.
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                *m.element_mut(i, j) = i + j;
            }
        }
        test_basic_indexing(&m);
    }

    #[test]
    fn matrix_construction_by_row() {
        let mut m = Owned::<usize>::from_element(4, 3, 0);
        assert!(m.as_slice().iter().all(|i| *i == 0));

        let ncols = m.ncols();
        for i in 0..m.nrows() {
            let row = m.row_mut(i);
            assert_eq!(row.len(), ncols);
            row[0] = i;
            row[1] = i + 1;
            row[2] = i + 2;
        }
        test_basic_indexing(&m);
    }

    #[test]
    fn matrix_construction_by_rowiter() {
        let mut m = Owned::<usize>::from_element(4, 3, 0);
        assert!(m.as_slice().iter().all(|i| *i == 0));

        let ncols = m.ncols();
        m.rows_mut().enumerate().for_each(|(i, row)| {
            assert_eq!(row.len(), ncols);
            row[0] = i;
            row[1] = i + 1;
            row[2] = i + 2;
        });
        test_basic_indexing(&m);
    }

    #[cfg(all(not(miri), feature = "rayon"))]
    #[test]
    fn matrix_construction_by_par_windows() {
        let mut m = Owned::<usize>::from_element(4, 3, 0);
        assert!(m.as_slice().iter().all(|i| *i == 0));

        let ncols = m.ncols();
        for batchsize in 1..=4 {
            m.par_window_iter_mut(batchsize)
                .enumerate()
                .for_each(|(i, mut submatrix)| {
                    let base = i * batchsize;
                    submatrix.rows_mut().enumerate().for_each(|(j, row)| {
                        assert_eq!(row.len(), ncols);
                        row[0] = base + j;
                        row[1] = base + j + 1;
                        row[2] = base + j + 2;
                    });
                });
            test_basic_indexing(&m);
        }
    }

    #[test]
    fn matrix_construction_happens_in_memory_order() {
        let mut i = 0;
        let ncols = 3;
        let initializer = |_| {
            let value = (i % ncols) + (i / ncols);
            i += 1;
            value
        };

        let m = Owned::from_fn(4, 3, initializer);
        test_basic_indexing(&m);
    }

    // Panics
    #[test]
    #[should_panic(expected = "tried to access row 3 of a matrix with 3 rows")]
    fn test_get_row_panics() {
        let m = Owned::<usize>::from_element(3, 7, 0);
        m.row(3);
    }

    #[test]
    #[should_panic(expected = "tried to access row 3 of a matrix with 3 rows")]
    fn test_get_row_mut_panics() {
        let mut m = Owned::<usize>::from_element(3, 7, 0);
        m.row_mut(3);
    }

    #[test]
    #[should_panic(expected = "row 3 is out of bounds (max: 3)")]
    fn test_element_panics_row() {
        let m = Owned::<usize>::from_element(3, 7, 0);
        assert!(m.get_element(3, 2).is_none());
        let _ = m.element(3, 2);
    }

    #[test]
    #[should_panic(expected = "col 7 is out of bounds (max: 7)")]
    fn test_element_panics_col() {
        let m = Owned::<usize>::from_element(3, 7, 0);
        assert!(m.get_element(2, 7).is_none());
        let _ = m.element(2, 7);
    }

    #[test]
    #[should_panic(expected = "row 3 is out of bounds (max: 3)")]
    fn test_element_mut_panics_row() {
        let mut m = Owned::<usize>::from_element(3, 7, 0);
        assert!(m.get_element_mut(3, 2).is_none());
        *m.element_mut(3, 2) = 1;
    }

    #[test]
    #[should_panic(expected = "col 7 is out of bounds (max: 7)")]
    fn test_element_mut_panics_col() {
        let mut m = Owned::<usize>::from_element(3, 7, 0);
        assert!(m.get_element_mut(2, 7).is_none());
        *m.element_mut(2, 7) = 1;
    }

    #[test]
    #[cfg(feature = "rayon")]
    #[should_panic(expected = "par_window_iter batchsize cannot be zero")]
    fn test_par_window_iter_panics() {
        let m = Owned::<usize>::from_element(4, 4, 0);
        let _ = m.par_window_iter(0);
    }

    #[test]
    #[cfg(feature = "rayon")]
    #[should_panic(expected = "par_window_iter_mut batchsize cannot be zero")]
    fn test_par_window_iter_mut_panics() {
        let mut m = Owned::<usize>::from_element(4, 4, 0);
        let _ = m.par_window_iter_mut(0);
    }

    // Additional tests for better coverage

    #[test]
    fn test_try_from_error_light() {
        // Incorrect slice
        let data = vec![1, 2, 3];
        let err = Ref::try_from_data(data.as_slice(), 2, 3).unwrap_err();

        // Test `as_static` method
        let err_static = err.as_static();
        let msg = err_static.to_string();
        assert_contains!(
            msg,
            "tried to construct a 2x3 matrix over a span of length 3",
        );
        // Test `into_inner` method
        let recovered_data = err.into_inner();
        assert_eq!(recovered_data, data.as_slice());

        // Invalid length.
        let err = Ref::try_from_data(data.as_slice(), 2, usize::MAX).unwrap_err();
        let msg = err.to_string();
        assert_contains!(msg, "usize::MAX");

        assert_eq!(data.as_slice(), err.into_inner());
    }

    #[test]
    fn test_map_errors() {
        #[derive(Debug, Clone, Copy)]
        struct Zst;

        // Create a large ZST slice without taking forever on debug builds.
        let b = Box::<[Zst]>::new_uninit_slice((isize::MAX as usize) + 1);

        // SAFETY: `b` has zero-sized elements, so all elements are initialized.
        let b = unsafe { b.assume_init() };

        let m = Owned::column_vector(b);
        let err = m.try_map(|_: &Zst| 0u8).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("isize::MAX"), "{msg}");

        // Panicking variant.
        let err = std::panic::catch_unwind(|| m.map(|_: &Zst| 0u8))
            .unwrap_err()
            .downcast::<String>()
            .unwrap();
        let msg = err.to_string();
        assert!(msg.contains("isize::MAX"), "{msg}");
    }

    #[test]
    fn test_get_row_optional() {
        let data = make_test_matrix();
        let m = Ref::try_from_data(data.as_slice(), 4, 3).unwrap();

        // Test successful get_row
        assert_eq!(m.get_row(0), Some(&[0, 1, 2][..]));
        assert_eq!(m.get_row(1), Some(&[1, 2, 3][..]));
        assert_eq!(m.get_row(3), Some(&[3, 4, 5][..]));

        // Test out-of-bounds get_row
        assert_eq!(m.get_row(4), None);
        assert_eq!(m.get_row(100), None);
    }

    #[test]
    fn test_unsafe_get_unchecked_methods() {
        let data = make_test_matrix();
        let mut m = Owned::try_from_data(data.into(), 4, 3).unwrap();

        // Safety: derives from known size of matrix and access element ids
        unsafe {
            assert_eq!(*m.element_unchecked(0, 0), 0);
            assert_eq!(*m.element_unchecked(1, 2), 3);
            assert_eq!(*m.element_unchecked(3, 1), 4);
        }

        // Safety: derives from known size of matrix and access element ids
        unsafe {
            *m.element_unchecked_mut(0, 0) = 100;
            *m.element_unchecked_mut(1, 2) = 200;
        }

        assert_eq!(*m.element(0, 0), 100);
        assert_eq!(*m.element(1, 2), 200);

        // Safety: derives from known size of matrix and access element ids
        unsafe {
            let row0 = m.row_unchecked(0);
            assert_eq!(row0[0], 100);
            assert_eq!(row0[1], 1);
            assert_eq!(row0[2], 2);
        }

        // Safety: derives from known size of matrix and access element ids
        unsafe {
            let row1 = m.row_unchecked_mut(1);
            row1[0] = 300;
        }

        assert_eq!(*m.element(1, 0), 300);
    }

    #[test]
    fn test_to_owned() {
        let data = make_test_matrix();
        let view = Ref::try_from_data(data.as_slice(), 4, 3).unwrap();

        // Test to_owned creates a proper clone
        let owned: Owned<_> = view.to_rowmajor_owned();
        assert_eq!(owned.nrows(), view.nrows());
        assert_eq!(owned.ncols(), view.ncols());
        assert_eq!(owned.as_slice(), view.as_slice());

        // Verify it's actually owned (different memory location)
        assert_ne!(owned.as_ptr(), view.as_ptr());

        // Test the owned matrix works properly
        test_basic_indexing(&owned);
    }

    #[test]
    fn test_matrix_from_conversions() {
        let data = make_test_matrix();
        let m = Owned::try_from_data(data.into(), 4, 3).unwrap();

        // Test Ref to slice conversion
        let view = m.as_view();
        let slice: &[usize] = view.into_slice();
        assert_eq!(slice.len(), 12);
        assert_eq!(slice[0], 0);
        assert_eq!(slice[11], 5);

        // Test Ref to slice conversion
        let data2 = make_test_matrix();
        let mut m2 = Owned::try_from_data(data2.into(), 4, 3).unwrap();
        let mut_view = m2.as_view_mut();
        let slice2: &[usize] = mut_view.as_slice();
        assert_eq!(slice2.len(), 12);
        assert_eq!(slice2[0], 0);
        assert_eq!(slice2[11], 5);
    }

    #[test]
    fn test_matrix_construction_edge_cases() {
        // Test 1x1 matrix
        let m = Owned::from_element(1, 1, 42);
        assert_eq!(m.nrows(), 1);
        assert_eq!(m.ncols(), 1);
        assert_eq!(*m.element(0, 0), 42);
        assert_eq!(*m.get_element(0, 0).unwrap(), 42);

        // Test single row matrix
        let m = Owned::from_element(1, 5, 7);
        assert_eq!(m.nrows(), 1);
        assert_eq!(m.ncols(), 5);
        assert!(m.as_slice().iter().all(|&x| x == 7));

        // Test single column matrix
        let m = Owned::from_element(5, 1, 9);
        assert_eq!(m.nrows(), 5);
        assert_eq!(m.ncols(), 1);
        assert!(m.as_slice().iter().all(|&x| x == 9));
    }

    #[test]
    fn test_matrix_view_edge_cases_with_data() {
        // Test matrix with actual data for edge cases
        let data = vec![10, 20];

        // 2x1 matrix
        let m = Ref::try_from_data(data.as_slice(), 2, 1).unwrap();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 1);
        assert_eq!(*m.element(0, 0), 10);
        assert_eq!(*m.element(1, 0), 20);
        assert_eq!(*m.get_element(0, 0).unwrap(), 10);
        assert_eq!(*m.get_element(1, 0).unwrap(), 20);
        assert_eq!(m.row(0), &[10]);
        assert_eq!(m.row(1), &[20]);

        // 1x2 matrix
        let m = Ref::try_from_data(data.as_slice(), 1, 2).unwrap();
        assert_eq!(m.nrows(), 1);
        assert_eq!(m.ncols(), 2);
        assert_eq!(*m.element(0, 0), 10);
        assert_eq!(*m.element(0, 1), 20);
        assert_eq!(*m.get_element(0, 0).unwrap(), 10);
        assert_eq!(*m.get_element(0, 1).unwrap(), 20);
        assert_eq!(m.row(0), &[10, 20]);
    }

    #[test]
    fn test_row_vector() {
        let data = vec![1, 2, 3];
        let m = Ref::row_vector(data.as_slice());
        assert_eq!(m.nrows(), 1);
        assert_eq!(m.ncols(), 3);
        assert_eq!(m.as_slice(), &[1, 2, 3]);
        assert_eq!(m.row(0), &[1, 2, 3]);

        // Empty
        let empty: &[i32] = &[];
        let m = Ref::row_vector(empty);
        assert_eq!(m.nrows(), 1);
        assert_eq!(m.ncols(), 0);

        // Owned
        let m = Owned::row_vector(vec![10u64, 20].into_boxed_slice());
        assert_eq!(m.nrows(), 1);
        assert_eq!(m.ncols(), 2);
        assert_eq!(*m.element(0, 0), 10);
        assert_eq!(*m.element(0, 1), 20);
    }

    #[test]
    fn test_column_vector() {
        let data = vec![1, 2, 3];
        let m = Ref::column_vector(data.as_slice());
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 1);
        assert_eq!(m.as_slice(), &[1, 2, 3]);
        assert_eq!(*m.element(0, 0), 1);
        assert_eq!(*m.element(1, 0), 2);
        assert_eq!(*m.element(2, 0), 3);
        assert_eq!(m.row(0), &[1]);
        assert_eq!(m.row(1), &[2]);
        assert_eq!(m.row(2), &[3]);

        // Empty
        let empty: &[i32] = &[];
        let m = Ref::column_vector(empty);
        assert_eq!(m.nrows(), 0);
        assert_eq!(m.ncols(), 1);

        // Owned
        let m = Owned::column_vector(vec![10u64, 20].into_boxed_slice());
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 1);
        assert_eq!(*m.element(0, 0), 10);
        assert_eq!(*m.element(1, 0), 20);
    }

    #[test]
    fn test_map() {
        let m = Owned::try_from_data(vec![1u32, 2, 3, 4].into(), 2, 2).unwrap();
        let doubled = m.map(|&x| x * 2);
        assert_eq!(doubled.as_slice(), &[2, 4, 6, 8]);
        assert_eq!(doubled.nrows(), 2);
        assert_eq!(doubled.ncols(), 2);

        // Type-changing map
        let as_f64 = m.map(|&x| x as f64);
        assert_eq!(as_f64.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_get_element() {
        let mut m = Owned::try_from_data(vec![1, 2, 3, 4, 5, 6].into(), 2, 3).unwrap();
        assert_eq!(m.get_element(0, 0), Some(&1));
        assert_eq!(m.get_element(1, 2), Some(&6));
        assert_eq!(m.get_element(2, 0), None);
        assert_eq!(m.get_element(0, 3), None);

        *m.get_element_mut(1, 2).unwrap() = 7;
        assert_eq!(m.get_element(1, 2), Some(&7));
        assert_eq!(m.get_element_mut(2, 0), None);
        assert_eq!(m.get_element_mut(0, 3), None);
    }

    #[test]
    fn test_subview() {
        let data = make_test_matrix();
        let m = Owned::try_from_data(data.into(), 4, 3).unwrap();

        // Create a subview of the first two rows
        {
            let subview = m.subview(0..4).unwrap();
            assert_eq!(subview.nrows(), 4);
            assert_eq!(subview.ncols(), 3);

            assert_eq!(subview.row(0), &[0, 1, 2]);
            assert_eq!(subview.row(1), &[1, 2, 3]);
            assert_eq!(subview.row(2), &[2, 3, 4]);
            assert_eq!(subview.row(3), &[3, 4, 5]);
            assert!(subview.get_row(4).is_none());
        }

        // Sub view over a subset that touches the end.
        {
            let subview = m.subview(1..4).unwrap();
            assert_eq!(subview.nrows(), 3);
            assert_eq!(subview.ncols(), 3);

            assert_eq!(subview.row(0), &[1, 2, 3]);
            assert_eq!(subview.row(1), &[2, 3, 4]);
            assert_eq!(subview.row(2), &[3, 4, 5]);
            assert!(subview.get_row(3).is_none());
        }

        // Sub view over a subset that is in the middle
        {
            let subview = m.subview(1..3).unwrap();
            assert_eq!(subview.nrows(), 2);
            assert_eq!(subview.ncols(), 3);

            assert_eq!(subview.row(0), &[1, 2, 3]);
            assert_eq!(subview.row(1), &[2, 3, 4]);
            assert!(subview.get_row(2).is_none());
        }

        // Empty sub-view.
        {
            let subview = m.subview(2..2).unwrap();
            assert_eq!(subview.nrows(), 0);
            assert_eq!(subview.ncols(), 3);
        }

        // Empty subview in bounds
        {
            let subview = m.subview(0..0).unwrap();
            assert_eq!(subview.nrows(), 0);
            assert_eq!(subview.ncols(), 3);

            let subview = m.subview(4..4).unwrap();
            assert_eq!(subview.nrows(), 0);
            assert_eq!(subview.ncols(), 3);
        }

        // Empty out-of-bounds subview
        assert!(m.subview(5..5).is_none());

        // View too-large
        assert!(m.subview(0..6).is_none());
        assert!(m.subview(2..10).is_none());

        // View disjoint.
        assert!(m.subview(10..100).is_none());

        // Negative bounds
        #[expect(
            clippy::reversed_empty_ranges,
            reason = "we want to make sure it doesn't work"
        )]
        let empty = 3..2;
        assert!(m.subview(empty).is_none());

        #[expect(
            clippy::reversed_empty_ranges,
            reason = "we want to make sure it doesn't work"
        )]
        let empty = 3..1;
        assert!(m.subview(empty).is_none());

        // Bounds that overflow.
        assert!(m.subview(usize::MAX - 1..usize::MAX).is_none());
        assert!(m.subview(0..usize::MAX).is_none());
    }

    #[expect(
        clippy::reversed_empty_ranges,
        reason = "we want to make sure it doesn't work"
    )]
    #[test]
    fn test_subview_zero_cols() {
        let m = Owned::from_element(10, 0, 0u32);

        // Out-of-bounds indexing
        assert!(m.subview(100..200).is_none());
        assert!(m.subview(200..100).is_none());

        assert!(m.subview(10..11).is_none());
        assert!(m.subview(11..10).is_none());

        assert!(m.subview(0..11).is_none());
        assert!(m.subview(11..0).is_none());

        assert!(m.subview(10..0).is_none());
        assert!(m.subview(5..4).is_none());

        // In-bounds.
        let v = m.subview(5..10).unwrap();
        assert_eq!(v.nrows(), 5);
        assert_eq!(v.ncols(), 0);

        let v = m.subview(0..0).unwrap();
        assert_eq!(v.nrows(), 0);
        assert_eq!(v.ncols(), 0);

        let v = m.subview(10..10).unwrap();
        assert_eq!(v.nrows(), 0);
        assert_eq!(v.ncols(), 0);

        let v = m.subview(0..10).unwrap();
        assert_eq!(v.nrows(), 10);
        assert_eq!(v.ncols(), 0);
    }

    #[test]
    #[cfg(all(not(miri), feature = "rayon"))]
    fn test_parallel_methods_edge_cases() {
        let data = make_test_matrix();
        let m = Owned::try_from_data(data.into(), 4, 3).unwrap();

        // Test par_window_iter with batchsize larger than matrix
        let windows: Vec<_> = m.par_window_iter(10).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].nrows(), 4);
        assert_eq!(windows[0].ncols(), 3);

        // Test par_row_iter
        let rows: Vec<_> = m.par_row_iter().collect();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0], &[0, 1, 2]);
        assert_eq!(rows[3], &[3, 4, 5]);

        // Test par_window_iter_mut and par_row_iter_mut
        let mut m2 = Owned::from_element(4, 3, 0);

        // Use par_row_iter_mut to set values
        m2.par_row_iter_mut().enumerate().for_each(|(i, row)| {
            for (j, elem) in row.iter_mut().enumerate() {
                *elem = i + j;
            }
        });
        test_basic_indexing(&m2);

        // Test par_window_iter_mut with larger batchsize
        let mut m3 = Owned::from_element(4, 3, 0);
        m3.par_window_iter_mut(10)
            .enumerate()
            .for_each(|(_, mut window)| {
                window.rows_mut().enumerate().for_each(|(i, row)| {
                    for (j, elem) in row.iter_mut().enumerate() {
                        *elem = i + j;
                    }
                });
            });
        test_basic_indexing(&m3);
    }

    #[test]
    fn test_matrix_pointers() {
        let mut m = Owned::from_element(3, 4, 42);

        // Test as_ptr and as_mut_ptr return the same address
        let const_ptr = m.as_ptr();
        let mut_ptr = m.as_mut_ptr();
        assert_eq!(const_ptr, mut_ptr as *const _);

        // Test that view pointers match original
        let view = m.as_view();
        assert_eq!(view.as_ptr(), const_ptr);

        let mut mut_view = m.as_view_mut();
        assert_eq!(mut_view.as_ptr(), const_ptr);
        assert_eq!(mut_view.as_mut_ptr(), mut_ptr);
    }

    #[test]
    fn test_matrix_iteration_empty_cases() {
        // Test construction of empty matrices (we don't iterate over 0x0 matrices
        // since chunks_exact requires non-zero chunk size)
        let empty_data: Vec<i32> = vec![];

        // Matrix with 0 rows but non-zero cols can be constructed
        let _empty_matrix = Ref::try_from_data(empty_data.as_slice(), 0, 5).unwrap();

        // Test with actual single row to verify iterator works normally
        let data = vec![1, 2, 3];
        let single_row = Ref::try_from_data(data.as_slice(), 1, 3).unwrap();
        let rows: Vec<_> = single_row.rows().collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], &[1, 2, 3]);

        // Test iteration over matrix with multiple rows but single column
        let data = vec![1, 2, 3];
        let single_col = Ref::try_from_data(data.as_slice(), 3, 1).unwrap();
        let rows: Vec<_> = single_col.rows().collect();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], &[1]);
        assert_eq!(rows[1], &[2]);
        assert_eq!(rows[2], &[3]);
    }

    #[test]
    fn test_matrix_init_generator_various_types() {
        // Test with different types and generators
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = AtomicUsize::new(0);
        let m = Owned::from_fn(2, 3, |_| counter.fetch_add(1, Ordering::SeqCst));

        // Should be filled in memory order
        assert_eq!(*m.element(0, 0), 0);
        assert_eq!(*m.element(0, 1), 1);
        assert_eq!(*m.element(0, 2), 2);
        assert_eq!(*m.element(1, 0), 3);
        assert_eq!(*m.element(1, 1), 4);
        assert_eq!(*m.element(1, 2), 5);
    }

    #[test]
    fn test_transpose() {
        {
            let v = Owned::from_element(0, 0, 0);
            let t = v.transpose();
            assert_eq!(t.nrows(), 0);
            assert_eq!(t.ncols(), 0);
        }

        {
            let v = Owned::from_element(0, 10, 0);
            let t = v.transpose();
            assert_eq!(t.nrows(), 10);
            assert_eq!(t.ncols(), 0);
        }

        {
            let v = Owned::from_element(10, 0, 0);
            let t = v.transpose();
            assert_eq!(t.nrows(), 0);
            assert_eq!(t.ncols(), 10);
        }

        {
            let v = Owned::<usize>::try_from_data(Box::new([1, 2, 3, 4, 5, 6]), 2, 3).unwrap();
            let t = v.transpose();

            assert_eq!(t.row(0), &[1, 4]);
            assert_eq!(t.row(1), &[2, 5]);
            assert_eq!(t.row(2), &[3, 6]);
        }
    }

    #[test]
    fn test_debug_error_formatting() {
        // Test Debug implementation for TryFromError
        let data = vec![1, 2, 3];
        let err = Owned::try_from_data(data.into(), 2, 3).unwrap_err();
        let debug_str = format!("{:?}", err);
        assert_contains!(debug_str, "TryFromError");

        // Ensure Debug doesn't require T: Debug by using a non-Debug type
        #[derive(Clone)]
        struct NonDebug(#[expect(dead_code)] i32);

        let non_debug_data: Box<[NonDebug]> = vec![NonDebug(1), NonDebug(2)].into();
        let non_debug_err = match Owned::try_from_data(non_debug_data, 1, 3) {
            Ok(_) => panic!("should not have succeeded!"),
            Err(err) => err,
        };
        let debug_str = format!("{:?}", non_debug_err);
        assert_contains!(debug_str, "TryFromError");
    }

    // Comprehensive tests for rayon-specific functionality

    #[test]
    #[cfg(feature = "rayon")]
    fn test_par_window_iter_comprehensive() {
        use rayon::prelude::*;

        // Create a larger test matrix for more comprehensive testing
        let data: Vec<usize> = (0..24).collect(); // 6x4 matrix
        let m = Ref::try_from_data(data.as_slice(), 6, 4).unwrap();

        // Test various batch sizes
        for batchsize in 1..=8 {
            let context = lazy_format!("batchsize = {}", batchsize);
            let windows: Vec<_> = m.par_window_iter(batchsize).collect();

            // Calculate expected number of windows
            let expected_windows = (m.nrows()).div_ceil(batchsize);
            assert_eq!(windows.len(), expected_windows, "{}", context);

            // Verify each window's properties
            let mut total_rows_seen = 0;
            for (window_idx, window) in windows.iter().enumerate() {
                let expected_rows = if window_idx == windows.len() - 1 {
                    // Last window may have fewer rows
                    m.nrows() - (windows.len() - 1) * batchsize
                } else {
                    batchsize
                };

                assert_eq!(
                    window.nrows(),
                    expected_rows,
                    "window {} - {}",
                    window_idx,
                    context
                );
                assert_eq!(
                    window.ncols(),
                    m.ncols(),
                    "window {} - {}",
                    window_idx,
                    context
                );

                // Verify data integrity
                for (row_idx, row) in window.rows().enumerate() {
                    let global_row = window_idx * batchsize + row_idx;
                    let expected: Vec<usize> =
                        (0..m.ncols()).map(|j| global_row * m.ncols() + j).collect();
                    assert_eq!(
                        row,
                        expected.as_slice(),
                        "window {}, row {} - {}",
                        window_idx,
                        row_idx,
                        context
                    );
                }

                total_rows_seen += window.nrows();
            }

            assert_eq!(total_rows_seen, m.nrows(), "{}", context);
        }

        // Test with batchsize equal to matrix rows
        let windows: Vec<_> = m.par_window_iter(m.nrows()).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].nrows(), m.nrows());
        assert_eq!(windows[0].ncols(), m.ncols());

        // Test with batchsize larger than matrix rows
        let windows: Vec<_> = m.par_window_iter(m.nrows() * 2).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].nrows(), m.nrows());
        assert_eq!(windows[0].ncols(), m.ncols());
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_par_window_iter_mut_comprehensive() {
        use rayon::prelude::*;

        // Test various matrix sizes and batch sizes
        for nrows in [1, 2, 3, 5, 8, 10] {
            for ncols in [1, 3, 4] {
                for batchsize in [1, 2, 3, 7] {
                    let context = lazy_format!("{}x{}, batchsize={}", nrows, ncols, batchsize);

                    let mut m = Owned::from_element(nrows, ncols, 0usize);

                    // Use par_window_iter_mut to fill matrix
                    m.par_window_iter_mut(batchsize).enumerate().for_each(
                        |(window_idx, mut window)| {
                            let base_row = window_idx * batchsize;
                            window.rows_mut().enumerate().for_each(|(row_offset, row)| {
                                let global_row = base_row + row_offset;
                                for (col, elem) in row.iter_mut().enumerate() {
                                    *elem = global_row * ncols + col;
                                }
                            });
                        },
                    );

                    // Verify the matrix was filled correctly
                    for row in 0..nrows {
                        for col in 0..ncols {
                            let expected = row * ncols + col;
                            assert_eq!(
                                *m.element(row, col),
                                expected,
                                "pos ({}, {}) - {}",
                                row,
                                col,
                                context
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_par_row_iter_comprehensive() {
        use rayon::prelude::*;

        // Create test matrix with predictable pattern
        let nrows = 7;
        let ncols = 5;
        let data: Vec<i32> = (0..(nrows * ncols) as i32).collect();
        let m = Ref::try_from_data(data.as_slice(), nrows, ncols).unwrap();

        // Test that par_row_iter preserves order and data
        let collected_rows: Vec<Vec<i32>> = m.par_row_iter().map(|row| row.to_vec()).collect();

        assert_eq!(collected_rows.len(), nrows);

        for (row_idx, row) in collected_rows.iter().enumerate() {
            assert_eq!(row.len(), ncols);
            let expected: Vec<i32> = ((row_idx * ncols)..((row_idx + 1) * ncols))
                .map(|x| x as i32)
                .collect();
            assert_eq!(row, &expected, "row {} mismatch", row_idx);
        }

        // Test parallel enumeration
        let enumerated_rows: Vec<(usize, Vec<i32>)> = m
            .par_row_iter()
            .enumerate()
            .map(|(idx, row)| (idx, row.to_vec()))
            .collect();

        // Sort by index to ensure we got all indices
        let mut sorted_rows = enumerated_rows;
        sorted_rows.sort_by_key(|(idx, _)| *idx);

        assert_eq!(sorted_rows.len(), nrows);
        for (expected_idx, (actual_idx, row)) in sorted_rows.iter().enumerate() {
            assert_eq!(*actual_idx, expected_idx);
            assert_eq!(row.len(), ncols);
        }

        // Test parallel reduction operations
        let sum: i32 = m.par_row_iter().map(|row| row.iter().sum::<i32>()).sum();

        let expected_sum: i32 = data.iter().sum();
        assert_eq!(sum, expected_sum);

        // Test parallel find operations
        let target_row = 3;
        let found_row = m
            .par_row_iter()
            .enumerate()
            .find_any(|(idx, _)| *idx == target_row)
            .map(|(_, row)| row.to_vec());

        assert!(found_row.is_some());
        let expected_row: Vec<i32> = ((target_row * ncols)..((target_row + 1) * ncols))
            .map(|x| x as i32)
            .collect();
        assert_eq!(found_row.unwrap(), expected_row);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_par_row_iter_mut_comprehensive() {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let nrows = 6;
        let ncols = 4;
        let mut m = Owned::from_element(nrows, ncols, 0u32);

        // Test parallel modification
        m.par_row_iter_mut().enumerate().for_each(|(row_idx, row)| {
            for (col_idx, elem) in row.iter_mut().enumerate() {
                *elem = (row_idx * ncols + col_idx) as u32;
            }
        });

        // Verify modifications were applied correctly
        for row in 0..nrows {
            for col in 0..ncols {
                let expected = (row * ncols + col) as u32;
                assert_eq!(*m.element(row, col), expected, "pos ({}, {})", row, col);
            }
        }

        // Test parallel accumulation with atomic counter
        let counter = AtomicUsize::new(0);
        m.par_row_iter_mut().for_each(|row| {
            counter.fetch_add(1, Ordering::Relaxed);
            // Multiply each element by 2
            for elem in row {
                *elem *= 2;
            }
        });

        assert_eq!(counter.load(Ordering::Relaxed), nrows);

        // Verify all elements were doubled
        for row in 0..nrows {
            for col in 0..ncols {
                let expected = ((row * ncols + col) * 2) as u32;
                assert_eq!(
                    *m.element(row, col),
                    expected,
                    "doubled pos ({}, {})",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_parallel_iterators_with_single_dimensions() {
        use rayon::prelude::*;

        // Test single row matrix
        let data = vec![1, 2, 3, 4, 5];
        let single_row = Ref::try_from_data(data.as_slice(), 1, 5).unwrap();

        let windows: Vec<_> = single_row.par_window_iter(1).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].nrows(), 1);
        assert_eq!(windows[0].ncols(), 5);

        let rows: Vec<_> = single_row.par_row_iter().collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], &[1, 2, 3, 4, 5]);

        // Test single column matrix
        let data = vec![1, 2, 3, 4, 5];
        let single_col = Ref::try_from_data(data.as_slice(), 5, 1).unwrap();

        let windows: Vec<_> = single_col.par_window_iter(2).collect();
        assert_eq!(windows.len(), 3); // ceil(5/2) = 3
        assert_eq!(windows[0].nrows(), 2);
        assert_eq!(windows[1].nrows(), 2);
        assert_eq!(windows[2].nrows(), 1); // Last window has remainder

        let rows: Vec<_> = single_col.par_row_iter().collect();
        assert_eq!(rows.len(), 5);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row, &[i + 1]);
        }

        // Test 1x1 matrix
        let data = vec![42];
        let tiny = Ref::try_from_data(data.as_slice(), 1, 1).unwrap();

        let windows: Vec<_> = tiny.par_window_iter(1).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(*windows[0].element(0, 0), 42);

        let rows: Vec<_> = tiny.par_row_iter().collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], &[42]);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_parallel_window_properties() {
        use rayon::prelude::*;

        // Test that windows maintain proper matrix properties
        let data: Vec<usize> = (0..30).collect();
        let m = Ref::try_from_data(data.as_slice(), 6, 5).unwrap();

        // Test window indexing works correctly
        m.par_window_iter(2)
            .enumerate()
            .for_each(|(window_idx, window)| {
                for row_idx in 0..window.nrows() {
                    for col_idx in 0..window.ncols() {
                        let global_row = window_idx * 2 + row_idx;
                        let expected = global_row * 5 + col_idx;
                        assert_eq!(
                            *window.element(row_idx, col_idx),
                            expected,
                            "window {}, pos ({}, {})",
                            window_idx,
                            row_idx,
                            col_idx
                        );
                    }
                }
            });

        // Test window as_slice consistency
        m.par_window_iter(3)
            .enumerate()
            .for_each(|(window_idx, window)| {
                let slice = window.as_slice();
                assert_eq!(slice.len(), window.nrows() * window.ncols());

                for (slice_idx, &value) in slice.iter().enumerate() {
                    let row = slice_idx / window.ncols();
                    let col = slice_idx % window.ncols();
                    assert_eq!(
                        value,
                        *window.element(row, col),
                        "window {}, slice_idx {}",
                        window_idx,
                        slice_idx
                    );
                }
            });

        // Test window row iteration
        m.par_window_iter(2).for_each(|window| {
            let rows_via_iter: Vec<_> = window.rows().collect();
            assert_eq!(rows_via_iter.len(), window.nrows());

            for (row_idx, row) in rows_via_iter.iter().enumerate() {
                assert_eq!(row.len(), window.ncols());
                for (col_idx, &value) in row.iter().enumerate() {
                    assert_eq!(value, *window.element(row_idx, col_idx));
                }
            }
        });
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_parallel_performance_characteristics() {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Create a larger matrix to test parallelism benefits
        let nrows = 100;
        let ncols = 10;
        let mut m = Owned::from_element(nrows, ncols, 0usize);

        // Test that parallel operations can be chained
        let work_counter = AtomicUsize::new(0);

        m.par_window_iter_mut(10)
            .enumerate()
            .for_each(|(window_idx, mut window)| {
                work_counter.fetch_add(1, Ordering::Relaxed);

                // Nested parallel operation within window
                window.rows_mut().enumerate().for_each(|(row_offset, row)| {
                    let global_row = window_idx * 10 + row_offset;
                    for (col, elem) in row.iter_mut().enumerate() {
                        *elem = global_row * ncols + col;
                    }
                });
            });

        // Should have processed 10 windows (100 rows / 10 batch size)
        assert_eq!(work_counter.load(Ordering::Relaxed), 10);

        // Verify correctness
        for row in 0..nrows {
            for col in 0..ncols {
                assert_eq!(*m.element(row, col), row * ncols + col);
            }
        }

        // Test parallel reduction across windows
        let total_sum: usize = m
            .par_window_iter(15)
            .map(|window| {
                window
                    .rows()
                    .map(|row| row.iter().sum::<usize>())
                    .sum::<usize>()
            })
            .sum();

        let expected_sum: usize = (0..(nrows * ncols)).sum();
        assert_eq!(total_sum, expected_sum);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_rayon_trait_bounds_validation() {
        use rayon::prelude::*;

        // Test that the Sync/Send bounds work correctly
        let data: Vec<u64> = (0..20).collect();
        let m = Ref::try_from_data(data.as_slice(), 4, 5).unwrap();

        // This should compile because u64 is Sync
        let _: Vec<_> = m.par_window_iter(2).collect();
        let _: Vec<_> = m.par_row_iter().collect();

        // Test with mutable matrix
        let mut m = Owned::from_element(4, 5, 0u64);

        // This should compile because u64 is Send
        m.par_window_iter_mut(2).for_each(|mut window| {
            window.rows_mut().for_each(|row| {
                for elem in row {
                    *elem = 42;
                }
            });
        });

        m.par_row_iter_mut().for_each(|row| {
            for elem in row {
                *elem += 1;
            }
        });

        // Verify all elements are 43
        assert!(m.as_slice().iter().all(|&x| x == 43));
    }
}
