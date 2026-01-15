/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt,
    ops::{Index, IndexMut},
};

#[cfg(feature = "rayon")]
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSlice, ParallelSliceMut};
use thiserror::Error;

/// Various view types (types such as [`MatrixView`] that add semantic meaning to blobs
/// of data) need both immutable and mutable variants.
///
/// This trait can be implemented by wrappers for immutable and mutable slice references,
/// allowing for a common code path for immutable and mutable view types.
///
/// The main goal is to provide a way of retrieving an underlying dense slice, which can
/// then be used as the building block for higher level abstractions.
///
/// # Safety
///
/// This trait is unsafe because it requires `as_slice` to be idempotent (and unsafe code
/// relies on this).
///
/// In other words: `as_slice` must **always** return the same slice with the same length.
pub unsafe trait DenseData {
    type Elem;

    /// Return the underlying data as a slice.
    fn as_slice(&self) -> &[Self::Elem];
}

/// A mutable companion to `DenseData`.
///
/// This trait allows mutable methods on view types to be selectively enabled when data
/// underlying the type is mutable.
///
/// # Safety
///
/// This trait is unsafe because it requires `as_slice` to be idempotent (and unsafe code
/// relies on this).
///
/// In other words: `as_slice` must **always** return the same slice with the same length.
///
/// Additionally, the returned slice must span the exact same memory as `as_slice`.
pub unsafe trait MutDenseData: DenseData {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

// SAFETY: This fulfills the idempotency requirement.
unsafe impl<T> DenseData for &[T] {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: This fulfills the idempotency requirement.
unsafe impl<T> DenseData for &mut [T] {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: This fulfills the idempotency requirement and returns a slice spanning the same
// range as `as_slice`.
unsafe impl<T> MutDenseData for &mut [T] {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

// SAFETY: This fulfills the idempotency requirement.
unsafe impl<T> DenseData for Box<[T]> {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: This fulfills the idempotency requirement and returns a slice spanning the same
// memory as `as_slice`.
unsafe impl<T> MutDenseData for Box<[T]> {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

////////////
// Matrix //
////////////

/// A view over dense chunk of memory, interpreting that memory as a 2-dimensional matrix
/// laid out in row-major order.
///
/// When this class view immutable memory, it is `Copy`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixBase<T>
where
    T: DenseData,
{
    data: T,
    nrows: usize,
    ncols: usize,
}

#[derive(Debug, Error)]
#[non_exhaustive]
#[error(
    "tried to construct a matrix view with {nrows} rows and {ncols} columns over a slice \
     of length {len}"
)]
pub struct TryFromErrorLight {
    len: usize,
    nrows: usize,
    ncols: usize,
}

#[derive(Error)]
#[non_exhaustive]
#[error(
    "tried to construct a matrix view with {nrows} rows and {ncols} columns over a slice \
     of length {}", data.as_slice().len()
)]
pub struct TryFromError<T: DenseData> {
    data: T,
    nrows: usize,
    ncols: usize,
}

// Manually implement `fmt::Debug` so we don't require `T::Debug`.
impl<T: DenseData> fmt::Debug for TryFromError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TryFromError")
            .field("data_len", &self.data.as_slice().len())
            .field("nrows", &self.nrows)
            .field("ncols", &self.ncols)
            .finish()
    }
}

impl<T: DenseData> TryFromError<T> {
    /// Consume the error and return the base data.
    pub fn into_inner(self) -> T {
        self.data
    }

    /// Return a variation of `Self` that is guaranteed to be `'static` by removing the
    /// data that was passed to the original constructor.
    pub fn as_static(&self) -> TryFromErrorLight {
        TryFromErrorLight {
            len: self.data.as_slice().len(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

/// A generator for initializing the entries in a matrix via `Matrix::new`.
pub trait Generator<T> {
    fn generate(&mut self) -> T;
}

impl<T> Generator<T> for T
where
    T: Clone,
{
    fn generate(&mut self) -> T {
        self.clone()
    }
}

/// A matrix initializer that invokes the provided lambda to initialize each element.
pub struct Init<F>(pub F);

impl<T, F> Generator<T> for Init<F>
where
    F: FnMut() -> T,
{
    fn generate(&mut self) -> T {
        (self.0)()
    }
}

impl<T> MatrixBase<Box<[T]>> {
    /// Construct a new Matrix initialized with the contents of the generator.
    ///
    /// Elements are initialized in memory order.
    pub fn new<U>(mut generator: U, nrows: usize, ncols: usize) -> Self
    where
        U: Generator<T>,
    {
        let data: Box<[T]> = (0..nrows * ncols).map(|_| generator.generate()).collect();
        debug_assert_eq!(data.len(), nrows * ncols);
        Self { data, nrows, ncols }
    }
}

impl<T> MatrixBase<T>
where
    T: DenseData,
{
    /// Try to construct a `MatrixBase` over the provided base. If the size of the base
    /// is incorrect, return a `TryFromError` containing the base.
    ///
    /// The length of the base must be equal to `nrows * ncols`.
    pub fn try_from(data: T, nrows: usize, ncols: usize) -> Result<Self, TryFromError<T>> {
        let len = data.as_slice().len();
        if len != nrows * ncols {
            Err(TryFromError { data, nrows, ncols })
        } else {
            Ok(Self { data, nrows, ncols })
        }
    }

    /// Return the number of columns in the matrix.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Return the number of rows in the matrix.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Return the underlying data as a slice.
    pub fn as_slice(&self) -> &[T::Elem] {
        self.data.as_slice()
    }

    /// Return the underlying data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T::Elem]
    where
        T: MutDenseData,
    {
        self.data.as_mut_slice()
    }

    /// Return row `row` as a slice.
    ///
    /// # Panic
    ///
    /// Panics if `row >= self.nrows()`.
    pub fn row(&self, row: usize) -> &[T::Elem] {
        assert!(
            row < self.nrows(),
            "tried to access row {row} of a matrix with {} rows",
            self.nrows()
        );

        // SAFETY: `row` is in-bounds.
        unsafe { self.get_row_unchecked(row) }
    }

    /// Construct a new `MatrixBase` over the raw data.
    ///
    /// The returned `MatrixBase` will only have a single row with contents equal to `data`.
    pub fn row_vector(data: T) -> Self {
        let ncols = data.as_slice().len();
        Self {
            data,
            nrows: 1,
            ncols,
        }
    }
    /// Return row `row` if `row < self.nrows()`. Otherwise, return `None`.
    pub fn get_row(&self, row: usize) -> Option<&[T::Elem]> {
        if row < self.nrows() {
            // SAFETY: `row` is in-bounds.
            Some(unsafe { self.get_row_unchecked(row) })
        } else {
            None
        }
    }

    /// Returns the requested row without boundschecking.
    ///
    /// # Safety
    ///
    /// The following conditions must hold to avoid undefined behavior:
    /// * `row < self.nrows()`.
    pub unsafe fn get_row_unchecked(&self, row: usize) -> &[T::Elem] {
        debug_assert!(row < self.nrows);
        let ncols = self.ncols;
        let start = row * ncols;

        debug_assert!(start + ncols <= self.as_slice().len());
        // SAFETY: The idempotency requirement of `as_slice` and our audited constructors
        // mean that `self.as_slice()` has a length of `self.nrows * self.ncols`.
        //
        // Therefore, this access is in-bounds.
        unsafe { self.as_slice().get_unchecked(start..start + ncols) }
    }

    /// Return row `row` as a mutable slice.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.nrows()`.
    pub fn row_mut(&mut self, row: usize) -> &mut [T::Elem]
    where
        T: MutDenseData,
    {
        assert!(
            row < self.nrows(),
            "tried to access row {row} of a matrix with {} rows",
            self.nrows()
        );

        // SAFETY: `row` is in-bounds.
        unsafe { self.get_row_unchecked_mut(row) }
    }

    /// Returns the requested row without boundschecking.
    ///
    /// # Safety
    ///
    /// The following conditions must hold to avoid undefined behavior:
    /// * `row < self.nrows()`.
    pub unsafe fn get_row_unchecked_mut(&mut self, row: usize) -> &mut [T::Elem]
    where
        T: MutDenseData,
    {
        debug_assert!(row < self.nrows);
        let ncols = self.ncols;
        let start = row * ncols;

        debug_assert!(start + ncols <= self.as_slice().len());
        // SAFETY: The idempotency requirement of `as_mut_slice` and our audited constructors
        // mean that `self.as_mut_slice()` has a length of `self.nrows * self.ncols`.
        //
        // Therefore, this access is in-bounds.
        unsafe {
            self.data
                .as_mut_slice()
                .get_unchecked_mut(start..start + ncols)
        }
    }

    /// Return a iterator over all rows in the matrix.
    ///
    /// Rows are yielded sequentially beginning with row 0.
    pub fn row_iter(&self) -> impl ExactSizeIterator<Item = &[T::Elem]> {
        self.data.as_slice().chunks_exact(self.ncols())
    }

    /// Return a mutable iterator over all rows in the matrix.
    ///
    /// Rows are yielded sequentially beginning with row 0.
    pub fn row_iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut [T::Elem]>
    where
        T: MutDenseData,
    {
        let ncols = self.ncols();
        self.data.as_mut_slice().chunks_exact_mut(ncols)
    }

    /// Return an iterator that divides the matrix into sub-matrices with (up to)
    /// `batchsize` rows with `self.ncols()` columns.
    ///
    /// It is possible for yielded sub-matrices to have fewer than `batchsize` rows if the
    /// number of rows in the parent matrix is not evenly divisible by `batchsize`.
    ///
    /// # Panics
    ///
    /// Panics if `batchsize = 0`.
    pub fn window_iter(&self, batchsize: usize) -> impl Iterator<Item = MatrixView<'_, T::Elem>>
    where
        T::Elem: Sync,
    {
        assert!(batchsize != 0, "window_iter batchsize cannot be zero");
        let ncols = self.ncols();
        self.data
            .as_slice()
            .chunks(ncols * batchsize)
            .map(move |data| {
                let blobsize = data.len();
                let nrows = blobsize / ncols;
                assert_eq!(blobsize % ncols, 0);
                MatrixView { data, nrows, ncols }
            })
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
    pub fn par_window_iter(
        &self,
        batchsize: usize,
    ) -> impl IndexedParallelIterator<Item = MatrixView<'_, T::Elem>>
    where
        T::Elem: Sync,
    {
        assert!(batchsize != 0, "par_window_iter batchsize cannot be zero");
        let ncols = self.ncols();
        self.data
            .as_slice()
            .par_chunks(ncols * batchsize)
            .map(move |data| {
                let blobsize = data.len();
                let nrows = blobsize / ncols;
                assert_eq!(blobsize % ncols, 0);
                MatrixView { data, nrows, ncols }
            })
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
    pub fn par_window_iter_mut(
        &mut self,
        batchsize: usize,
    ) -> impl IndexedParallelIterator<Item = MutMatrixView<'_, T::Elem>>
    where
        T: MutDenseData,
        T::Elem: Send,
    {
        assert!(
            batchsize != 0,
            "par_window_iter_mut batchsize cannot be zero"
        );
        let ncols = self.ncols();
        self.data
            .as_mut_slice()
            .par_chunks_mut(ncols * batchsize)
            .map(move |data| {
                let blobsize = data.len();
                let nrows = blobsize / ncols;
                assert_eq!(blobsize % ncols, 0);
                MutMatrixView { data, nrows, ncols }
            })
    }

    /// Return a parallel iterator over the rows of the matrix.
    #[cfg(feature = "rayon")]
    pub fn par_row_iter(&self) -> impl IndexedParallelIterator<Item = &[T::Elem]>
    where
        T::Elem: Sync,
    {
        self.as_slice().par_chunks_exact(self.ncols())
    }

    /// Return a parallel iterator over the rows of the matrix.
    #[cfg(feature = "rayon")]
    pub fn par_row_iter_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [T::Elem]>
    where
        T: MutDenseData,
        T::Elem: Send,
    {
        let ncols = self.ncols();
        self.as_mut_slice().par_chunks_exact_mut(ncols)
    }

    /// Consume the matrix, returning the inner representation.
    ///
    /// This loses the information about the number of rows and columnts.
    pub fn into_inner(self) -> T {
        self.data
    }

    /// Return a view over the matrix.
    pub fn as_view(&self) -> MatrixView<'_, T::Elem> {
        MatrixBase {
            data: self.as_slice(),
            nrows: self.nrows(),
            ncols: self.ncols(),
        }
    }

    /// Return a mutable view over the matrix.
    pub fn as_mut_view(&mut self) -> MutMatrixView<'_, T::Elem>
    where
        T: MutDenseData,
    {
        let nrows = self.nrows();
        let ncols = self.ncols();
        MatrixBase {
            data: self.as_mut_slice(),
            nrows,
            ncols,
        }
    }

    /// Return a pointer to the base of the matrix.
    pub fn as_ptr(&self) -> *const T::Elem {
        self.as_slice().as_ptr()
    }

    /// Return a pointer to the base of the matrix.
    pub fn as_mut_ptr(&mut self) -> *mut T::Elem
    where
        T: MutDenseData,
    {
        self.as_mut_slice().as_mut_ptr()
    }

    /// Returns a reference to an element without boundschecking.
    ///
    /// # Safety
    ///
    /// The following conditions must hold to avoid undefined behavior:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> &T::Elem {
        debug_assert!(row < self.nrows);
        debug_assert!(col < self.ncols);
        self.as_slice().get_unchecked(row * self.ncols + col)
    }

    /// Returns a mutable reference to an element without boundschecking.
    ///
    /// # Safety
    ///
    /// The following conditions must hold to avoid undefined behavior:
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    pub unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut T::Elem
    where
        T: MutDenseData,
    {
        let ncols = self.ncols;
        debug_assert!(row < self.nrows);
        debug_assert!(col < self.ncols);
        self.as_mut_slice().get_unchecked_mut(row * ncols + col)
    }

    pub fn to_owned(&self) -> Matrix<T::Elem>
    where
        T::Elem: Clone,
    {
        Matrix {
            data: self.data.as_slice().into(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

/// Represents an owning, 2-dimensional view of a contiguous block of memory,
/// interpreted as a matrix in row-major order.
pub type Matrix<T> = MatrixBase<Box<[T]>>;

/// Represents a non-owning, 2-dimensional view of a contiguous block of memory,
/// interpreted as a matrix in row-major order.
///
/// This type is useful for functions that need to read matrix data without taking ownership.
/// By accepting a `MatrixView`, such functions can operate on both owned matrices (by converting them
/// to a `MatrixView`) and existing non-owning views.
pub type MatrixView<'a, T> = MatrixBase<&'a [T]>;

/// Represents a mutable non-owning, 2-dimensional view of a contiguous block of memory,
/// interpreted as a matrix in row-major order.
///
/// This type is useful for functions that need to modify matrix data without taking ownership.
/// By accepting a `MutMatrixView`, such functions can operate on both owned matrices (by converting them
/// to a `MutMatrixView`) and existing non-owning mutable views.
pub type MutMatrixView<'a, T> = MatrixBase<&'a mut [T]>;

/// Allow matrix views to be converted directly to slices.
impl<'a, T> From<MatrixView<'a, T>> for &'a [T] {
    fn from(view: MatrixView<'a, T>) -> Self {
        view.data
    }
}

/// Allow mutable matrix views to be converted directly to slices.
impl<'a, T> From<MutMatrixView<'a, T>> for &'a [T] {
    fn from(view: MutMatrixView<'a, T>) -> Self {
        view.data
    }
}

/// Return a reference to the item at entry `(row, col)` in the matrix.
///
/// # Panics
///
/// Panics if `row >= self.nrows()` or `col >= self.ncols()`.
impl<T> Index<(usize, usize)> for MatrixBase<T>
where
    T: DenseData,
{
    type Output = T::Elem;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
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

        // SAFETY: We have checked that `row` and `col` are in-bounds.
        unsafe { self.get_unchecked(row, col) }
    }
}

/// Return a mutable reference to the item at entry `(row, col)` in the matrix.
///
/// # Panics
///
/// Panics if `row >= self.nrows()` or `col >= self.ncols()`.
impl<T> IndexMut<(usize, usize)> for MatrixBase<T>
where
    T: MutDenseData,
{
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
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

        // SAFETY: We have checked that `row` and `col` are in-bounds.
        unsafe { self.get_unchecked_mut(row, col) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_format;

    /// This function is only callable with copyable types.
    ///
    /// This lets us test for types we expect to be `Copy`.
    fn is_copyable<T: Copy>(_x: T) -> bool {
        true
    }

    /// Test the that provided representation yields a slice with the expected base pointer
    /// and length.
    fn test_dense_data_repr<T, Repr>(
        ptr: *const T,
        len: usize,
        repr: Repr,
        context: &dyn std::fmt::Display,
    ) where
        T: Copy,
        Repr: DenseData<Elem = T>,
    {
        let retrieved = repr.as_slice();
        assert_eq!(retrieved.len(), len, "{}", context);
        assert_eq!(retrieved.as_ptr(), ptr, "{}", context);
    }

    /// Set the underlying data for the provided representation to the following:
    ///
    /// [base, base + increment, base + increment + increment, ...]
    fn set_mut_dense_data_repr<T, Repr>(repr: &mut Repr, base: T, increment: T)
    where
        T: Copy + std::ops::Add<Output = T>,
        Repr: DenseData<Elem = T> + MutDenseData,
    {
        let slice = repr.as_mut_slice();
        for i in 0..slice.len() {
            if i == 0 {
                slice[i] = base;
            } else {
                slice[i] = slice[i - 1] + increment;
            }
        }
    }

    #[test]
    fn slice_implements_dense_data_repr() {
        for len in 0..10 {
            let context = lazy_format!("len = {}", len);
            let data: Vec<f32> = vec![0.0; len];
            let slice = data.as_slice();
            test_dense_data_repr(slice.as_ptr(), slice.len(), slice, &context);
        }
    }

    #[test]
    fn mut_slice_mplements_dense_data_repr() {
        for len in 0..10 {
            let context = lazy_format!("len = {}", len);
            let mut data: Vec<f32> = vec![0.0; len];
            let slice = data.as_mut_slice();

            let ptr = slice.as_ptr();
            let len = slice.len();
            test_dense_data_repr(ptr, len, slice, &context);
        }
    }

    #[test]
    fn mut_slice_implements_mut_dense_data_repr() {
        for len in 0..10 {
            let context = lazy_format!("len = {}", len);
            let mut data: Vec<f32> = vec![0.0; len];
            let mut slice = data.as_mut_slice();

            let base = 2.0;
            let increment = 1.0;
            set_mut_dense_data_repr(&mut slice, base, increment);

            for (i, &v) in slice.iter().enumerate() {
                let context = lazy_format!("entry {}, {}", i, context);
                assert_eq!(v, base + increment * (i as f32), "{}", context);
            }
        }
    }

    /////////////////
    // Matrix View //
    /////////////////

    #[test]
    fn try_from_error_misc() {
        let x = TryFromError::<&[f32]> {
            data: &[],
            nrows: 1,
            ncols: 2,
        };

        let debug = format!("{:?}", x);
        println!("debug = {}", debug);
        assert!(debug.contains("TryFromError"));
        assert!(debug.contains("data_len: 0"));
        assert!(debug.contains("nrows: 1"));
        assert!(debug.contains("ncols: 2"));
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
    fn test_basic_indexing_parallel(m: MatrixView<'_, usize>) {
        // Par window iters.
        let batchsize = 2;
        m.par_window_iter(batchsize)
            .enumerate()
            .for_each(|(i, submatrix)| {
                assert_eq!(submatrix.nrows(), batchsize);
                assert_eq!(submatrix.ncols(), m.ncols());

                // Make sure we are in the correct window of the original matrix.
                let base = i * batchsize;
                assert_eq!(submatrix[(0, 0)], base);
                assert_eq!(submatrix[(0, 1)], base + 1);
                assert_eq!(submatrix[(0, 2)], base + 2);

                assert_eq!(submatrix[(1, 0)], base + 1);
                assert_eq!(submatrix[(1, 1)], base + 2);
                assert_eq!(submatrix[(1, 2)], base + 3);
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
                    assert_eq!(submatrix[(0, 0)], 0);
                    assert_eq!(submatrix[(0, 1)], 1);
                    assert_eq!(submatrix[(0, 2)], 2);

                    assert_eq!(submatrix[(1, 0)], 1);
                    assert_eq!(submatrix[(1, 1)], 2);
                    assert_eq!(submatrix[(1, 2)], 3);

                    assert_eq!(submatrix[(2, 0)], 2);
                    assert_eq!(submatrix[(2, 1)], 3);
                    assert_eq!(submatrix[(2, 2)], 4);
                } else {
                    assert_eq!(submatrix.nrows(), 1);
                    assert_eq!(submatrix.ncols(), m.ncols());

                    // Check indexing
                    assert_eq!(submatrix[(0, 0)], 3);
                    assert_eq!(submatrix[(0, 1)], 4);
                    assert_eq!(submatrix[(0, 2)], 5);
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

    fn test_basic_indexing<T>(m: &MatrixBase<T>)
    where
        T: DenseData<Elem = usize> + Sync,
    {
        assert_eq!(m.nrows(), 4);
        assert_eq!(m.ncols(), 3);

        // Basic indexing
        assert_eq!(m[(0, 0)], 0);
        assert_eq!(m[(0, 1)], 1);
        assert_eq!(m[(0, 2)], 2);

        assert_eq!(m[(1, 0)], 1);
        assert_eq!(m[(1, 1)], 2);
        assert_eq!(m[(1, 2)], 3);

        assert_eq!(m[(2, 0)], 2);
        assert_eq!(m[(2, 1)], 3);
        assert_eq!(m[(2, 2)], 4);

        assert_eq!(m[(3, 0)], 3);
        assert_eq!(m[(3, 1)], 4);
        assert_eq!(m[(3, 2)], 5);

        // Row indexing.
        assert_eq!(m.row(0), &[0, 1, 2]);
        assert_eq!(m.row(1), &[1, 2, 3]);
        assert_eq!(m.row(2), &[2, 3, 4]);
        assert_eq!(m.row(3), &[3, 4, 5]);

        let rows: Vec<Vec<usize>> = m.row_iter().map(|x| x.to_vec()).collect();
        assert_eq!(m.row(0), &rows[0]);
        assert_eq!(m.row(1), &rows[1]);
        assert_eq!(m.row(2), &rows[2]);
        assert_eq!(m.row(3), &rows[3]);

        // Window Iters.
        let batchsize = 2;
        m.window_iter(batchsize)
            .enumerate()
            .for_each(|(i, submatrix)| {
                assert_eq!(submatrix.nrows(), batchsize);
                assert_eq!(submatrix.ncols(), m.ncols());

                // Make sure we are in the correct window of the original matrix.
                let base = i * batchsize;
                assert_eq!(submatrix[(0, 0)], base);
                assert_eq!(submatrix[(0, 1)], base + 1);
                assert_eq!(submatrix[(0, 2)], base + 2);

                assert_eq!(submatrix[(1, 0)], base + 1);
                assert_eq!(submatrix[(1, 1)], base + 2);
                assert_eq!(submatrix[(1, 2)], base + 3);
            });

        // Try again, but with a batch size of 3 to ensure that we correctly handle cases
        // where the last block is under-sized.
        let batchsize = 3;
        m.window_iter(batchsize)
            .enumerate()
            .for_each(|(i, submatrix)| {
                if i == 0 {
                    assert_eq!(submatrix.nrows(), batchsize);
                    assert_eq!(submatrix.ncols(), m.ncols());

                    // Check indexing
                    assert_eq!(submatrix[(0, 0)], 0);
                    assert_eq!(submatrix[(0, 1)], 1);
                    assert_eq!(submatrix[(0, 2)], 2);

                    assert_eq!(submatrix[(1, 0)], 1);
                    assert_eq!(submatrix[(1, 1)], 2);
                    assert_eq!(submatrix[(1, 2)], 3);

                    assert_eq!(submatrix[(2, 0)], 2);
                    assert_eq!(submatrix[(2, 1)], 3);
                    assert_eq!(submatrix[(2, 2)], 4);
                } else {
                    assert_eq!(submatrix.nrows(), 1);
                    assert_eq!(submatrix.ncols(), m.ncols());

                    // Check indexing
                    assert_eq!(submatrix[(0, 0)], 3);
                    assert_eq!(submatrix[(0, 1)], 4);
                    assert_eq!(submatrix[(0, 2)], 5);
                }
            });

        #[cfg(all(not(miri), feature = "rayon"))]
        test_basic_indexing_parallel(m.as_view());
    }

    #[test]
    fn matrix_happy_path() {
        let data = make_test_matrix();
        let m = Matrix::try_from(data.into(), 4, 3).unwrap();
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

        let m = Matrix::try_from(data.into(), 5, 4);
        assert!(m.is_err());
        let err = m.unwrap_err();
        assert_eq!(
            err.to_string(),
            "tried to construct a matrix view with 5 rows and 4 columns over a slice of length 12"
        );

        // Make sure that we can retrieve the original allocation from the interior.
        let data = err.into_inner();
        assert_eq!(data.as_ptr(), ptr);
        assert_eq!(data.len(), len);

        let m = MatrixView::try_from(&data, 5, 4);
        assert!(m.is_err());
        assert_eq!(
            m.unwrap_err().to_string(),
            "tried to construct a matrix view with 5 rows and 4 columns over a slice of length 12"
        );
    }

    #[test]
    fn matrix_mut_view() {
        let mut m = Matrix::<usize>::new(0, 4, 3);
        assert_eq!(m.nrows(), 4);
        assert_eq!(m.ncols(), 3);
        assert!(m.as_slice().iter().all(|&i| i == 0));
        let ptr = m.as_ptr();
        let mut_ptr = m.as_mut_ptr();
        assert_eq!(ptr, mut_ptr);

        let mut view = m.as_mut_view();
        assert_eq!(view.nrows(), 4);
        assert_eq!(view.ncols(), 3);
        assert_eq!(view.as_ptr(), ptr);
        assert_eq!(view.as_mut_ptr(), mut_ptr);

        // Construct the test matrix manually.
        for i in 0..view.nrows() {
            for j in 0..view.ncols() {
                view[(i, j)] = i + j;
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
        let m = MatrixView::try_from(data.as_slice(), 0, 10).unwrap();
        assert_eq!(m.nrows(), 0);
        assert_eq!(m.ncols(), 10);

        // Non-zero rows, but zero columns.
        let m = MatrixView::try_from(data.as_slice(), 3, 0).unwrap();
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 0);
        let empty: &[usize] = &[];
        assert_eq!(m.row(0), empty);
        assert_eq!(m.row(1), empty);
        assert_eq!(m.row(2), empty);

        // Zero rows and columns.
        let m = MatrixView::try_from(data.as_slice(), 0, 0).unwrap();
        assert_eq!(m.nrows(), 0);
        assert_eq!(m.ncols(), 0);
    }

    #[test]
    fn matrix_view_construction_elementwise() {
        let mut m = Matrix::<usize>::new(0, 4, 3);

        // Construct the test matrix manually.
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                m[(i, j)] = i + j;
            }
        }
        test_basic_indexing(&m);
    }

    #[test]
    fn matrix_construction_by_row() {
        let mut m = Matrix::<usize>::new(0, 4, 3);
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
        let mut m = Matrix::<usize>::new(0, 4, 3);
        assert!(m.as_slice().iter().all(|i| *i == 0));

        let ncols = m.ncols();
        m.row_iter_mut().enumerate().for_each(|(i, row)| {
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
        let mut m = Matrix::<usize>::new(0, 4, 3);
        assert!(m.as_slice().iter().all(|i| *i == 0));

        let ncols = m.ncols();
        for batchsize in 1..=4 {
            m.par_window_iter_mut(batchsize)
                .enumerate()
                .for_each(|(i, mut submatrix)| {
                    let base = i * batchsize;
                    submatrix.row_iter_mut().enumerate().for_each(|(j, row)| {
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
        let initializer = Init(|| {
            let value = (i % ncols) + (i / ncols);
            i += 1;
            value
        });

        let m = Matrix::new(initializer, 4, 3);
        test_basic_indexing(&m);
    }

    // Panics
    #[test]
    #[should_panic(expected = "tried to access row 3 of a matrix with 3 rows")]
    fn test_get_row_panics() {
        let m = Matrix::<usize>::new(0, 3, 7);
        m.row(3);
    }

    #[test]
    #[should_panic(expected = "tried to access row 3 of a matrix with 3 rows")]
    fn test_get_row_mut_panics() {
        let mut m = Matrix::<usize>::new(0, 3, 7);
        m.row_mut(3);
    }

    #[test]
    #[should_panic(expected = "row 3 is out of bounds (max: 3)")]
    fn test_index_panics_row() {
        let m = Matrix::<usize>::new(0, 3, 7);
        let _ = m[(3, 2)];
    }

    #[test]
    #[should_panic(expected = "col 7 is out of bounds (max: 7)")]
    fn test_index_panics_col() {
        let m = Matrix::<usize>::new(0, 3, 7);
        let _ = m[(2, 7)];
    }

    #[test]
    #[should_panic(expected = "row 3 is out of bounds (max: 3)")]
    fn test_index_mut_panics_row() {
        let mut m = Matrix::<usize>::new(0, 3, 7);
        m[(3, 2)] = 1;
    }

    #[test]
    #[should_panic(expected = "col 7 is out of bounds (max: 7)")]
    fn test_index_mut_panics_col() {
        let mut m = Matrix::<usize>::new(0, 3, 7);
        m[(2, 7)] = 1;
    }

    #[test]
    #[cfg(feature = "rayon")]
    #[should_panic(expected = "par_window_iter batchsize cannot be zero")]
    fn test_par_window_iter_panics() {
        let m = Matrix::<usize>::new(0, 4, 4);
        let _ = m.par_window_iter(0);
    }

    #[test]
    #[cfg(feature = "rayon")]
    #[should_panic(expected = "par_window_iter_mut batchsize cannot be zero")]
    fn test_par_window_iter_mut_panics() {
        let mut m = Matrix::<usize>::new(0, 4, 4);
        let _ = m.par_window_iter_mut(0);
    }

    // Additional tests for better coverage

    #[test]
    fn test_box_slice_dense_data_impls() {
        // Test Box<[T]> implementations
        let data: Box<[f32]> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into();
        let ptr = data.as_ptr();
        let len = data.len();

        // Test DenseData impl for Box<[T]>
        test_dense_data_repr(ptr, len, data, &lazy_format!("Box<[T]> DenseData"));

        // Test MutDenseData impl for Box<[T]>
        let mut data: Box<[f32]> = vec![0.0; 6].into();
        set_mut_dense_data_repr(&mut data, 1.0, 2.0);
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(
                v,
                1.0 + 2.0 * (i as f32),
                "Box<[T]> MutDenseData at index {}",
                i
            );
        }
    }

    #[test]
    fn test_try_from_error_light() {
        let data = vec![1, 2, 3];
        let err = MatrixView::try_from(data.as_slice(), 2, 3).unwrap_err();

        // Test as_static method
        let static_err = err.as_static();
        assert_eq!(static_err.len, 3);
        assert_eq!(static_err.nrows, 2);
        assert_eq!(static_err.ncols, 3);

        // Test Display for TryFromErrorLight
        let display_msg = format!("{}", static_err);
        assert!(display_msg.contains("tried to construct a matrix view with 2 rows and 3 columns"));
        assert!(display_msg.contains("slice of length 3"));

        // Test into_inner method
        let recovered_data = err.into_inner();
        assert_eq!(recovered_data, data.as_slice());
    }

    #[test]
    fn test_get_row_optional() {
        let data = make_test_matrix();
        let m = MatrixView::try_from(data.as_slice(), 4, 3).unwrap();

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
        let mut m = Matrix::try_from(data.into(), 4, 3).unwrap();

        // Safety: derives from known size of matrix and access element ids
        unsafe {
            assert_eq!(*m.get_unchecked(0, 0), 0);
            assert_eq!(*m.get_unchecked(1, 2), 3);
            assert_eq!(*m.get_unchecked(3, 1), 4);
        }

        // Safety: derives from known size of matrix and access element ids
        unsafe {
            *m.get_unchecked_mut(0, 0) = 100;
            *m.get_unchecked_mut(1, 2) = 200;
        }

        assert_eq!(m[(0, 0)], 100);
        assert_eq!(m[(1, 2)], 200);

        // Safety: derives from known size of matrix and access element ids
        unsafe {
            let row0 = m.get_row_unchecked(0);
            assert_eq!(row0[0], 100);
            assert_eq!(row0[1], 1);
            assert_eq!(row0[2], 2);
        }

        // Safety: derives from known size of matrix and access element ids
        unsafe {
            let row1 = m.get_row_unchecked_mut(1);
            row1[0] = 300;
        }

        assert_eq!(m[(1, 0)], 300);
    }

    #[test]
    fn test_to_owned() {
        let data = make_test_matrix();
        let view = MatrixView::try_from(data.as_slice(), 4, 3).unwrap();

        // Test to_owned creates a proper clone
        let owned = view.to_owned();
        assert_eq!(owned.nrows(), view.nrows());
        assert_eq!(owned.ncols(), view.ncols());
        assert_eq!(owned.as_slice(), view.as_slice());

        // Verify it's actually owned (different memory location)
        assert_ne!(owned.as_ptr(), view.as_ptr());

        // Test the owned matrix works properly
        test_basic_indexing(&owned);
    }

    #[test]
    fn test_generator_trait_impls() {
        // Test Generator impl for T where T: Clone
        let mut gen = 42i32;
        assert_eq!(gen.generate(), 42);
        assert_eq!(gen.generate(), 42); // Should be same value since it's cloned

        // Test Generator impl for Init<F>
        let mut counter = 0;
        let mut gen = Init(|| {
            counter += 1;
            counter
        });
        assert_eq!(gen.generate(), 1);
        assert_eq!(gen.generate(), 2);
        assert_eq!(gen.generate(), 3);
    }

    #[test]
    fn test_matrix_from_conversions() {
        let data = make_test_matrix();
        let m = Matrix::try_from(data.into(), 4, 3).unwrap();

        // Test MatrixView to slice conversion
        let view = m.as_view();
        let slice: &[usize] = view.into();
        assert_eq!(slice.len(), 12);
        assert_eq!(slice[0], 0);
        assert_eq!(slice[11], 5);

        // Test MutMatrixView to slice conversion
        let data2 = make_test_matrix();
        let mut m2 = Matrix::try_from(data2.into(), 4, 3).unwrap();
        let mut_view = m2.as_mut_view();
        let slice2: &[usize] = mut_view.into();
        assert_eq!(slice2.len(), 12);
        assert_eq!(slice2[0], 0);
        assert_eq!(slice2[11], 5);
    }

    #[test]
    fn test_matrix_construction_edge_cases() {
        // Test 1x1 matrix
        let m = Matrix::new(42, 1, 1);
        assert_eq!(m.nrows(), 1);
        assert_eq!(m.ncols(), 1);
        assert_eq!(m[(0, 0)], 42);

        // Test single row matrix
        let m = Matrix::new(7, 1, 5);
        assert_eq!(m.nrows(), 1);
        assert_eq!(m.ncols(), 5);
        assert!(m.as_slice().iter().all(|&x| x == 7));

        // Test single column matrix
        let m = Matrix::new(9, 5, 1);
        assert_eq!(m.nrows(), 5);
        assert_eq!(m.ncols(), 1);
        assert!(m.as_slice().iter().all(|&x| x == 9));
    }

    #[test]
    fn test_matrix_view_edge_cases_with_data() {
        // Test matrix with actual data for edge cases
        let data = vec![10, 20];

        // 2x1 matrix
        let m = MatrixView::try_from(data.as_slice(), 2, 1).unwrap();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 1);
        assert_eq!(m[(0, 0)], 10);
        assert_eq!(m[(1, 0)], 20);
        assert_eq!(m.row(0), &[10]);
        assert_eq!(m.row(1), &[20]);

        // 1x2 matrix
        let m = MatrixView::try_from(data.as_slice(), 1, 2).unwrap();
        assert_eq!(m.nrows(), 1);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m[(0, 0)], 10);
        assert_eq!(m[(0, 1)], 20);
        assert_eq!(m.row(0), &[10, 20]);
    }

    #[test]
    #[cfg(all(not(miri), feature = "rayon"))]
    fn test_parallel_methods_edge_cases() {
        let data = make_test_matrix();
        let m = Matrix::try_from(data.into(), 4, 3).unwrap();

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
        let mut m2 = Matrix::new(0, 4, 3);

        // Use par_row_iter_mut to set values
        m2.par_row_iter_mut().enumerate().for_each(|(i, row)| {
            for (j, elem) in row.iter_mut().enumerate() {
                *elem = i + j;
            }
        });
        test_basic_indexing(&m2);

        // Test par_window_iter_mut with larger batchsize
        let mut m3 = Matrix::new(0, 4, 3);
        m3.par_window_iter_mut(10)
            .enumerate()
            .for_each(|(_, mut window)| {
                window.row_iter_mut().enumerate().for_each(|(i, row)| {
                    for (j, elem) in row.iter_mut().enumerate() {
                        *elem = i + j;
                    }
                });
            });
        test_basic_indexing(&m3);
    }

    #[test]
    fn test_matrix_pointers() {
        let mut m = Matrix::new(42, 3, 4);

        // Test as_ptr and as_mut_ptr return the same address
        let const_ptr = m.as_ptr();
        let mut_ptr = m.as_mut_ptr();
        assert_eq!(const_ptr, mut_ptr as *const _);

        // Test that view pointers match original
        let view = m.as_view();
        assert_eq!(view.as_ptr(), const_ptr);

        let mut mut_view = m.as_mut_view();
        assert_eq!(mut_view.as_ptr(), const_ptr);
        assert_eq!(mut_view.as_mut_ptr(), mut_ptr);
    }

    #[test]
    fn test_matrix_iteration_empty_cases() {
        // Test construction of empty matrices (we don't iterate over 0x0 matrices
        // since chunks_exact requires non-zero chunk size)
        let empty_data: Vec<i32> = vec![];

        // Matrix with 0 rows but non-zero cols can be constructed
        let _empty_matrix = MatrixView::try_from(empty_data.as_slice(), 0, 5).unwrap();

        // Test with actual single row to verify iterator works normally
        let data = vec![1, 2, 3];
        let single_row = MatrixView::try_from(data.as_slice(), 1, 3).unwrap();
        let rows: Vec<_> = single_row.row_iter().collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], &[1, 2, 3]);

        // Test iteration over matrix with multiple rows but single column
        let data = vec![1, 2, 3];
        let single_col = MatrixView::try_from(data.as_slice(), 3, 1).unwrap();
        let rows: Vec<_> = single_col.row_iter().collect();
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
        let m = Matrix::new(Init(|| counter.fetch_add(1, Ordering::SeqCst)), 2, 3);

        // Should be filled in memory order
        assert_eq!(m[(0, 0)], 0);
        assert_eq!(m[(0, 1)], 1);
        assert_eq!(m[(0, 2)], 2);
        assert_eq!(m[(1, 0)], 3);
        assert_eq!(m[(1, 1)], 4);
        assert_eq!(m[(1, 2)], 5);
    }

    #[test]
    fn test_debug_error_formatting() {
        // Test Debug implementation for TryFromError
        let data = vec![1, 2, 3];
        let err = Matrix::try_from(data.into(), 2, 3).unwrap_err();

        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("TryFromError"));
        assert!(debug_str.contains("data_len: 3"));
        assert!(debug_str.contains("nrows: 2"));
        assert!(debug_str.contains("ncols: 3"));

        // Ensure Debug doesn't require T: Debug by using a non-Debug type
        #[derive(Clone, Debug)]
        struct NonDebug(#[allow(dead_code)] i32);

        let non_debug_data: Box<[NonDebug]> = vec![NonDebug(1), NonDebug(2)].into();
        let non_debug_err = Matrix::try_from(non_debug_data, 1, 3).unwrap_err();
        let debug_str = format!("{:?}", non_debug_err);
        assert!(debug_str.contains("TryFromError"));
    }

    // Comprehensive tests for rayon-specific functionality

    #[test]
    #[cfg(feature = "rayon")]
    fn test_par_window_iter_comprehensive() {
        use rayon::prelude::*;

        // Create a larger test matrix for more comprehensive testing
        let data: Vec<usize> = (0..24).collect(); // 6x4 matrix
        let m = MatrixView::try_from(data.as_slice(), 6, 4).unwrap();

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
                for (row_idx, row) in window.row_iter().enumerate() {
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

                    let mut m = Matrix::new(0usize, nrows, ncols);

                    // Use par_window_iter_mut to fill matrix
                    m.par_window_iter_mut(batchsize).enumerate().for_each(
                        |(window_idx, mut window)| {
                            let base_row = window_idx * batchsize;
                            window
                                .row_iter_mut()
                                .enumerate()
                                .for_each(|(row_offset, row)| {
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
                                m[(row, col)],
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
        let m = MatrixView::try_from(data.as_slice(), nrows, ncols).unwrap();

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
        let mut m = Matrix::new(0u32, nrows, ncols);

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
                assert_eq!(m[(row, col)], expected, "pos ({}, {})", row, col);
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
                assert_eq!(m[(row, col)], expected, "doubled pos ({}, {})", row, col);
            }
        }
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_parallel_iterators_with_single_dimensions() {
        use rayon::prelude::*;

        // Test single row matrix
        let data = vec![1, 2, 3, 4, 5];
        let single_row = MatrixView::try_from(data.as_slice(), 1, 5).unwrap();

        let windows: Vec<_> = single_row.par_window_iter(1).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].nrows(), 1);
        assert_eq!(windows[0].ncols(), 5);

        let rows: Vec<_> = single_row.par_row_iter().collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], &[1, 2, 3, 4, 5]);

        // Test single column matrix
        let data = vec![1, 2, 3, 4, 5];
        let single_col = MatrixView::try_from(data.as_slice(), 5, 1).unwrap();

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
        let tiny = MatrixView::try_from(data.as_slice(), 1, 1).unwrap();

        let windows: Vec<_> = tiny.par_window_iter(1).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0][(0, 0)], 42);

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
        let m = MatrixView::try_from(data.as_slice(), 6, 5).unwrap();

        // Test window indexing works correctly
        m.par_window_iter(2)
            .enumerate()
            .for_each(|(window_idx, window)| {
                for row_idx in 0..window.nrows() {
                    for col_idx in 0..window.ncols() {
                        let global_row = window_idx * 2 + row_idx;
                        let expected = global_row * 5 + col_idx;
                        assert_eq!(
                            window[(row_idx, col_idx)],
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
                        window[(row, col)],
                        "window {}, slice_idx {}",
                        window_idx,
                        slice_idx
                    );
                }
            });

        // Test window row iteration
        m.par_window_iter(2).for_each(|window| {
            let rows_via_iter: Vec<_> = window.row_iter().collect();
            assert_eq!(rows_via_iter.len(), window.nrows());

            for (row_idx, row) in rows_via_iter.iter().enumerate() {
                assert_eq!(row.len(), window.ncols());
                for (col_idx, &value) in row.iter().enumerate() {
                    assert_eq!(value, window[(row_idx, col_idx)]);
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
        let mut m = Matrix::new(0usize, nrows, ncols);

        // Test that parallel operations can be chained
        let work_counter = AtomicUsize::new(0);

        m.par_window_iter_mut(10)
            .enumerate()
            .for_each(|(window_idx, mut window)| {
                work_counter.fetch_add(1, Ordering::Relaxed);

                // Nested parallel operation within window
                window
                    .row_iter_mut()
                    .enumerate()
                    .for_each(|(row_offset, row)| {
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
                assert_eq!(m[(row, col)], row * ncols + col);
            }
        }

        // Test parallel reduction across windows
        let total_sum: usize = m
            .par_window_iter(15)
            .map(|window| {
                window
                    .row_iter()
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
        let m = MatrixView::try_from(data.as_slice(), 4, 5).unwrap();

        // This should compile because u64 is Sync
        let _: Vec<_> = m.par_window_iter(2).collect();
        let _: Vec<_> = m.par_row_iter().collect();

        // Test with mutable matrix
        let mut m = Matrix::new(0u64, 4, 5);

        // This should compile because u64 is Send
        m.par_window_iter_mut(2).for_each(|mut window| {
            window.row_iter_mut().for_each(|row| {
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
