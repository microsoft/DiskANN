/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt,
    ops::{Index, IndexMut},
};
use thiserror::Error;

use crate::views::{self, DenseData, MutDenseData};

/// A row-major strided matrix.
///
/// This is a generalization of the `MatrixBase` class as it does not mandate a dense
/// layout in memory.
///
/// ```text
///            |<------ cstride ----->|
///            |<-- ncols -->|
///            +-------------+
/// slice 0 -> | a0 a1 a2 a3 | a4 a5 a6     ^
/// slice 1 -> | b0 b1 b2 b3 | b4 b5 b6     |
/// slice 2 -> | c0 c1 c2 c3 | c4 c5 c6   nrows
/// slice 3 -> | d0 d1 d2 d3 | d4 d5 d6     |
/// slice 4 -> | e0 e1 e2 e3 | e4 e5 e6     |
/// slicf 5 -> | f0 f1 f2 f3 | f4 f5 f6     v
///            +-------------+
///                  ^
///                  |
///             StridedView
/// ```
///
/// This abstraction is useful when performing PQ related operations such as training or
/// compression as it provides a convenient abstraction for working with columnar subsets
/// of dense data in-place.
#[derive(Debug, Clone, Copy)]
pub struct StridedBase<T>
where
    T: DenseData,
{
    data: T,
    nrows: usize,
    ncols: usize,
    // The stride along the columns. This must be greater than or equal to `ncols`.
    cstride: usize,
}

/// Return the linear length of a slice underlying a `StridedBase` with the given parameters.
pub fn linear_length(nrows: usize, ncols: usize, cstride: usize) -> usize {
    (nrows.max(1) - 1) * cstride + nrows.min(1) * ncols
}

#[derive(Debug, Error)]
#[non_exhaustive]
#[error(
    "tried to construct a strided matrix with {nrows} rows and {ncols} cols and \
     column stride {cstride} over a slice of length {} (expected {})",
     len,
     linear_length(self.nrows, self.ncols, self.cstride)
)]
pub struct TryFromErrorLight {
    len: usize,
    nrows: usize,
    ncols: usize,
    cstride: usize,
}

#[derive(Error)]
#[non_exhaustive]
#[error(
    "tried to construct a strided matrix with {nrows} rows and {ncols} cols and \
     column stride {cstride} over a slice of length {} (expected {})",
     data.as_slice().len(),
     linear_length(self.nrows, self.ncols, self.cstride)
)]
pub struct TryFromError<T: views::DenseData> {
    data: T,
    nrows: usize,
    ncols: usize,
    cstride: usize,
}

// Manually implement `fmt::Debug` so we don't require `T::Debug`.
impl<T: DenseData> fmt::Debug for TryFromError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TryFromError")
            .field("data_len", &self.data.as_slice().len())
            .field("nrows", &self.nrows)
            .field("ncols", &self.ncols)
            .field("cstride", &self.cstride)
            .finish()
    }
}

impl<T: views::DenseData> TryFromError<T> {
    /// Consume the error and return the base data.
    pub fn into_inner(self) -> T {
        self.data
    }

    /// Drop the data portion of the error and return an equivalent error that is guaranteed
    /// to be `'static`.
    pub fn as_static(&self) -> TryFromErrorLight {
        TryFromErrorLight {
            len: self.data.as_slice().len(),
            nrows: self.nrows,
            ncols: self.ncols,
            cstride: self.cstride,
        }
    }
}

impl<'a, T> StridedBase<&'a [T]> {
    /// Construct a strided view over data slice, shrinking the slice as needed.
    ///
    /// Returns an error if `data` is shorter than the value returned by `linear_length`.
    ///
    /// # Panics
    ///
    /// * Panics if `cstride < ncols`.
    pub fn try_shrink_from(
        data: &'a [T],
        nrows: usize,
        ncols: usize,
        cstride: usize,
    ) -> Result<Self, TryFromError<&'a [T]>> {
        assert!(
            cstride >= ncols,
            "cstride must be greater than or equal to ncols"
        );
        let required_length = linear_length(nrows, ncols, cstride);
        match data.get(..required_length) {
            Some(data) => Ok(Self {
                data,
                nrows,
                ncols,
                cstride,
            }),
            None => Err(TryFromError {
                data,
                nrows,
                ncols,
                cstride,
            }),
        }
    }
}

impl<'a, T> StridedBase<&'a mut [T]> {
    /// Construct a strided view over data slice, shrinking the slice as needed.
    ///
    /// Returns an error if `data` is shorter than the value returned by `linear_length`.
    ///
    /// # Panics
    ///
    /// * Panics if `cstride < ncols`.
    pub fn try_shrink_from_mut(
        data: &'a mut [T],
        nrows: usize,
        ncols: usize,
        cstride: usize,
    ) -> Result<Self, TryFromError<&'a mut [T]>> {
        assert!(
            cstride >= ncols,
            "cstride must be greater than or equal to ncols"
        );
        let required_length = linear_length(nrows, ncols, cstride);
        if data.as_slice().len() >= required_length {
            Ok(Self {
                data: &mut data[..required_length],
                nrows,
                ncols,
                cstride,
            })
        } else {
            Err(TryFromError {
                data,
                nrows,
                ncols,
                cstride,
            })
        }
    }
}

impl<T> StridedBase<T>
where
    T: DenseData,
{
    /// Construct a strided view over data slice, shrinking the slice as needed.
    ///
    /// Returns an error if `data` is not equal to the expected length as determined
    /// by `linear_length`.
    ///
    /// # Panics
    ///
    /// * Panics if `cstride < ncols`.
    pub fn try_from(
        data: T,
        nrows: usize,
        ncols: usize,
        cstride: usize,
    ) -> Result<Self, TryFromError<T>> {
        assert!(
            cstride >= ncols,
            "cstride must be greater than or equal to ncols"
        );
        // This computation needs to be set up such that:
        // 1. When `nrows == 0`, the expected length is 0.
        // 2. We make a tight upper-bound on the expected length for the last row.
        let required_length = linear_length(nrows, ncols, cstride);
        if data.as_slice().len() == required_length {
            Ok(Self {
                data,
                nrows,
                ncols,
                cstride,
            })
        } else {
            Err(TryFromError {
                data,
                nrows,
                ncols,
                cstride,
            })
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

    /// Return the count of elements between the start of each row.
    pub fn cstride(&self) -> usize {
        self.cstride
    }

    /// Return the underlying data as a slice.
    ///
    /// # Note
    ///
    /// The underlying representation for a strided matrix is not necessarily dense.
    pub fn as_slice(&self) -> &[T::Elem] {
        self.data.as_slice()
    }

    /// Return the underlying data as a slice.
    ///
    /// # Note
    ///
    /// The underlying representation for a strided matrix is not necessarily dense.
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

    /// Returns the requested row without boundschecking.
    ///
    /// # Safety
    ///
    /// The following conditions must hold to avoid undefined behavior:
    /// * `row < self.nrows()`.
    pub unsafe fn get_row_unchecked(&self, row: usize) -> &[T::Elem] {
        debug_assert!(row < self.nrows);
        let cstride = self.cstride;
        let ncols = self.ncols;
        let start = row * cstride;

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
        let cstride = self.cstride;
        let ncols = self.ncols;
        let start = row * cstride;

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
    ///
    /// # Panics
    ///
    /// Panics if `self.ncols() == 0` (because the implementation does not work correctly
    /// in this case and it's too corner-case to bother fixing). This restriction may
    /// be lifted in the future.
    pub fn row_iter(&self) -> impl Iterator<Item = &[T::Elem]> {
        assert!(self.ncols() != 0);
        let ncols = self.ncols;

        self.data
            .as_slice()
            .chunks(self.cstride())
            .map(move |i| &i[..ncols])
    }

    /// Return a mutable iterator over all rows in the matrix.
    ///
    /// Rows are yielded sequentially beginning with row 0.
    ///
    /// # Panics
    ///
    /// Panics if `self.ncols() == 0` (because the implementation does not work correctly
    /// in this case and it's too corner-case to bother fixing). This restriction may
    /// be lifted in the future.
    pub fn row_iter_mut(&mut self) -> impl Iterator<Item = &mut [T::Elem]>
    where
        T: MutDenseData,
    {
        assert!(self.ncols() != 0);

        let ncols = self.ncols();
        let cstride = self.cstride();
        self.data
            .as_mut_slice()
            .chunks_mut(cstride)
            .map(move |i| &mut i[..ncols])
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
        self.as_slice().get_unchecked(row * self.cstride + col)
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
        let cstride = self.cstride;
        debug_assert!(row < self.nrows);
        debug_assert!(col < self.ncols);
        self.as_mut_slice().get_unchecked_mut(row * cstride + col)
    }

    /// Return a view over the matrix.
    pub fn as_view(&self) -> StridedView<'_, T::Elem> {
        StridedView {
            data: self.as_slice(),
            nrows: self.nrows,
            ncols: self.ncols,
            cstride: self.cstride,
        }
    }
}

pub type StridedView<'a, T> = StridedBase<&'a [T]>;
pub type MutStridedView<'a, T> = StridedBase<&'a mut [T]>;

/// Return a reference to the item at entry `(row, col)` in the matrix.
///
/// # Panics
///
/// Panics if `row >= self.nrows()` or `col >= self.ncols()`.
impl<T> Index<(usize, usize)> for StridedBase<T>
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
impl<T> IndexMut<(usize, usize)> for StridedBase<T>
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

impl<T, U> From<views::MatrixBase<T>> for StridedBase<U>
where
    T: DenseData,
    U: DenseData,
    T: Into<U>,
{
    fn from(matrix: views::MatrixBase<T>) -> Self {
        let nrows = matrix.nrows();
        let ncols = matrix.ncols();
        Self {
            data: matrix.into_inner().into(),
            nrows,
            ncols,
            cstride: ncols,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_length() {
        // If the number of rows is zero - the output should always be zero.
        assert_eq!(linear_length(0, 1, 1), 0);
        assert_eq!(linear_length(0, 2, 2), 0);
        assert_eq!(linear_length(0, 2, 3), 0);
        assert_eq!(linear_length(0, 2, 4), 0);

        // If `cstride == ncols`, then the computation should be trivial.
        for row in 1..10 {
            for col in 1..10 {
                assert_eq!(linear_length(row, col, col), row * col);
            }
        }

        // If there is only one row, then `cstride` should be ignored.
        assert_eq!(linear_length(1, 5, 10), 5);
        assert_eq!(linear_length(1, 7, 99), 7);

        // Otherwise, the computation is a block of `nrows - 1` chunks of `cstride` and then
        // `ncols`. Yes - this runs a bunch of computations.
        for row in 2..10 {
            for col in 0..10 {
                for cstride in col..12 {
                    assert_eq!(linear_length(row, col, cstride), (row - 1) * cstride + col);
                }
            }
        }
    }

    fn assert_is_static<T: 'static>(_x: &T) {}

    #[test]
    fn try_from_error_misc() {
        let x = TryFromError::<&[f32]> {
            data: &[],
            nrows: 1,
            ncols: 2,
            cstride: 3,
        };

        let display = format!("{}", x);
        let debug = format!("{:?}", x);
        println!("debug = {}", debug);
        assert!(debug.contains("TryFromError"));
        assert!(debug.contains("data_len: 0"));
        assert!(debug.contains("nrows: 1"));
        assert!(debug.contains("ncols: 2"));
        assert!(debug.contains("cstride: 3"));

        let x = x.as_static();
        assert_is_static(&x);
        assert_eq!(
            display,
            format!("{}", x),
            "static version of the error must hav ethe same message"
        );
    }

    fn expected_error(len: usize, nrows: usize, ncols: usize, cstride: usize) -> String {
        format!(
            "tried to construct a strided matrix with {nrows} rows and {ncols} cols and \
             column stride {cstride} over a slice of length {} (expected {})",
            len,
            linear_length(nrows, ncols, cstride)
        )
    }

    // Test that the contents of `dut` match those in the dense 2d matrix.
    fn test_indexing(dut: StridedView<'_, usize>, expected: views::MatrixView<'_, usize>) {
        assert_eq!(dut.nrows(), expected.nrows());
        assert_eq!(dut.ncols(), expected.ncols());

        // Check the underlying data.
        if dut.cstride() == dut.ncols() {
            assert_eq!(dut.as_slice(), expected.as_slice());
        } else {
            assert_ne!(dut.as_slice(), expected.as_slice());
        }

        // Compare via linear indexing.
        for row in 0..dut.nrows() {
            for col in 0..dut.ncols() {
                assert_eq!(
                    dut[(row, col)],
                    expected[(row, col)],
                    "failed on (row, col) = ({}, {})",
                    row,
                    col
                );
            }
        }

        // Compare via row.
        for row in 0..dut.nrows() {
            assert_eq!(dut.row(row), expected.row(row), "failed on row {}", row);
        }

        // Compare via row iterators.
        assert!(dut.row_iter().eq(expected.row_iter()));
    }

    // Create a base Matrix with the following pattern:
    // ```text
    //       0         1         2 ...   ncols-1
    //   ncols   ncols+1   ncols+2 ... 2*ncols-1
    // 2*ncols 2*ncols+1 2*ncols+2 ... 3*ncols-1
    // ...
    // ```
    fn create_test_matrix(nrows: usize, ncols: usize) -> views::Matrix<usize> {
        let mut i = 0;
        views::Matrix::new(
            views::Init(|| {
                let v = i;
                i += 1;
                v
            }),
            nrows,
            ncols,
        )
    }

    #[test]
    fn test_basic_indexing() {
        let m = create_test_matrix(5, 3);

        // First - test a dense StridedView over the entire matrix.
        let ptr = m.as_ptr();
        let v = StridedView::try_from(m.as_slice(), m.nrows(), m.ncols(), m.ncols()).unwrap();
        assert_eq!(v.as_ptr(), ptr, "base pointer was not preserved");

        assert_eq!(v.nrows(), m.nrows());
        assert_eq!(v.ncols(), m.ncols());
        assert_eq!(v.cstride(), m.ncols());
        test_indexing(v, m.as_view());

        // Now - create a truly strided view over the first two columns.
        let v = StridedView::try_from(
            &(m.as_slice()[..(4 * m.ncols() + 2)]),
            m.nrows(),
            2,
            m.ncols(),
        )
        .unwrap();
        assert_eq!(v.as_ptr(), ptr, "base pointer was not preserved");

        // Create the expected matrix.
        let mut expected = views::Matrix::new(0, 5, 2);
        for row in 0..expected.nrows() {
            for col in 0..expected.ncols() {
                expected[(row, col)] = m[(row, col)];
            }
        }
        test_indexing(v, expected.as_view());

        // Create a strided view over the last two columns.
        let v = StridedView::try_from(&(m.as_slice()[1..]), m.nrows(), 2, m.ncols()).unwrap();
        let mut expected = views::Matrix::new(0, 5, 2);
        for row in 0..expected.nrows() {
            for col in 0..expected.ncols() {
                expected[(row, col)] = m[(row, col + 1)];
            }
        }
        test_indexing(v, expected.as_view());
    }

    #[test]
    fn test_mutable_indexing() {
        // The source matrix.
        let src = create_test_matrix(5, 4);

        // Initialize using 2d indexing.
        {
            let mut dst = views::Matrix::<usize>::new(0, 5, 10);

            let ptr = dst.as_ptr();

            let ncols = src.ncols();
            let nrows = src.nrows();
            let cstride = dst.ncols();
            let mut dst_view =
                MutStridedView::try_shrink_from_mut(dst.as_mut_slice(), nrows, ncols, cstride)
                    .unwrap();

            assert_eq!(dst_view.as_ptr(), ptr);
            assert_eq!(dst_view.as_mut_ptr().cast_const(), ptr);
            assert_eq!(dst_view.nrows(), nrows);
            assert_eq!(dst_view.ncols(), ncols);
            assert_eq!(dst_view.cstride(), cstride);

            // Initialize using linear indexing.
            for row in 0..dst_view.nrows() {
                for col in 0..dst_view.ncols() {
                    dst_view[(row, col)] = src[(row, col)]
                }
            }

            // Check equality.
            test_indexing(dst_view.as_view(), src.as_view());
        }

        // Initialize using row-wise indexing.
        {
            let mut dst = views::Matrix::<usize>::new(0, 5, 10);

            let ptr = dst.as_ptr();

            let ncols = src.ncols();
            let nrows = src.nrows();
            let cstride = dst.ncols();
            let mut dst_view =
                MutStridedView::try_shrink_from_mut(dst.as_mut_slice(), nrows, ncols, cstride)
                    .unwrap();

            assert_eq!(dst_view.as_ptr(), ptr);
            assert_eq!(dst_view.as_mut_ptr().cast_const(), ptr);
            assert_eq!(dst_view.nrows(), nrows);
            assert_eq!(dst_view.ncols(), ncols);
            assert_eq!(dst_view.cstride(), cstride);

            // Initialize by looping over rows.
            for row in 0..dst_view.nrows() {
                dst_view.row_mut(row).copy_from_slice(src.row(row))
            }

            // Check equality.
            test_indexing(dst_view.as_view(), src.as_view());
        }

        // Initialize using row-iterator indexing.
        {
            let mut dst = views::Matrix::<usize>::new(0, 5, 10);

            let offset = 2;
            // SAFETY: The underlying allocation is valid for much more than 2 elements.
            let ptr = unsafe { dst.as_ptr().add(offset) };

            let ncols = src.ncols();
            let nrows = src.nrows();
            let cstride = dst.ncols();
            let mut dst_view = MutStridedView::try_shrink_from_mut(
                &mut dst.as_mut_slice()[2..],
                nrows,
                ncols,
                cstride,
            )
            .unwrap();

            assert_eq!(dst_view.as_ptr(), ptr);
            assert_eq!(dst_view.as_mut_ptr().cast_const(), ptr);
            assert_eq!(dst_view.nrows(), nrows);
            assert_eq!(dst_view.ncols(), ncols);
            assert_eq!(dst_view.cstride(), cstride);

            // Initialize using row iterators.
            for (d, s) in std::iter::zip(dst_view.row_iter_mut(), src.row_iter()) {
                d.copy_from_slice(s)
            }

            // Check equality.
            test_indexing(dst_view.as_view(), src.as_view());
        }
    }

    #[test]
    fn matrix_conversion() {
        let m = create_test_matrix(3, 4);
        let ptr = m.as_ptr();
        let v: StridedView<_> = m.as_view().into();
        assert_eq!(v.as_ptr(), ptr);
        test_indexing(v, m.as_view());
    }

    #[test]
    fn test_zero_sized() {
        let m = create_test_matrix(5, 5);
        let v = StridedView::try_shrink_from(m.as_slice(), 0, 4, 5).unwrap();

        assert_eq!(v.nrows(), 0);
        assert_eq!(v.ncols(), 4);
        assert_eq!(v.cstride(), 5);

        let v = StridedView::try_shrink_from(m.as_slice(), 5, 0, 5).unwrap();
        assert_eq!(v.nrows(), 5);
        assert_eq!(v.ncols(), 0);
        assert_eq!(v.cstride(), 5);

        for row in 0..v.nrows() {
            let empty: &[usize] = &[];
            assert_eq!(v.row(row), empty);
        }
    }

    #[test]
    #[should_panic]
    fn test_row_iter_panics() {
        let m = create_test_matrix(5, 5);
        let v = StridedView::try_shrink_from(m.as_slice(), 5, 0, 5).unwrap();
        let _ = v.row_iter();
    }

    #[test]
    #[should_panic]
    fn test_row_iter_mut_panics() {
        let mut m = create_test_matrix(5, 5);
        let mut v = MutStridedView::try_shrink_from_mut(m.as_mut_slice(), 5, 0, 5).unwrap();
        let _ = v.row_iter_mut();
    }

    #[test]
    fn test_try_shrink_from() {
        // Exact is okay.
        let m = views::Matrix::<usize>::new(0, 10, 10);
        let nrows = m.nrows();
        let ncols = m.ncols();
        let s = StridedView::try_shrink_from(m.as_slice(), nrows, ncols, ncols).unwrap();
        assert_eq!(s.as_slice(), m.as_slice());

        // Giving a slice that is too large is okay.
        let s = StridedView::try_shrink_from(m.as_slice(), nrows, 5, ncols).unwrap();
        assert_eq!(s.as_ptr(), m.as_ptr());

        // Too small is a problem.
        let s = StridedView::try_shrink_from(m.as_slice(), nrows, ncols, ncols + 1);
        assert!(s.is_err());
        let err = s.unwrap_err();
        assert_eq!(
            err.to_string(),
            expected_error(m.as_slice().len(), nrows, ncols, ncols + 1)
        );
        assert_eq!(err.into_inner(), m.as_slice());
    }

    #[test]
    #[should_panic(expected = "cstride must be greater than or equal to ncols")]
    fn test_try_shink_from_panics() {
        let m = views::Matrix::<usize>::new(0, 4, 4);
        let _ = StridedView::try_shrink_from(m.as_slice(), 2, 2, 1);
    }

    #[test]
    fn test_try_shrink_from_mut() {
        // Exact is okay.
        let mut m = views::Matrix::<usize>::new(0, 10, 10);

        let nrows = m.nrows();
        let ncols = m.ncols();
        let ptr = m.as_ptr();
        let len = m.as_slice().len();

        let s = MutStridedView::try_shrink_from_mut(m.as_mut_slice(), nrows, ncols, ncols).unwrap();
        assert_eq!(s.as_ptr(), ptr);
        assert_eq!(s.as_slice().len(), len);

        // Giving a slice that is too large is okay.
        let s = MutStridedView::try_shrink_from_mut(m.as_mut_slice(), nrows, 5, ncols).unwrap();
        assert_eq!(s.as_ptr(), ptr);

        // Too small is a problem.
        let s = MutStridedView::try_shrink_from_mut(m.as_mut_slice(), nrows, ncols, ncols + 1);
        assert!(s.is_err());
        let err = s.unwrap_err();
        assert_eq!(
            err.to_string(),
            expected_error(len, nrows, ncols, ncols + 1)
        );
    }

    #[test]
    #[should_panic(expected = "cstride must be greater than or equal to ncols")]
    fn test_try_shink_from_mut_panics() {
        let mut m = views::Matrix::<usize>::new(0, 4, 4);
        let _ = MutStridedView::try_shrink_from_mut(m.as_mut_slice(), 2, 2, 1);
    }

    #[test]
    fn test_try_from() {
        // Exact is okay.
        let m = views::Matrix::<usize>::new(0, 10, 10);
        let nrows = m.nrows();
        let ncols = m.ncols();
        let s = StridedView::try_from(m.as_slice(), nrows, ncols, ncols).unwrap();
        assert_eq!(s.as_slice(), m.as_slice());

        // Giving a slice that is too large is a problem.
        let s = StridedView::try_from(m.as_slice(), nrows, 5, ncols);
        assert!(s.is_err());
        let err = s.unwrap_err();
        assert_eq!(
            err.to_string(),
            expected_error(m.as_slice().len(), nrows, 5, ncols)
        );

        // Too small is a problem.
        let s = StridedView::try_from(m.as_slice(), nrows, ncols, ncols + 1);
        assert!(s.is_err());
        let err = s.unwrap_err();
        assert_eq!(
            err.to_string(),
            expected_error(m.as_slice().len(), nrows, ncols, ncols + 1)
        );
        assert_eq!(err.into_inner(), m.as_slice());
    }

    #[test]
    #[should_panic(expected = "cstride must be greater than or equal to ncols")]
    fn test_try_frompanics() {
        let mut m = views::Matrix::<usize>::new(0, 4, 4);
        let _ = MutStridedView::try_from(m.as_mut_slice(), 2, 2, 1);
    }

    #[test]
    #[should_panic(expected = "tried to access row 3 of a matrix with 3 rows")]
    fn test_get_row_panics() {
        let m = views::Matrix::<usize>::new(0, 3, 7);
        let v: StridedView<_> = m.as_view().into();
        v.row(3);
    }

    #[test]
    #[should_panic(expected = "tried to access row 3 of a matrix with 3 rows")]
    fn test_get_row_mut_panics() {
        let mut m = views::Matrix::<usize>::new(0, 3, 7);
        let mut v: MutStridedView<_> = m.as_mut_view().into();
        v.row_mut(3);
    }

    #[test]
    #[should_panic(expected = "row 3 is out of bounds (max: 3)")]
    fn test_index_panics_row() {
        let m = views::Matrix::<usize>::new(0, 3, 7);
        let v: StridedView<_> = m.as_view().into();
        let _ = v[(3, 2)];
    }

    #[test]
    #[should_panic(expected = "col 7 is out of bounds (max: 7)")]
    fn test_index_panics_col() {
        let m = views::Matrix::<usize>::new(0, 3, 7);
        let v: StridedView<_> = m.as_view().into();
        let _ = v[(2, 7)];
    }

    #[test]
    #[should_panic(expected = "row 3 is out of bounds (max: 3)")]
    fn test_index_mut_panics_row() {
        let mut m = views::Matrix::<usize>::new(0, 3, 7);
        let mut v: MutStridedView<_> = m.as_mut_view().into();
        v[(3, 2)] = 1;
    }

    #[test]
    #[should_panic(expected = "col 7 is out of bounds (max: 7)")]
    fn test_index_mut_panics_col() {
        let mut m = views::Matrix::<usize>::new(0, 3, 7);
        let mut v: MutStridedView<_> = m.as_mut_view().into();
        v[(2, 7)] = 1;
    }
}
