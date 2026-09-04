/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt, marker::PhantomData, ptr::NonNull};
use thiserror::Error;

use crate::{
    internal,
    views::{self},
    Reborrow,
};

/// The layout for [`Strided`].
///
/// This struct ensures that the [`Self::cstride`] is greater than [`Self::ncols`] and that
/// the linear length of the representation does not overflow `usize::MAX`.
///
/// The linear length of [`Strided]` is given by the forumula
/// ```text
/// self.nrows.saturating_sub(1) * self.cstride + self.nrows.min(1) * self.ncols
/// ```
/// This allows the last row to occupy less than a full stride.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    nrows: usize,
    ncols: usize,
    cstride: usize,
}

impl Layout {
    /// Construct a new [`Layout`].
    ///
    /// Errors if:
    ///
    /// * `ncols < cstride`.
    /// * The computation of the linear length overflows `usize::MAX`.
    pub fn new(nrows: usize, ncols: usize, cstride: usize) -> Result<Self, LayoutError> {
        LayoutError::check(nrows, ncols, cstride)?;
        Ok(Self {
            nrows,
            ncols,
            cstride,
        })
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Return the number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Return the stride between subsequent rows.
    pub fn cstride(&self) -> usize {
        self.cstride
    }

    /// Return the linear length of the [`Strided`] described by this [`Layout`].
    pub fn linear_length(&self) -> usize {
        self.nrows.saturating_sub(1) * self.cstride + self.nrows.min(1) * self.ncols
    }
}

fn linear_length(nrows: usize, ncols: usize, cstride: usize) -> Option<usize> {
    nrows
        .saturating_sub(1)
        .checked_mul(cstride)
        .and_then(|main| main.checked_add(nrows.min(1) * ncols))
}

/// Errors for [`Layout::new`].
#[derive(Debug)]
pub struct LayoutError(LayoutErrorInner);

impl LayoutError {
    fn check(nrows: usize, ncols: usize, cstride: usize) -> Result<usize, Self> {
        if cstride < ncols {
            Err(Self(LayoutErrorInner::InvalidStride { ncols, cstride }))
        } else {
            linear_length(nrows, ncols, cstride)
                .ok_or(Self(LayoutErrorInner::Overflow { nrows, cstride }))
        }
    }
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for LayoutError {}

#[derive(Debug)]
enum LayoutErrorInner {
    InvalidStride { ncols: usize, cstride: usize },
    Overflow { nrows: usize, cstride: usize },
}

impl fmt::Display for LayoutErrorInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStride { ncols, cstride } => write!(
                f,
                "column stride {} must be greater than or equal to number of columns {}",
                cstride, ncols
            ),
            Self::Overflow { nrows, cstride } => write!(
                f,
                "a {}x{} strided matrix has a length exceeding usize::MAX",
                nrows, cstride
            ),
        }
    }
}

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
#[derive(Debug)]
pub struct Strided<'a, T> {
    ptr: NonNull<T>,
    layout: Layout,
    _lifetime: PhantomData<&'a [T]>,
}

impl<'a, T> Strided<'a, T> {
    /// Construct a strided view over data slice, shrinking the slice as needed.
    ///
    /// Returns an error if the layout is invalid or `data` is not at least
    /// [`Layout::linear_length`].
    pub fn try_from_data(
        data: &'a [T],
        nrows: usize,
        ncols: usize,
        cstride: usize,
    ) -> Result<Self, TryFromError> {
        let layout = Layout::new(nrows, ncols, cstride).map_err(TryFromError::LayoutError)?;
        let expected = layout.linear_length();
        if data.len() < expected {
            Err(TryFromError::InvalidLength {
                got: data.len(),
                expected,
            })
        } else {
            Ok(Self {
                ptr: internal::slice_to_nonnull(data),
                layout,
                _lifetime: PhantomData,
            })
        }
    }

    fn as_nonnull(&self) -> NonNull<T> {
        self.ptr
    }

    /// Return the [`Layout`] for the matrix.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn as_ptr(&self) -> *const T {
        self.as_nonnull().as_ptr().cast_const()
    }

    /// Return the number of columns in the matrix.
    pub fn ncols(&self) -> usize {
        self.layout().ncols()
    }

    /// Return the number of rows in the matrix.
    pub fn nrows(&self) -> usize {
        self.layout().nrows()
    }

    /// Return the count of elements between the start of each row.
    pub fn cstride(&self) -> usize {
        self.layout().cstride()
    }

    /// Return the underlying data as a slice.
    ///
    /// # Note
    ///
    /// The underlying representation for a strided matrix is not necessarily dense.
    pub fn as_slice(&self) -> &[T] {
        let layout = self.layout();

        // SAFETY: Constructors verify that the backing memory has a length as least
        // `layout.linear_length()`.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), layout.linear_length()) }
    }

    // Element Access

    /// Return the specified element.
    ///
    /// # Safety
    ///
    /// * `row < self.nrows()`.
    /// * `col < self.ncols()`.
    pub unsafe fn element_unchecked(&self, row: usize, col: usize) -> &T {
        let layout = self.layout();
        debug_assert!(row < layout.nrows());
        debug_assert!(col < layout.ncols());

        // SAFETY: Constructors verify that the backing memory has a length of at least
        // `layout.linear_length`. Since `row` and `col` are in-bounds, the pointer offset
        // is valid and it is safe to return a reference.
        unsafe { &*self.as_ptr().add(layout.cstride() * row + col) }
    }

    /// Return the specified element if `row < self.nrows()` and `col < self.ncols()`.
    pub fn element(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.nrows() && col < self.ncols() {
            // SAFETY: `row` and `col` are in-bounds.
            Some(unsafe { self.element_unchecked(row, col) })
        } else {
            None
        }
    }

    /// Return the specified element.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.nrows()` or `col >= self.ncols()`.
    pub fn element_or_panic(&self, row: usize, col: usize) -> &T {
        assert!(
            row < self.nrows(),
            "row {} is out of bounds for a matrix with {} rows",
            row,
            self.nrows()
        );
        assert!(
            col < self.ncols(),
            "col {} is out of bounds for a matrix with {} cols",
            col,
            self.ncols()
        );

        // SAFETY: `row` and `col` are in-bounds.
        unsafe { self.element_unchecked(row, col) }
    }

    // Row Access

    /// Returns the requested row without boundschecking.
    ///
    /// # Safety
    ///
    /// Caller must ensure `row < self.nrows()`.
    pub unsafe fn row_unchecked(&self, row: usize) -> &[T] {
        let layout = self.layout();
        debug_assert!(row < layout.nrows());

        // SAFETY: Constructors verify that the backing memory has a length of at least
        // `layout.linear_length`. Since `row` is in-bounds, the pointer offset
        // is valid and it is safe to form a slice of length `layout.ncols()`.
        unsafe {
            std::slice::from_raw_parts(self.as_ptr().add(layout.cstride() * row), layout.ncols())
        }
    }

    /// Return the requested row if `row < self.nrows()`.
    pub fn row(&self, row: usize) -> Option<&[T]> {
        if row < self.nrows() {
            // SAFETY: `row` is in-bounds.
            Some(unsafe { self.row_unchecked(row) })
        } else {
            None
        }
    }

    /// Return row `row` as a slice.
    ///
    /// # Panic
    ///
    /// Panics if `row >= self.nrows()`.
    pub fn row_or_panic(&self, row: usize) -> &[T] {
        assert!(
            row < self.nrows(),
            "row {} is out of bounds for a matrix with {} rows",
            row,
            self.nrows()
        );

        // SAFETY: `row` is in-bounds.
        unsafe { self.row_unchecked(row) }
    }

    /// Return a iterator over all rows in the matrix.
    ///
    /// Rows are yielded sequentially beginning with row 0.
    pub fn rows(&self) -> Rows<'_, T> {
        Rows::new(*self)
    }
}

/// Errors for [`Strided::new`].
#[derive(Debug, Error)]
pub enum TryFromError {
    #[error(transparent)]
    LayoutError(LayoutError),
    #[error(
        "argument of length {} is shorter than the expected length {}",
        got,
        expected
    )]
    InvalidLength { got: usize, expected: usize },
}

impl<'a, T> From<views::MatrixView<'a, T>> for Strided<'a, T> {
    fn from(matrix: views::MatrixView<'a, T>) -> Self {
        // FIXME: `views::MatrixView` doesn't quite ensure that `nrows * ncols` doesn't
        // overflow. So for now, we need to be pessimistic.
        //
        // This will be fixed when `MatrixView` gets the same treatment.
        Self::try_from_data(
            matrix.into_inner(),
            matrix.nrows(),
            matrix.ncols(),
            matrix.ncols(),
        )
        .expect("this will be made infallible in the future")
    }
}

// SAFETY: `Strided` only exposes shared access to its `T` elements (via `&T`/`&[T]`), so
// sharing or transferring a `Strided<T>` across threads is sound whenever `T: Sync`.
unsafe impl<T> Send for Strided<'_, T> where T: Sync {}
// SAFETY: See above.
unsafe impl<T> Sync for Strided<'_, T> where T: Sync {}

impl<T> Clone for Strided<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Strided<'_, T> {}

impl<'a, T> Reborrow<'a> for Strided<'_, T> {
    type Target = Strided<'a, T>;
    fn reborrow(&'a self) -> Self::Target {
        *self
    }
}

/// Iterator for [`Strided::rows`].
#[derive(Debug)]
pub struct Rows<'a, T> {
    ptr: NonNull<T>,
    remaining: usize,
    ncols: usize,
    cstride: usize,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T> Rows<'a, T> {
    fn new(strided: Strided<'a, T>) -> Self {
        let layout = strided.layout();
        Self {
            ptr: strided.as_nonnull(),
            remaining: layout.nrows(),
            ncols: layout.ncols(),
            cstride: layout.cstride(),
            _lifetime: PhantomData,
        }
    }
}

// SAFETY: `Rows` only yields shared `&[T]` slices borrowed from a `Strided`, so sharing or
// transferring a `Rows<T>` across threads is sound whenever `T: Sync`.
unsafe impl<T> Send for Rows<'_, T> where T: Sync {}
// SAFETY: See above.
unsafe impl<T> Sync for Rows<'_, T> where T: Sync {}

impl<'a, T> Iterator for Rows<'a, T> {
    type Item = &'a [T];
    fn next(&mut self) -> Option<&'a [T]> {
        self.remaining.checked_sub(1).map(|remaining| {
            // SAFETY: The originating `Strided` guarantees `self.remaining` rows of
            // `self.ncols` elements each are readable starting from `self.ptr`, so the
            // current row is valid for `self.ncols` elements.
            let item =
                unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast_const(), self.ncols) };
            self.remaining = remaining;
            if remaining != 0 {
                // SAFETY: There is at least one more row remaining, so advancing by
                // `self.cstride` stays within (or one-past-the-end of) the originating
                // allocation, per `Layout::linear_length`.
                self.ptr = unsafe { self.ptr.add(self.cstride) };
            }
            item
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for Rows<'_, T> {}
impl<T> std::iter::FusedIterator for Rows<'_, T> {}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_length() {
        // If the number of rows is zero - the output should always be zero.
        assert_eq!(linear_length(0, 1, 1).unwrap(), 0);
        assert_eq!(linear_length(0, 2, 2).unwrap(), 0);
        assert_eq!(linear_length(0, 2, 3).unwrap(), 0);
        assert_eq!(linear_length(0, 2, 4).unwrap(), 0);

        // If `cstride == ncols`, then the computation should be trivial.
        for row in 1..10 {
            for col in 1..10 {
                assert_eq!(linear_length(row, col, col).unwrap(), row * col);
            }
        }

        // If there is only one row, then `cstride` should be ignored.
        assert_eq!(linear_length(1, 5, 10).unwrap(), 5);
        assert_eq!(linear_length(1, 7, 99).unwrap(), 7);

        // Otherwise, the computation is a block of `nrows - 1` chunks of `cstride` and then
        // `ncols`. Yes - this runs a bunch of computations.
        for row in 2..10 {
            for col in 0..10 {
                for cstride in col..12 {
                    assert_eq!(
                        linear_length(row, col, cstride).unwrap(),
                        (row - 1) * cstride + col
                    );
                }
            }
        }
    }

    #[test]
    fn test_layout_new() {
        // Valid layouts.
        let layout = Layout::new(3, 4, 4).unwrap();
        assert_eq!(layout.nrows(), 3);
        assert_eq!(layout.ncols(), 4);
        assert_eq!(layout.cstride(), 4);
        assert_eq!(layout.linear_length(), 12);

        let layout = Layout::new(3, 4, 6).unwrap();
        assert_eq!(layout.linear_length(), 2 * 6 + 4);

        // `cstride == ncols` is fine, even at zero.
        assert!(Layout::new(0, 0, 0).is_ok());

        // Invalid stride: `cstride < ncols`.
        let err = Layout::new(3, 4, 3).unwrap_err();
        assert_eq!(
            err.to_string(),
            "column stride 3 must be greater than or equal to number of columns 4"
        );

        // Overflow: linear length exceeds `usize::MAX`.
        let err = Layout::new(usize::MAX, usize::MAX, usize::MAX).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!(
                "a {}x{} strided matrix has a length exceeding usize::MAX",
                usize::MAX,
                usize::MAX
            )
        );
    }

    #[test]
    fn test_try_from_data_errors() {
        let m = views::Matrix::<usize>::new(0, 10, 10);
        let nrows = m.nrows();
        let ncols = m.ncols();

        // An invalid layout (`cstride < ncols`) is reported as a `LayoutError`, not a panic.
        let err = Strided::try_from_data(m.as_slice(), 2, 2, 1).unwrap_err();
        assert_eq!(
            err.to_string(),
            "column stride 1 must be greater than or equal to number of columns 2"
        );

        // A slice shorter than `Layout::linear_length` is reported as `InvalidLength`.
        let err = Strided::try_from_data(m.as_slice(), nrows, ncols, ncols + 1).unwrap_err();
        let expected_len = Layout::new(nrows, ncols, ncols + 1).unwrap().linear_length();
        assert_eq!(
            err.to_string(),
            format!(
                "argument of length {} is shorter than the expected length {}",
                m.as_slice().len(),
                expected_len
            )
        );
    }

    #[test]
    fn test_element_and_row_out_of_bounds() {
        let m = create_test_matrix(3, 4);
        let v = Strided::try_from_data(m.as_slice(), m.nrows(), m.ncols(), m.ncols()).unwrap();

        // In-bounds accesses succeed.
        assert!(v.element(2, 3).is_some());
        assert!(v.row(2).is_some());

        // Out-of-bounds row and/or col return `None` rather than panicking.
        assert!(v.element(3, 0).is_none(), "row out-of-bounds");
        assert!(v.element(0, 4).is_none(), "col out-of-bounds");
        assert!(v.element(3, 4).is_none(), "both out-of-bounds");
        assert!(v.row(3).is_none());
    }

    #[test]
    #[should_panic(expected = "row 3 is out of bounds for a matrix with 3 rows")]
    fn test_element_or_panic_panics_on_row() {
        let m = create_test_matrix(3, 4);
        let v = Strided::try_from_data(m.as_slice(), m.nrows(), m.ncols(), m.ncols()).unwrap();
        v.element_or_panic(3, 0);
    }

    #[test]
    #[should_panic(expected = "col 4 is out of bounds for a matrix with 4 cols")]
    fn test_element_or_panic_panics_on_col() {
        let m = create_test_matrix(3, 4);
        let v = Strided::try_from_data(m.as_slice(), m.nrows(), m.ncols(), m.ncols()).unwrap();
        v.element_or_panic(0, 4);
    }

    #[test]
    #[should_panic(expected = "row 3 is out of bounds for a matrix with 3 rows")]
    fn test_row_or_panic_panics() {
        let m = create_test_matrix(3, 4);
        let v = Strided::try_from_data(m.as_slice(), m.nrows(), m.ncols(), m.ncols()).unwrap();
        v.row_or_panic(3);
    }

    #[test]
    fn test_clone_copy_reborrow() {
        let m = create_test_matrix(3, 4);
        let v = Strided::try_from_data(m.as_slice(), m.nrows(), m.ncols(), m.ncols()).unwrap();

        // `Copy`/`Clone` produce an independent handle to the same data.
        let copied = v;
        let cloned = v.clone();
        assert_eq!(v.as_ptr(), copied.as_ptr());
        assert_eq!(v.as_ptr(), cloned.as_ptr());

        // `Reborrow` should yield an equivalent view.
        let reborrowed = v.reborrow();
        assert_eq!(reborrowed.as_ptr(), v.as_ptr());
        assert_eq!(reborrowed.nrows(), v.nrows());
        assert_eq!(reborrowed.ncols(), v.ncols());
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Strided<'_, u8>>();
        assert_send_sync::<Rows<'_, u8>>();
    }

    #[test]
    fn test_rows_iterator_properties() {
        let m = create_test_matrix(4, 3);
        let v = Strided::try_from_data(m.as_slice(), m.nrows(), m.ncols(), m.ncols()).unwrap();

        let mut rows = v.rows();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows.size_hint(), (4, Some(4)));

        for expected_row in 0..4 {
            let row = rows.next().unwrap();
            assert_eq!(row, m.row(expected_row));
        }

        // Exhausted iterators keep returning `None` (`FusedIterator`).
        assert_eq!(rows.next(), None);
        assert_eq!(rows.next(), None);
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_rows_iterator_zero_rows() {
        let m = create_test_matrix(5, 5);
        let v = Strided::try_from_data(m.as_slice(), 0, 4, 5).unwrap();

        let mut rows = v.rows();
        assert_eq!(rows.len(), 0);
        assert_eq!(rows.next(), None);
    }

    // Test that the contents of `dut` match those in the dense 2d matrix.
    fn test_indexing(dut: Strided<'_, usize>, expected: views::MatrixView<'_, usize>) {
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
                let e = expected[(row, col)];

                assert_eq!(
                    *dut.element_or_panic(row, col),
                    e,
                    "failed on (row, col) = ({}, {})",
                    row,
                    col
                );

                assert_eq!(
                    *dut.element(row, col).unwrap(),
                    e,
                    "failed on (row, col) = ({}, {})",
                    row,
                    col
                );
            }
        }

        // Compare via row.
        for row in 0..dut.nrows() {
            assert_eq!(
                dut.row_or_panic(row),
                expected.row(row),
                "failed on row {}",
                row
            );

            assert_eq!(
                dut.row(row).unwrap(),
                expected.row(row),
                "failed on row {}",
                row
            );
        }

        // Compare via row iterators.
        assert!(dut.rows().eq(expected.row_iter()));
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
        let v = Strided::try_from_data(m.as_slice(), m.nrows(), m.ncols(), m.ncols()).unwrap();
        assert_eq!(v.as_ptr(), ptr, "base pointer was not preserved");

        assert_eq!(v.nrows(), m.nrows());
        assert_eq!(v.ncols(), m.ncols());
        assert_eq!(v.cstride(), m.ncols());
        test_indexing(v, m.as_view());

        // Now - create a truly strided view over the first two columns.
        let v = Strided::try_from_data(
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
        let v = Strided::try_from_data(&(m.as_slice()[1..]), m.nrows(), 2, m.ncols()).unwrap();
        let mut expected = views::Matrix::new(0, 5, 2);
        for row in 0..expected.nrows() {
            for col in 0..expected.ncols() {
                expected[(row, col)] = m[(row, col + 1)];
            }
        }
        test_indexing(v, expected.as_view());
    }

    #[test]
    fn matrix_conversion() {
        let m = create_test_matrix(3, 4);
        let ptr = m.as_ptr();
        let v: Strided<_> = m.as_view().into();
        assert_eq!(v.as_ptr(), ptr);
        test_indexing(v, m.as_view());
    }

    #[test]
    fn test_zero_sized() {
        let m = create_test_matrix(5, 5);
        let v = Strided::try_from_data(m.as_slice(), 0, 4, 5).unwrap();

        assert_eq!(v.nrows(), 0);
        assert_eq!(v.ncols(), 4);
        assert_eq!(v.cstride(), 5);

        let v = Strided::try_from_data(m.as_slice(), 5, 0, 5).unwrap();
        assert_eq!(v.nrows(), 5);
        assert_eq!(v.ncols(), 0);
        assert_eq!(v.cstride(), 5);

        for row in 0..v.nrows() {
            let empty: &[usize] = &[];
            assert_eq!(v.row(row).unwrap(), empty);
        }
    }

    #[test]
    fn test_try_shrink_from() {
        // Exact is okay.
        let m = views::Matrix::<usize>::new(0, 10, 10);
        let nrows = m.nrows();
        let ncols = m.ncols();
        let s = Strided::try_from_data(m.as_slice(), nrows, ncols, ncols).unwrap();
        assert_eq!(s.as_slice(), m.as_slice());

        // Giving a slice that is too large is okay.
        let s = Strided::try_from_data(m.as_slice(), nrows, 5, ncols).unwrap();
        assert_eq!(s.as_ptr(), m.as_ptr());

        // Too small is a problem.
        let s = Strided::try_from_data(m.as_slice(), nrows, ncols, ncols + 1);
        assert!(s.is_err());
        let err = s.unwrap_err();
        // assert_eq!(
        //     err.to_string(),
        //     expected_error(m.as_slice().len(), nrows, ncols, ncols + 1)
        // );
        // assert_eq!(err.into_inner(), m.as_slice());
    }

    #[test]
    #[should_panic(expected = "cstride must be greater than or equal to ncols")]
    fn test_try_shink_from_panics() {
        let m = views::Matrix::<usize>::new(0, 4, 4);
        let _ = Strided::try_from_data(m.as_slice(), 2, 2, 1).unwrap();
    }

    // #[test]
    // fn test_try_from() {
    //     // Exact is okay.
    //     let m = views::Matrix::<usize>::new(0, 10, 10);
    //     let nrows = m.nrows();
    //     let ncols = m.ncols();
    //     let s = StridedView::try_from(m.as_slice(), nrows, ncols, ncols).unwrap();
    //     assert_eq!(s.as_slice(), m.as_slice());

    //     // Giving a slice that is too large is a problem.
    //     let s = StridedView::try_from(m.as_slice(), nrows, 5, ncols);
    //     assert!(s.is_err());
    //     let err = s.unwrap_err();
    //     assert_eq!(
    //         err.to_string(),
    //         expected_error(m.as_slice().len(), nrows, 5, ncols)
    //     );

    //     // Too small is a problem.
    //     let s = StridedView::try_from(m.as_slice(), nrows, ncols, ncols + 1);
    //     assert!(s.is_err());
    //     let err = s.unwrap_err();
    //     assert_eq!(
    //         err.to_string(),
    //         expected_error(m.as_slice().len(), nrows, ncols, ncols + 1)
    //     );
    //     assert_eq!(err.into_inner(), m.as_slice());
    // }

    // #[test]
    // #[should_panic(expected = "cstride must be greater than or equal to ncols")]
    // fn test_try_frompanics() {
    //     let mut m = views::Matrix::<usize>::new(0, 4, 4);
    //     let _ = MutStridedView::try_from(m.as_mut_slice(), 2, 2, 1);
    // }

    // #[test]
    // #[should_panic(expected = "tried to access row 3 of a matrix with 3 rows")]
    // fn test_get_row_panics() {
    //     let m = views::Matrix::<usize>::new(0, 3, 7);
    //     let v: StridedView<_> = m.as_view().into();
    //     v.row(3);
    // }

    // #[test]
    // #[should_panic(expected = "tried to access row 3 of a matrix with 3 rows")]
    // fn test_get_row_mut_panics() {
    //     let mut m = views::Matrix::<usize>::new(0, 3, 7);
    //     let mut v: MutStridedView<_> = m.as_mut_view().into();
    //     v.row_mut(3);
    // }

    // #[test]
    // #[should_panic(expected = "row 3 is out of bounds (max: 3)")]
    // fn test_index_panics_row() {
    //     let m = views::Matrix::<usize>::new(0, 3, 7);
    //     let v: StridedView<_> = m.as_view().into();
    //     let _ = v[(3, 2)];
    // }

    // #[test]
    // #[should_panic(expected = "col 7 is out of bounds (max: 7)")]
    // fn test_index_panics_col() {
    //     let m = views::Matrix::<usize>::new(0, 3, 7);
    //     let v: StridedView<_> = m.as_view().into();
    //     let _ = v[(2, 7)];
    // }

    // #[test]
    // #[should_panic(expected = "row 3 is out of bounds (max: 3)")]
    // fn test_index_mut_panics_row() {
    //     let mut m = views::Matrix::<usize>::new(0, 3, 7);
    //     let mut v: MutStridedView<_> = m.as_mut_view().into();
    //     v[(3, 2)] = 1;
    // }

    // #[test]
    // #[should_panic(expected = "col 7 is out of bounds (max: 7)")]
    // fn test_index_mut_panics_col() {
    //     let mut m = views::Matrix::<usize>::new(0, 3, 7);
    //     let mut v: MutStridedView<_> = m.as_mut_view().into();
    //     v[(2, 7)] = 1;
    // }
}
