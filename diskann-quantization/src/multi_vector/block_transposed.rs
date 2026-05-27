/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Block-transposed matrix types with configurable packing.
//!
//! This module provides block-transposed matrix types — [`BlockTransposed`] (owned),
//! [`BlockTransposedRef`] (shared view), and [`BlockTransposedMut`] (mutable view) —
//! where groups of `GROUP` rows are stored in transposed form to enable efficient SIMD
//! processing. An optional packing factor `PACK` interleaves adjacent columns within
//! each group, which can be used to feed SIMD instructions that operate on packed pairs
//! (e.g. `vpmaddwd` with `PACK = 2`).
//!
//! # Layout
//!
//! ## `PACK = 1` (standard block-transpose)
//!
//! Given a logical matrix with rows `a`, `b`, `c`, `d`, `e` (each with `K` columns)
//! and `GROUP = 3`:
//!
//! ```text
//!            Group Size (3)
//!            <---------->
//!
//!            +----------+    ^
//!            | a0 b0 c0 |    |
//!            | a1 b1 c1 |    |
//!            | a2 b2 c2 |    | Block Size (K)
//!  Block 0   | ...      |    |
//!  (Full)    | aK bK cK |    |
//!            +----------+    v
//!            +----------+
//!            | d0 e0 XX |
//!  Block 1   | d1 e1 XX |
//!  (Partial) | ...      |
//!            | dK eK XX |
//!            +----------+
//! ```
//!
//! ## `PACK = 2` (super-packed)
//!
//! With `GROUP = 4`, `PACK = 2`, and a logical matrix with rows `a`, `b`, `c`, `d`,
//! `e`, `f` (each with **5** columns — odd, to show padding), adjacent column-pairs
//! are interleaved per row within each group panel:
//!
//! ```text
//!              GROUP × PACK (4 × 2 = 8)
//!              <----------------------------->
//!
//!              +-----------------------------+    ^
//!              | a0 a1  b0 b1  c0 c1  d0 d1  |    |  col-pair (0, 1)
//!              | a2 a3  b2 b3  c2 c3  d2 d3  |    |  col-pair (2, 3)
//!    Block 0   | a4 __  b4 __  c4 __  d4 __  |    |  col-pair (4, pad)
//!    (Full)    +-----------------------------+    v
//!              +-----------------------------+
//!              | e0 e1  f0 f1  XX XX  XX XX  |       col-pair (0, 1)
//!    Block 1   | e2 e3  f2 f3  XX XX  XX XX  |       col-pair (2, 3)
//!    (Partial) | e4 __  f4 __  XX XX  XX XX  |       col-pair (4, pad)
//!              +-----------------------------+
//!
//!    __ = zero (column padding)    XX = zero (row padding)
//!    padded_ncols = 6  (5 rounded up to next multiple of PACK)
//!    Block Size  = padded_ncols / PACK = 3 physical rows per block
//! ```
//!
//! Each physical row of a block holds one column-pair across all `GROUP` rows.
//! For example, the first physical row stores columns `(0, 1)` for rows
//! `a, b, c, d` interleaved as `[a0, a1, b0, b1, c0, c1, d0, d1]`.
//!
//! Because `ncols = 5` is odd (not a multiple of `PACK = 2`), the last
//! column-pair `(4, pad)` is zero-padded: `[a4, 0, b4, 0, c4, 0, d4, 0]`.
//!
//! # Constraints
//!
//! - `GROUP > 0`
//! - `PACK > 0`
//! - `GROUP % PACK == 0`

use std::{alloc::Layout, marker::PhantomData, ptr::NonNull};

use diskann_utils::{
    Reborrow, ReborrowMut,
    strided::StridedView,
    views::{MatrixView, MutMatrixView},
};

use super::matrix::{
    Defaulted, LayoutError, Mat, MatMut, MatRef, NewMut, NewOwned, NewRef, Overflow, Repr, ReprMut,
    ReprOwned, SliceError,
};
use crate::bits::{AsMutPtr, AsPtr, MutSlicePtr, SlicePtr};
use crate::utils;

/// Round `ncols` up to the next multiple of `PACK`.
#[inline]
fn padded_ncols<const PACK: usize>(ncols: usize) -> usize {
    ncols.next_multiple_of(PACK)
}

/// Compute the total number of `T` elements required to store a block-transposed matrix
/// of `nrows x ncols` with group size `GROUP` and packing factor `PACK`.
///
/// This is the **unchecked** flavor — it assumes the caller has already validated that
/// the dimensions do not overflow (e.g. after construction). For use in the constructor,
/// prefer [`checked_compute_capacity`].
///
/// Compile-time constraints (`GROUP > 0`, `PACK > 0`, `GROUP % PACK == 0`) are enforced
/// by [`BlockTransposedRepr::_ASSERTIONS`]; this function does **not** duplicate them.
#[inline]
fn compute_capacity<const GROUP: usize, const PACK: usize>(nrows: usize, ncols: usize) -> usize {
    nrows.next_multiple_of(GROUP) * padded_ncols::<PACK>(ncols)
}

/// Checked variant of [`compute_capacity`] that returns `None` if any intermediate
/// arithmetic overflows. Used by the constructor to reject impossibly large dimensions
/// before committing to an allocation.
#[inline]
fn checked_compute_capacity<const GROUP: usize, const PACK: usize>(
    nrows: usize,
    ncols: usize,
) -> Option<usize> {
    nrows
        .checked_next_multiple_of(GROUP)?
        .checked_mul(ncols.checked_next_multiple_of(PACK)?)
}

/// Compute the linear index for the element at logical `(row, col)` in a block-transposed
/// layout with group size `GROUP`, packing factor `PACK`, and `ncols` logical columns.
#[inline]
fn linear_index<const GROUP: usize, const PACK: usize>(
    row: usize,
    col: usize,
    ncols: usize,
) -> usize {
    let pncols = padded_ncols::<PACK>(ncols);
    let block = row / GROUP;
    let row_in_block = row % GROUP;
    block * GROUP * pncols + (col / PACK) * GROUP * PACK + row_in_block * PACK + (col % PACK)
}

/// Compute the offset from a row's base pointer (at col=0) to the element at `col`.
///
/// This is purely a function of the column index and the const layout parameters, not
/// of any particular matrix's dimensions.
#[inline]
fn col_offset<const GROUP: usize, const PACK: usize>(col: usize) -> usize {
    (col / PACK) * GROUP * PACK + (col % PACK)
}

/// Internal layout descriptor for block-transposed matrices.
///
/// This is not part of the public API — use [`BlockTransposed`], [`BlockTransposedRef`],
/// or [`BlockTransposedMut`] instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockTransposedRepr<T, const GROUP: usize, const PACK: usize = 1> {
    nrows: usize,
    ncols: usize,
    _elem: PhantomData<T>,
}

impl<T: Copy, const GROUP: usize, const PACK: usize> BlockTransposedRepr<T, GROUP, PACK> {
    // Compile-time assertions — evaluated whenever any method references this constant.
    const _ASSERTIONS: () = {
        assert!(GROUP > 0, "group size GROUP must be positive");
        assert!(PACK > 0, "packing factor PACK must be positive");
        assert!(
            GROUP.is_multiple_of(PACK),
            "GROUP must be divisible by PACK"
        );
    };

    /// Create a new `BlockTransposedRepr` descriptor.
    ///
    /// Successful construction requires that the total memory for the backing allocation
    /// does not exceed `isize::MAX`.
    pub fn new(nrows: usize, ncols: usize) -> Result<Self, Overflow> {
        let () = Self::_ASSERTIONS;
        let capacity = checked_compute_capacity::<GROUP, PACK>(nrows, ncols)
            .ok_or_else(|| Overflow::for_type::<T>(nrows, ncols))?;
        Overflow::check_byte_budget::<T>(capacity, nrows, ncols)?;
        Ok(Self {
            nrows,
            ncols,
            _elem: PhantomData,
        })
    }

    // ── Query helpers ────────────────────────────────────────────────

    /// The total number of `T` elements in the backing allocation (including padding).
    #[inline]
    fn storage_len(&self) -> usize {
        compute_capacity::<GROUP, PACK>(self.nrows, self.ncols)
    }

    /// Number of logical rows.
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of logical columns (dimensionality).
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of physical (padded) columns — logical columns rounded up to
    /// the next multiple of `PACK`.
    #[inline]
    pub fn padded_ncols(&self) -> usize {
        padded_ncols::<PACK>(self.ncols)
    }

    /// Number of completely full blocks.
    #[inline]
    pub fn full_blocks(&self) -> usize {
        self.nrows / GROUP
    }

    /// Total number of blocks including a possible partially-filled tail.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.nrows.div_ceil(GROUP)
    }

    /// Number of valid elements in the last block, or 0 if all blocks are full.
    #[inline]
    pub fn remainder(&self) -> usize {
        self.nrows % GROUP
    }

    /// Total number of logical rows rounded up to the next multiple of `GROUP`.
    ///
    /// This is the number of "available" row slots in the backing allocation,
    /// including zero-padded rows in the last (possibly partial) block.
    #[inline]
    pub fn padded_nrows(&self) -> usize {
        self.num_blocks() * GROUP
    }

    /// The stride (in elements) between the start of consecutive blocks.
    #[inline]
    fn block_stride(&self) -> usize {
        GROUP * self.padded_ncols()
    }

    /// The linear offset of the start of `block`.
    #[inline]
    fn block_offset(&self, block: usize) -> usize {
        block * self.block_stride()
    }

    /// Verify that `slice` has exactly `self.storage_len()` elements.
    fn check_slice(&self, slice: &[T]) -> Result<(), SliceError> {
        let cap = self.storage_len();
        if slice.len() != cap {
            Err(SliceError::LengthMismatch {
                expected: cap,
                found: slice.len(),
            })
        } else {
            Ok(())
        }
    }

    /// Helper: wrap a `Box<[T]>` into a [`Mat`] without any further checks.
    ///
    /// # Safety
    ///
    /// `b.len()` must equal `self.storage_len()`.
    unsafe fn box_to_mat(self, b: Box<[T]>) -> Mat<Self> {
        debug_assert_eq!(b.len(), self.storage_len(), "safety contract violated");

        let ptr = utils::box_into_nonnull(b).cast::<u8>();

        // SAFETY: `ptr` is properly aligned and compatible with our layout.
        unsafe { Mat::from_raw_parts(self, ptr) }
    }
}

// ════════════════════════════════════════════════════════════════════
// Row view types
// ════════════════════════════════════════════════════════════════════

/// An immutable view of a single logical row in a block-transposed matrix.
///
/// Because the elements of a logical row are strided (not contiguous), this struct
/// provides indexed access and iteration over the row's elements.
#[derive(Debug, Clone, Copy)]
pub struct Row<'a, T, const GROUP: usize, const PACK: usize = 1> {
    /// Pointer to the element at `(row, col=0)` in the backing allocation.
    base: SlicePtr<'a, T>,
    ncols: usize,
}

impl<T: Copy, const GROUP: usize, const PACK: usize> Row<'_, T, GROUP, PACK> {
    /// Number of elements (columns) in this row.
    #[inline]
    pub fn len(&self) -> usize {
        self.ncols
    }

    /// Whether the row is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ncols == 0
    }

    /// Get a reference to the element at column `col`, or `None` if out of bounds.
    #[inline]
    pub fn get(&self, col: usize) -> Option<&T> {
        if col < self.ncols {
            // SAFETY: bounds checked, offset computed from validated layout.
            Some(unsafe { &*self.base.as_ptr().add(col_offset::<GROUP, PACK>(col)) })
        } else {
            None
        }
    }

    /// Return an iterator over the elements of this row.
    #[inline]
    pub fn iter(&self) -> RowIter<'_, T, GROUP, PACK> {
        RowIter {
            base: self.base,
            col: 0,
            ncols: self.ncols,
        }
    }
}

impl<T: Copy, const GROUP: usize, const PACK: usize> std::ops::Index<usize>
    for Row<'_, T, GROUP, PACK>
{
    type Output = T;

    #[inline]
    #[allow(clippy::panic)] // Index is expected to panic on OOB
    fn index(&self, col: usize) -> &Self::Output {
        self.get(col)
            .unwrap_or_else(|| panic!("column index {col} out of bounds (ncols = {})", self.ncols))
    }
}

/// Iterator over the elements of a [`Row`].
#[derive(Debug, Clone)]
pub struct RowIter<'a, T, const GROUP: usize, const PACK: usize = 1> {
    base: SlicePtr<'a, T>,
    col: usize,
    ncols: usize,
}

impl<T: Copy, const GROUP: usize, const PACK: usize> Iterator for RowIter<'_, T, GROUP, PACK> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.col >= self.ncols {
            return None;
        }
        // SAFETY: col < ncols means the offset is within the backing allocation.
        let val = unsafe { *self.base.as_ptr().add(col_offset::<GROUP, PACK>(self.col)) };
        self.col += 1;
        Some(val)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ncols - self.col;
        (remaining, Some(remaining))
    }
}

impl<T: Copy, const GROUP: usize, const PACK: usize> ExactSizeIterator
    for RowIter<'_, T, GROUP, PACK>
{
}
impl<T: Copy, const GROUP: usize, const PACK: usize> std::iter::FusedIterator
    for RowIter<'_, T, GROUP, PACK>
{
}

/// A mutable view of a single logical row in a block-transposed matrix.
#[derive(Debug)]
pub struct RowMut<'a, T, const GROUP: usize, const PACK: usize = 1> {
    base: MutSlicePtr<'a, T>,
    ncols: usize,
}

impl<T: Copy, const GROUP: usize, const PACK: usize> RowMut<'_, T, GROUP, PACK> {
    /// Number of elements (columns) in this row.
    #[inline]
    pub fn len(&self) -> usize {
        self.ncols
    }

    /// Whether the row is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ncols == 0
    }

    /// Get a reference to the element at column `col`, or `None` if out of bounds.
    #[inline]
    pub fn get(&self, col: usize) -> Option<&T> {
        if col < self.ncols {
            // SAFETY: bounds checked.
            Some(unsafe { &*self.base.as_ptr().add(col_offset::<GROUP, PACK>(col)) })
        } else {
            None
        }
    }

    /// Get a mutable reference to the element at column `col`, or `None` if out of bounds.
    #[inline]
    pub fn get_mut(&mut self, col: usize) -> Option<&mut T> {
        if col < self.ncols {
            // SAFETY: bounds checked.
            Some(unsafe { &mut *self.base.as_mut_ptr().add(col_offset::<GROUP, PACK>(col)) })
        } else {
            None
        }
    }

    /// Set the element at column `col`.
    ///
    /// # Panics
    ///
    /// Panics if `col >= self.len()`.
    #[inline]
    pub fn set(&mut self, col: usize, value: T) {
        assert!(
            col < self.ncols,
            "column index {col} out of bounds (ncols = {})",
            self.ncols
        );
        // SAFETY: bounds checked.
        unsafe { *self.base.as_mut_ptr().add(col_offset::<GROUP, PACK>(col)) = value };
    }
}

impl<T: Copy, const GROUP: usize, const PACK: usize> std::ops::Index<usize>
    for RowMut<'_, T, GROUP, PACK>
{
    type Output = T;

    #[inline]
    #[allow(clippy::panic)] // Index is expected to panic on OOB
    fn index(&self, col: usize) -> &Self::Output {
        self.get(col)
            .unwrap_or_else(|| panic!("column index {col} out of bounds (ncols = {})", self.ncols))
    }
}

impl<T: Copy, const GROUP: usize, const PACK: usize> std::ops::IndexMut<usize>
    for RowMut<'_, T, GROUP, PACK>
{
    #[inline]
    #[allow(clippy::panic)] // IndexMut is expected to panic on OOB
    fn index_mut(&mut self, col: usize) -> &mut Self::Output {
        let ncols = self.ncols;
        self.get_mut(col)
            .unwrap_or_else(|| panic!("column index {col} out of bounds (ncols = {ncols})"))
    }
}

// ════════════════════════════════════════════════════════════════════
// Repr / ReprMut / ReprOwned
// ════════════════════════════════════════════════════════════════════

// SAFETY: `get_row` produces a valid `Row` for valid indices. The layout
// reports the correct capacity for the block-transposed backing allocation.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> Repr
    for BlockTransposedRepr<T, GROUP, PACK>
{
    type Row<'a>
        = Row<'a, T, GROUP, PACK>
    where
        Self: 'a;

    fn nrows(&self) -> usize {
        self.nrows
    }

    fn layout(&self) -> Result<Layout, LayoutError> {
        Ok(Layout::array::<T>(self.storage_len())?)
    }

    unsafe fn get_row<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::Row<'a> {
        debug_assert!(i < self.nrows);

        // When ncols == 0 the backing allocation is zero-sized, so we must not
        // compute any pointer offset.  Return a dangling base instead.
        if self.ncols == 0 {
            return Row {
                // SAFETY: The row is empty (ncols == 0) so the pointer will never be
                // dereferenced. A dangling `NonNull` satisfies the non-null invariant.
                base: unsafe { SlicePtr::new_unchecked(NonNull::dangling()) },
                ncols: 0,
            };
        }

        let base_ptr = ptr.as_ptr().cast::<T>();
        let offset = linear_index::<GROUP, PACK>(i, 0, self.ncols);

        // SAFETY: The caller asserts `i < self.nrows()`. The backing allocation has at
        // least `self.storage_len()` elements, so the computed offset is in bounds.
        let row_base = unsafe { base_ptr.add(offset) };

        Row {
            // SAFETY: `row_base` is derived from a `NonNull<u8>` with a valid offset,
            // so it is non-null. The lifetime is tied to the caller's `'a`.
            base: unsafe { SlicePtr::new_unchecked(NonNull::new_unchecked(row_base)) },
            ncols: self.ncols,
        }
    }
}

// SAFETY: `get_row_mut` produces a valid `RowMut`. Disjoint row indices
// produce disjoint base pointers because each row within a block starts at a unique
// offset modulo GROUP.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> ReprMut
    for BlockTransposedRepr<T, GROUP, PACK>
{
    type RowMut<'a>
        = RowMut<'a, T, GROUP, PACK>
    where
        Self: 'a;

    unsafe fn get_row_mut<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::RowMut<'a> {
        debug_assert!(i < self.nrows);

        // When ncols == 0 the backing allocation is zero-sized, so we must not
        // compute any pointer offset.  Return a dangling base instead.
        if self.ncols == 0 {
            return RowMut {
                // SAFETY: The row is empty (ncols == 0) so the pointer will never be
                // dereferenced. A dangling `NonNull` satisfies the non-null invariant.
                base: unsafe { MutSlicePtr::new_unchecked(NonNull::dangling()) },
                ncols: 0,
            };
        }

        let base_ptr = ptr.as_ptr().cast::<T>();
        let offset = linear_index::<GROUP, PACK>(i, 0, self.ncols);

        // SAFETY: `i < self.nrows` (debug-asserted) guarantees the offset is within
        // the backing allocation. Same reasoning as `get_row`.
        let row_base = unsafe { base_ptr.add(offset) };

        RowMut {
            // SAFETY: `row_base` is derived from a `NonNull<u8>` with a valid offset,
            // so it is non-null. The lifetime is tied to the caller's `'a`.
            base: unsafe { MutSlicePtr::new_unchecked(NonNull::new_unchecked(row_base)) },
            ncols: self.ncols,
        }
    }
}

// SAFETY: Memory is deallocated by reconstructing the `Box<[T]>` that was created during
// `NewOwned`.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> ReprOwned
    for BlockTransposedRepr<T, GROUP, PACK>
{
    unsafe fn drop(self, ptr: NonNull<u8>) {
        // SAFETY: `ptr` was obtained from `Box::into_raw` with length `self.storage_len()`.
        unsafe {
            let slice_ptr =
                std::ptr::slice_from_raw_parts_mut(ptr.cast::<T>().as_ptr(), self.storage_len());
            let _ = Box::from_raw(slice_ptr);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Constructors
// ════════════════════════════════════════════════════════════════════

// SAFETY: The returned `Mat` contains a `Box` with exactly `self.storage_len()` elements.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> NewOwned<T>
    for BlockTransposedRepr<T, GROUP, PACK>
{
    type Error = crate::error::Infallible;

    fn new_owned(self, value: T) -> Result<Mat<Self>, Self::Error> {
        let b: Box<[T]> = vec![value; self.storage_len()].into_boxed_slice();

        // SAFETY: By construction, `b.len() == self.storage_len()`.
        Ok(unsafe { self.box_to_mat(b) })
    }
}

// SAFETY: This safely re-uses `<Self as NewOwned<T>>`.
unsafe impl<T: Copy + Default, const GROUP: usize, const PACK: usize> NewOwned<Defaulted>
    for BlockTransposedRepr<T, GROUP, PACK>
{
    type Error = crate::error::Infallible;

    fn new_owned(self, _: Defaulted) -> Result<Mat<Self>, Self::Error> {
        self.new_owned(T::default())
    }
}

// SAFETY: This checks slice length against storage_len.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> NewRef<T>
    for BlockTransposedRepr<T, GROUP, PACK>
{
    type Error = SliceError;

    fn new_ref(self, data: &[T]) -> Result<MatRef<'_, Self>, Self::Error> {
        self.check_slice(data)?;

        // SAFETY: `check_slice` verified the length.
        Ok(unsafe { MatRef::from_raw_parts(self, utils::as_nonnull(data).cast::<u8>()) })
    }
}

// SAFETY: This checks slice length against storage_len.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> NewMut<T>
    for BlockTransposedRepr<T, GROUP, PACK>
{
    type Error = SliceError;

    fn new_mut(self, data: &mut [T]) -> Result<MatMut<'_, Self>, Self::Error> {
        self.check_slice(data)?;

        // SAFETY: `check_slice` verified the length.
        Ok(unsafe { MatMut::from_raw_parts(self, utils::as_nonnull_mut(data).cast::<u8>()) })
    }
}

// ════════════════════════════════════════════════════════════════════
// Delegation macro
// ════════════════════════════════════════════════════════════════════

/// Generates a forwarding method that delegates to `self.as_view().$name(...)`.
///
/// The generated doc-comment links back to the canonical implementation on
/// [`BlockTransposedRef`], so documentation stays in sync automatically.
macro_rules! delegate_to_ref {
    // Safe function.
    ($(#[$m:meta])* $vis:vis fn $name:ident(&self $(, $a:ident: $t:ty)*) $(-> $r:ty)?) => {
        #[doc = concat!("See [`BlockTransposedRef::", stringify!($name), "`].")]
        $(#[$m])*
        #[inline]
        $vis fn $name(&self $(, $a: $t)*) $(-> $r)? {
            self.as_view().$name($($a),*)
        }
    };
    // Unsafe function.
    ($(#[$m:meta])* unsafe $vis:vis fn $name:ident(&self $(, $a:ident: $t:ty)*) $(-> $r:ty)?) => {
        #[doc = concat!("See [`BlockTransposedRef::", stringify!($name), "`].")]
        $(#[$m])*
        #[inline]
        $vis unsafe fn $name(&self $(, $a: $t)*) $(-> $r)? {
            // SAFETY: Caller upholds the safety contract of the delegated method.
            unsafe { self.as_view().$name($($a),*) }
        }
    };
}

// ════════════════════════════════════════════════════════════════════
// Public wrapper types
// ════════════════════════════════════════════════════════════════════

/// An owning block-transposed matrix.
///
/// Wraps an owned allocation of `T` elements laid out in block-transposed order.
/// See the [module-level documentation](self) for layout details.
///
/// For shared and mutable views, see [`BlockTransposedRef`] and [`BlockTransposedMut`].
///
/// # Row Types
///
/// Because rows are not contiguous in memory, the row types are view structs:
///
/// - [`Row`] — a `Copy` handle supporting `Index<usize>` and `.iter()`.
/// - [`RowMut`] — a mutable handle supporting `IndexMut<usize>`.
#[derive(Debug)]
pub struct BlockTransposed<T: Copy, const GROUP: usize, const PACK: usize = 1> {
    data: Mat<BlockTransposedRepr<T, GROUP, PACK>>,
}

/// A shared (immutable) view of a block-transposed matrix.
///
/// Created by [`BlockTransposed::as_view`].
#[derive(Debug, Clone, Copy)]
pub struct BlockTransposedRef<'a, T: Copy, const GROUP: usize, const PACK: usize = 1> {
    data: MatRef<'a, BlockTransposedRepr<T, GROUP, PACK>>,
}

/// A mutable view of a block-transposed matrix.
///
/// Created by [`BlockTransposed::as_view_mut`].
pub struct BlockTransposedMut<'a, T: Copy, const GROUP: usize, const PACK: usize = 1> {
    data: MatMut<'a, BlockTransposedRepr<T, GROUP, PACK>>,
}

// ── BlockTransposedRef (core read implementations) ───────────────

impl<'a, T: Copy, const GROUP: usize, const PACK: usize> BlockTransposedRef<'a, T, GROUP, PACK> {
    /// Returns the number of logical rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.data.repr().nrows()
    }

    /// Returns the number of logical columns (dimensionality).
    #[inline]
    pub fn ncols(&self) -> usize {
        self.data.repr().ncols()
    }

    /// Returns the number of physical (padded) columns.
    #[inline]
    pub fn padded_ncols(&self) -> usize {
        self.data.repr().padded_ncols()
    }

    /// Group size (blocking factor `GROUP`).
    pub const fn group_size(&self) -> usize {
        GROUP
    }

    /// Group size (blocking factor `GROUP`) as a `const` function on the *type*.
    pub const fn const_group_size() -> usize {
        GROUP
    }

    /// Packing factor `PACK`.
    pub const fn pack_size(&self) -> usize {
        PACK
    }

    /// Number of completely full blocks.
    #[inline]
    pub fn full_blocks(&self) -> usize {
        self.data.repr().full_blocks()
    }

    /// Total number of blocks including any partially-filled tail.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.data.repr().num_blocks()
    }

    /// Number of valid elements in the last partially-full block, or 0 if all
    /// blocks are full.
    #[inline]
    pub fn remainder(&self) -> usize {
        self.data.repr().remainder()
    }

    /// Total number of logical rows rounded up to the next multiple of `GROUP`.
    ///
    /// This is the number of "available" row slots in the backing allocation,
    /// including zero-padded rows in the last (possibly partial) block.
    #[inline]
    pub fn padded_nrows(&self) -> usize {
        self.data.repr().padded_nrows()
    }

    /// Return a raw typed pointer to the start of the backing data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_raw_ptr().cast::<T>()
    }

    /// Return the backing data as a shared slice.
    ///
    /// The returned slice has `storage_len()` elements — this includes all padding
    /// for partial blocks and column-group alignment.
    #[inline]
    pub fn as_slice(&self) -> &'a [T] {
        let len = self.data.repr().storage_len();
        // SAFETY: The backing allocation has exactly `storage_len()` elements of type T.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), len) }
    }

    /// Return a pointer to the start of the given block.
    ///
    /// The caller may assume that for the returned pointer `ptr`,
    /// `[ptr, ptr + GROUP * padded_ncols)` points to valid memory, even for the
    /// remainder block.
    ///
    /// # Safety
    ///
    /// `block` must be less than `self.num_blocks()`. No bounds check is
    /// performed in release builds; callers must verify the index themselves
    /// (e.g. by iterating `0..self.num_blocks()`).
    #[inline]
    pub unsafe fn block_ptr_unchecked(&self, block: usize) -> *const T {
        debug_assert!(block < self.num_blocks());
        // SAFETY: Caller asserts `block < self.num_blocks()`.
        unsafe { self.as_ptr().add(self.data.repr().block_offset(block)) }
    }

    /// Return a view over a full block as a [`MatrixView`].
    ///
    /// The returned view has `padded_ncols / PACK` rows and `GROUP * PACK`
    /// columns. For `PACK == 1` this simplifies to `ncols` rows and `GROUP`
    /// columns (the standard transposed interpretation).
    ///
    /// # Panics
    ///
    /// Panics if `block >= self.full_blocks()`.
    #[allow(clippy::expect_used)]
    pub fn block(&self, block: usize) -> MatrixView<'a, T> {
        assert!(block < self.full_blocks());
        let offset = self.data.repr().block_offset(block);
        let stride = self.data.repr().block_stride();
        // SAFETY: `block < full_blocks()` (asserted above) guarantees
        // `offset + stride` is within the backing allocation.
        let data: &[T] = unsafe { std::slice::from_raw_parts(self.as_ptr().add(offset), stride) };
        MatrixView::try_from(data, self.padded_ncols() / PACK, GROUP * PACK)
            .expect("base data should have been sized correctly")
    }

    /// Return a view over the remainder block, or `None` if there is no
    /// remainder.
    ///
    /// The returned view has the same dimensions as [`block()`](Self::block):
    /// `padded_ncols / PACK` rows and `GROUP * PACK` columns.
    #[allow(clippy::expect_used)]
    pub fn remainder_block(&self) -> Option<MatrixView<'a, T>> {
        if self.remainder() == 0 {
            None
        } else {
            let offset = self.data.repr().block_offset(self.full_blocks());
            let stride = self.data.repr().block_stride();
            // SAFETY: The remainder block exists (`remainder() != 0`),
            // so `offset + stride` is within the backing allocation.
            let data: &[T] =
                unsafe { std::slice::from_raw_parts(self.as_ptr().add(offset), stride) };
            Some(
                MatrixView::try_from(data, self.padded_ncols() / PACK, GROUP * PACK)
                    .expect("base data should have been sized correctly"),
            )
        }
    }

    /// Retrieve the value at the logical `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.nrows()` or `col >= self.ncols()`.
    #[inline]
    pub fn get_element(&self, row: usize, col: usize) -> T {
        assert!(
            row < self.nrows(),
            "row {row} out of bounds (nrows = {})",
            self.nrows()
        );
        assert!(
            col < self.ncols(),
            "col {col} out of bounds (ncols = {})",
            self.ncols()
        );
        let idx = linear_index::<GROUP, PACK>(row, col, self.ncols());
        // SAFETY: bounds checked above.
        unsafe { *self.as_ptr().add(idx) }
    }

    /// Get an immutable row view, or `None` if `i` is out of bounds.
    #[inline]
    pub fn get_row(&self, i: usize) -> Option<Row<'_, T, GROUP, PACK>> {
        self.data.get_row(i)
    }
}

// ── BlockTransposedMut ───────────────────────────────────────────

impl<'a, T: Copy, const GROUP: usize, const PACK: usize> BlockTransposedMut<'a, T, GROUP, PACK> {
    /// Borrow as an immutable [`BlockTransposedRef`].
    #[inline]
    pub fn as_view(&self) -> BlockTransposedRef<'_, T, GROUP, PACK> {
        BlockTransposedRef {
            data: self.data.as_view(),
        }
    }

    // ── Delegated read methods ───────────────────────────────────

    delegate_to_ref!(pub fn nrows(&self) -> usize);
    delegate_to_ref!(pub fn ncols(&self) -> usize);
    delegate_to_ref!(pub fn padded_ncols(&self) -> usize);
    delegate_to_ref!(pub fn full_blocks(&self) -> usize);
    delegate_to_ref!(pub fn num_blocks(&self) -> usize);
    delegate_to_ref!(pub fn remainder(&self) -> usize);
    delegate_to_ref!(pub fn padded_nrows(&self) -> usize);
    delegate_to_ref!(pub fn as_ptr(&self) -> *const T);
    delegate_to_ref!(pub fn as_slice(&self) -> &[T]);
    delegate_to_ref!(#[allow(clippy::missing_safety_doc)] unsafe pub fn block_ptr_unchecked(&self, block: usize) -> *const T);
    delegate_to_ref!(#[allow(clippy::expect_used)] pub fn block(&self, block: usize) -> MatrixView<'_, T>);
    delegate_to_ref!(#[allow(clippy::expect_used)] pub fn remainder_block(&self) -> Option<MatrixView<'_, T>>);
    delegate_to_ref!(pub fn get_element(&self, row: usize, col: usize) -> T);

    /// Group size (blocking factor `GROUP`).
    pub const fn group_size(&self) -> usize {
        GROUP
    }

    /// Group size as `const` function on the *type*.
    pub const fn const_group_size() -> usize {
        GROUP
    }

    /// Packing factor `PACK`.
    pub const fn pack_size(&self) -> usize {
        PACK
    }

    /// Get an immutable row view, or `None` if `i` is out of bounds.
    #[inline]
    pub fn get_row(&self, i: usize) -> Option<Row<'_, T, GROUP, PACK>> {
        self.data.get_row(i)
    }

    // ── Mutable methods ──────────────────────────────────────────
    //
    // The `_inner` variants consume `self` by value so that the lifetime of
    // the returned view is tied to `'a` (the underlying allocation), not to
    // a temporary reborrow. Public `&mut self` methods reborrow into a
    // short-lived `BlockTransposedMut` and then call the inner variant.

    /// Return the backing data as a mutable slice.
    ///
    /// The returned slice has `storage_len()` elements (including all padding).
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.reborrow_mut().mut_slice_inner()
    }

    fn mut_slice_inner(mut self) -> &'a mut [T] {
        let len = self.data.repr().storage_len();
        // SAFETY: We own exclusive access through `self`.
        unsafe { std::slice::from_raw_parts_mut(self.data.as_raw_mut_ptr().cast::<T>(), len) }
    }

    /// Return a mutable view over a full block.
    ///
    /// # Panics
    ///
    /// Panics if `block >= self.full_blocks()`.
    #[allow(clippy::expect_used)]
    pub fn block_mut(&mut self, block: usize) -> MutMatrixView<'_, T> {
        self.reborrow_mut().block_mut_inner(block)
    }

    #[allow(clippy::expect_used)]
    fn block_mut_inner(mut self, block: usize) -> MutMatrixView<'a, T> {
        let repr = *self.data.repr();
        assert!(block < repr.full_blocks());
        let offset = repr.block_offset(block);
        let stride = repr.block_stride();
        let pncols = repr.padded_ncols();
        // SAFETY: `block < full_blocks()`, so the range is within the allocation.
        let data: &mut [T] = unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_raw_mut_ptr().cast::<T>().add(offset),
                stride,
            )
        };
        MutMatrixView::try_from(data, pncols / PACK, GROUP * PACK)
            .expect("base data should have been sized correctly")
    }

    /// Return a mutable view over the remainder block, or `None` if there is no
    /// remainder.
    #[allow(clippy::expect_used)]
    pub fn remainder_block_mut(&mut self) -> Option<MutMatrixView<'_, T>> {
        self.reborrow_mut().remainder_block_mut_inner()
    }

    #[allow(clippy::expect_used)]
    fn remainder_block_mut_inner(mut self) -> Option<MutMatrixView<'a, T>> {
        let repr = *self.data.repr();
        if repr.remainder() == 0 {
            None
        } else {
            let offset = repr.block_offset(repr.full_blocks());
            let stride = repr.block_stride();
            let pncols = repr.padded_ncols();
            // SAFETY: Remainder block exists, so the range is within the allocation.
            let data: &mut [T] = unsafe {
                std::slice::from_raw_parts_mut(
                    self.data.as_raw_mut_ptr().cast::<T>().add(offset),
                    stride,
                )
            };
            Some(
                MutMatrixView::try_from(data, pncols / PACK, GROUP * PACK)
                    .expect("base data should have been sized correctly"),
            )
        }
    }

    /// Get a mutable row view, or `None` if `i` is out of bounds.
    #[inline]
    pub fn get_row_mut(&mut self, i: usize) -> Option<RowMut<'_, T, GROUP, PACK>> {
        self.data.get_row_mut(i)
    }

    // ── Private helpers ──────────────────────────────────────────

    fn reborrow_mut(&mut self) -> BlockTransposedMut<'_, T, GROUP, PACK> {
        BlockTransposedMut {
            data: self.data.reborrow_mut(),
        }
    }
}

// ── BlockTransposed (owned) ──────────────────────────────────────

impl<T: Copy, const GROUP: usize, const PACK: usize> BlockTransposed<T, GROUP, PACK> {
    /// Borrow as an immutable [`BlockTransposedRef`].
    pub fn as_view(&self) -> BlockTransposedRef<'_, T, GROUP, PACK> {
        BlockTransposedRef {
            data: self.data.as_view(),
        }
    }

    /// Borrow as a mutable [`BlockTransposedMut`].
    pub fn as_view_mut(&mut self) -> BlockTransposedMut<'_, T, GROUP, PACK> {
        BlockTransposedMut {
            data: self.data.as_view_mut(),
        }
    }

    // ── Delegated read methods ───────────────────────────────────

    delegate_to_ref!(pub fn nrows(&self) -> usize);
    delegate_to_ref!(pub fn ncols(&self) -> usize);
    delegate_to_ref!(pub fn padded_ncols(&self) -> usize);
    delegate_to_ref!(pub fn full_blocks(&self) -> usize);
    delegate_to_ref!(pub fn num_blocks(&self) -> usize);
    delegate_to_ref!(pub fn remainder(&self) -> usize);
    delegate_to_ref!(pub fn padded_nrows(&self) -> usize);
    delegate_to_ref!(pub fn as_ptr(&self) -> *const T);
    delegate_to_ref!(pub fn as_slice(&self) -> &[T]);
    delegate_to_ref!(#[allow(clippy::missing_safety_doc)] unsafe pub fn block_ptr_unchecked(&self, block: usize) -> *const T);
    delegate_to_ref!(#[allow(clippy::expect_used)] pub fn block(&self, block: usize) -> MatrixView<'_, T>);
    delegate_to_ref!(#[allow(clippy::expect_used)] pub fn remainder_block(&self) -> Option<MatrixView<'_, T>>);
    delegate_to_ref!(pub fn get_element(&self, row: usize, col: usize) -> T);

    /// Group size (blocking factor `GROUP`).
    pub const fn group_size(&self) -> usize {
        GROUP
    }

    /// Group size (blocking factor `GROUP`) as a `const` function on the *type*.
    pub const fn const_group_size() -> usize {
        GROUP
    }

    /// Packing factor `PACK`.
    pub const fn pack_size(&self) -> usize {
        PACK
    }

    /// Get an immutable row view, or `None` if `i` is out of bounds.
    #[inline]
    pub fn get_row(&self, i: usize) -> Option<Row<'_, T, GROUP, PACK>> {
        self.data.get_row(i)
    }

    // ── Mutable methods (delegated to BlockTransposedMut) ────────

    /// See [`BlockTransposedMut::as_mut_slice`].
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_view_mut().mut_slice_inner()
    }

    /// See [`BlockTransposedMut::block_mut`].
    #[allow(clippy::expect_used)]
    pub fn block_mut(&mut self, block: usize) -> MutMatrixView<'_, T> {
        self.as_view_mut().block_mut_inner(block)
    }

    /// See [`BlockTransposedMut::remainder_block_mut`].
    #[allow(clippy::expect_used)]
    pub fn remainder_block_mut(&mut self) -> Option<MutMatrixView<'_, T>> {
        self.as_view_mut().remainder_block_mut_inner()
    }

    /// Get a mutable row view, or `None` if `i` is out of bounds.
    #[inline]
    pub fn get_row_mut(&mut self, i: usize) -> Option<RowMut<'_, T, GROUP, PACK>> {
        self.data.get_row_mut(i)
    }
}

// ── Reborrow ─────────────────────────────────────────────────────

impl<'this, T: Copy, const GROUP: usize, const PACK: usize> Reborrow<'this>
    for BlockTransposed<T, GROUP, PACK>
{
    type Target = BlockTransposedRef<'this, T, GROUP, PACK>;

    #[inline]
    fn reborrow(&'this self) -> Self::Target {
        self.as_view()
    }
}

// ── Factory methods ──────────────────────────────────────────────

impl<T: Copy + Default, const GROUP: usize, const PACK: usize> BlockTransposed<T, GROUP, PACK> {
    /// Construct a default-initialized block-transposed matrix from dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the dimensions overflow the allocation budget.
    #[allow(clippy::expect_used)]
    pub fn new(nrows: usize, ncols: usize) -> Self {
        let repr = BlockTransposedRepr::<T, GROUP, PACK>::new(nrows, ncols)
            .expect("dimensions should not overflow");
        Self {
            data: Mat::new(repr, Defaulted).expect("infallible"),
        }
    }

    /// Fallible variant of [`new`](Self::new).
    pub fn try_new(nrows: usize, ncols: usize) -> Result<Self, Overflow> {
        let repr = BlockTransposedRepr::<T, GROUP, PACK>::new(nrows, ncols)?;
        Ok(Self {
            data: Mat::new(repr, Defaulted).expect("infallible"),
        })
    }

    /// Construct a block-transposed matrix by copying data from a [`StridedView`].
    ///
    /// Each source element at `(row, col)` is placed at the correct offset in the
    /// block-transposed layout. Padding positions (both partial-block rows and
    /// column-group padding when `ncols % PACK != 0`) are filled with
    /// `T::default()`.
    ///
    /// The loop iterates in physical (block-transposed) order — block, column-group,
    /// row-within-block, pack-lane — so that writes to the backing allocation are
    /// sequential. Source reads stride across rows of the [`StridedView`], which is
    /// acceptable because read-side prefetch is more effective than write-side.
    pub fn from_strided(v: StridedView<'_, T>) -> Self {
        let nrows = v.nrows();
        let ncols = v.ncols();
        let mut mat = Self::new(nrows, ncols);

        let repr = *mat.data.repr();
        let num_blocks = repr.num_blocks();
        let pncols = repr.padded_ncols();
        let num_col_groups = pncols / PACK;

        // Walk the backing allocation in physical order so that writes are
        // sequential. The allocation is default-initialized, so padding positions
        // already hold `T::default()` and can be skipped.
        let mut dst = mat.data.as_raw_mut_ptr().cast::<T>();
        for block in 0..num_blocks {
            let row_base = block * GROUP;
            for cg in 0..num_col_groups {
                let col_base = cg * PACK;
                for rib in 0..GROUP {
                    let row = row_base + rib;
                    if row < nrows {
                        // SAFETY: row < nrows is checked by the enclosing `if` condition.
                        let src_row = unsafe { v.get_row_unchecked(row) };
                        for p in 0..PACK {
                            let col = col_base + p;
                            if col < ncols {
                                // SAFETY: dst advances sequentially through the
                                // backing allocation which has exactly `storage_len`
                                // elements, and our loop visits each position once.
                                // `col < ncols` is checked above, and `src_row` has
                                // exactly `ncols` elements.
                                unsafe { *dst = *src_row.get_unchecked(col) };
                            }
                            // SAFETY: dst advances sequentially through the
                            // backing allocation which has exactly `storage_len`
                            // elements, and our loop visits each position once.
                            dst = unsafe { dst.add(1) };
                        }
                    } else {
                        // SAFETY: Entire row is padding — skip PACK positions.
                        // dst remains within the allocation.
                        dst = unsafe { dst.add(PACK) };
                    }
                }
            }
        }

        mat
    }

    /// Construct a block-transposed matrix by copying data from a [`MatrixView`].
    pub fn from_matrix_view(v: MatrixView<'_, T>) -> Self {
        Self::from_strided(v.into())
    }
}

// ════════════════════════════════════════════════════════════════════
// Index<(usize, usize)> for BlockTransposed
// ════════════════════════════════════════════════════════════════════

impl<T: Copy, const GROUP: usize, const PACK: usize> std::ops::Index<(usize, usize)>
    for BlockTransposed<T, GROUP, PACK>
{
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.nrows());
        assert!(col < self.ncols());
        let idx = linear_index::<GROUP, PACK>(row, col, self.ncols());
        // SAFETY: bounds checked above and the backing allocation has `storage_len()` elements.
        unsafe { &*self.as_ptr().add(idx) }
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    //! Test organisation:
    //!
    //!  1. **Helper functions** — `gen_*` element generators.
    //!  2. [`test_full_api`] — single parameterized function that exhaustively
    //!     exercises the full read + write API on all three wrapper types
    //!     (`BlockTransposed`, `BlockTransposedRef`, `BlockTransposedMut`).
    //!  3. **Test runners** — `#[test]` functions that call `test_full_api`
    //!     with various `(T, GROUP, PACK, nrows, ncols)` combinations.
    //!  4. [`test_block_layout_pack1`] — verifies that `PACK=1` blocks are
    //!     the standard row-to-column transposition.
    //!  5. **Focused tests** — edge cases that cannot be expressed as
    //!     parameters to `test_full_api` (`Send`/`Sync`, panic paths,
    //!     non-unit strides, concurrent mutation, etc.).

    use diskann_utils::{lazy_format, views::Matrix};

    use super::*;
    use crate::utils::div_round_up;

    // ── Per-type element generators ──────────────────────────────────
    //
    // Each generator maps a flat index to a non-zero `T` value so that
    // `T::default()` (zero) can be used unambiguously to verify padding.

    fn gen_f32(i: usize) -> f32 {
        (i + 1) as f32
    }
    fn gen_i32(i: usize) -> i32 {
        (i + 1) as i32
    }
    fn gen_u8(i: usize) -> u8 {
        ((i % 255) + 1) as u8
    }

    // ── Unified parameterized test ──────────────────────────────────

    /// Exhaustive test for the full `BlockTransposed` / `BlockTransposedRef` /
    /// `BlockTransposedMut` API surface, parameterized over element type `T`,
    /// group size `GROUP`, and packing factor `PACK`.
    ///
    /// Exercises: construction, query helpers, `Index` / `get_element`,
    /// immutable row views (`Row`), mutable row views (`RowMut`),
    /// `as_slice` / `as_mut_slice`, block views (immutable and mutable),
    /// `remainder_block` / `remainder_block_mut`, `block_ptr_unchecked`,
    /// `from_matrix_view`, OOB `get_row` returns, and both column and row
    /// padding verification.
    fn test_full_api<
        T: Copy + Default + PartialEq + std::fmt::Debug + 'static,
        const GROUP: usize,
        const PACK: usize,
    >(
        nrows: usize,
        ncols: usize,
        gen_element: fn(usize) -> T,
    ) {
        let context = lazy_format!(
            "T={}, GROUP={}, PACK={}, nrows={}, ncols={}",
            std::any::type_name::<T>(),
            GROUP,
            PACK,
            nrows,
            ncols,
        );

        // ── Construction ─────────────────────────────────────────

        let mut data = Matrix::new(T::default(), nrows, ncols);
        data.as_mut_slice()
            .iter_mut()
            .enumerate()
            .for_each(|(i, d)| *d = gen_element(i));

        let mut transpose = BlockTransposed::<T, GROUP, PACK>::from_strided(data.as_view().into());

        let expected_padded = div_round_up(ncols, PACK) * PACK;
        let expected_remainder = nrows % GROUP;
        let storage_len = transpose.as_slice().len();

        // ── Query methods on owned type ──────────────────────────

        assert_eq!(transpose.nrows(), nrows, "{}", context);
        assert_eq!(transpose.ncols(), ncols, "{}", context);
        assert_eq!(transpose.group_size(), GROUP, "{}", context);
        assert_eq!(
            BlockTransposed::<T, GROUP, PACK>::const_group_size(),
            GROUP,
            "{}",
            context
        );
        assert_eq!(transpose.pack_size(), PACK, "{}", context);
        assert_eq!(transpose.full_blocks(), nrows / GROUP, "{}", context);
        assert_eq!(
            transpose.num_blocks(),
            div_round_up(nrows, GROUP),
            "{}",
            context,
        );
        assert_eq!(transpose.remainder(), expected_remainder, "{}", context);
        assert_eq!(transpose.padded_ncols(), expected_padded, "{}", context);

        // ── Element access (owned) ───────────────────────────────

        for row in 0..nrows {
            for col in 0..ncols {
                assert_eq!(
                    data[(row, col)],
                    transpose[(row, col)],
                    "Index at ({}, {}) -- {}",
                    row,
                    col,
                    context,
                );
                assert_eq!(
                    data[(row, col)],
                    transpose.get_element(row, col),
                    "get_element at ({}, {}) -- {}",
                    row,
                    col,
                    context,
                );
            }
        }

        // ── Immutable row views (owned) ──────────────────────────

        let view = transpose.as_view();
        for row in 0..nrows {
            let row_view = view.get_row(row).unwrap();
            assert_eq!(row_view.len(), ncols, "{}", context);
            assert_eq!(row_view.is_empty(), ncols == 0, "{}", context);
            for col in 0..ncols {
                assert_eq!(
                    data[(row, col)],
                    row_view[col],
                    "row view at ({}, {}) -- {}",
                    row,
                    col,
                    context,
                );
            }
            // Row::get — in-bounds + OOB.
            if ncols > 0 {
                assert_eq!(row_view.get(0), Some(&data[(row, 0)]), "{}", context);
            }
            assert_eq!(row_view.get(ncols), None, "{}", context);

            // Iterator + ExactSizeIterator.
            let iter = row_view.iter();
            assert_eq!(iter.len(), ncols, "{}", context);
            let (lo, hi) = iter.size_hint();
            assert_eq!(lo, ncols, "{}", context);
            assert_eq!(hi, Some(ncols), "{}", context);

            let collected: Vec<T> = row_view.iter().collect();
            assert_eq!(collected.len(), ncols, "{}", context);
            for col in 0..ncols {
                assert_eq!(data[(row, col)], collected[col], "{}", context);
            }
        }
        // OOB row returns None.
        assert!(view.get_row(nrows).is_none(), "{}", context);
        let _ = view;

        // ── BlockTransposedRef API ───────────────────────────────

        {
            let view = transpose.as_view();
            assert_eq!(view.nrows(), nrows, "{}", context);
            assert_eq!(view.ncols(), ncols, "{}", context);
            assert_eq!(view.padded_ncols(), expected_padded, "{}", context);
            assert_eq!(view.group_size(), GROUP, "{}", context);
            assert_eq!(
                BlockTransposedRef::<T, GROUP, PACK>::const_group_size(),
                GROUP,
            );
            assert_eq!(view.pack_size(), PACK, "{}", context);
            assert_eq!(view.full_blocks(), nrows / GROUP, "{}", context);
            assert_eq!(view.num_blocks(), div_round_up(nrows, GROUP), "{}", context,);
            assert_eq!(view.remainder(), expected_remainder, "{}", context);
            assert_eq!(view.as_ptr(), transpose.as_ptr(), "{}", context);
            assert_eq!(view.as_slice(), transpose.as_slice(), "{}", context);

            for row in 0..nrows {
                for col in 0..ncols {
                    assert_eq!(
                        data[(row, col)],
                        view.get_element(row, col),
                        "Ref get_element at ({}, {}) -- {}",
                        row,
                        col,
                        context,
                    );
                }
                let row_view = view.get_row(row).unwrap();
                for col in 0..ncols {
                    assert_eq!(data[(row, col)], row_view[col], "{}", context);
                }
            }
            assert!(view.get_row(nrows).is_none(), "{}", context);
        }

        // ── BlockTransposedMut read API ──────────────────────────

        let expected_ptr = transpose.as_ptr();
        {
            let mut_view = transpose.as_view_mut();
            assert_eq!(mut_view.nrows(), nrows, "{}", context);
            assert_eq!(mut_view.ncols(), ncols, "{}", context);
            assert_eq!(mut_view.padded_ncols(), expected_padded, "{}", context);
            assert_eq!(mut_view.group_size(), GROUP, "{}", context);
            assert_eq!(
                BlockTransposedMut::<T, GROUP, PACK>::const_group_size(),
                GROUP,
            );
            assert_eq!(mut_view.pack_size(), PACK, "{}", context);
            assert_eq!(mut_view.full_blocks(), nrows / GROUP, "{}", context);
            assert_eq!(
                mut_view.num_blocks(),
                div_round_up(nrows, GROUP),
                "{}",
                context,
            );
            assert_eq!(mut_view.remainder(), expected_remainder, "{}", context);
            assert_eq!(mut_view.as_ptr(), expected_ptr, "{}", context);
            assert_eq!(mut_view.as_slice().len(), storage_len, "{}", context);

            for row in 0..nrows {
                for col in 0..ncols {
                    assert_eq!(
                        data[(row, col)],
                        mut_view.get_element(row, col),
                        "Mut get_element at ({}, {}) -- {}",
                        row,
                        col,
                        context,
                    );
                }
                let row_view = mut_view.get_row(row).unwrap();
                for col in 0..ncols {
                    assert_eq!(data[(row, col)], row_view[col], "{}", context);
                }
            }
            assert!(mut_view.get_row(nrows).is_none(), "{}", context);
        }

        // ── BlockTransposedMut::as_view() ────────────────────────

        {
            let mut_view = transpose.as_view_mut();
            let ref_from_mut = mut_view.as_view();
            assert_eq!(ref_from_mut.nrows(), nrows, "{}", context);
            for row in 0..nrows {
                for col in 0..ncols {
                    assert_eq!(
                        data[(row, col)],
                        ref_from_mut.get_element(row, col),
                        "{}",
                        context,
                    );
                }
            }
        }

        // ── as_mut_slice ─────────────────────────────────────────

        // Through BlockTransposedMut.
        {
            let mut mut_view = transpose.as_view_mut();
            assert_eq!(mut_view.as_mut_slice().len(), storage_len, "{}", context);
        }
        // Through BlockTransposed (owned).
        assert_eq!(transpose.as_mut_slice().len(), storage_len, "{}", context);

        // ── Immutable block views on all three types ─────────────

        let expected_block_nrows = expected_padded / PACK;
        let expected_block_ncols = GROUP * PACK;

        for b in 0..transpose.full_blocks() {
            let block_data: Vec<T>;
            let ptr: *const T;
            {
                let block = transpose.block(b);
                assert_eq!(block.nrows(), expected_block_nrows, "{}", context);
                assert_eq!(block.ncols(), expected_block_ncols, "{}", context);

                // SAFETY: b < full_blocks <= num_blocks.
                ptr = unsafe { transpose.block_ptr_unchecked(b) };
                assert_eq!(ptr, block.as_slice().as_ptr(), "{}", context);

                block_data = block.as_slice().to_vec();
            }

            // Same block via Ref.
            {
                let view = transpose.as_view();
                assert_eq!(view.block(b).as_slice(), &block_data[..], "{}", context);
                // SAFETY: `b` is in range `0..num_blocks` by the loop bound.
                assert_eq!(unsafe { view.block_ptr_unchecked(b) }, ptr, "{}", context);
            }

            // Same block via Mut (read path).
            {
                let mut_view = transpose.as_view_mut();
                assert_eq!(mut_view.block(b).as_slice(), &block_data[..], "{}", context);
                assert_eq!(
                    // SAFETY: `b` is in range `0..num_blocks` by the loop bound.
                    unsafe { mut_view.block_ptr_unchecked(b) },
                    ptr,
                    "{}",
                    context,
                );
            }
        }

        // Remainder block (immutable, all three types).
        if expected_remainder != 0 {
            let remainder_data: Vec<T>;
            let ptr: *const T;
            let fb = transpose.full_blocks();
            {
                let block = transpose.remainder_block().unwrap();
                assert_eq!(block.nrows(), expected_block_nrows, "{}", context);
                assert_eq!(block.ncols(), expected_block_ncols, "{}", context);

                // SAFETY: fb < num_blocks (remainder exists).
                ptr = unsafe { transpose.block_ptr_unchecked(fb) };
                assert_eq!(ptr, block.as_slice().as_ptr(), "{}", context);

                remainder_data = block.as_slice().to_vec();
            }

            // Via Ref.
            {
                let view = transpose.as_view();
                let ref_block = view.remainder_block().unwrap();
                assert_eq!(ref_block.as_slice(), &remainder_data[..], "{}", context);
            }
            // Via Mut (read path).
            {
                let mut_view = transpose.as_view_mut();
                let mut_block = mut_view.remainder_block().unwrap();
                assert_eq!(mut_block.as_slice(), &remainder_data[..], "{}", context);
            }
        } else {
            assert!(transpose.remainder_block().is_none(), "{}", context);
            {
                let view = transpose.as_view();
                assert!(view.remainder_block().is_none(), "{}", context);
            }
            {
                let mut_view = transpose.as_view_mut();
                assert!(mut_view.remainder_block().is_none(), "{}", context);
            }
        }

        // ── Mutable block views via BlockTransposedMut ───────────

        {
            let mut mut_view = transpose.as_view_mut();
            for b in 0..mut_view.full_blocks() {
                let block_mut = mut_view.block_mut(b);
                assert_eq!(block_mut.nrows(), expected_block_nrows, "{}", context);
                assert_eq!(block_mut.ncols(), expected_block_ncols, "{}", context);
            }
            if expected_remainder != 0 {
                let rem = mut_view.remainder_block_mut().unwrap();
                assert_eq!(rem.nrows(), expected_block_nrows, "{}", context);
                assert_eq!(rem.ncols(), expected_block_ncols, "{}", context);
            } else {
                assert!(mut_view.remainder_block_mut().is_none(), "{}", context);
            }
        }

        // Mutable block views via owned BlockTransposed.
        for b in 0..transpose.full_blocks() {
            let block_mut = transpose.block_mut(b);
            assert_eq!(block_mut.nrows(), expected_block_nrows, "{}", context);
            assert_eq!(block_mut.ncols(), expected_block_ncols, "{}", context);
        }
        if expected_remainder != 0 {
            let rem = transpose.remainder_block_mut().unwrap();
            assert_eq!(rem.nrows(), expected_block_nrows, "{}", context);
            assert_eq!(rem.ncols(), expected_block_ncols, "{}", context);
        } else {
            assert!(transpose.remainder_block_mut().is_none(), "{}", context);
        }

        // ── Mutable row views via BlockTransposedMut ─────────────

        {
            let mut mut_view = transpose.as_view_mut();
            for row in 0..nrows {
                let row_view = mut_view.get_row_mut(row).unwrap();
                assert_eq!(row_view.len(), ncols, "{}", context);
                assert_eq!(row_view.is_empty(), ncols == 0, "{}", context);
                for col in 0..ncols {
                    assert_eq!(data[(row, col)], row_view[col], "{}", context);
                }
            }
            assert!(mut_view.get_row_mut(nrows).is_none(), "{}", context);
        }

        // ── Row::get, RowMut::get, RowMut::get_mut ──────────────

        if nrows > 0 && ncols > 0 {
            // Row::get OOB.
            {
                let view = transpose.as_view();
                let row = view.get_row(0).unwrap();
                assert_eq!(row.get(ncols), None, "{}", context);
                assert_eq!(row.get(usize::MAX), None, "{}", context);
            }

            // RowMut::get OOB.
            let row = transpose.get_row_mut(0).unwrap();
            assert_eq!(row.get(ncols), None, "{}", context);

            // RowMut::get_mut — mutate and verify.
            let mut row = transpose.get_row_mut(0).unwrap();
            let sentinel = gen_element(usize::MAX / 2);
            let original = row[0];
            if let Some(v) = row.get_mut(0) {
                *v = sentinel;
            }
            assert_eq!(row.get_mut(ncols), None, "{}", context);
            // Explicit scope end so the mutable borrow is released before the next access.
            let _ = row;
            assert_eq!(transpose.get_element(0, 0), sentinel, "{}", context);
            // Restore original.
            transpose.get_row_mut(0).unwrap().set(0, original);
        }

        // ── Zero out via block_mut / remainder_block_mut ─────────

        for b in 0..transpose.full_blocks() {
            transpose.block_mut(b).as_mut_slice().fill(T::default());
        }
        if transpose.remainder() != 0 {
            transpose
                .remainder_block_mut()
                .unwrap()
                .as_mut_slice()
                .fill(T::default());
        }
        assert!(
            transpose.as_slice().iter().all(|v| *v == T::default()),
            "not fully zeroed -- {}",
            context,
        );

        // ── Padding verification (fresh construction) ────────────

        let transpose = BlockTransposed::<T, GROUP, PACK>::from_strided(data.as_view().into());
        let raw = transpose.as_slice();

        // Column padding.
        for row in 0..nrows {
            for col in ncols..expected_padded {
                let idx = linear_index::<GROUP, PACK>(row, col, ncols);
                assert_eq!(
                    raw[idx],
                    T::default(),
                    "col padding at ({}, {}) -- {}",
                    row,
                    col,
                    context,
                );
            }
        }

        // Row padding (within partial blocks).
        let padded_nrows = nrows.next_multiple_of(GROUP);
        for row in nrows..padded_nrows {
            for col in 0..expected_padded {
                let idx = linear_index::<GROUP, PACK>(row, col, ncols);
                assert_eq!(
                    raw[idx],
                    T::default(),
                    "row padding at ({}, {}) -- {}",
                    row,
                    col,
                    context,
                );
            }
        }

        // ── padded_nrows() returns padded row count ──────────────

        assert_eq!(
            transpose.as_view().padded_nrows(),
            padded_nrows,
            "padded_nrows() mismatch -- {}",
            context,
        );

        // ── from_matrix_view produces identical results ──────────

        if nrows > 0 && ncols > 0 {
            let via_matrix = BlockTransposed::<T, GROUP, PACK>::from_matrix_view(data.as_view());
            assert_eq!(via_matrix.as_slice(), transpose.as_slice(), "{}", context);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Test runners — each combination gets the full API surface.
    // ════════════════════════════════════════════════════════════════

    #[test]
    fn test_api_pack1_group16() {
        // Miri: boundary rows around GROUP=16 block transitions;
        // full run: exhaustive sweep.
        let rows: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 15, 16, 17, 33]
        } else {
            (0..128).collect()
        };
        let cols: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 2]
        } else {
            (0..5).collect()
        };
        for &nrows in &rows {
            for &ncols in &cols {
                test_full_api::<f32, 16, 1>(nrows, ncols, gen_f32);
            }
        }
    }

    #[test]
    fn test_api_pack1_group8() {
        // Miri: boundary rows around GROUP=8 block transitions;
        // full run: exhaustive sweep.
        let rows: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 7, 8, 9, 17]
        } else {
            (0..128).collect()
        };
        let cols: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 2]
        } else {
            (0..5).collect()
        };
        for &nrows in &rows {
            for &ncols in &cols {
                test_full_api::<f32, 8, 1>(nrows, ncols, gen_f32);
            }
        }
    }

    #[test]
    fn test_api_pack2() {
        // Miri: boundary rows around GROUP=4/8/16 transitions;
        // cols hit PACK=2 boundary (even/odd). Full run: exhaustive.
        let rows: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17]
        } else {
            (0..48).collect()
        };
        let cols: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 2, 3, 4, 5]
        } else {
            (0..9).collect()
        };
        for &nrows in &rows {
            for &ncols in &cols {
                test_full_api::<f32, 4, 2>(nrows, ncols, gen_f32);
                test_full_api::<f32, 8, 2>(nrows, ncols, gen_f32);
                test_full_api::<f32, 16, 2>(nrows, ncols, gen_f32);
            }
        }
    }

    #[test]
    fn test_api_pack4() {
        // Miri: boundary rows around GROUP=4/8/16 transitions;
        // cols hit PACK=4 boundary (0,1,3,4,5,8). Full run: exhaustive.
        let rows: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17]
        } else {
            (0..48).collect()
        };
        let cols: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 3, 4, 5, 8]
        } else {
            (0..9).collect()
        };
        for &nrows in &rows {
            for &ncols in &cols {
                test_full_api::<f32, 4, 4>(nrows, ncols, gen_f32);
                test_full_api::<f32, 8, 4>(nrows, ncols, gen_f32);
                test_full_api::<f32, 16, 4>(nrows, ncols, gen_f32);
            }
        }
    }

    /// Exercise the unified test with non-`f32` element types.
    #[test]
    fn test_api_non_f32() {
        // i32:  PACK=1 and PACK=2
        test_full_api::<i32, 4, 1>(10, 7, gen_i32);
        test_full_api::<i32, 8, 2>(12, 5, gen_i32);

        // u8:   PACK=1 and PACK=2
        test_full_api::<u8, 4, 2>(12, 5, gen_u8);
        test_full_api::<u8, 8, 1>(10, 7, gen_u8);
    }

    // ════════════════════════════════════════════════════════════════
    // Block layout verification (PACK=1 only)
    // ════════════════════════════════════════════════════════════════

    /// Verify that for PACK=1, each block is the standard row-to-column
    /// transposition of a GROUP-row slice of the source matrix.
    fn test_block_layout_pack1<
        T: Copy + Default + PartialEq + std::fmt::Debug + 'static,
        const GROUP: usize,
    >(
        nrows: usize,
        ncols: usize,
        gen_element: fn(usize) -> T,
    ) {
        let mut data = Matrix::new(T::default(), nrows, ncols);
        data.as_mut_slice()
            .iter_mut()
            .enumerate()
            .for_each(|(i, d)| *d = gen_element(i));

        let transpose = BlockTransposed::<T, GROUP, 1>::from_strided(data.as_view().into());

        // Full blocks.
        for b in 0..transpose.full_blocks() {
            let block = transpose.block(b);
            for i in 0..block.nrows() {
                for j in 0..block.ncols() {
                    assert_eq!(
                        block[(i, j)],
                        data[(GROUP * b + j, i)],
                        "block {} at ({}, {}) -- GROUP={}, nrows={}, ncols={}",
                        b,
                        i,
                        j,
                        GROUP,
                        nrows,
                        ncols,
                    );
                }
            }
        }

        // Remainder block.
        if transpose.remainder() != 0 {
            let fb = transpose.full_blocks();
            let block = transpose.remainder_block().unwrap();
            for i in 0..block.nrows() {
                for j in 0..transpose.remainder() {
                    assert_eq!(
                        block[(i, j)],
                        data[(GROUP * fb + j, i)],
                        "remainder at ({}, {}) -- GROUP={}, nrows={}, ncols={}",
                        i,
                        j,
                        GROUP,
                        nrows,
                        ncols,
                    );
                }
            }
        }
    }

    #[test]
    fn test_block_layout_pack1_group16() {
        let rows: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 15, 16, 17, 33]
        } else {
            (0..128).collect()
        };
        let cols: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 2]
        } else {
            (0..5).collect()
        };
        for &nrows in &rows {
            for &ncols in &cols {
                test_block_layout_pack1::<f32, 16>(nrows, ncols, gen_f32);
            }
        }
    }

    #[test]
    fn test_block_layout_pack1_group8() {
        let rows: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 7, 8, 9, 17]
        } else {
            (0..128).collect()
        };
        let cols: Vec<usize> = if cfg!(miri) {
            vec![0, 1, 2]
        } else {
            (0..5).collect()
        };
        for &nrows in &rows {
            for &ncols in &cols {
                test_block_layout_pack1::<f32, 8>(nrows, ncols, gen_f32);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Focused tests (not part of the unified parameterized test)
    // ════════════════════════════════════════════════════════════════

    // ── Send / Sync static assertions ───────────────────────────────

    #[test]
    fn test_row_view_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<Row<'_, f32, 16>>();
        assert_sync::<Row<'_, f32, 16>>();
        assert_send::<Row<'_, u8, 8, 2>>();
        assert_sync::<Row<'_, u8, 8, 2>>();

        assert_send::<RowMut<'_, f32, 16>>();
        assert_sync::<RowMut<'_, f32, 16>>();
        assert_send::<RowMut<'_, i32, 4, 4>>();
        assert_sync::<RowMut<'_, i32, 4, 4>>();
    }

    // ── NewRef / NewMut from raw slices ─────────────────────────────

    #[test]
    fn test_new_ref_and_new_mut() {
        let nrows = 5;
        let ncols = 3;
        let repr = BlockTransposedRepr::<f32, 4>::new(nrows, ncols).unwrap();

        let mat = BlockTransposed::<f32, 4>::new(nrows, ncols);
        let raw: &[f32] = mat.as_slice();

        let mat_ref = BlockTransposedRef {
            data: repr.new_ref(raw).unwrap(),
        };
        assert_eq!(mat_ref.nrows(), nrows);
        assert_eq!(mat_ref.ncols(), ncols);
        for row in 0..nrows {
            for col in 0..ncols {
                assert_eq!(mat_ref.get_element(row, col), mat.get_element(row, col));
            }
        }

        let mut buf = raw.to_vec();
        let mat_mut = BlockTransposedMut {
            data: repr.new_mut(&mut buf).unwrap(),
        };
        assert_eq!(mat_mut.nrows(), nrows);
        assert_eq!(mat_mut.ncols(), ncols);

        // Wrong-length slice should fail.
        let mut short = vec![0.0_f32; 2];
        assert!(repr.new_ref(&short).is_err());
        assert!(repr.new_mut(&mut short).is_err());
    }

    // ── Row view edge cases ─────────────────────────────────────────

    #[test]
    fn test_row_view_empty() {
        /// Verify that immutable and mutable empty-row views are sound for a
        /// given `GROUP`/`PACK` combination.
        fn check_empty<const GROUP: usize, const PACK: usize>() {
            let mut mat = BlockTransposed::<f32, GROUP, PACK>::new(4, 0);

            // Immutable views.
            let view = mat.as_view();
            for i in 0..4 {
                let row = view.get_row(i).unwrap();
                assert!(row.is_empty());
                assert_eq!(row.len(), 0);
                assert_eq!(row.iter().count(), 0);
            }

            // Mutable views.
            for i in 0..4 {
                let row = mat.get_row_mut(i).unwrap();
                assert!(row.is_empty());
                assert_eq!(row.len(), 0);
            }
        }

        check_empty::<16, 1>(); // default PACK
        check_empty::<4, 2>(); // PACK > 1
        check_empty::<4, 4>(); // PACK == GROUP
    }

    // ── Bounds-checking panic tests ─────────────────────────────────

    #[test]
    #[should_panic(expected = "column index 3 out of bounds")]
    fn test_row_view_index_oob() {
        let mat = BlockTransposed::<f32, 4>::new(4, 3);
        let view = mat.as_view();
        let row = view.get_row(0).unwrap();
        let _ = row[3];
    }

    #[test]
    #[should_panic(expected = "column index 3 out of bounds")]
    fn test_row_view_mut_index_oob() {
        let mut mat = BlockTransposed::<f32, 4>::new(4, 3);
        let row = mat.get_row_mut(0).unwrap();
        let _ = row[3];
    }

    #[test]
    #[should_panic(expected = "column index 3 out of bounds")]
    fn test_row_view_mut_index_mut_oob() {
        let mut mat = BlockTransposed::<f32, 4>::new(4, 3);
        let mut row = mat.get_row_mut(0).unwrap();
        row[3] = 1.0;
    }

    #[test]
    #[should_panic(expected = "column index 3 out of bounds")]
    fn test_row_view_set_oob() {
        let mut mat = BlockTransposed::<f32, 4>::new(4, 3);
        let mut row = mat.get_row_mut(0).unwrap();
        row.set(3, 1.0);
    }

    #[test]
    #[should_panic(expected = "row 4 out of bounds")]
    fn test_get_element_row_oob() {
        let mat = BlockTransposed::<f32, 4>::new(4, 3);
        mat.get_element(4, 0);
    }

    #[test]
    #[should_panic(expected = "col 3 out of bounds")]
    fn test_get_element_col_oob() {
        let mat = BlockTransposed::<f32, 4>::new(4, 3);
        mat.get_element(0, 3);
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_index_tuple_row_oob() {
        let mat = BlockTransposed::<f32, 4>::new(4, 3);
        let _ = mat[(4, 0)];
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_index_tuple_col_oob() {
        let mat = BlockTransposed::<f32, 4>::new(4, 3);
        let _ = mat[(0, 3)];
    }

    #[test]
    #[should_panic]
    fn test_block_oob() {
        let mat = BlockTransposed::<f32, 4>::new(4, 3);
        let _ = mat.block(1);
    }

    #[test]
    #[should_panic]
    fn test_block_mut_oob() {
        let mut mat = BlockTransposed::<f32, 4>::new(4, 3);
        let _ = mat.block_mut(1);
    }

    // ── from_strided with non-unit stride ───────────────────────────

    #[test]
    fn test_from_strided_nonunit_stride() {
        use diskann_utils::strided::StridedView;

        const GROUP: usize = 4;
        const PACK: usize = 2;
        let nrows = 5;
        let ncols = 3;
        let cstride = 8;

        let required_len = (nrows - 1) * cstride + ncols;
        let mut flat = vec![0.0_f32; required_len];
        for row in 0..nrows {
            for col in 0..ncols {
                flat[row * cstride + col] = (row * 100 + col + 1) as f32;
            }
        }

        let strided = StridedView::try_shrink_from(&flat, nrows, ncols, cstride)
            .expect("should construct strided view");
        let transpose = BlockTransposed::<f32, GROUP, PACK>::from_strided(strided);

        assert_eq!(transpose.nrows(), nrows);
        assert_eq!(transpose.ncols(), ncols);

        for row in 0..nrows {
            for col in 0..ncols {
                let expected = (row * 100 + col + 1) as f32;
                assert_eq!(
                    transpose[(row, col)],
                    expected,
                    "mismatch at ({}, {})",
                    row,
                    col,
                );
            }
        }

        let padded_ncols = ncols.next_multiple_of(PACK);
        let raw: &[f32] = transpose.as_slice();
        for row in 0..nrows {
            for col in ncols..padded_ncols {
                let idx = linear_index::<GROUP, PACK>(row, col, ncols);
                assert_eq!(
                    raw[idx], 0.0,
                    "column-padding at ({}, {}) should be zero",
                    row, col,
                );
            }
        }
    }

    // ── Concurrent multi-row mutation ───────────────────────────────

    #[test]
    fn test_concurrent_row_mutation() {
        const GROUP: usize = 8;
        const PACK: usize = 2;

        let (nrows, ncols, num_threads) = if cfg!(miri) { (8, 4, 2) } else { (64, 16, 4) };

        let mut mat = BlockTransposed::<f32, GROUP, PACK>::new(nrows, ncols);
        let rows: Vec<RowMut<'_, f32, GROUP, PACK>> = mat.data.rows_mut().collect();
        let rows_per_thread = nrows / num_threads;
        let mut rows = rows.into_boxed_slice();

        std::thread::scope(|s| {
            let mut remaining = &mut rows[..];
            for thread_id in 0..num_threads {
                let chunk_len = if thread_id == num_threads - 1 {
                    remaining.len()
                } else {
                    rows_per_thread
                };
                let (chunk, rest) = remaining.split_at_mut(chunk_len);
                remaining = rest;
                let start_row = thread_id * rows_per_thread;

                s.spawn(move || {
                    for (offset, row_view) in chunk.iter_mut().enumerate() {
                        let row = start_row + offset;
                        for col in 0..ncols {
                            let value = (thread_id * 10000 + row * 100 + col) as f32;
                            row_view.set(col, value);
                        }
                    }
                });
            }
        });

        for row in 0..nrows {
            let thread_id = (row / rows_per_thread).min(num_threads - 1);
            for col in 0..ncols {
                let expected = (thread_id * 10000 + row * 100 + col) as f32;
                assert_eq!(
                    mat.get_element(row, col),
                    expected,
                    "mismatch at ({}, {})",
                    row,
                    col,
                );
            }
        }
    }
}
