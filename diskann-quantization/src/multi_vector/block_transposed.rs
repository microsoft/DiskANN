// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Block-transposed matrix representation with configurable packing.
//!
//! This module provides a [`Repr`] implementation for block-transposed matrices, where
//! groups of `GROUP` rows are stored in transposed form to enable efficient SIMD
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
//! With `GROUP = 4` and `PACK = 2`, adjacent column-pairs are interleaved per row
//! within each group panel:
//!
//! ```text
//! Col-group(0,1):  [r0_c0, r0_c1, r1_c0, r1_c1, r2_c0, r2_c1, r3_c0, r3_c1]
//! Col-group(2,3):  [r0_c2, r0_c3, r1_c2, r1_c3, r2_c2, r2_c3, r3_c2, r3_c3]
//! ```
//!
//! If `ncols` is not a multiple of `PACK`, the last column-group is zero-padded.
//!
//! # Constraints
//!
//! - `GROUP > 0`
//! - `PACK > 0`
//! - `GROUP % PACK == 0`

use std::{alloc::Layout, marker::PhantomData, ptr::NonNull};

use diskann_utils::{
    strided::StridedView,
    views::{MatrixView, MutMatrixView},
};

use super::matrix::{
    Defaulted, LayoutError, Mat, MatMut, MatRef, NewMut, NewOwned, NewRef, Overflow, Repr, ReprMut,
    ReprOwned, SliceError,
};
use crate::utils;

/// Round `ncols` up to the next multiple of `PACK`.
#[inline]
fn padded_ncols<const PACK: usize>(ncols: usize) -> usize {
    utils::div_round_up(ncols, PACK) * PACK
}

/// Compute the total number of `T` elements required to store a block-transposed matrix
/// of `nrows x ncols` with group size `GROUP` and packing factor `PACK`.
fn compute_capacity<const GROUP: usize, const PACK: usize>(nrows: usize, ncols: usize) -> usize {
    const {
        assert!(GROUP > 0, "group size GROUP must be positive");
        assert!(PACK > 0, "packing factor PACK must be positive");
        assert!(GROUP % PACK == 0, "GROUP must be divisible by PACK");
    };
    let num_blocks = utils::div_round_up(nrows, GROUP);
    GROUP * padded_ncols::<PACK>(ncols) * num_blocks
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

/// Metadata for block-transposed matrices with optional packing.
///
/// Type parameter `T` is the element type (must be `Copy`), `GROUP` is the blocking
/// factor (number of logical rows per block), and `PACK` is the column packing factor
/// (number of adjacent columns interleaved within each group).
///
/// # Row Types
///
/// Because rows are not contiguous in memory, the row types are proxy structs:
///
/// - `Row<'a>`: [`BlockTransposedRow`] — a `Copy` handle supporting `Index<usize>` and
///   `.iter()`.
/// - `RowMut<'a>`: [`BlockTransposedRowMut`] — a mutable handle supporting
///   `IndexMut<usize>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockTransposed<T, const GROUP: usize, const PACK: usize = 1> {
    nrows: usize,
    ncols: usize,
    _elem: PhantomData<T>,
}

impl<T: Copy, const GROUP: usize, const PACK: usize> BlockTransposed<T, GROUP, PACK> {
    // Compile-time assertions — evaluated whenever any method references this constant.
    const _ASSERTIONS: () = {
        assert!(GROUP > 0, "group size GROUP must be positive");
        assert!(PACK > 0, "packing factor PACK must be positive");
        assert!(GROUP % PACK == 0, "GROUP must be divisible by PACK");
    };

    /// Create a new `BlockTransposed` descriptor.
    ///
    /// Successful construction requires that the total memory for the backing allocation
    /// does not exceed `isize::MAX`.
    pub fn new(nrows: usize, ncols: usize) -> Result<Self, Overflow> {
        let _ = Self::_ASSERTIONS;
        // Use the *capacity* (which includes padding for partial blocks and column
        // padding) for the overflow check so that the layout is always valid.
        let capacity = compute_capacity::<GROUP, PACK>(nrows, ncols);
        // Re-use Overflow::check but with the artificial "capacity x 1" shape so
        // the byte-budget is checked correctly.
        Overflow::check::<T>(capacity, 1)?;
        Ok(Self {
            nrows,
            ncols,
            _elem: PhantomData,
        })
    }

    // ── Query helpers ────────────────────────────────────────────────

    /// The total number of `T` elements in the backing allocation (including padding).
    fn capacity(&self) -> usize {
        compute_capacity::<GROUP, PACK>(self.nrows, self.ncols)
    }

    /// Number of logical rows.
    fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of logical columns (dimensionality).
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of physical (padded) columns — logical columns rounded up to
    /// the next multiple of `PACK`.
    pub fn padded_ncols(&self) -> usize {
        padded_ncols::<PACK>(self.ncols)
    }

    /// The group size (number of logical rows per complete block).
    pub const fn group_size() -> usize {
        GROUP
    }

    /// The packing factor.
    pub const fn pack_size() -> usize {
        PACK
    }

    /// Number of completely full blocks.
    pub fn full_blocks(&self) -> usize {
        self.nrows / GROUP
    }

    /// Total number of blocks including a possible partially-filled tail.
    pub fn num_blocks(&self) -> usize {
        utils::div_round_up(self.nrows, GROUP)
    }

    /// Number of valid elements in the last block, or 0 if all blocks are full.
    pub fn remainder(&self) -> usize {
        self.nrows % GROUP
    }

    /// The stride (in elements) between the start of consecutive blocks.
    fn block_stride(&self) -> usize {
        GROUP * self.padded_ncols()
    }

    /// The linear offset of the start of `block`.
    fn block_offset(&self, block: usize) -> usize {
        block * self.block_stride()
    }

    /// Verify that `slice` has exactly `self.capacity()` elements.
    fn check_slice(&self, slice: &[T]) -> Result<(), SliceError> {
        let cap = self.capacity();
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
    /// `b.len()` must equal `self.capacity()`.
    unsafe fn box_to_mat(self, b: Box<[T]>) -> Mat<Self> {
        debug_assert_eq!(b.len(), self.capacity(), "safety contract violated");

        // SAFETY: Box guarantees the returned pointer is non-null.
        let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(b)) }.cast::<u8>();

        // SAFETY: `ptr` is properly aligned and compatible with our layout.
        unsafe { Mat::from_raw_parts(self, ptr) }
    }
}

// ════════════════════════════════════════════════════════════════════
// Row proxy types
// ════════════════════════════════════════════════════════════════════

/// An immutable view of a single logical row in a block-transposed matrix.
///
/// Because the elements of a logical row are strided (not contiguous), this struct
/// acts as a lightweight proxy that supports indexed access and iteration.
#[derive(Debug, Clone, Copy)]
pub struct BlockTransposedRow<'a, T, const GROUP: usize, const PACK: usize = 1> {
    /// Pointer to the element at `(row, col=0)` in the backing allocation.
    base: *const T,
    ncols: usize,
    _lifetime: PhantomData<&'a T>,
}

impl<T: Copy, const GROUP: usize, const PACK: usize> BlockTransposedRow<'_, T, GROUP, PACK> {
    /// Number of elements (columns) in this row.
    pub fn len(&self) -> usize {
        self.ncols
    }

    /// Whether the row is empty.
    pub fn is_empty(&self) -> bool {
        self.ncols == 0
    }

    /// Get the element at column `col`, or `None` if out of bounds.
    pub fn get(&self, col: usize) -> Option<T> {
        if col < self.ncols {
            // SAFETY: bounds checked, offset computed from validated layout.
            Some(unsafe { *self.base.add(col_offset::<GROUP, PACK>(col)) })
        } else {
            None
        }
    }

    /// Return an iterator over the elements of this row.
    pub fn iter(&self) -> BlockTransposedRowIter<'_, T, GROUP, PACK> {
        BlockTransposedRowIter {
            base: self.base,
            col: 0,
            ncols: self.ncols,
            _lifetime: PhantomData,
        }
    }
}

impl<T: Copy, const GROUP: usize, const PACK: usize> std::ops::Index<usize>
    for BlockTransposedRow<'_, T, GROUP, PACK>
{
    type Output = T;

    fn index(&self, col: usize) -> &Self::Output {
        assert!(
            col < self.ncols,
            "column index {col} out of bounds (ncols = {})",
            self.ncols
        );
        // SAFETY: bounds checked.
        unsafe { &*self.base.add(col_offset::<GROUP, PACK>(col)) }
    }
}

/// Iterator over the elements of a [`BlockTransposedRow`].
#[derive(Debug, Clone)]
pub struct BlockTransposedRowIter<'a, T, const GROUP: usize, const PACK: usize = 1> {
    base: *const T,
    col: usize,
    ncols: usize,
    _lifetime: PhantomData<&'a T>,
}

impl<T: Copy, const GROUP: usize, const PACK: usize> Iterator
    for BlockTransposedRowIter<'_, T, GROUP, PACK>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.col >= self.ncols {
            return None;
        }
        // SAFETY: col < ncols means the offset is within the backing allocation.
        let val = unsafe { *self.base.add(col_offset::<GROUP, PACK>(self.col)) };
        self.col += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ncols - self.col;
        (remaining, Some(remaining))
    }
}

impl<T: Copy, const GROUP: usize, const PACK: usize> ExactSizeIterator
    for BlockTransposedRowIter<'_, T, GROUP, PACK>
{
}
impl<T: Copy, const GROUP: usize, const PACK: usize> std::iter::FusedIterator
    for BlockTransposedRowIter<'_, T, GROUP, PACK>
{
}

/// A mutable view of a single logical row in a block-transposed matrix.
#[derive(Debug)]
pub struct BlockTransposedRowMut<'a, T, const GROUP: usize, const PACK: usize = 1> {
    base: *mut T,
    ncols: usize,
    _lifetime: PhantomData<&'a mut T>,
}

impl<T: Copy, const GROUP: usize, const PACK: usize> BlockTransposedRowMut<'_, T, GROUP, PACK> {
    /// Number of elements (columns) in this row.
    pub fn len(&self) -> usize {
        self.ncols
    }

    /// Whether the row is empty.
    pub fn is_empty(&self) -> bool {
        self.ncols == 0
    }

    /// Get the element at column `col`, or `None` if out of bounds.
    pub fn get(&self, col: usize) -> Option<T> {
        if col < self.ncols {
            // SAFETY: bounds checked.
            Some(unsafe { *self.base.add(col_offset::<GROUP, PACK>(col)) })
        } else {
            None
        }
    }

    /// Set the element at column `col`.
    ///
    /// # Panics
    ///
    /// Panics if `col >= self.len()`.
    pub fn set(&mut self, col: usize, value: T) {
        assert!(
            col < self.ncols,
            "column index {col} out of bounds (ncols = {})",
            self.ncols
        );
        // SAFETY: bounds checked.
        unsafe { *self.base.add(col_offset::<GROUP, PACK>(col)) = value };
    }
}

impl<T: Copy, const GROUP: usize, const PACK: usize> std::ops::Index<usize>
    for BlockTransposedRowMut<'_, T, GROUP, PACK>
{
    type Output = T;

    fn index(&self, col: usize) -> &Self::Output {
        assert!(
            col < self.ncols,
            "column index {col} out of bounds (ncols = {})",
            self.ncols
        );
        // SAFETY: bounds checked.
        unsafe { &*self.base.add(col_offset::<GROUP, PACK>(col)) }
    }
}

impl<T: Copy, const GROUP: usize, const PACK: usize> std::ops::IndexMut<usize>
    for BlockTransposedRowMut<'_, T, GROUP, PACK>
{
    fn index_mut(&mut self, col: usize) -> &mut Self::Output {
        assert!(
            col < self.ncols,
            "column index {col} out of bounds (ncols = {})",
            self.ncols
        );
        // SAFETY: bounds checked.
        unsafe { &mut *self.base.add(col_offset::<GROUP, PACK>(col)) }
    }
}

// ════════════════════════════════════════════════════════════════════
// Repr / ReprMut / ReprOwned
// ════════════════════════════════════════════════════════════════════

// SAFETY: `get_row` produces a valid `BlockTransposedRow` for valid indices. The layout
// reports the correct capacity for the block-transposed backing allocation.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> Repr
    for BlockTransposed<T, GROUP, PACK>
{
    type Row<'a>
        = BlockTransposedRow<'a, T, GROUP, PACK>
    where
        Self: 'a;

    fn nrows(&self) -> usize {
        self.nrows
    }

    fn layout(&self) -> Result<Layout, LayoutError> {
        Ok(Layout::array::<T>(self.capacity())?)
    }

    unsafe fn get_row<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::Row<'a> {
        debug_assert!(i < self.nrows);

        let base_ptr = ptr.as_ptr().cast::<T>();
        let offset = linear_index::<GROUP, PACK>(i, 0, self.ncols);

        // SAFETY: The caller asserts `i < self.nrows()`. The backing allocation has at
        // least `self.capacity()` elements, so the computed offset is in bounds.
        let row_base = unsafe { base_ptr.add(offset) };

        BlockTransposedRow {
            base: row_base,
            ncols: self.ncols,
            _lifetime: PhantomData,
        }
    }
}

// SAFETY: `get_row_mut` produces a valid `BlockTransposedRowMut`. Disjoint row indices
// produce disjoint base pointers because each row within a block starts at a unique
// offset modulo GROUP.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> ReprMut
    for BlockTransposed<T, GROUP, PACK>
{
    type RowMut<'a>
        = BlockTransposedRowMut<'a, T, GROUP, PACK>
    where
        Self: 'a;

    unsafe fn get_row_mut<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::RowMut<'a> {
        debug_assert!(i < self.nrows);

        let base_ptr = ptr.as_ptr().cast::<T>();
        let offset = linear_index::<GROUP, PACK>(i, 0, self.ncols);

        // SAFETY: Same reasoning as `get_row`.
        let row_base = unsafe { base_ptr.add(offset) as *mut T };

        BlockTransposedRowMut {
            base: row_base,
            ncols: self.ncols,
            _lifetime: PhantomData,
        }
    }
}

// SAFETY: Memory is deallocated by reconstructing the `Box<[T]>` that was created during
// `NewOwned`.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> ReprOwned
    for BlockTransposed<T, GROUP, PACK>
{
    unsafe fn drop(self, ptr: NonNull<u8>) {
        // SAFETY: `ptr` was obtained from `Box::into_raw` with length `self.capacity()`.
        unsafe {
            let slice_ptr =
                std::ptr::slice_from_raw_parts_mut(ptr.cast::<T>().as_ptr(), self.capacity());
            let _ = Box::from_raw(slice_ptr);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Constructors
// ════════════════════════════════════════════════════════════════════

// SAFETY: The returned `Mat` contains a `Box` with exactly `self.capacity()` elements.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> NewOwned<T>
    for BlockTransposed<T, GROUP, PACK>
{
    type Error = crate::error::Infallible;

    fn new_owned(self, value: T) -> Result<Mat<Self>, Self::Error> {
        let b: Box<[T]> = (0..self.capacity()).map(|_| value).collect();

        // SAFETY: By construction, `b.len() == self.capacity()`.
        Ok(unsafe { self.box_to_mat(b) })
    }
}

// SAFETY: This safely re-uses `<Self as NewOwned<T>>`.
unsafe impl<T: Copy + Default, const GROUP: usize, const PACK: usize> NewOwned<Defaulted>
    for BlockTransposed<T, GROUP, PACK>
{
    type Error = crate::error::Infallible;

    fn new_owned(self, _: Defaulted) -> Result<Mat<Self>, Self::Error> {
        self.new_owned(T::default())
    }
}

// SAFETY: This checks slice length against capacity.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> NewRef<T>
    for BlockTransposed<T, GROUP, PACK>
{
    type Error = SliceError;

    fn new_ref(self, data: &[T]) -> Result<MatRef<'_, Self>, Self::Error> {
        self.check_slice(data)?;

        // SAFETY: `check_slice` verified the length.
        Ok(unsafe { MatRef::from_raw_parts(self, utils::as_nonnull(data).cast::<u8>()) })
    }
}

// SAFETY: This checks slice length against capacity.
unsafe impl<T: Copy, const GROUP: usize, const PACK: usize> NewMut<T>
    for BlockTransposed<T, GROUP, PACK>
{
    type Error = SliceError;

    fn new_mut(self, data: &mut [T]) -> Result<MatMut<'_, Self>, Self::Error> {
        self.check_slice(data)?;

        // SAFETY: `check_slice` verified the length.
        Ok(unsafe { MatMut::from_raw_parts(self, utils::as_nonnull_mut(data).cast::<u8>()) })
    }
}

// ════════════════════════════════════════════════════════════════════
// Inherent methods on Mat / MatRef / MatMut
// ════════════════════════════════════════════════════════════════════

/// Shared helper methods available on all three matrix flavours.
macro_rules! impl_block_transposed_accessors {
    ($ty:ident $(<$lt:lifetime>)? , $ptr_method:ident, $cast:ty) => {
        impl<$($lt,)? T: Copy, const GROUP: usize, const PACK: usize>
            $ty<$($lt,)? BlockTransposed<T, GROUP, PACK>>
        {
            /// Returns the number of logical rows.
            pub fn nrows(&self) -> usize {
                self.repr.nrows()
            }

            /// Returns the number of logical columns (dimensionality).
            pub fn ncols(&self) -> usize {
                self.repr.ncols()
            }

            /// Returns the number of physical (padded) columns.
            pub fn padded_ncols(&self) -> usize {
                self.repr.padded_ncols()
            }

            /// Alias for [`ncols`](Self::ncols) — the number of logical columns.
            pub fn block_size(&self) -> usize {
                self.repr.ncols()
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
            pub fn full_blocks(&self) -> usize {
                self.repr.full_blocks()
            }

            /// Total number of blocks including any partially-filled tail.
            pub fn num_blocks(&self) -> usize {
                self.repr.num_blocks()
            }

            /// Number of valid elements in the last partially-full block, or 0 if all
            /// blocks are full.
            pub fn remainder(&self) -> usize {
                self.repr.remainder()
            }

            /// Return a raw typed pointer to the start of the backing data.
            pub fn as_ptr(&self) -> *const T {
                self.ptr.as_ptr().cast::<T>()
            }

            /// Return a pointer to the start of the given block.
            ///
            /// The caller may assume that for the returned pointer `ptr`,
            /// `[ptr, ptr + GROUP * padded_ncols)` points to valid memory, even for the
            /// remainder block.
            ///
            /// # Safety
            ///
            /// `block` must be less than `self.num_blocks()`.
            pub unsafe fn block_ptr_unchecked(&self, block: usize) -> *const T {
                debug_assert!(block < self.num_blocks());
                // SAFETY: Caller asserts `block < self.num_blocks()`.
                unsafe { self.as_ptr().add(self.repr.block_offset(block)) }
            }

            /// Return a view over a full block as a [`MatrixView`].
            ///
            /// The returned view has `ncols` rows and `GROUP` columns (the transposed
            /// layout within a block). This interpretation is only meaningful when
            /// `PACK == 1`; for `PACK > 1` the in-block layout is interleaved and
            /// this view would not reflect the logical structure.
            ///
            /// # Panics
            ///
            /// Panics if `block >= self.full_blocks()`.
            #[allow(clippy::expect_used)]
            pub fn block(&self, block: usize) -> MatrixView<'_, T>
            where
                T: Default,
            {
                debug_assert!(PACK == 1, "block() view is only correct for PACK == 1");
                assert!(block < self.full_blocks());
                let offset = self.repr.block_offset(block);
                let stride = self.repr.block_stride();
                let data: &[T] = unsafe {
                    std::slice::from_raw_parts(self.as_ptr().add(offset), stride)
                };
                MatrixView::try_from(data, self.ncols(), GROUP)
                    .expect("base data should have been sized correctly")
            }

            /// Return a view over the remainder block, or `None` if there is no remainder.
            ///
            /// This interpretation is only meaningful when `PACK == 1`.
            #[allow(clippy::expect_used)]
            pub fn remainder_block(&self) -> Option<MatrixView<'_, T>>
            where
                T: Default,
            {
                debug_assert!(PACK == 1, "remainder_block() view is only correct for PACK == 1");
                if self.remainder() == 0 {
                    None
                } else {
                    let offset = self.repr.block_offset(self.full_blocks());
                    let stride = self.repr.block_stride();
                    let data: &[T] = unsafe {
                        std::slice::from_raw_parts(self.as_ptr().add(offset), stride)
                    };
                    Some(
                        MatrixView::try_from(data, self.ncols(), GROUP)
                            .expect("base data should have been sized correctly")
                    )
                }
            }

            /// Retrieve the value at the logical `(row, col)`.
            ///
            /// # Panics
            ///
            /// Panics if `row >= self.nrows()` or `col >= self.ncols()`.
            pub fn get_element(&self, row: usize, col: usize) -> T {
                assert!(row < self.nrows(), "row {row} out of bounds (nrows = {})", self.nrows());
                assert!(col < self.ncols(), "col {col} out of bounds (ncols = {})", self.ncols());
                let idx = linear_index::<GROUP, PACK>(row, col, self.ncols());
                // SAFETY: bounds checked above.
                unsafe { *self.as_ptr().add(idx) }
            }
        }
    };
}

impl_block_transposed_accessors!(Mat, as_raw_ptr, *const u8);
impl_block_transposed_accessors!(MatRef<'a>, as_raw_ptr, *const u8);
impl_block_transposed_accessors!(MatMut<'a>, as_raw_ptr, *const u8);

/// Mutable-specific block helpers.
impl<T: Copy + Default, const GROUP: usize, const PACK: usize>
    Mat<BlockTransposed<T, GROUP, PACK>>
{
    /// Return a mutable view over a full block.
    ///
    /// This interpretation is only meaningful when `PACK == 1`.
    ///
    /// # Panics
    ///
    /// Panics if `block >= self.full_blocks()`.
    #[allow(clippy::expect_used)]
    pub fn block_mut(&mut self, block: usize) -> MutMatrixView<'_, T> {
        debug_assert!(PACK == 1, "block_mut() view is only correct for PACK == 1");
        assert!(block < self.full_blocks());
        let offset = self.repr.block_offset(block);
        let stride = self.repr.block_stride();
        let ncols = self.ncols();
        let data: &mut [T] = unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast::<T>().add(offset), stride)
        };
        MutMatrixView::try_from(data, ncols, GROUP)
            .expect("base data should have been sized correctly")
    }

    /// Return a mutable view over the remainder block, or `None` if there is no remainder.
    ///
    /// This interpretation is only meaningful when `PACK == 1`.
    #[allow(clippy::expect_used)]
    pub fn remainder_block_mut(&mut self) -> Option<MutMatrixView<'_, T>> {
        debug_assert!(
            PACK == 1,
            "remainder_block_mut() view is only correct for PACK == 1"
        );
        if self.remainder() == 0 {
            None
        } else {
            let offset = self.repr.block_offset(self.full_blocks());
            let stride = self.repr.block_stride();
            let ncols = self.ncols();
            let data: &mut [T] = unsafe {
                std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast::<T>().add(offset), stride)
            };
            Some(
                MutMatrixView::try_from(data, ncols, GROUP)
                    .expect("base data should have been sized correctly"),
            )
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Factory methods (from_strided, from_matrix_view)
// ════════════════════════════════════════════════════════════════════

impl<const GROUP: usize, const PACK: usize> Mat<BlockTransposed<f32, GROUP, PACK>> {
    /// Construct a zero-initialized block-transposed matrix from dimensions.
    #[allow(clippy::expect_used)]
    pub fn new_block_transposed(nrows: usize, ncols: usize) -> Self {
        let repr = BlockTransposed::<f32, GROUP, PACK>::new(nrows, ncols)
            .expect("dimensions should not overflow");
        Mat::new(repr, Defaulted).expect("infallible")
    }

    /// Construct a block-transposed matrix by copying data from a `StridedView`.
    ///
    /// Each source element at `(row, col)` is placed at the correct offset in the
    /// block-transposed layout. Padding positions (both partial-block rows and
    /// column-group padding when `ncols % PACK != 0`) are zero.
    pub fn from_strided(v: StridedView<'_, f32>) -> Self {
        let nrows = v.nrows();
        let ncols = v.ncols();
        let mat = Self::new_block_transposed(nrows, ncols);

        // Fill using linear_index. The allocation is zero-initialized so padding
        // positions are already zero.
        let base: *mut f32 = mat.ptr.as_ptr().cast::<f32>();
        for row in 0..nrows {
            for col in 0..ncols {
                let idx = linear_index::<GROUP, PACK>(row, col, ncols);
                // SAFETY: idx < capacity by construction, base points to a valid allocation.
                unsafe { *base.add(idx) = v[(row, col)] };
            }
        }

        mat
    }

    /// Construct a block-transposed matrix by copying data from a `MatrixView`.
    pub fn from_matrix_view(v: MatrixView<'_, f32>) -> Self {
        Self::from_strided(v.into())
    }
}

// ════════════════════════════════════════════════════════════════════
// Index<(usize, usize)> for Mat
// ════════════════════════════════════════════════════════════════════

impl<T: Copy, const GROUP: usize, const PACK: usize> std::ops::Index<(usize, usize)>
    for Mat<BlockTransposed<T, GROUP, PACK>>
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.nrows());
        assert!(col < self.ncols());
        let idx = linear_index::<GROUP, PACK>(row, col, self.ncols());
        // SAFETY: bounds checked above and the backing allocation has `capacity()` elements.
        unsafe { &*self.as_ptr().add(idx) }
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use diskann_utils::{lazy_format, views::Matrix};

    use super::*;
    use crate::utils::div_round_up;

    /// Exhaustive test for `PACK = 1` (the standard block-transpose layout).
    fn test_block_transposed_impl<const GROUP: usize>(nrows: usize, ncols: usize) {
        let context = lazy_format!("GROUP = {}, nrows = {}, ncols = {}", GROUP, nrows, ncols);

        // Create initial data:
        //       0         1         2 ...   ncols-1
        //   ncols   ncols+1   ncols+2     2*ncols-1
        // ...
        let mut data = Matrix::new(0.0, nrows, ncols);
        data.as_mut_slice()
            .iter_mut()
            .enumerate()
            .for_each(|(i, d)| *d = i as f32);

        let mut transpose: Mat<BlockTransposed<f32, GROUP>> =
            Mat::from_strided(data.as_view().into());

        // Verify query methods.
        assert_eq!(transpose.nrows(), nrows, "{}", context);
        assert_eq!(transpose.ncols(), ncols, "{}", context);
        assert_eq!(transpose.block_size(), ncols, "{}", context);
        assert_eq!(transpose.group_size(), GROUP, "{}", context);
        assert_eq!(transpose.full_blocks(), nrows / GROUP, "{}", context);
        assert_eq!(
            transpose.num_blocks(),
            div_round_up(nrows, GROUP),
            "{}",
            context
        );
        assert_eq!(transpose.remainder(), nrows % GROUP, "{}", context);
        assert_eq!(transpose.padded_ncols(), ncols, "{}", context); // PACK=1 → no padding

        // Check regular row-column indexing.
        for row in 0..nrows {
            for col in 0..ncols {
                assert_eq!(
                    data[(row, col)],
                    transpose[(row, col)],
                    "failed for (row, col) = ({}, {})",
                    row,
                    col,
                );
            }
        }

        // Check get_element matches Index.
        for row in 0..nrows {
            for col in 0..ncols {
                assert_eq!(
                    data[(row, col)],
                    transpose.get_element(row, col),
                    "get_element failed for ({}, {})",
                    row,
                    col,
                );
            }
        }

        // Check row proxy.
        let view = transpose.as_view();
        for row in 0..nrows {
            let proxy = view.get_row(row).unwrap();
            assert_eq!(proxy.len(), ncols);
            for col in 0..ncols {
                assert_eq!(
                    data[(row, col)],
                    proxy[col],
                    "row proxy failed for ({}, {})",
                    row,
                    col,
                );
            }
            // Test iterator.
            let collected: Vec<f32> = proxy.iter().collect();
            assert_eq!(collected.len(), ncols);
            for col in 0..ncols {
                assert_eq!(data[(row, col)], collected[col]);
            }
        }

        // Check block-level views (only valid for PACK=1, the default here).
        for b in 0..transpose.full_blocks() {
            let block = transpose.block(b);
            assert_eq!(block.nrows(), ncols);
            assert_eq!(block.ncols(), GROUP);

            for i in 0..block.nrows() {
                for j in 0..block.ncols() {
                    assert_eq!(
                        block[(i, j)],
                        data[(GROUP * b + j, i)],
                        "failed in block {}, row {}, col {} -- {}",
                        b,
                        i,
                        j,
                        context,
                    );
                }
            }

            // Check block_ptr_unchecked.
            // SAFETY: b < full_blocks < num_blocks.
            let ptr = unsafe { transpose.block_ptr_unchecked(b) };
            assert_eq!(ptr, block.as_slice().as_ptr());

            // Mutable access — zero out the block.
            let mut block_mut = transpose.block_mut(b);
            assert_eq!(ptr, block_mut.as_slice().as_ptr());
            block_mut.as_mut_slice().fill(0.0);
        }

        let expected_remainder = nrows % GROUP;
        if expected_remainder != 0 {
            let b = transpose.full_blocks();
            let block = transpose.remainder_block().unwrap();
            assert_eq!(block.nrows(), ncols);
            assert_eq!(block.ncols(), GROUP);

            for i in 0..block.nrows() {
                for j in 0..expected_remainder {
                    assert_eq!(
                        block[(i, j)],
                        data[(GROUP * b + j, i)],
                        "failed in block {}, row {}, col {} -- {}",
                        b,
                        i,
                        j,
                        context,
                    );
                }
            }

            // SAFETY: b < num_blocks.
            let ptr = unsafe { transpose.block_ptr_unchecked(b) };
            assert_eq!(ptr, block.as_slice().as_ptr());

            let mut block_mut = transpose.remainder_block_mut().unwrap();
            assert_eq!(ptr, block_mut.as_slice().as_ptr());
            block_mut.as_mut_slice().fill(0.0);
        } else {
            assert!(transpose.remainder_block().is_none());
            assert!(transpose.remainder_block_mut().is_none());
        }

        // The entire backing store should now be zeroed.
        let raw: &[f32] =
            unsafe { std::slice::from_raw_parts(transpose.as_ptr(), transpose.repr.capacity()) };
        assert!(raw.iter().all(|i| *i == 0.0));
    }

    #[test]
    fn test_block_transposed_16() {
        // Miri interprets at MIR level and is ~100-1000× slower than native execution,
        // so we test only a single representative case under Miri to keep CI feasible.
        let row_range = if cfg!(miri) { 127..128 } else { 0..128 };
        let column_range = if cfg!(miri) { 4..5 } else { 0..5 };

        for nrows in row_range {
            for ncols in column_range.clone() {
                test_block_transposed_impl::<16>(nrows, ncols);
            }
        }
    }

    #[test]
    fn test_block_transposed_8() {
        // Narrow ranges under Miri to avoid excessive runtime (see test_block_transposed_16).
        let row_range = if cfg!(miri) { 127..128 } else { 0..128 };
        let column_range = if cfg!(miri) { 4..5 } else { 0..5 };

        for nrows in row_range {
            for ncols in column_range.clone() {
                test_block_transposed_impl::<8>(nrows, ncols);
            }
        }
    }

    #[test]
    fn test_row_proxy_empty() {
        // A 0-column matrix should yield empty row proxies.
        let mat: Mat<BlockTransposed<f32, 16>> = Mat::new_block_transposed(4, 0);
        let view = mat.as_view();
        for i in 0..4 {
            let row = view.get_row(i).unwrap();
            assert!(row.is_empty());
            assert_eq!(row.len(), 0);
            assert_eq!(row.iter().count(), 0);
        }
    }

    // ── PACK > 1 tests ──────────────────────────────────────────────

    /// Test block-transposed with PACK > 1 for various dimensions.
    fn test_packed_impl<const GROUP: usize, const PACK: usize>(nrows: usize, ncols: usize) {
        let context = lazy_format!(
            "GROUP = {}, PACK = {}, nrows = {}, ncols = {}",
            GROUP,
            PACK,
            nrows,
            ncols,
        );

        // Use nonzero values so we can distinguish from padding.
        let mut data = Matrix::new(0.0, nrows, ncols);
        data.as_mut_slice()
            .iter_mut()
            .enumerate()
            .for_each(|(i, d)| *d = (i + 1) as f32);

        let transpose: Mat<BlockTransposed<f32, GROUP, PACK>> =
            Mat::from_strided(data.as_view().into());

        // Check query methods.
        assert_eq!(transpose.nrows(), nrows, "{}", context);
        assert_eq!(transpose.ncols(), ncols, "{}", context);
        assert_eq!(transpose.group_size(), GROUP, "{}", context);
        assert_eq!(transpose.pack_size(), PACK, "{}", context);

        let expected_padded = div_round_up(ncols, PACK) * PACK;
        assert_eq!(transpose.padded_ncols(), expected_padded, "{}", context);

        assert_eq!(transpose.full_blocks(), nrows / GROUP, "{}", context);
        assert_eq!(
            transpose.num_blocks(),
            div_round_up(nrows, GROUP),
            "{}",
            context
        );
        assert_eq!(transpose.remainder(), nrows % GROUP, "{}", context);

        // Check row-column indexing via Index.
        for row in 0..nrows {
            for col in 0..ncols {
                assert_eq!(
                    data[(row, col)],
                    transpose[(row, col)],
                    "Index failed for (row, col) = ({}, {}) -- {}",
                    row,
                    col,
                    context,
                );
            }
        }

        // Check get_element.
        for row in 0..nrows {
            for col in 0..ncols {
                assert_eq!(
                    data[(row, col)],
                    transpose.get_element(row, col),
                    "get_element failed for ({}, {}) -- {}",
                    row,
                    col,
                    context,
                );
            }
        }

        // Check row proxy + iterator.
        let view = transpose.as_view();
        for row in 0..nrows {
            let proxy = view.get_row(row).unwrap();
            assert_eq!(proxy.len(), ncols, "{}", context);
            for col in 0..ncols {
                assert_eq!(
                    data[(row, col)],
                    proxy[col],
                    "row proxy failed for ({}, {}) -- {}",
                    row,
                    col,
                    context,
                );
            }
            let collected: Vec<f32> = proxy.iter().collect();
            assert_eq!(collected.len(), ncols, "{}", context);
            for col in 0..ncols {
                assert_eq!(data[(row, col)], collected[col], "{}", context);
            }
        }

        // Verify padding positions are zero.
        let raw: &[f32] =
            unsafe { std::slice::from_raw_parts(transpose.as_ptr(), transpose.repr.capacity()) };
        // Check that padded "columns" within each data row are zero.
        for row in 0..nrows {
            for col in ncols..expected_padded {
                let idx = linear_index::<GROUP, PACK>(row, col, ncols);
                assert_eq!(
                    raw[idx], 0.0,
                    "padding at ({}, {}) should be zero -- {}",
                    row, col, context,
                );
            }
        }
    }

    #[test]
    fn test_block_transposed_pack2() {
        // Narrow ranges under Miri to avoid excessive runtime (see test_block_transposed_16).
        let row_range = if cfg!(miri) { 31..32 } else { 0..48 };
        let column_range = if cfg!(miri) { 4..5 } else { 0..9 };

        for nrows in row_range {
            for ncols in column_range.clone() {
                test_packed_impl::<4, 2>(nrows, ncols);
                test_packed_impl::<8, 2>(nrows, ncols);
                test_packed_impl::<16, 2>(nrows, ncols);
            }
        }
    }

    #[test]
    fn test_block_transposed_pack4() {
        // Narrow ranges under Miri to avoid excessive runtime (see test_block_transposed_16).
        let row_range = if cfg!(miri) { 31..32 } else { 0..48 };
        let column_range = if cfg!(miri) { 4..5 } else { 0..9 };

        for nrows in row_range {
            for ncols in column_range.clone() {
                test_packed_impl::<4, 4>(nrows, ncols);
                test_packed_impl::<8, 4>(nrows, ncols);
                test_packed_impl::<16, 4>(nrows, ncols);
            }
        }
    }
}
