// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! [`Repr`] implementation for matrices whose rows are [`slice::Slice`]
//! — a vector of `T` elements paired with per-row metadata of type `M`.
//!
//! [`SliceMatRepr`] describes the layout; construct matrices with
//! [`Mat::new`], [`MatRef::new`], or [`MatMut::new`].

use std::{alloc::Layout, marker::PhantomData, ptr::NonNull};

use thiserror::Error;

use crate::{
    alloc::{AlignedAllocator, Poly},
    meta::slice::{self, SliceMut, SliceRef},
    num::PowerOfTwo,
};

use super::matrix::{
    Defaulted, LayoutError, Mat, MatMut, MatRef, NewCloned, NewMut, NewOwned, NewRef, Repr,
    ReprMut, ReprOwned,
};

/// Representation metadata for a matrix where each row is a [`Slice`](slice::Slice)
/// with element type `T` and metadata type `M`, stored using the canonical layout.
///
/// # Layout
///
/// Each row is stored using the canonical layout from [`SliceRef`], occupying
/// [`SliceRef::canonical_bytes(ncols)`](SliceRef::canonical_bytes) bytes of payload
/// padded to [`SliceRef::canonical_align()`](SliceRef::canonical_align) so that
/// every row starts at a properly aligned offset.
///
/// # Bounds
///
/// Both `T` and `M` must be [`bytemuck::Pod`], which is required by the canonical layout
/// functions in [`crate::meta::slice`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceMatRepr<T, M> {
    nrows: usize,
    ncols: usize,
    _elem: PhantomData<T>,
    _meta: PhantomData<M>,
}

/// Error for [`SliceMatRepr::new`].
#[derive(Debug, Clone, Copy, Error)]
#[non_exhaustive]
pub enum SliceMatReprError {
    /// The total byte size of the matrix exceeds `isize::MAX`.
    #[error(
        "a slice matrix of size {nrows} x {ncols} with row stride {row_stride} \
         would exceed isize::MAX bytes"
    )]
    Overflow {
        nrows: usize,
        ncols: usize,
        row_stride: usize,
    },
}

/// Error for constructing a [`MatRef`] or [`MatMut`] over a `&[u8]` slice.
#[derive(Debug, Clone, Copy, Error)]
#[non_exhaustive]
pub enum SliceMatError {
    /// The length of the provided slice does not match the expected total byte count.
    #[error("length mismatch: expected {expected} bytes, found {found}")]
    LengthMismatch { expected: usize, found: usize },

    /// The provided slice is not properly aligned.
    #[error("alignment mismatch: expected alignment of {expected}")]
    NotAligned { expected: usize },
}

impl<T, M> SliceMatRepr<T, M>
where
    T: bytemuck::Pod,
    M: bytemuck::Pod,
{
    /// Create a new `SliceMatRepr` for `nrows` rows, each with `ncols` elements of type
    /// `T` and metadata of type `M`.
    ///
    /// Returns an error if the total byte size exceeds `isize::MAX`.
    pub fn new(nrows: usize, ncols: usize) -> Result<Self, SliceMatReprError> {
        let stride = Self::stride(ncols);

        // Check that total bytes don't overflow or exceed isize::MAX.
        let total = nrows
            .checked_mul(stride)
            .ok_or(SliceMatReprError::Overflow {
                nrows,
                ncols,
                row_stride: stride,
            })?;

        if total > isize::MAX as usize {
            return Err(SliceMatReprError::Overflow {
                nrows,
                ncols,
                row_stride: stride,
            });
        }

        Ok(Self {
            nrows,
            ncols,
            _elem: PhantomData,
            _meta: PhantomData,
        })
    }

    /// Returns the vector dimension.
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the byte stride per row.
    ///
    /// This is [`canonical_bytes`](SliceRef::canonical_bytes) rounded up to
    /// [`canonical_align`](SliceRef::canonical_align) so that consecutive rows
    /// maintain the required alignment.
    ///
    /// # SAFETY:
    /// - when `canonical_bytes.next_multiple_of(align) >= isize::MAX` this calculation
    ///   will overflow.
    const fn stride(ncols: usize) -> usize {
        let bytes = slice::SliceRef::<T, M>::canonical_bytes(ncols);
        let align = Self::alignment().raw();
        bytes.next_multiple_of(align)
    }

    /// Returns the cached alignment as a [`PowerOfTwo`].
    const fn alignment() -> PowerOfTwo {
        slice::SliceRef::<T, M>::canonical_align()
    }

    /// Returns the total number of bytes for the entire matrix.
    fn total_bytes(&self) -> usize {
        // Safe because `new` already checked this doesn't overflow.
        self.nrows * Self::stride(self.ncols)
    }

    /// Check that `slice` has the correct length and alignment for this representation.
    fn check_slice(&self, slice: &[u8]) -> Result<(), SliceMatError> {
        let expected = self.total_bytes();
        if slice.len() != expected {
            return Err(SliceMatError::LengthMismatch {
                expected,
                found: slice.len(),
            });
        }

        let align = Self::alignment().raw();
        if !(slice.as_ptr() as usize).is_multiple_of(align) {
            return Err(SliceMatError::NotAligned { expected: align });
        }

        Ok(())
    }

    /// Construct a `Mat` around the contents of `b` **without** any checks.
    ///
    /// # Safety
    ///
    /// `b` must have length `self.total_bytes()` and its base pointer must be aligned
    /// to `Self::alignment()`.
    unsafe fn poly_to_mat(self, b: Poly<[u8], AlignedAllocator>) -> Mat<Self> {
        debug_assert_eq!(b.len(), self.total_bytes(), "safety contract violated");

        let (ptr, _allocator) = Poly::into_raw(b);

        let ptr = ptr.cast::<u8>();

        // SAFETY: `ptr` is properly aligned and points to memory with the required layout.
        // The drop logic in `ReprOwned` will reconstruct the `Poly` from this pointer.
        unsafe { Mat::from_raw_parts(self, ptr) }
    }
}

// SAFETY:
// - `get_row` correctly computes the row offset as `i * row_stride` and constructs a valid
//   `SliceRef` using the canonical layout.
// - The `layout` method reports the correct memory layout
// - `SliceRef` borrows the data immutably, so Send/Sync propagation is correct.
unsafe impl<T, M> Repr for SliceMatRepr<T, M>
where
    T: bytemuck::Pod,
    M: bytemuck::Pod,
{
    type Row<'a>
        = SliceRef<'a, T, M>
    where
        Self: 'a;

    fn nrows(&self) -> usize {
        self.nrows
    }

    fn layout(&self) -> Result<Layout, LayoutError> {
        let total = self.total_bytes();
        let align = Self::alignment().raw();

        if total == 0 {
            // Zero-size layouts are valid with any power-of-two alignment.
            Ok(Layout::from_size_align(0, align)?)
        } else {
            Ok(Layout::from_size_align(total, align)?)
        }
    }

    unsafe fn get_row<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::Row<'a> {
        debug_assert!(i < self.nrows);

        let stride = Self::stride(self.ncols);
        let canonical = slice::SliceRef::<T, M>::canonical_bytes(self.ncols);

        // SAFETY:
        // - The caller guarantees `ptr` is valid, aligned and `i < self.nrows`.
        // - `stride` is `canonical_bytes` rounded up to `canonical_align`, so
        //   `i * stride` preserves the base alignment.
        // - We pass `canonical` to `from_canonical_unchecked`,
        //   which expects `canonical_bytes` bytes. Padding is at the end.
        unsafe {
            let row_ptr = ptr.as_ptr().add(i * stride);
            let row_slice = std::slice::from_raw_parts(row_ptr, canonical);
            SliceRef::from_canonical_unchecked(row_slice, self.ncols)
        }
    }
}

// SAFETY:
// - `get_row_mut` correctly computes the row offset and constructs a valid `SliceMut`.
// - For disjoint `i`, the resulting `SliceMut` values reference non-overlapping memory
//   regions because each row occupies a distinct `[i*stride .. (i+1)*stride]` range.
unsafe impl<T, M> ReprMut for SliceMatRepr<T, M>
where
    T: bytemuck::Pod,
    M: bytemuck::Pod,
{
    type RowMut<'a>
        = SliceMut<'a, T, M>
    where
        Self: 'a;

    unsafe fn get_row_mut<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::RowMut<'a> {
        debug_assert!(i < self.nrows);

        let stride = Self::stride(self.ncols);
        let canonical = slice::SliceRef::<T, M>::canonical_bytes(self.ncols);

        // SAFETY:
        // - The caller guarantees `ptr` is valid, aligned, `i < self.nrows`,
        //   and exclusive access to row `i`.
        // - `stride` is `canonical_bytes` rounded up to `canonical_align`, so
        //   `i * stride` preserves alignment and disjoint rows do not overlap.
        // - We pass `canonical_bytes` to `from_canonical_mut_unchecked`.
        unsafe {
            let row_ptr = ptr.as_ptr().add(i * stride);
            let row_slice = std::slice::from_raw_parts_mut(row_ptr, canonical);
            SliceMut::from_canonical_mut_unchecked(row_slice, self.ncols)
        }
    }
}

// SAFETY: The `drop` implementation reconstructs the `Poly<[u8], AlignedAllocator>` from the raw
// pointer and lets it deallocate. This is compatible with all `NewOwned` implementations
// which allocate via `Poly::broadcast` with the same `AlignedAllocator` and alignment.
unsafe impl<T, M> ReprOwned for SliceMatRepr<T, M>
where
    T: bytemuck::Pod,
    M: bytemuck::Pod,
{
    unsafe fn drop(self, ptr: NonNull<u8>) {
        let total = self.total_bytes();
        let align = Self::alignment();

        // SAFETY: The caller guarantees `ptr` was obtained from `NewOwned`, which uses
        // `Poly::broadcast` with `AlignedAllocator::new(align)`. Reconstructing the
        // `Poly` with the same allocator and dropping it is the correct deallocation.
        unsafe {
            let fat_ptr =
                NonNull::new_unchecked(std::ptr::slice_from_raw_parts_mut(ptr.as_ptr(), total));
            let _ = Poly::from_raw(fat_ptr, AlignedAllocator::new(align));
        }
    }
}

//////////////////
// Constructors //
//////////////////

// SAFETY: Allocates via `Poly::broadcast` with correct alignment and size, fills with
// zeros, which is valid for `bytemuck::Pod` types as a default. The resulting pointer
// is non-null and aligned to `Slice::canonical_align()` for `T` and `M`.
// The allocation is valid for `Slice::canonical_bytes() * self.num_rows()` for types `T` and `M`.
unsafe impl<T, M> NewOwned<Defaulted> for SliceMatRepr<T, M>
where
    T: bytemuck::Pod,
    M: bytemuck::Pod,
{
    type Error = crate::alloc::AllocatorError;

    fn new_owned(self, _: Defaulted) -> Result<Mat<Self>, Self::Error> {
        let total = self.total_bytes();
        let align = Self::alignment();

        let buffer = Poly::broadcast(u8::default(), total, AlignedAllocator::new(align))?;

        // SAFETY: `buffer` has length `self.total_bytes()` and is aligned to
        // `Self::alignment()`. All bytes are zero, which is a valid representation
        // for `Pod` types.
        Ok(unsafe { self.poly_to_mat(buffer) })
    }
}

// SAFETY: Validates that the provided slice has the correct length and alignment for the
// canonical layout. The check is sufficient to satisfy the `Repr` contract.
unsafe impl<T, M> NewRef<u8> for SliceMatRepr<T, M>
where
    T: bytemuck::Pod,
    M: bytemuck::Pod,
{
    type Error = SliceMatError;

    fn new_ref(self, slice: &[u8]) -> Result<MatRef<'_, Self>, Self::Error> {
        self.check_slice(slice)?;

        let ptr = crate::utils::as_nonnull(slice).cast::<u8>();

        // SAFETY: We have verified length and alignment via `check_slice`.
        Ok(unsafe { MatRef::from_raw_parts(self, ptr) })
    }
}

// SAFETY: Validates that the provided mutable slice has the correct length and alignment.
// By mutable borrow rules in Rust, we're guaranteed the reference to data is unique until
// the resulting `MatMut` goes out of scope.
unsafe impl<T, M> NewMut<u8> for SliceMatRepr<T, M>
where
    T: bytemuck::Pod,
    M: bytemuck::Pod,
{
    type Error = SliceMatError;

    fn new_mut(self, slice: &mut [u8]) -> Result<MatMut<'_, Self>, Self::Error> {
        self.check_slice(slice)?;

        let ptr = crate::utils::as_nonnull_mut(slice).cast::<u8>();

        // SAFETY: We have verified length and alignment via `check_slice`.
        Ok(unsafe { MatMut::from_raw_parts(self, ptr) })
    }
}

impl<T, M> NewCloned for SliceMatRepr<T, M>
where
    T: bytemuck::Pod,
    M: bytemuck::Pod,
{
    fn new_cloned(v: MatRef<'_, Self>) -> Mat<Self> {
        let repr = *v.repr();
        let total = repr.total_bytes();
        let align = SliceMatRepr::<T, M>::alignment();

        // SAFETY: `total` and `align` come from a repr that backs an existing `MatRef`,
        // so the size/alignment pair is guaranteed valid.
        let layout = unsafe { std::alloc::Layout::from_size_align_unchecked(total, align.raw()) };

        let mut buffer = match Poly::broadcast(u8::default(), total, AlignedAllocator::new(align)) {
            Ok(buf) => buf,
            Err(_) => std::alloc::handle_alloc_error(layout),
        };

        // SAFETY: `v.ptr` points to `total` bytes of valid memory, and `buffer` has the
        // same length. The regions do not overlap because `buffer` is freshly allocated.
        unsafe {
            std::ptr::copy_nonoverlapping(v.ptr.as_ptr(), buffer.as_mut_ptr(), total);
        }

        // SAFETY: `buffer` has the correct length and alignment from above checks.
        unsafe { repr.poly_to_mat(buffer) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use diskann_utils::{Reborrow, ReborrowMut, lazy_format};

    use super::*;

    /// A simple 4-byte metadata type with no special alignment.
    #[derive(Debug, Default, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C)]
    struct Meta {
        scale: f32,
    }

    /// A metadata type with stricter alignment to exercise padding.
    #[derive(Debug, Default, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C, align(8))]
    struct AlignedMeta {
        a: f64,
    }

    /// A zero-sized metadata type.
    #[derive(Debug, Default, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C)]
    struct ZstMeta;

    /// A multi-field metadata type.
    #[derive(Debug, Default, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C)]
    struct RichMeta {
        min: f32,
        max: f32,
    }

    ///////////////////////
    // Helper Functions  //
    ///////////////////////

    const ROWS: &[usize] = &[0, 1, 2, 3, 5, 10];
    const COLS: &[usize] = &[0, 1, 2, 3, 5, 10];

    /// Fill each row of an owned `Mat` with deterministic test values.
    ///
    /// Row `i`, column `j` gets value `(10 * i + j)` cast into `T`.
    /// Metadata for row `i` is `from(i)`.
    fn fill_mat<T, M>(mat: &mut Mat<SliceMatRepr<T, M>>)
    where
        T: bytemuck::Pod + From<u8>,
        M: bytemuck::Pod + TestMeta,
    {
        let nrows = mat.num_vectors();
        let ncols = mat.repr().ncols();
        for i in 0..nrows {
            let mut row = mat.get_row_mut(i).unwrap();
            *row.meta_mut() = M::from_index(i);
            for j in 0..ncols {
                row.vector_mut()[j] = T::from((10 * i + j) as u8);
            }
        }
    }

    /// Fill a `MatMut` with the same deterministic pattern as [`fill_mat`].
    fn fill_mat_mut<T, M>(mat: &mut MatMut<'_, SliceMatRepr<T, M>>)
    where
        T: bytemuck::Pod + From<u8>,
        M: bytemuck::Pod + TestMeta,
    {
        let nrows = mat.num_vectors();
        let ncols = mat.repr().ncols();
        for i in 0..nrows {
            let mut row = mat.get_row_mut(i).unwrap();
            *row.meta_mut() = M::from_index(i);
            for j in 0..ncols {
                row.vector_mut()[j] = T::from((10 * i + j) as u8);
            }
        }
    }

    /// Fill via the `rows_mut` iterator.
    fn fill_rows_mut<T, M>(mat: &mut Mat<SliceMatRepr<T, M>>)
    where
        T: bytemuck::Pod + From<u8>,
        M: bytemuck::Pod + TestMeta,
    {
        let ncols = mat.repr().ncols();
        for (i, mut row) in mat.rows_mut().enumerate() {
            *row.meta_mut() = M::from_index(i);
            for j in 0..ncols {
                row.vector_mut()[j] = T::from((10 * i + j) as u8);
            }
        }
    }

    /// Check all rows of a `Mat` match the deterministic fill pattern.
    fn check_mat<T, M>(mat: &Mat<SliceMatRepr<T, M>>, repr: SliceMatRepr<T, M>, ctx: &dyn Display)
    where
        T: bytemuck::Pod + From<u8> + PartialEq + std::fmt::Debug,
        M: bytemuck::Pod + TestMeta + PartialEq + std::fmt::Debug,
    {
        let align = expected_alignment::<T, M>();
        assert_eq!(mat.num_vectors(), repr.nrows, "nrows mismatch: {ctx}");
        for i in 0..mat.num_vectors() {
            // Check that the row's metadata pointer is correctly aligned by
            // going through the low-level `Repr::get_row` path.
            // SAFETY: `i < nrows` and `mat` owns a valid, aligned buffer.
            unsafe {
                let ptr = NonNull::new_unchecked(mat.as_raw_ptr().cast_mut());
                let row = mat.repr().get_row(ptr, i);
                let meta_ptr = std::ptr::from_ref(row.meta()) as usize;
                assert!(
                    meta_ptr % align == 0,
                    "row {i} meta not {align}-aligned (ptr=0x{meta_ptr:x}): {ctx}"
                );
            }

            let row = mat.get_row(i).unwrap();
            assert_eq!(
                *row.meta(),
                M::from_index(i),
                "meta mismatch row {i}: {ctx}"
            );
            assert_eq!(
                row.vector().len(),
                repr.ncols,
                "ncols mismatch row {i}: {ctx}"
            );
            for j in 0..repr.ncols {
                assert_eq!(
                    row.vector()[j],
                    T::from((10 * i + j) as u8),
                    "data mismatch at [{i}, {j}]: {ctx}"
                );
            }
        }
        // Out-of-bounds access returns None.
        for oob in oob_indices(repr.nrows) {
            assert!(
                mat.get_row(oob).is_none(),
                "expected None for index {oob}: {ctx}"
            );
        }
    }

    /// Check all rows via a `MatRef`.
    fn check_mat_ref<T, M>(
        mat: MatRef<'_, SliceMatRepr<T, M>>,
        repr: SliceMatRepr<T, M>,
        ctx: &dyn Display,
    ) where
        T: bytemuck::Pod + From<u8> + PartialEq + std::fmt::Debug,
        M: bytemuck::Pod + TestMeta + PartialEq + std::fmt::Debug,
    {
        assert_eq!(mat.num_vectors(), repr.nrows, "nrows mismatch: {ctx}");
        for i in 0..mat.num_vectors() {
            let row = mat.get_row(i).unwrap();
            assert_eq!(
                *row.meta(),
                M::from_index(i),
                "meta mismatch row {i}: {ctx}"
            );
            assert_eq!(
                row.vector().len(),
                repr.ncols,
                "ncols mismatch row {i}: {ctx}"
            );
            for j in 0..repr.ncols {
                assert_eq!(
                    row.vector()[j],
                    T::from((10 * i + j) as u8),
                    "data mismatch at [{i}, {j}]: {ctx}"
                );
            }
        }
        for oob in oob_indices(repr.nrows) {
            assert!(
                mat.get_row(oob).is_none(),
                "expected None for index {oob}: {ctx}"
            );
        }
    }

    /// Check all rows via a `MatMut` (read-only check).
    fn check_mat_mut<T, M>(
        mat: MatMut<'_, SliceMatRepr<T, M>>,
        repr: SliceMatRepr<T, M>,
        ctx: &dyn Display,
    ) where
        T: bytemuck::Pod + From<u8> + PartialEq + std::fmt::Debug,
        M: bytemuck::Pod + TestMeta + PartialEq + std::fmt::Debug,
    {
        assert_eq!(mat.num_vectors(), repr.nrows, "nrows mismatch: {ctx}");
        for i in 0..mat.num_vectors() {
            let row = mat.get_row(i).unwrap();
            assert_eq!(
                *row.meta(),
                M::from_index(i),
                "meta mismatch row {i}: {ctx}"
            );
            assert_eq!(
                row.vector().len(),
                repr.ncols,
                "ncols mismatch row {i}: {ctx}"
            );
            for j in 0..repr.ncols {
                assert_eq!(
                    row.vector()[j],
                    T::from((10 * i + j) as u8),
                    "data mismatch at [{i}, {j}]: {ctx}"
                );
            }
        }
        for oob in oob_indices(repr.nrows) {
            assert!(
                mat.get_row(oob).is_none(),
                "expected None for index {oob}: {ctx}"
            );
        }
    }

    /// Check via the `Rows` iterator.
    fn check_rows<T, M>(
        rows: super::super::matrix::Rows<'_, SliceMatRepr<T, M>>,
        repr: SliceMatRepr<T, M>,
        ctx: &dyn Display,
    ) where
        T: bytemuck::Pod + From<u8> + PartialEq + std::fmt::Debug,
        M: bytemuck::Pod + TestMeta + PartialEq + std::fmt::Debug,
    {
        assert_eq!(rows.len(), repr.nrows, "rows len mismatch: {ctx}");
        for (i, row) in rows.enumerate() {
            assert_eq!(
                *row.meta(),
                M::from_index(i),
                "meta mismatch row {i}: {ctx}"
            );
            assert_eq!(
                row.vector().len(),
                repr.ncols,
                "ncols mismatch row {i}: {ctx}"
            );
            for j in 0..repr.ncols {
                assert_eq!(
                    row.vector()[j],
                    T::from((10 * i + j) as u8),
                    "data mismatch at [{i}, {j}]: {ctx}"
                );
            }
        }
    }

    /// Return indices that should be out of bounds for a matrix with `nrows` rows.
    fn oob_indices(nrows: usize) -> Vec<usize> {
        vec![nrows, nrows + 1, nrows + 11, usize::MAX]
    }

    /// Trait for constructing deterministic metadata from a row index.
    trait TestMeta: Sized {
        fn from_index(i: usize) -> Self;
    }

    impl TestMeta for Meta {
        fn from_index(i: usize) -> Self {
            Self { scale: i as f32 }
        }
    }

    impl TestMeta for AlignedMeta {
        fn from_index(i: usize) -> Self {
            Self { a: i as f64 }
        }
    }

    impl TestMeta for ZstMeta {
        fn from_index(_i: usize) -> Self {
            Self
        }
    }

    impl TestMeta for RichMeta {
        fn from_index(i: usize) -> Self {
            Self {
                min: -(i as f32),
                max: i as f32,
            }
        }
    }

    /// Independently computed alignment for `SliceMatRepr<T, M>`.
    ///
    /// This duplicates the logic from `canonical_align` so that tests verify the
    /// production code against a separately-written oracle.
    const fn expected_alignment<T, M>() -> usize {
        let t = std::mem::align_of::<T>();
        let m = std::mem::align_of::<M>();
        if m > t { m } else { t }
    }

    /// Independently computed metadata-prefix size (bytes before the `T` elements).
    const fn expected_meta_prefix<T, M>() -> usize {
        let m_size = std::mem::size_of::<M>();
        if m_size == 0 {
            0
        } else {
            m_size.next_multiple_of(std::mem::align_of::<T>())
        }
    }

    /// Independently computed row stride (canonical bytes per row, padded to alignment).
    const fn expected_stride<T, M>(ncols: usize) -> usize {
        let raw = expected_meta_prefix::<T, M>() + std::mem::size_of::<T>() * ncols;
        let align = expected_alignment::<T, M>();
        raw.next_multiple_of(align)
    }

    /// Independently computed total byte count for the full matrix.
    const fn expected_total_bytes<T, M>(nrows: usize, ncols: usize) -> usize {
        nrows * expected_stride::<T, M>(ncols)
    }

    // Expose private helpers so buffer-allocation helpers can call the production
    // code (we need the *actual* values to create correctly-sized buffers).
    impl<T: bytemuck::Pod, M: bytemuck::Pod> SliceMatRepr<T, M> {
        fn alignment_for_alloc() -> PowerOfTwo {
            Self::alignment()
        }

        fn total_bytes_for_alloc(&self) -> usize {
            self.total_bytes()
        }
    }

    /// Allocate a zeroed, correctly-aligned byte buffer for a given repr.
    fn aligned_buffer<T: bytemuck::Pod, M: bytemuck::Pod>(
        repr: SliceMatRepr<T, M>,
    ) -> Poly<[u8], AlignedAllocator> {
        let total = repr.total_bytes_for_alloc();
        let align = SliceMatRepr::<T, M>::alignment_for_alloc();
        Poly::broadcast(0u8, total, AlignedAllocator::new(align)).expect("test allocation")
    }

    /// Run the full fill-and-check cycle for a given `(T, M)` type combo over all
    /// `ROWS x COLS` sizes.
    fn test_roundtrip<T, M>()
    where
        T: bytemuck::Pod + From<u8> + PartialEq + std::fmt::Debug,
        M: bytemuck::Pod + TestMeta + Default + PartialEq + std::fmt::Debug,
    {
        for &nrows in ROWS {
            for &ncols in COLS {
                let repr = SliceMatRepr::<T, M>::new(nrows, ncols).unwrap();

                // -- Verify repr properties against oracle --
                assert_eq!(repr.nrows, nrows);
                assert_eq!(repr.ncols(), ncols);
                let layout = repr.layout().unwrap();
                assert_eq!(layout.size(), expected_total_bytes::<T, M>(nrows, ncols));
                assert_eq!(
                    repr.total_bytes(),
                    expected_total_bytes::<T, M>(nrows, ncols)
                );
                assert_eq!(layout.align(), expected_alignment::<T, M>());

                // When stride is 0 and nrows > 0, the zero-size allocation may have
                // insufficient alignment for T. This is a known limitation of the
                // allocator, not of SliceMatRepr itself.
                if expected_stride::<T, M>(ncols) == 0 && nrows > 0 {
                    continue;
                }

                let ctx_base = &lazy_format!(
                    "T={}, M={}, nrows={nrows}, ncols={ncols}",
                    std::any::type_name::<T>(),
                    std::any::type_name::<M>()
                );

                // -- Fill via &mut Mat, check all views --
                {
                    let ctx = &lazy_format!("{ctx_base} [fill_mat]");
                    let mut mat = Mat::new(repr, Defaulted).unwrap();
                    fill_mat(&mut mat);
                    check_mat(&mat, repr, ctx);
                    check_mat_ref(mat.reborrow(), repr, ctx);
                    check_mat_mut(mat.reborrow_mut(), repr, ctx);
                    check_rows(mat.rows(), repr, ctx);
                }

                // -- Fill via MatMut --
                {
                    let ctx = &lazy_format!("{ctx_base} [fill_mat_mut]");
                    let mut mat = Mat::new(repr, Defaulted).unwrap();
                    let mut matmut = mat.reborrow_mut();
                    fill_mat_mut(&mut matmut);
                    check_mat(&mat, repr, ctx);
                    check_mat_ref(mat.reborrow(), repr, ctx);
                    check_mat_mut(mat.reborrow_mut(), repr, ctx);
                    check_rows(mat.rows(), repr, ctx);
                }

                // -- Fill via rows_mut iterator --
                {
                    let ctx = &lazy_format!("{ctx_base} [fill_rows_mut]");
                    let mut mat = Mat::new(repr, Defaulted).unwrap();
                    fill_rows_mut(&mut mat);
                    check_mat(&mat, repr, ctx);
                    check_mat_ref(mat.reborrow(), repr, ctx);
                    check_mat_mut(mat.reborrow_mut(), repr, ctx);
                    check_rows(mat.rows(), repr, ctx);
                }
            }
        }
    }

    /// Run clone-related tests for a given `(T, M)` combo across all sizes.
    fn test_clone<T, M>()
    where
        T: bytemuck::Pod + From<u8> + PartialEq + std::fmt::Debug,
        M: bytemuck::Pod + TestMeta + Default + PartialEq + std::fmt::Debug,
    {
        for &nrows in ROWS {
            for &ncols in COLS {
                let repr = SliceMatRepr::<T, M>::new(nrows, ncols).unwrap();

                // See test_roundtrip for rationale.
                if expected_stride::<T, M>(ncols) == 0 && nrows > 0 {
                    continue;
                }

                let ctx_base = &lazy_format!(
                    "T={}, M={}, nrows={nrows}, ncols={ncols}",
                    std::any::type_name::<T>(),
                    std::any::type_name::<M>()
                );

                let mut mat = Mat::new(repr, Defaulted).unwrap();
                fill_mat(&mut mat);

                // Mat::clone
                {
                    let ctx = &lazy_format!("{ctx_base} [Mat::clone]");
                    let cloned = mat.clone();
                    check_mat(&cloned, repr, ctx);
                    check_mat_ref(cloned.reborrow(), repr, ctx);
                    check_rows(cloned.rows(), repr, ctx);
                }

                // MatRef::to_owned
                {
                    let ctx = &lazy_format!("{ctx_base} [MatRef::to_owned]");
                    let owned = mat.as_view().to_owned();
                    check_mat(&owned, repr, ctx);
                }

                // MatMut::to_owned
                {
                    let ctx = &lazy_format!("{ctx_base} [MatMut::to_owned]");
                    let owned = mat.as_view_mut().to_owned();
                    check_mat(&owned, repr, ctx);
                }
            }
        }
    }

    /////////////////////
    // Representation //
    ////////////////////

    #[test]
    fn repr_zero_dimensions() {
        // Zero rows always means zero total bytes.
        for ncols in [0, 3, 5] {
            let repr = SliceMatRepr::<u8, Meta>::new(0, ncols).unwrap();
            assert_eq!(repr.nrows, 0);
            assert_eq!(repr.ncols(), ncols);
            let layout = repr.layout().unwrap();
            assert_eq!(layout.size(), 0, "0 rows should yield 0-size layout");
        }

        // Zero cols but nonzero rows: meta still occupies space.
        let repr = SliceMatRepr::<u8, Meta>::new(3, 0).unwrap();
        assert_eq!(repr.nrows, 3);
        assert_eq!(repr.ncols(), 0);
        let layout = repr.layout().unwrap();
        assert_eq!(
            layout.size(),
            expected_total_bytes::<u8, Meta>(3, 0),
            "ncols=0 but meta has nonzero size"
        );
    }

    #[test]
    fn repr_is_copy() {
        let repr = SliceMatRepr::<u32, Meta>::new(2, 3).unwrap();
        let copy = repr;
        assert_eq!(repr, copy);
    }

    ////////////////////////
    // Construction Errors //
    ////////////////////////

    #[test]
    fn new_rejects_nrows_overflow() {
        // nrows * stride overflows usize.
        assert!(SliceMatRepr::<u8, Meta>::new(usize::MAX, 2).is_err());
    }

    #[test]
    fn new_rejects_isize_max() {
        // Total bytes would exceed isize::MAX.
        let stride = expected_stride::<u64, Meta>(1);
        let half = (isize::MAX as usize / stride) + 1;
        assert!(SliceMatRepr::<u64, Meta>::new(half, 1).is_err());
    }

    #[test]
    fn new_accepts_boundary_below_isize_max() {
        let stride = expected_stride::<u64, Meta>(1);
        let max_rows = isize::MAX as usize / stride;
        let repr = SliceMatRepr::<u64, Meta>::new(max_rows, 1).unwrap();
        assert_eq!(repr.nrows, max_rows);
    }

    ////////////////////////
    // Defaulted (NewOwned)//
    ////////////////////////

    #[test]
    fn new_owned_default_zeroes() {
        for &nrows in ROWS {
            for &ncols in COLS {
                let repr = SliceMatRepr::<u16, Meta>::new(nrows, ncols).unwrap();
                let mat = Mat::new(repr, Defaulted).unwrap();
                assert_eq!(mat.num_vectors(), nrows);
                for i in 0..nrows {
                    let row = mat.get_row(i).unwrap();
                    assert_eq!(*row.meta(), Meta::default());
                    assert!(row.vector().iter().all(|&v| v == 0));
                }
            }
        }
    }

    //////////////////////////
    // Roundtrip (fill/check) //
    //////////////////////////

    #[test]
    fn roundtrip_u8_meta() {
        test_roundtrip::<u8, Meta>();
    }

    #[test]
    fn roundtrip_u16_meta() {
        test_roundtrip::<u16, Meta>();
    }

    #[test]
    fn roundtrip_u32_meta() {
        test_roundtrip::<u32, Meta>();
    }

    #[test]
    fn roundtrip_f32_meta() {
        test_roundtrip::<f32, Meta>();
    }

    #[test]
    fn roundtrip_u16_aligned_meta() {
        test_roundtrip::<u16, AlignedMeta>();
    }

    #[test]
    fn roundtrip_f32_zst_meta() {
        test_roundtrip::<f32, ZstMeta>();
    }

    #[test]
    fn roundtrip_u8_rich_meta() {
        test_roundtrip::<u8, RichMeta>();
    }

    //////////////////////////
    // Clone Independence   //
    //////////////////////////

    #[test]
    fn clone_u8_meta() {
        test_clone::<u8, Meta>();
    }

    #[test]
    fn clone_u16_aligned_meta() {
        test_clone::<u16, AlignedMeta>();
    }

    #[test]
    fn clone_f32_zst_meta() {
        test_clone::<f32, ZstMeta>();
    }

    #[test]
    fn clone_mutation_independence() {
        let repr = SliceMatRepr::<u16, Meta>::new(2, 3).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();
        fill_mat(&mut mat);

        let cloned = mat.clone();

        // Mutate original.
        {
            let mut row = mat.get_row_mut(0).unwrap();
            *row.meta_mut() = Meta { scale: 999.0 };
            row.vector_mut()[0] = 0xFFFF;
        }

        // Clone must be unaffected.
        let orig = mat.get_row(0).unwrap();
        let copy = cloned.get_row(0).unwrap();
        assert_eq!(*orig.meta(), Meta { scale: 999.0 });
        assert_eq!(*copy.meta(), Meta::from_index(0));
        assert_eq!(orig.vector()[0], 0xFFFF);
        assert_eq!(copy.vector()[0], u16::from(0u8));
    }

    //////////////////////////
    // NewRef / NewMut      //
    //////////////////////////

    #[test]
    fn new_ref_roundtrip() {
        for &nrows in ROWS {
            for &ncols in COLS {
                let repr = SliceMatRepr::<u8, Meta>::new(nrows, ncols).unwrap();

                // Zero-size allocations yield a dangling pointer that won't
                // satisfy the alignment check in `check_slice`.
                if expected_total_bytes::<u8, Meta>(nrows, ncols) == 0 {
                    continue;
                }

                let mut buf = aligned_buffer(repr);

                // Fill via MatMut, then verify via MatRef on the same buffer.
                {
                    let mut view = MatMut::new(repr, &mut buf[..]).unwrap();
                    fill_mat_mut(&mut view);
                }

                let view = MatRef::new(repr, &buf[..]).unwrap();
                let ctx = &lazy_format!("NewRef nrows={nrows}, ncols={ncols}");
                check_mat_ref(view, repr, ctx);
            }
        }
    }

    #[test]
    fn new_mut_roundtrip() {
        for &nrows in ROWS {
            for &ncols in COLS {
                let repr = SliceMatRepr::<u8, Meta>::new(nrows, ncols).unwrap();

                if expected_total_bytes::<u8, Meta>(nrows, ncols) == 0 {
                    continue;
                }

                let mut buf = aligned_buffer(repr);

                {
                    let mut view = MatMut::new(repr, &mut buf[..]).unwrap();
                    fill_mat_mut(&mut view);
                }

                let ctx = &lazy_format!("NewMut nrows={nrows}, ncols={ncols}");
                let view = MatMut::new(repr, &mut buf[..]).unwrap();
                check_mat_mut(view, repr, ctx);
            }
        }
    }

    /// Allocate an oversized aligned buffer (total + alignment extra bytes).
    fn oversized_buffer<T: bytemuck::Pod, M: bytemuck::Pod>(
        repr: SliceMatRepr<T, M>,
    ) -> Poly<[u8], AlignedAllocator> {
        let total = repr.total_bytes_for_alloc();
        let align = SliceMatRepr::<T, M>::alignment_for_alloc();
        Poly::broadcast(0u8, total + align.raw(), AlignedAllocator::new(align))
            .expect("test allocation")
    }

    #[test]
    fn new_ref_length_mismatch() {
        let repr = SliceMatRepr::<u32, Meta>::new(3, 4).unwrap();
        let total = expected_total_bytes::<u32, Meta>(3, 4);
        let buf = oversized_buffer(repr);

        // Too short.
        if total > 0 {
            let result = MatRef::<SliceMatRepr<u32, Meta>>::new(repr, &buf[..total - 1]);
            assert!(
                matches!(result, Err(SliceMatError::LengthMismatch { .. })),
                "expected LengthMismatch for too-short slice"
            );
        }

        // Too long.
        let result = MatRef::<SliceMatRepr<u32, Meta>>::new(repr, &buf[..total + 1]);
        assert!(
            matches!(result, Err(SliceMatError::LengthMismatch { .. })),
            "expected LengthMismatch for too-long slice"
        );
    }

    #[test]
    fn new_mut_length_mismatch() {
        let repr = SliceMatRepr::<u32, Meta>::new(3, 4).unwrap();
        let total = expected_total_bytes::<u32, Meta>(3, 4);
        let mut buf = oversized_buffer(repr);

        // Too short.
        if total > 0 {
            let result = MatMut::<SliceMatRepr<u32, Meta>>::new(repr, &mut buf[..total - 1]);
            assert!(
                matches!(result, Err(SliceMatError::LengthMismatch { .. })),
                "expected LengthMismatch for too-short slice"
            );
        }

        // Too long.
        let result = MatMut::<SliceMatRepr<u32, Meta>>::new(repr, &mut buf[..total + 1]);
        assert!(
            matches!(result, Err(SliceMatError::LengthMismatch { .. })),
            "expected LengthMismatch for too-long slice"
        );
    }

    #[test]
    fn new_ref_alignment_mismatch() {
        let repr = SliceMatRepr::<u32, Meta>::new(2, 3).unwrap();
        let total = expected_total_bytes::<u32, Meta>(2, 3);
        let align = expected_alignment::<u32, Meta>();
        let buf = oversized_buffer(repr);

        // Offset by 1 byte to break alignment.
        if total > 0 && align > 1 {
            let result = MatRef::<SliceMatRepr<u32, Meta>>::new(repr, &buf[1..1 + total]);
            assert!(
                matches!(result, Err(SliceMatError::NotAligned { .. })),
                "expected NotAligned for misaligned slice"
            );
        }
    }

    #[test]
    fn new_mut_alignment_mismatch() {
        let repr = SliceMatRepr::<u32, Meta>::new(2, 3).unwrap();
        let total = expected_total_bytes::<u32, Meta>(2, 3);
        let align = expected_alignment::<u32, Meta>();
        let mut buf = oversized_buffer(repr);

        if total > 0 && align > 1 {
            let result = MatMut::<SliceMatRepr<u32, Meta>>::new(repr, &mut buf[1..1 + total]);
            assert!(
                matches!(result, Err(SliceMatError::NotAligned { .. })),
                "expected NotAligned for misaligned slice"
            );
        }
    }

    ////////////
    // Layout //
    ////////////

    #[test]
    fn layout_consistency() {
        for &nrows in ROWS {
            for &ncols in COLS {
                let repr = SliceMatRepr::<u32, AlignedMeta>::new(nrows, ncols).unwrap();
                let layout = repr.layout().unwrap();

                let oracle_total = expected_total_bytes::<u32, AlignedMeta>(nrows, ncols);
                let oracle_align = expected_alignment::<u32, AlignedMeta>();

                if nrows == 0 {
                    assert_eq!(layout.size(), 0);
                } else {
                    assert_eq!(layout.size(), oracle_total);
                }
                assert_eq!(layout.align(), oracle_align);
            }
        }
    }

    #[test]
    fn stride_matches_oracle() {
        for &ncols in COLS {
            let repr = SliceMatRepr::<u16, AlignedMeta>::new(1, ncols).unwrap();
            let layout = repr.layout().unwrap();
            let oracle = expected_stride::<u16, AlignedMeta>(ncols);
            assert_eq!(
                layout.size(),
                oracle,
                "layout size {0} != oracle stride {oracle} for ncols={ncols}",
                layout.size()
            );
        }
    }

    //////////////////////////
    // Reborrow             //
    //////////////////////////

    #[test]
    fn reborrow_write_visible_through_owner() {
        let repr = SliceMatRepr::<f32, Meta>::new(2, 3).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();

        // Write through MatMut reborrow.
        {
            let mut view = mat.reborrow_mut();
            let mut row = view.get_row_mut(0).unwrap();
            *row.meta_mut() = Meta { scale: 7.0 };
            row.vector_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        }

        // Read through MatRef reborrow.
        {
            let view = mat.reborrow();
            let row = view.get_row(0).unwrap();
            assert_eq!(*row.meta(), Meta { scale: 7.0 });
            assert_eq!(row.vector(), &[1.0, 2.0, 3.0]);
        }

        // Verify directly on Mat.
        let row = mat.get_row(0).unwrap();
        assert_eq!(*row.meta(), Meta { scale: 7.0 });
        assert_eq!(row.vector(), &[1.0, 2.0, 3.0]);
    }

    //////////////////////////
    // Row Isolation        //
    //////////////////////////

    #[test]
    fn rows_are_independent() {
        let repr = SliceMatRepr::<u8, Meta>::new(3, 4).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();

        // Write different patterns to adjacent rows.
        for i in 0..3 {
            let mut row = mat.get_row_mut(i).unwrap();
            *row.meta_mut() = Meta {
                scale: (i + 1) as f32 * 10.0,
            };
            row.vector_mut()
                .iter_mut()
                .for_each(|v| *v = (i as u8 + 1) * 0x11);
        }

        // Verify each row is distinct and hasn't bled into neighbours.
        for i in 0..3 {
            let row = mat.get_row(i).unwrap();
            assert_eq!(
                *row.meta(),
                Meta {
                    scale: (i + 1) as f32 * 10.0
                }
            );
            let expected_byte = (i as u8 + 1) * 0x11;
            assert!(
                row.vector().iter().all(|&v| v == expected_byte),
                "row {i} has unexpected data: {:?}",
                row.vector()
            );
        }
    }

    //////////////////////////
    // Multi-Type Roundtrips //
    //////////////////////////

    /// Exercise various (T, M) combos for alignment and padding coverage.
    #[test]
    fn roundtrip_u32_aligned_meta() {
        test_roundtrip::<u32, AlignedMeta>();
    }

    #[test]
    fn roundtrip_u8_zst_meta() {
        test_roundtrip::<u8, ZstMeta>();
    }

    #[test]
    fn clone_u8_rich_meta() {
        test_clone::<u8, RichMeta>();
    }
}
