// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

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
/// Each row occupies [`SliceRef::<T, M>::canonical_bytes(ncols)`] bytes, with alignment
/// [`SliceRef::<T, M>::canonical_align()`]. Rows are packed contiguously.
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
        let total = nrows.checked_mul(stride).ok_or(SliceMatReprError::Overflow {
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

    /// Returns the number of elements per row (the vector dimension).
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the cached byte stride per row.
    const fn stride(ncols : usize) -> usize {
        slice::SliceRef::<T, M>::canonical_bytes(ncols)
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
// - `SliceRef` using the canonical layout. The `layout` method reports the correct memory
//    layout. `SliceRef` borrows the data immutably, so Send/Sync propagation is correct.
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

        let stride = Self::stride(self.ncols());
        let row_ptr = ptr.as_ptr().add(i * stride);

        // SAFETY: 
        // - Base pointer `ptr` was initialized correctly by `NewRef`
        //   and so is within bounds. 
        // - `stride` was computed based on [`SliceRef::<T, M>::canonical_bytes`] and 
        //   was used for initialization also.
        let row_slice = std::slice::from_raw_parts(row_ptr, stride);

        // SAFETY: The row slice has the correct length and alignment for the canonical
        // layout because:
        // - The base pointer is aligned to [`SliceRef::canonical_align()`].
        // - Each row starts at `i * stride`, where `stride` is a multiple of the
        //   canonical alignment, thus preserving alignment.
        SliceRef::from_canonical_unchecked(row_slice, self.ncols)
    }
}

// SAFETY:
//
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

        let stride = Self::stride(self.ncols());
        
        let row_ptr = ptr.as_ptr().add(i * stride);

        // SAEFTY: 
        // - Base pointer `ptr` was initialized correctly by `NewMut` 
        //   and so is within bounds. 
        // - `stride` was computed based on [`SliceRef::<T, M>::canonical_bytes`] and 
        //   was used for initialization also.
        let row_slice = std::slice::from_raw_parts_mut(row_ptr, stride);

        // SAFETY: Same alignment and length guarantees as `get_row`. Additionally,
        // the caller guarantees exclusive access to row `i`, and disjoint rows do not
        // overlap because each row occupies its own `stride`-sized region (stride >=
        // canonical, and rows are at stride-aligned offsets).
        SliceMut::from_canonical_mut_unchecked(row_slice, self.ncols)
    }
}

// SAFETY: The `drop` implementation reconstructs the `Poly<[u8], AlignedAllocator>` from the raw
// pointer and lets it deallocate. This is compatible with all `NewOwned` implementations
// which allocate via `Poly::broadcast` with the same `AlignedAllocator`.
unsafe impl<T, M> ReprOwned for SliceMatRepr<T, M>
where
    T: bytemuck::Pod,
    M: bytemuck::Pod,
{
    unsafe fn drop(self, ptr: NonNull<u8>) {
        let total = self.total_bytes();
        let align = Self::alignment();

        // Reconstruct the fat pointer for the byte slice.
        let fat_ptr = NonNull::new_unchecked(std::ptr::slice_from_raw_parts_mut(
            ptr.as_ptr(),
            total,
        ));

        // SAFETY: The caller guarantees `ptr` was obtained from `NewOwned`, which uses
        // `Poly::broadcast` with `AlignedAllocator::new(align)`. Reconstructing the
        // `Poly` with the same allocator and dropping it is the correct deallocation.
        let _ = Poly::from_raw(fat_ptr, AlignedAllocator::new(align));
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

        let buffer = Poly::broadcast(0u8, total, AlignedAllocator::new(align))?;

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

        // Allocate a new buffer and copy the raw bytes.
        let mut buffer = Poly::broadcast(0u8, total, AlignedAllocator::new(align))
            .expect("allocation failed in NewCloned");

        // SAFETY: `v.ptr` points to `total` bytes of valid memory, and `buffer` has the
        // same length. The regions do not overlap because `buffer` is freshly allocated.
        unsafe {
            std::ptr::copy_nonoverlapping(v.ptr.as_ptr(), buffer.as_mut_ptr(), total);
        }

        // SAFETY: `buffer` has the correct length and alignment.
        unsafe { repr.poly_to_mat(buffer) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::{Reborrow, ReborrowMut};

    use super::*;

    /// A simple test metadata type.
    #[derive(Debug, Default, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C)]
    struct Meta {
        scale: f32,
    }

    #[test]
    fn new_and_basic_accessors() {
        let repr = SliceMatRepr::<f32, Meta>::new(3, 4).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();

        assert_eq!(mat.num_vectors(), 3);

        // All rows should be zeroed (default).
        for i in 0..3 {
            let row = mat.get_row(i).unwrap();
            assert_eq!(*row.meta(), Meta::default());
            assert_eq!(row.vector(), &[0.0f32; 4]);
        }

        // Write to each row.
        for i in 0..3 {
            let mut row = mat.get_row_mut(i).unwrap();
            *row.meta_mut() = Meta { scale: i as f32 };
            row.vector_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(j, v)| *v = (10 * i + j) as f32);
        }

        // Verify reads.
        for i in 0..3 {
            let row = mat.get_row(i).unwrap();
            assert_eq!(*row.meta(), Meta { scale: i as f32 });
            for j in 0..4 {
                assert_eq!(row.vector()[j], (10 * i + j) as f32);
            }
        }
    }

    #[test]
    fn out_of_bounds_returns_none() {
        let repr = SliceMatRepr::<u8, Meta>::new(2, 3).unwrap();
        let mat = Mat::new(repr, Defaulted).unwrap();

        assert!(mat.get_row(0).is_some());
        assert!(mat.get_row(1).is_some());
        assert!(mat.get_row(2).is_none());
        assert!(mat.get_row(usize::MAX).is_none());
    }

    #[test]
    fn clone_produces_independent_copy() {
        let repr = SliceMatRepr::<u16, Meta>::new(2, 3).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();

        // Write data.
        {
            let mut row = mat.get_row_mut(0).unwrap();
            *row.meta_mut() = Meta { scale: 1.5 };
            row.vector_mut().copy_from_slice(&[10, 20, 30]);
        }
        {
            let mut row = mat.get_row_mut(1).unwrap();
            *row.meta_mut() = Meta { scale: 2.5 };
            row.vector_mut().copy_from_slice(&[40, 50, 60]);
        }

        let cloned = mat.clone();

        // Verify the clone has the same data.
        for i in 0..2 {
            let orig = mat.get_row(i).unwrap();
            let copy = cloned.get_row(i).unwrap();
            assert_eq!(*orig.meta(), *copy.meta());
            assert_eq!(orig.vector(), copy.vector());
        }

        // Mutating original should not affect clone.
        {
            let mut row = mat.get_row_mut(0).unwrap();
            *row.meta_mut() = Meta { scale: 99.0 };
            row.vector_mut()[0] = 999;
        }

        let orig_row = mat.get_row(0).unwrap();
        let clone_row = cloned.get_row(0).unwrap();
        assert_eq!(*orig_row.meta(), Meta { scale: 99.0 });
        assert_eq!(*clone_row.meta(), Meta { scale: 1.5 });
        assert_eq!(orig_row.vector()[0], 999);
        assert_eq!(clone_row.vector()[0], 10);
    }

    #[test]
    fn zero_dimensions() {
        // Zero rows.
        let repr = SliceMatRepr::<f32, Meta>::new(0, 5).unwrap();
        let mat = Mat::new(repr, Defaulted).unwrap();
        assert_eq!(mat.num_vectors(), 0);
        assert!(mat.get_row(0).is_none());

        // Zero cols.
        let repr = SliceMatRepr::<f32, Meta>::new(3, 0).unwrap();
        let mat = Mat::new(repr, Defaulted).unwrap();
        assert_eq!(mat.num_vectors(), 3);
        for i in 0..3 {
            let row = mat.get_row(i).unwrap();
            assert_eq!(row.vector().len(), 0);
            assert_eq!(*row.meta(), Meta::default());
        }

        // Zero rows and zero cols.
        let repr = SliceMatRepr::<f32, Meta>::new(0, 0).unwrap();
        let mat = Mat::new(repr, Defaulted).unwrap();
        assert_eq!(mat.num_vectors(), 0);
    }

    #[test]
    fn rows_iterator() {
        let repr = SliceMatRepr::<u8, Meta>::new(4, 2).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();

        // Fill via get_row_mut.
        for i in 0..4 {
            let mut row = mat.get_row_mut(i).unwrap();
            *row.meta_mut() = Meta { scale: i as f32 };
            row.vector_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(j, v)| *v = (i * 10 + j) as u8);
        }

        // Verify via rows iterator.
        let rows: Vec<_> = mat.rows().collect();
        assert_eq!(rows.len(), 4);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(*row.meta(), Meta { scale: i as f32 });
            for j in 0..2 {
                assert_eq!(row.vector()[j], (i * 10 + j) as u8);
            }
        }
    }

    #[test]
    fn rows_mut_iterator() {
        let repr = SliceMatRepr::<u32, Meta>::new(3, 2).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();

        // Fill via rows_mut.
        for (i, mut row) in mat.rows_mut().enumerate() {
            *row.meta_mut() = Meta {
                scale: (i + 1) as f32,
            };
            row.vector_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(j, v)| *v = (i * 10 + j) as u32);
        }

        // Verify.
        for i in 0..3 {
            let row = mat.get_row(i).unwrap();
            assert_eq!(
                *row.meta(),
                Meta {
                    scale: (i + 1) as f32
                }
            );
            for j in 0..2 {
                assert_eq!(row.vector()[j], (i * 10 + j) as u32);
            }
        }
    }

    #[test]
    fn reborrow_views() {
        let repr = SliceMatRepr::<f32, Meta>::new(2, 3).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();

        // Write data.
        {
            let mut row = mat.get_row_mut(0).unwrap();
            *row.meta_mut() = Meta { scale: 1.0 };
            row.vector_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        }

        // Reborrow as MatRef.
        let view: MatRef<'_, _> = mat.reborrow();
        assert_eq!(view.num_vectors(), 2);
        let row = view.get_row(0).unwrap();
        assert_eq!(*row.meta(), Meta { scale: 1.0 });
        assert_eq!(row.vector(), &[1.0, 2.0, 3.0]);

        // Reborrow as MatMut.
        let mut view_mut: MatMut<'_, _> = mat.reborrow_mut();
        assert_eq!(view_mut.num_vectors(), 2);
        let mut row = view_mut.get_row_mut(0).unwrap();
        *row.meta_mut() = Meta { scale: 5.0 };

        // Verify the change is visible.
        let row = mat.get_row(0).unwrap();
        assert_eq!(*row.meta(), Meta { scale: 5.0 });
    }

    #[test]
    fn to_owned_from_view() {
        let repr = SliceMatRepr::<u8, Meta>::new(2, 4).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();

        {
            let mut row = mat.get_row_mut(0).unwrap();
            *row.meta_mut() = Meta { scale: 2.14 };
            row.vector_mut().copy_from_slice(&[1, 2, 3, 4]);
        }

        let owned = mat.as_view().to_owned();
        assert_eq!(owned.num_vectors(), 2);
        let row = owned.get_row(0).unwrap();
        assert_eq!(*row.meta(), Meta { scale: 2.14 });
        assert_eq!(row.vector(), &[1, 2, 3, 4]);
    }

    // #[test]
    // fn new_rejects_overflow() {
    //     // This should fail because the total byte size would overflow.
    //     assert!(SliceMatRepr::<u8, Meta>::new(usize::MAX, 2).is_err());
    //     assert!(SliceMatRepr::<u8, Meta>::new(2, usize::MAX).is_err());
    // }

    /// A metadata type with stricter alignment to test alignment handling.
    #[derive(Debug, Default, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C, align(8))]
    struct AlignedMeta {
        a: f64,
    }

    #[test]
    fn aligned_metadata() {
        let repr = SliceMatRepr::<u16, AlignedMeta>::new(2, 5).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();

        {
            let mut row = mat.get_row_mut(0).unwrap();
            *row.meta_mut() = AlignedMeta { a: 42.0 };
            row.vector_mut().copy_from_slice(&[1, 2, 3, 4, 5]);
        }

        let row = mat.get_row(0).unwrap();
        assert_eq!(*row.meta(), AlignedMeta { a: 42.0 });
        assert_eq!(row.vector(), &[1, 2, 3, 4, 5]);
    }

    /// Zero-sized metadata.
    #[derive(Debug, Default, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
    #[repr(C)]
    struct ZstMeta;

    #[test]
    fn zst_metadata() {
        let repr = SliceMatRepr::<f32, ZstMeta>::new(3, 2).unwrap();
        let mut mat = Mat::new(repr, Defaulted).unwrap();

        for i in 0..3 {
            let mut row = mat.get_row_mut(i).unwrap();
            assert_eq!(*row.meta(), ZstMeta);
            row.vector_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(j, v)| *v = (i * 10 + j) as f32);
        }

        for i in 0..3 {
            let row = mat.get_row(i).unwrap();
            assert_eq!(*row.meta(), ZstMeta);
            for j in 0..2 {
                assert_eq!(row.vector()[j], (i * 10 + j) as f32);
            }
        }
    }
}
