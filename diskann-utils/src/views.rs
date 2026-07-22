/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Row-major matrix types for multi-vector representations.
//!
//! This module provides flexible matrix abstractions that support different underlying
//! storage formats through the [`Repr`] trait. The primary types are:
//!
//! - [`Mat`]: An owning matrix that manages its own memory.
//! - [`MatRef`]: An immutable borrowed view of matrix data.
//! - [`MatMut`]: A mutable borrowed view of matrix data.
//!
//! # Representations
//!
//! Representation types interact with the [`Mat`] family of types using the following traits:
//!
//! - [`Repr`]: Read-only matrix representation.
//! - [`ReprMut`]: Mutable matrix representation.
//! - [`ReprOwned`]: Owning matrix representation.
//!
//! Each trait refinement has a corresponding constructor:
//!
//! - [`NewRef`]: Construct a read-only [`MatRef`] view over a slice.
//! - [`NewMut`]: Construct a mutable [`MatMut`] matrix view over a slice.
//! - [`NewOwned`]: Construct a new owning [`Mat`].
//!

use std::{alloc::Layout, iter::FusedIterator, marker::PhantomData, ptr::NonNull};

use crate::{Reborrow, ReborrowMut};
use thiserror::Error;

#[cfg(feature = "rayon")]
use rayon::iter::ParallelIterator;

// Pointer helpers, kept private so the matrix framework is self-contained in diskann-utils.
fn as_nonnull<T>(slice: &[T]) -> NonNull<T> {
    // SAFETY: Slices guarantee non-null pointers.
    unsafe { NonNull::new_unchecked(slice.as_ptr().cast_mut()) }
}

fn as_nonnull_mut<T>(slice: &mut [T]) -> NonNull<T> {
    // SAFETY: Slices guarantee non-null pointers.
    unsafe { NonNull::new_unchecked(slice.as_mut_ptr()) }
}

fn box_into_nonnull<T>(b: Box<[T]>) -> NonNull<T> {
    // SAFETY: `Box::into_raw` guarantees the returned pointer is non-null.
    unsafe { NonNull::new_unchecked(Box::into_raw(b).cast::<T>()) }
}

/// Representation trait describing the layout and access patterns for a matrix.
///
/// Implementations define how raw bytes are interpreted as typed rows. This enables
/// matrices over different storage formats (dense, quantized, etc.) using a single
/// generic [`Mat`] type.
///
/// # Associated Types
///
/// - `Row<'a>`: The immutable row type (e.g., `&[f32]`, `&[f16]`).
///
/// # Safety
///
/// Implementations must ensure:
///
/// - [`get_row`](Self::get_row) returns valid references for the given row index.
///   This call **must** be memory safe for `i < self.nrows()`, provided the caller upholds
///   the contract for the raw pointer.
///
/// - The objects implicitly managed by this representation inherit the `Send` and `Sync`
///   attributes of `Repr`. That is, `Repr: Send` implies that the objects in backing memory
///   are [`Send`], and likewise with `Sync`. This is necessary to apply [`Send`] and [`Sync`]
///   bounds to [`Mat`], [`MatRef`], and [`MatMut`].
pub unsafe trait Repr: Copy {
    /// Immutable row reference type.
    type Row<'a>
    where
        Self: 'a;

    /// Returns the number of rows in the matrix.
    ///
    /// # Safety Contract
    ///
    /// This function must be loosely pure in the sense that for any given instance of
    /// `self`, `self.nrows()` must return the same value.
    fn nrows(&self) -> usize;

    /// Returns the memory layout for a memory allocation containing [`Repr::nrows`] vectors
    /// each with vector dimension [`Repr::ncols`].
    ///
    /// # Safety Contract
    ///
    /// The [`Layout`] returned from this method must be consistent with the contract of
    /// [`Repr::get_row`].
    fn layout(&self) -> Result<Layout, LayoutError>;

    /// Returns an immutable reference to the `i`-th row.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to a slice with a layout compatible with [`Repr::layout`].
    /// - The entire range for this slice must be within a single allocation.
    /// - `i` must be less than [`Repr::nrows`].
    /// - The memory referenced by the returned [`Repr::Row`] must not be mutated for the
    ///   duration of lifetime `'a`.
    /// - The lifetime for the returned [`Repr::Row`] is inferred from its usage. Correct
    ///   usage must properly tie the lifetime to a source.
    unsafe fn get_row<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::Row<'a>;
}

/// Extension of [`Repr`] that supports mutable row access.
///
/// # Associated Types
///
/// - `RowMut<'a>`: The mutable row type (e.g., `&mut [f32]`).
///
/// # Safety
///
/// Implementors must ensure:
///
/// - [`get_row_mut`](Self::get_row_mut) returns valid references for the given row index.
///   This call **must** be memory safe for `i < self.nrows()`, provided the caller upholds
///   the contract for the raw pointer.
///
///   Additionally, since the implementation of the [`RowsMut`] iterator can give out rows
///   for all `i` in `0..self.nrows()`, the implementation of [`Self::get_row_mut`] must be
///   such that the result for disjoint `i` must not interfere with one another.
pub unsafe trait ReprMut: Repr {
    /// Mutable row reference type.
    type RowMut<'a>
    where
        Self: 'a;

    /// Returns a mutable reference to the i-th row.
    ///
    /// # Safety
    /// - `ptr` must point to a slice with a layout compatible with [`Repr::layout`].
    /// - The entire range for this slice must be within a single allocation.
    /// - `i` must be less than `self.nrows()`.
    /// - The memory referenced by the returned [`ReprMut::RowMut`] must not be accessed
    ///   through any other reference for the duration of lifetime `'a`.
    /// - The lifetime for the returned [`ReprMut::RowMut`] is inferred from its usage.
    ///   Correct usage must properly tie the lifetime to a source.
    unsafe fn get_row_mut<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::RowMut<'a>;
}

/// Extension trait for [`Repr`] that supports deallocation of owned matrices. This is used
/// in conjunction with [`NewOwned`] to create matrices.
///
/// Requires [`ReprMut`] since owned matrices should support mutation.
///
/// # Safety
///
/// Implementors must ensure that `drop` properly deallocates the memory in a way compatible
/// with all [`NewOwned`] implementations.
pub unsafe trait ReprOwned: ReprMut {
    /// Deallocates memory at `ptr` and drops `self`.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been obtained via [`NewOwned`] with the same value of `self`.
    /// - This method may only be called once for such a pointer.
    /// - After calling this method, the memory behind `ptr` may not be dereferenced at all.
    unsafe fn drop(self, ptr: NonNull<u8>);
}

/// A new-type version of `std::alloc::LayoutError` for cleaner error handling.
///
/// This is basically the same as [`std::alloc::LayoutError`], but constructible in
/// use code to allow implementors of [`Repr::layout`] to return it for reasons other than
/// those derived from `std::alloc::Layout`'s methods.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct LayoutError;

impl LayoutError {
    /// Construct a new opaque [`LayoutError`].
    pub fn new() -> Self {
        Self
    }
}

impl Default for LayoutError {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LayoutError")
    }
}

impl std::error::Error for LayoutError {}

impl From<std::alloc::LayoutError> for LayoutError {
    fn from(_: std::alloc::LayoutError) -> Self {
        LayoutError
    }
}

//////////////////
// Constructors //
//////////////////

/// Create a new [`MatRef`] over a slice.
///
/// # Safety
///
/// Implementations must validate the length (and any other requirements) of the provided
/// slice to ensure it is compatible with the implementation of [`Repr`].
pub unsafe trait NewRef<T>: Repr {
    /// Errors that can occur when initializing.
    type Error;

    /// Create a new [`MatRef`] over `slice`.
    fn new_ref(self, slice: &[T]) -> Result<MatRef<'_, Self>, Self::Error>;
}

/// Create a new [`MatMut`] over a slice.
///
/// # Safety
///
/// Implementations must validate the length (and any other requirements) of the provided
/// slice to ensure it is compatible with the implementation of [`ReprMut`].
pub unsafe trait NewMut<T>: ReprMut {
    /// Errors that can occur when initializing.
    type Error;

    /// Create a new [`MatMut`] over `slice`.
    fn new_mut(self, slice: &mut [T]) -> Result<MatMut<'_, Self>, Self::Error>;
}

/// Create a new [`Mat`] from an initializer.
///
/// # Safety
///
/// Implementations must ensure that the returned [`Mat`] is compatible with
/// `Self`'s implementation of [`ReprOwned`].
pub unsafe trait NewOwned<T>: ReprOwned {
    /// Errors that can occur when initializing.
    type Error;

    /// Create a new [`Mat`] initialized with `init`.
    fn new_owned(self, init: T) -> Result<Mat<Self>, Self::Error>;
}

/// Create a new [`Mat`] with every element default-initialized.
///
/// This is a distinct trait from [`NewOwned`] (rather than a `NewOwned<Marker>`) so that a repr
/// can offer both a value-fill and a default-fill constructor without the two impls overlapping
/// under coherence.
///
/// # Safety
///
/// Implementations must ensure that the returned [`Mat`] is compatible with
/// `Self`'s implementation of [`ReprOwned`].
pub unsafe trait NewDefault: ReprOwned {
    /// Errors that can occur when initializing.
    type Error;

    /// Create a new [`Mat`] with each element set to its [`Default`].
    fn new_default(self) -> Result<Mat<Self>, Self::Error>;
}

/// Create a new [`Mat`] cloned from a view.
pub trait NewCloned: ReprOwned {
    /// Clone the contents behind `v`, returning a new owning [`Mat`].
    ///
    /// Implementations should ensure the returned [`Mat`] is "semantically the same" as `v`.
    fn new_cloned(v: MatRef<'_, Self>) -> Mat<Self>;
}

//////////////
// RowMajor //
//////////////

/// Metadata for dense row-major matrices.
///
/// Rows are stored contiguously as `&[T]` slices. This is the default representation
/// type for standard floating-point multi-vectors.
///
/// # Row Types
///
/// - `Row<'a>`: `&'a [T]`
/// - `RowMut<'a>`: `&'a mut [T]`
#[derive(Debug)]
pub struct RowMajor<T> {
    nrows: usize,
    ncols: usize,
    _elem: PhantomData<T>,
}

// Hand-written so `RowMajor<T>` is `Copy`/`Clone`/`PartialEq`/`Eq` for every `T`: it only
// stores two `usize` and a `PhantomData<T>`, so derives would spuriously require the same
// bound on `T` (and the `Repr: Copy` supertrait must hold regardless of the element type).
impl<T> Copy for RowMajor<T> {}

impl<T> Clone for RowMajor<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> PartialEq for RowMajor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.nrows == other.nrows && self.ncols == other.ncols
    }
}

impl<T> Eq for RowMajor<T> {}

impl<T> RowMajor<T> {
    /// Create a new `RowMajor` for data of type `T`.
    ///
    /// Successful construction requires:
    ///
    /// * The total number of elements determined by `nrows * ncols` does not exceed
    ///   `usize::MAX`.
    /// * The total memory footprint defined by `ncols * nrows * size_of::<T>()` does not
    ///   exceed `isize::MAX`.
    pub fn new(nrows: usize, ncols: usize) -> Result<Self, Overflow> {
        Overflow::check::<T>(nrows, ncols)?;
        Ok(Self {
            nrows,
            ncols,
            _elem: PhantomData,
        })
    }

    /// Returns the number of total elements (`rows x cols`) in this matrix.
    pub fn num_elements(&self) -> usize {
        // Since we've constructed `self` - we know we cannot overflow.
        self.nrows() * self.ncols()
    }

    /// Returns `rows`, the number of rows in this matrix.
    fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns `ncols`, the number of elements in a row of this matrix.
    fn ncols(&self) -> usize {
        self.ncols
    }

    /// Checks the following:
    ///
    /// 1. Computation of the number of elements in `self` does not overflow.
    /// 2. Argument `slice` has the expected number of elements.
    fn check_slice(&self, slice: &[T]) -> Result<(), SliceError> {
        let len = self.num_elements();

        if slice.len() != len {
            Err(SliceError::LengthMismatch {
                expected: len,
                found: slice.len(),
            })
        } else {
            Ok(())
        }
    }

    /// Create a new [`Mat`] around the contents of `b` **without** any checks.
    ///
    /// # Safety
    ///
    /// The length of `b` must be exactly [`RowMajor::num_elements`].
    unsafe fn box_to_mat(self, b: Box<[T]>) -> Mat<Self> {
        debug_assert_eq!(b.len(), self.num_elements(), "safety contract violated");

        let ptr = box_into_nonnull(b).cast::<u8>();

        // SAFETY: `ptr` is properly aligned and points to a slice of the required length.
        // Additionally, it is dropped via `Box::from_raw`, which is compatible with obtaining
        // it from `Box::into_raw`.
        unsafe { Mat::from_raw_parts(self, ptr) }
    }
}

/// Error for [`RowMajor::new`].
#[derive(Debug, Clone, Copy)]
pub struct Overflow {
    nrows: usize,
    ncols: usize,
    elsize: usize,
}

impl Overflow {
    /// Construct an `Overflow` error for the given dimensions and element type.
    pub fn for_type<T>(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            elsize: std::mem::size_of::<T>(),
        }
    }

    /// Verify that `capacity` elements of type `T` fit within the `isize::MAX` byte
    /// budget required by Rust's allocation APIs.
    ///
    /// On failure the error reports the original `(nrows, ncols)` dimensions rather
    /// than the padded capacity.
    pub fn check_byte_budget<T>(capacity: usize, nrows: usize, ncols: usize) -> Result<(), Self> {
        let bytes = std::mem::size_of::<T>().saturating_mul(capacity);
        if bytes <= isize::MAX as usize {
            Ok(())
        } else {
            Err(Self::for_type::<T>(nrows, ncols))
        }
    }

    pub(crate) fn check<T>(nrows: usize, ncols: usize) -> Result<(), Self> {
        // Guard the element count itself so that `num_elements()` can never overflow.
        let capacity = nrows
            .checked_mul(ncols)
            .ok_or_else(|| Self::for_type::<T>(nrows, ncols))?;

        Self::check_byte_budget::<T>(capacity, nrows, ncols)
    }
}

impl std::fmt::Display for Overflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.elsize == 0 {
            write!(
                f,
                "ZST matrix with dimensions {} x {} has more than `usize::MAX` elements",
                self.nrows, self.ncols,
            )
        } else {
            write!(
                f,
                "a matrix of size {} x {} with element size {} would exceed isize::MAX bytes",
                self.nrows, self.ncols, self.elsize,
            )
        }
    }
}

impl std::error::Error for Overflow {}

/// Error types for [`RowMajor`].
#[derive(Debug, Clone, Copy, Error)]
#[non_exhaustive]
pub enum SliceError {
    #[error("Length mismatch: expected {expected}, found {found}")]
    LengthMismatch { expected: usize, found: usize },
}

// SAFETY: The implementation correctly computes row offsets as `i * ncols` and
// constructs valid slices of the appropriate length. The `layout` method correctly
// reports the memory layout requirements.
unsafe impl<T> Repr for RowMajor<T> {
    type Row<'a>
        = &'a [T]
    where
        T: 'a;

    fn nrows(&self) -> usize {
        self.nrows
    }

    fn layout(&self) -> Result<Layout, LayoutError> {
        Ok(Layout::array::<T>(self.num_elements())?)
    }

    unsafe fn get_row<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::Row<'a> {
        debug_assert!(ptr.cast::<T>().is_aligned());
        debug_assert!(i < self.nrows);

        // SAFETY: The caller asserts that `i` is less than `self.nrows()`. Since this type
        // audits the constructors for `Mat` and friends, we know that there is room for at
        // least `self.num_elements()` elements from the base pointer, so this access is safe.
        let row_ptr = unsafe { ptr.as_ptr().cast::<T>().add(i * self.ncols) };

        // SAFETY: The logic is the same as the previous `unsafe` block.
        unsafe { std::slice::from_raw_parts(row_ptr, self.ncols) }
    }
}

// SAFETY: The implementation correctly computes row offsets and constructs valid mutable
// slices.
unsafe impl<T> ReprMut for RowMajor<T> {
    type RowMut<'a>
        = &'a mut [T]
    where
        T: 'a;

    unsafe fn get_row_mut<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::RowMut<'a> {
        debug_assert!(ptr.cast::<T>().is_aligned());
        debug_assert!(i < self.nrows);

        // SAFETY: The caller asserts that `i` is less than `self.nrows()`. Since this type
        // audits the constructors for `Mat` and friends, we know that there is room for at
        // least `self.num_elements()` elements from the base pointer, so this access is safe.
        let row_ptr = unsafe { ptr.as_ptr().cast::<T>().add(i * self.ncols) };

        // SAFETY: The logic is the same as the previous `unsafe` block. Further, the caller
        // attests that creating a mutable reference is safe.
        unsafe { std::slice::from_raw_parts_mut(row_ptr, self.ncols) }
    }
}

// SAFETY: The drop implementation correctly reconstructs a Box from the raw pointer
// using the same length (nrows * ncols) that was used for allocation, allowing Box
// to properly deallocate the memory.
unsafe impl<T> ReprOwned for RowMajor<T> {
    unsafe fn drop(self, ptr: NonNull<u8>) {
        // SAFETY: The caller guarantees that `ptr` was obtained from an implementation of
        // `NewOwned` for an equivalent instance of `self`.
        //
        // We ensure that `NewOwned` goes through boxes, so here we reconstruct a Box to
        // let it handle deallocation.
        unsafe {
            let slice_ptr = std::ptr::slice_from_raw_parts_mut(
                ptr.cast::<T>().as_ptr(),
                self.nrows * self.ncols,
            );
            let _ = Box::from_raw(slice_ptr);
        }
    }
}

// SAFETY: The implementation uses guarantees from `Box` to ensure that the pointer
// initialized by it is non-null and properly aligned to the underlying type.
unsafe impl<T> NewOwned<T> for RowMajor<T>
where
    T: Clone,
{
    type Error = std::convert::Infallible;
    fn new_owned(self, value: T) -> Result<Mat<Self>, Self::Error> {
        let b: Box<[T]> = std::iter::repeat_n(value, self.num_elements()).collect();

        // SAFETY: By construction, `b` has length `self.num_elements()`.
        Ok(unsafe { self.box_to_mat(b) })
    }
}

// SAFETY: The implementation uses guarantees from `Box` to ensure that the pointer
// initialized by it is non-null and properly aligned to the underlying type.
unsafe impl<T> NewDefault for RowMajor<T>
where
    T: Default,
{
    type Error = std::convert::Infallible;
    fn new_default(self) -> Result<Mat<Self>, Self::Error> {
        let b: Box<[T]> = std::iter::repeat_with(T::default)
            .take(self.num_elements())
            .collect();

        // SAFETY: By construction, `b` has length `self.num_elements()`.
        Ok(unsafe { self.box_to_mat(b) })
    }
}

// SAFETY: This checks that the slice has the correct length, which is all that is
// required for [`Repr`].
unsafe impl<T> NewRef<T> for RowMajor<T> {
    type Error = SliceError;
    fn new_ref(self, data: &[T]) -> Result<MatRef<'_, Self>, Self::Error> {
        self.check_slice(data)?;

        // SAFETY: The function `check_slice` verifies that `data` is compatible with
        // the layout requirement of `RowMajor`.
        //
        // We've properly checked that the underlying pointer is okay.
        Ok(unsafe { MatRef::from_raw_parts(self, as_nonnull(data).cast::<u8>()) })
    }
}

// SAFETY: This checks that the slice has the correct length, which is all that is
// required for [`ReprMut`].
unsafe impl<T> NewMut<T> for RowMajor<T> {
    type Error = SliceError;
    fn new_mut(self, data: &mut [T]) -> Result<MatMut<'_, Self>, Self::Error> {
        self.check_slice(data)?;

        // SAFETY: The function `check_slice` verifies that `data` is compatible with
        // the layout requirement of `RowMajor`.
        //
        // We've properly checked that the underlying pointer is okay.
        Ok(unsafe { MatMut::from_raw_parts(self, as_nonnull_mut(data).cast::<u8>()) })
    }
}

impl<T> NewCloned for RowMajor<T>
where
    T: Clone,
{
    fn new_cloned(v: MatRef<'_, Self>) -> Mat<Self> {
        let b: Box<[T]> = v.as_slice().iter().cloned().collect();

        // SAFETY: By construction, `b` has length `v.repr().num_elements()`.
        unsafe { v.repr().box_to_mat(b) }
    }
}

/////////
// Mat //
/////////

/// An owning matrix that manages its own memory.
///
/// The matrix stores raw bytes interpreted according to representation type `T`.
/// Memory is automatically deallocated when the matrix is dropped.
#[derive(Debug)]
pub struct Mat<T: ReprOwned> {
    ptr: NonNull<u8>,
    repr: T,
    _invariant: PhantomData<fn(T) -> T>,
}

// SAFETY: [`Repr`] is required to propagate its `Send` bound.
unsafe impl<T> Send for Mat<T> where T: ReprOwned + Send {}

// SAFETY: [`Repr`] is required to propagate its `Sync` bound.
unsafe impl<T> Sync for Mat<T> where T: ReprOwned + Sync {}

impl<T: ReprOwned> Mat<T> {
    /// Create a new matrix using `init` as the initializer.
    pub fn from_repr<U>(repr: T, init: U) -> Result<Self, <T as NewOwned<U>>::Error>
    where
        T: NewOwned<U>,
    {
        repr.new_owned(init)
    }

    /// Create a new matrix with every element default-initialized.
    ///
    /// ```rust
    /// use diskann_utils::views::{Mat, RowMajor};
    /// let mat = Mat::from_default(RowMajor::<f32>::new(4, 3).unwrap()).unwrap();
    /// for i in 0..4 {
    ///     assert!(mat.get_row(i).unwrap().iter().all(|&x| x == 0.0f32));
    /// }
    /// ```
    pub fn from_default(repr: T) -> Result<Self, <T as NewDefault>::Error>
    where
        T: NewDefault,
    {
        repr.new_default()
    }

    /// Returns the number of rows (vectors) in the matrix.
    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.repr.nrows()
    }

    /// Returns a reference to the underlying representation.
    pub fn repr(&self) -> &T {
        &self.repr
    }

    /// Returns the `i`th row if `i < self.num_vectors()`.
    #[must_use]
    pub fn get_row(&self, i: usize) -> Option<T::Row<'_>> {
        if i < self.num_vectors() {
            // SAFETY: Bounds check passed, and the Mat was constructed
            // with valid representation and pointer.
            let row = unsafe { self.get_row_unchecked(i) };
            Some(row)
        } else {
            None
        }
    }

    /// Returns the `i`th row without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be less than `self.num_vectors()`.
    pub unsafe fn get_row_unchecked(&self, i: usize) -> T::Row<'_> {
        // SAFETY: Caller must ensure i < self.num_vectors(). The constructors for this type
        // ensure that `ptr` is compatible with `T`.
        unsafe { self.repr.get_row(self.ptr, i) }
    }

    /// Returns the `i`th mutable row if `i < self.num_vectors()`.
    #[must_use]
    pub fn get_row_mut(&mut self, i: usize) -> Option<T::RowMut<'_>> {
        if i < self.num_vectors() {
            // SAFETY: Bounds check passed, and we have exclusive access via &mut self.
            Some(unsafe { self.get_row_mut_unchecked(i) })
        } else {
            None
        }
    }

    pub(crate) unsafe fn get_row_mut_unchecked(&mut self, i: usize) -> T::RowMut<'_> {
        // SAFETY: Caller asserts that `i < self.num_vectors()`. The constructors for this
        // type ensure that `ptr` is compatible with `T`.
        unsafe { self.repr.get_row_mut(self.ptr, i) }
    }

    /// Returns an immutable view of the matrix.
    #[inline]
    pub fn as_view(&self) -> MatRef<'_, T> {
        MatRef {
            ptr: self.ptr,
            repr: self.repr,
            _lifetime: PhantomData,
        }
    }

    /// Returns a mutable view of the matrix.
    #[inline]
    pub fn as_view_mut(&mut self) -> MatMut<'_, T> {
        MatMut {
            ptr: self.ptr,
            repr: self.repr,
            _lifetime: PhantomData,
        }
    }

    /// Returns an iterator over immutable row references.
    pub fn rows(&self) -> Rows<'_, T> {
        Rows::new(self.reborrow())
    }

    /// Returns an iterator over mutable row references.
    pub fn rows_mut(&mut self) -> RowsMut<'_, T> {
        RowsMut::new(self.reborrow_mut())
    }

    /// Construct a new [`Mat`] over the raw pointer and representation without performing
    /// any validity checks.
    ///
    /// # Safety
    ///
    /// Argument `ptr` must be:
    ///
    /// 1. Point to memory compatible with [`Repr::layout`].
    /// 2. Be compatible with the drop logic in [`ReprOwned`].
    pub unsafe fn from_raw_parts(repr: T, ptr: NonNull<u8>) -> Self {
        Self {
            ptr,
            repr,
            _invariant: PhantomData,
        }
    }

    /// Return the base pointer for the [`Mat`].
    pub fn as_raw_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Return a mutable base pointer for the [`Mat`].
    pub fn as_raw_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl<T: ReprOwned> Drop for Mat<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr` was correctly initialized according to `layout`
        // and we are guaranteed exclusive access to the data due to Rust borrow rules.
        unsafe { self.repr.drop(self.ptr) };
    }
}

impl<T: NewCloned> Clone for Mat<T> {
    fn clone(&self) -> Self {
        T::new_cloned(self.as_view())
    }
}

// Delegation macro for the dense (`RowMajor`) read API. The canonical read
// implementations live once on `MatRef<'_, RowMajor<T>>` (`MatrixView`); the
// owning `Mat` and the mutable view forward to them through `self.as_view()`, so
// each method has a single body. Mirrors the `delegate_to_ref!` pattern used by
// `diskann-quantization`'s block-transposed layout.
macro_rules! delegate_read {
    ($(#[$m:meta])* $vis:vis fn $name:ident(&self $(, $a:ident: $t:ty)*) $(-> $r:ty)?) => {
        #[doc = "Delegates to the canonical immutable-view implementation."]
        $(#[$m])*
        #[inline]
        $vis fn $name(&self $(, $a: $t)*) $(-> $r)? {
            self.as_view().$name($($a),*)
        }
    };
    ($(#[$m:meta])* unsafe $vis:vis fn $name:ident(&self $(, $a:ident: $t:ty)*) $(-> $r:ty)?) => {
        #[doc = "Delegates to the canonical immutable-view implementation."]
        $(#[$m])*
        #[inline]
        $vis unsafe fn $name(&self $(, $a: $t)*) $(-> $r)? {
            // SAFETY: the caller upholds the delegated method's safety contract.
            unsafe { self.as_view().$name($($a),*) }
        }
    };
}

impl<T> Mat<RowMajor<T>> {
    /// Construct a [`Mat`] by filling each element in row-major order from `gen`.
    pub fn new<U: Generator<T>>(mut gen: U, nrows: usize, ncols: usize) -> Self {
        let repr = RowMajor::new(nrows, ncols).expect("dimension overflow");
        let b: Box<[T]> = (0..repr.num_elements()).map(|_| gen.generate()).collect();
        // SAFETY: `b` has length `repr.num_elements()` by construction.
        unsafe { repr.box_to_mat(b) }
    }

    /// Returns the raw dimension (columns) of the vectors in the matrix.
    #[inline]
    pub fn vector_dim(&self) -> usize {
        self.repr.ncols()
    }

    delegate_read!(pub fn as_slice(&self) -> &[T]);
    delegate_read!(pub fn as_matrix_view(&self) -> MatrixView<'_, T>);
}

////////////
// MatRef //
////////////

/// An immutable borrowed view of a matrix.
///
/// Provides read-only access to matrix data without ownership. Implements [`Copy`]
/// and can be freely cloned.
///
/// # Type Parameter
/// - `T`: A [`Repr`] implementation defining the row layout.
///
/// # Access
/// - [`get_row`](Self::get_row): Get an immutable row by index.
/// - [`rows`](Self::rows): Iterate over all rows.
#[derive(Debug, Clone, Copy)]
pub struct MatRef<'a, T: Repr> {
    ptr: NonNull<u8>,
    repr: T,
    /// Marker to tie the lifetime to the borrowed data.
    _lifetime: PhantomData<&'a T>,
}

// SAFETY: [`Repr`] is required to propagate its `Send` bound.
unsafe impl<T> Send for MatRef<'_, T> where T: Repr + Send {}

// SAFETY: [`Repr`] is required to propagate its `Sync` bound.
unsafe impl<T> Sync for MatRef<'_, T> where T: Repr + Sync {}

impl<'a, T: Repr> MatRef<'a, T> {
    /// Construct a new [`MatRef`] over `data`.
    pub fn from_repr<U>(repr: T, data: &'a [U]) -> Result<Self, T::Error>
    where
        T: NewRef<U>,
    {
        repr.new_ref(data)
    }

    /// Returns the number of rows (vectors) in the matrix.
    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.repr.nrows()
    }

    /// Returns a reference to the underlying representation.
    pub fn repr(&self) -> &T {
        &self.repr
    }

    /// Returns an immutable reference to the i-th row, or `None` if out of bounds.
    #[must_use]
    pub fn get_row(&self, i: usize) -> Option<T::Row<'_>> {
        if i < self.num_vectors() {
            // SAFETY: Bounds check passed, and the MatRef was constructed
            // with valid representation and pointer.
            let row = unsafe { self.get_row_unchecked(i) };
            Some(row)
        } else {
            None
        }
    }

    /// Returns the `i`th row without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be less than `self.num_vectors()`.
    #[inline]
    pub unsafe fn get_row_unchecked(&self, i: usize) -> T::Row<'_> {
        // SAFETY: Caller must ensure i < self.num_vectors().
        unsafe { self.repr.get_row(self.ptr, i) }
    }

    /// Returns an iterator over immutable row references.
    pub fn rows(&self) -> Rows<'_, T> {
        Rows::new(*self)
    }

    /// Return a [`Mat`] with the same contents as `self`.
    pub fn to_owned(&self) -> Mat<T>
    where
        T: NewCloned,
    {
        T::new_cloned(*self)
    }

    /// Construct a new [`MatRef`] over the raw pointer and representation without performing
    /// any validity checks.
    ///
    /// # Safety
    ///
    /// Argument `ptr` must point to memory compatible with [`Repr::layout`] and pass any
    /// validity checks required by `T`.
    pub unsafe fn from_raw_parts(repr: T, ptr: NonNull<u8>) -> Self {
        Self {
            ptr,
            repr,
            _lifetime: PhantomData,
        }
    }

    /// Return the base pointer for the [`MatRef`].
    pub fn as_raw_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
}

impl<'a, T> MatRef<'a, RowMajor<T>> {
    /// Returns the raw dimension (columns) of the vectors in the matrix.
    #[inline]
    pub fn vector_dim(&self) -> usize {
        self.repr.ncols()
    }

    /// Return the backing data as a contiguous slice of `T`.
    ///
    /// The returned slice has `num_vectors() * vector_dim()` elements in row-major order.
    #[inline]
    pub fn as_slice(&self) -> &'a [T] {
        let len = self.repr.num_elements();
        // SAFETY: `RowMajor<T>` guarantees `nrows * ncols` contiguous `T` elements starting
        // at `self.ptr`, valid for the view's lifetime `'a`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().cast::<T>(), len) }
    }

    /// Return a [`MatrixView`] over the backing data.
    #[inline]
    pub fn as_matrix_view(&self) -> MatrixView<'a, T> {
        *self
    }
}

// Reborrow: Mat -> MatRef
impl<'this, T: ReprOwned> Reborrow<'this> for Mat<T> {
    type Target = MatRef<'this, T>;

    fn reborrow(&'this self) -> Self::Target {
        self.as_view()
    }
}

// ReborrowMut: Mat -> MatMut
impl<'this, T: ReprOwned> ReborrowMut<'this> for Mat<T> {
    type Target = MatMut<'this, T>;

    fn reborrow_mut(&'this mut self) -> Self::Target {
        self.as_view_mut()
    }
}

// Reborrow: MatRef -> MatRef (with shorter lifetime)
impl<'this, 'a, T: Repr> Reborrow<'this> for MatRef<'a, T> {
    type Target = MatRef<'this, T>;

    fn reborrow(&'this self) -> Self::Target {
        MatRef {
            ptr: self.ptr,
            repr: self.repr,
            _lifetime: PhantomData,
        }
    }
}

////////////
// MatMut //
////////////

/// A mutable borrowed view of a matrix.
///
/// Provides read-write access to matrix data without ownership.
///
/// # Type Parameter
/// - `T`: A [`ReprMut`] implementation defining the row layout.
///
/// # Access
/// - [`get_row`](Self::get_row): Get an immutable row by index.
/// - [`get_row_mut`](Self::get_row_mut): Get a mutable row by index.
/// - [`as_view`](Self::as_view): Reborrow as immutable [`MatRef`].
/// - [`rows`](Self::rows), [`rows_mut`](Self::rows_mut): Iterate over rows.
#[derive(Debug)]
pub struct MatMut<'a, T: ReprMut> {
    ptr: NonNull<u8>,
    repr: T,
    /// Marker to tie the lifetime to the mutably borrowed data.
    _lifetime: PhantomData<&'a mut T>,
}

// SAFETY: [`ReprMut`] is required to propagate its `Send` bound.
unsafe impl<T> Send for MatMut<'_, T> where T: ReprMut + Send {}

// SAFETY: [`ReprMut`] is required to propagate its `Sync` bound.
unsafe impl<T> Sync for MatMut<'_, T> where T: ReprMut + Sync {}

impl<'a, T: ReprMut> MatMut<'a, T> {
    /// Construct a new [`MatMut`] over `data`.
    pub fn from_repr<U>(repr: T, data: &'a mut [U]) -> Result<Self, T::Error>
    where
        T: NewMut<U>,
    {
        repr.new_mut(data)
    }

    /// Returns the number of rows (vectors) in the matrix.
    #[inline]
    pub fn num_vectors(&self) -> usize {
        self.repr.nrows()
    }

    /// Returns a reference to the underlying representation.
    pub fn repr(&self) -> &T {
        &self.repr
    }

    /// Returns an immutable reference to the i-th row, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn get_row(&self, i: usize) -> Option<T::Row<'_>> {
        if i < self.num_vectors() {
            // SAFETY: Bounds check passed.
            Some(unsafe { self.get_row_unchecked(i) })
        } else {
            None
        }
    }

    /// Returns the i-th row without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be less than `self.num_vectors()`.
    #[inline]
    pub unsafe fn get_row_unchecked(&self, i: usize) -> T::Row<'_> {
        // SAFETY: Caller must ensure i < self.num_vectors().
        unsafe { self.repr.get_row(self.ptr, i) }
    }

    /// Returns a mutable reference to the `i`-th row, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn get_row_mut(&mut self, i: usize) -> Option<T::RowMut<'_>> {
        if i < self.num_vectors() {
            // SAFETY: Bounds check passed.
            Some(unsafe { self.get_row_mut_unchecked(i) })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the i-th row without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be less than [`num_vectors()`](Self::num_vectors).
    #[inline]
    pub(crate) unsafe fn get_row_mut_unchecked(&mut self, i: usize) -> T::RowMut<'_> {
        // SAFETY: Caller asserts that `i < self.num_vectors()`. The constructors for this
        // type ensure that `ptr` is compatible with `T`.
        unsafe { self.repr.get_row_mut(self.ptr, i) }
    }

    /// Reborrows as an immutable [`MatRef`].
    pub fn as_view(&self) -> MatRef<'_, T> {
        MatRef {
            ptr: self.ptr,
            repr: self.repr,
            _lifetime: PhantomData,
        }
    }

    /// Returns an iterator over immutable row references.
    pub fn rows(&self) -> Rows<'_, T> {
        Rows::new(self.reborrow())
    }

    /// Returns an iterator over mutable row references.
    pub fn rows_mut(&mut self) -> RowsMut<'_, T> {
        RowsMut::new(self.reborrow_mut())
    }

    /// Return a [`Mat`] with the same contents as `self`.
    pub fn to_owned(&self) -> Mat<T>
    where
        T: NewCloned,
    {
        T::new_cloned(self.as_view())
    }

    /// Construct a new [`MatMut`] over the raw pointer and representation without performing
    /// any validity checks.
    ///
    /// # Safety
    ///
    /// Argument `ptr` must point to memory compatible with [`Repr::layout`].
    pub unsafe fn from_raw_parts(repr: T, ptr: NonNull<u8>) -> Self {
        Self {
            ptr,
            repr,
            _lifetime: PhantomData,
        }
    }

    /// Return the base pointer for the [`MatMut`].
    pub fn as_raw_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Return a mutable base pointer for the [`MatMut`].
    pub fn as_raw_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

// Reborrow: MatMut -> MatRef
impl<'this, 'a, T: ReprMut> Reborrow<'this> for MatMut<'a, T> {
    type Target = MatRef<'this, T>;

    fn reborrow(&'this self) -> Self::Target {
        self.as_view()
    }
}

// ReborrowMut: MatMut -> MatMut (with shorter lifetime)
impl<'this, 'a, T: ReprMut> ReborrowMut<'this> for MatMut<'a, T> {
    type Target = MatMut<'this, T>;

    fn reborrow_mut(&'this mut self) -> Self::Target {
        MatMut {
            ptr: self.ptr,
            repr: self.repr,
            _lifetime: PhantomData,
        }
    }
}

impl<'a, T> MatMut<'a, RowMajor<T>> {
    /// Returns the raw dimension (columns) of the vectors in the matrix.
    #[inline]
    pub fn vector_dim(&self) -> usize {
        self.repr.ncols()
    }

    delegate_read!(pub fn as_slice(&self) -> &[T]);
    delegate_read!(pub fn as_matrix_view(&self) -> MatrixView<'_, T>);
}

//////////////////////////////
// Dense (RowMajor) API + aliases
//////////////////////////////

/// A generator for initializing matrix entries via [`Matrix::new`].
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

/// A [`Generator`] that invokes a closure to initialize each element.
pub struct Init<F>(pub F);

impl<T, F> Generator<T> for Init<F>
where
    F: FnMut() -> T,
{
    fn generate(&mut self) -> T {
        (self.0)()
    }
}

/// Owned dense row-major matrix.
pub type Matrix<T> = Mat<RowMajor<T>>;

/// Shared dense row-major view.
pub type MatrixView<'a, T> = MatRef<'a, RowMajor<T>>;

/// Mutable dense row-major view.
pub type MatrixViewMut<'a, T> = MatMut<'a, RowMajor<T>>;

/// A lightweight, `'static` version of [`TryFromError`] that drops the offending data.
#[derive(Debug, Error)]
#[non_exhaustive]
#[error("cannot view a slice of length {len} as a {nrows}x{ncols} matrix")]
pub struct TryFromErrorLight {
    pub len: usize,
    pub nrows: usize,
    pub ncols: usize,
}

/// Error returned when a slice length is incompatible with the requested dimensions.
///
/// Carries the original `data` so a failed construction does not deallocate it; recover it
/// with [`TryFromError::into_inner`].
#[derive(Error)]
#[non_exhaustive]
#[error("cannot view a slice of length {} as a {nrows}x{ncols} matrix", data.as_slice().len())]
pub struct TryFromError<T: DenseData> {
    data: T,
    nrows: usize,
    ncols: usize,
}

// Hand-written so `TryFromError<T>` does not require `T: Debug`.
impl<T: DenseData> std::fmt::Debug for TryFromError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TryFromError")
            .field("data_len", &self.data.as_slice().len())
            .field("nrows", &self.nrows)
            .field("ncols", &self.ncols)
            .finish()
    }
}

impl<T: DenseData> TryFromError<T> {
    /// Consume the error and return the original data, without deallocating it.
    pub fn into_inner(self) -> T {
        self.data
    }

    /// Return a `'static` copy of this error that drops the offending data.
    pub fn as_static(&self) -> TryFromErrorLight {
        TryFromErrorLight {
            len: self.data.as_slice().len(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<'a, T> MatRef<'a, RowMajor<T>> {
    /// View `data` as an `nrows x ncols` matrix.
    pub fn try_from(
        data: &'a [T],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, TryFromError<&'a [T]>> {
        let repr = match RowMajor::new(nrows, ncols) {
            Ok(repr) if repr.num_elements() == data.len() => repr,
            _ => return Err(TryFromError { data, nrows, ncols }),
        };
        // SAFETY: `data.len()` was checked to equal `repr.num_elements()`.
        Ok(unsafe { MatRef::from_raw_parts(repr, as_nonnull(data).cast::<u8>()) })
    }

    /// View `data` as a single-row matrix.
    pub fn row_vector(data: &'a [T]) -> Self {
        let repr = RowMajor {
            nrows: 1,
            ncols: data.len(),
            _elem: PhantomData,
        };
        // SAFETY: `data` is exactly `1 * data.len()` contiguous `T`, matching `repr`.
        unsafe { MatRef::from_raw_parts(repr, as_nonnull(data).cast::<u8>()) }
    }

    /// View `data` as a single-column matrix.
    pub fn column_vector(data: &'a [T]) -> Self {
        let repr = RowMajor {
            nrows: data.len(),
            ncols: 1,
            _elem: PhantomData,
        };
        // SAFETY: `data` is exactly `data.len() * 1` contiguous `T`, matching `repr`.
        unsafe { MatRef::from_raw_parts(repr, as_nonnull(data).cast::<u8>()) }
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.num_vectors()
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.vector_dim()
    }

    /// Row `i`. Panics if `i >= self.nrows()`.
    pub fn row(&self, i: usize) -> &'a [T] {
        assert!(
            i < self.nrows(),
            "tried to access row {i} of a matrix with {} rows",
            self.nrows()
        );
        let ncols = self.ncols();
        let start = i * ncols;
        // SAFETY: `i < self.nrows()` was asserted, and the backing slice holds
        // `nrows * ncols` elements, so `start..start + ncols` is in bounds.
        unsafe { self.as_slice().get_unchecked(start..start + ncols) }
    }

    /// Iterator over rows.
    pub fn row_iter(&self) -> impl ExactSizeIterator<Item = &'a [T]> {
        self.as_slice().chunks_exact(self.ncols())
    }

    /// Iterator over sub-matrices of up to `batchsize` rows.
    ///
    /// # Panics
    /// Panics if `batchsize == 0`.
    pub fn window_iter(&self, batchsize: usize) -> impl Iterator<Item = MatrixView<'a, T>> {
        assert!(batchsize != 0, "window_iter batchsize cannot be zero");
        let ncols = self.ncols();
        self.as_slice()
            .chunks(ncols * batchsize)
            .map(move |d| MatrixView::try_from(d, d.len() / ncols, ncols).expect("exact chunk"))
    }

    /// Parallel iterator over rows.
    #[cfg(feature = "rayon")]
    pub fn par_row_iter(&self) -> impl rayon::iter::IndexedParallelIterator<Item = &'a [T]>
    where
        T: Sync,
    {
        use rayon::slice::ParallelSlice;
        self.as_slice().par_chunks_exact(self.ncols())
    }

    /// Parallel iterator over sub-matrices of up to `batchsize` rows.
    ///
    /// # Panics
    /// Panics if `batchsize == 0`.
    #[cfg(feature = "rayon")]
    pub fn par_window_iter(
        &self,
        batchsize: usize,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = MatrixView<'a, T>>
    where
        T: Send + Sync,
    {
        use rayon::slice::ParallelSlice;
        assert!(batchsize != 0, "par_window_iter batchsize cannot be zero");
        let ncols = self.ncols();
        self.as_slice()
            .par_chunks(ncols * batchsize)
            .map(move |d| MatrixView::try_from(d, d.len() / ncols, ncols).expect("exact chunk"))
    }

    /// View of the rows in `rows`, or `None` if out of bounds.
    pub fn subview(&self, rows: std::ops::Range<usize>) -> Option<MatrixView<'a, T>> {
        let ncols = self.ncols();
        let lo = rows.start.checked_mul(ncols)?;
        let hi = rows.end.checked_mul(ncols)?;
        let data = self.as_slice().get(lo..hi)?;
        let repr = RowMajor {
            nrows: rows.len(),
            ncols,
            _elem: PhantomData,
        };
        // SAFETY: `data` is exactly `rows.len() * ncols` contiguous `T`, matching `repr`.
        Some(unsafe { MatRef::from_raw_parts(repr, as_nonnull(data).cast::<u8>()) })
    }

    /// Element at `(row, col)`, or `None` if out of bounds.
    pub fn try_get(&self, row: usize, col: usize) -> Option<&'a T> {
        if row >= self.nrows() || col >= self.ncols() {
            None
        } else {
            // SAFETY: `row` and `col` were just verified in-bounds.
            Some(unsafe { self.get_unchecked(row, col) })
        }
    }

    /// Base pointer of the backing data.
    pub fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    /// Apply `f` to every element, producing a new owned matrix of the same shape.
    pub fn map<F, R>(&self, f: F) -> Matrix<R>
    where
        F: FnMut(&T) -> R,
    {
        let data: Box<[R]> = self.as_slice().iter().map(f).collect();
        let repr = RowMajor {
            nrows: self.nrows(),
            ncols: self.ncols(),
            _elem: PhantomData,
        };
        // SAFETY: `data` has exactly `nrows * ncols` elements (one per source element).
        unsafe { repr.box_to_mat(data) }
    }
}

impl<'a, T> MatMut<'a, RowMajor<T>> {
    /// View `data` as an `nrows x ncols` mutable matrix.
    pub fn try_from(
        data: &'a mut [T],
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, TryFromError<&'a mut [T]>> {
        let repr = match RowMajor::new(nrows, ncols) {
            Ok(repr) if repr.num_elements() == data.len() => repr,
            _ => return Err(TryFromError { data, nrows, ncols }),
        };
        // SAFETY: `data.len()` was checked to equal `repr.num_elements()`.
        Ok(unsafe { MatMut::from_raw_parts(repr, as_nonnull_mut(data).cast::<u8>()) })
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.num_vectors()
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.vector_dim()
    }

    // Reads delegate to the canonical `MatrixView` implementation.
    delegate_read!(pub fn row(&self, i: usize) -> &[T]);
    delegate_read!(pub fn row_iter(&self) -> impl ExactSizeIterator<Item = &[T]>);
    delegate_read!(pub fn window_iter(&self, batchsize: usize) -> impl Iterator<Item = MatrixView<'_, T>>);
    delegate_read!(pub fn subview(&self, rows: std::ops::Range<usize>) -> Option<MatrixView<'_, T>>);
    delegate_read!(pub fn try_get(&self, row: usize, col: usize) -> Option<&T>);
    delegate_read!(pub fn as_ptr(&self) -> *const T);

    /// Parallel iterator over rows.
    #[cfg(feature = "rayon")]
    pub fn par_row_iter(&self) -> impl rayon::iter::IndexedParallelIterator<Item = &[T]>
    where
        T: Sync,
    {
        self.as_view().par_row_iter()
    }

    /// Parallel iterator over sub-matrices of up to `batchsize` rows.
    #[cfg(feature = "rayon")]
    pub fn par_window_iter(
        &self,
        batchsize: usize,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = MatrixView<'_, T>>
    where
        T: Send + Sync,
    {
        self.as_view().par_window_iter(batchsize)
    }

    /// Apply `f` to every element, producing a new owned matrix of the same shape.
    pub fn map<F, R>(&self, f: F) -> Matrix<R>
    where
        F: FnMut(&T) -> R,
    {
        self.as_view().map(f)
    }

    /// Backing data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.repr.num_elements();
        // SAFETY: the backing allocation holds `len` contiguous `T`, and `&mut self` grants
        // exclusive access for the returned slice's lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.as_raw_mut_ptr().cast::<T>(), len) }
    }

    /// Mutable row `i`. Panics if `i >= self.nrows()`.
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        assert!(
            i < self.nrows(),
            "tried to access row {i} of a matrix with {} rows",
            self.nrows()
        );
        let ncols = self.ncols();
        let start = i * ncols;
        // SAFETY: `i < self.nrows()` was asserted, and the backing slice holds
        // `nrows * ncols` elements, so `start..start + ncols` is in bounds.
        unsafe { self.as_mut_slice().get_unchecked_mut(start..start + ncols) }
    }

    /// Iterator over mutable rows.
    pub fn row_iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut [T]> {
        let ncols = self.ncols();
        self.as_mut_slice().chunks_exact_mut(ncols)
    }

    /// Parallel iterator over mutable rows.
    #[cfg(feature = "rayon")]
    pub fn par_row_iter_mut(&mut self) -> impl rayon::iter::IndexedParallelIterator<Item = &mut [T]>
    where
        T: Send,
    {
        use rayon::slice::ParallelSliceMut;
        let ncols = self.ncols();
        self.as_mut_slice().par_chunks_exact_mut(ncols)
    }

    /// Base pointer of the backing data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_slice().as_mut_ptr()
    }
}

impl<T> Mat<RowMajor<T>> {
    /// Take ownership of `data`, interpreting it as an `nrows x ncols` matrix.
    pub fn try_from(
        data: Box<[T]>,
        nrows: usize,
        ncols: usize,
    ) -> Result<Self, TryFromError<Box<[T]>>> {
        let repr = match RowMajor::new(nrows, ncols) {
            Ok(repr) if repr.num_elements() == data.len() => repr,
            _ => return Err(TryFromError { data, nrows, ncols }),
        };
        // SAFETY: `data.len()` was checked to equal `repr.num_elements()`.
        Ok(unsafe { repr.box_to_mat(data) })
    }

    /// Take ownership of `data` as a single-row matrix.
    pub fn row_vector(data: Box<[T]>) -> Self {
        let repr = RowMajor {
            nrows: 1,
            ncols: data.len(),
            _elem: PhantomData,
        };
        // SAFETY: `data` has exactly `1 * data.len()` elements matching `repr`.
        unsafe { repr.box_to_mat(data) }
    }

    /// Take ownership of `data` as a single-column matrix.
    pub fn column_vector(data: Box<[T]>) -> Self {
        let repr = RowMajor {
            nrows: data.len(),
            ncols: 1,
            _elem: PhantomData,
        };
        // SAFETY: `data` has exactly `data.len() * 1` elements matching `repr`.
        unsafe { repr.box_to_mat(data) }
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.num_vectors()
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.vector_dim()
    }

    // Reads delegate to the canonical `MatrixView` implementation.
    delegate_read!(pub fn row(&self, i: usize) -> &[T]);
    delegate_read!(pub fn row_iter(&self) -> impl ExactSizeIterator<Item = &[T]>);
    delegate_read!(pub fn window_iter(&self, batchsize: usize) -> impl Iterator<Item = MatrixView<'_, T>>);
    delegate_read!(pub fn subview(&self, rows: std::ops::Range<usize>) -> Option<MatrixView<'_, T>>);
    delegate_read!(pub fn try_get(&self, row: usize, col: usize) -> Option<&T>);
    delegate_read!(pub fn as_ptr(&self) -> *const T);

    /// Parallel iterator over rows.
    #[cfg(feature = "rayon")]
    pub fn par_row_iter(&self) -> impl rayon::iter::IndexedParallelIterator<Item = &[T]>
    where
        T: Sync,
    {
        self.as_view().par_row_iter()
    }

    /// Parallel iterator over sub-matrices of up to `batchsize` rows.
    #[cfg(feature = "rayon")]
    pub fn par_window_iter(
        &self,
        batchsize: usize,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = MatrixView<'_, T>>
    where
        T: Send + Sync,
    {
        self.as_view().par_window_iter(batchsize)
    }

    /// Apply `f` to every element, producing a new owned matrix of the same shape.
    pub fn map<F, R>(&self, f: F) -> Matrix<R>
    where
        F: FnMut(&T) -> R,
    {
        self.as_view().map(f)
    }

    /// Backing data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let len = self.repr.num_elements();
        // SAFETY: the backing allocation holds `len` contiguous `T`, and `&mut self` grants
        // exclusive access for the returned slice's lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.as_raw_mut_ptr().cast::<T>(), len) }
    }

    /// Mutable view over the whole matrix.
    pub fn as_mut_view(&mut self) -> MatrixViewMut<'_, T> {
        self.as_view_mut()
    }

    /// Mutable row `i`. Panics if `i >= self.nrows()`.
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        assert!(
            i < self.nrows(),
            "tried to access row {i} of a matrix with {} rows",
            self.nrows()
        );
        let ncols = self.ncols();
        let start = i * ncols;
        // SAFETY: `i < self.nrows()` was asserted, and the backing slice holds
        // `nrows * ncols` elements, so `start..start + ncols` is in bounds.
        unsafe { self.as_mut_slice().get_unchecked_mut(start..start + ncols) }
    }

    /// Iterator over mutable rows.
    pub fn row_iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut [T]> {
        let ncols = self.ncols();
        self.as_mut_slice().chunks_exact_mut(ncols)
    }

    /// Parallel iterator over mutable rows.
    #[cfg(feature = "rayon")]
    pub fn par_row_iter_mut(&mut self) -> impl rayon::iter::IndexedParallelIterator<Item = &mut [T]>
    where
        T: Send,
    {
        use rayon::slice::ParallelSliceMut;
        let ncols = self.ncols();
        self.as_mut_slice().par_chunks_exact_mut(ncols)
    }

    /// Parallel iterator over mutable sub-matrices of up to `batchsize` rows.
    #[cfg(feature = "rayon")]
    pub fn par_window_iter_mut(
        &mut self,
        batchsize: usize,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = MatrixViewMut<'_, T>>
    where
        T: Send,
    {
        use rayon::slice::ParallelSliceMut;
        assert!(
            batchsize != 0,
            "par_window_iter_mut batchsize cannot be zero"
        );
        let ncols = self.ncols();
        self.as_mut_slice()
            .par_chunks_mut(ncols * batchsize)
            .map(move |d| {
                let nrows = d.len() / ncols;
                MatrixViewMut::try_from(d, nrows, ncols).expect("exact chunk")
            })
    }

    /// Mutable base pointer of the backing data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_slice().as_mut_ptr()
    }
}

// Indexing by `(row, col)`.
impl<T> std::ops::Index<(usize, usize)> for Mat<RowMajor<T>> {
    type Output = T;
    fn index(&self, (row, col): (usize, usize)) -> &T {
        self.try_get(row, col).expect("index out of bounds")
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for Mat<RowMajor<T>> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(
            row < self.nrows() && col < self.ncols(),
            "index out of bounds"
        );
        // SAFETY: `row` and `col` were checked in-bounds above.
        unsafe { self.get_unchecked_mut(row, col) }
    }
}

impl<T> std::ops::Index<(usize, usize)> for MatRef<'_, RowMajor<T>> {
    type Output = T;
    fn index(&self, (row, col): (usize, usize)) -> &T {
        self.try_get(row, col).expect("index out of bounds")
    }
}

impl<T> std::ops::Index<(usize, usize)> for MatMut<'_, RowMajor<T>> {
    type Output = T;
    fn index(&self, (row, col): (usize, usize)) -> &T {
        self.try_get(row, col).expect("index out of bounds")
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for MatMut<'_, RowMajor<T>> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(
            row < self.nrows() && col < self.ncols(),
            "index out of bounds"
        );
        // SAFETY: `row` and `col` were checked in-bounds above.
        unsafe { self.get_unchecked_mut(row, col) }
    }
}

// Equality compares shape and contents.
impl<T: PartialEq> PartialEq for Mat<RowMajor<T>> {
    fn eq(&self, other: &Self) -> bool {
        self.nrows() == other.nrows()
            && self.ncols() == other.ncols()
            && self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq> PartialEq for MatRef<'_, RowMajor<T>> {
    fn eq(&self, other: &Self) -> bool {
        self.nrows() == other.nrows()
            && self.ncols() == other.ncols()
            && self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq> PartialEq for MatMut<'_, RowMajor<T>> {
    fn eq(&self, other: &Self) -> bool {
        self.nrows() == other.nrows()
            && self.ncols() == other.ncols()
            && self.as_slice() == other.as_slice()
    }
}

impl<'a, T> MatRef<'a, RowMajor<T>> {
    /// Reborrow as a shorter-lived view.
    pub fn as_view(&self) -> MatrixView<'_, T> {
        self.reborrow()
    }

    /// Element at `(row, col)` without bounds checking.
    ///
    /// # Safety
    /// `row < self.nrows()` and `col < self.ncols()`.
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> &'a T {
        debug_assert!(row < self.nrows());
        debug_assert!(col < self.ncols());
        let ncols = self.ncols();
        // SAFETY: guaranteed in-bounds by the caller.
        unsafe { self.as_slice().get_unchecked(row * ncols + col) }
    }
}

impl<'a, T> MatMut<'a, RowMajor<T>> {
    /// Reborrow as a shorter-lived mutable view.
    pub fn as_mut_view(&mut self) -> MatrixViewMut<'_, T> {
        self.reborrow_mut()
    }

    delegate_read!(
        /// Element at `(row, col)` without bounds checking.
        ///
        /// # Safety
        /// `row < self.nrows()` and `col < self.ncols()`.
        unsafe pub fn get_unchecked(&self, row: usize, col: usize) -> &T
    );

    /// Mutable element at `(row, col)` without bounds checking.
    ///
    /// # Safety
    /// `row < self.nrows()` and `col < self.ncols()`.
    pub unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut T {
        let ncols = self.ncols();
        // SAFETY: guaranteed in-bounds by the caller.
        unsafe { self.as_mut_slice().get_unchecked_mut(row * ncols + col) }
    }

    /// Parallel iterator over mutable sub-matrices of up to `batchsize` rows.
    #[cfg(feature = "rayon")]
    pub fn par_window_iter_mut(
        &mut self,
        batchsize: usize,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = MatrixViewMut<'_, T>>
    where
        T: Send,
    {
        use rayon::slice::ParallelSliceMut;
        assert!(
            batchsize != 0,
            "par_window_iter_mut batchsize cannot be zero"
        );
        let ncols = self.ncols();
        self.as_mut_slice()
            .par_chunks_mut(ncols * batchsize)
            .map(move |d| {
                let nrows = d.len() / ncols;
                MatrixViewMut::try_from(d, nrows, ncols).expect("exact chunk")
            })
    }
}

impl<T> Mat<RowMajor<T>> {
    delegate_read!(
        /// Element at `(row, col)` without bounds checking.
        ///
        /// # Safety
        /// `row < self.nrows()` and `col < self.ncols()`.
        unsafe pub fn get_unchecked(&self, row: usize, col: usize) -> &T
    );

    /// Mutable element at `(row, col)` without bounds checking.
    ///
    /// # Safety
    /// `row < self.nrows()` and `col < self.ncols()`.
    pub unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut T {
        let ncols = self.ncols();
        // SAFETY: guaranteed in-bounds by the caller.
        unsafe { self.as_mut_slice().get_unchecked_mut(row * ncols + col) }
    }

    /// Consume the matrix, returning its backing storage as a `Box<[T]>`.
    pub fn into_inner(self) -> Box<[T]> {
        let this = core::mem::ManuallyDrop::new(self);
        let slice = this.as_slice();
        let (ptr, len) = (slice.as_ptr().cast_mut(), slice.len());
        // SAFETY: the matrix is backed by a `Box<[T]>` of `len` elements (see
        // `RowMajor::box_to_mat`); `ManuallyDrop` prevents a double free via `Mat`'s `Drop`.
        unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) }
    }
}

// A dense view converts directly to its backing slice.
impl<'a, T> From<MatRef<'a, RowMajor<T>>> for &'a [T] {
    fn from(m: MatRef<'a, RowMajor<T>>) -> Self {
        m.as_slice()
    }
}

// A dense mutable view converts directly to its backing slice.
impl<'a, T> From<MatMut<'a, RowMajor<T>>> for &'a mut [T] {
    fn from(m: MatMut<'a, RowMajor<T>>) -> Self {
        let len = m.repr.num_elements();
        // SAFETY: `m: MatMut<'a, RowMajor<T>>` exclusively views `len` contiguous `T` valid for `'a`.
        unsafe { std::slice::from_raw_parts_mut(m.ptr.as_ptr().cast::<T>(), len) }
    }
}

//////////
// Rows //
//////////

/// Iterator over immutable row references of a matrix.
///
/// Created by [`Mat::rows`], [`MatRef::rows`], or [`MatMut::rows`].
#[derive(Debug)]
pub struct Rows<'a, T: Repr> {
    matrix: MatRef<'a, T>,
    current: usize,
}

impl<'a, T> Rows<'a, T>
where
    T: Repr,
{
    fn new(matrix: MatRef<'a, T>) -> Self {
        Self { matrix, current: 0 }
    }
}

impl<'a, T> Iterator for Rows<'a, T>
where
    T: Repr + 'a,
{
    type Item = T::Row<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current;
        if current >= self.matrix.num_vectors() {
            None
        } else {
            self.current += 1;
            // SAFETY: We make sure through the above check that
            // the access is within bounds.
            //
            // Extending the lifetime to `'a` is safe because the underlying
            // MatRef has lifetime `'a`.
            Some(unsafe { self.matrix.repr.get_row(self.matrix.ptr, current) })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.matrix.num_vectors() - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for Rows<'a, T> where T: Repr + 'a {}
impl<'a, T> FusedIterator for Rows<'a, T> where T: Repr + 'a {}

/////////////
// RowsMut //
/////////////

/// Iterator over mutable row references of a matrix.
///
/// Created by [`Mat::rows_mut`] or [`MatMut::rows_mut`].
#[derive(Debug)]
pub struct RowsMut<'a, T: ReprMut> {
    matrix: MatMut<'a, T>,
    current: usize,
}

impl<'a, T> RowsMut<'a, T>
where
    T: ReprMut,
{
    fn new(matrix: MatMut<'a, T>) -> Self {
        Self { matrix, current: 0 }
    }
}

impl<'a, T> Iterator for RowsMut<'a, T>
where
    T: ReprMut + 'a,
{
    type Item = T::RowMut<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current;
        if current >= self.matrix.num_vectors() {
            None
        } else {
            self.current += 1;
            // SAFETY: We make sure through the above check that
            // the access is within bounds.
            //
            // Extending the lifetime to `'a` is safe because:
            // 1. The underlying MatMut has lifetime `'a`.
            // 2. The iterator ensures that the mutable row indices are disjoint, so
            //    there is no aliasing as long as the implementation of `ReprMut` ensures
            //    there is not mutable sharing of the `RowMut` types.
            Some(unsafe { self.matrix.repr.get_row_mut(self.matrix.ptr, current) })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.matrix.num_vectors() - self.current;
        (remaining, Some(remaining))
    }
}

impl<'a, T> ExactSizeIterator for RowsMut<'a, T> where T: ReprMut + 'a {}
impl<'a, T> FusedIterator for RowsMut<'a, T> where T: ReprMut + 'a {}

///////////////
// DenseData //
///////////////

/// Abstraction over a type that can yield a dense slice of its contents.
///
/// # Safety
///
/// `as_slice` must be idempotent: it must **always** return the same slice with the same
/// length (unsafe code relies on this).
pub unsafe trait DenseData {
    type Elem;

    /// Return the underlying data as a slice.
    fn as_slice(&self) -> &[Self::Elem];
}

/// A mutable companion to [`DenseData`].
///
/// # Safety
///
/// `as_mut_slice` must be idempotent and must span the exact same memory as
/// [`DenseData::as_slice`].
pub unsafe trait MutDenseData: DenseData {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

// SAFETY: fulfills the idempotency requirement.
unsafe impl<T> DenseData for &[T] {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: fulfills the idempotency requirement.
unsafe impl<T> DenseData for &mut [T] {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: fulfills the idempotency requirement and spans the same memory as `as_slice`.
unsafe impl<T> MutDenseData for &mut [T] {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

// SAFETY: fulfills the idempotency requirement.
unsafe impl<T> DenseData for Box<[T]> {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: fulfills the idempotency requirement and spans the same memory as `as_slice`.
unsafe impl<T> MutDenseData for Box<[T]> {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt::Display;

    use crate::lazy_format;

    /// Helper to assert a type is Copy.
    fn assert_copy<T: Copy>(_: &T) {}

    // ── Variance assertions ──────────────────────────────────────
    //
    // These functions are never called. The test is that they compile:
    // covariant positions must accept subtype coercions.
    //
    // The negative (invariance) counterparts live in
    // `tests/compile-fail/multi/{mat,matmut}_invariant.rs`.

    /// `MatRef` is covariant in `'a`: a longer borrow can shorten.
    fn _assert_matref_covariant_lifetime<'long: 'short, 'short, T: Repr>(
        v: MatRef<'long, T>,
    ) -> MatRef<'short, T> {
        v
    }

    /// `MatRef` is covariant in `T`: `RowMajor<&'long u8>` → `RowMajor<&'short u8>`.
    fn _assert_matref_covariant_repr<'long: 'short, 'short, 'a>(
        v: MatRef<'a, RowMajor<&'long u8>>,
    ) -> MatRef<'a, RowMajor<&'short u8>> {
        v
    }

    /// `MatMut` is covariant in `'a`: a longer borrow can shorten.
    fn _assert_matmut_covariant_lifetime<'long: 'short, 'short, T: ReprMut>(
        v: MatMut<'long, T>,
    ) -> MatMut<'short, T> {
        v
    }

    fn edge_cases(nrows: usize) -> Vec<usize> {
        let max = usize::MAX;

        vec![
            nrows,
            nrows + 1,
            nrows + 11,
            nrows + 20,
            max / 2,
            max.div_ceil(2),
            max - 1,
            max,
        ]
    }

    fn fill_mat(x: &mut Mat<RowMajor<usize>>, repr: RowMajor<usize>) {
        assert_eq!(x.repr(), &repr);
        assert_eq!(x.num_vectors(), repr.nrows());
        assert_eq!(x.vector_dim(), repr.ncols());

        for i in 0..x.num_vectors() {
            let row = x.get_row_mut(i).unwrap();
            assert_eq!(row.len(), repr.ncols());
            row.iter_mut()
                .enumerate()
                .for_each(|(j, r)| *r = 10 * i + j);
        }

        for i in edge_cases(repr.nrows()).into_iter() {
            assert!(x.get_row_mut(i).is_none());
        }
    }

    fn fill_mat_mut(mut x: MatMut<'_, RowMajor<usize>>, repr: RowMajor<usize>) {
        assert_eq!(x.repr(), &repr);
        assert_eq!(x.num_vectors(), repr.nrows());
        assert_eq!(x.vector_dim(), repr.ncols());

        for i in 0..x.num_vectors() {
            let row = x.get_row_mut(i).unwrap();
            assert_eq!(row.len(), repr.ncols());

            row.iter_mut()
                .enumerate()
                .for_each(|(j, r)| *r = 10 * i + j);
        }

        for i in edge_cases(repr.nrows()).into_iter() {
            assert!(x.get_row_mut(i).is_none());
        }
    }

    fn fill_rows_mut(x: RowsMut<'_, RowMajor<usize>>, repr: RowMajor<usize>) {
        assert_eq!(x.len(), repr.nrows());
        // Materialize all rows at once.
        let mut all_rows: Vec<_> = x.collect();
        assert_eq!(all_rows.len(), repr.nrows());
        for (i, row) in all_rows.iter_mut().enumerate() {
            assert_eq!(row.len(), repr.ncols());
            row.iter_mut()
                .enumerate()
                .for_each(|(j, r)| *r = 10 * i + j);
        }
    }

    fn check_mat(x: &Mat<RowMajor<usize>>, repr: RowMajor<usize>, ctx: &dyn Display) {
        assert_eq!(x.repr(), &repr);
        assert_eq!(x.num_vectors(), repr.nrows());
        assert_eq!(x.vector_dim(), repr.ncols());

        for i in 0..x.num_vectors() {
            let row = x.get_row(i).unwrap();

            assert_eq!(row.len(), repr.ncols(), "ctx: {ctx}");
            row.iter().enumerate().for_each(|(j, r)| {
                assert_eq!(
                    *r,
                    10 * i + j,
                    "mismatched entry at row {}, col {} -- ctx: {}",
                    i,
                    j,
                    ctx
                )
            });
        }

        for i in edge_cases(repr.nrows()).into_iter() {
            assert!(x.get_row(i).is_none(), "ctx: {ctx}");
        }
    }

    fn check_mat_ref(x: MatRef<'_, RowMajor<usize>>, repr: RowMajor<usize>, ctx: &dyn Display) {
        assert_eq!(x.repr(), &repr);
        assert_eq!(x.num_vectors(), repr.nrows());
        assert_eq!(x.vector_dim(), repr.ncols());

        assert_copy(&x);
        for i in 0..x.num_vectors() {
            let row = x.get_row(i).unwrap();
            assert_eq!(row.len(), repr.ncols(), "ctx: {ctx}");

            row.iter().enumerate().for_each(|(j, r)| {
                assert_eq!(
                    *r,
                    10 * i + j,
                    "mismatched entry at row {}, col {} -- ctx: {}",
                    i,
                    j,
                    ctx
                )
            });
        }

        for i in edge_cases(repr.nrows()).into_iter() {
            assert!(x.get_row(i).is_none(), "ctx: {ctx}");
        }
    }

    fn check_mat_mut(x: MatMut<'_, RowMajor<usize>>, repr: RowMajor<usize>, ctx: &dyn Display) {
        assert_eq!(x.repr(), &repr);
        assert_eq!(x.num_vectors(), repr.nrows());
        assert_eq!(x.vector_dim(), repr.ncols());

        for i in 0..x.num_vectors() {
            let row = x.get_row(i).unwrap();
            assert_eq!(row.len(), repr.ncols(), "ctx: {ctx}");

            row.iter().enumerate().for_each(|(j, r)| {
                assert_eq!(
                    *r,
                    10 * i + j,
                    "mismatched entry at row {}, col {} -- ctx: {}",
                    i,
                    j,
                    ctx
                )
            });
        }

        for i in edge_cases(repr.nrows()).into_iter() {
            assert!(x.get_row(i).is_none(), "ctx: {ctx}");
        }
    }

    fn check_rows(x: Rows<'_, RowMajor<usize>>, repr: RowMajor<usize>, ctx: &dyn Display) {
        assert_eq!(x.len(), repr.nrows(), "ctx: {ctx}");
        let all_rows: Vec<_> = x.collect();
        assert_eq!(all_rows.len(), repr.nrows(), "ctx: {ctx}");
        for (i, row) in all_rows.iter().enumerate() {
            assert_eq!(row.len(), repr.ncols(), "ctx: {ctx}");
            row.iter().enumerate().for_each(|(j, r)| {
                assert_eq!(
                    *r,
                    10 * i + j,
                    "mismatched entry at row {}, col {} -- ctx: {}",
                    i,
                    j,
                    ctx
                )
            });
        }
    }

    //////////////
    // RowMajor //
    //////////////

    #[test]
    fn standard_representation() {
        let repr = RowMajor::<f32>::new(4, 3).unwrap();
        assert_eq!(repr.nrows(), 4);
        assert_eq!(repr.ncols(), 3);

        let layout = repr.layout().unwrap();
        assert_eq!(layout.size(), 4 * 3 * std::mem::size_of::<f32>());
        assert_eq!(layout.align(), std::mem::align_of::<f32>());
    }

    #[test]
    fn standard_zero_dimensions() {
        for (nrows, ncols) in [(0, 0), (0, 5), (5, 0)] {
            let repr = RowMajor::<u8>::new(nrows, ncols).unwrap();
            assert_eq!(repr.nrows(), nrows);
            assert_eq!(repr.ncols(), ncols);
            let layout = repr.layout().unwrap();
            assert_eq!(layout.size(), 0);
        }
    }

    #[test]
    fn standard_check_slice() {
        let repr = RowMajor::<u32>::new(3, 4).unwrap();

        // Correct length succeeds
        let data = vec![0u32; 12];
        assert!(repr.check_slice(&data).is_ok());

        // Too short fails
        let short = vec![0u32; 11];
        assert!(matches!(
            repr.check_slice(&short),
            Err(SliceError::LengthMismatch {
                expected: 12,
                found: 11
            })
        ));

        // Too long fails
        let long = vec![0u32; 13];
        assert!(matches!(
            repr.check_slice(&long),
            Err(SliceError::LengthMismatch {
                expected: 12,
                found: 13
            })
        ));

        // Overflow case
        let overflow_repr = RowMajor::<u8>::new(usize::MAX, 2).unwrap_err();
        assert!(matches!(overflow_repr, Overflow { .. }));
    }

    #[test]
    fn standard_new_rejects_element_count_overflow() {
        // nrows * ncols overflows usize even though per-element size is small.
        assert!(RowMajor::<u8>::new(usize::MAX, 2).is_err());
        assert!(RowMajor::<u8>::new(2, usize::MAX).is_err());
        assert!(RowMajor::<u8>::new(usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn standard_new_rejects_byte_count_exceeding_isize_max() {
        // Element count fits in usize, but total bytes exceed isize::MAX.
        let half = (isize::MAX as usize / std::mem::size_of::<u64>()) + 1;
        assert!(RowMajor::<u64>::new(half, 1).is_err());
        assert!(RowMajor::<u64>::new(1, half).is_err());
    }

    #[test]
    fn standard_new_accepts_boundary_below_isize_max() {
        // Largest allocation that still fits in isize::MAX bytes.
        let max_elems = isize::MAX as usize / std::mem::size_of::<u64>();
        let repr = RowMajor::<u64>::new(max_elems, 1).unwrap();
        assert_eq!(repr.num_elements(), max_elems);
    }

    #[test]
    fn standard_new_zst_rejects_element_count_overflow() {
        // For ZSTs the byte count is always 0, but element-count overflow
        // must still be caught so that `num_elements()` never wraps.
        assert!(RowMajor::<()>::new(usize::MAX, 2).is_err());
        assert!(RowMajor::<()>::new(usize::MAX / 2 + 1, 3).is_err());
    }

    #[test]
    fn standard_new_zst_accepts_large_non_overflowing() {
        // Large-but-valid ZST matrix: element count fits in usize.
        let repr = RowMajor::<()>::new(usize::MAX, 1).unwrap();
        assert_eq!(repr.num_elements(), usize::MAX);
        assert_eq!(repr.layout().unwrap().size(), 0);
    }

    #[test]
    fn standard_new_overflow_error_display() {
        let err = RowMajor::<u32>::new(usize::MAX, 2).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("would exceed isize::MAX bytes"), "{msg}");

        let zst_err = RowMajor::<()>::new(usize::MAX, 2).unwrap_err();
        let zst_msg = zst_err.to_string();
        assert!(zst_msg.contains("ZST matrix"), "{zst_msg}");
        assert!(zst_msg.contains("usize::MAX"), "{zst_msg}");
    }

    #[test]
    fn try_from_error_recovers_owned_data() {
        // A failed owned `try_from` must hand the `Box` back, not deallocate it.
        let data: Box<[i32]> = vec![1, 2, 3, 4, 5, 6].into_boxed_slice();
        let err = Mat::try_from(data, 4, 4).unwrap_err();
        assert_eq!(&*err.into_inner(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn eq_distinguishes_row_count_for_zero_columns() {
        // With zero columns the backing slice is empty for any row count, so equality
        // must compare the row count too: a 5x0 and a 3x0 matrix are not equal.
        let a = Mat::from_repr(RowMajor::<i32>::new(5, 0).unwrap(), 0).unwrap();
        let b = Mat::from_repr(RowMajor::<i32>::new(3, 0).unwrap(), 0).unwrap();
        assert!(a != b);
        assert!(a == Mat::from_repr(RowMajor::<i32>::new(5, 0).unwrap(), 0).unwrap());
    }

    /////////
    // Mat //
    /////////

    #[test]
    fn mat_new_and_basic_accessors() {
        let mat = Mat::from_repr(RowMajor::<usize>::new(3, 4).unwrap(), 42usize).unwrap();
        let base: *const u8 = mat.as_raw_ptr();

        assert_eq!(mat.num_vectors(), 3);
        assert_eq!(mat.vector_dim(), 4);

        let repr = mat.repr();
        assert_eq!(repr.nrows(), 3);
        assert_eq!(repr.ncols(), 4);

        for (i, r) in mat.rows().enumerate() {
            assert_eq!(r, &[42, 42, 42, 42]);
            let ptr = r.as_ptr().cast::<u8>();
            assert_eq!(
                ptr,
                base.wrapping_add(std::mem::size_of::<usize>() * mat.repr().ncols() * i),
            );
        }
    }

    #[test]
    fn mat_new_with_default() {
        let mat = Mat::from_default(RowMajor::<usize>::new(2, 3).unwrap()).unwrap();
        let base: *const u8 = mat.as_raw_ptr();

        assert_eq!(mat.num_vectors(), 2);
        for (i, row) in mat.rows().enumerate() {
            assert!(row.iter().all(|&v| v == 0));

            let ptr = row.as_ptr().cast::<u8>();
            assert_eq!(
                ptr,
                base.wrapping_add(std::mem::size_of::<usize>() * mat.repr().ncols() * i),
            );
        }
    }

    const ROWS: &[usize] = &[0, 1, 2, 3, 5, 10];
    const COLS: &[usize] = &[0, 1, 2, 3, 5, 10];

    #[test]
    fn test_mat() {
        for nrows in ROWS {
            for ncols in COLS {
                let repr = RowMajor::<usize>::new(*nrows, *ncols).unwrap();
                let ctx = &lazy_format!("nrows = {}, ncols = {}", nrows, ncols);

                // Populate the matrix using `&mut Mat`
                {
                    let ctx = &lazy_format!("{ctx} - direct");
                    let mut mat = Mat::from_default(repr).unwrap();

                    assert_eq!(mat.num_vectors(), *nrows);
                    assert_eq!(mat.vector_dim(), *ncols);

                    fill_mat(&mut mat, repr);

                    check_mat(&mat, repr, ctx);
                    check_mat_ref(mat.reborrow(), repr, ctx);
                    check_mat_mut(mat.reborrow_mut(), repr, ctx);
                    check_rows(mat.rows(), repr, ctx);

                    // Check reborrow preserves pointers.
                    assert_eq!(mat.as_raw_ptr(), mat.reborrow().as_raw_ptr());
                    assert_eq!(mat.as_raw_ptr(), mat.reborrow_mut().as_raw_ptr());
                }

                // Populate the matrix using `MatMut`
                {
                    let ctx = &lazy_format!("{ctx} - matmut");
                    let mut mat = Mat::from_default(repr).unwrap();
                    let matmut = mat.reborrow_mut();

                    assert_eq!(matmut.num_vectors(), *nrows);
                    assert_eq!(matmut.vector_dim(), *ncols);

                    fill_mat_mut(matmut, repr);

                    check_mat(&mat, repr, ctx);
                    check_mat_ref(mat.reborrow(), repr, ctx);
                    check_mat_mut(mat.reborrow_mut(), repr, ctx);
                    check_rows(mat.rows(), repr, ctx);
                }

                // Populate the matrix using `RowsMut`
                {
                    let ctx = &lazy_format!("{ctx} - rows_mut");
                    let mut mat = Mat::from_default(repr).unwrap();
                    fill_rows_mut(mat.rows_mut(), repr);

                    check_mat(&mat, repr, ctx);
                    check_mat_ref(mat.reborrow(), repr, ctx);
                    check_mat_mut(mat.reborrow_mut(), repr, ctx);
                    check_rows(mat.rows(), repr, ctx);
                }
            }
        }
    }

    #[test]
    fn test_mat_clone() {
        for nrows in ROWS {
            for ncols in COLS {
                let repr = RowMajor::<usize>::new(*nrows, *ncols).unwrap();
                let ctx = &lazy_format!("nrows = {}, ncols = {}", nrows, ncols);

                let mut mat = Mat::from_default(repr).unwrap();
                fill_mat(&mut mat, repr);

                // Clone via Mat::clone
                {
                    let ctx = &lazy_format!("{ctx} - Mat::clone");
                    let cloned = mat.clone();

                    assert_eq!(cloned.num_vectors(), *nrows);
                    assert_eq!(cloned.vector_dim(), *ncols);

                    check_mat(&cloned, repr, ctx);
                    check_mat_ref(cloned.reborrow(), repr, ctx);
                    check_rows(cloned.rows(), repr, ctx);

                    // Cloned allocation is independent.
                    if repr.num_elements() > 0 {
                        assert_ne!(mat.as_raw_ptr(), cloned.as_raw_ptr());
                    }
                }

                // Clone via MatRef::to_owned
                {
                    let ctx = &lazy_format!("{ctx} - MatRef::to_owned");
                    let owned = mat.as_view().to_owned();

                    check_mat(&owned, repr, ctx);
                    check_mat_ref(owned.reborrow(), repr, ctx);
                    check_rows(owned.rows(), repr, ctx);

                    if repr.num_elements() > 0 {
                        assert_ne!(mat.as_raw_ptr(), owned.as_raw_ptr());
                    }
                }

                // Clone via MatMut::to_owned
                {
                    let ctx = &lazy_format!("{ctx} - MatMut::to_owned");
                    let owned = mat.as_view_mut().to_owned();

                    check_mat(&owned, repr, ctx);
                    check_mat_ref(owned.reborrow(), repr, ctx);
                    check_rows(owned.rows(), repr, ctx);

                    if repr.num_elements() > 0 {
                        assert_ne!(mat.as_raw_ptr(), owned.as_raw_ptr());
                    }
                }
            }
        }
    }

    #[test]
    fn test_mat_refmut() {
        for nrows in ROWS {
            for ncols in COLS {
                let repr = RowMajor::<usize>::new(*nrows, *ncols).unwrap();
                let ctx = &lazy_format!("nrows = {}, ncols = {}", nrows, ncols);

                // Populate the matrix using `&mut Mat`
                {
                    let ctx = &lazy_format!("{ctx} - by matmut");
                    let mut b: Box<[_]> = (0..repr.num_elements()).map(|_| 0usize).collect();
                    let ptr = b.as_ptr().cast::<u8>();
                    let mut matmut = MatMut::from_repr(repr, &mut b).unwrap();

                    assert_eq!(
                        ptr,
                        matmut.as_raw_ptr(),
                        "underlying memory should be preserved",
                    );

                    fill_mat_mut(matmut.reborrow_mut(), repr);

                    check_mat_mut(matmut.reborrow_mut(), repr, ctx);
                    check_mat_ref(matmut.reborrow(), repr, ctx);
                    check_rows(matmut.rows(), repr, ctx);
                    check_rows(matmut.reborrow().rows(), repr, ctx);

                    let matref = MatRef::from_repr(repr, &b).unwrap();
                    check_mat_ref(matref, repr, ctx);
                    check_mat_ref(matref.reborrow(), repr, ctx);
                    check_rows(matref.rows(), repr, ctx);
                }

                // Populate the matrix using `RowsMut`
                {
                    let ctx = &lazy_format!("{ctx} - by rows");
                    let mut b: Box<[_]> = (0..repr.num_elements()).map(|_| 0usize).collect();
                    let ptr = b.as_ptr().cast::<u8>();
                    let mut matmut = MatMut::from_repr(repr, &mut b).unwrap();

                    assert_eq!(
                        ptr,
                        matmut.as_raw_ptr(),
                        "underlying memory should be preserved",
                    );

                    fill_rows_mut(matmut.rows_mut(), repr);

                    check_mat_mut(matmut.reborrow_mut(), repr, ctx);
                    check_mat_ref(matmut.reborrow(), repr, ctx);
                    check_rows(matmut.rows(), repr, ctx);
                    check_rows(matmut.reborrow().rows(), repr, ctx);

                    let matref = MatRef::from_repr(repr, &b).unwrap();
                    check_mat_ref(matref, repr, ctx);
                    check_mat_ref(matref.reborrow(), repr, ctx);
                    check_rows(matref.rows(), repr, ctx);
                }
            }
        }
    }

    //////////////////
    // Constructors //
    //////////////////

    #[test]
    fn test_standard_new_owned() {
        let rows = [0, 1, 2, 3, 5, 10];
        let cols = [0, 1, 2, 3, 5, 10];

        for nrows in rows {
            for ncols in cols {
                let m = Mat::from_repr(RowMajor::new(nrows, ncols).unwrap(), 1usize).unwrap();
                let rows_iter = m.rows();
                let len = <_ as ExactSizeIterator>::len(&rows_iter);
                assert_eq!(len, nrows);
                for r in rows_iter {
                    assert_eq!(r.len(), ncols);
                    assert!(r.iter().all(|i| *i == 1usize));
                }
            }
        }
    }

    #[test]
    fn test_mat_from_fn() {
        let rows = [0, 1, 2, 5];
        let cols = [0, 1, 3, 7];

        for nrows in rows {
            for ncols in cols {
                let mut counter = 0u32;
                let m = Mat::new(
                    Init(|| {
                        let v = counter;
                        counter += 1;
                        v
                    }),
                    nrows,
                    ncols,
                );

                assert_eq!(counter as usize, nrows * ncols);
                for (i, row) in m.rows().enumerate() {
                    assert_eq!(row.len(), ncols);
                    for (j, &v) in row.iter().enumerate() {
                        assert_eq!(v, (i * ncols + j) as u32);
                    }
                }
            }
        }
    }

    #[test]
    fn matref_new_slice_length_error() {
        let repr = RowMajor::<u32>::new(3, 4).unwrap();

        // Correct length succeeds
        let data = vec![0u32; 12];
        assert!(MatRef::from_repr(repr, &data).is_ok());

        // Too short fails
        let short = vec![0u32; 11];
        assert!(matches!(
            MatRef::from_repr(repr, &short),
            Err(SliceError::LengthMismatch {
                expected: 12,
                found: 11
            })
        ));

        // Too long fails
        let long = vec![0u32; 13];
        assert!(matches!(
            MatRef::from_repr(repr, &long),
            Err(SliceError::LengthMismatch {
                expected: 12,
                found: 13
            })
        ));
    }

    #[test]
    fn matmut_new_slice_length_error() {
        let repr = RowMajor::<u32>::new(3, 4).unwrap();

        // Correct length succeeds
        let mut data = vec![0u32; 12];
        assert!(MatMut::from_repr(repr, &mut data).is_ok());

        // Too short fails
        let mut short = vec![0u32; 11];
        assert!(matches!(
            MatMut::from_repr(repr, &mut short),
            Err(SliceError::LengthMismatch {
                expected: 12,
                found: 11
            })
        ));

        // Too long fails
        let mut long = vec![0u32; 13];
        assert!(matches!(
            MatMut::from_repr(repr, &mut long),
            Err(SliceError::LengthMismatch {
                expected: 12,
                found: 13
            })
        ));
    }

    #[test]
    fn as_matrix_view_roundtrip() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        // MatRef
        let matref = MatRef::from_repr(RowMajor::new(2, 3).unwrap(), &data).unwrap();
        let view = matref.as_matrix_view();
        assert_eq!(view.nrows(), 2);
        assert_eq!(view.ncols(), 3);
        for row in 0..2 {
            for col in 0..3 {
                assert_eq!(view[(row, col)], data[row * 3 + col]);
            }
        }
        assert_eq!(matref.as_slice(), &data);

        // Mat
        let mut mat = Mat::from_repr(RowMajor::<f32>::new(2, 3).unwrap(), 0.0f32).unwrap();
        for i in 0..2 {
            let r = mat.get_row_mut(i).unwrap();
            for j in 0..3 {
                r[j] = data[i * 3 + j];
            }
        }
        let view = mat.as_matrix_view();
        assert_eq!(view.nrows(), 2);
        assert_eq!(view.ncols(), 3);
        for row in 0..2 {
            for col in 0..3 {
                assert_eq!(view[(row, col)], data[row * 3 + col]);
            }
        }
        assert_eq!(mat.as_slice(), &data);

        // MatMut
        let mut buf = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matmut = MatMut::from_repr(RowMajor::new(2, 3).unwrap(), &mut buf).unwrap();
        let view = matmut.as_matrix_view();
        assert_eq!(view.nrows(), 2);
        assert_eq!(view.ncols(), 3);
        for row in 0..2 {
            for col in 0..3 {
                assert_eq!(view[(row, col)], data[row * 3 + col]);
            }
        }
        assert_eq!(matmut.as_slice(), &data);
    }

    #[test]
    fn test_standard_non_copy_element() {
        let repr = RowMajor::<String>::new(2, 3).unwrap();

        // Owned fill via NewOwned<T> (Clone).
        let filled = Mat::from_repr(repr, String::from("x")).unwrap();
        assert_eq!(filled.num_vectors(), 2);
        assert!(filled.rows().flatten().all(|s| s == "x"));

        // NewDefault (Default).
        let defaulted = Mat::from_default(repr).unwrap();
        assert!(defaulted.rows().flatten().all(String::is_empty));

        // from_fn.
        let mut counter = 0usize;
        let mut mat = Mat::new(
            Init(|| {
                let s = counter.to_string();
                counter += 1;
                s
            }),
            2,
            3,
        );
        assert_eq!(counter, 6);
        assert_eq!(mat.get_row(1).unwrap()[0], "3");

        // Mutation via get_row_mut.
        mat.get_row_mut(0).unwrap()[0] = String::from("mutated");
        assert_eq!(mat.get_row(0).unwrap()[0], "mutated");

        // Clone via NewCloned (Clone): independent allocation, equal contents.
        let cloned = mat.clone();
        assert_ne!(mat.as_raw_ptr(), cloned.as_raw_ptr());
        assert_eq!(cloned.get_row(0).unwrap()[0], "mutated");

        // Immutable view over a non-Copy slice (NewRef).
        let data = [String::from("a"), String::from("b")];
        let view = MatRef::from_repr(RowMajor::new(2, 1).unwrap(), &data).unwrap();
        assert_eq!(view.get_row(1).unwrap()[0], "b");

        // Mutable view over a non-Copy slice (NewMut).
        let mut data_mut = [String::from("a"), String::from("b")];
        let mut view_mut = MatMut::from_repr(RowMajor::new(1, 2).unwrap(), &mut data_mut).unwrap();
        view_mut.get_row_mut(0).unwrap()[1] = String::from("z");
        assert_eq!(data_mut[1], "z");
    }

    // ── Dense read API (canonical on `MatrixView`, delegated by `Matrix`/`MatrixViewMut`) ──

    #[test]
    fn dense_subview_and_try_get() {
        let m: Matrix<i32> =
            Matrix::try_from(vec![1, 2, 3, 4, 5, 6].into_boxed_slice(), 3, 2).unwrap();
        let sv = m.subview(1..3).unwrap();
        assert_eq!((sv.nrows(), sv.ncols()), (2, 2));
        assert_eq!(sv.row(0), &[3, 4]);
        assert_eq!(sv.row(1), &[5, 6]);
        assert!(m.subview(2..4).is_none());
        assert_eq!(m.try_get(2, 1), Some(&6));
        assert!(m.try_get(3, 0).is_none());
        assert!(m.try_get(0, 2).is_none());
    }

    #[test]
    fn dense_map_and_vectors() {
        let m: Matrix<i32> = Matrix::try_from(vec![1, 2, 3, 4].into_boxed_slice(), 2, 2).unwrap();
        let doubled = m.map(|&x| x * 2);
        assert_eq!(doubled.as_slice(), &[2, 4, 6, 8]);
        assert_eq!((doubled.nrows(), doubled.ncols()), (2, 2));

        let rv = Matrix::row_vector(vec![7, 8, 9].into_boxed_slice());
        assert_eq!((rv.nrows(), rv.ncols()), (1, 3));
        let cv = Matrix::column_vector(vec![7, 8, 9].into_boxed_slice());
        assert_eq!((cv.nrows(), cv.ncols()), (3, 1));
    }

    #[test]
    fn dense_window_iter_batches_and_remainder() {
        let m: Matrix<i32> =
            Matrix::try_from((0..9).collect::<Vec<_>>().into_boxed_slice(), 3, 3).unwrap();
        let rows: Vec<_> = m.window_iter(2).map(|w| w.nrows()).collect();
        assert_eq!(rows, vec![2, 1]);
    }

    #[test]
    #[should_panic(expected = "window_iter batchsize cannot be zero")]
    fn dense_window_iter_zero_panics() {
        let m: Matrix<i32> = Matrix::try_from(vec![1, 2].into_boxed_slice(), 1, 2).unwrap();
        let _ = m.window_iter(0).count();
    }

    #[test]
    #[should_panic(expected = "tried to access row 5 of a matrix with 1 rows")]
    fn dense_row_out_of_bounds_panics() {
        let m: Matrix<i32> = Matrix::try_from(vec![1, 2].into_boxed_slice(), 1, 2).unwrap();
        let _ = m.row(5);
    }

    #[test]
    fn dense_get_unchecked_reads_and_writes() {
        let mut m: Matrix<i32> =
            Matrix::try_from(vec![1, 2, 3, 4].into_boxed_slice(), 2, 2).unwrap();
        // SAFETY: `(1, 1)` is in bounds for a 2x2 matrix.
        assert_eq!(unsafe { *m.get_unchecked(1, 1) }, 4);
        // SAFETY: `(0, 0)` is in bounds.
        unsafe { *m.get_unchecked_mut(0, 0) = 9 };
        assert_eq!(m.as_slice(), &[9, 2, 3, 4]);
    }

    #[test]
    fn dense_reads_delegate_consistently() {
        let mut m: Matrix<i32> =
            Matrix::try_from(vec![1, 2, 3, 4, 5, 6].into_boxed_slice(), 3, 2).unwrap();
        let owned = m.row(1).to_vec();
        let via_view = m.as_view().row(1).to_vec();
        let via_mut_view = m.as_mut_view().row(1).to_vec();
        assert_eq!(owned, &[3, 4]);
        assert_eq!(owned, via_view);
        assert_eq!(owned, via_mut_view);
        // The mutable view exposes the full read set (subview/window_iter/as_ptr/...).
        assert!(m.as_mut_view().subview(0..1).is_some());
        assert_eq!(m.as_mut_view().window_iter(2).count(), 2);
    }
}
