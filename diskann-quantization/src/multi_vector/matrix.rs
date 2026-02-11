// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

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

use diskann_utils::{Reborrow, ReborrowMut};
use thiserror::Error;

use crate::utils;

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

/// An initializer argument to [`NewOwned`] that uses a type's [`Default`] implementation
/// to initialize a matrix.
///
/// ```rust
/// use diskann_quantization::multi_vector::{Mat, Standard, Defaulted};
/// let mat = Mat::new(Standard::<f32>::new(4, 3).unwrap(), Defaulted).unwrap();
/// for i in 0..4 {
///     assert!(mat.get_row(i).unwrap().iter().all(|&x| x == 0.0f32));
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Defaulted;

/// Create a new [`Mat`] cloned from a view.
pub trait NewCloned: ReprOwned {
    fn new_cloned(v: MatRef<'_, Self>) -> Mat<Self>;
}

//////////////
// Standard //
//////////////

/// Metadata for dense row-major matrices of `Copy` types.
///
/// Rows are stored contiguously as `&[T]` slices. This is the default representation
/// type for standard floating-point multi-vectors.
///
/// # Row Types
///
/// - `Row<'a>`: `&'a [T]`
/// - `RowMut<'a>`: `&'a mut [T]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Standard<T> {
    nrows: usize,
    ncols: usize,
    _elem: PhantomData<T>,
}

impl<T: Copy> Standard<T> {
    /// Create a new `Standard` for data of type `T`.
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
    /// The value [`Standard::num_elements`] must return `Some(v)` and `b` must have length `v`.
    unsafe fn box_to_mat(self, b: Box<[T]>) -> Mat<Self> {
        debug_assert_eq!(
            b.len(),
            self.num_elements()
                .expect("safety contract requires `self` to be well-formed"),
            "safety contract violated"
        );

        // SAFETY: Box [guarantees](https://doc.rust-lang.org/std/boxed/struct.Box.html#method.into_raw)
        // the returned pointer is non-null.
        let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(b)) }.cast::<u8>();

        // SAFETY: `ptr` is properly aligned and points to a slice of the required length.
        // Additionally, it is dropped via `Box::from_raw`, which is compatible with obtaining
        // it from `Box::into_raw`.
        unsafe { Mat::from_raw_parts(self, ptr) }
    }
}

/// Error for [`Standard::new`].
#[derive(Debug, Clone, Copy)]
pub struct Overflow {
    nrows: usize,
    ncols: usize,
    elsize: usize,
}

impl Overflow {
    fn check<T>(nrows: usize, ncols: usize) -> Result<(), Self> {
        let elsize = std::mem::size_of::<T>();
        // Guard the element count itself so that `num_elements()` can never overflow.
        let elements = nrows.checked_mul(ncols).ok_or(Self {
            nrows,
            ncols,
            elsize,
        })?;

        let bytes = elsize.saturating_mul(elements);
        if bytes <= isize::MAX as usize {
            Ok(())
        } else {
            Err(Self {
                nrows,
                ncols,
                elsize,
            })
        }
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

/// Error types for [`Standard`].
#[derive(Debug, Clone, Copy, Error)]
#[non_exhaustive]
pub enum SliceError {
    #[error("Length mismatch: expected {expected}, found {found}")]
    LengthMismatch { expected: usize, found: usize },
}

// SAFETY: The implementation correctly computes row offsets as `i * ncols` and
// constructs valid slices of the appropriate length. The `layout` method correctly
// reports the memory layout requirements.
unsafe impl<T: Copy> Repr for Standard<T> {
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

        let row_ptr = ptr.as_ptr().cast::<T>().add(i * self.ncols);
        std::slice::from_raw_parts(row_ptr, self.ncols)
    }
}

// SAFETY: The implementation correctly computes row offsets and constructs valid mutable
// slices.
unsafe impl<T: Copy> ReprMut for Standard<T> {
    type RowMut<'a>
        = &'a mut [T]
    where
        T: 'a;

    unsafe fn get_row_mut<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::RowMut<'a> {
        debug_assert!(ptr.cast::<T>().is_aligned());
        debug_assert!(i < self.nrows);

        let row_ptr = ptr.as_ptr().cast::<T>().add(i * self.ncols);
        std::slice::from_raw_parts_mut(row_ptr, self.ncols)
    }
}

// SAFETY: The drop implementation correctly reconstructs a Box from the raw pointer
// using the same length (nrows * ncols) that was used for allocation, allowing Box
// to properly deallocate the memory.
unsafe impl<T: Copy> ReprOwned for Standard<T> {
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
unsafe impl<T> NewOwned<T> for Standard<T>
where
    T: Copy,
{
    type Error = crate::error::Infallible;
    fn new_owned(self, value: T) -> Result<Mat<Self>, Self::Error> {
        let b: Box<[T]> = (0..self.num_elements().unwrap()).map(|_| value).collect();

        // SAFETY: By construction, `b` has length `self.num_elements()`. Since we did
        // not panic when creating `b`, we know that `num_elements()` is well formed.
        Ok(unsafe { self.box_to_mat(b) })
    }
}

// SAFETY: This safely reuses `<Self as NewOwned<T>>`.
unsafe impl<T> NewOwned<Defaulted> for Standard<T>
where
    T: Copy + Default,
{
    type Error = crate::error::Infallible;
    fn new_owned(self, _: Defaulted) -> Result<Mat<Self>, Self::Error> {
        self.new_owned(T::default())
    }
}

// SAFETY: This checks that the slice has the correct length, which is all that is
// required for [`Repr`].
unsafe impl<T> NewRef<T> for Standard<T>
where
    T: Copy,
{
    type Error = SliceError;
    fn new_ref(self, data: &[T]) -> Result<MatRef<'_, Self>, Self::Error> {
        self.check_slice(data)?;

        // SAFETY: The function `check_slice` verifies that `data` is compatible with
        // the layout requirement of `Standard`.
        //
        // We've properly checked that the underlying pointer is okay.
        Ok(unsafe { MatRef::from_raw_parts(self, utils::as_nonnull(data).cast::<u8>()) })
    }
}

// SAFETY: This checks that the slice has the correct length, which is all that is
// required for [`ReprMut`].
unsafe impl<T> NewMut<T> for Standard<T>
where
    T: Copy,
{
    type Error = SliceError;
    fn new_mut(self, data: &mut [T]) -> Result<MatMut<'_, Self>, Self::Error> {
        self.check_slice(data)?;

        // SAFETY: The function `check_slice` verifies that `data` is compatible with
        // the layout requirement of `Standard`.
        //
        // We've properly checked that the underlying pointer is okay.
        Ok(unsafe { MatMut::from_raw_parts(self, utils::as_nonnull_mut(data).cast::<u8>()) })
    }
}

impl<T> NewCloned for Standard<T>
where
    T: Copy,
{
    fn new_cloned(v: MatRef<'_, Self>) -> Mat<Self> {
        let b: Box<[T]> = v.rows().flatten().copied().collect();

        // SAFETY: By construction, `b` has length `v.repr().num_elements()`. Furthermore,
        // since `v` is a valid `MatRef`, we know that `v.repr().num_elements()` cannot overflow.
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
}

// SAFETY: [`Repr`] is required to propagate its `Send` bound.
unsafe impl<T> Send for Mat<T> where T: ReprOwned + Send {}

// SAFETY: [`Repr`] is required to propagate its `Sync` bound.
unsafe impl<T> Sync for Mat<T> where T: ReprOwned + Sync {}

impl<T: ReprOwned> Mat<T> {
    /// Create a new matrix using `init` as the initializer.
    pub fn new<U>(repr: T, init: U) -> Result<Self, <T as NewOwned<U>>::Error>
    where
        T: NewOwned<U>,
    {
        repr.new_owned(init)
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

    pub(crate) unsafe fn get_row_unchecked(&self, i: usize) -> T::Row<'_> {
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
    pub(crate) unsafe fn from_raw_parts(repr: T, ptr: NonNull<u8>) -> Self {
        Self { ptr, repr }
    }

    #[cfg(test)]
    fn as_ptr(&self) -> NonNull<u8> {
        self.ptr
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

impl<T: Copy> Mat<Standard<T>> {
    /// Returns the raw dimension (columns) of the vectors in the matrix.
    #[inline]
    pub fn vector_dim(&self) -> usize {
        self.repr.ncols()
    }
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
    pub(crate) ptr: NonNull<u8>,
    pub(crate) repr: T,
    /// Marker to tie the lifetime to the borrowed data.
    pub(crate) _lifetime: PhantomData<&'a [u8]>,
}

// SAFETY: [`Repr`] is required to propagate its `Send` bound.
unsafe impl<T> Send for MatRef<'_, T> where T: Repr + Send {}

// SAFETY: [`Repr`] is required to propagate its `Sync` bound.
unsafe impl<T> Sync for MatRef<'_, T> where T: Repr + Sync {}

impl<'a, T: Repr> MatRef<'a, T> {
    /// Construct a new [`MatRef`] over `data`.
    pub fn new<U>(repr: T, data: &'a [U]) -> Result<Self, T::Error>
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

    /// Returns the i-th row without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be less than `self.num_vectors()`.
    #[inline]
    pub(crate) unsafe fn get_row_unchecked(&self, i: usize) -> T::Row<'_> {
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
}

impl<'a, T: Copy> MatRef<'a, Standard<T>> {
    /// Returns the raw dimension (columns) of the vectors in the matrix.
    #[inline]
    pub fn vector_dim(&self) -> usize {
        self.repr.ncols()
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
    pub(crate) ptr: NonNull<u8>,
    pub(crate) repr: T,
    /// Marker to tie the lifetime to the mutably borrowed data.
    pub(crate) _lifetime: PhantomData<&'a mut [u8]>,
}

// SAFETY: [`ReprMut`] is required to propagate its `Send` bound.
unsafe impl<T> Send for MatMut<'_, T> where T: ReprMut + Send {}

// SAFETY: [`ReprMut`] is required to propagate its `Sync` bound.
unsafe impl<T> Sync for MatMut<'_, T> where T: ReprMut + Sync {}

impl<'a, T: ReprMut> MatMut<'a, T> {
    /// Construct a new [`MatMut`] over `data`.
    pub fn new<U>(repr: T, data: &'a mut [U]) -> Result<Self, T::Error>
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
    pub(crate) unsafe fn get_row_unchecked(&self, i: usize) -> T::Row<'_> {
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

impl<'a, T: Copy> MatMut<'a, Standard<T>> {
    /// Returns the raw dimension (columns) of the vectors in the matrix.
    #[inline]
    pub fn vector_dim(&self) -> usize {
        self.repr.ncols()
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt::Display;

    use diskann_utils::lazy_format;

    /// Helper to assert a type is Copy.
    fn assert_copy<T: Copy>(_: &T) {}

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

    fn fill_mat(x: &mut Mat<Standard<usize>>, repr: Standard<usize>) {
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

    fn fill_mat_mut(mut x: MatMut<'_, Standard<usize>>, repr: Standard<usize>) {
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

    fn fill_rows_mut(x: RowsMut<'_, Standard<usize>>, repr: Standard<usize>) {
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

    fn check_mat(x: &Mat<Standard<usize>>, repr: Standard<usize>, ctx: &dyn Display) {
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

    fn check_mat_ref(x: MatRef<'_, Standard<usize>>, repr: Standard<usize>, ctx: &dyn Display) {
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

    fn check_mat_mut(x: MatMut<'_, Standard<usize>>, repr: Standard<usize>, ctx: &dyn Display) {
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

    fn check_rows(x: Rows<'_, Standard<usize>>, repr: Standard<usize>, ctx: &dyn Display) {
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
    // Standard //
    //////////////

    #[test]
    fn standard_representation() {
        let repr = Standard::<f32>::new(4, 3).unwrap();
        assert_eq!(repr.nrows(), 4);
        assert_eq!(repr.ncols(), 3);

        let layout = repr.layout().unwrap();
        assert_eq!(layout.size(), 4 * 3 * std::mem::size_of::<f32>());
        assert_eq!(layout.align(), std::mem::align_of::<f32>());
    }

    #[test]
    fn standard_zero_dimensions() {
        for (nrows, ncols) in [(0, 0), (0, 5), (5, 0)] {
            let repr = Standard::<u8>::new(nrows, ncols).unwrap();
            assert_eq!(repr.nrows(), nrows);
            assert_eq!(repr.ncols(), ncols);
            let layout = repr.layout().unwrap();
            assert_eq!(layout.size(), 0);
        }
    }

    #[test]
    fn standard_check_slice() {
        let repr = Standard::<u32>::new(3, 4).unwrap();

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
        let overflow_repr = Standard::<u8>::new(usize::MAX, 2).unwrap_err();
        assert!(matches!(overflow_repr, Overflow { .. }));
    }

    #[test]
    fn standard_new_rejects_element_count_overflow() {
        // nrows * ncols overflows usize even though per-element size is small.
        assert!(Standard::<u8>::new(usize::MAX, 2).is_err());
        assert!(Standard::<u8>::new(2, usize::MAX).is_err());
        assert!(Standard::<u8>::new(usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn standard_new_rejects_byte_count_exceeding_isize_max() {
        // Element count fits in usize, but total bytes exceed isize::MAX.
        let half = (isize::MAX as usize / std::mem::size_of::<u64>()) + 1;
        assert!(Standard::<u64>::new(half, 1).is_err());
        assert!(Standard::<u64>::new(1, half).is_err());
    }

    #[test]
    fn standard_new_accepts_boundary_below_isize_max() {
        // Largest allocation that still fits in isize::MAX bytes.
        let max_elems = isize::MAX as usize / std::mem::size_of::<u64>();
        let repr = Standard::<u64>::new(max_elems, 1).unwrap();
        assert_eq!(repr.num_elements(), max_elems);
    }

    #[test]
    fn standard_new_zst_rejects_element_count_overflow() {
        // For ZSTs the byte count is always 0, but element-count overflow
        // must still be caught so that `num_elements()` never wraps.
        assert!(Standard::<()>::new(usize::MAX, 2).is_err());
        assert!(Standard::<()>::new(usize::MAX / 2 + 1, 3).is_err());
    }

    #[test]
    fn standard_new_zst_accepts_large_non_overflowing() {
        // Large-but-valid ZST matrix: element count fits in usize.
        let repr = Standard::<()>::new(usize::MAX, 1).unwrap();
        assert_eq!(repr.num_elements(), usize::MAX);
        assert_eq!(repr.layout().unwrap().size(), 0);
    }

    #[test]
    fn standard_new_overflow_error_display() {
        let err = Standard::<u32>::new(usize::MAX, 2).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("would exceed isize::MAX bytes"), "{msg}");

        let zst_err = Standard::<()>::new(usize::MAX, 2).unwrap_err();
        let zst_msg = zst_err.to_string();
        assert!(zst_msg.contains("ZST matrix"), "{zst_msg}");
        assert!(zst_msg.contains("usize::MAX"), "{zst_msg}");
    }

    /////////
    // Mat //
    /////////

    #[test]
    fn mat_new_and_basic_accessors() {
        let mat = Mat::new(Standard::<usize>::new(3, 4).unwrap(), 42usize).unwrap();
        let base: *const u8 = mat.as_ptr().as_ptr();

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
        let mat = Mat::new(Standard::<usize>::new(2, 3).unwrap(), Defaulted).unwrap();
        let base: *const u8 = mat.as_ptr().as_ptr();

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
                let repr = Standard::<usize>::new(*nrows, *ncols).unwrap();
                let ctx = &lazy_format!("nrows = {}, ncols = {}", nrows, ncols);

                // Populate the matrix using `&mut Mat`
                {
                    let ctx = &lazy_format!("{ctx} - direct");
                    let mut mat = Mat::new(repr, Defaulted).unwrap();

                    assert_eq!(mat.num_vectors(), *nrows);
                    assert_eq!(mat.vector_dim(), *ncols);

                    fill_mat(&mut mat, repr);

                    check_mat(&mat, repr, ctx);
                    check_mat_ref(mat.reborrow(), repr, ctx);
                    check_mat_mut(mat.reborrow_mut(), repr, ctx);
                    check_rows(mat.rows(), repr, ctx);
                }

                // Populate the matrix using `MatMut`
                {
                    let ctx = &lazy_format!("{ctx} - matmut");
                    let mut mat = Mat::new(repr, Defaulted).unwrap();
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
                    let mut mat = Mat::new(repr, Defaulted).unwrap();
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
                let repr = Standard::<usize>::new(*nrows, *ncols);
                let ctx = &lazy_format!("nrows = {}, ncols = {}", nrows, ncols);

                let mut mat = Mat::new(repr, Defaulted).unwrap();
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
                    if repr.num_elements().unwrap_or(0) > 0 {
                        assert_ne!(mat.as_ptr(), cloned.as_ptr());
                    }
                }

                // Clone via MatRef::to_owned
                {
                    let ctx = &lazy_format!("{ctx} - MatRef::to_owned");
                    let owned = mat.as_view().to_owned();

                    check_mat(&owned, repr, ctx);
                    check_mat_ref(owned.reborrow(), repr, ctx);
                    check_rows(owned.rows(), repr, ctx);

                    if repr.num_elements().unwrap_or(0) > 0 {
                        assert_ne!(mat.as_ptr(), owned.as_ptr());
                    }
                }

                // Clone via MatMut::to_owned
                {
                    let ctx = &lazy_format!("{ctx} - MatMut::to_owned");
                    let owned = mat.as_view_mut().to_owned();

                    check_mat(&owned, repr, ctx);
                    check_mat_ref(owned.reborrow(), repr, ctx);
                    check_rows(owned.rows(), repr, ctx);

                    if repr.num_elements().unwrap_or(0) > 0 {
                        assert_ne!(mat.as_ptr(), owned.as_ptr());
                    }
                }
            }
        }
    }

    #[test]
    fn test_mat_refmut() {
        for nrows in ROWS {
            for ncols in COLS {
                let repr = Standard::<usize>::new(*nrows, *ncols).unwrap();
                let ctx = &lazy_format!("nrows = {}, ncols = {}", nrows, ncols);

                // Populate the matrix using `&mut Mat`
                {
                    let ctx = &lazy_format!("{ctx} - by matmut");
                    let mut b: Box<[_]> = (0..repr.num_elements()).map(|_| 0usize).collect();
                    let mut matmut = MatMut::new(repr, &mut b).unwrap();

                    fill_mat_mut(matmut.reborrow_mut(), repr);

                    check_mat_mut(matmut.reborrow_mut(), repr, ctx);
                    check_mat_ref(matmut.reborrow(), repr, ctx);
                    check_rows(matmut.rows(), repr, ctx);
                    check_rows(matmut.reborrow().rows(), repr, ctx);

                    let matref = MatRef::new(repr, &b).unwrap();
                    check_mat_ref(matref, repr, ctx);
                    check_mat_ref(matref.reborrow(), repr, ctx);
                    check_rows(matref.rows(), repr, ctx);
                }

                // Populate the matrix using `RowsMut`
                {
                    let ctx = &lazy_format!("{ctx} - by rows");
                    let mut b: Box<[_]> = (0..repr.num_elements()).map(|_| 0usize).collect();
                    let mut matmut = MatMut::new(repr, &mut b).unwrap();

                    fill_rows_mut(matmut.rows_mut(), repr);

                    check_mat_mut(matmut.reborrow_mut(), repr, ctx);
                    check_mat_ref(matmut.reborrow(), repr, ctx);
                    check_rows(matmut.rows(), repr, ctx);
                    check_rows(matmut.reborrow().rows(), repr, ctx);

                    let matref = MatRef::new(repr, &b).unwrap();
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
                let m = Mat::new(Standard::new(nrows, ncols).unwrap(), 1usize).unwrap();
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
    fn matref_new_slice_length_error() {
        let repr = Standard::<u32>::new(3, 4).unwrap();

        // Correct length succeeds
        let data = vec![0u32; 12];
        assert!(MatRef::new(repr, &data).is_ok());

        // Too short fails
        let short = vec![0u32; 11];
        assert!(matches!(
            MatRef::new(repr, &short),
            Err(SliceError::LengthMismatch {
                expected: 12,
                found: 11
            })
        ));

        // Too long fails
        let long = vec![0u32; 13];
        assert!(matches!(
            MatRef::new(repr, &long),
            Err(SliceError::LengthMismatch {
                expected: 12,
                found: 13
            })
        ));
    }

    #[test]
    fn matmut_new_slice_length_error() {
        let repr = Standard::<u32>::new(3, 4).unwrap();

        // Correct length succeeds
        let mut data = vec![0u32; 12];
        assert!(MatMut::new(repr, &mut data).is_ok());

        // Too short fails
        let mut short = vec![0u32; 11];
        assert!(matches!(
            MatMut::new(repr, &mut short),
            Err(SliceError::LengthMismatch {
                expected: 12,
                found: 11
            })
        ));

        // Too long fails
        let mut long = vec![0u32; 13];
        assert!(matches!(
            MatMut::new(repr, &mut long),
            Err(SliceError::LengthMismatch {
                expected: 12,
                found: 13
            })
        ));
    }
}
