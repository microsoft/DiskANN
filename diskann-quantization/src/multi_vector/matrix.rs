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

use diskann_utils::{Reborrow, ReborrowMut};
use thiserror::Error;

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

    /// Returns the memory layout for an allocation containing [`Repr::nrows`] vectors.
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
#[derive(Debug, Clone, Copy)]
pub struct Defaulted;

/// Create a new [`Mat`] cloned from a view.
pub trait NewCloned: ReprOwned {
    /// Clone the contents behind `v`, returning a new owning [`Mat`].
    ///
    /// Implementations should ensure the returned [`Mat`] is "semantically the same" as `v`.
    fn new_cloned(v: MatRef<'_, Self>) -> Mat<Self>;
}

/// Error for [`Standard::new`].
#[derive(Debug, Clone, Copy)]
pub struct Overflow {
    nrows: usize,
    ncols: usize,
    elsize: usize,
}

impl Overflow {
    /// Construct an `Overflow` error for the given dimensions and element type.
    pub(crate) fn for_type<T>(nrows: usize, ncols: usize) -> Self {
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
    pub(crate) fn check_byte_budget<T>(
        capacity: usize,
        nrows: usize,
        ncols: usize,
    ) -> Result<(), Self> {
        let bytes = std::mem::size_of::<T>().saturating_mul(capacity);
        if bytes <= isize::MAX as usize {
            Ok(())
        } else {
            Err(Self::for_type::<T>(nrows, ncols))
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
        Self {
            ptr,
            repr,
            _invariant: PhantomData,
        }
    }

    /// Return a mutable base pointer for the [`Mat`].
    pub(crate) fn as_raw_mut_ptr(&mut self) -> *mut u8 {
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

    /// Returns an iterator over mutable row references.
    pub fn rows_mut(&mut self) -> RowsMut<'_, T> {
        RowsMut::new(self.reborrow_mut())
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

    /// Return a mutable base pointer for the [`MatMut`].
    pub(crate) fn as_raw_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
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

    /// `MatMut` is covariant in `'a`: a longer borrow can shorten.
    fn _assert_matmut_covariant_lifetime<'long: 'short, 'short, T: ReprMut>(
        v: MatMut<'long, T>,
    ) -> MatMut<'short, T> {
        v
    }
}
