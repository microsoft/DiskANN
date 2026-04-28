/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::marker::PhantomData;

/// A minimally functional span over a potentially unaligned slice of elements of type `T`.
///
/// Like `&[T]`, this type guarantees that the memory
/// `[self.as_ptr(), self.as_ptr().add(self.len()))` is valid for reads.
///
/// However, unlike `&[T]`, the pointer [`Self::as_ptr`] is **not** guaranteed to be aligned
/// to `std::mem::align_of::<T>()`.
///
/// If the type `T` is [`Copy`], then [`std::ptr::read_unaligned`] can be used on the valid
/// memory region for this slice to access values of type `T`.
#[derive(Debug)]
pub struct UnalignedSlice<'a, T> {
    ptr: *const T,
    len: usize,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T> UnalignedSlice<'a, T> {
    /// Construct a new [`UnalignedSlice`] over the region `[ptr, ptr.add(len))`.
    ///
    /// # Safety
    ///
    /// The provided memory region must be valid for reading via [`std::ptr::read_unaligned`]
    /// for the duration of the associated lifetime.
    ///
    /// Note that `ptr` need not be aligned to `std::mem::align_of::<T>()`.
    ///
    /// Argument `ptr` may be null if `len == 0`.
    pub const unsafe fn new(ptr: *const T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _lifetime: PhantomData,
        }
    }

    /// Return the number of elements of type `T` available for reading from [`Self::as_ptr`].
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return `true` only if the associated slice is empty.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the base pointer of the slice.
    ///
    /// **NOTE**: It is **not** guaranteed that the returned pointer is aligned!
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<T> Clone for UnalignedSlice<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for UnalignedSlice<'_, T> {}

impl<'a, T> From<&'a [T]> for UnalignedSlice<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        // SAFETY: Slices are inherently valid, so this construction is safe.
        unsafe { Self::new(slice.as_ptr(), slice.len()) }
    }
}

impl<'a, T, const N: usize> From<&'a [T; N]> for UnalignedSlice<'a, T> {
    fn from(slice: &'a [T; N]) -> Self {
        // SAFETY: Slices are inherently valid, so this construction is safe.
        unsafe { Self::new(slice.as_ptr(), N) }
    }
}

/// View `self` as an [`UnalignedSlice`].
pub trait AsUnaligned {
    /// The element type of the slice.
    type Element;

    /// Return an [`UnalignedSlice`] view of `self`.
    fn as_unaligned(&self) -> UnalignedSlice<'_, Self::Element>;
}

impl<T> AsUnaligned for UnalignedSlice<'_, T> {
    type Element = T;
    fn as_unaligned(&self) -> UnalignedSlice<'_, T> {
        *self
    }
}

impl<T> AsUnaligned for &[T] {
    type Element = T;
    fn as_unaligned(&self) -> UnalignedSlice<'_, T> {
        (*self).into()
    }
}

impl<T, const N: usize> AsUnaligned for &[T; N] {
    type Element = T;
    fn as_unaligned(&self) -> UnalignedSlice<'_, T> {
        (*self).into()
    }
}

impl<T, const N: usize> AsUnaligned for [T; N] {
    type Element = T;
    fn as_unaligned(&self) -> UnalignedSlice<'_, T> {
        self.into()
    }
}

/// A utility that offsets a collection of `T` by one byte to test the guarantee that
/// distance functions can operate on unaligned pointers.
///
/// # Invariant
///
/// We maintain the following invariants
/// * `self.data.as_ptr().add(1)` is always safe (i.e., `data.len() >= 1`)
/// * `data.len() == self.len * std::mem::size_of::<T>() + 1`: We can construct an
///   [`UnalignedSlice`] of length `self.len` starting from `self.data.as_ptr().add(1)`.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct Buffer<T>
where
    T: bytemuck::Pod,
{
    data: Vec<u8>,
    len: usize,
    _type: PhantomData<T>,
}

#[cfg(test)]
impl<T> Default for Buffer<T>
where
    T: bytemuck::Pod,
{
    fn default() -> Self {
        Self {
            // NOTE: Length 1 is important to maintain struct invariants.
            data: vec![0u8; 1],
            len: 0,
            _type: PhantomData,
        }
    }
}

#[cfg(test)]
impl<T> Buffer<T>
where
    T: bytemuck::Pod,
{
    pub(crate) fn new(x: &[T]) -> Self {
        let mut this = Self::default();
        this.copy(x);
        this
    }

    pub(crate) fn copy(&mut self, x: &[T]) {
        let bytes = std::mem::size_of_val(x);
        self.data.resize(bytes.checked_add(1).unwrap(), 0u8);

        // SAFETY: We maintain the invariant that `data.len() >= 1`, so `ptr::add` is valid.
        let dst = unsafe { self.data.as_mut_ptr().add(1) };

        // SAFETY: The `bytemuck::Pod` bound guarantees `Copy`, so we need not worry about
        // destructors. Further:
        //
        // * `src` is valid for `bytes` since that is how we obtained the value `bytes` in
        //   the first place.
        // * `dst` is valid for `bytes` since we just resized.
        // * Both `src` and `dst` are `u8` and are trivially aligned.
        // * Neither region can overlap because we borrow self by mutable reference -
        //   therefore `data` must be disjoint from the memory for `x`.
        unsafe {
            std::ptr::copy_nonoverlapping::<u8>(
                bytemuck::must_cast_slice::<T, u8>(x).as_ptr(),
                dst,
                bytes,
            );
        }

        self.len = x.len();
    }

    pub(crate) fn as_unaligned(&self) -> UnalignedSlice<'_, T> {
        // SAFETY: The invariants maintained by this class guarantee the validity of the
        // memory of the returned slice.
        //
        // The `bytemuck::Pod` bound means the resulting slice is useable via
        // [`ptr::read_unaligned`].
        unsafe { UnalignedSlice::new(self.data.as_ptr().add(1).cast::<T>(), self.len) }
    }
}
