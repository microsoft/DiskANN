/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, ptr::NonNull};

pub use sealed::{AsMutPtr, AsPtr, Precursor};

////////////////////
// Pointer Traits //
////////////////////

/// A constant pointer with an associated lifetime.
///
/// The only safe way to construct this type is through implementations
/// of the `crate::bits::slice::sealed::Precursor` trait or by reborrowing a bitslice.
#[derive(Debug, Clone, Copy)]
pub struct SlicePtr<'a, T> {
    ptr: NonNull<T>,
    lifetime: PhantomData<&'a T>,
}

impl<T> SlicePtr<'_, T> {
    /// # Safety
    ///
    /// It's the callers responsibility to ensure the correct lifetime is attached to
    /// the underlying pointer and that the constraints of any unsafe traits implemented by
    /// this type are upheld.
    pub(super) unsafe fn new_unchecked(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            lifetime: PhantomData,
        }
    }
}

// SAFETY: Slices are `Send` when `T: Send`.
unsafe impl<T: Send> Send for SlicePtr<'_, T> {}

// SAFETY: Slices are `Sync` when `T: Sync`.
unsafe impl<T: Sync> Sync for SlicePtr<'_, T> {}

/// A mutable pointer with an associated lifetime.
///
/// The only safe way to construct this type is through implementations
/// of the `crate::bits::slice::sealed::Precursor` trait or by reborrowing a bitslice.
#[derive(Debug)]
pub struct MutSlicePtr<'a, T> {
    ptr: NonNull<T>,
    lifetime: PhantomData<&'a mut T>,
}

impl<T> MutSlicePtr<'_, T> {
    /// # Safety
    ///
    /// It's the callers responsibility to ensure the correct lifetime is attached to
    /// the underlying pointer and that the constraints of any unsafe traits implemented by
    /// this type are upheld.
    pub(super) unsafe fn new_unchecked(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            lifetime: PhantomData,
        }
    }
}

// SAFETY: Mutable slices are `Send` when `T: Send`.
unsafe impl<T: Send> Send for MutSlicePtr<'_, T> {}

// SAFETY: Mutable slices are `Sync` when `T: Sync`.
unsafe impl<T: Sync> Sync for MutSlicePtr<'_, T> {}

mod sealed {
    use std::{marker::PhantomData, ptr::NonNull};

    use super::{MutSlicePtr, SlicePtr};
    use crate::alloc::{AllocatorCore, Poly};

    /// A precursor for a pointer type that is used as the base of a `BitSlice`.
    ///
    /// This trait is *unsafe* because implementing it incorrectly will lead to lifetime
    /// violations, out-of-bounds accesses, and other undefined behavior.
    ///
    /// # Safety
    ///
    /// There are two components to a safe implementation of `Precursor`.
    ///
    /// ## Memory Safety
    ///
    /// It is the implementors responsibility to ensure all the preconditions necessary so
    /// the following expression is safe:
    /// ```ignore
    /// let x: impl Precursor ...
    /// let len = x.len();
    /// // The length and derived pointer **must** maintain the following:
    /// unsafe { std::slice::from_raw_parts(x.precursor_into().as_ptr(), len) }
    /// ```
    /// Furthermore, if `Target: AsMutPtr`, then the preconditions for
    /// `std::slice::from_raw_parts_mut` must be upheld.
    ///
    /// ## Lifetime Safety
    ///
    /// If is the implementors responsibility that `Target` correctly captures lifetimes
    /// so the slice obtained from the precursor obeys Rust's rules for references.
    pub unsafe trait Precursor<Target>
    where
        Target: AsPtr,
    {
        /// Consume `self` and convert into `Target`.
        fn precursor_into(self) -> Target;

        /// Return the number of elements of type `<Target as AsPtr>::Type` that are valid
        /// from the pointer returned by `self.precursor_into().as_ptr()`.
        fn precursor_len(&self) -> usize;
    }

    /// Safety: Slices implicitly fit the pre-conditions for `std::slice::from_raw_parts`,
    /// meaning in particular they will not return null pointers, even with zero size.
    ///
    /// This implementation simply breaks the slice into its raw parts.
    ///
    /// The `SlicePtr` captures the reference lifetime.
    unsafe impl<'a, T> Precursor<SlicePtr<'a, T>> for &'a [T] {
        fn precursor_into(self) -> SlicePtr<'a, T> {
            SlicePtr {
                // Safety: Slices cannot yield null pointers.
                //
                // The `cast_mut()` is safe because we do not provide a way to retrieve a
                // mutable pointer from `SlicePtr`.
                ptr: unsafe { NonNull::new_unchecked(self.as_ptr().cast_mut()) },
                lifetime: PhantomData,
            }
        }
        fn precursor_len(&self) -> usize {
            <[T]>::len(self)
        }
    }

    /// Safety: Slices implicitly fit the pre-conditions for `std::slice::from_raw_parts`,
    /// meaning in particular they will not return null pointers, even with zero size.
    ///
    /// This implementation simply breaks the slice into its raw parts.
    ///
    /// The `SlicePtr` captures the reference lifetime. Decaying a mutable borrow to a
    /// normal borrow is allowed.
    unsafe impl<'a, T> Precursor<SlicePtr<'a, T>> for &'a mut [T] {
        fn precursor_into(self) -> SlicePtr<'a, T> {
            SlicePtr {
                // Safety: Slices cannot yield null pointers.
                //
                // The `cast_mut()` is safe because we do not provide a way to retrieve a
                // mutable pointer from `SlicePtr`.
                ptr: unsafe { NonNull::new_unchecked(self.as_ptr().cast_mut()) },
                lifetime: PhantomData,
            }
        }
        fn precursor_len(&self) -> usize {
            self.len()
        }
    }

    /// Safety: Slices implicitly fit the pre-conditions for `std::slice::from_raw_parts`
    /// and `std::slice::from_raw_parts_mut` meaning in particular they will not return
    /// null pointers, even with zero size.
    ///
    /// This implementation simply breaks the slice into its raw parts.
    ///
    /// The `SlicePtr` captures the mutable reference lifetime.
    unsafe impl<'a, T> Precursor<MutSlicePtr<'a, T>> for &'a mut [T] {
        fn precursor_into(self) -> MutSlicePtr<'a, T> {
            MutSlicePtr {
                // Safety: Slices cannot yield null pointers.
                ptr: unsafe { NonNull::new_unchecked(self.as_mut_ptr()) },
                lifetime: PhantomData,
            }
        }
        fn precursor_len(&self) -> usize {
            self.len()
        }
    }

    /// Safety: This implementation forwards through the [`Poly`], whose implementation of
    /// `AsPtr` and `AsPtrMut` follow the same logic as those of captured slices.
    ///
    /// Since a [`Poly`] owns its contents, the lifetime requirements are satisfied.
    unsafe impl<T, A> Precursor<Poly<[T], A>> for Poly<[T], A>
    where
        A: AllocatorCore,
    {
        fn precursor_into(self) -> Poly<[T], A> {
            self
        }
        fn precursor_len(&self) -> usize {
            self.len()
        }
    }

    /// Obtain a constant base pointer for a slice of data.
    ///
    /// # Safety
    ///
    /// The returned pointer must never be null!
    pub unsafe trait AsPtr {
        type Type;
        fn as_ptr(&self) -> *const Self::Type;
    }

    /// Obtain a mutable base pointer for a slice of data.
    ///
    /// # Safety
    ///
    /// The returned pointer must never be null! Furthermore, the mutable pointer must
    /// originally be derived from a mutable pointer.
    pub unsafe trait AsMutPtr: AsPtr {
        fn as_mut_ptr(&mut self) -> *mut Self::Type;
    }

    /// Safety: SlicePtr may only contain non-null pointers.
    unsafe impl<T> AsPtr for SlicePtr<'_, T> {
        type Type = T;
        fn as_ptr(&self) -> *const T {
            self.ptr.as_ptr().cast_const()
        }
    }

    /// Safety: SlicePtr may only contain non-null pointers.
    unsafe impl<T> AsPtr for MutSlicePtr<'_, T> {
        type Type = T;
        fn as_ptr(&self) -> *const T {
            // The const-cast is allowed by variance.
            self.ptr.as_ptr().cast_const()
        }
    }

    /// Safety: SlicePtr may only contain non-null pointers. The only way to construct
    /// a `MutSlicePtr` is from a mutable reference, so the underlying pointer is indeed
    /// mutable.
    unsafe impl<T> AsMutPtr for MutSlicePtr<'_, T> {
        fn as_mut_ptr(&mut self) -> *mut T {
            self.ptr.as_ptr()
        }
    }

    /// Safety: Slices never return a null pointer.
    unsafe impl<T, A> AsPtr for Poly<[T], A>
    where
        A: AllocatorCore,
    {
        type Type = T;
        fn as_ptr(&self) -> *const T {
            <[T]>::as_ptr(self)
        }
    }

    /// Safety: Slices never return a null pointer. A mutable reference to `self` signals
    /// an exclusive borrow - so the underlying pointer is indeed mutable.
    unsafe impl<T, A> AsMutPtr for Poly<[T], A>
    where
        A: AllocatorCore,
    {
        fn as_mut_ptr(&mut self) -> *mut T {
            <[T]>::as_mut_ptr(&mut *self)
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{GlobalAllocator, Poly};

    ///////////////////////////////
    // SlicePtr from Const Slice //
    ///////////////////////////////

    fn slice_ptr_from_const_slice(base: &[u8]) {
        let ptr = base.as_ptr();
        assert!(!ptr.is_null(), "slices must not return null pointers");

        let slice_ptr: SlicePtr<'_, u8> = base.precursor_into();
        assert_eq!(slice_ptr.as_ptr(), ptr);
        assert_eq!(base.precursor_len(), base.len());

        // Safety: Check with Miri.
        let derived = unsafe {
            std::slice::from_raw_parts(base.precursor_into().as_ptr(), base.precursor_len())
        };
        assert_eq!(derived.as_ptr(), ptr);
        assert_eq!(derived.len(), base.len());
    }

    #[test]
    fn test_slice_ptr_from_const_slice() {
        slice_ptr_from_const_slice(&[]);
        slice_ptr_from_const_slice(&[1]);
        slice_ptr_from_const_slice(&[1, 2]);

        for len in 0..10 {
            let base: Vec<u8> = vec![0; len];
            slice_ptr_from_const_slice(&base);
        }
    }

    /////////////////////////////
    // SlicePtr from Mut Slice //
    /////////////////////////////

    fn slice_ptr_from_mut_slice(base: &mut [u8]) {
        let ptr = base.as_ptr();
        let len = base.len();
        assert!(!ptr.is_null(), "slices must not return null pointers");

        let precursor_len = <&mut [u8] as Precursor<SlicePtr<'_, u8>>>::precursor_len(&base);
        assert_eq!(precursor_len, base.len());

        let slice_ptr: SlicePtr<'_, u8> = base.precursor_into();
        assert_eq!(slice_ptr.as_ptr(), ptr);

        // Safety: Check with Miri.
        let derived = unsafe { std::slice::from_raw_parts(slice_ptr.as_ptr(), precursor_len) };

        assert_eq!(derived.as_ptr(), ptr);
        assert_eq!(derived.len(), len);
    }

    #[test]
    fn test_slice_ptr_from_mut_slice() {
        slice_ptr_from_mut_slice(&mut []);
        slice_ptr_from_mut_slice(&mut [1]);
        slice_ptr_from_mut_slice(&mut [1, 2]);

        for len in 0..10 {
            let mut base: Vec<u8> = vec![0; len];
            slice_ptr_from_mut_slice(&mut base);
        }
    }

    /////////////////////////////
    // SlicePtr from Mut Slice //
    /////////////////////////////

    fn mut_slice_ptr_from_mut_slice(base: &mut [u8]) {
        let ptr = base.as_mut_ptr();
        let len = base.len();

        assert!(!ptr.is_null(), "slices must not return null pointers");

        let precursor_len = <&mut [u8] as Precursor<SlicePtr<'_, u8>>>::precursor_len(&base);
        assert_eq!(precursor_len, base.len());

        let mut slice_ptr: MutSlicePtr<'_, u8> = base.precursor_into();
        assert_eq!(slice_ptr.as_ptr(), ptr.cast_const());
        assert_eq!(slice_ptr.as_mut_ptr(), ptr);

        // Safety: Check with Miri.
        let derived =
            unsafe { std::slice::from_raw_parts_mut(slice_ptr.as_mut_ptr(), precursor_len) };

        assert_eq!(derived.as_ptr(), ptr);
        assert_eq!(derived.len(), len);
    }

    #[test]
    fn test_mut_slice_ptr_from_mut_slice() {
        mut_slice_ptr_from_mut_slice(&mut []);
        mut_slice_ptr_from_mut_slice(&mut [1]);
        mut_slice_ptr_from_mut_slice(&mut [1, 2]);

        for len in 0..10 {
            let mut base: Vec<u8> = vec![0; len];
            mut_slice_ptr_from_mut_slice(&mut base);
        }
    }

    /////////
    // Box //
    /////////

    fn box_from_box(base: Poly<[u8], GlobalAllocator>) {
        let ptr = base.as_ptr();
        let len = base.len();

        assert!(!ptr.is_null(), "slices must not return null pointers");

        assert_eq!(base.precursor_len(), len);
        let mut derived = base.precursor_into();

        assert_eq!(derived.as_ptr(), ptr);
        assert_eq!(derived.as_mut_ptr(), ptr.cast_mut());
        assert_eq!(derived.len(), len);
    }

    #[test]
    fn test_box() {
        box_from_box(Poly::from_iter([].into_iter(), GlobalAllocator).unwrap());
        box_from_box(Poly::from_iter([1].into_iter(), GlobalAllocator).unwrap());
        box_from_box(Poly::from_iter([1, 2].into_iter(), GlobalAllocator).unwrap());

        for len in 0..10 {
            let base = Poly::broadcast(0, len, GlobalAllocator).unwrap();
            box_from_box(base);
        }
    }
}
