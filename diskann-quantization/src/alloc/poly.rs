/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    alloc::Layout,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

#[cfg(feature = "flatbuffers")]
use super::Allocator;
use super::{AllocatorCore, AllocatorError, GlobalAllocator, TryClone};

/// An owning pointer type like `std::Box` that supports custom allocators.
///
/// # Examples
///
/// ## Creating and Mutating a Simple Type
///
/// This example demonstrates that `Poly` behaves pretty much like a `Box` with allocator
/// support for simple types.
/// ```
/// use diskann_quantization::alloc::{Poly, GlobalAllocator};
///
/// // `Poly` constructors can fail due to allocator errors, so return `Results`.
/// let mut x = Poly::new(10usize, GlobalAllocator).unwrap();
/// assert_eq!(*x, 10);
///
/// *x = 50;
/// assert_eq!(*x, 50);
/// ```
///
/// ## Creating and Mutating a Slice
///
/// The standard library trait [`FromIterator`] is not implemented for `Poly` because an
/// allocator is required for all construction operations. Instead, the inherent method
/// [`Poly::from_iter`] can be used, provided the iterator is one of the select few for
/// which [`TrustedIter`] is implemented, indicating that the length of the iterator can
/// be relied on in unsafe code.
///
/// ```
/// use diskann_quantization::alloc::{Poly, GlobalAllocator};
/// let v = vec![
///     "foo".to_string(),
///     "bar".to_string(),
///     "baz".to_string(),
/// ];
///
/// let poly = Poly::from_iter(v.into_iter(), GlobalAllocator).unwrap();
/// assert_eq!(poly.len(), 3);
/// assert_eq!(poly[0], "foo");
/// assert_eq!(poly[1], "bar");
/// assert_eq!(poly[2], "baz");
/// ```
///
/// ## Using a Custom Allocator
///
/// This crate provides a handful of custom allocators, including the [`super::BumpAllocator`].
/// It can be used to group together allocations into a single arena.
///
/// ```
/// use diskann_quantization::{
///     alloc::{Poly, BumpAllocator},
///     num::PowerOfTwo
/// };
///
/// let dim = 10;
///
/// // Estimate how many bytes are needed to create two such slices. We can control the
/// // alignment of the base pointer for the `BumpAllocator` to avoid extra memory used
/// // to satisfy alignment.
/// let alloc = BumpAllocator::new(
///     dim * (std::mem::size_of::<f64>() + std::mem::size_of::<u8>()),
///     PowerOfTwo::new(64).unwrap(),
/// ).unwrap();
///
/// let foo = Poly::<[f64], _>::from_iter(
///    (0..dim).map(|i| i as f64),
///    alloc.clone(),
/// ).unwrap();
///
/// let bar = Poly::<[u8], _>::from_iter(
///    (0..dim).map(|i| i as u8),
///    alloc.clone(),
/// ).unwrap();
///
/// // The base pointer for the allocated object in `foo` is the base pointer of the arena
/// // owned by the BumpAllocator.
/// assert_eq!(alloc.as_ptr(), Poly::as_ptr(&foo).cast::<u8>());
///
/// // The base pointer for the allocated object in `bar` is also within the arena owned
/// // by the BumpAllocator as well.
/// assert_eq!(
///     unsafe { alloc.as_ptr().add(std::mem::size_of_val(&*foo)) },
///     Poly::as_ptr(&bar).cast::<u8>(),
/// );
///
/// // The allocator is now full - so further allocations will fail.
/// assert!(Poly::new(10usize, alloc.clone()).is_err());
///
/// // If we drop the allocator, the clones inside the `Poly` containers will keep the
/// // backing memory alive.
/// std::mem::forget(alloc);
/// assert!(foo.iter().enumerate().all(|(i, v)| i as f64 == *v));
/// assert!(bar.iter().enumerate().all(|(i, v)| i as u8 == *v));
/// ```
///
/// ## Using Trait Object
///
/// `Poly` is compatible with trait objects as well using the [`crate::poly!`] macro.
/// A macro is needed because traits such as
/// [`Unsize`](https://doc.rust-lang.org/std/marker/trait.Unsize.html) and
/// [`CoerceUnsized`](https://doc.rust-lang.org/std/ops/trait.CoerceUnsized.html) are
/// unstable.
///
/// ```
/// use diskann_quantization::{
///     poly,
///     alloc::BumpAllocator,
///     num::PowerOfTwo,
/// };
/// use std::fmt::Display;
///
/// let message = "hello world";
///
/// let alloc = BumpAllocator::new(512, PowerOfTwo::new(64).unwrap()).unwrap();
///
/// // Create a new `Poly` trait object for `std::fmt::Display`. Due to limitations in the
/// // macro matching rules, identifiers are needed for the object and allocator.
/// let clone = alloc.clone();
/// let poly = poly!(std::fmt::Display, message, clone).unwrap();
/// assert_eq!(poly.to_string(), "hello world");
///
/// // Here - we demonstrate the full type of the returned `Poly`.
/// let clone = alloc.clone();
/// let _: diskann_quantization::alloc::Poly<dyn Display, _> = poly!(
///     Display,
///     message,
///     clone
/// ).unwrap();
///
/// // If additional auto traits are needed like `Send`, the brace-style syntax can be used
/// let clone = alloc.clone();
/// let poly = poly!({ std::fmt::Display + Send + Sync }, message, clone).unwrap();
///
/// // Existing `Poly` objects can be converted using the same macro.
/// let poly = diskann_quantization::alloc::Poly::new(message, alloc.clone()).unwrap();
/// let poly = poly!(std::fmt::Display, poly);
/// assert_eq!(poly.to_string(), "hello world");
/// ```
///
/// Naturally, the implementation of the trait is checked for validity.
#[derive(Debug)]
#[repr(C)]
pub struct Poly<T, A = GlobalAllocator>
where
    T: ?Sized,
    A: AllocatorCore,
{
    ptr: NonNull<T>,
    allocator: A,
}

// SAFETY: `Poly` is `Send` when the pointed-to object and allocator are `Send`.
unsafe impl<T, A> Send for Poly<T, A>
where
    T: ?Sized + Send,
    A: AllocatorCore + Send,
{
}

// SAFETY: `Poly` is `Sync` when the pointed-to object and allocator are `Sync`.
unsafe impl<T, A> Sync for Poly<T, A>
where
    T: ?Sized + Sync,
    A: AllocatorCore + Sync,
{
}

/// Error type returned from [`Poly::new_with`].
#[derive(Debug, Clone, Copy)]
pub enum CompoundError<E> {
    /// An allocator error occurred while allocating the base `Poly`.
    Allocator(AllocatorError),
    /// An error occurred while running the closure.
    Constructor(E),
}

impl<T, A> Poly<T, A>
where
    A: AllocatorCore,
{
    /// Allocate memory from `allocator` and place `value` into that location.
    pub fn new(value: T, allocator: A) -> Result<Self, AllocatorError> {
        if std::mem::size_of::<T>() == 0 {
            Ok(Self {
                ptr: NonNull::dangling(),
                allocator,
            })
        } else {
            let ptr = allocator.allocate(Layout::new::<T>())?;

            // SAFETY: On success, [`Allocator::allocate`] is required to return a suitable
            // aligned pointer to a slice of size at least `std::mem::size_of::<T>()`.
            //
            // Therefore, the cast is valid.
            //
            // The write is safe because there is no existing object at the pointed to location.
            let ptr: NonNull<T> = unsafe {
                let ptr = ptr.cast::<T>();
                ptr.as_ptr().write(value);
                ptr
            };

            Ok(Self { ptr, allocator })
        }
    }

    /// Allocate memory from `allocator` for `T`, then run the provided closure, moving
    /// the result into the allocated memory.
    ///
    /// Because this allocates the storage for the object first, it can be used in
    /// situations where the object to be allocated will use the same allocator, but should
    /// be allocated after the base.
    pub fn new_with<F, E>(f: F, allocator: A) -> Result<Self, CompoundError<E>>
    where
        F: FnOnce(A) -> Result<T, E>,
        A: Clone,
    {
        // Construct an uninitialized version of `Self` first before invoking the constructor
        // closure.
        let mut this = Self::new_uninit(allocator.clone()).map_err(CompoundError::Allocator)?;
        this.write(f(allocator).map_err(CompoundError::Constructor)?);

        // SAFETY: We wrote to the `MaybeUninit` with the valid object returned from `f`.
        Ok(unsafe { this.assume_init() })
    }

    /// Construct a new [`Poly`] with uninitialized contents in `allocator`.
    pub fn new_uninit(allocator: A) -> Result<Poly<MaybeUninit<T>, A>, AllocatorError> {
        if std::mem::size_of::<T>() == 0 {
            Ok(Poly {
                ptr: NonNull::dangling(),
                allocator,
            })
        } else {
            let ptr = allocator.allocate(Layout::new::<MaybeUninit<T>>())?;

            // SAFETY: This cast is valid because
            //
            // 1. [`Allocator::allocate]` is required to on success to return a pointer to a
            //    slice compatible with the provided layout.
            //
            // 2. This memory has not been initialized. Since `MaybeUninit` does not `Drop`
            //    its contents, it is okay to hand out.
            let ptr: NonNull<MaybeUninit<T>> = ptr.cast::<MaybeUninit<T>>();
            Ok(Poly { ptr, allocator })
        }
    }
}

impl<T, A> Poly<T, A>
where
    T: ?Sized,
    A: AllocatorCore,
{
    /// Consume `Self`, returning the wrapped pointer and allocator.
    ///
    /// This function does not trigger any drop logic nor deallocation.
    pub fn into_raw(this: Self) -> (NonNull<T>, A) {
        let ptr = this.ptr;

        // SAFETY: This creates a bit-size copy of the allocator in `this`. Since we
        // immediately forget `this`, this behaves like moving the allocator out of `this`,
        // which is safe because `this` is taken by value.
        let allocator = unsafe { std::ptr::read(&this.allocator) };
        std::mem::forget(this);
        (ptr, allocator)
    }

    /// Construct a [`Poly`] from a raw pointer and allocator.
    ///
    /// After calling this function, the returned [`Poly`] will assume ownership of the
    /// provided pointer.
    ///
    /// # Safety
    ///
    /// The value of `ptr` must have runtime alignment compatible with
    /// ```text
    /// std::mem::Layout::for_value(&*ptr.as_ptr())
    /// ```
    /// and point to a valid object of type `T`.
    ///
    /// The pointer must point to memory currently allocated in `allocator`.
    pub unsafe fn from_raw(ptr: NonNull<T>, allocator: A) -> Self {
        Poly { ptr, allocator }
    }

    /// Return a pointer to the object managed by this `Poly`.
    pub fn as_ptr(this: &Self) -> *const T {
        this.ptr.as_ptr().cast_const()
    }

    /// Return a reference to the underlying allocator.
    pub fn allocator(&self) -> &A {
        &self.allocator
    }
}

impl<T, A> Poly<MaybeUninit<T>, A>
where
    A: AllocatorCore,
{
    /// Converts to `Poly<T, A>`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the value has truly been initialized.
    pub unsafe fn assume_init(self) -> Poly<T, A> {
        let (ptr, allocator) = Poly::into_raw(self);
        // SAFETY: It's the caller's responsibility to ensure that the pointed-to value
        // has truely been initialized.
        unsafe { Poly::from_raw(ptr.cast::<T>(), allocator) }
    }
}

impl<T, A> Poly<[T], A>
where
    A: AllocatorCore,
{
    /// Construct a new `Poly` containing an uninitialized slice of length `len` with
    /// memory allocated from `allocator`.
    pub fn new_uninit_slice(
        len: usize,
        allocator: A,
    ) -> Result<Poly<[MaybeUninit<T>], A>, AllocatorError> {
        let layout = Layout::array::<T>(len).map_err(|_| AllocatorError)?;
        let ptr = if layout.size() == 0 {
            // SAFETY: We're either constructing a slice of zero sized types, or a slice
            // of length zero. In either case, `NonNull::dangling()` ensures proper
            // alignment of the non-null base pointer.
            unsafe {
                NonNull::new_unchecked(std::ptr::slice_from_raw_parts_mut(
                    NonNull::dangling().as_ptr(),
                    len,
                ))
            }
        } else {
            let ptr = allocator.allocate(layout)?;
            debug_assert_eq!(ptr.len(), layout.size());

            // SAFETY: `Allocator` is required to provide a properly sized and aligned
            // slice upon success. Wrapping the raw memory in `MaybeUninit` is okay because
            // we will not try to `Drop` values of type `T` until they've been properly
            // initialized.
            unsafe {
                NonNull::new_unchecked(std::ptr::slice_from_raw_parts_mut(
                    ptr.as_ptr().cast::<MaybeUninit<T>>(),
                    len,
                ))
            }
        };

        // SAFETY: `ptr` points to a properly initialized object allocated from `allocator`.
        Ok(unsafe { Poly::from_raw(ptr, allocator) })
    }

    /// Construct a new `Poly` from the iterator.
    pub fn from_iter<I>(iter: I, allocator: A) -> Result<Self, AllocatorError>
    where
        I: TrustedIter<Item = T>,
    {
        // A guard that drops the initialized portion of a partially constructed slice
        // in the event that `iter` panics.
        struct Guard<'a, T, A>
        where
            A: AllocatorCore,
        {
            uninit: &'a mut Poly<[MaybeUninit<T>], A>,
            initialized_to: usize,
        }

        impl<T, A> Drop for Guard<'_, T, A>
        where
            A: AllocatorCore,
        {
            fn drop(&mut self) {
                // Performance optimization: skip if `T` doesn't need to be dropped.
                //
                // Not needed for release builds since the drop loop will be optimized away,
                // but can make debug build run a little faster.
                //
                // See: https://doc.rust-lang.org/std/mem/fn.needs_drop.html
                if std::mem::needs_drop::<T>() {
                    self.uninit
                        .iter_mut()
                        .take(self.initialized_to)
                        .for_each(|u|
                            // SAFETY: `self.initialized_to` is only incremented after a
                            // successful write and therefore `u` is initialized.
                            unsafe { u.assume_init_drop() });
                }
            }
        }

        let mut uninit = Poly::<[T], A>::new_uninit_slice(iter.len(), allocator)?;

        let mut guard = Guard {
            uninit: &mut uninit,
            initialized_to: 0,
        };

        std::iter::zip(iter, guard.uninit.iter_mut()).for_each(|(src, dst)| {
            dst.write(src);
            guard.initialized_to += 1;
        });

        debug_assert_eq!(
            guard.initialized_to,
            guard.uninit.len(),
            "an incorrect number of elements was initialized",
        );

        // Forget the guard so its destructor doesn't run and there-by ruin all the good
        // work we just did.
        std::mem::forget(guard);

        // SAFETY: Since `iter` has a trusted length, we know every element in `uninit` has
        // been properly initialized.
        Ok(unsafe { uninit.assume_init() })
    }
}

impl<T, A> Poly<[MaybeUninit<T>], A>
where
    A: AllocatorCore,
{
    /// Converts to `Poly<[T], A>`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the value has truly been initialized.
    pub unsafe fn assume_init(self) -> Poly<[T], A> {
        let len = self.deref().len();
        let (ptr, allocator) = Poly::into_raw(self);

        // SAFETY: The slice cast is valid because
        //
        // 1. The caller has asserted that `self` has been initialized.
        // 2. `MaybeUninit<T>` is ABI compatible with `T`.
        //
        // The unchecked `NonNull` is valid because `self.ptr` is `NonNull`.
        let ptr = unsafe {
            NonNull::new_unchecked(std::ptr::slice_from_raw_parts_mut(
                ptr.as_ptr().cast::<T>(),
                len,
            ))
        };

        // SAFETY: The memory layout and pointed-to contents of of `[T]` is exactly the
        // same as `[MaybeUninit<T>]`. So it is acceptable to deallocate this derived `[T]`
        // from `allocator`.
        //
        // Additionally, the caller has asserted that the pointed-to slice is valid.
        unsafe { Poly::<[T], A>::from_raw(ptr, allocator) }
    }
}

impl<T, A> Poly<[T], A>
where
    A: AllocatorCore,
    T: Clone,
{
    /// Construct a new `Poly` slice with each entry initialized to `value`.
    pub fn broadcast(value: T, len: usize, allocator: A) -> Result<Self, AllocatorError> {
        Self::from_iter((0..len).map(|_| value.clone()), allocator)
    }
}

impl<T, A> Drop for Poly<T, A>
where
    T: ?Sized,
    A: AllocatorCore,
{
    fn drop(&mut self) {
        // SAFETY: Because `self` hasn't been dropped quite yet, the pointed to object is
        // still valid.
        let layout = Layout::for_value(unsafe { self.ptr.as_ref() });

        // SAFETY: `Poly` owns the pointed-to object, so we can drop it when the `Poly`
        // is dropped.
        unsafe { std::ptr::drop_in_place(self.ptr.as_ptr()) };

        if layout.size() != 0 {
            // This cast is safe because `u8`'s alignment requirements are equal to or less
            // than `T`. Additionally, the layout was derived from the pointed to object.
            let as_slice =
                std::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr().cast::<u8>(), layout.size());

            // SAFETY: `self.ptr` was non-null.
            let ptr = unsafe { NonNull::new_unchecked(as_slice) };

            // SAFETY: The construction of `Poly` means that the pointer is always passed
            // around with its corresponding allocator.
            unsafe { self.allocator.deallocate(ptr, layout) }
        }
    }
}

impl<T, A> Deref for Poly<T, A>
where
    T: ?Sized,
    A: AllocatorCore,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        // SAFETY: As long as `Self` is alive, the pointed-to object is valid.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T, A> DerefMut for Poly<T, A>
where
    T: ?Sized,
    A: AllocatorCore,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: As long as `Self` is alive, the pointed-to object is valid.
        //
        // Since there is a mutable reference to `Self`, the access to the pointed-to
        // object is exclusive, so it is safe to return a mutable reference.
        unsafe { self.ptr.as_mut() }
    }
}

///////////////
// From Iter //
///////////////

/// A local marker type for iterators with a trusted length.
///
/// # Safety
///
/// Implementation must ensure that the implementation of `ExactSizeIterator` is such that
/// that unsafe code can rely on the returned value.
pub unsafe trait TrustedIter: ExactSizeIterator {}

//---------------//
// Raw Iterators //
//---------------//

// SAFETY: `std::slice` is trusted.
unsafe impl<T> TrustedIter for std::slice::Iter<'_, T> {}
// SAFETY: `std::vec` is trusted.
unsafe impl<T> TrustedIter for std::vec::IntoIter<T> {}
// SAFETY: `std::ops::Range` is trusted.
unsafe impl TrustedIter for std::ops::Range<usize> {}
// SAFETY: `std::array::IntoIter` is trusted.
unsafe impl<T, const N: usize> TrustedIter for std::array::IntoIter<T, N> {}
// SAFETY: `rand::seq::index::IndexVecIntoIter` is trusted.
unsafe impl TrustedIter for rand::seq::index::IndexVecIntoIter {}

#[cfg(feature = "flatbuffers")]
// SAFETY: We trust the implementors of `flatbuffer` return trustable lengths for this iterator.
unsafe impl<'a, T> TrustedIter for flatbuffers::VectorIter<'a, T> where T: flatbuffers::Follow<'a> {}

//---------------//
// Map Iterators //
//---------------//

// SAFETY: Maps of trusted iterators are trusted.
unsafe impl<I, U, F> TrustedIter for std::iter::Map<I, F>
where
    I: TrustedIter,
    F: FnMut(I::Item) -> U,
{
}

// SAFETY: Enumerates of trusted iterators are trusted.
unsafe impl<I> TrustedIter for std::iter::Enumerate<I> where I: TrustedIter {}

// SAFETY: Clones of trusted iterators are trusted.
unsafe impl<'a, I, T> TrustedIter for std::iter::Cloned<I>
where
    I: TrustedIter<Item = &'a T>,
    T: 'a + Clone,
{
}

// SAFETY: Copies of trusted iterators are trusted.
unsafe impl<'a, I, T> TrustedIter for std::iter::Copied<I>
where
    I: TrustedIter<Item = &'a T>,
    T: 'a + Copy,
{
}

// SAFETY: Zip of trusted iterators is trusted.
unsafe impl<T, U> TrustedIter for std::iter::Zip<T, U>
where
    T: TrustedIter,
    U: TrustedIter,
{
}

//////////////////
// Trait Object //
//////////////////

#[macro_export]
macro_rules! poly {
    // Creating a new poly types.
    ({ $($traits:tt)+ }, $v:ident, $alloc:ident) => {{
        $crate::alloc::Poly::new($v, $alloc).map(|poly| {
            $crate::alloc::poly!({ $($traits)+ }, poly)
        })
    }};
    ($trait:path, $v:ident, $alloc:ident) => {{
        $crate::alloc::poly!({ $trait }, $v, $alloc)
    }};

    // Coercing an existing `poly`.
    ({ $($traits:tt)+ }, $poly:ident) => {{
        let (ptr, alloc) = $crate::alloc::Poly::into_raw($poly);

        // The deduction chain goes that we need to coerce the pointer from `*const T` for
        // some concrete type `T` to `*const dyn $traits...`.
        //
        // Putting the dyn trait in the turbo-fish for the `Poly::from_raw` forces the
        // corresponding pointer argument to be the dynamic pointer.
        //
        // As such, Rust will check to see if Unsized coercion applies. If not, we get
        // a compilation error.
        //
        // SAFETY: The unsafe part is the call to `from_raw`, which is safe because we just
        // obtained `ptr` from `into_raw` and the pointed-to object is still the same, so
        // may be safely deallocated with `alloc`.
        unsafe { $crate::alloc::Poly::<dyn $($traits)*, _>::from_raw(ptr, alloc) }
    }};
    ($trait:path, $poly:ident) => {{
        $crate::alloc::poly!({ $trait }, $poly)
    }};

    // Literal array constructor.
    ([$($x:expr),* $(,)?], $alloc:ident) => {{
        Poly::from_iter([$($x,)*].into_iter(), $alloc)
    }}
}

pub use poly;

///////////////
// Try Clone //
///////////////

impl<T, A> TryClone for Poly<T, A>
where
    T: Clone,
    A: AllocatorCore + Clone,
{
    fn try_clone(&self) -> Result<Self, AllocatorError> {
        let clone = (*self).clone();
        Poly::new(clone, self.allocator().clone())
    }
}

impl<T, A> TryClone for Poly<[T], A>
where
    T: Clone,
    A: AllocatorCore + Clone,
{
    fn try_clone(&self) -> Result<Self, AllocatorError> {
        Poly::from_iter(self.iter().cloned(), self.allocator().clone())
    }
}

impl<T, A> TryClone for Option<Poly<T, A>>
where
    T: ?Sized,
    A: AllocatorCore,
    Poly<T, A>: super::TryClone,
{
    fn try_clone(&self) -> Result<Self, AllocatorError> {
        Ok(match self {
            Some(v) => Some(v.try_clone()?),
            None => None,
        })
    }
}

////////////////////////////////
// Recursively Defined Traits //
////////////////////////////////

impl<T, A> PartialEq for Poly<T, A>
where
    T: ?Sized + PartialEq,
    A: AllocatorCore,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

////////////////
// Conversion //
////////////////

impl<T> From<Box<[T]>> for Poly<[T], GlobalAllocator> {
    fn from(value: Box<[T]>) -> Self {
        // SAFETY: The underlying pointer for `Box` is always `NonNull`, and
        // `GlobalAllocator` is the same as the global allocator used by `std::Box`.
        unsafe {
            Poly::from_raw(
                NonNull::new_unchecked(Box::into_raw(value)),
                GlobalAllocator,
            )
        }
    }
}

////////////////////////
// Flatbuffer Support //
////////////////////////

#[cfg(feature = "flatbuffers")]
// SAFETY: We correctly report the length of the buffer and allocate downwards.
//
// Also - clippy doesn't work if the safety comment comes before the `cfg`.
unsafe impl<A> flatbuffers::Allocator for Poly<[u8], A>
where
    A: Allocator,
{
    type Error = AllocatorError;

    // Double the size of `self` and move the current results to the end of the new buffer.
    fn grow_downwards(&mut self) -> Result<(), Self::Error> {
        // `self.len()` is constrained to be less than `isize::MAX`, so we can double it and
        // still fit within `usize`.
        let next_len = (2 * self.len()).max(1);
        let mut next = Poly::broadcast(0u8, next_len, self.allocator().clone())?;
        next[next_len - self.len()..].copy_from_slice(self);
        *self = next;
        Ok(())
    }

    fn len(&self) -> usize {
        (**self).len()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::AlwaysFails;

    struct HasHoles {
        s: String,
        a: u32,
        b: u8,
    }

    impl HasHoles {
        fn new(s: String, a: u32, b: u8) -> Self {
            Self { s, a, b }
        }
    }

    fn assert_is_send<T>(_: &T)
    where
        T: Send,
    {
    }

    //-------//
    // Sizes //
    //-------//

    #[test]
    fn size_check() {
        assert_eq!(std::mem::size_of::<Poly<usize>>(), 8);
        assert_eq!(std::mem::size_of::<Option<Poly<usize>>>(), 8);
    }

    //-------//
    // Basic //
    //-------//

    #[test]
    fn basic_test_copy() {
        let x = 10usize;
        let poly = Poly::new(x, GlobalAllocator).unwrap();
        assert_eq!(*poly, 10);
    }

    #[test]
    fn basic_test_borrow() {
        let x = &10usize;
        let poly = Poly::<&usize>::new(x, GlobalAllocator).unwrap();
        assert_eq!(**poly, 10);
    }

    #[test]
    fn test_with_drop() {
        let poly = Poly::<String>::new("hello world".to_string(), GlobalAllocator).unwrap();
        assert_eq!(&**poly, "hello world");
    }

    #[test]
    fn test_mutate() {
        let mut poly = Poly::<String>::new("foo".to_string(), GlobalAllocator).unwrap();
        assert_eq!(&**poly, "foo");
        *poly = "bar".to_string();
        assert_eq!(&**poly, "bar");
    }

    //------------------//
    // Zero Sized Types //
    //------------------//

    #[test]
    fn zero_sized() {
        let _ = Poly::<()>::new((), GlobalAllocator).unwrap();
    }

    #[test]
    fn zero_sized_raw() {
        let x = Poly::<()>::new((), GlobalAllocator).unwrap();
        let (ptr, alloc) = Poly::into_raw(x);
        // SAFETY: `ptr` and `alloc` were obtained from `into_raw`.
        let _ = unsafe { Poly::from_raw(ptr, alloc) };
    }

    #[test]
    fn zero_sized_uninit() {
        let _ = Poly::<()>::new_uninit(GlobalAllocator).unwrap();
    }

    #[test]
    fn zero_sized_uninit_to_init() {
        let x = Poly::<()>::new_uninit(GlobalAllocator).unwrap();
        // SAFETY: No initialization is required for zero sized types.
        let _ = unsafe { x.assume_init() };
    }

    #[test]
    fn zero_sized_slice() {
        let x = Poly::<[()]>::from_iter((0..0).map(|_| ()), GlobalAllocator).unwrap();
        assert!(x.is_empty());

        let x = Poly::<[()]>::from_iter((0..10).map(|_| ()), GlobalAllocator).unwrap();
        assert_eq!(x.len(), 10);

        let x = Poly::<[usize]>::from_iter(0..0, GlobalAllocator).unwrap();
        assert!(x.is_empty());

        let x =
            Poly::<[String]>::from_iter((0..0).map(|i| i.to_string()), GlobalAllocator).unwrap();
        assert!(x.is_empty());
    }

    //--------//
    // Uninit //
    //--------//

    #[test]
    fn dropping_uninit_is_okay() {
        // `String` has a non-trivial `Drop` implementation.
        //
        // If we return an uninitialized `Poly<String>` and do not initialize the
        // contents, dropping the `Poly<MaybeUninit<String>>` should not trigger undefined
        // behavior.
        let _ = Poly::<HasHoles>::new_uninit(GlobalAllocator).unwrap();
    }

    #[test]
    fn test_assume_init() {
        let mut poly = Poly::<HasHoles>::new_uninit(GlobalAllocator).unwrap();
        poly.write(HasHoles::new("hello world".into(), 10, 5));
        // SAFETY: We just initialized `poly`.
        let poly: Poly<HasHoles> = unsafe { poly.assume_init() };
        assert_eq!(poly.s, "hello world");
        assert_eq!(poly.a, 10);
        assert_eq!(poly.b, 5);
    }

    #[test]
    fn test_assume_init_slice_copy() {
        let mut poly = Poly::<[usize]>::new_uninit_slice(10, GlobalAllocator).unwrap();
        assert_eq!(poly.len(), 10);
        for (i, v) in poly.iter_mut().enumerate() {
            v.write(i);
        }
        // SAFETY: We just initialized `poly`.
        let poly: Poly<[usize]> = unsafe { poly.assume_init() };

        for (i, v) in poly.iter().enumerate() {
            assert_eq!(*v, i);
        }
    }

    #[test]
    fn test_assume_init_slice_drop() {
        let mut poly = Poly::<[HasHoles]>::new_uninit_slice(10, GlobalAllocator).unwrap();
        assert_eq!(poly.len(), 10);
        for (i, v) in poly.iter_mut().enumerate() {
            v.write(HasHoles::new(
                i.to_string(),
                i.try_into().unwrap(),
                i.try_into().unwrap(),
            ));
        }
        // SAFETY: We just initialized `poly`.
        let poly: Poly<[HasHoles]> = unsafe { poly.assume_init() };

        for (i, v) in poly.iter().enumerate() {
            assert_eq!(v.s, i.to_string());
            assert_eq!(v.a as usize, i);
            assert_eq!(v.b as usize, i);
        }
    }

    //-----------//
    // From Iter //
    //-----------//

    #[test]
    fn from_iter_strings() {
        let p =
            Poly::<[String], _>::from_iter((0..5).map(|i| i.to_string()), GlobalAllocator).unwrap();

        assert_eq!(&*p, &["0", "1", "2", "3", "4"])
    }

    /// Test for undefined behavior if an iterator panics on the first item. In this
    /// situation nothing should be dropped.
    ///
    /// This must be tested using Miri.
    #[test]
    #[should_panic(expected = "first")]
    fn from_iter_cleanup_first() {
        Poly::<[String], _>::from_iter((0..5).map(|_| panic!("first")), GlobalAllocator).unwrap();
    }

    /// This test induces a panic in `from_iter` in the middle of iteration to test the
    /// incremental drop logic.
    ///
    /// A non-compliant implementation will leak memory, which Miri can detect.
    #[test]
    #[should_panic(expected = "middle")]
    fn from_iter_cleanup_middle() {
        Poly::<[String], _>::from_iter(
            (0..5).map(|i| {
                if i == 3 {
                    panic!("middle");
                } else {
                    i.to_string()
                }
            }),
            GlobalAllocator,
        )
        .unwrap();
    }

    /// This test is like `from_iter_cleanup_middle` but just panics at the very end.
    ///
    /// A non-compliant implementation will leak memory, which Miri can detect.
    #[test]
    #[should_panic(expected = "last")]
    fn from_iter_cleanup_last() {
        Poly::<[String], _>::from_iter(
            (0..5).map(|i| {
                let string = i.to_string();
                if i == 4 {
                    panic!("last");
                }
                string
            }),
            GlobalAllocator,
        )
        .unwrap();
    }

    //------------------//
    // Allocator Errors //
    //------------------//

    #[test]
    fn new_error() {
        let _ = Poly::new(10usize, AlwaysFails).unwrap_err();
    }

    #[test]
    fn new_with_error() {
        let err = Poly::new_with(
            |_| -> Result<u8, std::convert::Infallible> { Ok(0) },
            AlwaysFails,
        )
        .unwrap_err();
        assert!(matches!(err, CompoundError::Allocator(_)));

        let err = Poly::new_with(
            |_| -> Result<u8, std::num::TryFromIntError> {
                let x: u8 = (1000usize).try_into()?;
                Ok(x)
            },
            GlobalAllocator,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            CompoundError::Constructor(std::num::TryFromIntError { .. })
        ));
    }

    #[test]
    fn new_uninit_error() {
        let _ = Poly::<String, _>::new_uninit(AlwaysFails).unwrap_err();
    }

    #[test]
    fn new_uninit_slice_error() {
        let _ = Poly::<[usize], _>::new_uninit_slice(10, AlwaysFails).unwrap_err();
    }

    #[test]
    fn new_from_iter_error() {
        let _ = Poly::<[usize], _>::from_iter(0..10, AlwaysFails).unwrap_err();
    }

    //---------------//
    // Trait Objects //
    //---------------//

    trait Describe {
        fn describe(&self) -> String;
        fn describe_mut(&mut self) -> String;
    }

    struct ImplsDescribe;

    impl Describe for ImplsDescribe {
        fn describe(&self) -> String {
            "describe const".to_string()
        }

        fn describe_mut(&mut self) -> String {
            "describe mut".to_string()
        }
    }

    struct AlsoImplsDescribe(String);

    impl Describe for AlsoImplsDescribe {
        fn describe(&self) -> String {
            format!("describe const: {}", self.0)
        }

        fn describe_mut(&mut self) -> String {
            format!("describe mut: {}", self.0)
        }
    }

    struct DescribeLifetime<'a>(&'a str);

    impl Describe for DescribeLifetime<'_> {
        fn describe(&self) -> String {
            format!("describe const: {}", self.0)
        }

        fn describe_mut(&mut self) -> String {
            format!("describe mut: {}", self.0)
        }
    }

    trait Foo<T> {
        fn foo(&self, v: T) -> T;
    }

    impl Foo<f32> for f32 {
        fn foo(&self, v: f32) -> f32 {
            *self + v
        }
    }

    #[test]
    fn test_dyn_trait() {
        // Traits without generic parameters
        {
            let mut poly0 = poly!(Describe, ImplsDescribe, GlobalAllocator).unwrap();

            let also = AlsoImplsDescribe("foo".to_string());
            let mut poly1 = poly!({ Describe + Send }, also, GlobalAllocator).unwrap();
            assert_is_send::<Poly<dyn Describe + Send, _>>(&poly1);

            assert_eq!(poly1.describe(), "describe const: foo");
            assert_eq!(poly1.describe_mut(), "describe mut: foo");

            assert_eq!(poly0.describe(), "describe const");
            assert_eq!(poly0.describe_mut(), "describe mut");
        }

        {
            // Transform a `Poly<T>` to `Poly<dyn T>`.
            let mut poly =
                Poly::new(AlsoImplsDescribe("bar".to_string()), GlobalAllocator).unwrap();
            assert_is_send::<Poly<AlsoImplsDescribe>>(&poly);

            assert_eq!(poly.describe(), "describe const: bar");
            assert_eq!(poly.describe_mut(), "describe mut: bar");

            let mut poly = poly!({ Describe + Send }, poly);

            assert_is_send::<Poly<dyn Describe + Send>>(&poly);

            assert_eq!(poly.describe(), "describe const: bar");
            assert_eq!(poly.describe_mut(), "describe mut: bar");
        }

        // Traits with generic parameters
        {
            let f = 1.0f32;
            let poly = poly!({ Foo<f32> }, f, GlobalAllocator).unwrap();
            assert_eq!(poly.foo(2.0), 3.0);
        }

        {
            let poly = Poly::new(1.0f32, GlobalAllocator).unwrap();
            let poly = poly!({ Foo<f32> + Send }, poly);

            assert_is_send::<Poly<dyn Foo<f32> + Send>>(&poly);

            assert_eq!(poly.foo(2.0), 3.0);
        }

        // Traits with generic parameters in a function.
        //
        // An improper implementation of the `poly!` macro won't be able to use the generic
        // parameter `T` due to the "cannot use generic parameter from outer item"
        // constraint.
        fn test<'a, T>(x: T) -> Poly<dyn Foo<T> + 'a>
        where
            T: Foo<T> + 'a,
        {
            poly!({ Foo<T> }, x, GlobalAllocator).unwrap()
        }

        {
            let x = test(1.0f32);
            assert_eq!(x.foo(2.0), 3.0);
        }
    }

    #[test]
    fn test_dyn_trait_with_lifetime() {
        let base: String = "foo".into();
        let describe = DescribeLifetime(&base);

        let mut poly: Poly<dyn Describe> = poly!({ Describe }, describe, GlobalAllocator).unwrap();
        assert_eq!(poly.describe(), "describe const: foo");
        assert_eq!(poly.describe_mut(), "describe mut: foo");
    }

    #[test]
    fn test_try_clone_item() {
        let x = Poly::<String>::new("hello".to_string(), GlobalAllocator).unwrap();
        let y = x.try_clone().unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn test_try_clone_slice() {
        let x = Poly::<[String]>::from_iter(
            ["foo".to_string(), "bar".to_string(), "baz".to_string()].into_iter(),
            GlobalAllocator,
        )
        .unwrap();

        let y = x.try_clone().unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn test_try_clone_option() {
        let mut x = Some(Poly::<String>::new("hello".to_string(), GlobalAllocator).unwrap());
        let y = x.try_clone().unwrap();
        assert_eq!(x, y);

        x = None;
        let y = x.try_clone().unwrap();
        assert_eq!(x, y);
    }

    #[cfg(feature = "flatbuffers")]
    #[test]
    fn test_grow_downwards() {
        let mut x = Poly::from_iter([1u8, 2u8, 3u8].into_iter(), GlobalAllocator).unwrap();
        <_ as flatbuffers::Allocator>::grow_downwards(&mut x).unwrap();
        assert_eq!(&*x, &[0, 0, 0, 1, 2, 3]);

        <_ as flatbuffers::Allocator>::grow_downwards(&mut x).unwrap();
        assert_eq!(&*x, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]);
    }
}
