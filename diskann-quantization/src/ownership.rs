/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use diskann_utils::{Reborrow, ReborrowMut};

/// A variation of [`Deref`] that copies the underlying value instead of returning a
/// reference. This allows structs like [`Ref`] to hold unaligned references.
pub trait CopyRef {
    type Target: Copy;
    fn copy_ref(&self) -> Self::Target;
}

/// A variation of [`DerefMut`] that copies the underlying value instead of returning a
/// mutable reference. This allows structs like [`Mut`] to hold unaligned references.
pub trait CopyMut: CopyRef {
    fn copy_mut(&mut self, value: Self::Target);
}

/// Wrapper struct for owning a value of type T.
#[derive(Debug, Clone, Copy, Default)]
#[repr(transparent)]
pub struct Owned<T>(pub T)
where
    T: 'static;

impl<T> From<T> for Owned<T>
where
    T: 'static,
{
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T> Deref for Owned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for Owned<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> CopyRef for Owned<T>
where
    T: Copy,
{
    type Target = T;
    fn copy_ref(&self) -> T {
        self.0
    }
}

impl<T> CopyMut for Owned<T>
where
    T: Copy,
{
    fn copy_mut(&mut self, value: T) {
        self.0 = value;
    }
}

impl<'a, T> Reborrow<'a> for Owned<T>
where
    T: Copy,
{
    type Target = Ref<'a, T>;
    fn reborrow(&'a self) -> Self::Target {
        Ref::from(&self.0)
    }
}

impl<'a, T> ReborrowMut<'a> for Owned<T>
where
    T: Copy,
{
    type Target = Mut<'a, T>;
    fn reborrow_mut(&'a mut self) -> Self::Target {
        Mut::from(&mut self.0)
    }
}

/// A wrapper struct for a constant unaligned reference.
#[derive(Debug)]
pub struct Ref<'a, T: ?Sized> {
    ptr: NonNull<T>,
    _lifetime: std::marker::PhantomData<&'a T>,
}

impl<T> Ref<'_, T>
where
    T: ?Sized,
{
    /// Construct a new `Ref` around the pointer.
    ///
    /// # Safety
    ///
    /// Callers must ensure that `ptr` satisfies all the requirements of a traditional Rust
    /// reference for the associated lifetime, with the relaxation that `ptr` need not be
    /// aligned to `std::mem::align_of::<T>()`.
    pub unsafe fn new(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _lifetime: std::marker::PhantomData,
        }
    }
}

impl<T> Clone for Ref<'_, T>
where
    T: ?Sized,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Ref<'_, T> where T: ?Sized {}

// SAFETY: `Ref` behaves like a reference, so is `Send` if `T` is `Send`.
unsafe impl<T> Send for Ref<'_, T> where T: ?Sized + Send {}

// SAFETY: `Ref` behaves like a reference, so is `Sync` if `T` is `Sync`.
unsafe impl<T> Sync for Ref<'_, T> where T: ?Sized + Sync {}

impl<T> Deref for Ref<'_, T>
where
    T: ?Sized,
{
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: Constructors of `Ref` are required to ensure that the underlying pointer
        // points to valid memory and that there is not a simultaneous exclusive borrow of
        // said reference.
        //
        // Therefore, the pointer must be safe to dereference.
        unsafe { &*self.ptr.as_ptr().cast_const() }
    }
}

impl<T> CopyRef for Ref<'_, T>
where
    T: Copy,
{
    type Target = T;
    fn copy_ref(&self) -> T {
        // SAFETY: Constructors of `Ref` are required to ensure that the underlying pointer
        // points to valid memory and that there is not a simultaneous exclusive borrow of
        // said reference.
        //
        // Therefore, the pointer must be safe to dereference.
        unsafe { self.ptr.read_unaligned() }
    }
}

impl<'a, T> From<&'a T> for Ref<'a, T>
where
    T: ?Sized,
{
    fn from(r: &'a T) -> Self {
        // SAFETY: Normal references automatically satisfy the reference-like requirements
        // of `Self::New`.
        //
        // The resulting object does not allow mutable access to the underlying object, so
        // constructing a `NonNull` is safe.
        unsafe { Self::new(r.into()) }
    }
}

impl<'a, T> From<&'a mut T> for Ref<'a, T>
where
    T: ?Sized,
{
    fn from(r: &'a mut T) -> Self {
        // SAFETY: Normal references automatically satisfy the reference-like requirements
        // of `Self::New`. Further, exclusive references are safe to decay to shared reference.
        unsafe { Self::new(r.into()) }
    }
}

impl<'short, T> Reborrow<'short> for Ref<'_, T>
where
    T: ?Sized,
{
    type Target = Ref<'short, T>;
    fn reborrow(&'short self) -> Self::Target {
        *self
    }
}

/// A wrapper struct for an exclusive unaligned reference.
#[derive(Debug)]
pub struct Mut<'a, T: ?Sized> {
    ptr: NonNull<T>,
    _lifetime: std::marker::PhantomData<&'a mut T>,
}

// SAFETY: `Mut` behaves like an exclusive reference, so is `Send` if `T` is `Send`.
unsafe impl<T> Send for Mut<'_, T> where T: ?Sized + Send {}

// SAFETY: `Mut` behaves like an exclusive reference, so is `Sync` if `T` is `Sync`.
unsafe impl<T> Sync for Mut<'_, T> where T: ?Sized + Sync {}

impl<T> Mut<'_, T>
where
    T: ?Sized,
{
    /// Construct a new `Mut` around the pointer.
    ///
    /// # Safety
    ///
    /// Callers must ensure that `ptr` satisfies all the requirements of a traditional Rust
    /// exclusive reference for the associated lifetime, with the relaxation that `ptr` need
    /// not be aligned to `std::mem::align_of::<T>()`.
    pub unsafe fn new(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _lifetime: std::marker::PhantomData,
        }
    }
}

impl<T> Deref for Mut<'_, T>
where
    T: ?Sized,
{
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: Constructors of `Mut` are required to ensure that the underlying pointer
        // points to valid memory and are exclusive.
        //
        // Therefore, the pointer must be safe to dereference.
        unsafe { &*self.ptr.as_ptr().cast_const() }
    }
}

impl<T> DerefMut for Mut<'_, T>
where
    T: ?Sized,
{
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: Constructors of `Mut` are required to ensure that the underlying pointer
        // points to valid memory and are exclusive.
        //
        // Therefore, the pointer must be safe to mutably dereference.
        unsafe { &mut *self.ptr.as_ptr() }
    }
}

impl<T> CopyRef for Mut<'_, T>
where
    T: Copy,
{
    type Target = T;
    fn copy_ref(&self) -> T {
        // SAFETY: Constructors of `Mut` are required to ensure that the underlying pointer
        // points to valid memory and are exclusive.
        //
        // Therefore, the pointer must be safe to dereference.
        unsafe { self.ptr.read_unaligned() }
    }
}

impl<T> CopyMut for Mut<'_, T>
where
    T: Copy,
{
    fn copy_mut(&mut self, value: T) {
        // SAFETY: Constructors of `Mut` are required to ensure that the underlying pointer
        // points to valid memory and are exclusive.
        //
        // Therefore, the pointer must be safe to dereference and to write.
        //
        // The contained object is required to be `Copy`, so we need not worry about
        // invoking destructors.
        unsafe { self.ptr.write_unaligned(value) }
    }
}

impl<'a, T> From<&'a mut T> for Mut<'a, T>
where
    T: ?Sized,
{
    fn from(r: &'a mut T) -> Self {
        // SAFETY: Exclusive references automatically satisfy the requirements of `Self::new`.
        unsafe { Self::new(r.into()) }
    }
}

impl<'short, T> Reborrow<'short> for Mut<'_, T>
where
    T: ?Sized,
{
    type Target = Ref<'short, T>;
    fn reborrow(&'short self) -> Self::Target {
        // SAFETY: `self.ptr` must point to a valid object that is exclusively borrowed for
        // the lifetime of `Self`. Therefore, it is safe to construct a shared reference
        // to the same underlying object for a shorter lifetime.
        unsafe { Ref::new(self.ptr) }
    }
}

impl<'short, T> ReborrowMut<'short> for Mut<'_, T>
where
    T: ?Sized,
{
    type Target = Mut<'short, T>;
    fn reborrow_mut(&'short mut self) -> Self::Target {
        // SAFETY: `self.ptr` must point to a valid object that is exclusively borrowed for
        // the lifetime of `Self`. Therefore, it is safe to construct an exclusive reference
        // to the same underlying object for a shorter lifetime.
        unsafe { Mut::new(self.ptr) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_is_ref<T>(_x: Ref<'_, T>) {}
    fn assert_is_mut_ref<T>(_x: Mut<'_, T>) {}

    #[test]
    fn test_owned() {
        let from: f32 = 10.0;
        let mut owned: Owned<f32> = from.into();
        assert_eq!(*owned, 10.0);
        *owned.deref_mut() = 5.0;
        assert_eq!(*owned, 5.0);

        assert_eq!(owned.reborrow().copy_ref(), 5.0);
        assert_is_ref(owned.reborrow());

        owned.reborrow_mut().copy_mut(1.0);
        assert_eq!(*owned, 1.0);
        assert_is_mut_ref(owned.reborrow_mut());
    }

    #[test]
    fn test_ref() {
        let from: f32 = 10.0;
        let r: Ref<f32> = (&from).into();
        assert_eq!(r.copy_ref(), 10.0);
        assert_eq!(*r, 10.0);
        assert_eq!(r.reborrow().copy_ref(), 10.0);
        assert_is_ref(r.reborrow());

        let mut from: f32 = 10.0;
        let r: Ref<f32> = (&mut from).into();
        assert_eq!(r.copy_ref(), 10.0);
    }

    #[test]
    fn test_ref_mut() {
        let mut from: f32 = 10.0;

        let mut r: Mut<f32> = (&mut from).into();
        assert_eq!(r.copy_ref(), 10.0);
        assert_eq!(*r, 10.0);
        assert_eq!(r.reborrow().copy_ref(), 10.0);
        assert_is_ref(r.reborrow());
        assert_is_mut_ref(r.reborrow_mut());

        // CopyMut
        r.copy_mut(5.0);
        assert_eq!(r.copy_ref(), 5.0);
        assert_eq!(*r, 5.0);
        assert_eq!(from, 5.0);

        // DerefMut
        let mut r: Mut<f32> = (&mut from).into();
        *r = 10.0;
        assert_eq!(r.copy_ref(), 10.0);
        assert_eq!(*r, 10.0);
        assert_eq!(from, 10.0);

        // Re-create the Mut.
        let mut r: Mut<f32> = (&mut from).into();
        r.reborrow_mut().copy_mut(1.0);
        assert_eq!(r.copy_ref(), 1.0);
        assert_eq!(*r, 1.0);
        assert_eq!(from, 1.0);
    }
}
