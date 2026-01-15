/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tools to pass objects with lifetimes across the funtion pointer API.
//!
//! Useful helpers include
//!
//! * [`As`]: Feed through the generic parameter unaltered with no lifetime.
//! * [`Ref`]: Pass an argument through a shared reference.
//! * [`Mut`]: Pass an argument through an exclusive reference.
//!
//! Common primitives like integers and floating point numbers pass through by value.

use std::marker::PhantomData;

/// A lifetime annotator for the function pointer API of [`crate::Architecture`].
///
/// This mainly works around limitations the Rust compiler's ability to infer the proper
/// of dispatched function pointers.
pub trait AddLifetime: 'static {
    /// The type with a lifetime (if any).
    type Of<'a>;
}

macro_rules! self_lifetime {
    ($T:ty) => {
        impl $crate::lifetime::AddLifetime for $T {
            type Of<'a> = $T;
        }
    };
    ($($Ts:ty),+ $(,)?) => {
        $(self_lifetime!($Ts);)+
    }
}

self_lifetime!(
    bool,
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    usize,
    half::f16,
    f32,
    f64,
    String
);

/// An [`AddLifetime`] helper that passes `T` by value.
#[derive(Debug)]
pub struct As<T> {
    _marker: PhantomData<T>,
}

impl<T> As<T> {
    /// Construct a new instance of [`Self`].
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for As<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for As<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for As<T> {}

impl<T> AddLifetime for As<T>
where
    T: 'static,
{
    type Of<'a> = T;
}

/// An [`AddLifetime`] helper that passes `&T`.
#[derive(Debug)]
pub struct Ref<T: ?Sized> {
    _marker: PhantomData<T>,
}

impl<T: ?Sized> Ref<T> {
    /// Construct a new instance of [`Self`].
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for Ref<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ?Sized> Clone for Ref<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Ref<T> {}

impl<T> AddLifetime for Ref<T>
where
    T: ?Sized + 'static,
{
    type Of<'a> = &'a T;
}

/// An [`AddLifetime`] helper that passes `&mut T`.
#[derive(Debug)]
pub struct Mut<T: ?Sized> {
    _marker: PhantomData<T>,
}

impl<T: ?Sized> Mut<T> {
    /// Construct a new instance of [`Self`].
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for Mut<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ?Sized> Clone for Mut<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> Copy for Mut<T> {}

impl<T> AddLifetime for Mut<T>
where
    T: ?Sized + 'static,
{
    type Of<'a> = &'a mut T;
}
