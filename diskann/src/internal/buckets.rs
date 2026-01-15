/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    future::Future,
    ops::{Deref, DerefMut},
};

use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A naive way of amortizing concurrent access to containers by partitioning the ID space
/// into buckets.
///
/// This is used instead of a proper concurrent hash map like
///
/// * [`DashMap`](https://github.com/xacrimon/dashmap)
/// * [`scc::HashMap`](https://docs.rs/scc/latest/scc/hash_map/struct.HashMap.html)
///
/// to keep the interface simple and async-friendly. The internal data structures that use
/// this favor simplicity, control, and test coverage over performance.
#[derive(Debug)]
pub(crate) struct Buckets<T, const N: usize> {
    buckets: Box<[RwLock<T>; N]>,
}

impl<T, const N: usize> Buckets<T, N> {
    /// Construct a new `Buckets` array with buckets initialized to `T::default()`.
    pub(crate) fn new() -> Self
    where
        T: Default,
    {
        Self {
            buckets: Box::new(core::array::from_fn(|_| RwLock::new(T::default()))),
        }
    }

    /// Acquire bucket `i % N` for shared read access.
    pub(crate) fn blocking_read(&self, i: usize) -> Read<'_, T> {
        Read {
            guard: self.buckets[i % N].blocking_read(),
        }
    }

    /// Acquire bucket `i % N` for shared read access.
    pub(crate) fn read(&self, i: usize) -> impl Future<Output = Read<'_, T>> {
        use futures_util::FutureExt;
        self.buckets[i % N].read().map(|guard| Read { guard })
    }

    /// Acquire bucket `i % N` for exclusive write access.
    pub(crate) fn write(&self, i: usize) -> impl Future<Output = Write<'_, T>> {
        use futures_util::FutureExt;
        self.buckets[i % N].write().map(|guard| Write { guard })
    }

    /// Return a mutable reference to bucket `i % N`. This is safe because it receives
    /// by mutable reference.
    pub(crate) fn get_mut(&mut self, i: usize) -> &mut T {
        self.buckets[i % N].get_mut()
    }
}

#[derive(Debug)]
pub(crate) struct Read<'a, T> {
    guard: RwLockReadGuard<'a, T>,
}

impl<T> Deref for Read<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

#[derive(Debug)]
pub(crate) struct Write<'a, T> {
    guard: RwLockWriteGuard<'a, T>,
}

impl<T> Deref for Write<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> DerefMut for Write<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}
