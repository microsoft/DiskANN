/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::RwLock;

use diskann::{graph::AdjacencyList, utils::IntoUsize};
use thiserror::Error;

use crate::{
    arbiter::Buffer,
    num::{Align, Bytes},
};

type Id = u32;

const LOCK_GRANULARITY: usize = 16;

fn lock_index(i: usize) -> usize {
    i / LOCK_GRANULARITY
}

#[derive(Debug)]
pub struct Neighbors {
    neighbors: Buffer,
    // One lock for each slot in `neighbors`.
    locks: Vec<RwLock<()>>,
}

impl Neighbors {
    pub fn new(entries: usize, max_length: usize) -> Self {
        let bytes = Bytes((max_length + 1) * std::mem::size_of::<Id>());
        let neighbors = Buffer::new(entries, bytes, Align(128));
        let locks = std::iter::repeat_with(|| RwLock::new(()))
            .take(entries.div_ceil(LOCK_GRANULARITY))
            .collect();

        Self { neighbors, locks }
    }

    /// Return the maximum length for any adjacency list.
    pub fn max_length(&self) -> usize {
        // We reserve 4 bytes at the beginning for the length of the adjacency list.
        (self.neighbors.stride().0 - std::mem::size_of::<Id>()) / std::mem::size_of::<Id>()
    }

    pub fn entries(&self) -> usize {
        self.neighbors.len()
    }

    pub fn get(&self, i: usize, neighbors: &mut AdjacencyList<u32>) -> Result<(), OutOfBounds> {
        self.check(i)?;

        let lock = unsafe { self.locks.get_unchecked(lock_index(i)) };

        let _guard = dismiss_poison(lock.read());

        // SAFETY: By consruction `self.buffer` has the same number of entries as
        // `self.locks` and we have already checked that `i` is in-bounds there.
        let (prefix, rest) =
            unsafe { self.neighbors.get_unchecked(i) }.split(std::mem::size_of::<Id>());

        debug_assert_eq!(prefix.len(), std::mem::size_of::<Id>());
        debug_assert!(prefix.as_ptr().is_aligned());

        // SAFETY: We hold the read-lock, so reading is safe. From our bounds checks, we
        // know that this pointer is valid.
        let len: usize = unsafe { prefix.as_ptr().cast::<Id>().read() }
            .into_usize()
            .min(self.max_length());

        let mut resizer = neighbors.resize(len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                rest.as_ptr().as_ptr(),
                resizer.as_mut_ptr().cast::<u8>(),
                len * std::mem::size_of::<Id>(),
            )
        };
        resizer.finish(len);
        Ok(())
    }

    pub fn lock(&self, i: usize) -> Result<Lock<'_>, OutOfBounds> {
        self.check(i)?;
        Ok(unsafe { self.lock_unchecked(i) })
    }

    unsafe fn lock_unchecked(&self, i: usize) -> Lock<'_> {
        let lock = dismiss_poison(unsafe { self.locks.get_unchecked(lock_index(i)) }.write());

        // SAFETY: By consruction `self.buffer` has the same number of entries as
        // `self.locks` and we have already checked that `i` is in-bounds there.
        let slice = unsafe { self.neighbors.get_unchecked(i) };

        debug_assert!(slice.as_ptr().is_aligned());

        let raw = unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_ptr().as_ptr().cast::<Id>(),
                slice.len() / std::mem::size_of::<Id>(),
            )
        };

        Lock { raw, lock }
    }

    pub fn set(&self, i: usize, neighbors: &[u32]) -> Result<(), SetError> {
        self.check(i).map_err(SetError::OutOfBounds)?;

        // We can check the length of `neighbors` before acquiring any locks as an early exit.
        if neighbors.len() > self.max_length() {
            return Err(SetError::TooLong(TooLong));
        }

        let lock = unsafe { self.lock_unchecked(i) };
        unsafe { lock.write_unchecked(neighbors) };
        Ok(())
    }

    fn check(&self, i: usize) -> Result<(), OutOfBounds> {
        if i >= self.entries() {
            Err(OutOfBounds(i))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("index {} is out-of-bounds", self.0)]
pub struct OutOfBounds(usize);

#[derive(Debug, Clone, Copy, Error)]
pub enum SetError {
    #[error(transparent)]
    OutOfBounds(OutOfBounds),
    #[error(transparent)]
    TooLong(TooLong),
}

// We carefully guard where locks are acquired in this function, so that panicking while
// holding a lock won't happen and if it does, we know we're still in decent shape.
fn dismiss_poison<T>(r: std::sync::LockResult<T>) -> T {
    match r {
        Ok(v) => v,
        Err(poison) => poison.into_inner(),
    }
}

/// A locked adjacency list to implement atomic read-modify-write operations.
#[derive(Debug)]
pub struct Lock<'a> {
    // The raw adjacency list with the actual length stored as the first element.
    //
    // This **must** have a length of at least one.
    //
    // Also, `raw.len()` must be less than `u32::MAX`.
    raw: &'a mut [u32],
    // VERY IMPORTANT: `lock` has to be **after** `raw` because `lock` is guarding `raw`
    // and thus must be dropped **after** `raw`.
    lock: std::sync::RwLockWriteGuard<'a, ()>,
}

impl Lock<'_> {
    /// Return the capacity of the neighbor buffer.
    pub fn capacity(&self) -> usize {
        self.raw.len() - 1
    }

    /// Return the current length of the neighbor list.
    ///
    /// This is guaranteed to be less than [`capacity`](Self::capacity).
    pub fn len(&self) -> usize {
        // SAFETY: By construction, `self.raw` has a length of at least 1.
        //
        // The `min` operation is to be conservative.
        unsafe { self.raw.get_unchecked(0) }
            .into_usize()
            .min(self.capacity())
    }

    /// View the current contents of the locked adjacency list as a slice.
    pub fn as_slice(&self) -> &[u32] {
        let len = self.len();
        unsafe { self.raw.get_unchecked(1..len + 1) }
    }

    /// Consume the [`Lock`] - copying the contents of `neighbors`.
    ///
    /// Returns an error if `neighbors.len() > self.capacity()` without copying any of the
    /// contents of `neighbors`.
    pub fn write(self, neighbors: &[u32]) -> Result<(), TooLong> {
        if neighbors.len() > self.capacity() {
            return Err(TooLong);
        }

        unsafe { self.write_unchecked(neighbors) };
        Ok(())
    }

    unsafe fn write_unchecked(self, neighbors: &[u32]) {
        let len = neighbors.len();
        debug_assert!(len <= self.capacity());
        unsafe {
            std::ptr::copy_nonoverlapping(neighbors.as_ptr(), self.raw.as_mut_ptr().add(1), len)
        }
        *unsafe { self.raw.get_unchecked_mut(0) } = len as u32;
        // `self.raw` is dropped first, then `self.lock` which was guarding it.
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("too long")]
pub struct TooLong;

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
}
