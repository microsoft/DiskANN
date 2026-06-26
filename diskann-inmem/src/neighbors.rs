/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A Concurrent Graph Structure
//!
//! The [`Neighbors`] data structure is a concurrent graph managed out of a single allocation.
//! The use of a single allocation puts a hard upper-bound on the length each adjacency list,
//! which is enforced by the types in this module.
//!
//! Concurrency is obtained using sharded read/write locks, with [`Neighbors::get`] and
//! [`Neighbors::set`] acquiring read and write locks respectively.
//!
//! To implement atomic read-modify-write operations, [`Neighbors::lock`] can be used to
//! obtain a [`Lock`]ed list.
//!
//! Due to lock sharding, attempting to acquire multiple [`Lock`]s to a single [`Neighbors`]
//! simultaneously can lead to dead-lock.
//!
//! ## Performance Considerations
//!
//! Adjacency lists written through the APIs exposed in this module are not validated for
//! uniqueness nor for being in-bounds. These are the caller's responsibility.

use std::ptr::NonNull;

use diskann::{graph::AdjacencyList, utils::IntoUsize};
use parking_lot::{RwLock, RwLockWriteGuard};
use thiserror::Error;

use crate::{
    buffer::{Buffer, BufferError},
    num::{Align, Bytes},
};

type Id = u32;

/// Locks are shared among groups of adjacency lists.
///
/// Adjacency lists whose indices map to the same lock group (i.e. `i / LOCK_GRANULARITY`)
/// share a single `RwLock`. This means that holding a [`Lock`] on slot `i` will also block
/// operations on any slot `j` in the same group.
///
/// **Deadlock hazard**: attempting to acquire two [`Lock`]s simultaneously can deadlock if
/// they fall in the same lock group — or even across groups, depending on acquisition order.
/// Callers must not hold more than one [`Lock`] at a time.
const LOCK_GRANULARITY: usize = 16;

fn lock_index(i: u32) -> usize {
    i.into_usize() / LOCK_GRANULARITY
}

/// A concurrent graph data structure with a fixed number of adjacency lists and a fixed
/// upper-bound for each adjacency list's length. See the [module level docs](self) for
/// more detail.
///
/// Adjacency lists are indexed by `[0, Neighbors::entries)`.
#[derive(Debug)]
pub(crate) struct Neighbors {
    neighbors: Buffer,
    locks: Vec<RwLock<()>>,
}

impl Neighbors {
    /// Construct a new [`Neighbors`] capable of holding `entries` adjacency lists with a
    /// maximum length of `max_length`.
    ///
    /// # Errors
    ///
    /// Returns an error if `(max_length + 1) * size_of::<u32>()` overflows `usize`
    /// (unreachable on 64-bit targets) or the resulting allocation would exceed
    /// `isize::MAX` bytes.
    pub(crate) fn new(entries: u32, max_length: u32) -> Result<Self, NeighborsError> {
        let bytes = max_length
            .into_usize()
            .checked_add(1)
            .and_then(|len| len.checked_mul(std::mem::size_of::<Id>()))
            .map(Bytes::new)
            .ok_or(NeighborsError::Overflow(max_length))?;

        // We materialize slices of `Id` into the raw byte buffers.
        //
        // To make this sound, the base allocation must be that of `Id` so the slice
        // materialization is properly aligned.
        const ALIGN: Align = Align::_128;
        const {
            assert!(
                ALIGN.value() >= Align::of::<Id>().value(),
                "buffer alignment must be at least that of the ID"
            );
        }

        let neighbors = Buffer::new(entries.into_usize(), bytes, ALIGN)?;

        let locks = std::iter::repeat_with(|| RwLock::new(()))
            .take(entries.into_usize().div_ceil(LOCK_GRANULARITY))
            .collect();

        Ok(Self { neighbors, locks })
    }

    /// Return the maximum length for any adjacency list.
    pub(crate) fn max_length(&self) -> usize {
        // We reserve 4 bytes at the beginning for the length of the adjacency list.
        (self.neighbors.stride().value() - std::mem::size_of::<Id>()) / std::mem::size_of::<Id>()
    }

    /// Return the maximum length for any adjacency list as a 32-bit integer.
    pub(crate) fn max_length_u32(&self) -> u32 {
        // Lossless by the invariants on `Self::new`.
        self.max_length() as u32
    }

    /// Return the number of adjacency lists contained by this graph.
    pub(crate) fn entries(&self) -> u32 {
        // Cast is lossless by construction.
        self.neighbors.len() as u32
    }

    /// Copy the contents of adjacency list `i` into `neighbors`.
    ///
    /// Returns an error if `i` exceeds [`Self::entries`].
    pub(crate) fn get(
        &self,
        i: u32,
        neighbors: &mut AdjacencyList<u32>,
    ) -> Result<(), OutOfBounds> {
        self.check(i)?;

        // SAFETY: We've checked that `i` is in-bounds.
        let lock = unsafe { self.locks.get_unchecked(lock_index(i)) };

        let _guard = lock.read();

        // SAFETY: By construction `self.buffer` has the same number of entries as
        // `self.locks` and we have already checked that `i` is in-bounds there.
        let (prefix, rest) =
            unsafe { self.neighbors.get_unchecked(i.into_usize()) }.split(Bytes::size_of::<Id>());

        debug_assert_eq!(prefix.len(), Bytes::size_of::<Id>());
        debug_assert!(prefix.as_ptr().cast::<Id>().is_aligned());

        // SAFETY: We hold the read-lock, so reading is safe. From our bounds checks, we
        // know that this pointer is valid.
        let len: usize = unsafe { prefix.as_ptr().cast::<Id>().read() }
            .min(self.max_length_u32())
            .into_usize();

        let mut resizer = neighbors.resize(len);

        // SAFETY: We've validated that the two slices are valid. They cannot overlap
        // because `neighbors` is provided externally by exclusive reference.
        unsafe {
            std::ptr::copy_nonoverlapping(
                rest.as_mut_ptr(),
                resizer.as_mut_ptr().cast::<u8>(),
                len * std::mem::size_of::<Id>(),
            )
        };
        resizer.finish(len);
        Ok(())
    }

    /// Lock adjacency list `i` for read-modify-write operations.
    ///
    /// Returns an error if `i` exceeds [`Self::entries`].
    pub(crate) fn lock(&self, i: u32) -> Result<Lock<'_>, OutOfBounds> {
        self.check(i)?;

        // SAFETY: `i` is in-bounds.
        Ok(unsafe { self.lock_unchecked(i) })
    }

    /// Lock adjacency-list `i` without bounds-checking.
    ///
    /// # SAFETY
    ///
    /// `i` must be in-bounds.
    unsafe fn lock_unchecked(&self, i: u32) -> Lock<'_> {
        // SAFETY: `i` is in-bounds.
        let lock = unsafe { self.locks.get_unchecked(lock_index(i)) }.write();

        // SAFETY: By construction `self.buffer` has the same number of entries as
        // `self.locks` and we have already checked that `i` is in-bounds there.
        let slice = unsafe { self.neighbors.get_unchecked(i.into_usize()) };

        debug_assert!(slice.as_ptr().cast::<Id>().is_aligned());

        Lock {
            ptr: slice.as_non_null().cast::<Id>(),
            capacity: self.max_length().into_usize(),
            _lock: lock,
        }
    }

    /// Overwrite the contents of adjacency list `i` with `neighbors`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * `i` exceeds [`Self::entries`].
    /// * `neighbors.len()` exceeds [`Self::max_length_u32`].
    ///
    /// If an error is returned, the graph is left unmodified.
    pub(crate) fn set(&self, i: u32, neighbors: &[u32]) -> Result<(), SetError> {
        self.check(i).map_err(SetError::OutOfBounds)?;

        // We can check the length of `neighbors` before acquiring any locks as an early exit.
        if neighbors.len() > self.max_length().into_usize() {
            return Err(SetError::TooLong(TooLong {
                got: neighbors.len(),
                max: self.max_length_u32(),
            }));
        }

        // SAFETY: We've checked `i` is in-bounds.
        let lock = unsafe { self.lock_unchecked(i) };

        // SAFETY: `neighbors.len() <= self.max_length()`.
        unsafe { lock.write_unchecked(neighbors) };
        Ok(())
    }

    fn check(&self, i: u32) -> Result<(), OutOfBounds> {
        if i >= self.entries() {
            Err(OutOfBounds(i))
        } else {
            Ok(())
        }
    }
}

/// Errors returned by [`Neighbors::new`].
#[derive(Debug, Error)]
pub(crate) enum NeighborsError {
    /// Computing the per-list byte size `(max_length + 1) * size_of::<u32>()` overflowed
    /// `usize`.
    ///
    /// Unreachable on 64-bit targets.
    #[error("adjacency list length of {0} is too long")]
    Overflow(u32),

    /// Allocation of the underlying buffer failed.
    ///
    /// This can occur if the total allocation size (`entries * per-list bytes`)
    /// would exceed `isize::MAX`, or if the underlying allocator returns an error.
    #[error("neighbor buffer allocation failed")]
    AllocationFailed(#[from] BufferError),
}

/// Attempted to access a [`Neighbors`] at an out-of-bounds index.
#[derive(Debug, Clone, Copy, Error)]
#[error("index {} is out-of-bounds", self.0)]
pub(crate) struct OutOfBounds(u32);

crate::opaque!(OutOfBounds);

/// A neighbor list was longer than the configured per-list capacity.
///
/// `got` is the caller-supplied length (any `usize`); `max` is the per-list capacity,
/// which is bounded by `u32` per [`Neighbors::new`].
#[derive(Debug, Clone, Copy, Error)]
#[error("length {} exceeds the max length {}", self.got, self.max)]
pub(crate) struct TooLong {
    got: usize,
    max: u32,
}

crate::opaque!(TooLong);

/// Errors during [`Neighbors::set`].
#[derive(Debug, Clone, Copy, Error)]
pub(crate) enum SetError {
    /// Attempted to access an out-of-bounds index.
    #[error(transparent)]
    OutOfBounds(OutOfBounds),

    /// The new adjacency list was too long.
    #[error(transparent)]
    TooLong(TooLong),
}

crate::opaque!(SetError);

/// A locked adjacency list to implement atomic read-modify-write operations.
///
/// Callers must not hold more than one `Lock` at a time. See [`LOCK_GRANULARITY`] for
/// details on the deadlock hazard.
pub(crate) struct Lock<'a> {
    ptr: NonNull<u32>,
    capacity: usize,
    _lock: RwLockWriteGuard<'a, ()>,
}

impl Lock<'_> {
    /// Return the capacity of the neighbor buffer.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the current length of the neighbor list.
    ///
    /// This is guaranteed to be less than or equal to [`capacity`](Self::capacity).
    pub(crate) fn len(&self) -> usize {
        // SAFETY: By construction, `self.raw` has a length of at least 1.
        //
        // The `min` operation defensively clamps in case the stored length has been
        // corrupted; under normal operation it should already be `<= capacity`.
        unsafe { self.ptr.read() }.into_usize().min(self.capacity())
    }

    /// Consume `self`, appending `neighbors` to the list.
    ///
    /// Returns an error if the concatenated list would exceed [`Self::capacity`] without
    /// modify the adjacency list.
    ///
    /// This method does not attempt to deduplicate `neighbors`.
    pub(crate) fn append(self, neighbors: &[u32]) -> Result<(), TooLong> {
        let len = self.len();
        let newlen = len.saturating_add(neighbors.len());

        if newlen > self.capacity() {
            return Err(TooLong {
                got: newlen,
                max: self.capacity as u32,
            });
        }

        // SAFETY: We've verified that both regions are in-bounds.
        //
        // The slices have to be disjoint because `self` effectively owns its data while
        // it is alive and this method receives by-value.
        unsafe {
            std::ptr::copy_nonoverlapping(
                neighbors.as_ptr(),
                self.ptr.add(len + 1).as_ptr(),
                neighbors.len(),
            )
        }

        // SAFETY: `self.ptr` is guaranteed to be valid for at least 4-bytes, and we own the
        // underlying data until `drop`.
        unsafe { self.ptr.write(newlen as u32) };

        Ok(())
    }

    /// Write the contents of `neighbors` into `self` without validating lenghts.
    ///
    /// # Safety
    ///
    /// `neighbors.len() <= self.capacity()`.
    unsafe fn write_unchecked(self, neighbors: &[u32]) {
        let len = neighbors.len();
        debug_assert!(len <= self.capacity());

        // SAFETY: the caller asserts that the pointer arithmetic is sound.
        //
        // The slices are disjoint because `self` owns its data and this method receives
        // by value.
        unsafe { std::ptr::copy_nonoverlapping(neighbors.as_ptr(), self.ptr.as_ptr().add(1), len) }

        // SAFETY: `self.ptr` is guaranteed to be valid for at least 4-bytes, and we own the
        // underlying data until `drop`.
        unsafe { self.ptr.write(len as u32) };
    }

    #[cfg(test)]
    fn as_slice(&self) -> &[u32] {
        let len = self.len();

        // SAFETY: by construction - this access is in-bounds and `Lock` has exclusive
        // access too its data, so we're free to hand out a raw slice.
        unsafe { std::slice::from_raw_parts(self.ptr.add(1).as_ptr().cast_const(), len) }
    }

    #[cfg(test)]
    fn write(self, neighbors: &[u32]) -> Result<(), TooLong> {
        if neighbors.len() > self.capacity() {
            return Err(TooLong {
                got: neighbors.len(),
                max: self.capacity as u32,
            });
        }

        // SAFETY: We've checked that `neighbors.len() <= self.capacity()`.
        unsafe { self.write_unchecked(neighbors) };
        Ok(())
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for Lock<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lock")
            .field("ptr", &self.ptr)
            .field("capacity", &self.capacity)
            .field("lock", &())
            .finish()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test::Sequencer;

    // -- OutOfBounds checks --

    #[test]
    fn out_of_bounds_rejects_indices_beyond_entries() {
        let n = Neighbors::new(4, 4).unwrap();
        // entries == 4, so valid indices are 0..=3.
        // Regression test: a buggy `check` using `i == entries()` would let
        // `entries+1`, `entries+2`, ... slip through to UB.
        let mut out = AdjacencyList::with_capacity(4);
        for bad in [4u32, 5, 100, u32::MAX] {
            assert!(matches!(n.get(bad, &mut out), Err(OutOfBounds(_))));
            assert!(matches!(n.set(bad, &[]), Err(SetError::OutOfBounds(_))));
            assert!(matches!(n.lock(bad), Err(OutOfBounds(_))));
        }
    }

    #[test]
    fn empty_neighbors_rejects_all_access() {
        let n = Neighbors::new(0, 4).unwrap();
        let mut out = AdjacencyList::with_capacity(4);
        for i in [0u32, 1, u32::MAX] {
            assert!(matches!(n.get(i, &mut out), Err(OutOfBounds(_))));
            assert!(matches!(n.set(i, &[]), Err(SetError::OutOfBounds(_))));
            assert!(matches!(n.lock(i), Err(OutOfBounds(_))));
        }
    }

    // TooLong errors

    #[test]
    fn set_rejects_oversized_neighbors() {
        let n = Neighbors::new(4, 3).unwrap();
        let too_many = &[1, 2, 3, 4];
        assert!(matches!(n.set(0, too_many), Err(SetError::TooLong(_))));
    }

    #[test]
    fn lock_write_rejects_oversized_neighbors() {
        let n = Neighbors::new(4, 3).unwrap();
        let lock = n.lock(0).unwrap();
        assert!(lock.write(&[1, 2, 3, 4]).is_err());
    }

    #[test]
    fn lock_append_rejects_overflow() {
        let n = Neighbors::new(4, 3).unwrap();
        n.set(0, &[1, 2]).unwrap();
        let lock = n.lock(0).unwrap();
        assert!(lock.append(&[3, 4]).is_err());
    }

    #[test]
    fn lock_implements_debug() {
        let n = Neighbors::new(4, 3).unwrap();
        let lock = n.lock(0).unwrap();
        let _ = format!("{:?}", lock);
    }

    // -- Lock::append --

    #[test]
    fn append_preserves_existing_and_adds_new() {
        let n = Neighbors::new(4, 6).unwrap();
        n.set(0, &[10, 20]).unwrap();

        let lock = n.lock(0).unwrap();
        assert_eq!(lock.as_slice(), &[10, 20]);
        lock.append(&[30, 40, 50]).unwrap();

        let mut out = AdjacencyList::with_capacity(6);
        n.get(0, &mut out).unwrap();
        assert_eq!(&*out, &[10, 20, 30, 40, 50]);
    }

    #[test]
    fn append_to_empty() {
        let n = Neighbors::new(4, 4).unwrap();

        let lock = n.lock(0).unwrap();
        assert_eq!(lock.as_slice(), &[]);
        lock.append(&[1, 2, 3]).unwrap();

        let mut out = AdjacencyList::with_capacity(4);
        n.get(0, &mut out).unwrap();
        assert_eq!(&*out, &[1, 2, 3]);
    }

    #[test]
    fn append_fills_to_capacity() {
        let n = Neighbors::new(1, 3).unwrap();
        n.set(0, &[1]).unwrap();

        let lock = n.lock(0).unwrap();
        lock.append(&[2, 3]).unwrap();

        let mut out = AdjacencyList::with_capacity(3);
        n.get(0, &mut out).unwrap();
        assert_eq!(&*out, &[1, 2, 3]);
    }

    #[test]
    fn append_empty_slice_is_noop() {
        let n = Neighbors::new(1, 4).unwrap();
        n.set(0, &[10, 20]).unwrap();

        let lock = n.lock(0).unwrap();
        lock.append(&[]).unwrap();

        let mut out = AdjacencyList::with_capacity(4);
        n.get(0, &mut out).unwrap();
        assert_eq!(&*out, &[10, 20]);
    }

    #[test]
    fn write_overwrites_longer_list() {
        let n = Neighbors::new(1, 5).unwrap();
        n.set(0, &[1, 2, 3, 4, 5]).unwrap();

        // Overwrite with a shorter list.
        let lock = n.lock(0).unwrap();
        assert_eq!(lock.len(), 5);
        lock.write(&[99]).unwrap();

        // The length must reflect the new shorter list, not the old one.
        let mut out = AdjacencyList::with_capacity(5);
        n.get(0, &mut out).unwrap();
        assert_eq!(&*out, &[99]);
    }

    // Clear the adjacency list in `neighbors`.
    //
    // Receives by `&mut` to ensure exclusivity.
    fn clear(neighbors: &mut Neighbors) {
        for i in 0..neighbors.entries() {
            neighbors.set(i, &[]).unwrap();
        }

        assert_is_cleared(neighbors);
    }

    fn assert_is_cleared(neighbors: &mut Neighbors) {
        for i in 0..neighbors.entries() {
            assert!(neighbors.lock(i).unwrap().is_empty());
        }
    }

    #[test]
    fn basic_test() {
        let mut neighbors = Neighbors::new(10, 4).unwrap();
        assert_eq!(neighbors.entries(), 10);
        assert_eq!(neighbors.max_length(), 4);

        let mut list = AdjacencyList::new();
        for i in 0..neighbors.entries() {
            list.clear();
            list.extend_from_slice(&[1, 2, 3, 4]);
            neighbors.get(i, &mut list).unwrap();
            assert!(list.is_empty());

            let lock = neighbors.lock(i).unwrap();
            assert_eq!(lock.capacity(), neighbors.max_length());
            assert_eq!(lock.len(), 0);
            assert!(lock.is_empty());
            assert_eq!(lock.as_slice(), &[]);
        }

        // Verify out-of-bounds accesses error.
        let oob = neighbors.entries();
        assert!(matches!(neighbors.get(oob, &mut list), Err(OutOfBounds(_))));
        assert!(matches!(neighbors.lock(oob), Err(OutOfBounds(_))));
        assert!(matches!(
            neighbors.set(oob, &[1, 2, 3, 4, 5, 6]),
            Err(SetError::OutOfBounds(_))
        ));

        let generate =
            |round: u32, entry: u32| -> Vec<u32> { (0..(round + 1)).map(|r| entry + r).collect() };

        // Test mutation via `Neighbors::set`.
        for round in 0..neighbors.max_length_u32() {
            for i in 0..neighbors.entries() {
                let v = generate(round, i);
                neighbors.set(i, &v).unwrap();
            }

            for i in 0..neighbors.entries() {
                let expected = generate(round, i);
                neighbors.get(i, &mut list).unwrap();
                assert_eq!(&*list, &*expected);

                let lock = neighbors.lock(i).unwrap();
                assert_eq!(lock.as_slice(), &*expected);
            }
        }

        clear(&mut neighbors);

        // Test mutation via `lock + write`.
        for round in 0..neighbors.max_length_u32() {
            for i in 0..neighbors.entries() {
                let v = generate(round, i);
                neighbors.lock(i).unwrap().write(&v).unwrap();
            }

            for i in 0..neighbors.entries() {
                let expected = generate(round, i);
                neighbors.get(i, &mut list).unwrap();
                assert_eq!(&*list, &*expected);

                let lock = neighbors.lock(i).unwrap();
                assert_eq!(lock.as_slice(), &*expected);
            }
        }

        clear(&mut neighbors);

        // Test mutation via `lock + append`.
        for round in 0..neighbors.max_length_u32() {
            for i in 0..neighbors.entries() {
                neighbors.lock(i).unwrap().append(&[round + i]).unwrap();
            }

            for i in 0..neighbors.entries() {
                let expected = generate(round, i);

                neighbors.get(i, &mut list).unwrap();
                assert_eq!(&*list, &*expected);

                let lock = neighbors.lock(i).unwrap();
                assert_eq!(lock.as_slice(), &*expected);
            }
        }

        clear(&mut neighbors);
    }

    //-------------------//
    // Concurrency Tests //
    //-------------------//

    // Verify that holding a `Lock` correctly blocks reads for the same adjacency list.
    #[test]
    fn lock_blocks_get() {
        for _ in 0..10 {
            let neighbors = Neighbors::new(3, 4).unwrap();
            let seq = Sequencer::new();

            std::thread::scope(|s| {
                let handle = s.spawn(|| {
                    seq.wait_for(0);
                    let mut list = AdjacencyList::new();
                    neighbors.get(0, &mut list).unwrap();
                    list
                });

                seq.until_waiting_for(0);
                let lock = neighbors.lock(0).unwrap();
                seq.advance_past(0);

                lock.write(&[1, 2, 3, 4]).unwrap();
                let list = handle.join().unwrap();
                assert_eq!(&*list, &[1, 2, 3, 4]);
            });
        }
    }

    #[test]
    fn many_appends() {
        let max_length = if cfg!(miri) { 100 } else { 1000 };

        let neighbors = Neighbors::new(1, max_length).unwrap();

        let num_threads = 4;
        let barrier = std::sync::Barrier::new(num_threads);

        std::thread::scope(|s| {
            let neighbors_ref = &neighbors;
            let barrier_ref = &barrier;

            for thread_id in 0..num_threads {
                s.spawn(move || {
                    barrier_ref.wait();
                    let mut i = thread_id as u32;
                    let upper = neighbors_ref.max_length() as u32;
                    while i < upper {
                        neighbors_ref.lock(0).unwrap().append(&[i]).unwrap();
                        i += num_threads as u32;
                    }
                });
            }
        });

        let mut list = AdjacencyList::new();
        let expected: Vec<_> = (0..neighbors.max_length()).map(|i| i as u32).collect();
        neighbors.get(0, &mut list).unwrap();
        list.sort();

        assert_eq!(&*list, &*expected);
    }
}
