/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::atomic::{AtomicU64, Ordering};

/// An atomic [`Generation`] tag.
///
/// Access is performed through [`Ref`] and [`Mut`].
#[derive(Debug)]
#[repr(transparent)]
pub struct Tag(AtomicU64);

impl Tag {
    /// Construct a new [`Tag`] initialized to `generation`.
    pub const fn new(generation: Generation) -> Self {
        Self(AtomicU64::new(generation.value()))
    }

    /// Return a read-only [`Ref`] to `self`.
    pub fn as_ref(&self) -> Ref<'_> {
        Ref::new(&self.0)
    }

    /// Return a read-write [`Mut`] to `self`.
    pub fn as_mut(&self) -> Mut<'_> {
        Mut::new(&self.0)
    }

    /// Creates a new reference to a `Tag` from a raw pointer.
    ///
    /// # Safety
    ///
    /// * `ptr` must be aligned to `align_of::<Tag>()`.
    /// * `ptr` must be valid for both reads and writes for the whole lifetime `'a`.
    /// * This must adhere to the memory model for atomic accesses. In particular, it must
    ///   not admit conflicting atomic and non-atomic accesses, or atomic accesses of
    ///   different sizes without synchronization.
    ///
    /// See: <https://doc.rust-lang.org/std/sync/atomic/index.html#memory-model-for-atomic-accesses>
    pub unsafe fn from_ptr<'a>(ptr: *mut Tag) -> &'a Self {
        unsafe { &*ptr }
    }
}

/// A generation tag for controlling concurrent access to data.
///
/// Generally, generations are decremented from `Generation::MAX`, with higher values
/// representing older generations. This allows zero to stand for "unused" as it is newer
/// than any valid generation.
///
/// Certain low-numbered generations are reserved for special uses. Any generation less
/// than or equal to [`Generation::RESERVED`] is reserved.
///
/// # Reserved Generations
///
/// * [`Generation::AVAILABLE`]: The associated slot is not currently storing valid data
///   and is available to use.
///
///   To acquire ownership, an atomic compare-exchange must be used away from this state.
///
/// * [`Generation::OWNED`]: The associated data is owned by some thread. Only the thread
///   owning this slot may update it.
///
///   Note that ownership may be transferred between threads as long as this ownership
///   transfer is unambiguous and properly synchronized.
///
/// * [`Generation::FROZEN`]: This data is protected and is not expected to be mutated.
///
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Generation(u64);

impl Generation {
    /// The maximum generation. This is the oldest possible generation.
    pub const MAX: Self = Self::new(u64::MAX);

    // Reserved generations.
    //
    // These all have small values, with `0` marking the "available" state.
    // In this way, zeroed allocations for tags naturally begin in the "available" state and
    // don't require additional initialization.
    //
    // If you add states - make sure to increment the `RESERVED` marker!

    /// See [`Generation`].
    pub const AVAILABLE: Self = Self::new(0);

    /// See [`Generation`].
    pub const OWNED: Self = Self::new(1);

    /// See [`Generation`].
    pub const FROZEN: Self = Self::new(2);

    /// The maximum reserved generation. See [`Generation`].
    pub const RESERVED: Self = Self::FROZEN;

    /// Return `true` if `self` belongs to a reserved generation.
    #[must_use = "this function has no side-effects"]
    pub(crate) fn is_reserved(self) -> bool {
        self <= Self::RESERVED
    }

    /// Construct a new [`Generation`] with `value`.
    #[inline]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the value of `self`.
    #[inline]
    pub const fn value(self) -> u64 {
        self.0
    }

    #[cfg(test)]
    const fn add(self, v: u64) -> Self {
        Self(self.0 + v)
    }

    #[cfg(test)]
    const fn sub(self, v: u64) -> Self {
        Self(self.0 - v)
    }
}

/// A read-only handle to a [`Tag`].
///
/// Provides atomic load access to the underlying generation value.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Ref<'a>(&'a AtomicU64);

impl<'a> Ref<'a> {
    #[inline]
    pub(crate) fn new(slot: &'a AtomicU64) -> Self {
        Self(slot)
    }

    #[inline]
    fn inner(&self) -> &'a AtomicU64 {
        self.0
    }

    /// Load the current [`Generation`] with the given ordering.
    #[inline]
    pub fn get(&self, ordering: Ordering) -> Generation {
        Generation::new(self.0.load(ordering))
    }
}

/// A read-write handle to a [`Tag`].
///
/// Provides atomic store and compare-exchange access in addition to the read access
/// inherited from [`Ref`] via [`Deref`](std::ops::Deref).
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Mut<'a>(Ref<'a>);

impl<'a> Mut<'a> {
    #[inline]
    pub(crate) fn new(slot: &'a AtomicU64) -> Self {
        Self(Ref::new(slot))
    }

    /// Attempt to atomically update the generation from `current` to `new`.
    ///
    /// Returns `Ok(current)` on success, or `Err(actual)` if the value was not `current`.
    #[inline]
    pub fn try_set(
        &self,
        current: Generation,
        new: Generation,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Generation, Generation> {
        self.inner()
            .compare_exchange(current.value(), new.value(), success, failure)
            .map(Generation::new)
            .map_err(Generation::new)
    }

    /// Atomically store a [`Generation`] with the given ordering.
    #[inline]
    pub fn set(&self, generation: Generation, ordering: Ordering) {
        self.inner().store(generation.value(), ordering)
    }
}

impl<'a> std::ops::Deref for Mut<'a> {
    type Target = Ref<'a>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::{thread, sync::Barrier};

    use crate::{num::{Bytes,Align}, arbiter::Buffer};

    fn spin_decrement(m: Mut<'_>, count: usize) {
        for i in 0..count {
            let mut current = m.get(Ordering::Relaxed);
            while let Err(c) = m.try_set(
                current,
                current.sub(1),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                current = c;
            }
        }
    }

    #[test]
    fn test_atomic() {
        let threads = 4;
        let barrier = &Barrier::new(threads);

        // This dance basically verifies that we can view the tag though a proper-aligned
        // raw pointer.
        let buffer = Buffer::new(1, Bytes::size_of::<Tag>(), Align::of::<Tag>()).unwrap();
        let ptr = buffer.get(0).unwrap().as_mut_ptr().cast::<Tag>();

        {
            let tag = unsafe { Tag::from_ptr(ptr) };
            tag.as_mut().set(Generation::MAX, Ordering::Relaxed);
        }

        let count = 1000;
        thread::scope(|s| {
            for i in 0..threads {
                s.spawn(|| {
                    // Re-derive `p` to avoid issues with `Send`.
                    let p = buffer.get(0).unwrap().as_mut_ptr().cast::<Tag>();
                    let tag = unsafe { Tag::from_ptr(p) };
                    barrier.wait();
                    spin_decrement(tag.as_mut(), count);
                });
            }
        });

        {
            let tag = unsafe { Tag::from_ptr(ptr) };
            let g = tag.as_ref().get(Ordering::Relaxed);
            assert_eq!(g, Generation::MAX.sub((count * threads) as u64));
        }
    }

    #[test]
    fn test_is_reserved() {
        assert!(Generation::AVAILABLE.is_reserved());
        assert!(Generation::OWNED.is_reserved());
        assert!(Generation::FROZEN.is_reserved());
        assert!(!Generation::FROZEN.add(1).is_reserved());
    }
}
