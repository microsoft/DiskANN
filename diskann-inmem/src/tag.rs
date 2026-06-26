/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! State tags for slots participating in the EBR protocol.
//!
//! This module defines [`Tag`] and [`AtomicTag`], a small state machine used to label
//! individual slots in concurrent data structures. Tags pair with the epoch-based
//! reclamation machinery in [`super::epoch`]: epochs decide *when* it is safe to reclaim a
//! slot, while tags decide *whether* a given slot is currently readable, owned, or in
//! transition.
//!
//! Note that the type system does not enforce the tag protocol — only the documented
//! transitions on [`Tag`] are sound, and it is the caller's responsibility to follow them.

use std::sync::atomic::{AtomicU8, Ordering};

/// A tag for controlling concurrent access to data.
///
/// Tag updates and reads should use [`AtomicTag`].
///
/// A reader holding a [`Guard`](super::epoch::Guard) performs an [`Ordering::Acquire`] load
/// on an [`AtomicTag`]; if [`Tag::can_read`] returns `true`, the reader may access the
/// data this tag protects.
///
/// # Named Tags
///
/// * [`Tag::PUBLISHED`]: The associated slot has been published and may be freely accessed
///   by readers.
///
/// * [`Tag::FROZEN`]: This data is protected and is not expected to be mutated. Readers
///   may still freely access this data. `FROZEN` has no defined transitions in this
///   protocol; once a slot is frozen it remains so for the lifetime of the structure.
///
/// * [`Tag::AVAILABLE`]: The associated slot is not currently storing valid data
///   and is available to use.
///
///   Ownership is acquired via a CAS from `AVAILABLE` to `OWNED`.
///
/// * [`Tag::OWNED`]: The associated data is owned by some thread. Only the thread
///   owning this slot may update it.
///
///   Note that ownership may be transferred between threads as long as this ownership
///   transfer is unambiguous and properly synchronized.
///
///   In this state, the owning thread may write to the associated data.
///
/// * [`Tag::RETIRING`]: Indicates that this slot is currently being [retired](super::epoch).
///   Readers may not access associated data after reading this tag, but readers who accessed
///   the tag before retirement may still exist.
///
///   Only transition away from this value when the corresponding slot is returned from a
///   [`Drain`](super::epoch::Drain).
///
/// # Allowed Transitions
///
/// The following protocol must be used when working with [`AtomicTag`]ged data and a
/// [`Registry`](super::Registry).
///
/// * [`Tag::AVAILABLE`] -> [`Tag::OWNED`]: Use a CAS to ensure unique ownership. Once in
///   the owned state, unsynchronized writes can be made to associated data.
///
/// * [`Tag::OWNED`] -> [`Tag::PUBLISHED`]: Must be done as an [`Ordering::Release`] store
///   and only by the thread that acquired ownership.
///
/// * [`Tag::PUBLISHED`] -> [`Tag::RETIRING`]: Must be done while under a
///   [`Guard`](super::epoch::Guard) and may be done with relaxed atomics. Writes to
///   associated data may not be made. Place into [`Guard::retire`](super::epoch::Guard::retire)
///   for final reclamation.
///
/// * [`Tag::RETIRING`] -> [`Tag::AVAILABLE`]: May only be done if the corresponding slot is
///   retrieved from a [`Drain`](super::epoch::Drain). Writes may occur to associated data
///   and if so, this transition must be made with [`Ordering::Release`].
///
/// # Reading
///
/// Checks to [`Tag::can_read`] can be made following [`Ordering::Acquire`] loads.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub(crate) struct Tag(u8);

impl Tag {
    //-------------//
    // High Values //
    //-------------//

    /// The slot is permanently readable and never mutated again. See [`Tag`].
    pub(crate) const FROZEN: Self = Self::new(u8::MAX);

    /// The slot has been published and is freely readable. See [`Tag`].
    pub(crate) const PUBLISHED: Self = Self::new(u8::MAX - 1);

    //------------//
    // Low Values //
    //------------//

    /// The slot holds no valid data and may be claimed via CAS to [`Tag::OWNED`].
    /// See [`Tag`].
    pub(crate) const AVAILABLE: Self = Self::new(0);

    /// The slot is exclusively owned by a single thread that may write its data.
    /// See [`Tag`].
    pub(crate) const OWNED: Self = Self::new(1);

    /// The slot is in the process of being retired and is no longer readable to new
    /// readers. See [`Tag`].
    pub(crate) const RETIRING: Self = Self::new(2);

    /// NOTE: We rely on reserved values being contiguous so `is_reserved` can be
    /// implemented relatively efficiently.
    const RESERVED: Self = Self::RETIRING;

    /// Return `true` if `self` is one of the protocol's reserved tag values.
    ///
    /// Reserved tags are part of the protocol's fixed vocabulary and are never delivered
    /// as retirement payloads.
    #[must_use = "this function has no side-effects"]
    pub(crate) fn is_reserved(self) -> bool {
        (self <= Self::RESERVED) || (self == Self::FROZEN)
    }

    /// Return `true` if `self` is in a state where it is legal to access tagged data.
    #[must_use = "this function has no side-effects"]
    pub(crate) fn can_read(self) -> bool {
        // Tags are split into `high` (readable) and `low` (non-readable) values so this
        // check reduces to a single comparison.
        self >= Self::PUBLISHED
    }

    /// Construct a new [`Tag`] with `value`.
    #[inline]
    const fn new(value: u8) -> Self {
        Self(value)
    }

    /// Return the value of `self`.
    #[inline]
    const fn value(self) -> u8 {
        self.0
    }
}

impl std::fmt::Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let me = *self;
        if me == Self::AVAILABLE {
            f.write_str("Tag(AVAILABLE)")
        } else if me == Self::OWNED {
            f.write_str("Tag(OWNED)")
        } else if me == Self::RETIRING {
            f.write_str("Tag(RETIRING)")
        } else if me == Self::FROZEN {
            f.write_str("Tag(FROZEN)")
        } else if me == Self::PUBLISHED {
            f.write_str("Tag(PUBLISHED)")
        } else {
            write!(f, "Tag({})", me.value())
        }
    }
}

/// An atomic [`Tag`].
///
/// Memory orderings are the caller's responsibility and must be chosen consistent with the
/// protocol described on [`Tag`].
#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct AtomicTag(AtomicU8);

impl AtomicTag {
    /// Construct a new [`AtomicTag`] initialized to `tag`.
    pub(crate) const fn new(tag: Tag) -> Self {
        Self(AtomicU8::new(tag.value()))
    }

    /// Creates a new reference to a `AtomicTag` from a raw pointer.
    ///
    /// # Safety
    ///
    /// * `ptr` must be aligned to `align_of::<AtomicTag>()`.
    /// * `ptr` must be valid for both reads and writes for the whole lifetime `'a`.
    /// * The caller chooses `'a`; the underlying allocation must outlive `'a`.
    /// * This must adhere to the memory model for atomic accesses. In particular, it must
    ///   not admit conflicting atomic and non-atomic accesses, or atomic accesses of
    ///   different sizes without synchronization.
    ///
    /// See: <https://doc.rust-lang.org/std/sync/atomic/index.html#memory-model-for-atomic-accesses>
    pub(crate) unsafe fn from_ptr<'a>(ptr: *mut AtomicTag) -> &'a Self {
        unsafe { &*ptr }
    }

    /// Perform an atomic compare-exchange with the provided orderings.
    ///
    /// Note that this does not enforce the [`Tag`] transition protocol; the caller must
    /// ensure `current` and `new` correspond to a legal transition.
    ///
    /// See: [`AtomicU8::compare_exchange`].
    pub(crate) fn compare_exchange(
        &self,
        current: Tag,
        new: Tag,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Tag, Tag> {
        self.0
            .compare_exchange(current.value(), new.value(), success, failure)
            .map(Tag::new)
            .map_err(Tag::new)
    }

    /// Perform an atomic load with the provided ordering.
    ///
    /// See: [`AtomicU8::load`].
    pub(crate) fn load(&self, ordering: Ordering) -> Tag {
        Tag::new(self.0.load(ordering))
    }

    /// Perform an atomic store with the provided ordering.
    ///
    /// See: [`AtomicU8::store`].
    pub(crate) fn store(&self, val: Tag, ordering: Ordering) {
        self.0.store(val.value(), ordering)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::{sync::Barrier, thread};

    use crate::{
        buffer::Buffer,
        num::{Align, Bytes},
    };

    fn spin_decrement(m: &AtomicTag, count: usize) {
        for _ in 0..count {
            let mut current = m.load(Ordering::Relaxed);
            while let Err(c) = m.compare_exchange(
                current,
                Tag::new(current.value().wrapping_sub(1)),
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
        let buffer =
            Buffer::new(1, Bytes::size_of::<AtomicTag>(), Align::of::<AtomicTag>()).unwrap();
        let ptr = buffer.get(0).unwrap().as_mut_ptr().cast::<AtomicTag>();

        {
            let tag = unsafe { AtomicTag::from_ptr(ptr) };
            tag.store(Tag::FROZEN, Ordering::Relaxed);
        }

        let count = 1000;
        thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    // Re-derive `p` to avoid issues with `Send`.
                    let p = buffer.get(0).unwrap().as_mut_ptr().cast::<AtomicTag>();
                    let tag = unsafe { AtomicTag::from_ptr(p) };
                    barrier.wait();
                    spin_decrement(&tag, count);
                });
            }
        });

        {
            let g = unsafe { AtomicTag::from_ptr(ptr) }.load(Ordering::Relaxed);
            assert_eq!(g, Tag::new(u8::MAX.wrapping_sub((count * threads) as u8)));
        }
    }

    #[test]
    fn test_is_reserved() {
        assert!(Tag::FROZEN.is_reserved());
        assert!(!Tag::PUBLISHED.is_reserved());

        assert!(Tag::AVAILABLE.is_reserved());
        assert!(Tag::OWNED.is_reserved());
        assert!(Tag::RETIRING.is_reserved());
    }

    #[test]
    fn test_can_read() {
        assert!(Tag::FROZEN.can_read());
        assert!(Tag::PUBLISHED.can_read());

        assert!(!Tag::AVAILABLE.can_read());
        assert!(!Tag::OWNED.can_read());
        assert!(!Tag::RETIRING.can_read());
    }

    #[test]
    fn test_display() {
        assert_eq!(Tag::AVAILABLE.to_string(), "Tag(AVAILABLE)");
        assert_eq!(Tag::OWNED.to_string(), "Tag(OWNED)");
        assert_eq!(Tag::RETIRING.to_string(), "Tag(RETIRING)");
        assert_eq!(Tag::FROZEN.to_string(), "Tag(FROZEN)");
        assert_eq!(Tag::PUBLISHED.to_string(), "Tag(PUBLISHED)");

        // Guard against future changes.
        assert_eq!(Tag::new(Tag::RETIRING.value() + 1).to_string(), "Tag(3)");
        assert_eq!(Tag::new(Tag::PUBLISHED.value() - 1).to_string(), "Tag(253)");
    }
}
