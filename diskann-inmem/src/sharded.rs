/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::hash::Hash;

use dashmap::{
    DashMap,
    mapref::entry::{self, OccupiedEntry},
};
use diskann::utils::IntoUsize;
use parking_lot::{RwLock, RwLockWriteGuard};
use thiserror::Error;

const SHARD_SIZE: usize = 1024;

/// Bidirectional mapping between an external id `I` and a dense internal `u32` id.
#[derive(Debug)]
pub(crate) struct Sharded<I>
where
    I: Hash + Eq,
{
    forward: DashMap<I, u32>,
    backward: Vec<RwLock<Box<[Option<I>]>>>,
    capacity: usize,
}

impl<I> Sharded<I>
where
    I: Hash + Eq,
{
    pub(crate) fn new(capacity: usize) -> Self {
        let backward = std::iter::repeat_with(|| {
            let shard = std::iter::repeat_with(|| None).take(SHARD_SIZE).collect();
            RwLock::new(shard)
        })
        .take(capacity.div_ceil(SHARD_SIZE))
        .collect();

        Self {
            forward: DashMap::new(),
            backward,
            capacity,
        }
    }

    /// Establish a mapping between `external` and `internal`.
    ///
    /// # Errors
    ///
    /// Returns [`InsertError::OutOfBounds`] if `internal` is outside the table's capacity.
    /// Returns [`InsertError::ExternalExists`] if `external` is already mapped.
    /// Returns [`InsertError::InternalExists`] if `internal` is already mapped.
    pub(crate) fn insert(&self, external: I, internal: u32) -> Result<(), InsertError>
    where
        I: Eq + Hash + Clone,
    {
        if internal.into_usize() >= self.capacity {
            return Err(InsertError::OutOfBounds);
        }

        let Shard { outer, inner } = self.shard(internal);

        // Take the forward entry first and hold it vacant until the reverse slot is
        // confirmed empty. This makes the pair-write atomic with respect to other
        // `insert` callers: another thread racing on the same `external` will block
        // on the dashmap shard, and another thread racing on the same `internal` will
        // block on the backward shard's write lock.
        let forward = match self.forward.entry(external.clone()) {
            entry::Entry::Occupied(_) => return Err(InsertError::ExternalExists),
            entry::Entry::Vacant(vacant) => vacant,
        };

        let mut shard = self.backward[outer].write();
        if shard[inner].is_some() {
            // Forward entry drops as vacant — no insertion happened.
            return Err(InsertError::InternalExists);
        }
        shard[inner] = Some(external);
        forward.insert(internal);
        Ok(())
    }

    pub(crate) fn contains_external<Q>(&self, external: &Q) -> bool
    where
        I: std::borrow::Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.forward.contains_key(external)
    }

    /// Look up the internal id for an external id.
    pub(crate) fn to_internal<Q>(&self, external: &Q) -> Option<u32>
    where
        I: std::borrow::Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.forward.get(external).map(|v| *v)
    }

    /// Look up the external id for an internal id.
    pub(crate) fn to_external(&self, internal: u32) -> Option<I>
    where
        I: Clone,
    {
        if internal.into_usize() >= self.capacity {
            return None;
        }

        let Shard { outer, inner } = self.shard(internal);
        self.backward[outer].read()[inner].clone()
    }

    /// Validate that a mapping exists for `external` and return an [`Entry`] if successful.
    ///
    /// The [`Entry`] provides a means of error-free deferred deletion to enable coordinated
    /// deletion of slots among multiple stores.
    pub(crate) fn occupied_entry(&self, external: I) -> Option<Entry<'_, I>>
    where
        I: Eq + Hash,
    {
        match self.forward.entry(external) {
            entry::Entry::Vacant(_) => None,
            entry::Entry::Occupied(forward) => {
                let internal = *forward.get();
                let Shard { outer, inner } = self.shard(internal);
                let backward = self.backward[outer].write();
                assert!(
                    backward[inner].is_some(),
                    "id {} removed improperly",
                    internal
                );

                Some(Entry {
                    forward,
                    backward,
                    entry: inner,
                })
            }
        }
    }

    fn shard(&self, i: u32) -> Shard {
        let i = i.into_usize();
        Shard {
            outer: i / SHARD_SIZE,
            inner: i % SHARD_SIZE,
        }
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.capacity
    }

}

struct Shard {
    outer: usize,
    inner: usize,
}

#[derive(Debug, Error)]
pub(crate) enum InsertError {
    #[error("internal id is out of bounds")]
    OutOfBounds,
    #[error("the external id is already mapped")]
    ExternalExists,
    #[error("the internal id is already mapped")]
    InternalExists,
}

/// A handle to a valid entry in a [`Sharded`].
///
/// This can be used to guarantee the presence of an entry prior to deletion to support
/// atomic deletes.
pub(crate) struct Entry<'a, I>
where
    I: Eq + Hash,
{
    forward: OccupiedEntry<'a, I, u32>,
    backward: RwLockWriteGuard<'a, Box<[Option<I>]>>,
    entry: usize,
}

impl<'a, I> Entry<'a, I>
where
    I: Eq + Hash,
{
    pub(crate) fn internal(&self) -> u32 {
        *self.forward.get()
    }

    pub(crate) fn delete(mut self) {
        self.forward.remove();
        self.backward[self.entry] = None;
    }

    #[cfg(test)]
    fn external(&self) -> &I {
        self.forward.key()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_reports_capacity() {
        for capacity in [0, 1, SHARD_SIZE - 1, SHARD_SIZE, SHARD_SIZE + 1, 3 * SHARD_SIZE] {
            let map = Sharded::<u32>::new(capacity);
            assert_eq!(map.capacity(), capacity);
        }
    }

    #[test]
    fn insert_round_trips() {
        let map = Sharded::<u32>::new(16);
        assert!(map.insert(100, 3).is_ok());

        assert_eq!(map.to_internal(&100), Some(3));
        assert_eq!(map.to_external(3), Some(100));
        assert!(map.contains_external(&100));

        // Unmapped ids return nothing.
        assert_eq!(map.to_internal(&101), None);
        assert_eq!(map.to_external(4), None);
        assert!(!map.contains_external(&101));
    }

    #[test]
    fn insert_rejects_out_of_bounds_internal() {
        let map = Sharded::<u32>::new(16);
        assert!(matches!(map.insert(0, 16), Err(InsertError::OutOfBounds)));
        assert!(matches!(
            map.insert(0, u32::MAX),
            Err(InsertError::OutOfBounds)
        ));

        // The largest in-bounds id is accepted.
        assert!(map.insert(0, 15).is_ok());
    }

    #[test]
    fn insert_rejects_duplicate_external_and_preserves_state() {
        let map = Sharded::<u32>::new(16);
        map.insert(7, 5).unwrap();

        assert!(matches!(
            map.insert(7, 6),
            Err(InsertError::ExternalExists)
        ));

        // The failed insert must not have established any partial mapping.
        assert_eq!(map.to_internal(&7), Some(5));
        assert_eq!(map.to_external(6), None);
        assert!(!map.contains_external(&6));
    }

    #[test]
    fn insert_rejects_duplicate_internal_and_preserves_state() {
        let map = Sharded::<u32>::new(16);
        map.insert(7, 5).unwrap();

        assert!(matches!(
            map.insert(8, 5),
            Err(InsertError::InternalExists)
        ));

        // The failed insert must not have established any partial mapping.
        assert_eq!(map.to_external(5), Some(7));
        assert_eq!(map.to_internal(&8), None);
        assert!(!map.contains_external(&8));
    }

    #[test]
    fn to_external_handles_bounds_and_empty_slots() {
        let map = Sharded::<u32>::new(16);
        // In-bounds but unmapped slot.
        assert_eq!(map.to_external(5), None);
        // Out-of-bounds slot.
        assert_eq!(map.to_external(16), None);
    }

    #[test]
    fn mappings_span_shard_boundaries() {
        let capacity = 3 * SHARD_SIZE;
        let map = Sharded::<u32>::new(capacity);

        // Ids straddling every internal shard boundary.
        let ids: [u32; 6] = [
            0,
            (SHARD_SIZE - 1) as u32,
            SHARD_SIZE as u32,
            (2 * SHARD_SIZE - 1) as u32,
            (2 * SHARD_SIZE) as u32,
            (capacity - 1) as u32,
        ];

        for (external, &internal) in ids.iter().enumerate() {
            map.insert(external as u32, internal).unwrap();
        }

        for (external, &internal) in ids.iter().enumerate() {
            assert_eq!(map.to_internal(&(external as u32)), Some(internal));
            assert_eq!(map.to_external(internal), Some(external as u32));
        }
    }

    #[test]
    fn lookup_supports_borrowed_query() {
        let map = Sharded::<String>::new(16);
        map.insert("alpha".to_string(), 1).unwrap();

        // Borrowed `&str` lookups against `String` keys.
        assert!(map.contains_external("alpha"));
        assert_eq!(map.to_internal("alpha"), Some(1));
        assert!(!map.contains_external("beta"));
        assert_eq!(map.to_internal("beta"), None);
    }

    #[test]
    fn occupied_entry_exposes_mapping() {
        let map = Sharded::<u32>::new(16);
        map.insert(42, 9).unwrap();

        let entry = map.occupied_entry(42).expect("entry should exist");
        assert_eq!(entry.internal(), 9);
        assert_eq!(*entry.external(), 42);
    }

    #[test]
    fn occupied_entry_absent_for_unmapped() {
        let map = Sharded::<u32>::new(16);
        assert!(map.occupied_entry(42).is_none());
    }

    #[test]
    fn entry_delete_clears_both_directions() {
        let map = Sharded::<u32>::new(16);
        map.insert(42, 9).unwrap();

        // Just creating and dropping an `occupied_entry` does not clear it.
        {
            let _ = map.occupied_entry(42).unwrap();
            assert!(map.contains_external(&42));
            assert_eq!(map.to_internal(&42), Some(9));
            assert_eq!(map.to_external(9), Some(42));
        }

        map.occupied_entry(42).expect("entry should exist").delete();

        assert!(!map.contains_external(&42));
        assert_eq!(map.to_internal(&42), None);
        assert_eq!(map.to_external(9), None);

        // The freed external and internal ids can be reused.
        assert!(map.insert(42, 9).is_ok());
    }
}
