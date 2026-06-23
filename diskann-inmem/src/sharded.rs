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

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
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

    pub(crate) fn external(&self) -> &I {
        self.forward.key()
    }

    pub(crate) fn delete(mut self) {
        self.forward.remove();
        self.backward[self.entry] = None;
    }
}
