/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use hashbrown::{HashMap, hash_map::Entry};

use crate::{
    num::{LogicalId, SlotId},
    repr,
};

/// A test distance that simply sums scalar floating point values.
#[derive(Debug)]
pub(super) struct TestDistance;

impl repr::internal::RawDistance for TestDistance {
    type Error = diskann::error::Infallible;

    fn eval(&self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error> {
        assert_eq!(x.len(), std::mem::size_of::<f32>());
        assert_eq!(y.len(), std::mem::size_of::<f32>());

        let x = unsafe { x.as_ptr().cast::<f32>().read_unaligned() };
        let y = unsafe { y.as_ptr().cast::<f32>().read_unaligned() };

        Ok(x + y)
    }
}

/// A test query distance that simply sums scalar floating point values.
#[derive(Debug)]
pub(super) struct TestQueryDistance {
    query: f32,
}

impl TestQueryDistance {
    pub(super) fn new(query: f32) -> Self {
        Self { query }
    }
}

impl repr::internal::RawQueryDistance for TestQueryDistance {
    type Error = diskann::error::Infallible;

    fn eval(&self, x: &[u8]) -> Result<f32, Self::Error> {
        assert_eq!(x.len(), std::mem::size_of::<f32>());

        let x = unsafe { x.as_ptr().cast::<f32>().read_unaligned() };

        Ok(self.query + x)
    }
}

////////////////////
// Reference Data //
////////////////////

/// A reference dataset used in testing to track what is in a [`repr::Representation`].
#[derive(Debug)]
pub(super) struct Reference {
    dim: usize,
    data: HashMap<SlotId, (LogicalId, Vec<f32>)>,
    id: HashMap<LogicalId, SlotId>,
}

impl Reference {
    /// Create a new [`Reference`] for data of the specified dimension.
    pub(super) fn new(dim: usize) -> Self {
        Self {
            dim,
            data: HashMap::new(),
            id: HashMap::new(),
        }
    }

    /// Insert the data with the given ID map into the reference dataset.
    ///
    /// # Panics
    ///
    /// Panics if any of the IDs provided are already used or if the dimension of data is
    /// not that expected by the reference.
    pub(super) fn insert(&mut self, logical: LogicalId, slot: SlotId, data: &[f32]) {
        assert_eq!(data.len(), self.dim);
        match self.id.entry(logical) {
            Entry::Vacant(id_entry) => match self.data.entry(slot) {
                Entry::Vacant(data_entry) => {
                    data_entry.insert((logical, data.into()));
                    id_entry.insert(slot);
                }
                Entry::Occupied(_) => panic!(
                    "reference already contains a mapping for {:?}/{:?}",
                    logical, slot
                ),
            },
            Entry::Occupied(_) => panic!(
                "reference already contains a mapping for {:?}/{:?}",
                logical, slot
            ),
        }
    }

    /// Delete the entry corresponding to the logical ID.
    ///
    /// # Panics
    ///
    /// Panics if no mapping for `logical` exists.
    pub(super) fn delete(&mut self, logical: LogicalId) -> SlotId {
        let slot = match self.id.remove(&logical) {
            Some(slot) => slot,
            None => panic!("No entry present for {:?}", logical),
        };

        if self.data.remove(&slot).is_none() {
            panic!("No entry present for {:?}", slot);
        }

        slot
    }

    #[must_use = "this function has no side-effects"]
    pub(super) fn contains_id<I>(&self, i: &I) -> bool
    where
        I: ReferenceLookup,
    {
        self.get(i).is_some()
    }

    /// Get the data payload for the id `i`.
    pub(super) fn get<I>(&self, i: &I) -> Option<&[f32]>
    where
        I: ReferenceLookup,
    {
        i.lookup(self)
    }

    /// Return the [`SlotId`] for `id` if it exists.
    pub(super) fn slot_id_for(&self, id: LogicalId) -> SlotId {
        match self.id.get(&id).copied() {
            Some(id) => id,
            None => panic!("no slot id for {:?}", id),
        }
    }

    /// Return the [`LogicalId`] for `id` if it exists.
    pub(super) fn logical_id_for(&self, id: SlotId) -> LogicalId {
        match self.data.get(&id).map(|(slot, _)| *slot) {
            Some(id) => id,
            None => panic!("no logical id for {:?}", id),
        }
    }
}

impl<I> std::ops::Index<I> for Reference
where
    I: ReferenceLookup,
{
    type Output = [f32];
    fn index(&self, idx: I) -> &[f32] {
        match self.get(&idx) {
            Some(v) => v,
            None => panic!("index {:?} is not in the reference data", idx),
        }
    }
}

pub(super) trait ReferenceLookup: std::fmt::Debug + Sized {
    fn lookup<'a>(&self, reference: &'a Reference) -> Option<&'a [f32]>;
}

impl ReferenceLookup for SlotId {
    fn lookup<'a>(&self, reference: &'a Reference) -> Option<&'a [f32]> {
        reference.data.get(self).map(|(_, v)| &**v)
    }
}

impl ReferenceLookup for LogicalId {
    fn lookup<'a>(&self, reference: &'a Reference) -> Option<&'a [f32]> {
        let slot_id = reference.id.get(self)?;
        Some(
            slot_id
                .lookup(reference)
                .expect("reference is in an inconsistent state"),
        )
    }
}
