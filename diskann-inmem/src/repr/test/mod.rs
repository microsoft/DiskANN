/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use hashbrown::{HashMap, hash_map::Entry};

use crate::{
    num::{LogicalId, SlotId},
    repr,
};

#[cfg(feature = "quantization")]
use diskann::neighbor::Neighbor;

#[cfg(feature = "quantization")]
use crate::num::IdLimit;

/// A test distance that simply sums scalar floating point values.
#[derive(Debug)]
pub(super) struct TestDistance;

impl repr::internal::RawDistance for TestDistance {
    type Error = diskann::error::Infallible;

    fn eval(&self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error> {
        let x: f32 = bytemuck::pod_read_unaligned(x);
        let y: f32 = bytemuck::pod_read_unaligned(y);
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
        let x: f32 = bytemuck::pod_read_unaligned(x);
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
    #[cfg(feature = "quantization")]
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

//-------------//
// Expand Beam //
//-------------//

/// Performs the following set of tests:
///
/// * [`ExpandBeam::id_limit`] is equal to `id_limit`.
///
/// * [`ExpandBeam::evaluate`]: For each id in `ids` - attempt to evaluate the distance
///   through [`ExpandBeam::evaluate`]. If the id is present in `distances`, assert that
///   the value in `distances` agrees with the result of the `ExpandBeam` method.
///
///   Otherwise, assert that `ExpandBeam` returns `None`.
///
/// * [`ExpandBeam::expand_beam`]: Provide all `ids` to `expand_beam`. Verify that ids not
///   present in `distances` get removed and all remaining ids are present and have a
///   distance value equal to the corresponding entry in `distances`.
#[cfg(feature = "quantization")]
pub(super) fn test_expand_beam(
    accessor: &dyn repr::ExpandBeam,
    id_limit: IdLimit,
    distances: HashMap<SlotId, f32>,
    ids: &[SlotId],
    ctx: &dyn std::fmt::Display,
) {
    assert_eq!(accessor.id_limit(), id_limit, "{ctx}");

    for slot_id in ids {
        if let Some(distance) = distances.get(slot_id) {
            assert_eq!(
                accessor.evaluate(slot_id.value()).unwrap(),
                Some(*distance),
                "failed on slot id {} -- {}",
                slot_id,
                ctx,
            );
        } else {
            assert!(
                accessor.evaluate(slot_id.value()).unwrap().is_none(),
                "failed on slot id {} -- {}",
                slot_id,
                ctx
            );
        }
    }

    // Test via `expand_beam`.
    let list: Vec<u32> = ids.iter().map(|slot_id| slot_id.value()).collect();
    let mut buffer = vec![Neighbor::default(); list.len()];
    let len = repr::safe_expand_beam(accessor, &list, &mut buffer).unwrap();

    let expected: Vec<Neighbor<u32>> = ids
        .iter()
        .filter_map(|slot_id| {
            distances
                .get(slot_id)
                .map(|distance| Neighbor::new(slot_id.value(), *distance))
        })
        .collect();

    assert_eq!(
        expected.len(),
        len,
        "`expand_beam` returned the incorrect number of items -- {}",
        ctx,
    );

    for (i, (got, expected)) in std::iter::zip(buffer.iter(), expected.iter()).enumerate() {
        assert_eq!(
            got.id(),
            expected.id(),
            "failed on entry {} of {} -- {}",
            i,
            len,
            ctx
        );
        assert_eq!(
            got.distance(),
            expected.distance(),
            "failed on entry {} of {} -- {}",
            i,
            len,
            ctx,
        );
    }
}

//-------//
// Prune //
//-------//

/// Test that the [`repr::Prune`] computes distances according to the ground truth in
/// `distances`.
///
/// This assumes that `distances` contains all valid (i.e., between undeleted) entries
/// in `ids` - including self distances.
///
/// For example, if `ids` contains `[0, 1, 2, 3(deleted)]`, then `distances` should contain
/// the keys:
///
/// (0, 0), (0, 1), (0, 2)
/// (1, 0), (1, 1), (1, 2)
/// (2, 0), (2, 1), (2, 2)
#[cfg(feature = "quantization")]
pub(super) fn test_prune(
    accessor: &mut dyn repr::Prune,
    distances: HashMap<(SlotId, SlotId), f32>,
    ids: &[SlotId],
    ctx: &dyn std::fmt::Display,
) {
    let num_present_ids = ids
        .iter()
        .filter(|&&slot_id| distances.contains_key(&(slot_id, slot_id)))
        .count();

    let mut items: HashMap<u32, Option<repr::PruneKey>> =
        ids.iter().map(|slot_id| (slot_id.value(), None)).collect();

    let count = accessor.prepare(items.iter_mut()).unwrap();
    assert_eq!(count, num_present_ids, "{ctx}");

    let mut visited = 0;
    for slot_id0 in ids.iter() {
        if let Some(key0) = items[&slot_id0.value()] {
            for slot_id1 in ids.iter() {
                if let Some(key1) = items[&slot_id1.value()] {
                    let d = accessor.evaluate(key0, key1);
                    let expected = distances[&(*slot_id0, *slot_id1)];
                    assert_eq!(
                        d, expected,
                        "failed for {} x {} -- {}",
                        slot_id0, slot_id1, ctx
                    );

                    visited += 1;
                }
            }
        }
    }

    assert_eq!(
        visited,
        distances.len(),
        "not all distances were visited -- {}",
        ctx
    );
}
