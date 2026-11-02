/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::neighbor::Neighbor;
use hashbrown::{HashMap, hash_map::Entry};

use crate::{
    num::{IdLimit, LogicalId, SlotId},
    repr,
};

/// A test distance that simply sums values as slices of `f32`.
#[derive(Debug)]
pub(super) struct TestDistance;

impl repr::internal::RawDistance for TestDistance {
    type Error = diskann::error::Infallible;

    fn eval(&self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error> {
        assert_eq!(x.len(), y.len());
        assert!(x.len().is_multiple_of(4));
        assert!(y.len().is_multiple_of(4));

        let mut sum: f32 = 0.0;
        for (x, y) in std::iter::zip(x.chunks_exact(4), y.chunks_exact(4)) {
            let x: f32 = bytemuck::pod_read_unaligned(x);
            let y: f32 = bytemuck::pod_read_unaligned(y);
            sum += x + y;
        }

        Ok(sum)
    }
}

/// A test query distance that simply sums scalar floating point values.
#[derive(Debug)]
pub(super) struct TestQueryDistance<'a> {
    query: &'a [f32],
}

impl<'a> TestQueryDistance<'a> {
    pub(super) fn new(query: &'a [f32]) -> Self {
        Self { query }
    }
}

impl repr::internal::RawQueryDistance for TestQueryDistance<'_> {
    type Error = diskann::error::Infallible;

    fn eval(&self, x: &[u8]) -> Result<f32, Self::Error> {
        assert_eq!(x.len(), std::mem::size_of_val(self.query));

        let mut sum = 0.0;
        for (q, x) in std::iter::zip(self.query.iter(), x.chunks_exact(4)) {
            let x: f32 = bytemuck::pod_read_unaligned(x);
            sum += q + x;
        }
        Ok(sum)
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
            Entry::Occupied(o) => panic!(
                "reference already contains a mapping for {:?}/{:?}: {:?}",
                logical,
                slot,
                o.get()
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

    /// Return an iterator over the contents of the reference.
    pub(super) fn iter(&self) -> impl Iterator<Item = (LogicalId, SlotId, &[f32])> {
        self.data
            .iter()
            .map(|(slot_id, (logical_id, data))| (*logical_id, *slot_id, &**data))
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
pub(super) fn test_prune(
    accessor: &mut dyn repr::Prune,
    distances: HashMap<(SlotId, SlotId), f32>,
    ids: &[SlotId],
    ctx: &dyn std::fmt::Display,
) {
    let mut visited = 0;
    for slot_id0 in ids.iter() {
        let v0 = {
            let v = accessor.get(slot_id0.value());
            if !distances.contains_key(&(*slot_id0, *slot_id0)) {
                assert!(v.is_none(), "failed on {}", slot_id0);
                continue;
            }
            v.unwrap()
        };

        for slot_id1 in ids.iter() {
            let v1 = {
                let v = accessor.get(slot_id1.value());
                if !distances.contains_key(&(*slot_id1, *slot_id1)) {
                    assert!(v.is_none(), "failed on {}", slot_id1);
                    continue;
                }
                v.unwrap()
            };

            let d = accessor.evaluate(v0, v1);
            let expected = distances[&(*slot_id0, *slot_id1)];
            assert_eq!(
                d, expected,
                "failed for {} x {} -- {}",
                slot_id0, slot_id1, ctx
            );

            visited += 1;
        }
    }

    assert_eq!(
        visited,
        distances.len(),
        "not all distances were visited -- {}",
        ctx
    );
}
