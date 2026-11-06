/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared infrastructure for implementing [`repr::ExpandBeam`] and [`repr::Prune`] on top
//! of the [`crate::store::intrusive::Intrusive`].

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, error::IntoANNResult, neighbor::Neighbor, utils::IntoUsize};

use crate::{
    num::IdLimit,
    prefetch::{self, Prefetch},
    repr, store,
};

use super::{OutOfBounds, RawDistance, RawQueryDistance};

////////////////
// ExpandBeam //
////////////////

/// A [`repr::ExpandBeam`] implementation for the invasive memory store.
///
/// Implementations can choose a [`RawQueryDistance`] and a [`Prefetch`] implementation.
/// The distance implementation provided must work with slices of the length provided by
/// `reader`.
///
/// At this moment, unsafe code may *not* rely on the assumption that only slices of length
/// [`store::intrusive::Reader::bytes`] will be provided, but this should always be the case.
#[derive(Debug)]
pub(in crate::repr) struct ExpandBeam<'a, D, P> {
    /// The data reader.
    reader: store::intrusive::Reader<'a>,

    /// The distance computation. Ensure this is only provided with slices of length
    /// `reader.bytes()`.
    distance: D,

    /// Prefetcher. This must be compatible with `reader.prefetch_bytes()` and is checked
    /// in [`Self::new`].
    prefetch: prefetch::Checked<P>,

    /// The number of expand-beam iterators to fetch ahead.
    lookahead: Option<NonZeroUsize>,
}

impl<'a, D, P> ExpandBeam<'a, D, P> {
    /// Create a new [`ExpandBeam`] object.
    ///
    /// Callers may generally assume that `distance` being provided with slices with length
    /// `reader.bytes()`, but unsafe code may not yet use this.
    ///
    /// # Panics
    ///
    /// Panics if `prefetch` is not compatible with [`store::intrusive::Reader::prefetch_bytes`].
    pub(in crate::repr) fn new(
        reader: store::intrusive::Reader<'a>,
        distance: D,
        prefetch: P,
        lookahead: Option<NonZeroUsize>,
    ) -> Self
    where
        P: Prefetch,
    {
        #[expect(
            clippy::expect_used,
            reason = "internal APIs should only provide valid prefetchers"
        )]
        let prefetch = prefetch::Checked::new(prefetch, reader.prefetch_bytes())
            .expect("internal APIs should only provide valid prefetchers");

        Self {
            reader,
            distance,
            prefetch,
            lookahead,
        }
    }

    /// Wrap `self` in a `Box`.
    pub(in crate::repr) fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

// SAFETY: Our implementation of `repr::ExpandBeam::id_limit` is consistent with our
// `repr::ExpandBeam::expand_beam` requirements. They are both dependent on
// `intrusive::Reader`'s internal bounds.
unsafe impl<D, P> repr::ExpandBeam for ExpandBeam<'_, D, P>
where
    P: Prefetch,
    D: RawQueryDistance,
{
    fn id_limit(&self) -> IdLimit {
        self.reader.id_limit()
    }

    fn evaluate(&self, i: u32) -> ANNResult<Option<f32>> {
        if !self.reader.is_in_bounds(i.into_usize()) {
            Err(ANNError::new(OutOfBounds::new(i)))
        } else {
            // SAFETY: We have checked that `i` is in-bounds.
            match unsafe { self.reader.read_in_bounds(i.into_usize()) } {
                Some(data) => {
                    let distance = self.distance.eval(data).into_ann_result()?;
                    Ok(Some(distance))
                }
                None => Ok(None),
            }
        }
    }

    unsafe fn expand_beam(&self, list: &[u32], buffer: &mut [Neighbor<u32>]) -> ANNResult<usize> {
        debug_assert!(buffer.len() >= list.len());

        let len = list.len();
        let lookahead = self.lookahead.map(|l| l.get()).unwrap_or(0).min(len);

        for j in list.iter().take(lookahead) {
            // SAFETY: The in-bounds constraint is assured by the caller, both for `j` as well
            // as the validity of the prefetch bounds.
            //
            // We validated `self.prefetch` with `self.reader.prefetch_bytes()` upon construction.
            //
            // The `RawSlice` and tag pointer are only used for prefetching.
            unsafe {
                let (raw, tag) = self.reader.read_for_prefetch(j.into_usize());

                prefetch::prefetch_cacheline(tag);
                self.prefetch.prefetch(raw.as_ptr(), raw.len());
            }
        }

        // Disable prefetching if the lookahead is 0.
        let mut j = if lookahead == 0 { len } else { lookahead };
        let mut processed = 0;
        for &i in list.iter() {
            if j != len {
                // SAFETY: The in-bounds constraint is assured by the caller, both for `j` as
                // well as the validity of the prefetch bounds.
                //
                // We validated `self.prefetch` with `self.reader.prefetch_bytes()` upon
                // construction.
                //
                // The `RawSlice` and tag pointer are only used for prefetching.
                unsafe {
                    let (raw, tag) = self
                        .reader
                        .read_for_prefetch(list.get_unchecked(j).into_usize());
                    prefetch::prefetch_cacheline(tag);
                    self.prefetch.prefetch(raw.as_ptr(), raw.len());
                }
                j += 1;
            }

            // SAFETY: Caller asserts that `i` is in-bounds.
            if let Some(data) = unsafe { self.reader.read_in_bounds(i.into_usize()) } {
                let distance = self.distance.eval(data).into_ann_result()?;

                // SAFETY: Inherited from caller.
                *unsafe { buffer.get_unchecked_mut(processed) } = Neighbor::new(i, distance);
                processed += 1;
            }
        }

        Ok(processed)
    }
}

///////////
// Prune //
///////////

/// An implementation of [`repr::Prune`].
///
/// Implementations must provide a [`RawDistance`].
#[derive(Debug)]
pub(in crate::repr) struct Prune<'a, D> {
    /// The reader into the intrusive store.
    reader: store::intrusive::Reader<'a>,

    /// The distance computation. Ensure this is only provided with slices of length
    /// `reader.bytes()`.
    distance: D,
}

impl<'a, D> Prune<'a, D> {
    /// Construct a new [`Prune`], using `distance` to compute distances for the raw data
    /// retrieved from `reader`.
    ///
    /// Callers may generally assume that only slices of length
    /// [`store::intrusive::Reader::bytes`] will be passed to `distance`. However, unsafe
    /// code may not rely on this.
    pub(in crate::repr) fn new(reader: store::intrusive::Reader<'a>, distance: D) -> Self {
        Self { reader, distance }
    }

    /// Wrap `self` in a `Box`.
    pub(in crate::repr) fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

// SAFETY: The embedded pointers are equivalent to storing `&[u8]`, which is `Send`.
unsafe impl<D> Send for Prune<'_, D> where D: Send {}
// SAFETY: The embedded pointers are equivalent to storing `&[u8]`, which is `Sync`.
unsafe impl<D> Sync for Prune<'_, D> where D: Sync {}

impl<D> repr::Prune for Prune<'_, D>
where
    D: RawDistance,
{
    fn get(&self, id: u32) -> Option<&[u8]> {
        self.reader.read(id.into_usize())
    }

    fn evaluate(&self, a: &[u8], b: &[u8]) -> f32 {
        #[expect(
            clippy::expect_used,
            reason = "diskann currently does not support fallible prune"
        )]
        self.distance
            .eval(a, b)
            .expect("diskann currently does not support fallible prune")
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::num::{NonZeroU32, NonZeroUsize};

    use hashbrown::HashMap;

    use crate::{
        num::{Bytes, IdLimit, LogicalId, MaxDegree, SlotId},
        repr::test::{Reference, TestDistance, TestQueryDistance},
        store::intrusive::Intrusive,
    };

    /// Create a [`store::Store`] with the given `IdLimit` and space to store `dim` values
    /// of type `f32` per slot.
    ///
    /// Then, `fill_to` entries with logical ids `0..fill_to` will be inserted, each with
    /// its logical id broadcast across all `dim` entries. These points will be published
    /// with [`store::Exclusive::publish`].
    ///
    /// Finally, `frozen` entries with logical ids `(usize::MAX - frozen + 1)..=usize::MAX`
    /// with values `-(0..frozen)` broadcast across dimensions will be inserted. These points
    /// will be frozen with [`store::Exclusive::freeze`].
    ///
    /// A [`Reference`] will be returned containing all inserted points.
    fn test_store(
        id_limit: IdLimit,
        dim: usize,
        fill_to: usize,
        frozen: usize,
    ) -> (store::Store<Intrusive>, Reference) {
        assert!(fill_to + frozen <= id_limit.value().into_usize());

        let bytes = Bytes::size_of::<f32>().checked_mul(dim).unwrap();
        let store = store::Store::new(
            store::Layout::new(id_limit, MaxDegree::new(0)),
            store::Config::__exhaustive(
                NonZeroUsize::new(4).unwrap(),
                NonZeroU32::new(16).unwrap(),
            ),
            Intrusive::config(bytes),
        )
        .unwrap();

        let mut reference = Reference::new(dim);
        let mut data = vec![0.0; dim];
        for i in (0..fill_to).rev() {
            data.fill(i as f32);
            let mut exclusive = store.acquire().unwrap();
            exclusive
                .data()
                .as_mut_slice()
                .copy_from_slice(bytemuck::must_cast_slice(&data));
            let slot = exclusive.slot();
            reference.insert(LogicalId(i), SlotId(slot), &data);
            exclusive.publish();
        }

        for i in 0..frozen {
            data.fill(-(i as f32));
            let mut exclusive = store.acquire().unwrap();
            exclusive
                .data()
                .as_mut_slice()
                .copy_from_slice(bytemuck::must_cast_slice(&data));
            let slot = exclusive.slot();
            reference.insert(LogicalId(usize::MAX - i), SlotId(slot), &data);
            exclusive.freeze();
        }

        (store, reference)
    }

    /////////////////
    // Expand Beam //
    /////////////////

    /// A Miri-friendly test for [`ExpandBeam`].
    ///
    /// This test covers the following:
    ///
    /// 1. Prefetches are in-bounds for all tested lookaheads.
    /// 2. [`ExpandBeam`] doesn't lie about its [`IdLimit`].
    /// 3. [`ExpandBeam`]'s methods are internally consistent with each other and
    ///    consistent with the reference data for item readability.
    /// 4. [`ExpandBeam::expand_beam`] preserves input order and visits every item in the
    ///    input list.
    fn test_expand_beam_inner(dim: usize) {
        let id_limit = IdLimit::new(22);
        let (store, reference) = test_store(id_limit, dim, 10, 2);

        assert_eq!(store.id_limit(), id_limit);

        // Lookaheads to try.
        let lookaheads: &[Option<NonZeroUsize>] = &[
            None,
            NonZeroUsize::new(1),
            NonZeroUsize::new(5),
            NonZeroUsize::new(100),
        ];

        // Compute the distance between the query and all internal data points.
        let query = vec![-1.0f32; dim];
        let distances: HashMap<_, _> = reference
            .iter()
            .map(|(_, slot, data)| {
                let distance: f32 = std::iter::zip(data.iter(), query.iter())
                    .map(|(d, q)| d + q)
                    .sum();
                (slot, distance)
            })
            .collect();

        // This is the main loop for testing `ExpandBeam`.
        //
        // We do several things.
        //
        // 1. We insert two additional IDs but hold their `Exclusive` without publishing.
        //    This tests that items remain unreadable until they are published.
        //
        // 2. We attempt to insert two more points but abort to check that they remain
        //    unreadable.
        //
        // 3. We publish two new points and immediately retire them.
        //    This tests that we correctly make these points unreadable.
        //
        for lookahead in lookaheads {
            // Retired
            let g0 = store.acquire().unwrap();
            let g1 = store.acquire().unwrap();

            // Abort
            let g2 = store.acquire().unwrap();
            let g3 = store.acquire().unwrap();

            // Held
            let g4 = store.acquire().unwrap();
            let g5 = store.acquire().unwrap();

            // Do the submit + retire loop.
            {
                let g0_id = g0.slot();
                g0.publish();
                store.retire(g0_id.into_usize()).unwrap();
            }

            {
                let g1_id = g1.slot();
                g1.publish();
                store.retire(g1_id.into_usize()).unwrap();
            }

            // Abort
            std::mem::drop(g2);
            std::mem::drop(g3);

            let expand = ExpandBeam::new(
                store
                    .read_guard(|intrusive, guard| intrusive.reader(guard))
                    .unwrap(),
                TestQueryDistance::new(&query),
                prefetch::Loop::new(),
                *lookahead,
            );

            assert_eq!(<_ as repr::ExpandBeam>::id_limit(&expand), id_limit);

            let mut list = Vec::<SlotId>::new();

            // Use triangular indexing from `0..id_limit` with `distances` serving as the
            // ground truth.
            //
            // Note that we purposely make `list` extra long with redundant indices to help
            // catch indexing bugs inside `ExpandBeam`.
            for i in 0..=id_limit.value() {
                list.clear();
                list.extend((0..i).rev().map(SlotId));
                list.extend((0..i).map(SlotId));

                repr::test::test_expand_beam(
                    &expand,
                    id_limit,
                    distances.clone(),
                    &list,
                    &format_args!("intrusive::ExpandBeam 0..{}", i),
                )
            }

            assert!(
                <_ as repr::ExpandBeam>::evaluate(&expand, id_limit.value()).is_err(),
                "`ExpandBeam::evaluate` should catch out-of-bounds errors",
            );

            // Ensure we hold onto `g4` and `g5` for the duration of the above check.
            drop(g4);
            drop(g5);
        }
    }

    // Even though zero-dimensional data doesn't make sense, we make sure that we can still
    // handle it safely.
    #[test]
    fn test_expand_beam_zero() {
        test_expand_beam_inner(0);
    }

    // Low dimensions with all data living on a single cache line.
    #[test]
    fn test_expand_beam_scalar() {
        test_expand_beam_inner(1);
    }

    // This tests that if we emit prefetch instructions disjoint from the tag that they
    // are in-bounds.
    //
    // A dimension of 16 means that all the data sits on one cache line. The tag sits on
    // another one.
    #[test]
    fn test_expand_beam_one_disjoint() {
        test_expand_beam_inner(16);
    }

    // This tests that if we emit prefetch instructions disjoint from the tag that they
    // are in-bounds.
    //
    // A dimension of 31 means that the data spans two cache lines, but the second cache
    // line is shared with the tag.
    #[test]
    fn test_expand_beam_one_overlap() {
        test_expand_beam_inner(31);
    }

    // This tests that if we emit prefetch instructions disjoint from the tag that they
    // are in-bounds.
    //
    // A dimension of 32 means that all the data sits on two cache lines. The tag sits on
    // another one.
    #[test]
    fn test_expand_beam_two_disjoint() {
        test_expand_beam_inner(32);
    }

    ///////////
    // Prune //
    ///////////

    /// A Miri-friendly test for `Prune`.
    #[test]
    fn test_prune() {
        let id_limit = IdLimit::new(22);
        let dim = 1;
        let (store, reference) = test_store(id_limit, dim, 10, 2);

        // Populate distances.
        let mut distances = HashMap::new();
        for (_, slot0, data0) in reference.iter() {
            for (_, slot1, data1) in reference.iter() {
                let distance: f32 = std::iter::zip(data0.iter(), data1.iter())
                    .map(|(l, r)| l + r)
                    .sum();

                distances.insert((slot0, slot1), distance);
            }
        }

        // We do several things.
        //
        // 1. We insert two additional IDs but hold their guards without publishing.
        //    This tests that items remain unreadable until they are published.
        //
        // 2. We publish two new points and immediately retire them.
        //    This tests that we correctly make these points unreadable.
        let g0 = store.acquire().unwrap();
        let g1 = store.acquire().unwrap();
        let g2 = store.acquire().unwrap();
        let g3 = store.acquire().unwrap();

        {
            let g0_id = g0.slot();
            g0.publish();
            store.retire(g0_id.into_usize()).unwrap();
        }

        {
            let g1_id = g1.slot();
            g1.publish();
            store.retire(g1_id.into_usize()).unwrap();
        }

        let mut prune = Prune::new(
            store
                .read_guard(|intrusive, guard| intrusive.reader(guard))
                .unwrap(),
            TestDistance,
        );

        // Note that we emit reads above the `IdLimit`, which we expect to be silently
        // rejected.
        for i in 0..=(id_limit.value() + 5) {
            let mut ids: Vec<SlotId> = (0..i).map(SlotId).collect();

            // Take the collection of distances where both entries in the tuple are within
            // the bounds.
            let distances_for_test: HashMap<_, _> = distances
                .iter()
                .filter_map(|(k, v)| {
                    if k.0.value() < i && k.1.value() < i {
                        Some((*k, *v))
                    } else {
                        None
                    }
                })
                .collect();

            repr::test::test_prune(
                &mut prune,
                distances_for_test.clone(),
                &ids,
                &format_args!("intrusive::Prune - ids 0..{}", i),
            );

            ids.reverse();
            repr::test::test_prune(
                &mut prune,
                distances_for_test,
                &ids,
                &format_args!("intrusive::Prune - ids {}..0", i),
            );
        }

        // Drop the guards after holding them in limbo during the test.
        drop(g2);
        drop(g3);
    }
}
