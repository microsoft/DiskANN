/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared infrastructure for implementing [`repr::ExpandBeam`] and [`repr::Prune`] on top
//! of [`crate::store::packed::Packed`].

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, error::IntoANNResult, neighbor::Neighbor};

use crate::{
    num::IdLimit,
    prefetch::{self, Prefetch},
    repr, store,
};

use super::{OutOfBounds, RawDistance, RawQueryDistance};

////////////////
// ExpandBeam //
////////////////

/// A [`repr::ExpandBeam`] implementation for the packed store.
///
/// Implementations can choose a [`RawQueryDistance`] and a [`Prefetch`] implementation.
/// The distance implementation provided must work with slices of the length provided by
/// `reader`.
///
/// At this moment, unsafe code may *not* rely on the assumption that only slices of length
/// [`store::packed::Reader::bytes`] will be provided, but this should always be the case.
///
/// # Prefetching
///
/// Prefetching is done in two independent streams: tags and data. Readability tags are
/// stored densely as a bit vector and as such are much more likely to be found in
/// the CPU's upper-level caches. Tag and data prefetch lookaheads can be tuned independently.
#[derive(Debug)]
pub(in crate::repr) struct ExpandBeam<'a, D, P> {
    /// The data reader.
    reader: store::packed::Reader<'a>,

    /// The distance computation. Ensure this is only provided with slices of length
    /// `reader.bytes()`.
    distance: D,

    /// This must be compatible with `reader.bytes()` and is checked in [`Self::new`].
    prefetch: prefetch::Checked<P>,

    /// How many expand-beam iterations ahead to prefetch the readability tags.
    tag_lookahead: Option<NonZeroUsize>,

    /// How many expand-beam iterations ahead to prefetch the data cache lines.
    data_lookahead: Option<NonZeroUsize>,
}

impl<'a, D, P> ExpandBeam<'a, D, P> {
    /// Create a new [`ExpandBeam`] object.
    ///
    /// Callers may generally assume that `distance` is provided with slices of length
    /// `reader.bytes()`, but unsafe code may not rely on this.
    ///
    /// # Panics
    ///
    /// Panics if `prefetch` is not compatible with [`store::packed::Reader::bytes`].
    pub(in crate::repr) fn new(
        reader: store::packed::Reader<'a>,
        distance: D,
        prefetch: P,
        tag_lookahead: Option<NonZeroUsize>,
        data_lookahead: Option<NonZeroUsize>,
    ) -> Self
    where
        P: Prefetch,
    {
        #[expect(
            clippy::expect_used,
            reason = "internal APIs should only provide valid prefetchers"
        )]
        let prefetch = prefetch::Checked::new(prefetch, reader.bytes())
            .expect("internal APIs should only provide valid prefetchers");

        Self {
            reader,
            distance,
            prefetch,
            tag_lookahead,
            data_lookahead,
        }
    }

    /// Wrap `self` in a `Box`.
    pub(in crate::repr) fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
}

// SAFETY: Our implementation of `repr::ExpandBeam::id_limit` is consistent with our
// `repr::ExpandBeam::expand_beam` requirements. They are both dependent on
// `packed::Reader`'s internal bounds.
unsafe impl<D, P> repr::ExpandBeam for ExpandBeam<'_, D, P>
where
    P: Prefetch,
    D: RawQueryDistance,
{
    fn id_limit(&self) -> IdLimit {
        self.reader.id_limit()
    }

    fn evaluate(&self, i: u32) -> ANNResult<Option<f32>> {
        if !self.reader.is_in_bounds(i) {
            Err(ANNError::new(OutOfBounds::new(i)))
        } else {
            // SAFETY: We have checked that `i` is in-bounds.
            match unsafe { self.reader.read_in_bounds(i) } {
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
        let tag_lookahead = self.tag_lookahead.map(|l| l.get()).unwrap_or(0).min(len);
        let data_lookahead = self.data_lookahead.map(|l| l.get()).unwrap_or(0).min(len);

        for j in list.iter().take(tag_lookahead) {
            // SAFETY: Caller guarantees that all items in `list` are less than `self.id_limit()`,
            // so this entry is in-bounds.
            unsafe { self.reader.prefetch_tag(*j) };
        }

        for k in list.iter().take(data_lookahead) {
            // SAFETY: Caller guarantees that all items in `list` are less than `self.id_limit()`,
            // so this entry is in-bounds.
            //
            // Further, the returned `RawSlice` ia guaranteed to point to valid memory,
            // so prefetching is safe.
            unsafe {
                let raw = self.reader.read_raw_unchecked(*k);
                self.prefetch.prefetch(raw.as_ptr(), raw.len());
            }
        }

        // Disable prefetching if the lookahead is 0.
        let maybe_disable = |lookahead| if lookahead == 0 { len } else { lookahead };

        let mut j = maybe_disable(tag_lookahead);
        let mut k = maybe_disable(data_lookahead);

        let mut processed = 0;
        for &i in list.iter() {
            if j != len {
                // SAFETY: `j < len`, so accessing `list` is in-bounds.
                //
                // Caller guarantees that all items in `list` are less than `self.id_limit()`,
                // so this entry is in-bounds.
                unsafe { self.reader.prefetch_tag(*list.get_unchecked(j)) };
                j += 1;
            }

            if k != len {
                // SAFETY: `j < len`, so accessing `list` is in-bounds.
                //
                // Caller guarantees that all items in `list` are less than `self.id_limit()`,
                // so this entry is in-bounds.
                //
                // Further, the returned `RawSlice` ia guaranteed to point to valid memory,
                // so prefetching is safe.
                unsafe {
                    let raw = self.reader.read_raw_unchecked(*list.get_unchecked(k));
                    self.prefetch.prefetch(raw.as_ptr(), raw.len());
                }
                k += 1;
            }

            // SAFETY: Caller asserts that `i` is in-bounds.
            if let Some(data) = unsafe { self.reader.read_in_bounds(i) } {
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
    /// The reader into the packed store.
    reader: store::packed::Reader<'a>,

    /// The distance computation. Ensure this is only provided with slices of length
    /// `reader.bytes()`.
    distance: D,
}

impl<'a, D> Prune<'a, D> {
    /// Construct a new [`Prune`], using `distance` to compute distances for the raw data
    /// retrieved from `reader`.
    ///
    /// Callers may generally assume that only slices of length
    /// [`store::packed::Reader::bytes`] will be passed to `distance`. However, unsafe
    /// code may not rely on this.
    pub(in crate::repr) fn new(reader: store::packed::Reader<'a>, distance: D) -> Self {
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
        self.reader.read(id)
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

    use diskann::utils::IntoUsize;
    use hashbrown::HashMap;
    use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

    use crate::{
        num::{Bytes, IdLimit, LogicalId, MaxDegree, SlotId},
        repr::test::{Reference, TestDistance, TestQueryDistance},
        store::packed::Packed,
    };

    /// Create a [`store::Store`] with the given `IdLimit` and space to store `dim` values
    /// of type `f32` per slot.
    ///
    /// Internal IDs are sampled without replacement across the entire ID space, allowing
    /// sparse stores to exercise tag prefetching across multiple cache lines.
    ///
    /// Then, `fill_to` entries with logical IDs `0..fill_to` will be inserted, each with
    /// its logical ID broadcast across all `dim` entries. These points will be published
    /// with [`store::Exclusive::publish`].
    ///
    /// Finally, `frozen` entries with logical IDs `(usize::MAX - frozen + 1)..=usize::MAX`
    /// with values `-(0..frozen)` broadcast across dimensions will be inserted. These points
    /// will be frozen with [`store::Exclusive::freeze`].
    ///
    /// A [`Reference`] will be returned containing all inserted points.
    fn test_store(
        id_limit: IdLimit,
        dim: usize,
        fill_to: usize,
        frozen: usize,
        rng: &mut impl Rng,
    ) -> (store::Store<Packed>, Reference) {
        assert!(fill_to + frozen <= id_limit.value().into_usize());

        let slots = rand::seq::index::sample(rng, id_limit.value().into_usize(), fill_to + frozen)
            .into_vec();

        let (published_slots, frozen_slots) = slots.split_at(fill_to);

        let bytes = Bytes::size_of::<f32>().checked_mul(dim).unwrap();
        let store = store::Store::new(
            store::Layout::new(id_limit, MaxDegree::new(0)),
            store::Config::__exhaustive(
                NonZeroUsize::new(4).unwrap(),
                NonZeroU32::new(16).unwrap(),
            ),
            Packed::config(bytes),
        )
        .unwrap();

        let mut reference = Reference::new(dim);
        let mut data = vec![0.0; dim];
        for (i, &slot) in (0..fill_to).rev().zip(published_slots) {
            data.fill(i as f32);
            let mut exclusive = store.slot(u32::try_from(slot).unwrap()).unwrap();
            exclusive
                .data()
                .as_mut_slice()
                .copy_from_slice(bytemuck::must_cast_slice(&data));
            let slot = exclusive.slot();
            reference.insert(LogicalId(i), SlotId(slot), &data);
            exclusive.publish();
        }

        for (i, &slot) in frozen_slots.iter().enumerate() {
            data.fill(-(i as f32));
            let mut exclusive = store.slot(u32::try_from(slot).unwrap()).unwrap();
            exclusive
                .data()
                .as_mut_slice()
                .copy_from_slice(bytemuck::must_cast_slice(&data));
            let slot = exclusive.slot();
            reference.insert(LogicalId(usize::MAX - i), SlotId(slot), &data);
            exclusive.freeze();
        }

        assert_eq!(reference.iter().count(), fill_to + frozen);
        assert!(
            reference
                .iter()
                .all(|(_, slot, _)| id_limit.is_in_bounds(slot.value()))
        );

        (store, reference)
    }

    /// Assemble a list of IDs to test for `expand_beam` and `prune`.
    ///
    /// This consists of:
    ///
    /// * All valid slot IDs in `reference`.
    /// * Slot IDs `0` and `id_limit - 1`.
    /// * The direct IDs specified by `lifecycle_ids`.
    /// * `select` randomly selected IDs in `[0, id_limit)`.
    ///
    /// All returned IDs will be unique and in a random order.
    fn test_ids(
        reference: &Reference,
        lifecycle_ids: &[u32],
        id_limit: IdLimit,
        select: usize,
        rng: &mut impl Rng,
    ) -> Vec<SlotId> {
        assert!(
            select <= id_limit.value().into_usize(),
            "cannot randomly select more than {id_limit} ids, tried to select {select}"
        );

        let mut ids: Vec<_> = reference.iter().map(|(_, slot, _)| slot).collect();
        ids.extend(lifecycle_ids.iter().copied().map(SlotId));
        ids.extend([SlotId(0), SlotId(id_limit.value() - 1)]);
        ids.extend(
            rand::seq::index::sample(rng, id_limit.value().into_usize(), select)
                .into_iter()
                .map(|id| SlotId(u32::try_from(id).unwrap())),
        );
        ids.sort_unstable_by_key(|id| id.value());
        ids.dedup();
        ids.shuffle(rng);
        ids
    }

    fn sample_lengths(max: usize) -> Vec<usize> {
        let mut l = vec![0, 1, 2, 3, 4, 5, 6, max];
        l.extend((0..max).step_by(5));
        l.retain(|i| *i <= max);
        l.sort_unstable();
        l.dedup();
        l
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
        let mut rng = StdRng::seed_from_u64(0x7e67f14b840c2a91);
        let id_limit = IdLimit::new(4096);
        let (store, reference) = test_store(id_limit, dim, 10, 2, &mut rng);

        assert_eq!(store.id_limit(), id_limit);

        // Lookaheads to try.
        let lookaheads: &[Option<NonZeroUsize>] =
            &[None, NonZeroUsize::new(5), NonZeroUsize::new(100)];

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
        for tag_lookahead in lookaheads {
            for data_lookahead in lookaheads {
                // Retired
                let g0 = store.acquire().unwrap();
                let g1 = store.acquire().unwrap();

                // Abort
                let g2 = store.acquire().unwrap();
                let g3 = store.acquire().unwrap();

                // Held
                let g4 = store.acquire().unwrap();
                let g5 = store.acquire().unwrap();

                let ids = test_ids(
                    &reference,
                    &[
                        g0.slot(),
                        g1.slot(),
                        g2.slot(),
                        g3.slot(),
                        g4.slot(),
                        g5.slot(),
                    ],
                    id_limit,
                    12,
                    &mut rng,
                );

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
                        .read_guard(|packed, guard| packed.reader(guard))
                        .unwrap(),
                    TestQueryDistance::new(&query),
                    prefetch::Loop::new(),
                    *tag_lookahead,
                    *data_lookahead,
                );

                assert_eq!(<_ as repr::ExpandBeam>::id_limit(&expand), id_limit);

                // Sweep prefixes of the sampled IDs with `distances` serving as ground truth.
                for len in sample_lengths(ids.len()) {
                    repr::test::test_expand_beam(
                        &expand,
                        id_limit,
                        distances.clone(),
                        &ids[..len],
                        &format_args!("packed::ExpandBeam sampled prefix {}", len),
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

    // This tests that we properly emit prefetches for data across multiple cache lines.
    #[test]
    fn test_expand_beam_one_overlap() {
        test_expand_beam_inner(31);
    }

    ///////////
    // Prune //
    ///////////

    /// A Miri-friendly test for [`Prune`].
    #[test]
    fn test_prune() {
        let mut rng = StdRng::seed_from_u64(0x5fa43bd762819ce0);
        let id_limit = IdLimit::new(4096);
        let dim = 1;
        let (store, reference) = test_store(id_limit, dim, 10, 2, &mut rng);

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

        let mut candidates = test_ids(
            &reference,
            &[g0.slot(), g1.slot(), g2.slot(), g3.slot()],
            id_limit,
            12,
            &mut rng,
        );
        candidates.extend((id_limit.value()..=(id_limit.value() + 5)).map(SlotId));
        candidates.shuffle(&mut rng);

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
                .read_guard(|packed, guard| packed.reader(guard))
                .unwrap(),
            TestDistance,
        );

        // Note that we emit reads above the `IdLimit`, which we expect to be silently
        // rejected.
        for len in sample_lengths(candidates.len()) {
            let mut ids = candidates[..len].to_vec();

            // Only keep distances whose two IDs are present in this prefix.
            let distances_for_test: HashMap<_, _> = distances
                .iter()
                .filter_map(|(k, v)| {
                    if ids.contains(&k.0) && ids.contains(&k.1) {
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
                &format_args!("packed::Prune - sampled prefix {}", len),
            );

            ids.reverse();
            repr::test::test_prune(
                &mut prune,
                distances_for_test,
                &ids,
                &format_args!("packed::Prune - reversed sampled prefix {}", len),
            );
        }

        // Drop the guards after holding them in limbo during the test.
        drop(g2);
        drop(g3);
    }
}
