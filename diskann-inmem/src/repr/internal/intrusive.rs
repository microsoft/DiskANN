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

    /// Prefetcher. This must be compatible with `reader.bytes_plus_tag()` and is checked
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
    /// Panics if `prefetch` is not compatible with [`store::intrusive::Reader::bytes_plus_tag`].
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
        let prefetch = prefetch::Checked::new(prefetch, reader.bytes_plus_tag())
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
            // We validated `self.prefetch` with `self.reader.bytes_with_tag()` upon construction.
            //
            // We do not materialize the `RawSlice` as a reference.
            unsafe {
                let raw = self.reader.read_raw_unchecked(j.into_usize());
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
                // We validated `self.prefetch` with `self.reader.bytes_with_tag()` upon
                // construction.
                //
                // We do not materialize the `RawSlice` as a reference.
                unsafe {
                    let raw = self
                        .reader
                        .read_raw_unchecked(list.get_unchecked(j).into_usize());
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

    /// Buffered pointers to slices in `reader`. These
    ///
    /// * Must come from slices yielded by `reader`.
    ///
    /// * Must have a length of exactly `reader.bytes()`.
    ///
    /// * Have no lifetime because they need to be outlived `reader`. We only materialize
    ///   them as slices in internal methods taking `&[mut] self` where we know the reader
    ///   is valid.
    buffer: Vec<*const u8>,
}

impl<'a, D> Prune<'a, D> {
    /// Construct a new [`Prune`], using `distance` to compute distances for the raw data
    /// retrieved from `reader`.
    ///
    /// Callers may generally assume that only slices of length
    /// [`store::intrusive::Reader::bytes`] will be passed to `distance`. However, unsafe
    /// code may not rely on this.
    pub(in crate::repr) fn new(reader: store::intrusive::Reader<'a>, distance: D) -> Self {
        Self {
            reader,
            distance,
            buffer: Vec::new(),
        }
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
    fn prepare(
        &mut self,
        items: hashbrown::hash_map::IterMut<'_, u32, Option<repr::PruneKey>>,
    ) -> ANNResult<usize> {
        let mut counter = repr::PruneKey::counter();
        self.buffer.clear();
        self.buffer.reserve(items.len());

        for (id, key) in items {
            if let Some(v) = self.reader.read(id.into_usize()) {
                self.buffer.push(v.as_ptr());

                *key = Some(counter);

                // Potential overflow issue - but it's exceedingly unlikely that
                // someone will provide a prune list exceeding `u16::MAX`.
                //
                // In addition, `diskann` limits this bound as well.
                counter = counter.increment()?;
            }
        }

        Ok(counter.index())
    }

    fn evaluate(&self, a: repr::PruneKey, b: repr::PruneKey) -> f32 {
        let len = self.reader.bytes().value();

        // SAFETY: The only way a pointer ends up in `self.buffer` is from `self.prepare`,
        // where they are obtained from slices returned from `self.reader`.
        //
        // Further, `self.reader` must be retained for the lifetime of `self`. Thus, they
        // point to valid data of length `len` and are protected from concurrent mutation.
        let a = unsafe { std::slice::from_raw_parts(self.buffer[a.index()], len) };

        // SAFETY: See above.
        let b = unsafe { std::slice::from_raw_parts(self.buffer[b.index()], len) };

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

    // /// A Miri-friendly test for [`ExpandBeam`].
    // ///
    // /// This test covers the following:
    // ///
    // /// 1. Prefetches are in-bounds for all lookaheads.
    // /// 2. [`ExpandBeam`] doesn't lie about its [`IdLimit`].
    // /// 3. [`ExpandBeam`]'s methods are internally consistent with each other and
    // ///    consistent with the parent [`Full`] for item readability.
    // /// 4. [`ExpandBeam::expand_beam`] preserves input order and visits every item in the
    // ///    input list.
    // #[test]
    // fn test_expand_beam() {
    //     let capacity = Capacity::new(20);
    //     let id_limit = IdLimit::new(22);

    //     let (mut full, mut points) = test_full(capacity);

    //     assert_eq!(<_ as repr::Representation>::capacity(&full), capacity);
    //     assert_eq!(<_ as repr::Representation>::id_limit(&full), id_limit);

    //     let mut available: HashSet<u32> = (0..capacity.value()).map(|i| i as u32).collect();

    //     // Insert the values 0 to 10.
    //     for i in 0u32..10 {
    //         let guard = <_ as repr::Set<&[f32]>>::set(&full, &[i as f32]).unwrap();

    //         let id = <_ as repr::Guard>::id(&guard);

    //         assert!(
    //             available.remove(&id),
    //             "insertion should return available slots",
    //         );

    //         assert!(
    //             points.insert(id, i as f32).is_none(),
    //             "insertion should not repeat",
    //         );

    //         <_ as repr::Guard>::publish(guard);
    //     }

    //     // Lookaheads to try.
    //     let lookaheads: &[Option<NonZeroUsize>] = &[
    //         None,
    //         NonZeroUsize::new(1),
    //         NonZeroUsize::new(2),
    //         NonZeroUsize::new(5),
    //         NonZeroUsize::new(10),
    //         NonZeroUsize::new(100),
    //     ];

    //     // This is the main loop for testing `ExpandBeam`.
    //     //
    //     // We do several things.
    //     //
    //     // 1. We insert two additional IDs but hold their guards without publishing.
    //     //    This tests that items remain unreadable until they are published.
    //     //
    //     // 2. We publish two new points and immediately retire them.
    //     //    This tests that we correctly make these points unreadable.
    //     for lookahead in lookaheads {
    //         full.lookahead = *lookahead;

    //         let g0 = <_ as repr::Set<&[f32]>>::set(&full, &[1000.0]).unwrap();
    //         let g1 = <_ as repr::Set<&[f32]>>::set(&full, &[2000.0]).unwrap();
    //         let g2 = <_ as repr::Set<&[f32]>>::set(&full, &[3000.0]).unwrap();
    //         let g3 = <_ as repr::Set<&[f32]>>::set(&full, &[4000.0]).unwrap();

    //         {
    //             let g0_id = <_ as repr::Guard>::id(&g0);
    //             <_ as repr::Guard>::publish(g0);
    //             <_ as repr::Representation>::retire(&full, g0_id).unwrap();
    //         }

    //         {
    //             let g1_id = <_ as repr::Guard>::id(&g1);
    //             <_ as repr::Guard>::publish(g1);
    //             <_ as repr::Representation>::retire(&full, g1_id).unwrap();
    //         }

    //         let query = -1.0f32;

    //         let into =
    //             IntoExpandBeam::new(&full, Calf::Borrowed(std::slice::from_ref(&query))).unwrap();

    //         let expand = into.into_expand_beam(repr::test::TestDistance, prefetch::Loop::new());

    //         assert_eq!(<_ as repr::ExpandBeam>::id_limit(&expand), id_limit);

    //         let mut buf = Vec::<Neighbor<u32>>::new();
    //         let mut list = Vec::<u32>::new();

    //         // Use triangular indexing from `0..id_limit` with `points` serving as the
    //         // groundtruth.
    //         //
    //         // Note that we purposely make `list` extra long with redundant indices to help
    //         // catch indexing bugs inside `ExpandBeam`.
    //         for i in 0..=id_limit.value() {
    //             list.clear();
    //             list.extend((0..i).rev());
    //             list.extend(0..i);

    //             buf.resize(list.len(), Default::default());

    //             let read = repr::safe_expand_beam(&expand, &list, &mut buf).unwrap();

    //             let expected: Vec<(u32, f32)> = list
    //                 .iter()
    //                 .copied()
    //                 .filter_map(|id| match points.get(&id) {
    //                     Some(point) => {
    //                         let expected = point + query;

    //                         assert!(
    //                             <_ as repr::Representation>::is_readable(&full, id).unwrap(),
    //                             "point should be readable"
    //                         );

    //                         assert_eq!(
    //                             <_ as repr::ExpandBeam>::evaluate(&expand, id).unwrap(),
    //                             Some(expected),
    //                             "readable points should return valid distances",
    //                         );

    //                         Some((id, expected))
    //                     }
    //                     None => {
    //                         assert!(
    //                             !<_ as repr::Representation>::is_readable(&full, id).unwrap(),
    //                             "points not yielded by ExpandBeam should be unreadable"
    //                         );

    //                         assert!(
    //                             <_ as repr::ExpandBeam>::evaluate(&expand, id)
    //                                 .unwrap()
    //                                 .is_none(),
    //                             "unreadable points should return `None` for their distance",
    //                         );

    //                         None
    //                     }
    //                 })
    //                 .collect();

    //             for i in 0..read {
    //                 assert_eq!(*buf[i].id(), expected[i].0, "i = {i}");
    //                 assert_eq!(*buf[i].distance(), expected[i].1, "i = {i}");
    //             }
    //         }

    //         assert!(
    //             <_ as repr::ExpandBeam>::evaluate(&expand, id_limit.value()).is_err(),
    //             "`ExpandBeam::evaluate` should catch out-of-bounds errors",
    //         );

    //         // Ensure we hold onto `g2` and `g3` for the duration of the above check.
    //         drop(g2);
    //         drop(g3);
    //     }
    // }

    // fn test_prune_inner(
    //     points: &HashMap<u32, f32>,
    //     prune: &mut repr::internal::intrusive::Prune<'_, repr::test::TestDistance>,
    //     ids: &[u32],
    // ) {
    //     let mut items: HashMap<u32, Option<repr::PruneKey>> =
    //         ids.iter().map(|id| (*id, None)).collect();

    //     let processed = <_ as repr::Prune>::prepare(prune, items.iter_mut()).unwrap();
    //     assert_eq!(processed, items.values().filter(|i| i.is_some()).count());

    //     // Ensure that `prepare` agrees with `points`.
    //     for (k, v) in items.iter() {
    //         match v {
    //             Some(_) => assert!(points.contains_key(k)),
    //             None => assert!(!points.contains_key(k)),
    //         }
    //     }

    //     fn filter((k, v): (&u32, &Option<repr::PruneKey>)) -> Option<(u32, repr::PruneKey)> {
    //         v.map(|v| (*k, v))
    //     }

    //     // Ensure that distances agree.
    //     for (k0, v0) in items.iter().filter_map(filter) {
    //         for (k1, v1) in items.iter().filter_map(filter) {
    //             // Manually implement `TestDistance`.
    //             let expected = points[&k0] + points[&k1];
    //             let got = <_ as repr::Prune>::evaluate(prune, v0, v1);
    //             assert_eq!(expected, got);
    //         }
    //     }
    // }

    // /// A Miri-friendly test for `Prune`.
    // #[test]
    // fn test_prune() {
    //     let capacity = Capacity::new(20);
    //     let id_limit = IdLimit::new(22);

    //     let (full, mut points) = test_full(capacity);

    //     assert_eq!(<_ as repr::Representation>::capacity(&full), capacity);
    //     assert_eq!(<_ as repr::Representation>::id_limit(&full), id_limit);

    //     let mut available: HashSet<u32> = (0..capacity.value()).map(|i| i as u32).collect();

    //     // Insert the values 0 to 10.
    //     for i in 0u32..10 {
    //         let guard = <_ as repr::Set<&[f32]>>::set(&full, &[i as f32]).unwrap();

    //         let id = <_ as repr::Guard>::id(&guard);

    //         assert!(
    //             available.remove(&id),
    //             "insertion should return available slots",
    //         );

    //         assert!(
    //             points.insert(id, i as f32).is_none(),
    //             "insertion should not repeat",
    //         );

    //         <_ as repr::Guard>::publish(guard);
    //     }

    //     // We do several things.
    //     //
    //     // 1. We insert two additional IDs but hold their guards without publishing.
    //     //    This tests that items remain unreadable until they are published.
    //     //
    //     // 2. We publish two new points and immediately retire them.
    //     //    This tests that we correctly make these points unreadable.
    //     let g0 = <_ as repr::Set<&[f32]>>::set(&full, &[1000.0]).unwrap();
    //     let g1 = <_ as repr::Set<&[f32]>>::set(&full, &[2000.0]).unwrap();
    //     let g2 = <_ as repr::Set<&[f32]>>::set(&full, &[3000.0]).unwrap();
    //     let g3 = <_ as repr::Set<&[f32]>>::set(&full, &[4000.0]).unwrap();

    //     {
    //         let g0_id = <_ as repr::Guard>::id(&g0);
    //         <_ as repr::Guard>::publish(g0);
    //         <_ as repr::Representation>::retire(&full, g0_id).unwrap();
    //     }

    //     {
    //         let g1_id = <_ as repr::Guard>::id(&g1);
    //         <_ as repr::Guard>::publish(g1);
    //         <_ as repr::Representation>::retire(&full, g1_id).unwrap();
    //     }

    //     let mut prune =
    //         repr::internal::intrusive::Prune::new(full.reader().unwrap(), repr::test::TestDistance);

    //     // Note that we emit reads above the `IdLimit`, which we expect to be silently
    //     // rejected.
    //     for i in 0..=(id_limit.value() + 5) {
    //         let mut ids: Vec<u32> = (0..i).collect();
    //         test_prune_inner(&points, &mut prune, &ids);

    //         ids.reverse();
    //         test_prune_inner(&points, &mut prune, &ids);
    //     }

    //     // Drop the guards - verifying that they are held in-limbo during the test.
    //     drop(g2);
    //     drop(g3);
    // }
}
