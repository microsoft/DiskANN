/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_vector::distance::{Distance, DistanceProvider};
use half::f16;

use crate::{
    counters::LocalCounters,
    epoch,
    num::Bytes,
    repr::{
        self,
        internal::{Calf, quantization::Metric},
    },
    store::{
        self,
        optional::Optional,
        simple::{self, Simple},
    },
};

/// Choose how data is going to be reranked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::repr) enum Rerank {
    /// No reranking will be performed and no space for higher precision vectors will be
    /// allocated.
    None,

    /// Use 16-bit floating point numbers to store the higher precision representation.
    /// These will be used automatically during search to rerank candidates.
    F16,
}

/// Internal representation of [`Rerank`].
///
/// This is used for computing distances among the raw values in the auxiliary store.
#[derive(Debug)]
pub(in crate::repr) enum Reranker {
    None,
    F16(Distance<f32, f16>),
}

impl Reranker {
    /// Construct a new [`Reranker`] and a [`store::slots::SlotsConfig`]  for the auxiliary
    /// store.
    pub(in crate::repr) fn new_with_config(
        rerank: Rerank,
        metric: Metric,
        dim: usize,
    ) -> (Self, Option<simple::Config>) {
        let this = match rerank {
            Rerank::None => Self::None,
            Rerank::F16 => {
                let distance = <f32 as DistanceProvider<f16>>::distance_comparer(
                    metric.as_vector_metric(),
                    Some(dim),
                );

                Self::F16(distance)
            }
        };

        let config = match &this {
            Self::None => None,
            Self::F16(_) => Some(Simple::config(this.bytes_for(dim))),
        };

        (this, config)
    }

    #[expect(
        clippy::expect_used,
        reason = "the arithmetic should not overflow for the feasible `dim` values"
    )]
    fn bytes_for(&self, dim: usize) -> Bytes {
        match self {
            Self::None => Bytes::new(0),
            Self::F16(_) => Bytes::new(dim.checked_mul(2).expect("f16 is smaller than the f32")),
        }
    }

    /// Create a [`repr::PostProcess`].
    ///
    /// This assumes that `simple` has the same dimensions as `self`'s contained distance
    /// computation and that `guard` belongs to `simple`.
    ///
    /// # Pre-conditions
    ///
    /// This requires that `slots` is the [`store::slots::Slots`] created from the
    /// configuration returned in [`Self::new_with_config`].
    #[expect(
        clippy::panic,
        reason = "this is an internal method that must be set up correctly"
    )]
    pub(in crate::repr) fn post_process<'a>(
        &'a self,
        query: &'a [f32],
        guard: &epoch::Guard<'a>,
        slots: &'a Optional<store::simple::Simple>,
        counters: &LocalCounters<'a>,
    ) -> Option<Box<dyn repr::PostProcess + 'a>> {
        match (self, slots.slots()) {
            (Self::None, None) => None,
            (Self::F16(distance), Some(simple)) => {
                let distance = repr::full::QueryDistance::new(Calf::Borrowed(query), *distance);
                let reader = simple.reader(guard.share());
                let post_process =
                    repr::internal::simple::Reranker::new(reader, distance, counters.fork());
                Some(Box::new(post_process))
            }
            _ => panic!("invalid combination of arguments"),
        }
    }

    /// Store the vector `v` into the raw buffer `buf`.
    ///
    /// # Pre-conditions
    ///
    /// `buf` must be consistent with the configuration returned from [`Self::new_with_config`],
    /// and may only be `None` if that configuration was `None`.
    ///
    /// If it is `Some`, this function may panic if its length is not consistent with the
    /// original configuration.
    #[expect(
        clippy::panic,
        reason = "this is an internal method that must be set up correctly"
    )]
    pub(in crate::repr) fn store(&self, v: &[f32], buf: &mut Option<simple::Exclusive<'_>>) {
        match (self, buf) {
            (Self::None, None) => {}
            (Self::F16(_), Some(exclusive)) => {
                use diskann_vector::conversion::CastFromSlice;
                bytemuck::cast_slice_mut::<u8, f16>(exclusive.as_mut_slice()).cast_from_slice(v);
            }
            _ => panic!("invalid combination of arguments"),
        }
    }
}

/// Test that [`repr::PostProcess`] reranks correctly.
///
/// Pass all `ids` to [`repr::PostProcess::post_process`]. Verify that all ids not
/// present in `distances` have been removed and the remaining ids are present, sorted,
/// and have distance values matching those in `distances`.
#[cfg(test)]
pub(in crate::repr) fn test_rerank(
    post_process: &mut dyn repr::PostProcess,
    distances: hashbrown::HashMap<crate::num::SlotId, f32>,
    ids: &[crate::num::SlotId],
    ctx: &dyn std::fmt::Display,
) {
    use diskann::neighbor::Neighbor;

    use crate::num::SlotId;

    let mut buffer: Vec<_> = ids
        .iter()
        .map(|slot_id| Neighbor::new(slot_id.value(), 0.0))
        .collect();

    post_process.post_process(&mut buffer).unwrap();
    let mut previous = f32::NEG_INFINITY;
    assert_eq!(buffer.len(), distances.len(), "{ctx}");
    for (pos, neighbor) in buffer.iter().enumerate() {
        let current = *neighbor.distance();
        assert_eq!(
            current,
            distances[&SlotId(*neighbor.id())],
            "failed in position {} of {:?} -- {ctx}",
            pos,
            buffer
        );

        assert!(
            current >= previous,
            "distances is not monotonically increasing, previous = {}, current = {} -- {}",
            previous,
            current,
            ctx,
        );

        previous = current;
    }

    assert!(
        previous > f32::NEG_INFINITY,
        "previous = {} -- {}",
        previous,
        ctx
    );
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::assert_matches;

    use diskann::utils::IntoUsize;

    use crate::{
        counters::Counters,
        num::{Capacity, LogicalId, MaxDegree, SlotId},
        repr::test::Reference,
        store::Store,
    };

    const TEST_DIM: usize = 1;

    /// Create a reranker of dim 5 for the given config and metric.
    fn make_store(rerank: Rerank, metric: Metric) -> (Reranker, Store<Optional<Simple>>) {
        let (reranker, config) = Reranker::new_with_config(rerank, metric, TEST_DIM);

        let store = Store::new(
            store::Layout::new(Capacity::new(10), MaxDegree::new(0), 0),
            store::Config::new(),
            config,
        )
        .unwrap();

        (reranker, store)
    }

    #[test]
    fn test_disabled_rerank() {
        let (reranker, store) = make_store(Rerank::None, Metric::SquaredL2);
        assert_matches!(reranker, Reranker::None);
        assert!(
            store.slots().slots().is_none(),
            "Rerank::None should create empty optional slots"
        );

        let query = std::slice::from_ref(&1.0);
        let counters = Counters::new();

        let post_processor = store
            .guard(|slots, guard| reranker.post_process(query, &guard, slots, &counters.local()))
            .unwrap();

        assert!(post_processor.is_none());
    }

    #[test]
    fn test_reranker_f16() {
        for metric in Metric::all() {
            let (reranker, store) = make_store(Rerank::F16, metric);
            let mut map = Reference::new(TEST_DIM);

            // Insert the following values by logical id:
            //
            // LogicalID    Value   Deleted
            //         5      2.5        No
            //         4      1.5        No
            //         3      0.5       Yes
            //         2     -0.5       Yes
            //         1     -1.5        No
            //         0     -2.5       Yes
            //
            // The value -1.0 is used as the query, which generates the following values for
            // the metrics being tested (note, we're using similarity scores here):
            //
            // LogicalId    Value    SquaredL2   InnerProduct    Cosine
            //         5      2.5        12.25            2.5       2.0
            //         4      1.5         6.25            1.5       2.0
            //         1     -1.5         0.25           -1.5       0.0

            let distances = match metric {
                Metric::SquaredL2 => [
                    (LogicalId(5), 12.25),
                    (LogicalId(4), 6.25),
                    (LogicalId(1), 0.25),
                ],
                Metric::InnerProduct => [
                    (LogicalId(5), 2.5),
                    (LogicalId(4), 1.5),
                    (LogicalId(1), -1.5),
                ],
                Metric::Cosine => [
                    (LogicalId(5), 2.0),
                    (LogicalId(4), 2.0),
                    (LogicalId(1), 0.0),
                ],
            };

            // Insert
            for i in (0..=5).rev() {
                let v = (i as f32) - 2.5;

                let mut exclusive = store.acquire().unwrap();

                map.insert(
                    LogicalId(i),
                    SlotId(exclusive.slot()),
                    std::slice::from_ref(&v),
                );

                reranker.store(std::slice::from_ref(&v), exclusive.data());

                exclusive.publish();
            }

            // Delete
            store
                .retire(map.slot_id_for(LogicalId(3)).value().into_usize())
                .unwrap();
            store
                .retire(map.slot_id_for(LogicalId(2)).value().into_usize())
                .unwrap();
            store
                .retire(map.slot_id_for(LogicalId(0)).value().into_usize())
                .unwrap();

            let counters = Counters::new();
            let query = std::slice::from_ref(&-1.0);

            // Create the post-processor.
            let mut post_process = store
                .guard(|slots, guard| {
                    reranker
                        .post_process(query, &guard, slots, &counters.local())
                        .unwrap()
                })
                .unwrap();

            let expected: hashbrown::HashMap<_, _> = distances
                .map(|(logical_id, distance)| (map.slot_id_for(logical_id), distance))
                .into_iter()
                .collect();

            test_rerank(
                &mut *post_process,
                expected,
                &[6, 5, 4, 3, 2, 1, 0].map(SlotId),
                &format_args!("metric = {:?}", metric),
            );
        }
    }
}
