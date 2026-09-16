/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult, neighbor::Neighbor, utils::IntoUsize};

use crate::{counters::LocalCounters, repr, store};

use super::RawQueryDistance;

//////////////
// Reranker //
//////////////

/// A reranker for data stored in a [`store::simple::Simple`].
#[derive(Debug)]
pub(in crate::repr) struct Reranker<'a, D> {
    reader: store::simple::Reader<'a>,
    distance: D,
    counters: LocalCounters<'a>,
}

impl<'a, D> Reranker<'a, D> {
    /// Construct a new reranker over `reader`.
    pub(in crate::repr) fn new(
        reader: store::simple::Reader<'a>,
        distance: D,
        counters: LocalCounters<'a>,
    ) -> Self {
        Self {
            reader,
            distance,
            counters,
        }
    }
}

impl<'a, D> repr::PostProcess for Reranker<'a, D>
where
    D: RawQueryDistance,
{
    fn post_process(&mut self, buffer: &mut Vec<Neighbor<u32>>) -> ANNResult<()> {
        let mut result: Result<(), D::Error> = Ok(());
        buffer.retain_mut(|neighbor| {
            if result.is_err() {
                return false;
            }

            let id = *neighbor.id();
            if let Some(data) = self.reader.read(id.into_usize()) {
                match self.distance.eval(data) {
                    Ok(distance) => *neighbor = Neighbor::new(id, distance),
                    Err(err) => result = Err(err),
                }
                true
            } else {
                false
            }
        });

        buffer.sort_unstable_by(diskann::neighbor::ord::fast_distance);

        // Update counters.
        //
        // We can do this in bulk because we remove entries where reading failed. The
        // result has a one-to-one correspondence with distance and vector fetches.
        self.counters.get_vector(buffer.len() as u64);
        self.counters.query_distance(buffer.len() as u64);

        result.map_err(ANNError::new)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use diskann::utils::IntoUsize;

    use crate::{
        counters::Counters,
        num::{Bytes, Capacity, MaxDegree},
        store::{self, Store},
        test::Sequencer,
    };

    #[test]
    fn test_simple_rerank() {
        // Since we don't necessarily guarantee monotonic slot accesses, this hash map
        // serves as a translation layer for slot ids to the expected payload.
        let mut map = HashMap::<u32, f32>::new();

        let store = Store::new(
            store::Layout::new(Capacity::new(10), MaxDegree::new(0), 0),
            store::Config::default(),
            store::simple::Simple::config(Bytes::new(std::mem::size_of::<f32>())),
        )
        .unwrap();

        for i in 0..10 {
            let v = (9 - i) as f32;

            let mut exclusive = store.acquire().unwrap();
            assert!(map.insert(exclusive.slot(), v).is_none());
            exclusive
                .data()
                .as_mut_slice()
                .copy_from_slice(bytemuck::bytes_of(&v));
            exclusive.publish();
        }

        // If we rerank - IDs should come out in reverse order.
        let counters = Counters::new();
        let mut reranker = Reranker::new(
            store.guard(|simple, guard| simple.reader(guard)).unwrap(),
            repr::test::TestQueryDistance::new(-1.0),
            counters.local(),
        );

        // Single threaded.
        let mut neighbors: Vec<_> = (0..10).map(|i| Neighbor::new(i, -500.0)).collect();
        repr::PostProcess::post_process(&mut reranker, &mut neighbors).unwrap();

        // Check that everything is in the right order.
        //
        // We expect these to be by increasing distance. Since we inserted floating point
        // values from 0 to 9, we expected distances from -1 to 8.
        assert_eq!(neighbors.len(), 10);
        for (i, n) in neighbors.iter().enumerate() {
            assert_eq!(
                *n.distance(),
                -1.0 + (i as f32),
                "mismatch for id {} at position {}", n.id(), i
            );

            // Verify that the ID matches what we think it should.
            assert_eq!(map[n.id()], i as f32);
        }

        // Multi-threaded.
        //
        // In this test - we go again. However, another thread deletes all odd distances.
        let mut neighbors: Vec<_> = (0..10).map(|i| Neighbor::new(i, -500.0)).collect();
        let seq = Sequencer::new();

        std::thread::scope(|s| {
            s.spawn(|| {
                for (id, v) in map.iter() {
                    if (*v as u32).is_multiple_of(2) {
                        store.retire(id.into_usize()).unwrap();
                    }
                }

                seq.wait_for(0);
            });

            // Wait until we know the IDs have been deleted.
            seq.until_waiting_for(0);

            // Now rerank again - all even distances should be filtered out.
            repr::PostProcess::post_process(&mut reranker, &mut neighbors).unwrap();

            // Check that everything is in the right order.
            //
            // We expect these to be by increasing distance. Since we inserted floating point
            // values from 0 to 9, we expected distances from -1 to 8.
            assert_eq!(neighbors.len(), 5);
            for (i, n) in neighbors.iter().enumerate() {
                assert_eq!(
                    *n.distance(),
                    -1.0 + ((2 * i + 1) as f32),
                    "mismatch for id {} at position {}", n.id(), i
                );

                // Verify that the ID matches what we think it should.
                assert_eq!(map[n.id()], (2 * i + 1) as f32);
            }

            seq.advance_past(0);
        });

    }
}
