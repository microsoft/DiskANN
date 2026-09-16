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
