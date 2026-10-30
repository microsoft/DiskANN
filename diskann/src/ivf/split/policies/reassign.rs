/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Reassignment of points between linked lists.

use diskann_utils::Reborrow;

use super::{Assign, Buckets, NearestCentroid, SplitError};
use crate::{
    ANNError, ANNResult,
    ivf::{
        region::{Update, View},
        split::{Links, Reassign},
    },
    utils::VectorId,
};


////////////
// Linked //
////////////
/// Each point stays or moves to a list linked to its own, as chosen by `A`.
pub struct Linked<A = NearestCentroid> {
    assign: A,
}

impl<A> Linked<A> {
    pub fn new(assign: A) -> Self {
        Self { assign }
    }
}

impl<I, L, R, A> Reassign<I, L, R> for Linked<A>
where
    I: Copy,
    L: VectorId,
    R: View<I, L>,
    A: for<'a> Assign<R::VectorsRef<'a>>,
{
    type Error = ANNError;

    fn reassign<U>(&mut self, region: &R, update: &mut U, links: &Links<L>) -> ANNResult<()>
    where
        U: Update<I, L>,
    {
        let get = move |list: L| {
            region
                .get_list(list)
                .ok_or(SplitError::UnableToRetrieve(list))
        };

        let mut moved = Buckets::new(links.len());
        let mut candidates = Vec::new();

        for (i, &list) in links.lists().iter().enumerate() {
            let view = get(list)?;

            let vectors = region
                .get_vectors(list)
                .ok_or(SplitError::UnableToRetrieve(list))?;

            // A point can stay if the current centroid is closer.
            candidates.clear();
            candidates.push((i, view.centroid));

            for &j in links.links(i) {
                candidates.push((j, get(links.lists()[j])?.centroid));
            }

            self.assign
                .assign(view.members, vectors.reborrow(), &candidates, |id, dst| {
                    if dst != i {
                        moved.push(id, dst);
                    }
                });

            update.remove_members(list, moved.seal())?;
        }

        let incoming = moved.group();

        for (i, &list) in links.lists().iter().enumerate() {
            update.append_members(list, incoming.get(i))?;
        }

        Ok(())
    }
}
