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
        Parallelism,
        region::{Update, View},
        split::{Links, Reassign},
    },
    utils::VectorId,
};

////////////
// Linked //
////////////

/// Reassigns points between linked lists, choosing destinations with `A`.
///
/// # Algorithm
///
/// 1. For each list, independently and possibly in parallel: offer each member the list's
///    own centroid first, then the centroids of its linked lists in link order, and let `A`
///    choose among them. This step only reads the region, so its failures leave the update
///    untouched.
///
/// 2. In the order of [`Links::lists`], remove the members that move from each list. Then
///    append each moved point to its destination.
///
/// Each point is considered once, from the list it was filled in, and no centroid changes.
pub struct Linked<A = NearestCentroid> {
    assign: A,
}

impl<A> Linked<A> {
    /// Construct a policy that chooses destinations with `assign`.
    pub fn new(assign: A) -> Self {
        Self { assign }
    }

    /// Return the members of list `i` in `links` that move, each with the index of its
    /// destination in [`Links::lists`].
    ///
    /// This reads only `region`, so lists can be planned in parallel.
    fn moves<I, L, R>(&self, region: &R, links: &Links<L>, i: usize) -> ANNResult<Vec<(I, usize)>>
    where
        I: Copy,
        L: VectorId,
        R: View<I, L>,
        A: for<'a> Assign<R::VectorsRef<'a>>,
    {
        let get = |list: L| {
            region
                .get_list(list)
                .ok_or(SplitError::UnableToRetrieve(list))
        };

        let list = links.lists()[i];
        let view = get(list)?;
        let vectors = region
            .get_vectors(list)
            .ok_or(SplitError::UnableToRetrieve(list))?;

        // Offer the list's own centroid first, so points can stay in place.
        let mut candidates = vec![(i, view.centroid)];
        for &j in links.links(i) {
            candidates.push((j, get(links.lists()[j])?.centroid));
        }

        let mut moves = Vec::new();
        self.assign
            .assign(view.members, vectors.reborrow(), &candidates, |id, dst| {
                if dst != i {
                    moves.push((id, dst));
                }
            });

        Ok(moves)
    }
}

impl<I, L, R, A> Reassign<I, L, R> for Linked<A>
where
    I: Copy + Send,
    L: VectorId,
    R: View<I, L>,
    A: for<'a> Assign<R::VectorsRef<'a>>,
{
    type Error = ANNError;

    fn reassign<U>(
        &mut self,
        region: &R,
        update: &mut U,
        links: &Links<L>,
        parallelism: Parallelism<'_>,
    ) -> ANNResult<()>
    where
        U: Update<I, L>,
    {
        let this = &*self;
        let moves = parallelism.try_map(links.len(), |i| this.moves(region, links, i))?;

        // Write in list order, so the changes do not depend on `parallelism`.
        let mut moved = Buckets::new(links.len());

        for (&list, moves) in links.lists().iter().zip(moves) {
            for (id, dst) in moves {
                moved.push(id, dst);
            }

            update.remove_members(list, moved.seal())?;
        }

        let incoming = moved.group();

        for (i, &list) in links.lists().iter().enumerate() {
            update.append_members(list, incoming.get(i))?;
        }

        Ok(())
    }
}
