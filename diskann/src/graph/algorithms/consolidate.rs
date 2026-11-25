/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! FIXME: All kinds of races between checking adjacency lists and trying to access things.
//!
//! Lots of races.

use std::{borrow::BorrowMut, fmt::Debug};

use diskann_utils::future::SendFuture;
use hashbrown::HashSet;

use crate::graph::{
    AdjacencyList, Config, adjacencylist::Empty, algorithms, glue::PruneAccessor, internal::prune,
};

use crate::{
    ANNResult,
    error::ErrorExt,
    provider::{HasId, NeighborAccessorMut},
    utils::{Bool, VectorId},
};

pub trait Access: HasId {
    type PruneAccessor<'a>: PruneAccessor<Id = Self::Id>
    where
        Self: 'a;

    fn prune_accessor(&mut self) -> impl std::borrow::BorrowMut<Self::PruneAccessor<'_>> + Send;

    fn neighbors(
        &mut self,
        id: Self::Id,
        neighbors: Empty<'_, Self::Id>,
    ) -> impl SendFuture<ANNResult<Bool>>;

    /// Out-of-bounds IDs may be considered as deleted.
    fn partition_live(
        &mut self,
        ids: &mut AdjacencyList<Self::Id>,
        deleted: Option<Empty<'_, Self::Id>>,
    ) -> impl SendFuture<ANNResult<()>>;
}

// enum used to return the status of the vector that `consolidate_vector`
// was called on: Deleted if the vector was already deleted, and Complete
// if the vector was not deleted (and thus is now consolidated)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsolidateKind {
    /// Consolidate was called on a deleted vector.
    Deleted,

    /// Consolidate was called on valid vector, but retrieving the data for that vector
    /// failed with a transient error.
    FailedVectorRetrieval,

    /// Consolidate completed successfully.
    Complete,
}

#[derive(Debug)]
pub struct Workspace<I> {
    live: AdjacencyList<I>,
    deleted: AdjacencyList<I>,
    neighborhood: HashSet<I>,
    prune: prune::Scratch<I>,
}

impl<I> Workspace<I> {
    pub fn new() -> Self {
        Self {
            live: AdjacencyList::new(),
            deleted: AdjacencyList::new(),
            neighborhood: HashSet::new(),
            prune: prune::Scratch::new(),
        }
    }

    pub fn run<A>(
        &mut self,
        access: &mut A,
        id: A::Id,
        config: &Config,
    ) -> impl SendFuture<ANNResult<ConsolidateKind>>
    where
        I: VectorId,
        A: Access<Id = I> + Send,
    {
        async move {
            if !access.neighbors(id, self.live.as_empty()).await?.get() {
                return Ok(ConsolidateKind::Deleted);
            }

            self.deleted.clear();
            self.neighborhood.clear();

            access
                .partition_live(&mut self.live, Some(self.deleted.as_empty()))
                .await?;
            if self.deleted.is_empty() && self.live.len() <= config.pruned_degree().get() {
                return Ok(ConsolidateKind::Complete);
            }

            self.neighborhood.extend(self.live.iter().copied());

            for &deleted in self.deleted.iter() {
                // Always consolidate the neighborhood, even if there is a race and this deleted
                // neighbor gets resurrected.
                let _ = access.neighbors(deleted, self.live.as_empty()).await?;
                access.partition_live(&mut self.live, None).await?;
                self.neighborhood.extend(self.live.iter().copied());
            }

            // Remove self-loop if it snuck in.
            self.neighborhood.remove(&id);
            self.live
                .replace_from_iter_unique(self.neighborhood.iter().copied());

            let mut prune_accessor = access.prune_accessor();
            let update: &[I] = {
                // FIXME: Use of `pruned_degree` is probably incorrect but copied over from an
                // earlier version of this algorithm. Kept for consistency.
                if self.live.len() < config.pruned_degree().get() {
                    &self.live
                } else {
                    // Force saturation is mainly used for the retry insert logic.
                    //
                    // No need to force during consolidation.
                    let options = prune::Options {
                        force_saturate: false,
                    };

                    if algorithms::insert::robust_prune_list(
                        prune_accessor.borrow_mut(),
                        id,
                        &self.live,
                        &mut self.prune,
                        options,
                        config,
                    )
                    .await
                    .allow_transient("vector retrieval is allowed to fail during consolidate")?
                    .is_none()
                    {
                        return Ok(ConsolidateKind::FailedVectorRetrieval);
                    }

                    &self.prune.neighbors
                }
            };

            prune_accessor
                .borrow_mut()
                .neighbors()
                .borrow_mut()
                .set_neighbors(id, update)
                .await?;
            Ok(ConsolidateKind::Complete)
        }
    }
}
