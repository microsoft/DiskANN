/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt::Debug,
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering},
};

use diskann_utils::future::SendFuture;
use hashbrown::HashMap;

use crate::{
    ANNResult,
    error::ErrorExt,
    graph::{
        AdjacencyList, Config, algorithms,
        glue::PruneAccessor,
        internal::{BackedgeBuffer, prune},
    },
    utils::{VectorId, async_tools, balancer::{Balancer, BatchPosition}},
};

//--------------//
// ForwardEdges //
//--------------//

#[derive(Debug)]
pub struct ForwardEdges<I> {
    edges: Vec<ForwardEdge<I>>,
}

impl<I> ForwardEdges<I> {
    pub fn new() -> Self {
        Self {
            edges: Vec::new()
        }
    }

    pub fn merge(&mut self, other: ForwardEdges<I>) {
        self.edges.extend(other.edges.into_iter());
    }

    fn push(&mut self, edge: ForwardEdge<I>) {
        self.edges.push(edge)
    }

    fn len(&self) -> usize {
        self.edges.len()
    }
}

#[derive(Debug)]
pub struct ForwardEdgeIter<I> {
    iter: std::vec::IntoIter<ForwardEdge<I>>,
}

impl<I> Iterator for ForwardEdgeIter<I> {
    type Item = ForwardEdge<I>;

    fn next(&mut self) -> Option<ForwardEdge<I>> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I> ExactSizeIterator for ForwardEdgeIter<I> {}

#[derive(Debug)]
pub struct ForwardEdge<I> {
    source: I,
    destinations: AdjacencyList<I>,
}

impl<I> ForwardEdge<I> {
    fn new(source: I, destinations: AdjacencyList<I>) -> Self {
        Self {
            source,
            destinations,
        }
    }

    pub fn source(&self) -> &I {
        &self.source
    }

    pub fn destinations(&self) -> &[I] {
        &self.destinations
    }

    pub fn into_inner(self) -> (I, AdjacencyList<I>) {
        (self.source, self.destinations)
    }
}

//-----------//
// BackEdges //
//-----------//

#[derive(Debug)]
pub struct BackEdges<I> {
    edges: HashMap<I, BackedgeBuffer<I>>,
}

impl<I> BackEdges<I> {
    pub fn new() -> Self {
        Self { edges: HashMap::new() }
    }

    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn skip_bootstrap(self) -> BackEdgeIter<I> {
        BackEdgeIter::new(self.edges.into_iter())
    }
}

impl<I> BackEdges<I>
where
    I: VectorId,
{
    pub fn accumulate(&mut self, forward: &ForwardEdges<I>) {
        for ForwardEdge {
            source,
            destinations,
        } in forward.edges.iter()
        {
            for destination in destinations.iter() {
                self.edges
                    .entry(*destination)
                    .and_modify(|b| {
                        b.push(*source);
                    })
                    .or_insert_with(|| BackedgeBuffer::new(*source));
            }
        }
    }

    pub fn merge(&mut self, other: Self) {
        for (src, dst) in other.edges.into_iter() {
            self.edges
                .entry(src)
                .and_modify(|b| {
                    for d in dst.iter() {
                        b.push(*d);
                    }
                })
                .or_insert_with(|| dst);
        }
    }
}

impl<I> Default for BackEdges<I> {
    fn default() -> Self {
        Self {
            edges: Default::default(),
        }
    }
}

#[derive(Debug)]
pub struct BackEdgeIter<I> {
    iter: hashbrown::hash_map::IntoIter<I, BackedgeBuffer<I>>,
}

impl<I> BackEdgeIter<I> {
    fn new(iter: hashbrown::hash_map::IntoIter<I, BackedgeBuffer<I>>) -> Self {
        Self { iter }
    }
}

impl<I> Iterator for BackEdgeIter<I> {
    type Item = BackEdge<I>;

    fn next(&mut self) -> Option<BackEdge<I>> {
        let (source, destinations) = self.iter.next()?;
        Some(BackEdge {
            source,
            destinations,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I> ExactSizeIterator for BackEdgeIter<I> {}

#[derive(Debug)]
pub struct BackEdge<I> {
    source: I,
    destinations: BackedgeBuffer<I>,
}

impl<I> BackEdge<I> {
    fn new(source: I, destinations: BackedgeBuffer<I>) -> Self {
        Self {
            source,
            destinations,
        }
    }
}

//-------------//
// BatchInsert //
//-------------//

#[derive(Debug)]
pub struct BatchInsert<I>
where
    I: VectorId,
{
    find: algorithms::FindInsertCandidates<I>,
    buf: AdjacencyList<I>,
}

impl<I> BatchInsert<I>
where
    I: VectorId,
{
    pub fn new() -> Self {
        Self {
            find: algorithms::FindInsertCandidates::new(),
            buf: AdjacencyList::new()
        }
    }

    pub fn find_forward_edges<A>(
        &mut self,
        forward_edges: &mut ForwardEdges<I>,
        accessor: &mut A,
        batch: BatchPosition<'_, I>,
        config: &Config,
    ) -> impl SendFuture<ANNResult<()>>
    where
        A: algorithms::Insert<Id = I>,
    {
        async move {
            // Add other vector id pairs in the mini batch to the candidate pool.
            let intra_batch_candidates = config.intra_batch_candidates().get(batch.len());
            let id = *batch.current();

            self.find
                .find_insert_candidates(
                    accessor,
                    id,
                    config,
                    batch.around(intra_batch_candidates).copied(),
                )
                .await?;

            forward_edges.push(ForwardEdge::new(id, self.find.candidates().clone()));
            Ok(())
        }
    }

    async fn bootstrap<A>(
        &mut self,
        forward_edges: &mut ForwardEdges<I>,
        accessor: &mut A,
        batch: BatchPosition<'_, ForwardEdge<I>>,
        config: &Config,
    ) -> ANNResult<()>
    where
        A: PruneAccessor<Id = I>,
    {
        self.prepare_bootstrap_candidates(batch);

        // During bootstrap, we want the graph to be as dense as possible to aid
        // in early navigation.
        //
        // Enabling saturation help achieve that.
        let options = prune::Options {
            force_saturate: true,
        };

        let id = batch.current().source;
        algorithms::robust_prune_list(
            accessor,
            id,
            &self.buf,
            &mut self.find.prune(),
            options,
            config,
        )
        .await
        .escalate("retrieving inserted vector must succeed")?;

        forward_edges.push(ForwardEdge::new(id, self.find.candidates().clone()));
        Ok(())
    }

    fn prepare_bootstrap_candidates(&mut self, batch: BatchPosition<'_, ForwardEdge<I>>) {
        let current = batch.current();
        self.buf
            .replace_from_iter_untrusted(current.destinations.iter().copied().chain(
                batch.all().into_iter().filter_map(|i| {
                    // Avoid self loops.
                    if i.source == current.source {
                        None
                    } else {
                        Some(i.source)
                    }
                }),
            ));
    }

    async fn apply_back_edge<A>(
        &mut self,
        accessor: &mut A,
        back_edge: &BackEdge<I>,
        config: &Config,
    ) -> ANNResult<()>
    where
        A: PruneAccessor<Id = I>,
    {
        let BackEdge {
            source,
            destinations,
        } = back_edge;
        // Even though `BackEdge` maintains the same uniqueness property, the replacement
        // algorithm will sort the results, which can improve determinism.
        self.buf
            .replace_from_iter_untrusted(destinations.iter().copied());
        algorithms::add_edge_and_prune(
            accessor,
            &self.buf,
            *source,
            self.find.prune(),
            None,
            config,
        )
        .await?;
        Ok(())
    }
}

#[derive(Debug)]
pub enum Candidates<I> {
    Ready {
        forward: ForwardEdges<I>,
        backward: BackEdgeIter<I>,
    },
    BootstrapNeeded {
        forward: ForwardEdges<I>,
    },
}

impl<I> Candidates<I> {
    pub fn check(config: &Config, forward: ForwardEdges<I>, backward: BackEdges<I>) -> Self {
        // Check if number of unique back edges source is very small. If so, we do kick
        // off the bootstrap routine and add edges from within the batch.
        //
        // Bootstrap is skipped when intra-batch candidates cover the full batch (i.e.
        // `All` or a `Max(n)` that resolves to >= batch size), since every item already
        // had full visibility of the batch during pruning.
        //
        // Setting the max to 1 ensures that if `work.len() == 1` (i.e., there is only
        // one item in the batch) that we don't trigger bootstrap. It wouldn't do anything.
        let batch_size = forward.len();
        let resolved_candidates = config.intra_batch_candidates().get(batch_size).max(1);
        if resolved_candidates < batch_size && backward.len().div_ceil(8) <= batch_size {
            Self::BootstrapNeeded { forward }
        } else {
            Self::Ready {
                forward,
                backward: backward.skip_bootstrap(),
            }
        }
    }
}

