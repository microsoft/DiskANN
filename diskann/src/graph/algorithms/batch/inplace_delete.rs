/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Debug;

use diskann_utils::future::SendFuture;
use hashbrown::{HashMap, HashSet};

use crate::{
    ANNResult,
    graph::{
        Config, algorithms,
        algorithms::inplace_delete::{self, OneHop, TwoHopAndOneHop, VisitedAndTopK},
        glue::PruneAccessor,
        internal::BackedgeBuffer,
    },
    utils::VectorId,
};

#[derive(Debug)]
pub struct Edges<I> {
    edges: HashMap<I, BackedgeBuffer<I>>,
}

impl<I> Edges<I> {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    pub fn into_iter(self) -> EdgeIter<I> {
        EdgeIter::new(self.edges.into_iter())
    }
}

impl<I> Edges<I>
where
    I: VectorId,
{
    pub fn merge_from(&mut self, other: &Edges<I>) {
        self.merge_from_inner(&other.edges)
    }

    fn merge_from_inner(&mut self, edges: &HashMap<I, BackedgeBuffer<I>>) {
        for (src, dst) in edges.iter() {
            self.edges
                .entry(*src)
                .and_modify(|b| {
                    for d in dst.iter() {
                        b.push(*d);
                    }
                })
                .or_insert_with(|| dst.clone());
        }
    }

    pub fn merge(&mut self, other: Edges<I>) {
        self.merge_inner(other.edges)
    }

    fn merge_inner(&mut self, edges: HashMap<I, BackedgeBuffer<I>>) {
        for (src, dst) in edges.into_iter() {
            self.edges
                .entry(src)
                .and_modify(|b| {
                    for d in dst.into_iter() {
                        b.push(*d);
                    }
                })
                .or_insert_with(|| dst);
        }
    }
}

#[derive(Debug)]
pub struct EdgeIter<I> {
    iter: hashbrown::hash_map::IntoIter<I, BackedgeBuffer<I>>,
}

impl<I> EdgeIter<I> {
    fn new(iter: hashbrown::hash_map::IntoIter<I, BackedgeBuffer<I>>) -> Self {
        Self { iter }
    }
}

impl<I> Iterator for EdgeIter<I> {
    type Item = Edge<I>;

    fn next(&mut self) -> Option<Edge<I>> {
        let (source, destinations) = self.iter.next()?;
        Some(Edge::new(source, destinations))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I> ExactSizeIterator for EdgeIter<I> {}

#[derive(Debug)]
pub struct Edge<I> {
    source: I,
    destinations: BackedgeBuffer<I>,
}

impl<I> Edge<I> {
    fn new(source: I, destinations: BackedgeBuffer<I>) -> Self {
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
}

#[derive(Debug)]
pub struct Workspace<I>
where
    I: VectorId,
{
    scratch: inplace_delete::Workspace<I>,
}

impl<I> Workspace<I>
where
    I: VectorId,
{
    pub fn new() -> Self {
        Self {
            scratch: inplace_delete::Workspace::new(),
        }
    }

    pub fn find_one_hop<A>(
        &mut self,
        edges: &mut Edges<A::Id>,
        accessor: &mut A,
        one_hop: &OneHop,
        id: I,
    ) -> impl SendFuture<ANNResult<()>>
    where
        A: inplace_delete::Access<Id = I>,
    {
        async move {
            one_hop.discover(accessor, id, &mut self.scratch).await?;
            edges.merge_from_inner(self.scratch.delta());
            Ok(())
        }
    }

    pub fn find_two_hop_and_one_hop<A>(
        &mut self,
        edges: &mut Edges<A::Id>,
        accessor: &mut A,
        two_hop: &TwoHopAndOneHop,
        id: I,
    ) -> impl SendFuture<ANNResult<()>>
    where
        A: inplace_delete::Access<Id = I>,
    {
        async move {
            two_hop.discover(accessor, id, &mut self.scratch).await?;
            edges.merge_from_inner(self.scratch.delta());
            Ok(())
        }
    }

    pub fn find_visited_and_top_k<A>(
        &mut self,
        edges: &mut Edges<A::Id>,
        accessor: &mut A,
        visited_and_top_k: &VisitedAndTopK,
        id: I,
        config: &Config,
    ) -> impl SendFuture<ANNResult<()>>
    where
        A: inplace_delete::TopKAccess<Id = I>,
    {
        async move {
            visited_and_top_k
                .discover(accessor, id, config, &mut self.scratch)
                .await?;
            edges.merge_from_inner(self.scratch.delta());
            Ok(())
        }
    }

    pub async fn apply_edge<A>(
        &mut self,
        accessor: &mut A,
        edge: &Edge<I>,
        deleted: &HashSet<I>,
        config: &Config,
    ) -> ANNResult<()>
    where
        A: PruneAccessor<Id = I>,
    {
        let Edge {
            source,
            destinations,
        } = edge;
        algorithms::insert::add_edge_and_prune(
            accessor,
            &destinations,
            *source,
            self.scratch.prune(),
            Some(&algorithms::insert::Exclude::new(deleted)),
            config,
        )
        .await?;

        Ok(())
    }
}

//---------//
// BuiltIn //
//---------//
