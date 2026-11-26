/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{borrow::BorrowMut, fmt::Debug, sync::Arc};

use diskann_utils::future::{AsyncFriendly, SendFuture};
use hashbrown::{HashMap, HashSet};
use tokio::sync::{mpsc, oneshot};

use crate::{
    ANNError, ANNResult, HasId,
    graph::{
        Config, algorithms,
        algorithms::inplace_delete::{self, OneHop, TwoHopAndOneHop, VisitedAndTopK},
        glue::{NeighborAccessorMut, PruneAccessor},
        internal::BackedgeBuffer,
    },
    utils::{VectorId, balancer::Balancer},
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

pub trait Standard: std::fmt::Debug + HasId + Send + Sync + 'static {
    type PruneAccessor<'a>: PruneAccessor<Id = Self::Id>;

    fn discover(
        &mut self,
        id: Self::Id,
        workspace: &mut Workspace<Self::Id>,
        edges: &mut Edges<Self::Id>,
        config: &Config,
    ) -> impl SendFuture<ANNResult<()>>;

    fn prune_accessor(
        &mut self,
    ) -> ANNResult<impl std::borrow::BorrowMut<Self::PruneAccessor<'_>> + Send>;
}

pub async fn standard<S, F>(
    spawn: &impl crate::tokio::Spawn,
    batch: Arc<[S::Id]>,
    config: &Config,
    make: F,
) -> ANNResult<()>
where
    F: FnMut() -> S,
    S: Standard,
{
    let ntasks = config.max_minibatch_par();

    // Spawn all worker tasks.
    let mut tasks = Vec::<tokio::task::JoinHandle<()>>::with_capacity(ntasks.get());
    let mut commands = Vec::<mpsc::Sender<Command<S::Id>>>::with_capacity(ntasks.get());
    let mut discover_responses = Vec::with_capacity(ntasks.get());

    let deleted: Arc<HashSet<_>> = Arc::new(batch.iter().copied().collect());
    let balancer = Arc::new(Balancer::new(batch.clone()));

    std::iter::repeat_with(make)
        .take(ntasks.get())
        .for_each(|standard| {
            let deleted_clone = deleted.clone();
            let config_clone = config.clone();
            let balancer_clone = balancer.clone();

            let (discover_sender, discover_receiver) = oneshot::channel();
            let (command_sender, command_receiver) = mpsc::channel(1);

            let h = spawn.spawn(worker(
                standard,
                deleted_clone,
                config_clone,
                command_receiver,
                balancer_clone,
                discover_sender,
            ));

            tasks.push(h);
            discover_responses.push(discover_receiver);
            commands.push(command_sender)
        });

    run_batch_inplace_delete(tasks, commands, discover_responses, batch).await
}

async fn run_batch_inplace_delete<I>(
    tasks: Vec<tokio::task::JoinHandle<()>>,
    commands: Vec<mpsc::Sender<Command<I>>>,
    discover_responses: Vec<oneshot::Receiver<ANNResult<Edges<I>>>>,
    ids: Arc<[I]>,
) -> ANNResult<()>
where
    I: VectorId,
{
    // FIXME: Handle errors by scanning over tasks and finding panicked workers.
    run_batch_happy(commands, discover_responses, ids).await
}

async fn run_batch_happy<I>(
    commands: Vec<mpsc::Sender<Command<I>>>,
    discover_responses: Vec<oneshot::Receiver<ANNResult<Edges<I>>>>,
    ids: Arc<[I]>,
) -> ANNResult<()>
where
    I: VectorId,
{
    use futures_util::{StreamExt, stream::FuturesUnordered};

    // Aggregate all edges as they become available.
    let mut edges = Edges::new();
    {
        let mut pending: FuturesUnordered<_> = discover_responses.into_iter().collect();
        while let Some(response) = pending.next().await {
            let discovered = response.map_err(ANNError::new)??;
            edges.merge(discovered);
        }
    }

    // Run all updates.
    {
        let balancer = Arc::new(Balancer::new(edges.into_iter().collect()));
        let mut pending: FuturesUnordered<_> = commands
            .iter()
            .map(|command| Command::apply_edge(command, &balancer))
            .collect();

        while let Some(response) = pending.next().await {
            response?;
        }
    }

    // Drop adjacency lists.
    {
        let balancer = Arc::new(Balancer::new(ids));
        let mut pending: FuturesUnordered<_> = commands
            .iter()
            .map(|command| Command::clear(command, &balancer))
            .collect();

        while let Some(response) = pending.next().await {
            response?;
        }
    }

    Ok(())
}

#[derive(Debug)]
enum Command<I> {
    ApplyEdge {
        balancer: Arc<Balancer<Edge<I>, Vec<Edge<I>>>>,
        response: oneshot::Sender<ANNResult<()>>,
    },
    Clear {
        balancer: Arc<Balancer<I, Arc<[I]>>>,
        response: oneshot::Sender<ANNResult<()>>,
    },
}

impl<I> Command<I>
where
    I: AsyncFriendly,
{
    async fn apply_edge(
        command: &mpsc::Sender<Self>,
        balancer: &Arc<Balancer<Edge<I>, Vec<Edge<I>>>>,
    ) -> ANNResult<()> {
        let (response, receiver) = oneshot::channel();
        let balancer = balancer.clone();
        command
            .send(Self::ApplyEdge { balancer, response })
            .await
            .map_err(ANNError::new)?;
        receiver.await.map_err(ANNError::new)?
    }

    async fn clear(
        command: &mpsc::Sender<Self>,
        balancer: &Arc<Balancer<I, Arc<[I]>>>,
    ) -> ANNResult<()> {
        let (response, receiver) = oneshot::channel();
        let balancer = balancer.clone();
        command
            .send(Self::Clear { balancer, response })
            .await
            .map_err(ANNError::new)?;
        receiver.await.map_err(ANNError::new)?
    }
}

async fn worker<S>(
    mut worker: S,
    deleted: Arc<HashSet<S::Id>>,
    config: Config,
    mut command: mpsc::Receiver<Command<S::Id>>,
    discover_balancer: Arc<Balancer<S::Id, Arc<[S::Id]>>>,
    discover_response: oneshot::Sender<ANNResult<Edges<S::Id>>>,
) where
    S: Standard,
{
    let mut workspace = Workspace::new();

    // Immediately on start up, begin discovering edges.
    let result = discover_edges(&mut worker, &mut workspace, &discover_balancer, &config).await;

    if discover_response.send(result).is_err() {
        // Gracefully exit.
        return;
    }

    // Enter command loop.
    while let Some(command) = command.recv().await {
        match command {
            Command::ApplyEdge { balancer, response } => {
                let result =
                    apply_edges(&mut worker, &mut workspace, &balancer, &deleted, &config).await;
                if response.send(result).is_err() {
                    // Gracefully exit.
                    return;
                }
            }
            Command::Clear { balancer, response } => {
                let result = clear_edges(&mut worker, &balancer).await;

                if response.send(result).is_err() {
                    // Gracefully exit.
                    return;
                }
            }
        }
    }
}

async fn discover_edges<S>(
    standard: &mut S,
    workspace: &mut Workspace<S::Id>,
    balancer: &Balancer<S::Id, Arc<[S::Id]>>,
    config: &Config,
) -> ANNResult<Edges<S::Id>>
where
    S: Standard,
{
    let mut edges = Edges::new();
    while let Some(batch) = balancer.next() {
        standard
            .discover(*batch.current(), workspace, &mut edges, &config)
            .await?;
    }
    Ok(edges)
}

async fn apply_edges<S>(
    standard: &mut S,
    workspace: &mut Workspace<S::Id>,
    balancer: &Balancer<Edge<S::Id>, Vec<Edge<S::Id>>>,
    deleted: &HashSet<S::Id>,
    config: &Config,
) -> ANNResult<()>
where
    S: Standard,
{
    let mut accessor = standard.prune_accessor()?;
    while let Some(edge) = balancer.next() {
        workspace
            .apply_edge(accessor.borrow_mut(), edge.current(), deleted, config)
            .await?;
    }
    Ok(())
}

async fn clear_edges<S>(standard: &mut S, balancer: &Balancer<S::Id, Arc<[S::Id]>>) -> ANNResult<()>
where
    S: Standard,
{
    let mut accessor = standard.prune_accessor()?;
    let mut neighbors = accessor.borrow_mut().neighbors();

    while let Some(id) = balancer.next() {
        neighbors.borrow_mut().clear(*id.current()).await?;
    }
    Ok(())
}
