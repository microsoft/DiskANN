/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{borrow::BorrowMut, fmt::Debug, sync::Arc};

use diskann_utils::future::{AsyncFriendly, SendFuture};
use hashbrown::HashMap;
use tokio::sync::{mpsc, oneshot};

use crate::{
    ANNError, ANNResult,
    error::ErrorExt,
    graph::{
        AdjacencyList, Config, algorithms,
        glue::PruneAccessor,
        internal::{BackedgeBuffer, prune},
    },
    provider::HasId,
    utils::{
        VectorId,
        balancer::{Balancer, BatchPosition},
    },
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
        Self { edges: Vec::new() }
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

    fn into_inner(self) -> Vec<ForwardEdge<I>> {
        self.edges
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
        Self {
            edges: HashMap::new(),
        }
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
        Some(BackEdge::new(source, destinations))
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

//-----------//
// Workspace //
//-----------//

#[derive(Debug)]
pub struct Workspace<I>
where
    I: VectorId,
{
    find: algorithms::insert::Workspace<I>,
    buf: AdjacencyList<I>,
}

impl<I> Workspace<I>
where
    I: VectorId,
{
    pub fn new() -> Self {
        Self {
            find: algorithms::insert::Workspace::new(),
            buf: AdjacencyList::new(),
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
        A: algorithms::insert::Access<Id = I>,
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

    pub async fn bootstrap<A>(
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
        algorithms::insert::robust_prune_list(
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

    pub async fn apply_back_edge<A>(
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

        algorithms::insert::add_edge_and_prune(
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

//----------//
// Provided //
//----------//

pub trait Standard: std::fmt::Debug + HasId + Send + Sync + 'static {
    type InsertAccessor<'a>: algorithms::insert::Access<Id = Self::Id>;
    type PruneAccessor<'a>: PruneAccessor<Id = Self::Id>;

    fn insert_accessor(
        &mut self,
        id: Self::Id,
    ) -> ANNResult<impl std::borrow::BorrowMut<Self::InsertAccessor<'_>> + Send>;

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
    let mut forward_responses = Vec::with_capacity(ntasks.get());

    let balancer = Arc::new(Balancer::new(batch));
    std::iter::repeat_with(make)
        .take(ntasks.get())
        .for_each(|standard| {
            let config_clone = config.clone();
            let balancer_clone = balancer.clone();

            let (forward_sender, forward_receiver) = oneshot::channel();
            let (command_sender, command_receiver) = mpsc::channel(1);

            let h = spawn.spawn(worker(
                standard,
                config_clone,
                command_receiver,
                balancer_clone,
                forward_sender,
            ));

            tasks.push(h);
            forward_responses.push(forward_receiver);
            commands.push(command_sender);
        });

    run_batch_insert(tasks, commands, forward_responses, config).await
}

/// This is the actual orchestration implementation. It is only generic over the ID type `I`
/// to only monomorphize once regardless of the type of the [`Standard`] or the construction
/// closure.
///
/// This is split into an inner happy path and outer error builder. Decomposing the
/// implementation this way allows the happy path to use standard "?" for error propagation.
async fn run_batch_insert<I>(
    tasks: Vec<tokio::task::JoinHandle<()>>,
    commands: Vec<mpsc::Sender<Command<I>>>,
    forward_responses: Vec<oneshot::Receiver<ANNResult<ForwardEdges<I>>>>,
    config: &Config,
) -> ANNResult<()>
where
    I: VectorId,
{
    // FIXME: Handle errors by scanning over tasks and finding panicked workers.
    run_batch_happy(commands, forward_responses, config).await
}

async fn run_batch_happy<I>(
    commands: Vec<mpsc::Sender<Command<I>>>,
    forward_responses: Vec<oneshot::Receiver<ANNResult<ForwardEdges<I>>>>,
    config: &Config,
) -> ANNResult<()>
where
    I: VectorId,
{
    use futures_util::{StreamExt, stream::FuturesUnordered};

    // Aggregate all forward and back edges.
    //
    // A `FuturesUnordered` is used too process results potentially out of order. The
    // control task eagerly merges edges to try to get a bit more concurrency.
    let mut forward = ForwardEdges::new();
    let mut backward = BackEdges::new();

    {
        let mut pending: FuturesUnordered<_> = forward_responses.into_iter().collect();
        while let Some(response) = pending.next().await {
            let edges = response.map_err(ANNError::new)??;
            backward.accumulate(&edges);
            forward.merge(edges);
        }
    }

    // Perform bootstrap if needed.
    let (forward, backward) = match Candidates::check(config, forward, backward) {
        Candidates::Ready { forward, backward } => (forward, backward),
        Candidates::BootstrapNeeded { forward } => {
            let mut recomputed_forward = ForwardEdges::new();
            let mut recomputed_backward = BackEdges::new();

            let balancer = Arc::new(Balancer::new(forward.into_inner()));
            let mut pending: FuturesUnordered<_> = commands
                .iter()
                .map(|command| Command::bootstrap(command, &balancer))
                .collect();

            while let Some(response) = pending.next().await {
                let edges = response?;
                recomputed_backward.accumulate(&edges);
                recomputed_forward.merge(edges);
            }

            (recomputed_forward, recomputed_backward.skip_bootstrap())
        }
    };

    // Apply forward edges.
    {
        let balancer = Arc::new(Balancer::new(forward.into_inner()));
        let mut pending: FuturesUnordered<_> = commands
            .iter()
            .map(|command| Command::apply_forward(command, &balancer))
            .collect();

        while let Some(response) = pending.next().await {
            response?;
        }
    }

    // Apply back edges.
    {
        let balancer = Arc::new(Balancer::new(backward.collect::<Vec<_>>()));
        let mut pending: FuturesUnordered<_> = commands
            .iter()
            .map(|command| Command::apply_backward(command, &balancer))
            .collect();

        while let Some(response) = pending.next().await {
            response?;
        }
    }

    Ok(())
}

#[derive(Debug)]
enum Command<I> {
    Bootstrap {
        balancer: Arc<Balancer<ForwardEdge<I>, Vec<ForwardEdge<I>>>>,
        response: oneshot::Sender<ANNResult<ForwardEdges<I>>>,
    },
    ApplyForward {
        balancer: Arc<Balancer<ForwardEdge<I>, Vec<ForwardEdge<I>>>>,
        response: oneshot::Sender<ANNResult<()>>,
    },
    ApplyBackward {
        balancer: Arc<Balancer<BackEdge<I>, Vec<BackEdge<I>>>>,
        response: oneshot::Sender<ANNResult<()>>,
    },
}

impl<I> Command<I>
where
    I: AsyncFriendly,
{
    async fn bootstrap(
        command: &mpsc::Sender<Self>,
        balancer: &Arc<Balancer<ForwardEdge<I>, Vec<ForwardEdge<I>>>>,
    ) -> ANNResult<ForwardEdges<I>> {
        let (response, receiver) = oneshot::channel();
        let balancer = balancer.clone();
        command
            .send(Self::Bootstrap { balancer, response })
            .await
            .map_err(ANNError::new)?;
        receiver.await.map_err(ANNError::new)?
    }

    async fn apply_forward(
        command: &mpsc::Sender<Self>,
        balancer: &Arc<Balancer<ForwardEdge<I>, Vec<ForwardEdge<I>>>>,
    ) -> ANNResult<()> {
        let (response, receiver) = oneshot::channel();
        let balancer = balancer.clone();
        command
            .send(Self::ApplyForward { balancer, response })
            .await
            .map_err(ANNError::new)?;
        receiver.await.map_err(ANNError::new)?
    }

    async fn apply_backward(
        command: &mpsc::Sender<Self>,
        balancer: &Arc<Balancer<BackEdge<I>, Vec<BackEdge<I>>>>,
    ) -> ANNResult<()> {
        let (response, receiver) = oneshot::channel();
        let balancer = balancer.clone();
        command
            .send(Self::ApplyBackward { balancer, response })
            .await
            .map_err(ANNError::new)?;
        receiver.await.map_err(ANNError::new)?
    }
}

async fn worker<S>(
    mut worker: S,
    config: Config,
    mut command: mpsc::Receiver<Command<S::Id>>,
    forward_balancer: Arc<Balancer<S::Id, Arc<[S::Id]>>>,
    forward_response: oneshot::Sender<ANNResult<ForwardEdges<S::Id>>>,
) where
    S: Standard,
{
    let mut workspace = Workspace::new();

    // Immediately on start up, begin computing forward edges.
    let result = compute_forward(&mut worker, &mut workspace, &forward_balancer, &config).await;

    if forward_response.send(result).is_err() {
        // Gracefully exit.
        return;
    }

    // Enter command loop.
    while let Some(command) = command.recv().await {
        match command {
            Command::Bootstrap { balancer, response } => {
                let result = bootstrap(&mut worker, &mut workspace, &balancer, &config).await;
                if response.send(result).is_err() {
                    // Gracefully exit.
                    return;
                }
            }
            Command::ApplyForward { balancer, response } => {
                let result = apply_forward(&mut worker, &balancer).await;

                if response.send(result).is_err() {
                    // Gracefully exit.
                    return;
                }
            }
            Command::ApplyBackward { balancer, response } => {
                let result = apply_backward(&mut worker, &mut workspace, &balancer, &config).await;

                if response.send(result).is_err() {
                    // Gracefully exit.
                    return;
                }
            }
        }
    }
}

async fn compute_forward<S>(
    standard: &mut S,
    workspace: &mut Workspace<S::Id>,
    balancer: &Balancer<S::Id, Arc<[S::Id]>>,
    config: &Config,
) -> ANNResult<ForwardEdges<S::Id>>
where
    S: Standard,
{
    let mut edges = ForwardEdges::new();
    while let Some(batch) = balancer.next() {
        let mut accessor = standard.insert_accessor(*batch.current())?;
        workspace
            .find_forward_edges(&mut edges, accessor.borrow_mut(), batch, config)
            .await?;
    }
    Ok(edges)
}

async fn bootstrap<S>(
    standard: &mut S,
    workspace: &mut Workspace<S::Id>,
    balancer: &Balancer<ForwardEdge<S::Id>, Vec<ForwardEdge<S::Id>>>,
    config: &Config,
) -> ANNResult<ForwardEdges<S::Id>>
where
    S: Standard,
{
    let mut edges = ForwardEdges::new();
    let mut accessor = standard.prune_accessor()?;
    while let Some(batch) = balancer.next() {
        workspace
            .bootstrap(&mut edges, accessor.borrow_mut(), batch, config)
            .await?;
    }
    Ok(edges)
}

async fn apply_forward<S>(
    standard: &mut S,
    balancer: &Balancer<ForwardEdge<S::Id>, Vec<ForwardEdge<S::Id>>>,
) -> ANNResult<()>
where
    S: Standard,
{
    use crate::provider::NeighborAccessorMut;

    let mut accessor = standard.prune_accessor()?;
    while let Some(edge) = balancer.next() {
        let edge = edge.current();
        accessor
            .borrow_mut()
            .neighbors()
            .borrow_mut()
            .set_neighbors(*edge.source(), edge.destinations())
            .await?;
    }
    Ok(())
}

async fn apply_backward<S>(
    standard: &mut S,
    workspace: &mut Workspace<S::Id>,
    balancer: &Balancer<BackEdge<S::Id>, Vec<BackEdge<S::Id>>>,
    config: &Config,
) -> ANNResult<()>
where
    S: Standard,
{
    let mut accessor = standard.prune_accessor()?;
    while let Some(edge) = balancer.next() {
        workspace
            .apply_back_edge(accessor.borrow_mut(), edge.current(), config)
            .await?;
    }
    Ok(())
}
