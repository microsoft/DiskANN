/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Compatibility contracts for provider- and strategy-based integrations.
//!
//! These contracts describe lifecycle management and accessor construction around the
//! accessor-based algorithms. Post-processing uses the traits in [`crate::graph::glue`].

use std::{borrow::BorrowMut, future::Future, num::NonZeroUsize, sync::Arc};

use diskann_utils::{Reborrow, future::SendFuture};
use hashbrown::{HashMap, HashSet};
use thiserror::Error;

use crate::{
    ANNError, ANNResult, HasId,
    error::{ErrorExt, IntoANNResult, StandardError, ToRanked},
    graph::{
        self, AdjacencyList, SearchOutputBuffer,
        adjacencylist::Empty,
        algorithms::{
            self,
            consolidate::ConsolidateKind,
            inplace_delete::{self, OneHop, TwoHopAndOneHop, VisitedAndTopK},
        },
        glue::{
            Batch, DefaultPostProcessor, FilteredAccessor, NeighborAccessor, NeighborAccessorMut,
            PruneAccessor, SearchAccessor, SearchPostProcess,
        },
        search::{FilteredSearch, PagedSearch, Search},
    },
    utils::{Bool, VectorId},
};

///////////
// Index //
///////////

#[derive(Debug)]
pub struct DiskANNIndex<DP> {
    provider: DP,
    config: graph::Config,
}

impl<DP> DiskANNIndex<DP> {
    pub fn new(provider: DP, config: graph::Config) -> Self {
        Self { provider, config }
    }

    pub fn provider(&self) -> &DP {
        &self.provider
    }

    pub fn provider_mut(&mut self) -> &mut DP {
        &mut self.provider
    }

    pub fn config(&self) -> &graph::Config {
        &self.config
    }

    //--------//
    // Search //
    //--------//

    /// Search using the accessor's default post-processor.
    pub fn search<'a, S, T, O, OB, P>(
        &'a self,
        search_params: P,
        strategy: &'a S,
        context: &'a DP::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<P::Output>>
    where
        DP: DataProvider,
        S: SearchStrategy<'a, DP, T>,
        S::SearchAccessor: SearchAccessor + DefaultPostProcessor<O>,
        P: Search<DP::InternalId> + Send,
        T: Send,
        O: Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(self.provider(), context, query)
                .into_ann_result()?;
            let processor = accessor.default_post_processor();
            search_params.search(&mut accessor, processor, output).await
        }
    }

    /// Search using an explicit post-processor.
    pub fn search_with<'a, S, T, O, OB, P, PP>(
        &'a self,
        search_params: P,
        strategy: &'a S,
        processor: PP,
        context: &'a DP::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<P::Output>>
    where
        DP: DataProvider,
        S: SearchStrategy<'a, DP, T>,
        S::SearchAccessor: SearchAccessor,
        P: Search<DP::InternalId> + Send,
        PP: SearchPostProcess<S::SearchAccessor, O> + Send,
        T: Send,
        O: Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(self.provider(), context, query)
                .into_ann_result()?;
            search_params.search(&mut accessor, processor, output).await
        }
    }

    /// Filtered search using the accessor's default post-processor.
    pub fn filtered_search<'a, S, T, O, OB, P>(
        &'a self,
        search_params: P,
        strategy: &'a S,
        context: &'a DP::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<P::Output>>
    where
        DP: DataProvider,
        S: SearchStrategy<'a, DP, T>,
        S::SearchAccessor: FilteredAccessor + DefaultPostProcessor<O>,
        P: FilteredSearch<DP::InternalId> + Send,
        T: Send,
        O: Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(self.provider(), context, query)
                .into_ann_result()?;
            let processor = accessor.default_post_processor();
            search_params
                .filtered_search(&mut accessor, processor, output)
                .await
        }
    }

    /// Filtered search using an explicit post-processor.
    pub fn filtered_search_with<'a, S, T, O, OB, P, PP>(
        &'a self,
        search_params: P,
        strategy: &'a S,
        processor: PP,
        context: &'a DP::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<P::Output>>
    where
        DP: DataProvider,
        S: SearchStrategy<'a, DP, T>,
        S::SearchAccessor: FilteredAccessor,
        P: FilteredSearch<DP::InternalId> + Send,
        PP: SearchPostProcess<S::SearchAccessor, O> + Send,
        T: Send,
        O: Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(self.provider(), context, query)
                .into_ann_result()?;
            search_params
                .filtered_search(&mut accessor, processor, output)
                .await
        }
    }

    /// Begin a paged search using the accessor's starting points.
    ///
    /// # Errors
    ///
    /// Returns an error if `l_value` is zero, accessor construction fails, or search
    /// initialization fails.
    pub fn paged_search<'a, S, T>(
        &'a self,
        strategy: &'a S,
        context: &'a DP::Context,
        query: T,
        l_value: usize,
    ) -> impl SendFuture<ANNResult<PagedSearch<S::SearchAccessor>>>
    where
        DP: DataProvider,
        S: SearchStrategy<'a, DP, T>,
        S::SearchAccessor: SearchAccessor,
        T: Send,
    {
        async move {
            let l_value = NonZeroUsize::new(l_value)
                .ok_or_else(|| ANNError::message("paged search requires a non-zero l_value"))?;
            let accessor = strategy
                .search_accessor(self.provider(), context, query)
                .into_ann_result()?;
            PagedSearch::new(accessor, l_value).await
        }
    }

    //-------------//
    // Maintenance //
    //-------------//

    pub fn is_any_neighbor_deleted<NA>(
        &self,
        context: &DP::Context,
        accessor: &mut NA,
        vector_id: DP::InternalId,
    ) -> impl SendFuture<ANNResult<bool>>
    where
        DP: Delete,
        NA: NeighborAccessor<Id = DP::InternalId>,
    {
        async move {
            let mut neighbors = AdjacencyList::new();
            accessor
                .get_neighbors(vector_id, neighbors.as_empty())
                .await?;
            for &id in neighbors.iter() {
                let status = self
                    .provider
                    .status_by_internal_id(context, id)
                    .await
                    .allow_transient("a neighbor may disappear during a deletion check")?;
                if !status.is_some_and(|status| status.is_valid()) {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }

    pub fn drop_deleted_neighbors<NA>(
        &self,
        context: &DP::Context,
        accessor: &mut NA,
        vector_id: DP::InternalId,
        only_orphans: bool,
    ) -> impl SendFuture<ANNResult<ConsolidateKind>>
    where
        DP: Delete,
        NA: NeighborAccessorMut<Id = DP::InternalId>,
    {
        async move {
            let status = self
                .provider
                .status_by_internal_id(context, vector_id)
                .await
                .escalate("`drop_deleted_neighbors` should only be called on valid IDs")?;
            if status.is_deleted() {
                return Ok(ConsolidateKind::Deleted);
            }

            let mut neighbors = AdjacencyList::new();
            accessor
                .get_neighbors(vector_id, neighbors.as_empty())
                .await?;
            let mut retained = AdjacencyList::with_capacity(neighbors.len());
            let mut outgoing = AdjacencyList::new();
            for &id in neighbors.iter() {
                let status = self
                    .provider
                    .status_by_internal_id(context, id)
                    .await
                    .allow_transient("a neighbor may disappear during deletion cleanup")?;
                let Some(status) = status else {
                    continue;
                };
                if status.is_valid() {
                    retained.push_unique(id);
                } else if only_orphans {
                    accessor.get_neighbors(id, outgoing.as_empty()).await?;
                    if !outgoing.is_empty() {
                        retained.push_unique(id);
                    }
                }
            }

            if retained.len() != neighbors.len() {
                accessor.set_neighbors(vector_id, &retained).await?;
            }
            Ok(ConsolidateKind::Complete)
        }
    }

    /// Remove all outgoing edges from `vector_id`.
    ///
    /// This does not logically delete the vector or remove incoming references to it.
    pub fn drop_adj_list<NA>(
        &self,
        accessor: &mut NA,
        vector_id: DP::InternalId,
    ) -> impl SendFuture<ANNResult<()>>
    where
        DP: DataProvider,
        NA: NeighborAccessorMut<Id = DP::InternalId>,
    {
        accessor.clear(vector_id)
    }

    /// Count distinct nodes reachable from `start_points`.
    ///
    /// Memory consumption grows with the number of reachable nodes.
    pub fn count_reachable_nodes<NA>(
        &self,
        start_points: &[DP::InternalId],
        accessor: &mut NA,
    ) -> impl SendFuture<ANNResult<usize>>
    where
        DP: DataProvider,
        NA: NeighborAccessor<Id = DP::InternalId>,
    {
        algorithms::measure::reachable_from(accessor, start_points)
    }

    /// Logically delete a vector, repair its neighborhood, and clear its outgoing edges.
    ///
    /// Remaining incoming references require a separate cleanup pass. This does not release
    /// the vector's storage, and a failure after logical deletion does not roll it back.
    ///
    /// # Compat
    ///
    /// Counts must be non-zero. Delete-search candidates use the accessor's traversal
    /// distances and provider liveness queries, not the legacy strategy's post-processor.
    ///
    /// # Errors
    ///
    /// Returns an error for zero replacement/search counts, or if ID translation, accessor
    /// construction, deletion, liveness queries, or graph repair fails.
    pub fn inplace_delete<S>(
        &self,
        strategy: S,
        context: &DP::Context,
        id: &DP::ExternalId,
        num_to_replace: usize,
        inplace_delete_method: InplaceDeleteMethod,
    ) -> impl SendFuture<ANNResult<()>>
    where
        DP: Delete,
        S: InplaceDeleteStrategy<DP>,
    {
        async move {
            let replacements = NonZeroUsize::new(num_to_replace).ok_or_else(|| {
                ANNError::message("inplace delete requires non-zero replacements")
            })?;
            let internal_id = self
                .provider
                .to_internal_id(context, id)
                .escalate("id translation for `inplace_delete` must succeed")?;
            let prune_strategy = strategy.prune_strategy();
            let prune = prune_strategy
                .prune_accessor(
                    self.provider(),
                    context,
                    self.config.max_occlusion_size().get(),
                )
                .into_ann_result()?;
            let mut accessor = InplaceDelete {
                provider: self.provider(),
                context,
                prune,
                search: (),
                live: AdjacencyList::new(),
            };
            let mut workspace = inplace_delete::Workspace::new();

            match inplace_delete_method {
                InplaceDeleteMethod::OneHop => {
                    self.provider
                        .delete(context, id)
                        .await
                        .escalate("`inplace_delete` requires a successful delete")?;
                    workspace
                        .one_hop(
                            &OneHop::new(replacements),
                            &mut accessor,
                            internal_id,
                            &self.config,
                        )
                        .await
                }
                InplaceDeleteMethod::TwoHopAndOneHop => {
                    self.provider
                        .delete(context, id)
                        .await
                        .escalate("`inplace_delete` requires a successful delete")?;
                    workspace
                        .two_hop_and_one_hop(
                            &TwoHopAndOneHop::new(replacements),
                            &mut accessor,
                            internal_id,
                            &self.config,
                        )
                        .await
                }
                InplaceDeleteMethod::VisitedAndTopK { k_value, l_value } => {
                    let k_value = NonZeroUsize::new(k_value).ok_or_else(|| {
                        ANNError::message("inplace delete requires a non-zero k_value")
                    })?;
                    let l_value = NonZeroUsize::new(l_value).ok_or_else(|| {
                        ANNError::message("inplace delete requires a non-zero l_value")
                    })?;
                    let element = strategy
                        .get_delete_element(self.provider(), context, internal_id)
                        .await
                        .into_ann_result()?;
                    self.provider
                        .delete(context, id)
                        .await
                        .escalate("`inplace_delete` requires a successful delete")?;
                    let search_strategy = strategy.search_strategy();
                    let search = search_strategy
                        .search_accessor(self.provider(), context, element.reborrow())
                        .into_ann_result()?;
                    let mut accessor = accessor.with_search(search);
                    workspace
                        .visited_and_topk(
                            &VisitedAndTopK {
                                replacements,
                                l_value,
                                k_value,
                            },
                            &mut accessor,
                            internal_id,
                            &self.config,
                        )
                        .await
                }
            }
        }
    }

    pub fn multi_inplace_delete<S>(
        self: &Arc<Self>,
        strategy: S,
        context: &DP::Context,
        ids: Arc<[DP::ExternalId]>,
        num_to_replace: usize,
        inplace_delete_method: InplaceDeleteMethod,
    ) -> impl SendFuture<ANNResult<()>>
    where
        DP: Delete,
        S: InplaceDeleteStrategy<DP>,
    {
        async move {
            let replacements = NonZeroUsize::new(num_to_replace).ok_or_else(|| {
                ANNError::message("multi inplace delete requires non-zero replacements")
            })?;
            let kind = match inplace_delete_method {
                InplaceDeleteMethod::OneHop => OneHop::new(replacements).into(),
                InplaceDeleteMethod::TwoHopAndOneHop => TwoHopAndOneHop::new(replacements).into(),
                InplaceDeleteMethod::VisitedAndTopK { k_value, l_value } => {
                    let k_value = NonZeroUsize::new(k_value).ok_or_else(|| {
                        ANNError::message("multi inplace delete requires a non-zero k_value")
                    })?;
                    let l_value = NonZeroUsize::new(l_value).ok_or_else(|| {
                        ANNError::message("multi inplace delete requires a non-zero l_value")
                    })?;
                    inplace_delete::InplaceDeleteMethod::VisitedAndTopK(VisitedAndTopK {
                        replacements,
                        l_value,
                        k_value,
                    })
                }
            };

            let mut internal_ids = Vec::with_capacity(ids.len());
            let mut unique = HashSet::with_capacity(ids.len());
            for id in ids.iter() {
                let internal_id = self
                    .provider
                    .to_internal_id(context, id)
                    .escalate("id translation for `multi_inplace_delete` must succeed")?;
                if !unique.insert(internal_id) {
                    return Err(ANNError::message(
                        "multi inplace delete requires unique internal IDs",
                    ));
                }
                internal_ids.push(internal_id);
            }
            if internal_ids.is_empty() {
                return Ok(());
            }

            // Capture queries before logical deletion can invalidate their external mapping.
            let mut elements = HashMap::new();
            if matches!(kind, inplace_delete::InplaceDeleteMethod::VisitedAndTopK(_)) {
                for &id in &internal_ids {
                    let element = strategy
                        .get_delete_element(self.provider(), context, id)
                        .await
                        .into_ann_result()?;
                    elements.insert(id, element);
                }
            }

            let prune = strategy.prune_strategy();
            for id in ids.iter() {
                self.provider
                    .delete(context, id)
                    .await
                    .escalate("`multi_inplace_delete` requires successful deletes")?;
            }

            let worker = Arc::new(BatchInplaceDelete {
                index: self.clone(),
                context: context.clone(),
                strategy,
                prune,
                kind,
                elements,
            });
            algorithms::batch::inplace_delete::standard(
                &ScopedSpawn(context),
                internal_ids.into(),
                &self.config,
                || worker.clone(),
            )
            .await
        }
    }

    /// Consider the neighbors of `vector_id`, look for deleted nodes and compute replacements for them.
    pub fn consolidate_vector<S>(
        &self,
        strategy: &S,
        context: &DP::Context,
        vector_id: DP::InternalId,
    ) -> impl SendFuture<ANNResult<ConsolidateKind>>
    where
        DP: Delete,
        S: PruneStrategy<DP>,
    {
        async move {
            let prune = strategy
                .prune_accessor(
                    self.provider(),
                    context,
                    self.config.max_occlusion_size().get(),
                )
                .into_ann_result()?;
            let mut accessor = Consolidate {
                provider: self.provider(),
                context,
                prune,
                partition: Vec::new(),
            };
            algorithms::consolidate::Workspace::new()
                .run(&mut accessor, vector_id, &self.config)
                .await
        }
    }

    pub fn prune_range<S, I>(
        &self,
        strategy: &S,
        context: &DP::Context,
        ids: I,
    ) -> impl SendFuture<ANNResult<()>>
    where
        DP: DataProvider,
        S: PruneStrategy<DP>,
        I: IntoIterator<Item = DP::InternalId, IntoIter: Send> + Send,
    {
        async move {
            let mut accessor = strategy
                .prune_accessor(
                    self.provider(),
                    context,
                    self.config.max_occlusion_size().get(),
                )
                .into_ann_result()?;
            let mut workspace = algorithms::insert::Workspace::new();
            for id in ids {
                workspace
                    .prune_node(&mut accessor, id, &self.config)
                    .await?;
            }
            Ok(())
        }
    }

    //--------//
    // Insert //
    //--------//

    /// Insert a vector into the index with the given `id` and `vector`.
    ///
    /// # Compat
    ///
    /// This function has minimal behavior changes relative to version 0.62.0.
    pub fn insert<'a, S, T>(
        &'a self,
        strategy: &'a S,
        context: &'a DP::Context,
        id: &DP::ExternalId,
        vector: T,
    ) -> impl SendFuture<ANNResult<()>>
    where
        S: InsertStrategy<'a, DP, T>,
        DP: SetElement<T>,
        T: Copy + Send,
    {
        self.insert_inner(strategy, context, id, vector)
    }

    async fn insert_inner<'a, S, T>(
        &'a self,
        strategy: &'a S,
        context: &'a DP::Context,
        id: &DP::ExternalId,
        vector: T,
    ) -> ANNResult<()>
    where
        S: InsertStrategy<'a, DP, T>,
        DP: SetElement<T>,
        T: Copy + Send,
    {
        let guard = self
            .provider
            .set_element(context, id, vector)
            .await
            .escalate("insertion requires a successful `set_element`")?;

        let prune_strategy = strategy.prune_strategy();
        let mut insert = Insert {
            search: strategy
                .insert_search_accessor(&self.provider, context, vector)
                .into_ann_result()?,
            prune: prune_strategy
                .prune_accessor(
                    &self.provider,
                    context,
                    self.config.max_occlusion_size().get(),
                )
                .into_ann_result()?,
        };

        let mut workspace = algorithms::insert::Workspace::new();
        workspace.run(&mut insert, guard.id(), &self.config).await?;

        guard.complete().await;
        Ok(())
    }

    /// # Compat
    ///
    /// Changes relative to 0.62.0:
    ///
    /// * Invocation of batch `SetElement` is single-threaded instead of partitioned across
    ///   multiple `tokio` tasks.
    ///
    /// * Total number of tasks spawns reduced to `self.config().max_minibatch_par()`.
    ///   Spawned tasks are reused for all insert stages.
    pub fn multi_insert<S, B>(
        self: &Arc<Self>,
        strategy: S,
        context: &DP::Context,
        vectors: Arc<B>,
        ids: Arc<[DP::ExternalId]>,
    ) -> impl SendFuture<ANNResult<()>>
    where
        Self: 'static,
        S: MultiInsertStrategy<DP, B>,
        B: Batch,
        DP: for<'a> SetElement<B::Element<'a>>,
    {
        async move {
            if vectors.len() != ids.len() {
                return Err(ANNError::new(BatchIdMismatch {
                    batch_len: vectors.len(),
                    ids_len: ids.len(),
                }));
            }

            // Set all elements.
            let mut map = HashMap::new();
            let mut internal_ids = Vec::new();
            let mut guards = Vec::with_capacity(ids.len());
            for (position, id) in ids.iter().enumerate() {
                let guard = self
                    .provider
                    .set_element(context, id, vectors.get(position))
                    .await
                    .escalate("cannot support failures during `set_element` in multi-insert")
                    .into_ann_result()?;

                map.insert(guard.id(), position);
                internal_ids.push(guard.id());
                guards.push(guard);
            }

            let seed = strategy
                .finish(
                    self.provider(),
                    context,
                    &vectors,
                    guards.iter().map(|i| i.id()),
                )
                .await
                .into_ann_result()?;

            let insert = strategy.insert_strategy();
            let batch = Arc::new(BatchInsert {
                index: self.clone(),
                context: context.clone(),
                strategy,
                insert,
                vectors,
                map,
                seed,
            });

            // Run multi-insert.
            algorithms::batch::insert::standard(
                &ScopedSpawn(context),
                internal_ids.into_iter().collect(),
                &self.config,
                || batch.clone(),
            )
            .await?;

            // Acknowledge all guards.
            for guard in guards {
                guard.complete().await;
            }
            Ok(())
        }
    }
}

//--------//
// Insert //
//--------//

#[derive(Debug)]
struct Insert<S, P> {
    search: S,
    prune: P,
}

impl<S, P> HasId for Insert<S, P>
where
    S: HasId,
    P: HasId<Id = S::Id>,
{
    type Id = S::Id;
}

impl<S, P> algorithms::insert::Access for Insert<S, P>
where
    S: SearchAccessor,
    P: PruneAccessor<Id = S::Id>,
{
    type SearchAccessor<'a>
        = S
    where
        Self: 'a;

    type PruneAccessor<'a>
        = P
    where
        Self: 'a;

    fn search_accessor(&mut self) -> impl BorrowMut<S> + Send {
        &mut self.search
    }
    fn prune_accessor(&mut self) -> impl BorrowMut<P> + Send {
        &mut self.prune
    }
}

//----------------//
// Inplace Delete //
//----------------//

// enum used to encode the algorithmic choices for inplace delete
// the first term indicates what is used to approximate the in-neighbors
// the second term indicates what is used to approximate the replace
// candidates
// also includes any params specific to that choice
#[derive(Copy, Clone, Debug)]
pub enum InplaceDeleteMethod {
    VisitedAndTopK { k_value: usize, l_value: usize },
    TwoHopAndOneHop,
    OneHop,
}

struct InplaceDelete<'a, DP: DataProvider, P, S = ()> {
    provider: &'a DP,
    context: &'a DP::Context,
    prune: P,
    search: S,
    live: AdjacencyList<DP::InternalId>,
}

impl<'a, DP: DataProvider, P> InplaceDelete<'a, DP, P> {
    fn with_search<S>(self, search: S) -> InplaceDelete<'a, DP, P, S> {
        InplaceDelete {
            provider: self.provider,
            context: self.context,
            prune: self.prune,
            search,
            live: self.live,
        }
    }
}

impl<DP: DataProvider, P, S> HasId for InplaceDelete<'_, DP, P, S> {
    type Id = DP::InternalId;
}

impl<DP, P, S> inplace_delete::Access for InplaceDelete<'_, DP, P, S>
where
    DP: Delete,
    P: PruneAccessor<Id = DP::InternalId>,
    S: Send,
{
    type PruneAccessor<'a>
        = P
    where
        Self: 'a;

    fn prune_accessor(&mut self) -> impl BorrowMut<P> + Send {
        &mut self.prune
    }

    fn keep_live(&mut self, ids: &mut AdjacencyList<Self::Id>) -> impl SendFuture<ANNResult<()>> {
        async move {
            self.live.clear();
            for &id in ids.iter() {
                let status = self
                    .provider
                    .status_by_internal_id(self.context, id)
                    .await
                    .allow_transient("a candidate may disappear during inplace delete")?;
                if status.is_some_and(|status| status.is_valid()) {
                    self.live.push_unique(id);
                }
            }
            std::mem::swap(ids, &mut self.live);
            self.live.clear();
            Ok(())
        }
    }
}

impl<DP, P, S> inplace_delete::TopKAccess for InplaceDelete<'_, DP, P, S>
where
    DP: Delete,
    P: PruneAccessor<Id = DP::InternalId>,
    S: SearchAccessor<Id = DP::InternalId>,
{
    type SearchAccessor<'a>
        = S
    where
        Self: 'a;

    fn search_accessor(&mut self) -> impl BorrowMut<S> + Send {
        &mut self.search
    }
}

//--------------------//
// BatchInplaceDelete //
//--------------------//

struct BatchInplaceDelete<DP, S>
where
    DP: DataProvider,
    S: InplaceDeleteStrategy<DP>,
{
    index: Arc<DiskANNIndex<DP>>,
    context: DP::Context,
    strategy: S,
    prune: S::PruneStrategy,
    kind: inplace_delete::InplaceDeleteMethod,
    elements: HashMap<DP::InternalId, S::DeleteElementGuard>,
}

impl<DP, S> std::fmt::Debug for BatchInplaceDelete<DP, S>
where
    DP: DataProvider,
    S: InplaceDeleteStrategy<DP>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BatchInplaceDelete {{ .. }}")
    }
}

impl<DP, S> HasId for Arc<BatchInplaceDelete<DP, S>>
where
    DP: DataProvider,
    S: InplaceDeleteStrategy<DP>,
{
    type Id = DP::InternalId;
}

impl<DP, S> algorithms::batch::inplace_delete::Standard for Arc<BatchInplaceDelete<DP, S>>
where
    DP: Delete,
    S: InplaceDeleteStrategy<DP>,
{
    type PruneAccessor<'a>
        = <S::PruneStrategy as PruneStrategy<DP>>::PruneAccessor<'a>
    where
        Self: 'a;

    fn discover(
        &mut self,
        id: Self::Id,
        workspace: &mut algorithms::batch::inplace_delete::Workspace<Self::Id>,
        edges: &mut algorithms::batch::inplace_delete::Edges<Self::Id>,
        config: &graph::Config,
    ) -> impl SendFuture<ANNResult<()>> {
        async move {
            let prune = self
                .prune
                .prune_accessor(
                    self.index.provider(),
                    &self.context,
                    config.max_occlusion_size().get(),
                )
                .into_ann_result()?;
            let mut access = InplaceDelete {
                provider: self.index.provider(),
                context: &self.context,
                prune,
                search: (),
                live: AdjacencyList::new(),
            };
            match &self.kind {
                inplace_delete::InplaceDeleteMethod::OneHop(kind) => {
                    workspace.find_one_hop(edges, &mut access, kind, id).await
                }
                inplace_delete::InplaceDeleteMethod::TwoHopAndOneHop(kind) => {
                    workspace
                        .find_two_hop_and_one_hop(edges, &mut access, kind, id)
                        .await
                }
                inplace_delete::InplaceDeleteMethod::VisitedAndTopK(kind) => {
                    let element = self.elements.get(&id).ok_or_else(|| {
                        ANNError::message("missing captured query for batch inplace delete")
                    })?;
                    let search_strategy = self.strategy.search_strategy();
                    let search = search_strategy
                        .search_accessor(self.index.provider(), &self.context, element.reborrow())
                        .into_ann_result()?;
                    let mut access = access.with_search(search);
                    workspace
                        .find_visited_and_top_k(edges, &mut access, kind, id, config)
                        .await
                }
            }
        }
    }

    fn prune_accessor(&mut self) -> ANNResult<impl BorrowMut<Self::PruneAccessor<'_>> + Send> {
        self.prune
            .prune_accessor(
                self.index.provider(),
                &self.context,
                self.index.config().max_occlusion_size().get(),
            )
            .into_ann_result()
    }
}

//-------------//
// Consolidate //
//-------------//

#[derive(Debug)]
struct Consolidate<'a, DP: DataProvider, P> {
    provider: &'a DP,
    context: &'a DP::Context,
    prune: P,
    partition: Vec<bool>,
}

impl<DP, P> HasId for Consolidate<'_, DP, P>
where
    DP: DataProvider,
    P: HasId<Id = DP::InternalId>,
{
    type Id = DP::InternalId;
}

impl<DP, P> algorithms::consolidate::Access for Consolidate<'_, DP, P>
where
    DP: Delete,
    P: PruneAccessor<Id = DP::InternalId>,
{
    type PruneAccessor<'a>
        = P
    where
        Self: 'a;

    fn prune_accessor(&mut self) -> impl BorrowMut<P> + Send {
        &mut self.prune
    }

    async fn neighbors(&mut self, id: Self::Id, neighbors: Empty<'_, Self::Id>) -> ANNResult<Bool> {
        // RACE WARNING: There is a TOCTOU race between the deletion check and the neighbor
        // retrieval. This existed in older versions of this code and is part of the reason
        // the `DataProvider` interface is being deprecated.
        let status = self
            .provider
            .status_by_internal_id(self.context, id)
            .await
            .allow_transient("a vertex may disappear during consolidation")?;
        let Some(status) = status else {
            return Ok(false.into());
        };

        self.prune
            .neighbors()
            .borrow_mut()
            .get_neighbors(id, neighbors)
            .await?;
        Ok(status.is_valid().into())
    }

    async fn partition_live(
        &mut self,
        ids: &mut AdjacencyList<Self::Id>,
        deleted: Option<Empty<'_, Self::Id>>,
    ) -> ANNResult<()> {
        self.partition.clear();
        for &id in ids.iter() {
            let status = self
                .provider
                .status_by_internal_id(self.context, id)
                .await
                .allow_transient("a candidate may disappear during consolidation")?;
            self.partition
                .push(status.is_some_and(|status| status.is_valid()));
        }

        let mut deleted = deleted.map(Empty::take);
        let mut position = 0;
        ids.retain(|id| {
            let live = self.partition[position];
            position += 1;
            if !live && let Some(deleted) = deleted.as_mut() {
                deleted.push_unique(*id);
            }
            live
        });
        Ok(())
    }
}

//-------------//
// BatchInsert //
//-------------//

struct BatchInsert<DP, B, S>
where
    DP: DataProvider,
    B: Batch,
    S: MultiInsertStrategy<DP, B>,
{
    index: Arc<DiskANNIndex<DP>>,
    context: DP::Context,
    strategy: S,
    insert: S::InsertStrategy,
    vectors: Arc<B>,
    map: HashMap<DP::InternalId, usize>,
    seed: S::Seed,
}

impl<DP, B, S> std::fmt::Debug for BatchInsert<DP, B, S>
where
    DP: DataProvider,
    B: Batch,
    S: MultiInsertStrategy<DP, B>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BatchInsert {{ .. }}")
    }
}

impl<DP, B, S> HasId for Arc<BatchInsert<DP, B, S>>
where
    DP: DataProvider,
    B: Batch,
    S: MultiInsertStrategy<DP, B>,
{
    type Id = DP::InternalId;
}

impl<DP, B, S> algorithms::batch::insert::Standard for Arc<BatchInsert<DP, B, S>>
where
    DP: DataProvider,
    B: Batch,
    S: MultiInsertStrategy<DP, B>,
{
    type InsertAccessor<'a>
        = Insert<
        <S::InsertStrategy as InsertStrategy<'a, DP, B::Element<'a>>>::SearchAccessor,
        <S::PruneStrategy as PruneStrategy<DP>>::PruneAccessor<'a>,
    >
    where
        Self: 'a;

    type PruneAccessor<'a>
        = <S::PruneStrategy as PruneStrategy<DP>>::PruneAccessor<'a>
    where
        Self: 'a;

    fn insert_accessor(
        &mut self,
        id: Self::Id,
    ) -> ANNResult<impl BorrowMut<Self::InsertAccessor<'_>> + Send> {
        let vector = self.vectors.get(self.map[&id]);
        let provider = self.index.provider();

        let insert = Insert {
            search: self
                .insert
                .insert_search_accessor(provider, &self.context, vector)
                .into_ann_result()?,
            prune: self
                .strategy
                .seeded_prune_accessor(
                    provider,
                    &self.context,
                    &self.seed,
                    self.index.config().max_occlusion_size().get(),
                )
                .into_ann_result()?,
        };

        Ok(insert)
    }

    fn prune_accessor(&mut self) -> ANNResult<impl BorrowMut<Self::PruneAccessor<'_>> + Send> {
        self.strategy
            .seeded_prune_accessor(
                self.index.provider(),
                &self.context,
                &self.seed,
                self.index.config().max_occlusion_size().get(),
            )
            .into_ann_result()
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("batch length ({}) does not match ids length ({})", self.batch_len, self.ids_len)]
struct BatchIdMismatch {
    batch_len: usize,
    ids_len: usize,
}

////////////////
// Compat API //
////////////////

/// Integration state forwarded through operations and accessor construction.
pub trait ExecutionContext: Send + Sync + Clone + 'static {
    /// Wrap a future before it is spawned, for example to propagate profiling state.
    fn wrap_spawn<F, T>(&self, future: F) -> impl Future<Output = T> + Send + 'static
    where
        F: Future<Output = T> + Send + 'static,
    {
        future
    }
}

#[derive(Debug)]
struct ScopedSpawn<'a, E>(&'a E);

impl<E> crate::tokio::Spawn for ScopedSpawn<'_, E>
where
    E: ExecutionContext,
{
    fn spawn<F>(&self, fut: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future<Output: Send + 'static> + Send + 'static,
    {
        tokio::spawn(self.0.wrap_spawn(fut))
    }
}

/// An execution context with no additional state or spawn behavior.
#[derive(Default, Clone)]
pub struct DefaultContext;

impl std::fmt::Display for DefaultContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "default context")
    }
}

impl ExecutionContext for DefaultContext {}

/// The identity mapping and operation types of a legacy integration.
pub trait DataProvider: Sized + Send + Sync + 'static {
    type Context: ExecutionContext;
    type InternalId: VectorId;
    type ExternalId: PartialEq + Send + Sync + 'static;
    type Error: ToRanked + std::fmt::Debug + Send + Sync + 'static;

    /// The insertion guard, owned across worker-task boundaries when necessary.
    type Guard: Guard<Id = Self::InternalId> + 'static;

    /// Translate an external ID previously assigned through [`SetElement`].
    ///
    /// The mapping is undefined until assignment.
    fn to_internal_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error>;
}

/// Assignment and insertion-completion notification for an integration.
pub trait SetElement<T>: DataProvider {
    type SetError: ToRanked + std::fmt::Debug + Send + Sync + 'static;

    /// Store `element`, associate it with `id`, and return its operation guard.
    ///
    /// Storage need not be lossless. Implementations should roll back internal state
    /// if assignment fails. The caller completes the guard only after insertion succeeds.
    /// Dropping an incomplete guard may undo mappings without reclaiming the local slot.
    fn set_element(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
        element: T,
    ) -> impl Future<Output = Result<Self::Guard, Self::SetError>> + Send;
}

/// An insertion's internal ID and success notification.
///
/// Dropping without calling [`Self::complete`] allows implementation-specific cleanup;
/// it does not guarantee rollback of graph mutations.
pub trait Guard: Send + Sync + 'static {
    type Id;

    /// Notify the provider that insertion completed successfully.
    fn complete(self) -> impl Future<Output = ()> + Send;

    fn id(&self) -> Self::Id;
}

/// A guard whose completion and destruction have no side effects.
#[derive(Debug, Default)]
pub struct NoopGuard<I>(I);

impl<I> NoopGuard<I> {
    pub fn new(id: I) -> Self {
        Self(id)
    }
}

impl<I> Guard for NoopGuard<I>
where
    I: Send + Sync + Copy + 'static,
{
    type Id = I;

    async fn complete(self) {}

    fn id(&self) -> Self::Id {
        self.0
    }
}

/// Logical deletion, liveness queries, and eventual storage release.
pub trait Delete: DataProvider {
    /// Exclude an item from search results while retaining internal access for traversal.
    fn delete(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Release storage after all incoming references have been removed.
    ///
    /// The caller must ensure that no accessor will retrieve this internal ID again
    /// until a subsequent [`SetElement`] operation yields it.
    fn release(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    fn status_by_internal_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> impl Future<Output = Result<ElementStatus, Self::Error>> + Send;

    fn status_by_external_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> impl Future<Output = Result<ElementStatus, Self::Error>> + Send;

    /// Query statuses without promising callback order.
    fn statuses_unordered<Itr, F>(
        &self,
        context: &Self::Context,
        itr: Itr,
        mut f: F,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send
    where
        Itr: Iterator<Item = Self::InternalId> + Send,
        F: FnMut(Result<ElementStatus, Self::Error>, Self::InternalId) + Send,
    {
        async move {
            for id in itr {
                f(self.status_by_internal_id(context, id).await, id);
            }
            Ok(())
        }
    }
}

/// Logical liveness; deleted storage may remain accessible until release.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementStatus {
    Valid,
    Deleted,
}

impl ElementStatus {
    pub fn is_valid(self) -> bool {
        self == Self::Valid
    }

    pub fn is_deleted(self) -> bool {
        self == Self::Deleted
    }
}

/// A provider's default accessor, without selecting a strategy.
pub trait DefaultAccessor: DataProvider {
    type Accessor<'a>: HasId<Id = Self::InternalId>
    where
        Self: 'a;

    fn default_accessor(&self) -> Self::Accessor<'_>;
}

/// Construct a query-specific accessor.
///
/// The accessor need not support unfiltered search; filtered strategies can return
/// an accessor with the capabilities required by their search algorithm.
pub trait SearchStrategy<'a, Provider, T>: Send + Sync
where
    Provider: DataProvider,
{
    type SearchAccessorError: StandardError;
    type SearchAccessor: HasId<Id = Provider::InternalId> + Send + Sync;

    fn search_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        query: T,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError>;
}

/// Construct the search accessor and select pruning for insertion.
pub trait InsertStrategy<'a, Provider, T>: Send + Sync + 'static
where
    Provider: DataProvider,
{
    type SearchAccessor: SearchAccessor<Id = Provider::InternalId>;
    type SearchAccessorError: StandardError;
    type PruneStrategy: PruneStrategy<Provider>;

    fn insert_search_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        vector: T,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError>;

    fn prune_strategy(&self) -> Self::PruneStrategy;
}

/// Construct pruning accessors independently of a query.
pub trait PruneStrategy<Provider>: Send + Sync + 'static
where
    Provider: DataProvider,
{
    type PruneAccessor<'a>: PruneAccessor<Id = Provider::InternalId>;
    type PruneAccessorError: StandardError;

    /// Construct an accessor sized for at most `capacity` IDs per fill.
    fn prune_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        capacity: usize,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError>;
}

/// Share batch-local data, such as a vector overlay, among insertion workers.
pub trait MultiInsertStrategy<Provider, B>: Send + Sync + 'static
where
    Provider: DataProvider,
    B: Batch,
{
    type Seed: Send + Sync + 'static;
    type FinishError: Into<ANNError> + std::fmt::Debug + Send + Sync;
    type PruneStrategy: PruneStrategy<Provider>;
    type InsertStrategy: for<'a> InsertStrategy<'a, Provider, B::Element<'a>, PruneStrategy = Self::PruneStrategy>;

    fn insert_strategy(&self) -> Self::InsertStrategy;

    /// Package an assigned batch into a seed.
    ///
    /// The caller must provide exactly one internal ID per batch element, in batch order.
    fn finish<Itr>(
        &self,
        provider: &Provider,
        context: &Provider::Context,
        batch: &Arc<B>,
        ids: Itr,
    ) -> impl Future<Output = Result<Self::Seed, Self::FinishError>> + Send
    where
        Itr: ExactSizeIterator<Item = Provider::InternalId> + Send;

    /// Construct a seeded accessor sized for at most `capacity` IDs per fill.
    fn seeded_prune_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        seed: &'a Self::Seed,
        capacity: usize,
    ) -> ANNResult<<Self::PruneStrategy as PruneStrategy<Provider>>::PruneAccessor<'a>>;
}

/// Select pruning and prepare query data for search-based inplace deletion.
pub trait InplaceDeleteStrategy<Provider>: Send + Sync + 'static
where
    Provider: DataProvider,
{
    type DeleteElement<'a>: Copy + Send + Sync;

    /// Own the query data for as long as a delete-search accessor may borrow it.
    type DeleteElementGuard: Send
        + Sync
        + for<'a> Reborrow<'a, Target = Self::DeleteElement<'a>>
        + 'static;

    type DeleteElementError: StandardError;
    type PruneStrategy: PruneStrategy<Provider>;
    type DeleteSearchAccessor<'a>: SearchAccessor<Id = Provider::InternalId>;
    type SearchPostProcessor: for<'a> SearchPostProcess<Self::DeleteSearchAccessor<'a>> + Send;
    type SearchStrategy: for<'a> SearchStrategy<
            'a,
            Provider,
            Self::DeleteElement<'a>,
            SearchAccessor = Self::DeleteSearchAccessor<'a>,
        >;

    fn prune_strategy(&self) -> Self::PruneStrategy;

    fn search_strategy(&self) -> Self::SearchStrategy;

    fn search_post_processor(&self) -> Self::SearchPostProcessor;

    fn get_delete_element<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        id: Provider::InternalId,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send;
}
