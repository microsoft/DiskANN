/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    cmp,
    collections::HashMap,
    num::NonZeroUsize,
    ops::Range,
    sync::{Arc, Mutex},
};

use diskann_utils::{
    Reborrow,
    future::{AssertSend, AsyncFriendly, SendFuture, boxit},
    reborrow::AsyncLower,
};
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction};
use futures_util::FutureExt;
use hashbrown::HashSet;
use thiserror::Error;
use tokio::task::JoinSet;
use tracing::{debug, trace};

use super::{
    AdjacencyList, Config, ConsolidateKind, InplaceDeleteMethod, RangeSearchParams, SearchParams,
    glue::{
        self, AsElement, ExpandBeam, FillSet, HybridPredicate, IdIterator, InplaceDeleteStrategy,
        InsertStrategy, Predicate, PredicateMut, PruneStrategy, SearchExt, SearchPostProcess,
        SearchStrategy, aliases,
    },
    internal::{BackedgeBuffer, SortedNeighbors, prune},
    search::{
        record::{NoopSearchRecord, SearchRecord, VisitedSearchRecord},
        scratch::{self, PriorityQueueConfiguration, SearchScratch, SearchScratchParams},
    },
    search_output_buffer,
};

#[cfg(feature = "experimental_diversity_search")]
use super::DiverseSearchParams;

use crate::{
    ANNError, ANNErrorKind, ANNResult,
    error::{ErrorExt, IntoANNResult},
    neighbor::{Neighbor, NeighborPriorityQueue, NeighborQueue},
    provider::{
        Accessor, AsNeighbor, AsNeighborMut, BuildDistanceComputer, BuildQueryComputer,
        DataProvider, Delete, ElementStatus, ExecutionContext, Guard, NeighborAccessor,
        NeighborAccessorMut, SetElement,
    },
    tracked_error,
    utils::{
        IntoUsize, TryIntoVectorId, VectorId,
        async_tools::{self, DynamicBalancer, VectorIdBoxSlice},
        object_pool::{ObjectPool, PooledRef},
    },
};

pub struct DiskANNIndex<DP: DataProvider> {
    /// Index config
    pub config: Config,

    /// The data provider.
    pub data_provider: DP,
    scratch_pool: ObjectPool<SearchScratch<DP::InternalId>>,
}

/// Decision returned by [`QueryLabelProvider::on_visit`] to control search traversal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueryVisitDecision<I: VectorId> {
    /// Accept this node into the frontier for further traversal.
    Accept(Neighbor<I>),
    /// Reject this node; do not add it to the frontier.
    Reject,
    /// Stop the search immediately without accepting this node.
    Terminate,
}

pub trait QueryLabelProvider<V: VectorId>: std::fmt::Debug + Send + Sync {
    /// This is a query scoped provider
    /// Check if the vec_id's label match the query label
    fn is_match(&self, vec_id: V) -> bool;

    /// Inspect a candidate before it is inserted into the frontier.
    /// Implementations can tweak the distance, reject the candidate, or
    /// request early termination. The default implementation accepts if
    /// `is_match` returns true, rejects otherwise.
    fn on_visit(&self, neighbor: Neighbor<V>) -> QueryVisitDecision<V> {
        if self.is_match(neighbor.id) {
            QueryVisitDecision::Accept(neighbor)
        } else {
            QueryVisitDecision::Reject
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DegreeStats {
    pub max_degree: u32,
    pub avg_degree: f32,
    pub min_degree: u32,
    pub cnt_less_than_two: usize, // Number of vertices with degree less than 2
}

/// Statistics collected during a search operation.
///
/// This struct provides detailed metrics about the search process, including
/// the number of nodes visited, the number of distance computations performed,
/// the number of hops taken during the search, and the total number of results returned.
pub struct SearchStats {
    /// The total number of distance computations performed during the search.
    pub cmps: u32,

    /// The number of hops (iterations) taken during the search traversal.
    pub hops: u32,

    /// The total number of results returned by the search.
    pub result_count: u32,

    /// Whether the range search continued to the second round
    pub range_search_second_round: bool,
}

/// Result from [`DiskANNIndex::get_undeleted_neighbors`].
pub struct PartitionedNeighbors<I> {
    pub undeleted: Vec<I>,
    pub deleted: Vec<I>,
}

/// Placeholder for extra state.
///
/// The contents of the search state are designed for the synchronous index.
/// However, use cases in the asynchronous index require some extra state.
///
/// This placeholder is used in the synchronous code-paths.
pub struct NoExtraState;

/// Represents the state of the pagged search.
/// It can be used to do paged search by doing multiple `nextSearchResults()` queries.
///
/// Generic extra state can be included to facilitate extra use-cases.
/// However, this extra state **must** be '`static' as we do not know how long the search
/// state will live for.
#[derive(Debug)]
pub struct SearchState<VectorIdType: VectorId, ExtraState: 'static = NoExtraState> {
    /// Scratch space for query processing.
    pub scratch: SearchScratch<VectorIdType>,
    /// The computed search results ready to be returned in `nextSearchResults()` query
    pub computed_result: Vec<Neighbor<VectorIdType>>,
    /// The index of the next result to be returned.
    pub next_result_index: usize,
    /// The search computes results in the multiple of `search_param_l`.
    pub search_param_l: usize,
    /// Any extra data needed by down-stream implementations.
    pub extra: ExtraState,
}

/// Edge pending submission for multi-insert.
#[derive(Debug)]
struct PendingEdge<I> {
    source: I,
    edges: AdjacencyList<I>,
}

impl<I> PendingEdge<I> {
    fn new(source: I, edges: AdjacencyList<I>) -> Self {
        Self { source, edges }
    }
}

fn aggregate_backedges<I>(lists: &[PendingEdge<I>]) -> HashMap<I, BackedgeBuffer<I>>
where
    I: Eq
        + std::hash::Hash
        + Copy
        + Default
        + diskann_vector::contains::ContainsSimd
        + std::fmt::Debug,
{
    let mut map = HashMap::<I, BackedgeBuffer<I>>::new();
    for PendingEdge { source, edges } in lists {
        for target in edges.iter() {
            map.entry(*target)
                .and_modify(|buf| {
                    buf.push(*source);
                })
                .or_insert_with(|| BackedgeBuffer::new(*source));
        }
    }
    map
}

/// A `Result` that indicates an error, but returns a value of type `T` on both the `Ok`
/// and `Err` paths.
type BatchResult<T> = Result<T, (T, ANNError)>;

/// State used during by paged search to perform multiple, consecutive searches over the index.
///
/// Type parameters:
///
/// * `DP`: The type of the [`DataProvider`].
/// * `S`: The type of the [`SearchStrategy`].
/// * `C`: The type of `S`'s [`BuildQueryComputer`] computer. This exists as a separate
///   type parameter because the type of the query computer depends on the type of the query.
pub type PagedSearchState<DP, S, C> = SearchState<<DP as DataProvider>::InternalId, (S, C)>;

/// The result of invoking [`DiskANNIndex::set_elements`], which invokes [`SetElement`] on
/// a batch of vector ids and returns the guards for the batch (as well as the batch data
/// with the appropriate internal ids.
struct SetBatchElements<G, I, T> {
    /// The [`SetElement::Guard`]s for every item in the set.
    guards: Vec<G>,

    /// The batch with the corresponding internal ids. The elements here have a position-wide
    /// correspondence with guards. That is, the guard at index `i` is the guard for
    /// the data vector as index `i`.
    batch: Arc<[VectorIdBoxSlice<I, T>]>,
}

pub struct NotInMutWithLabelCheck<'a, K>
where
    K: VectorId,
{
    visited_set: &'a mut hashbrown::HashSet<K>,
    query_label_evaluator: &'a dyn QueryLabelProvider<K>,
}

impl<'a, K> NotInMutWithLabelCheck<'a, K>
where
    K: VectorId,
{
    /// Construct a new `NotInMutWithLabelCheck` around `visited_set`.
    pub fn new(
        visited_set: &'a mut hashbrown::HashSet<K>,
        query_label_evaluator: &'a dyn QueryLabelProvider<K>,
    ) -> Self {
        Self {
            visited_set,
            query_label_evaluator,
        }
    }
}

impl<K> Predicate<K> for NotInMutWithLabelCheck<'_, K>
where
    K: VectorId,
{
    fn eval(&self, item: &K) -> bool {
        !self.visited_set.contains(item) && self.query_label_evaluator.is_match(*item)
    }
}

impl<K> PredicateMut<K> for NotInMutWithLabelCheck<'_, K>
where
    K: VectorId,
{
    fn eval_mut(&mut self, item: &K) -> bool {
        if self.query_label_evaluator.is_match(*item) {
            return self.visited_set.insert(*item);
        }
        false
    }
}

impl<K> HybridPredicate<K> for NotInMutWithLabelCheck<'_, K> where K: VectorId {}

impl<DP> DiskANNIndex<DP>
where
    DP: DataProvider,
{
    pub fn new(config: Config, data_provider: DP, thread_hint: Option<NonZeroUsize>) -> Self {
        let num_threads = thread_hint.map_or(0, |x| x.get());

        let scratch_pool = ObjectPool::new(
            &SearchScratchParams {
                l_value: config.l_build().get(),
                max_degree: config.max_degree().into(),
                num_frozen_pts: 0,
            },
            num_threads,       //initial_create_size
            Some(num_threads), //capacity
        );

        Self {
            config,
            data_provider,
            scratch_pool,
        }
    }

    /// Return scoped scratch space to use for index search.
    ///
    /// * `l`: The default window size to use.
    /// * `additional`: Extra capacity, usually to allow start points to be filtered from
    ///   the result.
    fn search_scratch(
        &self,
        l: usize,
        additional: usize,
    ) -> PooledRef<'_, SearchScratch<DP::InternalId>> {
        let params = SearchScratchParams {
            l_value: l,
            max_degree: self.max_degree_with_slack(),
            num_frozen_pts: additional,
        };
        self.scratch_pool.get_ref(&params)
    }

    /// Returns the configured target degree of the index.
    pub(crate) fn pruned_degree(&self) -> usize {
        self.config.pruned_degree().get()
    }

    /// Return the construction search list size.
    pub(crate) fn l_build(&self) -> usize {
        self.config.l_build().get()
    }

    /// Return the configured max occlusion size for this index.
    pub(crate) fn max_occlusion_size(&self) -> usize {
        self.config.max_occlusion_size().get()
    }

    /// Return the configured maximum size of an adjacency list accounting for the graph
    /// slack factor.
    pub(crate) fn max_degree_with_slack(&self) -> usize {
        self.config.max_degree().get()
    }

    /// Provide a rough estimate for the number of nodes visited during a graph traversal.
    ///
    /// If `search_l` is provided, it will be used for the computation. Otherwise, the
    /// current value of `self.l_build()` will be used.
    pub(crate) fn estimate_visited_set_capacity(&self, search_l: Option<usize>) -> usize {
        let effective_l = search_l.unwrap_or(self.l_build());
        scratch::estimate_node_visited_set_size(self.pruned_degree(), effective_l)
    }

    pub fn provider(&self) -> &DP {
        &self.data_provider
    }

    /// Insert a vector into the index with the given `id` and `vector`.
    pub fn insert<S, T>(
        &self,
        strategy: S,
        context: &DP::Context,
        id: &DP::ExternalId,
        vector: &T,
    ) -> impl SendFuture<ANNResult<()>>
    where
        S: InsertStrategy<DP, T>,
        T: Sync + ?Sized,
        DP: SetElement<T>,
    {
        async move {
            let guard = self
                .data_provider
                .set_element(context, id, vector)
                .await
                .escalate("insertion requires a successful `set_element`")?;

            let internal_id = guard.id();

            // NOTE: Use the API `insert_search_accessor` to allow `Accessor` customization.
            let mut accessor = strategy
                .insert_search_accessor(&self.data_provider, context)
                .into_ann_result()?;

            let computer = accessor.build_query_computer(vector).into_ann_result()?;

            // NOTE: We don't filter the start points out of `visited_nodes`, as those are
            // needed to generate out edges from the start points.
            let start_ids = accessor.starting_points().await?;

            let prune_strategy = strategy.prune_strategy();
            let mut prune_accessor = prune_strategy
                .prune_accessor(&self.data_provider, context)
                .into_ann_result()?;

            let mut scratch = self.search_scratch(self.l_build(), start_ids.len());
            let mut search_l = scratch.best.search_l();

            // If the experimental config is present, use it to obtain the maximum number
            // of retries. Otherwise, we stick with the default of 1.
            let insert_retry = self.config.experimental_insert_retry();
            let num_insert_attempts = insert_retry.map_or(1, |v| v.max_retries().get());

            // N.B.: Working set needs to be outlived by `accessor`.
            let mut working_set = HashMap::default();
            let mut prune_scratch = prune::Scratch::new();
            let mut new_neighbors = AdjacencyList::with_capacity(self.max_degree_with_slack());

            for attempt in 0..num_insert_attempts {
                let mut search_record =
                    VisitedSearchRecord::new(self.estimate_visited_set_capacity(Some(search_l)));

                self.search_internal(
                    None, // beam_width
                    &start_ids,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                    &mut search_record,
                )
                .await?;

                let context = prune::Context {
                    pool: SortedNeighbors::new(
                        &mut search_record.visited,
                        self.max_occlusion_size(),
                    ),
                    occlude_factor: &mut prune_scratch.occlude_factor,
                    neighbors: &mut new_neighbors,
                    last_checked: &mut prune_scratch.last_checked,
                };

                let options = prune::Options {
                    force_saturate: insert_retry.is_some_and(|v| v.should_saturate(attempt)),
                };

                self.robust_prune(
                    &mut prune_accessor,
                    internal_id,
                    context,
                    &mut working_set,
                    options,
                )
                .await?;

                let should_retry =
                    insert_retry.is_some_and(|v| v.should_retry(attempt, new_neighbors.len()));
                if !should_retry {
                    break;
                }

                search_l *= 2;
                scratch.resize(search_l);
                scratch.clear();
            }

            trace!(
                "Inserting out edges for vector_id: {} new_out_neighbors: {:?}",
                internal_id, new_neighbors,
            );

            // insert out edges
            prune_accessor
                .set_neighbors(internal_id, &new_neighbors)
                .await?;

            // Number of back edges this insert is allowed to add
            let num_back_edges = self.config.max_backedges().get();

            // add edges from `new_neighbors` to `internal_id` and prune if necessary
            for source in new_neighbors.iter().take(num_back_edges) {
                self.add_edge_and_prune(
                    &prune_strategy,
                    context,
                    std::slice::from_ref(&internal_id),
                    *source,
                    &mut prune_scratch,
                    &mut working_set,
                    None,
                )
                .await?;
            }

            guard.complete().await;
            Ok(())
        }
    }

    /// Perform a search for the given `vector_id, vector`, prune and return the visited set.
    /// Returns a tuple of the `vector_id` and a list of nodes from which to append an edge to `vector_id`.
    fn search_and_prune<S, T>(
        &self,
        strategy: &S,
        context: &DP::Context,
        vector_id_pair: &VectorIdBoxSlice<DP::InternalId, T>,
        position: usize,
        batch: &[VectorIdBoxSlice<DP::InternalId, T>],
        prune_scratch: &mut prune::Scratch<DP::InternalId>,
    ) -> impl SendFuture<ANNResult<PendingEdge<DP::InternalId>>>
    where
        T: Sync,
        DP: SetElement<[T]>,
        S: InsertStrategy<DP, [T]>,
        for<'a> aliases::InsertPruneAccessor<'a, S, DP, [T]>: AsElement<&'a [T]>,
    {
        async move {
            // Copy vectors to the vector provider, quantize them and set quant vec provider if necessary
            let internal_id = vector_id_pair.vector_id;
            let vector = vector_id_pair.vector.as_ref();

            // NOTE: Use the `insert_search_accessor` API to allow insert-specific customization.
            let mut accessor = strategy
                .insert_search_accessor(&self.data_provider, context)
                .into_ann_result()?;

            let computer = accessor.build_query_computer(vector).into_ann_result()?;
            let start_ids = accessor.starting_points().await?;

            let mut scratch = self.search_scratch(self.l_build(), start_ids.len());
            let mut search_l = scratch.best.search_l();

            // NOTE: We don't filter the start points out of `visited_nodes`, as those are
            // needed to generate out edges from the start points.
            let mut new_out_neighbors = AdjacencyList::with_capacity(self.max_degree_with_slack());

            // If the experimental config is present, use it to obtain the maximum number
            // of retries. Otherwise, we stick with the default of 1.
            let insert_retry = self.config.experimental_insert_retry();
            let num_insert_attempts = insert_retry.map_or(1, |v| v.max_retries().get());

            for attempt in 0..num_insert_attempts {
                let mut search_record = VisitedSearchRecord::new(
                    self.estimate_visited_set_capacity(Some(scratch.best.search_l())),
                );

                self.search_internal(
                    None, // beam_width
                    &start_ids,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                    &mut search_record,
                )
                .await?;

                // Add other vector id pairs in the mini batch to the candidate pool
                let prune_strategy = strategy.prune_strategy();
                let mut prune_accessor = prune_strategy
                    .prune_accessor(&self.data_provider, context)
                    .into_ann_result()?;

                let mut working_set = HashMap::new();
                let candidates = self.config.intra_batch_candidates().get(batch.len());
                if candidates != 0 {
                    let prune_computer =
                        prune_accessor.build_distance_computer().into_ann_result()?;
                    let this_vector: <
                        <S::PruneStrategy as PruneStrategy<DP>>::PruneAccessor<'_> as Accessor>::Extended = prune_accessor
                        .as_element(vector, internal_id)
                        .await
                        .escalate("Retrieving the inserted vector must succeed")?
                        .into();

                    for other in async_tools::around(batch, position, candidates) {
                        let id = other.vector_id;
                        if let Some(element) = prune_accessor
                            .as_element(&other.vector, id)
                            .await
                            .allow_transient(
                                "Failure to retrieve others in the batch is acceptable",
                            )?
                        {
                            search_record.push(Neighbor::new(
                                id,
                                prune_computer.evaluate_similarity(
                                    this_vector.reborrow(),
                                    element.reborrow(),
                                ),
                            ));

                            working_set.insert(id, element.into());
                        }
                    }
                }

                let context = prune::Context {
                    pool: SortedNeighbors::new(
                        &mut search_record.visited,
                        self.max_occlusion_size(),
                    ),
                    occlude_factor: &mut prune_scratch.occlude_factor,
                    neighbors: &mut new_out_neighbors,
                    last_checked: &mut prune_scratch.last_checked,
                };

                let options = prune::Options {
                    force_saturate: insert_retry.is_some_and(|v| v.should_saturate(attempt)),
                };

                self.robust_prune(
                    &mut prune_accessor,
                    internal_id,
                    context,
                    &mut working_set,
                    options,
                )
                .await?;

                let should_retry =
                    insert_retry.is_some_and(|v| v.should_retry(attempt, new_out_neighbors.len()));
                if !should_retry {
                    break;
                }

                search_l *= 2;
                scratch.resize(search_l);
                scratch.clear();
            }

            Ok(PendingEdge::new(internal_id, new_out_neighbors))
        }
    }

    /// Invoke `DP::set_element` for each vector in `vectors`. For each vector `v, returns:
    ///
    /// 1. The [`SetElement::Guard`] associated with the insertion.
    /// 2. The raw data of the vector.
    ///
    /// These results are aggregated into a vector, with an error being returned should any
    /// call to [`DP::set_element`] fail.
    ///
    /// This is the leaf task for the batch [`Self::set_elements`] method.
    async fn set_chunk<T>(
        &self,
        context: &DP::Context,
        vectors: Vec<VectorIdBoxSlice<DP::ExternalId, T>>,
    ) -> ANNResult<Vec<(DP::Guard, Box<[T]>)>>
    where
        DP: SetElement<[T]>,
    {
        let mut output = Vec::with_capacity(vectors.len());
        for pair in vectors {
            let id = pair.vector_id;
            let data = pair.vector;

            let guard = self
                .provider()
                .set_element(context, &id, &data)
                .await
                .escalate("cannot support failures during `set_element` in multi-insert")
                .into_ann_result()?;

            output.push((guard, data));
        }
        Ok(output)
    }

    /// Parallelise the invocation of applying [`DP::set_element`] to all vectors `v` in
    /// `vectors` using up to `ntasks` tasks.
    ///
    /// Return a pair consisting of the [`DP::set_element`] guards for each vector `v` in
    /// vectors and an [`Arc`] slice of the translated vectors. This function bails if any
    /// invocation of [`DP::set_element`] fails.
    ///
    /// The ordering of the guards will be consistent with the ordering of the translated
    /// vectors, though this need not necessarily be the order of the input vectors.
    ///
    /// The data backing each translated vector will be the same as that in `vectors`. That
    /// is, the backing data is *moved* internally, not copied.
    fn set_elements<T>(
        self: &Arc<Self>,
        context: &DP::Context,
        vectors: Box<[VectorIdBoxSlice<DP::ExternalId, T>]>,
        ntasks: NonZeroUsize,
    ) -> impl SendFuture<ANNResult<SetBatchElements<DP::Guard, DP::InternalId, T>>>
    where
        Self: 'static,
        T: AsyncFriendly,
        DP: SetElement<[T]>,
    {
        async move {
            let len = vectors.len();
            let partitions = async_tools::PartitionIter::new(vectors.len(), ntasks);

            // In the loop below, we chunk `itr` according to the lengths in `partitions`.
            let mut itr = vectors.into_iter();
            let handles: Vec<_> = partitions
                .map(|r| {
                    let self_clone = self.clone();

                    // Note: `by_ref`: avoids consuming `itr` but still take ownership of the
                    // yielded elements.
                    let chunk: Vec<_> = itr.by_ref().take(r.len()).collect();
                    let context_clone = context.clone();

                    // The task assigned to each round of `set_element`.
                    let future = async move { self_clone.set_chunk(&context_clone, chunk).await };

                    tokio::spawn(context.wrap_spawn(future))
                })
                .collect();

            // The collection of all the insert guards for the batch.
            let mut guards = Vec::with_capacity(len);

            // The repackage input data.
            let mut batch = Vec::with_capacity(len);

            for h in handles {
                let processed = h
                    .await
                    .map_err(|err| ANNError::new(ANNErrorKind::IndexError, err))??;
                for (guard, data) in processed {
                    let id = guard.id();
                    guards.push(guard);
                    batch.push(VectorIdBoxSlice {
                        vector_id: id,
                        vector: data,
                    });
                }
            }

            Ok(SetBatchElements {
                guards,
                batch: batch.into(),
            })
        }
    }

    /// Continually retrieve items from `work`, invoking `search_and_prune` on each claimed
    /// item. Return a vector of all results processed by this task.
    ///
    /// # Return Value and Error Handling
    ///
    /// Returns a pair `(edges, result)`. Whether or not `result` is an error, `edges` will
    /// contain all edges successfully processed by this task.
    fn search_and_prune_batch<S, T>(
        &self,
        strategy: &S,
        context: &DP::Context,
        work: &DynamicBalancer<VectorIdBoxSlice<DP::InternalId, T>>,
    ) -> impl SendFuture<BatchResult<Vec<PendingEdge<DP::InternalId>>>>
    where
        T: Send + Sync,
        DP: SetElement<[T]>,
        S: InsertStrategy<DP, [T]>,
        for<'a> aliases::InsertPruneAccessor<'a, S, DP, [T]>: AsElement<&'a [T]>,
    {
        async move {
            let mut output = Vec::new();
            let mut prune_scratch = prune::Scratch::new();
            while let Some((vector, position)) = work.next() {
                match self
                    .search_and_prune(
                        strategy,
                        context,
                        vector,
                        position,
                        work.all(),
                        &mut prune_scratch,
                    )
                    .await
                {
                    Ok(item) => output.push(item),
                    Err(err) => return Err((output, err)),
                }
            }
            Ok(output)
        }
    }

    /// The leaf element for the multi-insert bootstrapping algorithm.
    ///
    /// This algorithm:
    ///
    /// 1. Constructs a new candidates array from (A) the `current` pending edge and
    ///    (B) all other members in the `batch`.
    /// 2. Runs pruning on the aggregated list.
    /// 3. Returns the new pruned result as a replacement for `current`.
    fn multi_insert_bootstrap_leaf<S>(
        &self,
        strategy: &S,
        context: &DP::Context,
        current: &PendingEdge<DP::InternalId>,
        batch: &[PendingEdge<DP::InternalId>],
    ) -> impl SendFuture<ANNResult<PendingEdge<DP::InternalId>>>
    where
        S: PruneStrategy<DP>,
    {
        async move {
            // Collect all the current edges and all elements in the batch.
            let candidates =
                AdjacencyList::from_iter_untrusted(current.edges.iter().copied().chain(
                    batch.iter().filter_map(|i| {
                        // Avoid self loops.
                        if i.source == current.source {
                            None
                        } else {
                            Some(i.source)
                        }
                    }),
                ));

            let mut accessor = strategy
                .prune_accessor(&self.data_provider, context)
                .into_ann_result()?;

            let mut prune_scratch = prune::Scratch::new();
            let mut working_set = HashMap::new();

            // During bootstrap, we want the graph to be as dense as possible to aid
            // in early navigation.
            //
            // Enabling saturation help achieve that.
            let options = prune::Options {
                force_saturate: true,
            };

            self.robust_prune_list(
                &mut accessor,
                current.source,
                &candidates,
                &mut prune_scratch,
                &mut working_set,
                options,
            )
            .await
            .escalate("retrieving inserted vector must succeed")?;

            Ok(PendingEdge::new(current.source, prune_scratch.neighbors))
        }
    }

    /// Task level step in multi-insert bootstrap.
    ///
    /// This algorithm constructs a new candidates array from the `current` pending edge
    /// and all other members in the `batch`, runs pruning, and returns the new pruned result.
    fn multi_insert_bootstrap_task<S>(
        &self,
        strategy: &S,
        context: &DP::Context,
        work: &DynamicBalancer<PendingEdge<DP::InternalId>>,
    ) -> impl SendFuture<BatchResult<Vec<PendingEdge<DP::InternalId>>>>
    where
        S: PruneStrategy<DP>,
    {
        async move {
            let mut output = Vec::new();
            while let Some((pending, _)) = work.next() {
                let item = match self
                    .multi_insert_bootstrap_leaf::<S>(strategy, context, pending, work.all())
                    .await
                {
                    Ok(item) => item,
                    Err(err) => return Err((output, err)),
                };
                output.push(item);
            }
            Ok(output)
        }
    }

    /// Entry point for the mulit-insert bootstrap routine and a parallelized version of
    /// [`Self::multi_insert_bootstrap_task`]. Refer to that method for documentation on the
    /// routine applied to each edge in `edges`.
    ///
    /// If `self.config.max_minibatch_par()` is 1 or `edge.len() == 1`, then no additional
    /// spawns are made by this function.
    fn multi_insert_bootstrap<S>(
        self: Arc<Self>,
        strategy: S,
        context: DP::Context,
        edges: Vec<PendingEdge<DP::InternalId>>,
    ) -> impl SendFuture<ANNResult<Vec<PendingEdge<DP::InternalId>>>>
    where
        S: PruneStrategy<DP> + Clone,
    {
        async move {
            let num_items = edges.len();
            let work = Arc::new(DynamicBalancer::new(edges.into()));
            let num_tasks = self.config.max_minibatch_par().get().min(num_items).max(1);

            // Note: `num_tasks - 1` cannot underflow because `num_tasks` is guaranteed
            // to be at least 1.
            let handles: Vec<_> = (0..num_tasks - 1)
                .map(|_| {
                    let self_clone = self.clone();
                    let strategy_clone = strategy.clone();
                    let context_clone = context.clone();
                    let work_clone = work.clone();
                    tokio::spawn(context.wrap_spawn(async move {
                        self_clone
                            .multi_insert_bootstrap_task(
                                &strategy_clone,
                                &context_clone,
                                &work_clone,
                            )
                            .await
                    }))
                })
                .collect();

            // Process work on this thread.
            let mut next = match self
                .multi_insert_bootstrap_task(&strategy, &context, &work)
                .await
            {
                Ok(v) => v,
                Err((v, err)) => {
                    tracked_error!("main bootstrap task failed: {}", err);
                    v
                }
            };

            for h in handles {
                match h.await {
                    Ok(maybe_ok) => match maybe_ok {
                        Ok(mut v) => next.append(&mut v),
                        Err((mut v, err)) => {
                            next.append(&mut v);
                            tracked_error!("bootstrap task failed: {}", err);
                        }
                    },
                    Err(err) => tracked_error!("boostrap spawn failed: {}", err),
                }
            }
            Ok(next)
        }
    }

    /// Insert a small set of vectors into the index. The call parallelizes the search for
    /// each vector as well as the subsequent pruning and edge addition.
    ///
    /// Each `multi_insert` makes at most one update (either `set` or `append`) per id to
    /// the [`DataProvider`]. The algorithm resolves conflicting updates to adjacency lists
    /// caused by multiple inserts without involving the provider in the resolution.
    ///
    /// This method receives `self` by `Arc` to allow cloning for parallel task spawns.
    ///
    /// - `strategy`: The [`InsertStrategy`] to use for insert searches and prunes.
    /// - `context` is the context to pass through to providers.
    /// - `vectors` the vectors and associated ids to insert.
    ///
    /// # Configuration
    ///
    /// Multi-insert specific configuration includes:
    ///
    /// 1. [`diskann::index::Config::max_minibatch_par()`]: Control the maximum number
    ///    of concurrent task spawns used by the implementation. This puts an upper bound
    ///    on the extracted parallelism of this function.
    ///
    /// 2. [`diskann::index::Config::intra_batch_candidates()`]: Controls the maximum
    ///    number of candidates from within the batch that are considered as neighbors.
    ///
    ///    When the batch size is very high or the inserted data has high self similarity,
    ///    it is recommended to set this a moderate value such as 32 to ensure edges are
    ///    formed within a batch. However, setting this too high can slow down ingestion
    ///    for high batch sizes.
    ///
    ///    If this is set to zero, then candidates will not be considered among the batch
    ///    **unless** the number of unique back edge sources is fewer than eight times the
    ///    batch size. This helps with initial graph formation when the index is initially
    ///    empty. The value "8" is a rough heuristic, chosen to trigger frequently during
    ///    the initial phases of build but rarely again.
    ///
    /// # Error Handling
    ///
    /// The error handling for this function is currently undergoing a revision. Currently,
    /// errors encountered during tasks spawns are suppressed. The revision will inform
    /// the caller of failed insertions in a more precise manner.
    ///
    /// # Dev-Docs on Flow
    ///
    /// Multi-Insert works in three phases: (1) Set Elements, (2) Candidate Generation, and
    /// (3) graph update.
    ///
    /// ## Set Elements
    ///
    /// Vectors taken from external sources need to be inserted into the underlying data
    /// provider and an internal ID needs to be generated for these items. This phase
    /// parallelizes the insertion and collects the internal IDs for the inserted vectors.
    ///
    /// ## Candidate Generation
    ///
    /// This is made up of a graph search followed by a prune on the resulting candidate
    /// list. If [`diskann::index::Config::intra_batch_candidates()`] is non-zero, then
    /// candidates from within the batch are added at this step prior to prune.
    ///
    /// ### Bootstrap
    ///
    /// If no intra-batch candidates are to be used, the optional bootstrap routine (defined
    /// in the above section titled "configuration") is run after the initial candidate
    /// generation.
    ///
    /// ## Graph Update
    ///
    /// After all candidates have been retrieved, backedges are aggregated and the graph is
    /// updated. First with the generated candidates, and second to commit the backedges,
    /// triggering secondary prunes if necessary. This phase is partitioned so each parallel
    /// task updates a disjoint set of elements.
    pub fn multi_insert<S, T>(
        self: &Arc<Self>,
        strategy: S,
        context: &DP::Context,
        vectors: Box<[VectorIdBoxSlice<DP::ExternalId, T>]>,
    ) -> impl SendFuture<ANNResult<()>>
    where
        Self: 'static,
        T: AsyncFriendly,
        S: InsertStrategy<DP, [T]> + Clone + AsyncFriendly,
        DP: SetElement<[T]>,
        S::PruneStrategy: Clone,
        for<'a> aliases::InsertPruneAccessor<'a, S, DP, [T]>: AsElement<&'a [T]>,
    {
        async move {
            let num_tasks = self.config.max_minibatch_par();

            //--------------//
            // Set Elements //
            //--------------//

            let SetBatchElements { guards, batch } =
                boxit(self.set_elements(context, vectors, num_tasks)).await?;

            //----------------------//
            // Candidate Generation //
            //----------------------//

            // Dynamically partition the work across tasks. The time spent processing each
            // item (measured in the hundreds of micro-seconds) likely far exceeds the
            // synchronization overhead of the atomic increment.
            let work = Arc::new(DynamicBalancer::new(batch));

            // Launch `max_minibatch_par - 1` tasks to do work, running the last task on
            // the local thread.
            let handles: Vec<_> = (0..num_tasks.get() - 1)
                .map(|_| {
                    let self_clone = self.clone();
                    let strategy_clone = strategy.clone();
                    let context_clone = context.clone();
                    let work_clone = work.clone();
                    let future = async move {
                        self_clone
                            .search_and_prune_batch(&strategy_clone, &context_clone, &work_clone)
                            .await
                    };
                    tokio::spawn(context.wrap_spawn(future))
                })
                .collect();

            // Defer dealing with the `result` until after we have joined the other tasks.
            let mut edges = match self.search_and_prune_batch(&strategy, context, &work).await {
                Ok(v) => v,
                Err((v, err)) => {
                    tracked_error!("search_prune_and_search main failed: {}", err);
                    v
                }
            };

            // At this point - all the other tasks should be close to completing.
            for h in handles {
                match h.await {
                    Ok(Ok(mut v)) => edges.append(&mut v),
                    Ok(Err((mut v, err))) => {
                        edges.append(&mut v);
                        tracked_error!("search_prune_and_search failed: {}", err)
                    }
                    Err(err) => tracked_error!("Tokio spawned task join error: {}", err),
                }
            }

            let mut backedges = aggregate_backedges(&edges);

            //-----------//
            // Bootstrap //
            //-----------//

            // Check if number of unique back edges source is very small. If so, we do kick
            // off the bootstrap routine and add edges from within the batch.
            //
            // If `work.len() == 1`, then there is nothing to bootstrap since there are no
            // other edges in the batch.
            if self.config.intra_batch_candidates().is_none()
                && backedges.len().div_ceil(8) <= work.len() /* NB: update docs if 8 changes */
                && work.len() != 1
            {
                edges = boxit(self.clone().multi_insert_bootstrap(
                    strategy.prune_strategy(),
                    context.clone(),
                    edges,
                ))
                .await?;

                backedges = aggregate_backedges(&edges);
            }

            let backedges = Arc::new(backedges);

            //--------------//
            // Graph Update //
            //--------------//

            // Sequential assignment of the current neighbors. This does not yet appear
            // to be a huge bottleneck (compared to backedge aggregation).
            {
                let prune_strategy = strategy.prune_strategy();
                let accessor = &mut prune_strategy
                    .prune_accessor(&self.data_provider, context)
                    .into_ann_result()?;
                accessor
                    .set_neighbors_bulk(
                        edges
                            .into_iter()
                            .map(|PendingEdge { source, edges }| (source, edges)),
                    )
                    .await?;
            }

            // Spawn backedge insertions.
            let handles: Vec<_> = (0..num_tasks.get())
                .map(|i| {
                    let self_clone = self.clone();
                    let context_clone = context.clone();
                    let strategy_clone = strategy.prune_strategy();
                    let backedges_clone = backedges.clone();
                    tokio::spawn(context.wrap_spawn(async move {
                        // Get the range of items to process in this task.
                        let range = async_tools::partition(backedges_clone.len(), num_tasks, i)?;
                        let itr = backedges_clone.iter().skip(range.start).take(range.len());

                        let mut prune_scratch = prune::Scratch::new();
                        let mut working_set = HashMap::new();

                        for (source, adj_list) in itr {
                            // FIXME: Give providers control over the size of the working
                            // set.
                            working_set.clear();
                            self_clone
                                .add_edge_and_prune(
                                    &strategy_clone,
                                    &context_clone,
                                    adj_list,
                                    *source,
                                    &mut prune_scratch,
                                    &mut working_set,
                                    None,
                                )
                                .await?;
                        }
                        ANNResult::<()>::Ok(())
                    }))
                })
                .collect();

            for handle in handles {
                let result = handle.await;
                match result {
                    Err(err) => {
                        tracked_error!("Tokio task error in multi_insert: {}", err);
                    }
                    Ok(Err(err)) => {
                        tracked_error!("Error in `add_edge_and_prune: {}", err);
                    }
                    Ok(Ok(())) => {}
                }
            }

            // Indicate the batch as complete.
            for guard in guards {
                guard.complete().await;
            }

            Ok(())
        }
    }

    pub fn is_any_neighbor_deleted<NA>(
        &self,
        context: &DP::Context,
        accessor: &mut NA,
        vector_id: DP::InternalId,
    ) -> impl SendFuture<ANNResult<bool>>
    where
        DP: Delete,
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        async move {
            let mut neighbors = AdjacencyList::new();
            accessor.get_neighbors(vector_id, &mut neighbors).await?;

            for neighbor_id in neighbors.iter() {
                let is_deleted = self
                    .data_provider
                    .status_by_internal_id(context, *neighbor_id)
                    .await
                    .unwrap_or(ElementStatus::Deleted);
                if is_deleted == ElementStatus::Deleted {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }

    pub fn drop_adj_list(
        &self,
        accessor: &mut impl AsNeighborMut<Id = DP::InternalId>,
        vector_id: DP::InternalId,
    ) -> impl SendFuture<ANNResult<()>> {
        accessor
            .set_neighbors(vector_id, &[])
            .map(|x| x.map(|_| ()))
    }

    /// Explore the adjacency list for `id`, invoke `on_present` for neighbors that
    /// are present and `on_deleted` for neighbors that are deleted.
    fn on_neighbors<F, G, NA>(
        &self,
        context: &DP::Context,
        accessor: &mut NA,
        id: DP::InternalId,
        mut on_present: F,
        mut on_deleted: G,
    ) -> impl SendFuture<ANNResult<()>>
    where
        DP: Delete,
        F: FnMut(DP::InternalId) + Send,
        G: FnMut(DP::InternalId) + Send,
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        async move {
            let mut neighbors = AdjacencyList::new();
            accessor.get_neighbors(id, &mut neighbors).await?;

            self.data_provider
                .statuses_unordered(context, neighbors.iter().cloned(), |status, id| {
                    // Treat errors as deleted.
                    match status {
                        Err(_) => on_deleted(id),
                        Ok(status) => {
                            if status.is_deleted() {
                                on_deleted(id);
                            } else {
                                on_present(id);
                            }
                        }
                    }
                })
                .await
                .allow_transient("failures may occur during `on_neighbors`")?;

            Ok(())
        }
    }

    pub fn get_undeleted_neighbors<NA>(
        &self,
        context: &DP::Context,
        accessor: &mut NA,
        vector_id: DP::InternalId,
    ) -> impl SendFuture<ANNResult<PartitionedNeighbors<DP::InternalId>>>
    where
        DP: Delete,
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        async move {
            let mut undeleted = Vec::new();
            let mut deleted = Vec::new();

            self.on_neighbors(
                context,
                accessor,
                vector_id,
                |i| undeleted.push(i),
                |i| deleted.push(i),
            )
            .await?;

            Ok(PartitionedNeighbors { undeleted, deleted })
        }
    }

    /// Return the list of ids with refs to a particular (deleted) vertex
    pub fn return_refs_to_deleted_vertex<NA>(
        &self,
        accessor: &mut NA,
        vector_id: DP::InternalId,
        candidate_list: &[DP::InternalId],
    ) -> impl SendFuture<ANNResult<Vec<DP::InternalId>>>
    where
        DP: Delete,
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        async move {
            let mut return_candidates = Vec::new();
            let mut candidate_adj_list = AdjacencyList::new();
            for candidate_id in candidate_list.iter() {
                accessor
                    .get_neighbors(*candidate_id, &mut candidate_adj_list)
                    .await?;

                if candidate_adj_list.contains(vector_id) {
                    return_candidates.push(*candidate_id);
                }
            }
            Ok(return_candidates)
        }
    }

    /// Obtain the approximate in-neighbors and replace candidates using
    /// the visited set to approximate in-neighbors and the approximate top-k
    /// neighbors to approximate the replace candidates
    fn get_candidates_using_visited_and_topk<S>(
        &self,
        strategy: &S,
        context: &DP::Context,
        id: DP::InternalId,
        l_value: usize,
        k_value: usize,
    ) -> impl SendFuture<ANNResult<InplaceDeleteWorkList<DP::InternalId>>>
    where
        S: InplaceDeleteStrategy<DP> + Sync,
        DP: Delete,
    {
        async move {
            let v = strategy
                .get_delete_element(&self.data_provider, context, id)
                .await
                .into_ann_result()?;

            let search_strategy = strategy.search_strategy();
            let mut search_accessor = search_strategy
                .search_accessor(&self.data_provider, context)
                .into_ann_result()?;

            let computer = search_accessor
                .build_query_computer(&v.async_lower())
                .into_ann_result()?;

            let start_ids = search_accessor.starting_points().await?;

            let mut scratch = self.search_scratch(l_value, start_ids.len());

            self.search_internal(
                None, // beam_width
                &start_ids,
                &mut search_accessor,
                &computer,
                &mut scratch,
                &mut NoopSearchRecord::new(),
            )
            .await?;

            let mut output = vec![Neighbor::<DP::InternalId>::default(); l_value];

            // NOTE: We rely on `post_process` to remove deleted items from the results
            // placed into the output.
            let proxy = v.async_lower();
            let num_results = search_strategy
                .post_processor()
                .post_process(
                    &mut search_accessor,
                    &*proxy,
                    &computer,
                    scratch.best.iter(),
                    output.as_mut_slice(),
                )
                .send()
                .await
                .into_ann_result()?;

            let mut undeleted_ids: Vec<_> = output
                .iter()
                .take(num_results)
                .map(|neighbor| neighbor.id)
                .collect();

            // Collect IDs whose adjacency lists need to be updated.
            let ids_to_modify = self
                .return_refs_to_deleted_vertex(&mut search_accessor, id, &undeleted_ids)
                .await?;

            undeleted_ids.truncate(k_value);
            Ok(InplaceDeleteWorkList {
                replace_candidates: undeleted_ids,
                in_neighbors: ids_to_modify,
            })
        }
    }

    /// An alternative, experimental approach to finding the in-neighbors
    /// of a deleted point as well as the candidates to replace it with
    /// Uses the two-hop neighborhood of deleted point to find its in-neighbors,
    /// and the one-hop neighborhood of the deleted point as the replace candidates
    fn get_candidates_using_twohop_and_onehop<NA>(
        &self,
        context: &DP::Context,
        accessor: &mut NA,
        id: DP::InternalId,
    ) -> impl SendFuture<ANNResult<InplaceDeleteWorkList<DP::InternalId>>>
    where
        DP: Delete,
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        async move {
            let mut two_hop_nbhs = HashSet::new();
            let mut undeleted_one_hop_nbhs = Vec::new();
            self.on_neighbors(
                context,
                accessor,
                id,
                |i| {
                    undeleted_one_hop_nbhs.push(i);
                },
                |_| {},
            )
            .await?;

            let mut nbh_adj_list = AdjacencyList::new();
            for nbh in undeleted_one_hop_nbhs.iter() {
                two_hop_nbhs.insert(*nbh);
                accessor.get_neighbors(*nbh, &mut nbh_adj_list).await?;
                for nbh_nbh in nbh_adj_list.iter() {
                    two_hop_nbhs.insert(*nbh_nbh);
                }
            }

            let mut undeleted_two_hop_nbhs = Vec::new();

            for neighbor in two_hop_nbhs.iter() {
                let is_deleted = self
                    .data_provider
                    .status_by_internal_id(context, *neighbor)
                    .await
                    .allow_transient("assuming transient delete call means \"deleted\"")?
                    .is_none_or(|v| v.is_deleted());

                if !is_deleted {
                    undeleted_two_hop_nbhs.push(*neighbor);
                }
            }

            let in_neighbors = self
                .return_refs_to_deleted_vertex(accessor, id, &undeleted_two_hop_nbhs)
                .await?;

            Ok(InplaceDeleteWorkList {
                replace_candidates: undeleted_one_hop_nbhs,
                in_neighbors,
            })
        }
    }

    /// An alternative, experimental approach to finding the in-neighbors
    /// of a deleted point as well as the candidates to replace it with
    /// Uses the one-hop neighborhood of deleted point to find its in-neighbors,
    /// and the one-hop neighborhood of the deleted point as the replace candidates
    fn get_candidates_using_onehop<NA>(
        &self,
        context: &DP::Context,
        accessor: &mut NA,
        id: DP::InternalId,
    ) -> impl SendFuture<ANNResult<InplaceDeleteWorkList<DP::InternalId>>>
    where
        DP: Delete,
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        async move {
            let mut undeleted_one_hop_nbhs = Vec::new();
            self.on_neighbors(
                context,
                accessor,
                id,
                |i| {
                    undeleted_one_hop_nbhs.push(i);
                },
                |_| {},
            )
            .await?;

            let in_neighbors = self
                .return_refs_to_deleted_vertex(accessor, id, &undeleted_one_hop_nbhs)
                .await?;

            Ok(InplaceDeleteWorkList {
                replace_candidates: undeleted_one_hop_nbhs,
                in_neighbors,
            })
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
        Self: 'static + Send + Sync,
        S: InplaceDeleteStrategy<DP>
            + for<'a> SearchStrategy<DP, S::DeleteElement<'a>>
            + Send
            + Sync
            + Clone,
        DP: Delete + Send + Sync,
    {
        async move {
            let max_minibatch_par = self.config.max_minibatch_par();
            let chunk_iter = async_tools::arc_chunks(ids, max_minibatch_par);
            for chunk in chunk_iter {
                // compute edge updates for each inplace delete, running in parallel
                let handles: Vec<_> = (0..chunk.len())
                    .map(|i| {
                        let self_clone = Arc::clone(self);
                        let chunk_clone = chunk.clone();
                        let context_clone = context.clone();
                        let strategy_clone = strategy.clone();
                        let future = async move {
                            self_clone
                                .inplace_delete_inner(
                                    &strategy_clone,
                                    &context_clone,
                                    chunk_clone.get(i),
                                    num_to_replace,
                                    &inplace_delete_method,
                                )
                                .await
                        };
                        tokio::spawn(context.wrap_spawn(future))
                    })
                    .collect();

                let mut edge_collection = Vec::with_capacity(handles.len());
                for h in handles {
                    let res = h.await.map_err(|err| {
                        #[derive(Debug, Error)]
                        #[error("Spawning a task failed in inplace-delete: {0}")]
                        struct LocalError(tokio::task::JoinError);

                        ANNError::log_async_error(LocalError(err))
                    });
                    edge_collection.push(res);
                }

                // check for errors and collect ids to modify in one hashset
                let mut ids_to_modify = HashSet::<DP::InternalId>::with_capacity(
                    self.pruned_degree() * 2 * chunk.len(),
                );
                let mut edge_hashmaps = Vec::with_capacity(chunk.len());

                for output in edge_collection {
                    match output {
                        Ok(Ok(edges)) => {
                            for neighbor in edges.keys() {
                                ids_to_modify.insert(*neighbor);
                            }
                            edge_hashmaps.push(edges);
                        }
                        Ok(Err(ann_error)) => {
                            tracked_error!(
                                "inplace_delete returned error in multi_inplace_delete: {}",
                                ann_error
                            );
                        }
                        Err(err) => {
                            tracked_error!(
                                "Tokio spawned task has a join error in multi_inplace_delete: {}",
                                err
                            );
                        }
                    }
                }

                // next, insert and prune, adding the option to remove all the deleted neighbors
                // at each prune. this runs in parallel and respects the max_minibatch_par

                // convert external ids in chunk to internal ids
                let mut ids_to_delete = HashSet::with_capacity(chunk.len());
                for i in 0..chunk.len() {
                    let vector_id = self
                        .data_provider
                        .to_internal_id(context, chunk.get(i))
                        .escalate("id translation for `inplace_delete` must succeed")?;
                    ids_to_delete.insert(vector_id);
                }

                let num_tasks = NonZeroUsize::new(ids_to_modify.len())
                    .unwrap_or(max_minibatch_par)
                    .min(max_minibatch_par);

                let edges_to_add = Arc::new(Mutex::new(ids_to_modify.into_iter()));
                let ids_to_delete = Arc::new(ids_to_delete);
                let edge_hashmaps = Arc::new(edge_hashmaps);

                let mut tasks = JoinSet::new();
                for _ in 0..num_tasks.get() {
                    let self_clone = self.clone();
                    let context_clone = context.clone();
                    let strategy_clone = strategy.prune_strategy();
                    let edges_clone = edges_to_add.clone();
                    let ids_to_delete_clone = ids_to_delete.clone();
                    let edge_hashmaps_clone = edge_hashmaps.clone();
                    tasks.spawn(async move {
                        loop {
                            let result = {
                                let mut guard = edges_clone.lock().map_err(|_| {
                                    ANNError::log_async_error("Poisoned mutex during construction")
                                })?;
                                guard.next()
                            };

                            let mut prune_scratch = prune::Scratch::new();
                            let mut working_set = HashMap::new();

                            match result {
                                Some(source) => {
                                    let mut adj_list = Vec::new();
                                    for edge_hashmap in edge_hashmaps_clone.iter() {
                                        if let Some(edges) = edge_hashmap.get(&source) {
                                            // note: we don't deduplicate here because it's faster to let add_edge_and_prune handle it
                                            adj_list.extend_from_slice(edges);
                                        }
                                    }

                                    // FIXME: Give providers more control over the working
                                    // set.
                                    working_set.clear();
                                    self_clone
                                        .add_edge_and_prune(
                                            &strategy_clone,
                                            &context_clone,
                                            &adj_list,
                                            source,
                                            &mut prune_scratch,
                                            &mut working_set,
                                            Some(&ids_to_delete_clone), // delete any edges to a deleted point as part of add_edge_and_prune
                                        )
                                        .await?;
                                }
                                None => break,
                            }
                        }
                        ANNResult::Ok(())
                    });
                }

                // Wait for all tasks to complete.
                while let Some(result) = tasks.join_next().await {
                    if let Err(_e) = result {
                        tracked_error!("Tokio task JoinError in multi_inplace_delete");
                    } else if let Ok(Err(e)) = result {
                        tracked_error!("Error in add_edge_and_prune: {}", e);
                    }
                }

                // finally, drop each deleted neighbor's edges, this can run sequentially
                let prune_strategy = strategy.prune_strategy();
                let mut accessor = prune_strategy
                    .prune_accessor(&self.data_provider, context)
                    .into_ann_result()?;

                for vector_id in ids_to_delete.iter() {
                    self.drop_adj_list(&mut accessor, *vector_id).await?;
                }
            }

            Ok(())
        }
    }

    /// This method deletes a vertex without needing to loop over the entire graph to find
    /// its out-neighbors. It is meant to be called in conjunction with
    /// [`drop_deleted_neighbors`] run occasionally as a background process.
    ///
    /// See `https://arxiv.org/abs/2502.13826` for full description and experiments
    pub fn inplace_delete<S>(
        &self,
        strategy: S,
        context: &DP::Context,
        id: &DP::ExternalId,
        num_to_replace: usize,
        inplace_delete_method: InplaceDeleteMethod,
    ) -> impl SendFuture<ANNResult<()>>
    where
        S: InplaceDeleteStrategy<DP> + Sync + Clone,
        DP: Delete,
    {
        async move {
            let edges_to_add = self
                .inplace_delete_inner(
                    &strategy,
                    context,
                    id,
                    num_to_replace,
                    &inplace_delete_method,
                )
                .await?;

            let vector_id = self
                .data_provider
                .to_internal_id(context, id)
                .escalate("id translation for `inplace_delete` must succeed")?;

            let mut delete_set = HashSet::with_capacity(1);
            delete_set.insert(vector_id);
            let prune_strategy = strategy.prune_strategy();

            let mut working_set = HashMap::new();
            let mut prune_scratch = prune::Scratch::new();

            for (neighbor, edges) in edges_to_add.iter() {
                // FIXME: Allow providers to set the maximum size of the working set to
                // avoid always needing clearing.
                working_set.clear();

                self.add_edge_and_prune(
                    &prune_strategy,
                    context,
                    edges,
                    *neighbor,
                    &mut prune_scratch,
                    &mut working_set,
                    Some(&delete_set), // delete the edge to the deleted point as part of add_edge_and_prune
                )
                .await?
            }

            let mut accessor = prune_strategy
                .prune_accessor(&self.data_provider, context)
                .into_ann_result()?;

            self.drop_adj_list(&mut accessor, vector_id).await?;

            Ok(())
        }
    }

    /// To assist with multi_delete, this function computes the edge updates for an
    /// inplace delete, but returns them instead of immediately adding them to the index
    fn inplace_delete_inner<'a, S>(
        &'a self,
        strategy: &'a S,
        context: &'a DP::Context,
        id: &'a DP::ExternalId,
        num_to_replace: usize,
        inplace_delete_method: &'a InplaceDeleteMethod,
    ) -> impl SendFuture<ANNResult<HashMap<DP::InternalId, Vec<DP::InternalId>>>>
    where
        S: InplaceDeleteStrategy<DP> + Sync,
        DP: Delete,
    {
        async move {
            self.data_provider
                .delete(context, id)
                .await
                .escalate("`inplace_delete` requires a successful delete")?;
            let vector_id = self
                .data_provider
                .to_internal_id(context, id)
                .escalate("id translation for `inplace_delete` must succeed")?;

            let search_strategy = strategy.search_strategy();
            let accessor = &mut search_strategy
                .search_accessor(&self.data_provider, context)
                .into_ann_result()?;

            let InplaceDeleteWorkList {
                replace_candidates,
                in_neighbors,
            } = match inplace_delete_method {
                InplaceDeleteMethod::VisitedAndTopK {
                    k_value: k,
                    l_value: l,
                } => {
                    self.get_candidates_using_visited_and_topk(strategy, context, vector_id, *l, *k)
                        .await?
                }
                InplaceDeleteMethod::TwoHopAndOneHop => {
                    self.get_candidates_using_twohop_and_onehop(context, accessor, vector_id)
                        .await?
                }
                InplaceDeleteMethod::OneHop => {
                    self.get_candidates_using_onehop(context, accessor, vector_id)
                        .await?
                }
            };

            let prune_strategy = strategy.prune_strategy();
            let mut accessor = prune_strategy
                .prune_accessor(&self.data_provider, context)
                .into_ann_result()?;

            let mut working_set = HashMap::new();
            let mut edges_to_add = HashMap::<DP::InternalId, Vec<DP::InternalId>>::new();

            let computer = accessor.build_distance_computer().into_ann_result()?;
            accessor
                .fill_set(&mut working_set, replace_candidates.iter().copied())
                .await
                .allow_transient("observed failure during working set population")?;

            accessor
                .fill_set(&mut working_set, in_neighbors.iter().copied())
                .await
                .allow_transient("observed failure during working set population")?;

            // For each candidate `c` with an edge to `p`, compute the closest nodes to `c`
            // from the `best_k_candidates`.
            let mut pool = Vec::<Neighbor<DP::InternalId>>::new();
            for neighbor in in_neighbors.iter() {
                pool.clear();
                let candidate = match working_set.get(neighbor) {
                    Some(candidate) => candidate,
                    None => continue,
                };

                for other_id in replace_candidates.iter() {
                    // Don't add a self edge.
                    if other_id == neighbor {
                        continue;
                    }

                    let other = match working_set.get(other_id) {
                        Some(other) => other,
                        None => continue,
                    };

                    pool.push(Neighbor::new(
                        *other_id,
                        computer.evaluate_similarity(candidate.reborrow(), other.reborrow()),
                    ));
                }

                pool.sort_unstable();
                let best = pool.iter().take(num_to_replace).map(|x| x.id).collect();
                edges_to_add.insert(*neighbor, best);
            }

            // fetch the filtered adjacency list of `p`.
            let PartitionedNeighbors {
                undeleted: adjacency_list,
                ..
            } = self
                .get_undeleted_neighbors(context, &mut accessor, vector_id)
                .await?;

            for neighbor in adjacency_list {
                pool.clear();
                // We can accept a transient error on this candidate retrieval by skipping this
                // computation.
                let candidate = match accessor
                    .get_element(neighbor)
                    .await
                    .allow_transient_with(|| format!("skipping candidate {}", neighbor))?
                {
                    Some(candidate) => candidate,
                    None => continue,
                };

                for other_id in replace_candidates.iter() {
                    // Don't add a self edge.
                    if other_id == &neighbor {
                        continue;
                    }

                    let other = match working_set.get(other_id) {
                        Some(other) => other,
                        None => continue,
                    };

                    pool.push(Neighbor::new(
                        *other_id,
                        computer.evaluate_similarity(candidate.reborrow(), other.reborrow()),
                    ));
                }

                pool.sort_unstable();
                pool.iter().take(num_to_replace).for_each(|n| {
                    edges_to_add
                        .entry(n.id)
                        .or_insert_with(Vec::new)
                        .push(neighbor);
                });
            }

            Ok(edges_to_add)
        }
    }

    /// Look for deleted nodes in the neighbors of `vector_id`, and remove them if found
    /// This is used in conjunction with inplace deletes
    /// If `only_orphans` is set to true, we check to make sure a node's adjacency list
    /// is empty before removing a link to it. This is in case `drop_deleted_neighbors` is
    /// called with an inplace delete still pending
    pub fn drop_deleted_neighbors<NA>(
        &self,
        context: &DP::Context,
        accessor: &mut NA,
        vector_id: DP::InternalId,
        only_orphans: bool,
    ) -> impl SendFuture<ANNResult<ConsolidateKind>>
    where
        DP: Delete,
        NA: AsNeighborMut<Id = DP::InternalId>,
    {
        async move {
            let status = self
                .data_provider
                .status_by_internal_id(context, vector_id)
                .await
                .escalate("`drop_deleted_neighbors` should only be called on valid IDs")?;
            if status.is_deleted() {
                return Ok(ConsolidateKind::Deleted);
            }

            let degree = self.pruned_degree();

            let PartitionedNeighbors {
                undeleted: mut pool,
                deleted: deleted_neighbors,
            } = self
                .get_undeleted_neighbors(context, accessor, vector_id)
                .await?;

            // don't delete an edge to a vertex `v` if `v`'s adjacency list is still present
            if only_orphans {
                let mut neighbors = AdjacencyList::new();
                for deleted_nbh in deleted_neighbors.iter() {
                    accessor.get_neighbors(*deleted_nbh, &mut neighbors).await?;
                    if !neighbors.is_empty() {
                        pool.push(*deleted_nbh);
                    }
                }
            }

            // If nothing was deleted & prune not required, do nothing and return
            if deleted_neighbors.is_empty() && pool.len() <= degree {
                debug!(
                    "Consolidate_vector: Nothing to do for vector_id: {}",
                    vector_id
                );
                return Ok(ConsolidateKind::Complete);
            }

            trace!(
                "Consolidate_vector: Setting new AdjList for vector_id {} to {:?}.",
                vector_id, pool,
            );

            accessor.set_neighbors(vector_id, &pool).await?;

            Ok(ConsolidateKind::Complete)
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
            let is_deleted = self
                .data_provider
                .status_by_internal_id(context, vector_id)
                .await
                .escalate("`consolidate_vector` should only be called on valid IDs")?
                .is_deleted();
            if is_deleted {
                trace!("Called consolidate on deleted vector");
                return Ok(ConsolidateKind::Deleted);
            }
            let degree = self.pruned_degree();

            let mut pool = HashSet::new();
            let mut deleted_neighbors = Vec::new();
            let accessor = &mut strategy
                .prune_accessor(&self.data_provider, context)
                .into_ann_result()?;

            self.on_neighbors(
                context,
                accessor,
                vector_id,
                |i| {
                    pool.insert(i);
                },
                |i| deleted_neighbors.push(i),
            )
            .await?;

            // If nothing was deleted & prune not required, do nothing and return
            if deleted_neighbors.is_empty() && pool.len() <= degree {
                debug!(
                    "Consolidate_vector: Nothing to do for vector_id: {}",
                    vector_id
                );
                return Ok(ConsolidateKind::Complete);
            }

            // If something was deleted, get neighbors of deleted nodes and add them to new pool
            for id in deleted_neighbors.iter() {
                self.on_neighbors(
                    context,
                    accessor,
                    *id,
                    |i| {
                        pool.insert(i);
                    },
                    |_| {},
                )
                .await?;
            }

            // Remove self-loop if it snuck in.
            pool.remove(&vector_id);

            // Prune the new pool if it exceeds the max degree
            let adj_list = {
                let neighbors = AdjacencyList::from_iter_unique(pool.into_iter());
                if neighbors.len() < degree {
                    neighbors
                } else {
                    let mut prune_scratch = prune::Scratch::new();
                    let mut working_set = HashMap::new();

                    // Force saturation is mainly used for the retry insert logic.
                    //
                    // No need to force during consolidation.
                    let options = prune::Options {
                        force_saturate: false,
                    };

                    if self
                        .robust_prune_list(
                            accessor,
                            vector_id,
                            &neighbors,
                            &mut prune_scratch,
                            &mut working_set,
                            options,
                        )
                        .await
                        .allow_transient("vector retrieval is allowed to fail during consolidate")?
                        .is_none()
                    {
                        return Ok(ConsolidateKind::FailedVectorRetrieval);
                    }

                    prune_scratch.neighbors
                }
            };

            trace!(
                "Consolidate_vector: Setting new AdjList for vector_id {} to {:?}.",
                vector_id, adj_list
            );

            accessor.set_neighbors(vector_id, &adj_list).await?;
            Ok(ConsolidateKind::Complete)
        }
    }

    // A is the accessor type, T is the query type used for BuildQueryComputer
    fn search_internal<A, T, SR, Q>(
        &self,
        beam_width: Option<usize>,
        start_ids: &[DP::InternalId],
        accessor: &mut A,
        computer: &A::QueryComputer,
        scratch: &mut SearchScratch<DP::InternalId, Q>,
        search_record: &mut SR,
    ) -> impl SendFuture<ANNResult<InternalSearchStats>>
    where
        A: ExpandBeam<T, Id = DP::InternalId> + SearchExt,
        T: ?Sized,
        SR: SearchRecord<DP::InternalId> + ?Sized,
        Q: NeighborQueue<DP::InternalId>,
    {
        async move {
            let beam_width = beam_width.unwrap_or(1);

            // paged search can call search_internal multiple times, we only need to initialize
            // state if not already initialized.
            if scratch.visited.is_empty() {
                for id in start_ids {
                    scratch.visited.insert(*id);
                    let element = accessor
                        .get_element(*id)
                        .await
                        .escalate("start point retrieval must succeed")?;
                    let dist = computer.evaluate_similarity(element.reborrow());
                    scratch.best.insert(Neighbor::new(*id, dist));
                    scratch.cmps += 1;
                }
            }

            let mut neighbors = Vec::with_capacity(self.max_degree_with_slack());
            while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
                scratch.beam_nodes.clear();

                // In this loop we are going to find the beam_width number of nodes that are closest to the query.
                // Each of these nodes will be a frontier node.
                while scratch.best.has_notvisited_node() && scratch.beam_nodes.len() < beam_width {
                    let closest_node = scratch.best.closest_notvisited();
                    search_record.record(closest_node, scratch.hops, scratch.cmps);
                    scratch.beam_nodes.push(closest_node.id);
                }

                neighbors.clear();
                accessor
                    .expand_beam(
                        scratch.beam_nodes.iter().copied(),
                        computer,
                        glue::NotInMut::new(&mut scratch.visited),
                        |distance, id| neighbors.push(Neighbor::new(id, distance)),
                    )
                    .await?;

                // The predicate ensures that the contents of `neighbors` are unique.
                //
                // We insert into the priority queue outside of the expansion for
                // code-locality purposes.
                neighbors
                    .iter()
                    .for_each(|neighbor| scratch.best.insert(*neighbor));

                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;
            }

            Ok(InternalSearchStats {
                cmps: scratch.cmps,
                hops: scratch.hops,
                range_search_second_round: false,
            })
        }
    }

    // A is the accessor type, T is the query type used for BuildQueryComputer
    // scratch.in_range is guaranteed to include the starting points
    fn range_search_internal<A, T>(
        &self,
        search_params: &RangeSearchParams,
        accessor: &mut A,
        computer: &A::QueryComputer,
        scratch: &mut SearchScratch<DP::InternalId>,
    ) -> impl SendFuture<ANNResult<InternalSearchStats>>
    where
        A: ExpandBeam<T, Id = DP::InternalId> + SearchExt,
        T: ?Sized,
    {
        async move {
            let beam_width = search_params.beam_width.unwrap_or(1);

            for neighbor in &scratch.in_range {
                scratch.range_frontier.push_back(neighbor.id);
            }

            let mut neighbors = Vec::with_capacity(self.max_degree_with_slack());

            let max_returned = search_params.max_returned.unwrap_or(usize::MAX);

            while !scratch.range_frontier.is_empty() {
                scratch.beam_nodes.clear();

                // In this loop we are going to find the beam_width number of remaining nodes within the radius
                // Each of these nodes will be a frontier node.
                while !scratch.range_frontier.is_empty() && scratch.beam_nodes.len() < beam_width {
                    let next = scratch.range_frontier.pop_front();
                    if let Some(next_node) = next {
                        scratch.beam_nodes.push(next_node);
                    }
                }

                neighbors.clear();
                accessor
                    .expand_beam(
                        scratch.beam_nodes.iter().copied(),
                        computer,
                        glue::NotInMut::new(&mut scratch.visited),
                        |distance, id| neighbors.push(Neighbor::new(id, distance)),
                    )
                    .await?;

                // The predicate ensure that the contents of `neighbors` are unique.
                for neighbor in neighbors.iter() {
                    if neighbor.distance <= search_params.radius * search_params.range_search_slack
                        && scratch.in_range.len() < max_returned
                    {
                        scratch.in_range.push(*neighbor);
                        scratch.range_frontier.push_back(neighbor.id);
                    }
                }
                scratch.cmps += neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;
            }

            Ok(InternalSearchStats {
                cmps: scratch.cmps,
                hops: scratch.hops,
                range_search_second_round: true,
            })
        }
    }

    // A is the accessor type, T is the query type used for BuildQueryComputer
    fn multihop_search_internal<A, T, SR>(
        &self,
        search_params: &SearchParams,
        accessor: &mut A,
        computer: &A::QueryComputer,
        scratch: &mut SearchScratch<DP::InternalId>,
        search_record: &mut SR,
        query_label_evaluator: &dyn QueryLabelProvider<DP::InternalId>,
    ) -> impl SendFuture<ANNResult<InternalSearchStats>>
    where
        A: ExpandBeam<T, Id = DP::InternalId> + SearchExt,
        T: ?Sized,
        SR: SearchRecord<DP::InternalId> + ?Sized,
    {
        async move {
            let beam_width = search_params.beam_width.unwrap_or(1);

            // Helper to build the final stats from scratch state.
            let make_stats = |scratch: &SearchScratch<DP::InternalId>| InternalSearchStats {
                cmps: scratch.cmps,
                hops: scratch.hops,
                range_search_second_round: false,
            };

            // Initialize search state if not already initialized.
            // This allows paged search to call multihop_search_internal multiple times
            if scratch.visited.is_empty() {
                let start_ids = accessor.starting_points().await?;

                for id in start_ids {
                    scratch.visited.insert(id);
                    let element = accessor
                        .get_element(id)
                        .await
                        .escalate("start point retrieval must succeed")?;
                    let dist = computer.evaluate_similarity(element.reborrow());
                    scratch.best.insert(Neighbor::new(id, dist));
                }
            }

            // Pre-allocate with good capacity to avoid repeated allocations
            let mut one_hop_neighbors = Vec::with_capacity(self.max_degree_with_slack());
            let mut two_hop_neighbors = Vec::with_capacity(self.max_degree_with_slack());
            let mut candidates_two_hop_expansion = Vec::with_capacity(self.max_degree_with_slack());

            while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
                scratch.beam_nodes.clear();
                one_hop_neighbors.clear();
                candidates_two_hop_expansion.clear();
                two_hop_neighbors.clear();

                // In this loop we are going to find the beam_width number of nodes that are closest to the query.
                // Each of these nodes will be a frontier node.
                while scratch.best.has_notvisited_node() && scratch.beam_nodes.len() < beam_width {
                    let closest_node = scratch.best.closest_notvisited();
                    search_record.record(closest_node, scratch.hops, scratch.cmps);
                    scratch.beam_nodes.push(closest_node.id);
                }

                // compute distances from query to one-hop neighbors, and mark them visited
                accessor
                    .expand_beam(
                        scratch.beam_nodes.iter().copied(),
                        computer,
                        glue::NotInMut::new(&mut scratch.visited),
                        |distance, id| one_hop_neighbors.push(Neighbor::new(id, distance)),
                    )
                    .await?;

                // Process one-hop neighbors based on on_visit() decision
                for neighbor in one_hop_neighbors.iter().copied() {
                    match query_label_evaluator.on_visit(neighbor) {
                        QueryVisitDecision::Accept(accepted) => {
                            scratch.best.insert(accepted);
                        }
                        QueryVisitDecision::Reject => {
                            // Rejected nodes: still add to two-hop expansion so we can traverse through them
                            candidates_two_hop_expansion.push(neighbor);
                        }
                        QueryVisitDecision::Terminate => {
                            scratch.cmps += one_hop_neighbors.len() as u32;
                            scratch.hops += scratch.beam_nodes.len() as u32;
                            return Ok(make_stats(scratch));
                        }
                    }
                }

                scratch.cmps += one_hop_neighbors.len() as u32;
                scratch.hops += scratch.beam_nodes.len() as u32;

                // sort the candidates for two-hop expansion by distance to query point
                candidates_two_hop_expansion.sort_unstable_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // limit the number of two-hop candidates to avoid too many expansions
                candidates_two_hop_expansion.truncate(self.max_degree_with_slack() / 2);

                // Expand each two-hop candidate: if its neighbor is a match, compute its distance
                // to the query and insert into `scratch.visited`
                // If it is not a match, do nothing
                let two_hop_expansion_candidate_ids: Vec<DP::InternalId> =
                    candidates_two_hop_expansion.iter().map(|n| n.id).collect();

                accessor
                    .expand_beam(
                        two_hop_expansion_candidate_ids.iter().copied(),
                        computer,
                        NotInMutWithLabelCheck::new(&mut scratch.visited, query_label_evaluator),
                        |distance, id| {
                            two_hop_neighbors.push(Neighbor::new(id, distance));
                        },
                    )
                    .await?;

                // Next, insert the new matches into `scratch.best` and increment stats counters
                two_hop_neighbors
                    .iter()
                    .for_each(|neighbor| scratch.best.insert(*neighbor));

                scratch.cmps += two_hop_neighbors.len() as u32;
                scratch.hops += two_hop_expansion_candidate_ids.len() as u32;
            }

            Ok(make_stats(scratch))
        }
    }

    /// Filter out start nodes from the best candidates in the scratch.
    fn filter_search_candidates(
        &self,
        start_points: &[DP::InternalId],
        l_value: usize,
        best: &mut NeighborPriorityQueue<DP::InternalId>,
    ) -> impl SendFuture<ANNResult<(Vec<Neighbor<DP::InternalId>>, usize)>> {
        async move {
            let mut total = 0usize;
            let mut candidates = Vec::with_capacity(l_value);
            for n in best.iter() {
                total += 1;
                if !start_points.contains(&n.id) {
                    candidates.push(n);
                    if candidates.len() >= l_value {
                        break;
                    }
                }
            }

            debug_assert!(
                l_value.min(best.size().saturating_sub(start_points.len())) <= candidates.len(),
                "Not enough candidates after filtering starting points",
            );

            Ok((candidates, total))
        }
    }

    /// Performs a graph-based search towards a target query vector recording the path taken.
    ///
    /// This method executes a search using the provided `strategy` to access and process elements.
    /// It computes the similarity between the query vector and the elements in the index, moving towards the
    /// nearest neighbors according to the search parameters.
    /// The path taken is recorded according to the search_record object passed in.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The search strategy to use for accessing and processing elements.
    /// * `context` - The context to pass through to providers.
    /// * `query` - The query vector for which nearest neighbors are sought.
    /// * `search_params` - Parameters controlling the search behavior, such as search depth (`l_value`) and beam width.
    /// * `output` - A mutable buffer to store the search results. Must be pre-allocated by the caller.
    /// * `search_record` - A mutable reference to a search record object that will record the path taken during the search.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// - An optional vector of visited nodes (if requested in `search_params`).
    /// - The number of distance computations performed.
    /// - The number of hops (always zero for flat search, as no graph traversal occurs).
    ///
    /// # Errors
    ///
    /// Returns an error if there is a failure accessing elements or if the provided parameters are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn search_recorded<S, T, O, OB, SR>(
        &self,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        search_params: &SearchParams,
        output: &mut OB,
        search_record: &mut SR,
    ) -> impl SendFuture<ANNResult<SearchStats>>
    where
        T: Sync + ?Sized,
        S: SearchStrategy<DP, T, O>,
        O: Send,
        OB: search_output_buffer::SearchOutputBuffer<O> + Send + ?Sized,
        SR: SearchRecord<DP::InternalId> + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&self.data_provider, context)
                .into_ann_result()?;

            let computer = accessor.build_query_computer(query).into_ann_result()?;
            let start_ids = accessor.starting_points().await?;

            let mut scratch = self.search_scratch(search_params.l_value, start_ids.len());

            let stats = self
                .search_internal(
                    search_params.beam_width,
                    &start_ids,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                    search_record,
                )
                .await?;

            let result_count = strategy
                .post_processor()
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    scratch.best.iter().take(search_params.l_value.into_usize()),
                    output,
                )
                .send()
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }

    /// Performs a graph-based search towards a target query vector.
    ///
    /// This method executes a search using the provided `strategy` to access and process elements.
    /// It computes the similarity between the query vector and the elements in the index, moving towards the
    /// nearest neighbors according to the search parameters.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The search strategy to use for accessing and processing elements.
    /// * `context` - The context to pass through to providers.
    /// * `query` - The query vector for which nearest neighbors are sought.
    /// * `search_params` - Parameters controlling the search behavior, such as search depth (`l_value`) and beam width.
    /// * `output` - A mutable buffer to store the search results. Must be pre-allocated by the caller.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// - An optional vector of visited nodes (if requested in `search_params`).
    /// - The number of distance computations performed.
    /// - The number of hops (always zero for flat search, as no graph traversal occurs).
    ///
    /// # Errors
    ///
    /// Returns an error if there is a failure accessing elements or if the provided parameters are invalid.
    pub fn search<S, T, O, OB>(
        &self,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        search_params: &SearchParams,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<SearchStats>>
    where
        T: Sync + ?Sized,
        S: SearchStrategy<DP, T, O>,
        O: Send,
        OB: search_output_buffer::SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            self.search_recorded(
                strategy,
                context,
                query,
                search_params,
                output,
                &mut NoopSearchRecord::new(),
            )
            .await
        }
    }

    /// Performs a brute-force flat search over the points matching a provided filter function.
    ///
    /// This method executes a linear scan through all points in the index, applying the provided
    /// `vector_filter` to select candidate points. It computes the similarity between the query
    /// vector and each candidate, returning the top results according to the provided search parameters.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The search strategy to use for accessing and processing elements.
    /// * `context` - The context to pass through to providers.
    /// * `query` - The query vector for which nearest neighbors are sought.
    /// * `vector_filter` - A predicate function used to filter candidate vectors based on their external IDs.
    /// * `search_params` - Parameters controlling the search behavior, such as search depth (`l_value`) and beam width.
    /// * `output` - A mutable buffer to store the search results. Must be pre-allocated by the caller.
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// - An optional vector of visited nodes (if requested in `search_params`).
    /// - The number of distance computations performed.
    /// - The number of hops (always zero for flat search, as no graph traversal occurs).
    ///
    /// # Errors
    ///
    /// Returns an error if there is a failure accessing elements or if the provided parameters are invalid.
    ///
    /// # Notes
    ///
    /// This method is computationally expensive for large datasets, as it does not leverage the graph structure
    /// and instead performs a linear scan of all filtered points.
    pub async fn flat_search<'a, S, T, O, OB, I>(
        &'a self,
        strategy: &'a S,
        context: &'a DP::Context,
        query: &T,
        vector_filter: &(dyn Fn(&DP::ExternalId) -> bool + Send + Sync),
        search_params: &SearchParams,
        output: &mut OB,
    ) -> ANNResult<SearchStats>
    where
        T: ?Sized,
        S: SearchStrategy<DP, T, O, SearchAccessor<'a>: IdIterator<I>>,
        I: Iterator<Item = <DP as DataProvider>::InternalId>,
        O: Send,
        OB: search_output_buffer::SearchOutputBuffer<O> + Send,
    {
        let mut accessor = strategy
            .search_accessor(&self.data_provider, context)
            .into_ann_result()?;
        let computer = accessor.build_query_computer(query).into_ann_result()?;

        let mut scratch = {
            let num_start_points = accessor.starting_points().await?.len();
            self.search_scratch(search_params.l_value, num_start_points)
        };

        let id_iterator = accessor.id_iterator().await?;
        for id in id_iterator {
            let external_id = self
                .data_provider
                .to_external_id(context, id)
                .escalate("external id should be found")?;

            if vector_filter(&external_id) {
                scratch.visited.insert(id);
                let element = accessor
                    .get_element(id)
                    .await
                    .escalate("matched point retrieval must succeed")?;
                let dist = computer.evaluate_similarity(element.reborrow());
                scratch.best.insert(Neighbor::new(id, dist));
                scratch.cmps += 1;
            }
        }

        let result_count = strategy
            .post_processor()
            .post_process(
                &mut accessor,
                query,
                &computer,
                scratch.best.iter().take(search_params.l_value.into_usize()),
                output,
            )
            .send()
            .await
            .into_ann_result()?;

        Ok(SearchStats {
            cmps: scratch.cmps,
            hops: scratch.hops,
            result_count: result_count as u32,
            range_search_second_round: false,
        })
    }

    /// A helper function for range search that allows an external application
    /// to perform their own post-processing on the raw in-range results
    #[allow(clippy::type_complexity)]
    pub fn range_search_raw<S, T, O>(
        &self,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        search_params: &RangeSearchParams,
    ) -> impl SendFuture<ANNResult<(SearchStats, Vec<Neighbor<DP::InternalId>>)>>
    where
        T: Sync + ?Sized,
        S: SearchStrategy<DP, T, O>,
        O: Send + Default + Clone + Copy,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&self.data_provider, context)
                .into_ann_result()?;
            let computer = accessor.build_query_computer(query).into_ann_result()?;
            let start_ids = accessor.starting_points().await?;

            let mut scratch = self.search_scratch(search_params.starting_l_value, start_ids.len());

            let initial_stats = self
                .search_internal(
                    search_params.beam_width,
                    &start_ids,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                    &mut NoopSearchRecord::new(),
                )
                .await?;

            let mut in_range = Vec::with_capacity(search_params.starting_l_value.into_usize());

            for neighbor in scratch
                .best
                .iter()
                .take(search_params.starting_l_value.into_usize())
            {
                if neighbor.distance <= search_params.radius {
                    in_range.push(neighbor);
                }
            }

            // clear the visited set and repopulate it with just the in-range points
            scratch.visited.clear();
            for neighbor in in_range.iter() {
                scratch.visited.insert(neighbor.id);
            }
            scratch.in_range = in_range;

            let stats = if scratch.in_range.len()
                >= ((search_params.starting_l_value as f32) * search_params.initial_search_slack)
                    as usize
            {
                // Move to range search
                let range_stats = self
                    .range_search_internal(search_params, &mut accessor, &computer, &mut scratch)
                    .await?;

                InternalSearchStats {
                    cmps: initial_stats.cmps,
                    hops: initial_stats.hops + range_stats.hops,
                    range_search_second_round: true,
                }
            } else {
                initial_stats
            };

            Ok((
                stats.finish(scratch.in_range.len() as u32),
                scratch.in_range.to_vec(),
            ))
        }
    }

    /// Given a `query` vector, search for all results within a specified radius
    /// `l_value` is the search depth of the initial search phase
    ///
    /// Note that the radii in `search_params` are raw distances, not similarity scores;
    /// the user is expected to execute any necessary transformations to their desired
    /// radius before calling this function.
    ///
    /// We allow complicated types here to avoid needing an entirely new type definition
    /// for just one function
    #[allow(clippy::type_complexity)]
    pub fn range_search<S, T, O>(
        &self,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        search_params: &RangeSearchParams,
    ) -> impl SendFuture<ANNResult<(SearchStats, Vec<O>, Vec<f32>)>>
    where
        T: Sync + ?Sized,
        S: SearchStrategy<DP, T, O>,
        O: Send + Default + Clone + Copy,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&self.data_provider, context)
                .into_ann_result()?;
            let computer = accessor.build_query_computer(query).into_ann_result()?;

            let (mut stats, in_range) = self
                .range_search_raw(strategy, context, query, search_params)
                .await?;
            // create a new output buffer for the range search
            // need to initialize distance buffer to max value because of later filtering step
            let mut result_ids: Vec<O> = vec![O::default(); in_range.len()];
            let mut result_dists: Vec<f32> = vec![f32::MAX; in_range.len()];

            let mut output_buffer = search_output_buffer::IdDistance::new(
                result_ids.as_mut_slice(),
                result_dists.as_mut_slice(),
            );

            let _ = strategy
                .post_processor()
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    in_range.into_iter(),
                    &mut output_buffer,
                )
                .send()
                .await
                .into_ann_result()?;

            // Filter the output buffer for points with distance between inner and outer radius
            // Note this takes a dependency on the output of `post_process` being sorted by distance

            let inner_cutoff = if let Some(inner_radius) = search_params.inner_radius {
                result_dists
                    .iter()
                    .position(|dist| *dist > inner_radius)
                    .unwrap_or(result_dists.len())
            } else {
                0
            };

            let outer_cutoff = result_dists
                .iter()
                .position(|dist| *dist > search_params.radius)
                .unwrap_or(result_dists.len());

            result_ids.truncate(outer_cutoff);
            result_ids.drain(0..inner_cutoff);

            result_dists.truncate(outer_cutoff);
            result_dists.drain(0..inner_cutoff);

            let result_count = result_ids.len();

            stats.result_count = result_count as u32;

            Ok((stats, result_ids, result_dists))
        }
    }

    /// Graph search that takes into account label filter matching by expanding
    /// each non-matching neighborhood to search for matching nodes
    /// Label provider must be included as a function argument
    /// Note that if the Strategy is of type BetaFilter, this function assumes
    /// but does not enforce that the label provider used in the strategy
    /// is the same as the one in the function argument
    pub fn multihop_search<S, T, OB>(
        &self,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        search_params: &SearchParams,
        output: &mut OB,
        query_label_evaluator: &dyn QueryLabelProvider<DP::InternalId>,
    ) -> impl SendFuture<ANNResult<SearchStats>>
    where
        T: Sync + ?Sized,
        S: SearchStrategy<DP, T>,
        OB: search_output_buffer::SearchOutputBuffer<DP::InternalId> + Send,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&self.data_provider, context)
                .into_ann_result()?;
            let computer = accessor.build_query_computer(query).into_ann_result()?;

            let start_ids = accessor.starting_points().await?;

            let mut scratch = self.search_scratch(search_params.l_value, start_ids.len());

            let stats = self
                .multihop_search_internal(
                    search_params,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                    &mut NoopSearchRecord::new(),
                    query_label_evaluator,
                )
                .await?;

            let result_count = strategy
                .post_processor()
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    scratch.best.iter().take(search_params.l_value.into_usize()),
                    output,
                )
                .send()
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }

    //////////////////
    // Paged Search //
    //////////////////

    pub fn start_paged_search<S, T>(
        &self,
        strategy: S,
        context: &DP::Context,
        query: &T,
        l_value: usize,
    ) -> impl SendFuture<ANNResult<PagedSearchState<DP, S, S::QueryComputer>>>
    where
        S: SearchStrategy<DP, T> + 'static,
        T: Sync + ?Sized,
    {
        async move {
            self.start_paged_search_with_init_ids(strategy, context, query, l_value, None)
                .await
        }
    }

    pub fn start_paged_search_with_init_ids<S, T>(
        &self,
        strategy: S,
        context: &DP::Context,
        query: &T,
        l_value: usize,
        init_ids: Option<&[DP::InternalId]>,
    ) -> impl SendFuture<ANNResult<PagedSearchState<DP, S, S::QueryComputer>>>
    where
        S: SearchStrategy<DP, T> + 'static,
        T: Sync + ?Sized,
    {
        async move {
            let (computer, scratch) = {
                let mut accessor = strategy
                    .search_accessor(&self.data_provider, context)
                    .into_ann_result()?;

                let computer = accessor.build_query_computer(query).into_ann_result()?;

                let start_ids = accessor.starting_points().await?;
                let num_start_points = start_ids.len();

                let init_ids: std::borrow::Cow<'_, [DP::InternalId]> = match init_ids {
                    Some(ids) => ids.into(),
                    None => start_ids.into(),
                };

                // NOTE: init_ids are real points, and shouldn't be excluded from candidates
                let mut scratch = SearchScratch::new(
                    PriorityQueueConfiguration::Resizable(l_value + num_start_points),
                    None,
                );

                let mut neighbors = Vec::with_capacity(self.max_degree_with_slack());
                scratch.visited.extend(init_ids.iter().copied());
                accessor
                    .expand_beam(
                        init_ids.iter().copied(),
                        &computer,
                        glue::NotInMut::new(&mut scratch.visited),
                        |distance, id| neighbors.push(Neighbor::new(id, distance)),
                    )
                    .await?;

                // The predicate `NotInMut` ensures that the contents of `neighbors` are unique.
                neighbors
                    .iter()
                    .for_each(|neighbor| scratch.best.insert(*neighbor));

                (computer, scratch)
            };

            ANNResult::Ok(SearchState {
                scratch,
                computed_result: vec![Neighbor::default(); l_value],
                next_result_index: l_value,
                search_param_l: l_value,
                extra: (strategy, computer),
            })
        }
    }

    pub fn next_search_results<S, T>(
        &self,
        context: &DP::Context,
        search_state: &mut SearchState<DP::InternalId, (S, S::QueryComputer)>,
        k: usize,
        result_output: &mut [Neighbor<DP::InternalId>],
    ) -> impl SendFuture<ANNResult<usize>>
    where
        S: SearchStrategy<DP, T>,
        T: Send + Sync + ?Sized,
    {
        async move {
            if k > search_state.search_param_l {
                return ANNResult::Err(ANNError::log_paged_search_error(
                    "k should be less than or equal to search_param_l".to_string(),
                ));
            }
            if k == 0 {
                return ANNResult::Err(ANNError::log_paged_search_error(
                    "k should be greater than 0".to_string(),
                ));
            }
            if result_output.len() < k {
                return ANNResult::Err(ANNError::log_paged_search_error(
                    "The size of result_output should be greater than or equal to k".to_string(),
                ));
            }

            let copy_to_output =
                |search_state: &mut SearchState<DP::InternalId, (S, S::QueryComputer)>,
                 count: usize,
                 result_output: &mut [Neighbor<DP::InternalId>],
                 result_output_offset: usize| {
                    result_output[result_output_offset..result_output_offset + count]
                        .copy_from_slice(
                            &search_state.computed_result[search_state.next_result_index
                                ..search_state.next_result_index + count],
                        );
                    search_state.next_result_index += count;
                };

            let used_computed_result_count: usize = cmp::min(
                k,
                search_state.computed_result.len() - search_state.next_result_index,
            );
            if used_computed_result_count > 0 {
                copy_to_output(
                    search_state,
                    used_computed_result_count,
                    result_output,
                    0, // result_output_offset
                );

                if used_computed_result_count == k {
                    return ANNResult::Ok(k);
                }
            }

            let start_points = {
                let mut accessor = search_state
                    .extra
                    .0
                    .search_accessor(&self.data_provider, context)
                    .into_ann_result()?;

                let start_ids = accessor.starting_points().await?;
                self.search_internal(
                    None, // beam_width
                    &start_ids,
                    &mut accessor,
                    &search_state.extra.1,
                    &mut search_state.scratch,
                    &mut NoopSearchRecord::new(),
                )
                .await?;

                start_ids
            };

            let (mut candidates, total_considered) = self
                .filter_search_candidates(&start_points, k, &mut search_state.scratch.best)
                .await?;
            search_state.scratch.best.drain_best(total_considered);

            let computed_result_count = candidates.len();
            search_state.computed_result.clear();
            search_state.computed_result.append(&mut candidates);

            search_state.next_result_index = 0;
            if computed_result_count != search_state.search_param_l {
                search_state.computed_result.truncate(computed_result_count);
            }

            let leftover_results = cmp::min(k - used_computed_result_count, computed_result_count);

            copy_to_output(
                search_state,
                leftover_results, // count of results to copy
                result_output,
                used_computed_result_count, // result_output_offset
            );

            ANNResult::Ok(used_computed_result_count + leftover_results)
        }
    }

    /// Count the number of nodes in the graph reachable from the given `start_points`.
    ///
    /// This function has a large memory footprint for large graphs and should not be called
    /// frequently. This is mainly for analysis and sanity tests.
    pub fn count_reachable_nodes<NA>(
        &self,
        start_points: &[DP::InternalId],
        accessor: &mut NA,
    ) -> impl SendFuture<ANNResult<usize>>
    where
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        async move {
            let mut expanded_nodes = HashSet::<DP::InternalId>::new();
            let mut queue = std::collections::VecDeque::<DP::InternalId>::new();

            for id in start_points {
                queue.push_back(*id);
            }

            let mut neighbors = AdjacencyList::with_capacity(self.max_degree_with_slack());
            while let Some(id) = queue.pop_front() {
                if expanded_nodes.insert(id) {
                    accessor.get_neighbors(id, &mut neighbors).await?;
                    for neighbor_id in neighbors.iter() {
                        queue.push_back(*neighbor_id);
                    }
                }
            }

            Ok(expanded_nodes.len())
        }
    }

    pub fn get_degree_stats<NA>(&self, accessor: &mut NA) -> impl SendFuture<ANNResult<DegreeStats>>
    where
        for<'a> &'a DP: IntoIterator<Item = DP::InternalId, IntoIter: Send>,
        NA: AsNeighbor<Id = DP::InternalId>,
    {
        async move {
            let mut max_degree_usize: usize = 0;
            let mut min_degree_usize: usize = usize::MAX;
            let mut total = 0;
            let mut cnt_less_than_two: usize = 0;
            let mut total_live_points = 0;

            let mut neighbors = AdjacencyList::with_capacity(self.max_degree_with_slack());
            for id in &self.data_provider {
                total_live_points += 1;
                accessor.get_neighbors(id, &mut neighbors).await?;
                let pool_size = neighbors.len();
                max_degree_usize = cmp::max(max_degree_usize, pool_size);
                min_degree_usize = cmp::min(min_degree_usize, pool_size);
                total += pool_size;
                if pool_size < 2 {
                    cnt_less_than_two += 1;
                }
            }

            let total_f32 = total as f32;
            Ok(DegreeStats {
                max_degree: u32::try_from(max_degree_usize)?,
                avg_degree: total_f32 / total_live_points as f32,
                min_degree: u32::try_from(min_degree_usize)?,
                cnt_less_than_two,
            })
        }
    }
}

impl<DP> DiskANNIndex<DP>
where
    DP: DataProvider,
{
    /// A method to add edges from `source` to `targets`, and prune the adjacency list
    /// `source` if its degree exceeds `degree*graph_slack_factor`.
    /// Uses `append_neighbors` to append the new edge if a prune is not required.
    /// Uses `set_neighbors` to set the new adjacency list of `source` if a prune is required.
    /// This method mutates the adjacency list of the `source` node and no other node.
    ///
    /// * `context` - The index context passed through from upstream to providers.
    /// * `targets` - A list of indices of points to link to from the `source` point
    /// * `source` is the index of the source point
    /// * `scratch_option` is an option with mutable reference to a scratch space that can be reused for intermediate computations
    /// * `working_set` is an option with cache of vectors that can be used to avoid repeated reads
    /// * `to_remove` is an option to remove vectors from the adjacency list before adding the new edges
    ///
    /// # Lint
    ///
    /// We're going through a temporary phase of moving the `Context` into the `Strategy`.
    /// For now - we unfortunately need both.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_edge_and_prune<'a, S>(
        &'a self,
        strategy: &'a S,
        context: &'a DP::Context,
        targets: &[DP::InternalId],
        source: DP::InternalId,
        scratch: &mut prune::Scratch<DP::InternalId>,
        working_set: &mut HashMap<DP::InternalId, <S::PruneAccessor<'a> as Accessor>::Extended>,
        to_remove: Option<&HashSet<DP::InternalId>>,
    ) -> impl SendFuture<ANNResult<()>>
    where
        S: PruneStrategy<DP>,
    {
        async move {
            let mut accessor = strategy
                .prune_accessor(&self.data_provider, context)
                .into_ann_result()?;

            let mut adj_list = AdjacencyList::with_capacity(self.max_degree_with_slack());
            accessor.get_neighbors(source, &mut adj_list).await?;

            let mut did_remove = false;
            if let Some(set) = to_remove {
                adj_list.retain(|i| {
                    let contains = set.contains(i);
                    did_remove |= contains;
                    !contains
                })
            }

            let num_new_edges = adj_list.extend_from_slice(targets);
            if num_new_edges == 0 && !did_remove {
                trace!(
                    "Skipping edge insertion from {} to {:?} all edges already exist",
                    source, targets,
                );
                return Ok(());
            }

            if adj_list.len() <= self.max_degree_with_slack() {
                // No pruning is needed; we can just append.
                trace!("Appending back-edge from {} to {:?}.", source, targets,);
                if did_remove {
                    accessor.set_neighbors(source, &adj_list).await?;
                } else if let Some(edges) = adj_list.last(num_new_edges) {
                    accessor.append_vector(source, edges).await?;
                }
            } else {
                // During back-edge insertion, we are working with a limited number
                // of candidates and trying to keep the best of said candidates.
                //
                // No need to enable saturation.
                let options = prune::Options {
                    force_saturate: false,
                };

                self.robust_prune_list(
                    &mut accessor,
                    source,
                    &adj_list,
                    scratch,
                    working_set,
                    options,
                )
                .await
                .escalate("retrieving inserted vector must succeed")?;

                trace!(
                    "Setting new AdjList for vector_id {} to {:?}.",
                    source, scratch.neighbors
                );

                accessor.set_neighbors(source, &scratch.neighbors).await?;
            }

            Ok(())
        }
    }

    /// Run the pruning algorithm on [`prune::Context::pool`] and write the pruned
    /// list into [`prune::Context::neighbors`].
    ///
    /// # Errors
    ///
    /// Forwards critical errors from [`FillSet::fill_set`]. Transient errors from
    /// this API are suppressed and any IDs that failed will be skipped during prune.
    ///
    /// Errors due to [`BuildDistanceComputer::build_distance_computer`] are propagated.
    fn robust_prune<'a, A>(
        &'a self,
        accessor: &mut A,
        location: DP::InternalId,
        mut context: prune::Context<'_, DP::InternalId>,
        working_set: &mut HashMap<DP::InternalId, A::Extended>,
        options: prune::Options,
    ) -> impl SendFuture<ANNResult<()>>
    where
        A: Accessor<Id = DP::InternalId> + BuildDistanceComputer + FillSet + 'a,
    {
        async move {
            // Early exit.
            if context.pool.is_empty() {
                return Ok(());
            }

            let _: Option<()> = accessor
                .fill_set(working_set, context.pool.iter().map(|n| n.id))
                .send()
                .await
                .allow_transient("failures during working set fill are okay")?;

            // Note: Turbofish needed to help inference.
            self.occlude_list::<A::Extended, _, _>(
                &accessor.build_distance_computer().into_ann_result()?,
                &mut context,
                working_set,
                |id| id == location,
                options,
            );

            Ok(())
        }
    }

    /// A specialization of [`Self::robust_prune`] that prunes the IDs in an [`AdjacencyList`].
    ///
    /// The resulting candidates list will be placed into `state's`
    /// [`prune::Scratch::neighbors`] field.
    ///
    /// All other fields of `state` are clobbered.
    ///
    /// This works by first retrieving `location` and all ids in `list` into `working_set`
    /// via [`FillSet`] and then computing distances to populate the candidate pool.
    ///
    /// # Errors
    ///
    /// Forwards critical errors from [`FillSet::fill_set`]. Transient errors from
    /// this API are suppressed and any IDs that failed will be skipped during prune unless
    /// the vector that was not retrieved was `location`. If this is the case,
    /// [`prune::ListError::FailedVectorRetrieval`] is returned to delegate escalation to
    /// the caller.
    ///
    /// Errors due to [`BuildDistanceComputer::build_distance_computer`] are propagated.
    fn robust_prune_list<'a, A>(
        &'a self,
        accessor: &mut A,
        location: DP::InternalId,
        list: &AdjacencyList<DP::InternalId>,
        scratch: &mut prune::Scratch<DP::InternalId>,
        working_set: &mut HashMap<DP::InternalId, A::Extended>,
        options: prune::Options,
    ) -> impl SendFuture<Result<(), prune::ListError<DP::InternalId>>>
    where
        A: Accessor<Id = DP::InternalId> + BuildDistanceComputer + FillSet + 'a,
    {
        async move {
            // Early exit.
            if list.is_empty() {
                return Ok(());
            }

            // Fetch into the working set.
            let _: Option<()> = accessor
                .fill_set(
                    working_set,
                    std::iter::once(location).chain(list.iter().copied()),
                )
                .send()
                .await
                .allow_transient("failures during working set fill are okay")?;

            scratch.pool.clear();
            scratch.pool.reserve(list.len());

            let computer = accessor.build_distance_computer().into_ann_result()?;
            if let Some(vector) = working_set.get(&location) {
                for id in list.iter().filter(|&&i| i != location) {
                    if let Some(other) = working_set.get(id) {
                        scratch.pool.push(Neighbor::new(
                            *id,
                            computer.evaluate_similarity(vector.reborrow(), other.reborrow()),
                        ));
                    }
                }
            } else {
                return Err(prune::ListError::failed_retrieval(location));
            }

            // Note: Turbofish needed to help inference.
            self.occlude_list::<A::Extended, _, _>(
                &computer,
                &mut scratch.as_context(self.max_occlusion_size()),
                working_set,
                |id| id == location,
                options,
            );

            Ok(())
        }
    }

    /// Private implementation of the DiskANN pruning algorithm.
    ///
    /// Run the pruning algorithm using [`prune::Context::pool`] as the list of candidate.
    /// The `computer` will be used to perform distance computations, pulling elements
    /// from `map`. The closure `exclude` can be used to filter out unwanted ids from
    /// the candidates list.
    ///
    /// # Output
    ///
    /// After returning, the pruned neighbors can be found in [`prune::Context::neighbors`]
    ///
    /// ## Clobbers
    ///
    /// Clobbers the following scratch fields:
    ///
    /// * [`prune::Context::occlude_factor`]
    /// * [`prune::Context::last_checked`]
    ///
    /// ## Note
    ///
    /// The API for this function is a little awkward in an effort to minimize
    /// allocations. Non-trivial inputs, scratch space, and output all reside within
    /// [`prune::Context`].
    ///
    /// After `occlude_list` returns, the pruned neighbors can be found in
    /// [`prune::Context::neighbors`].
    ///
    /// # Options
    ///
    /// This algorithm saturates the adjacency list if configured in the global
    /// configuration or if forced via [`prune::Options`].
    fn occlude_list<V, C, F>(
        &self,
        computer: &C,
        context: &mut prune::Context<'_, DP::InternalId>,
        map: &HashMap<DP::InternalId, V>,
        exclude: F,
        options: prune::Options,
    ) where
        for<'a> V: Reborrow<'a>,
        C: for<'a, 'b> DistanceFunction<
                <V as Reborrow<'a>>::Target,
                <V as Reborrow<'b>>::Target,
                f32,
            >,
        F: Fn(DP::InternalId) -> bool,
    {
        if context.pool.is_empty() {
            return;
        }

        let prune::Context {
            pool,
            occlude_factor,
            neighbors: dst,
            last_checked,
        } = context;

        dst.clear();
        occlude_factor.clear();

        let alpha = self.config.alpha();
        let degree = self.config.pruned_degree().get();

        occlude_factor.clear();
        occlude_factor.resize(pool.len(), 0.0);

        last_checked.clear();
        last_checked.resize(pool.len(), 0u16);

        let mut current_alpha = 1.0f32;
        let increment_factor = alpha.min(1.2);

        // To avoid many hash lookups, we pull out just the candidates we're going to prune
        // into an auxiliary vector which can be accessed linearly.
        //
        // During the pruning phase, we store results by their relative position in the
        // cache, and only resolve to their `local_id` at the end.
        let cache: Vec<(f32, Option<&_>)> = pool
            .iter()
            .map(|neighbor| {
                // Filter out self loops.
                let id = &neighbor.id;
                if exclude(*id) {
                    (neighbor.distance, None)
                } else {
                    (neighbor.distance, map.get(id))
                }
            })
            .collect();

        // Loop will also terminate after cur_alpha reaches alpha
        while dst.len() < degree {
            for (i, (neighbor_distance, neighbor)) in cache.iter().enumerate() {
                if dst.len() >= degree {
                    break;
                }

                let factor = &mut occlude_factor[i];

                // If the occlusion factor for this neighbor is too high, skip it.
                if *factor > current_alpha {
                    continue;
                }

                // Retrieval from the cache might not be perfect.
                //
                // This neighbor did not end up in the cache, then just skip it.
                let neighbor = match neighbor {
                    Some(n) => n,
                    None => {
                        *factor = f32::MAX;
                        continue;
                    }
                };

                // This neighbor has not been exluded.
                //
                // To determine whether or not to add it, we must compute its occlusion
                // factor against all elements in `result` that appear before it in `pool`.
                //
                // This computation may be resumed from previous `alpha` values, so we need
                // to access our scratch data structures.
                let position = &mut last_checked[i];

                // During computation, we've found an occlusion factor greater than alpha
                // and wish to abort this neighbor from consideration.
                let mut skip: bool = false;

                // Increment `position` until we've compared with all current entries in
                // `result`.
                //
                // When the list is empty, the loop is skipped allowing the first undeleted
                // element to be added.
                while *position as usize != dst.len() {
                    let result_to_check = *position;
                    // Increment the position pointer.
                    *position += 1;

                    let result_position = dst[result_to_check as usize].into_usize();

                    // If the position of this result in `pool` is greater than or equal
                    // the current working position, then skip this candidate.
                    if result_position >= i {
                        continue;
                    }

                    // Otherwise, compute the distance between the result and this neighbor
                    // and update the occlude factor.
                    let distance = match cache[result_position] {
                        (_, Some(v)) => {
                            computer.evaluate_similarity((*neighbor).reborrow(), v.reborrow())
                        }
                        (_, None) => f32::MAX,
                    };

                    // Update occlude factor
                    *factor = self.config.prune_kind().update_occlude_factor(
                        *neighbor_distance,
                        distance,
                        *factor,
                        current_alpha,
                    );

                    // Check if the most recent update to the occlusion factor removes this
                    // neighbor from consideration.
                    if *factor > current_alpha {
                        // Don't add this neighbor.
                        skip = true;
                        break;
                    }
                }

                // N.B.: We can get here straight from the self-neighbor check when the
                // neighbor and the data at `location` have the same encoding.
                //
                // SO, we use short-circuiting logic to avoid checking the occlusion factor
                // twice, though that might not really save much.
                if skip || *factor > current_alpha {
                    continue;
                }

                // This neighbor has passed all the requirements of being a candidate.
                *factor = f32::MAX;

                // This conversion should always succeed.
                #[expect(
                    clippy::expect_used,
                    reason = "`i` cannot exceed `u16::MAX` so conversion should succeed"
                )]
                dst.push(
                    i.try_into_vector_id()
                        .expect("argument should not exceed u16::MAX"),
                );
            }

            // Exit if we completed the final iteration.
            if current_alpha == alpha {
                break;
            }
            // Update current alpha for the next iteration.
            current_alpha = (current_alpha * increment_factor).min(alpha);
        }

        // Cleanup `result` by undoing the local indirection.
        dst.remap_trusted(|r| *r = pool[r.into_usize()].id);
        debug_assert!(dst.len() <= degree, "max degree bound violated");

        // Post processing saturation if enabled.
        if options.force_saturate || (self.config.saturate_after_prune() && alpha > 1.0f32) {
            for neighbor in context.pool.iter() {
                if dst.len() >= degree {
                    break;
                }

                if !exclude(neighbor.id) {
                    // `AdjacencyList` filters out duplicates. No need to explicitly
                    // check.
                    dst.push(neighbor.id);
                }
            }
        }
    }

    /// Prune all nodes in the graph.
    ///
    /// This is used as the final step of graph construction.
    pub fn prune_range<S>(
        &self,
        strategy: &S,
        context: &DP::Context,
        range: Range<DP::InternalId>,
    ) -> impl SendFuture<ANNResult<()>>
    where
        S: PruneStrategy<DP>,
    {
        async move {
            let mut working_set = HashMap::default();

            let start: u64 = range.start.into();
            let end: u64 = range.end.into();

            let mut accessor = strategy
                .prune_accessor(&self.data_provider, context)
                .into_ann_result()?;

            let mut neighbors = AdjacencyList::with_capacity(self.max_degree_with_slack());
            let mut prune_scratch = prune::Scratch::<DP::InternalId>::new();

            for id in start..end {
                let id = id.try_into_vector_id()?;
                accessor.get_neighbors(id, &mut neighbors).await?;
                if neighbors.len() <= self.pruned_degree() {
                    continue;
                }

                working_set.clear();

                // Saturation is controlled by the index configuration.
                let options = prune::Options {
                    force_saturate: false,
                };

                self.robust_prune_list(
                    &mut accessor,
                    id,
                    &neighbors,
                    &mut prune_scratch,
                    &mut working_set,
                    options,
                )
                .await
                .escalate("prune-range does not support transient errors")?;

                accessor.set_neighbors(id, &prune_scratch.neighbors).await?;
            }

            Ok(())
        }
    }
}

/// A tracking variable to assist with book-keeping for lazy pruning.
/// This does imply the the maximum candidate pool size is now implicitly limited to
/// ```ignore
/// 65535
/// ```
/// We must conservatively protect against this in the pruning code.
struct InplaceDeleteWorkList<I> {
    /// Candidate replacements nodes for edges pointing to the node-to-delete.
    replace_candidates: Vec<I>,

    /// A collection of nodes with an edge pointing to the node-to-delete.
    in_neighbors: Vec<I>,
}

/// Private internal struct for recording search statistics.
struct InternalSearchStats {
    cmps: u32,
    hops: u32,
    range_search_second_round: bool,
}

impl InternalSearchStats {
    fn finish(self, result_count: u32) -> SearchStats {
        SearchStats {
            cmps: self.cmps,
            hops: self.hops,
            result_count,
            range_search_second_round: self.range_search_second_round,
        }
    }
}

#[cfg(feature = "experimental_diversity_search")]
impl<DP> DiskANNIndex<DP>
where
    DP: DataProvider,
{
    /// Create a diverse search scratch with DiverseNeighborQueue
    fn create_diverse_scratch(
        &self,
        l_value: usize,
        beam_width: Option<usize>,
        diverse_params: &DiverseSearchParams,
        k_value: usize,
    ) -> SearchScratch<DP::InternalId, crate::neighbor::DiverseNeighborQueue<DP::InternalId>> {
        use crate::neighbor::DiverseNeighborQueue;

        let attr_provider = diverse_params.attr_provider.clone();
        let diverse_queue = DiverseNeighborQueue::new(
            l_value,
            // SAFETY: k_value is guaranteed to be non-zero by SearchParams validation by caller
            #[allow(clippy::expect_used)]
            NonZeroUsize::new(k_value).expect("k_value must be non-zero"),
            diverse_params.diverse_results_k,
            attr_provider,
        );

        SearchScratch {
            best: diverse_queue,
            visited: HashSet::with_capacity(self.estimate_visited_set_capacity(Some(l_value))),
            id_scratch: Vec::with_capacity(self.max_degree_with_slack()),
            beam_nodes: Vec::with_capacity(beam_width.unwrap_or(1)),
            range_frontier: std::collections::VecDeque::new(),
            in_range: Vec::new(),
            hops: 0,
            cmps: 0,
        }
    }

    /// Experimental diverse search implementation using DiverseNeighborQueue.
    ///
    /// This method performs a graph-based search with diversity constraints, using the provided
    /// diverse search parameters to filter results based on attribute values.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The search strategy to use for accessing and processing elements.
    /// * `context` - The context to pass through to providers.
    /// * `query` - The query vector for which nearest neighbors are sought.
    /// * `search_params` - Parameters controlling the search behavior, including l_value, beam width, and k_value.
    /// * `diverse_params` - Diversity parameters including attribute provider and alpha value.
    /// * `output` - A mutable buffer to store the search results. Must be pre-allocated by the caller.
    /// * `search_record` - A mutable reference to a search record object that will record the path taken during the search.
    ///
    /// # Returns
    ///
    /// Returns search statistics including comparisons and hops performed.
    ///
    /// # Errors
    ///
    /// Returns an error if there is a failure accessing elements or if the provided parameters are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn diverse_search_experimental<S, T, O, OB, SR>(
        &self,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        search_params: &SearchParams,
        diverse_params: &DiverseSearchParams,
        output: &mut OB,
        search_record: &mut SR,
    ) -> impl SendFuture<ANNResult<SearchStats>>
    where
        T: Sync + ?Sized,
        S: glue::SearchStrategy<DP, T, O>,
        O: Send,
        OB: search_output_buffer::SearchOutputBuffer<O> + Send,
        SR: super::search::record::SearchRecord<DP::InternalId> + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&self.data_provider, context)
                .into_ann_result()?;

            let computer = accessor.build_query_computer(query).into_ann_result()?;
            let start_ids = accessor.starting_points().await?;

            // Use diverse search with DiverseNeighborQueue
            // TODO: Use scratch pool in future PRs to avoid allocation.
            let mut diverse_scratch = self.create_diverse_scratch(
                search_params.l_value,
                search_params.beam_width,
                diverse_params,
                search_params.k_value,
            );

            let stats = self
                .search_internal(
                    search_params.beam_width,
                    &start_ids,
                    &mut accessor,
                    &computer,
                    &mut diverse_scratch,
                    search_record,
                )
                .await?;

            // Post-process diverse results to keep only diverse_results_k items
            diverse_scratch.best.post_process();

            // TODO: Post processing will change for diverse search in future PRs
            let result_count = strategy
                .post_processor()
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    diverse_scratch
                        .best
                        .iter()
                        .take(search_params.l_value.into_usize()),
                    output,
                )
                .send()
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}
