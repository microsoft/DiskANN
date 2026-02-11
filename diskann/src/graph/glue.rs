/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Strategies provide the "glue" between [`DataProvider`]s and index operations like search,
//! insertion, deletion, etc.
//!
//! They do this by tying together [`Accessor`]s, computers, and other operations required
//! by the various indexing algorithms.
//!
//! The relationship between between strategies and indexing algorithms can feel like having
//! "action at a distance", so this module-level documentation describes the logical flow
//! behind the strategy types defined here.
//!
//! Type level documentation contains more detailed description behind the actual mechanics.
//!
//! * [`SearchStrategy`]: This is used for default graph-based searches. The flow of a search
//!   is as follows:
//!
//!   1. Create the `SearchAccessor` defined by the strategy. This is the object that will
//!      be used to actually retrieve data from the [`DataProvider`].
//!
//!   2. Use the `SearchAccessor` to create the `QueryComputer` from the query object.
//!      This computer will be used for all distances between elements retrieved from the
//!      search accessor.
//!
//!   3. Run greedy-search using the accessor-computer combination.
//!
//!   4. After search, post-processing is run, which enables operations like filtering
//!      of start or deleted points, reranking etc.
//!
//!      See [`SearchPostProcess`].
//!
//! * [`SearchPostProcess`]/[`SearchPostProcessStep`]: These traits enable cascadable
//!   post-processing pipelines for the results of search. The first is responsible for
//!   transforming an input iterator of search results into a [`SearchOutputBuffer`].
//!
//!   The second enables modification of the input or output streams and associated accessors.
//!
//!   An arbitrary number of [`SearchPostProcessStep`]s can be cascaded together using
//!   a [`Pipeline`].
//!
//! * [`InsertStrategy`]: Graph insertion consists of accepting the value to insert,
//!   invoking [`crate::model::graph::traits::data_provider::SetElement`] on that value,
//!   then performing a graph search.
//!
//!   Following graph search, candidates are passed to a pruning phase.
//!
//!   Most of the [`InsertStrategy`] is dedicated to the initial graph search, delegating
//!   pruning to an associated [`PruneStrategy`].
//!
//!   The graph search portion works by constructing the `InsertAccessor` and then an
//!   `InsertQueryComputer` from that accessor and the value that was just inserted.
//!
//!   This accessor/computer pair is then used for search, followed by pruning.
//!
//! * [`PruneStrategy`]: The pruning strategy is largely straightforward, consisting of
//!   an [`Accessor`] and a random access [`DistanceFunction`] for performing distance
//!   calculations on the retrieve elements.
//!
//!   One subtle aspect is the use of the [`FillSet`] trait. For clients with expensive
//!   vector retrieval calls, we wish to only retrieve vector IDs once for pruning.
//!
//!   This is done by using a hash-map to store the elements retrieved by the
//!   `PruneAccessor`. To give implementers the ability to perform a hybrid prune
//!   (consisting of a mix of full-precision and quantized vectors), **without** the index
//!   algorithm being aware of these two levels, the trait [`FillSet`] is used to delegate
//!   the responsibility of populating this cache to the [`DataProvider`].
//!
//! * [`InplaceDeleteStrategy`]: This follows the trend of defining accessors and related
//!   strategies. One difference for inplace-deletion is that we use an element accessed
//!   from the [`DataProvider`] itself for search. Therefore, methods like
//!   [`crate::index::diskann_async::DiskANNIndex::inplace_delete`] also require the
//!   [`InplaceDeleteStrategy`] to implement an appropriate [`SearchStrategy`].
//!
//!   Like insertion, this trait delegates pruning to a dedicated `PruneStrategy`.

use std::{collections::HashMap, future::Future};

use diskann_utils::future::AssertSend;
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction};

use crate::{
    ANNError, ANNResult,
    error::{ErrorExt, RankedError, StandardError, ToRanked, TransientError},
    graph::{AdjacencyList, SearchOutputBuffer},
    neighbor::Neighbor,
    provider::{
        Accessor, AsNeighbor, AsNeighborMut, BuildDistanceComputer, BuildQueryComputer,
        DataProvider, HasId, NeighborAccessor,
    },
    utils::VectorId,
};

/// A trait to override search constraints such as early termination based on constraints
/// by implementer.
pub trait SearchExt: Accessor {
    /// Return a `Vec` containing the starting points.
    fn starting_points(&self)
    -> impl std::future::Future<Output = ANNResult<Vec<Self::Id>>> + Send;

    /// Default is to never terminate early.
    fn terminate_early(&mut self) -> bool {
        false
    }

    //////////////////////
    // Provided methods //
    //////////////////////

    /// Return a `'static` closure that returns `true` if a provided `id` is not a start
    /// point - otherwise returning `false`.
    ///
    /// The provided implementation using `self.starting_points()` to obtain the collection
    /// of start points. Implementations may choose to specialize this if they have a more
    /// efficient means of providing the filter.
    fn is_not_start_point(
        &self,
    ) -> impl std::future::Future<
        Output = ANNResult<impl Fn(Self::Id) -> bool + Send + Sync + 'static>,
    > + Send {
        async move {
            let starting_points = self.starting_points().await?;
            Ok(move |id| !starting_points.contains(&id))
        }
    }
}

/// An predicate evaluating `item`.
pub trait Predicate<T> {
    fn eval(&self, item: &T) -> bool;
}

/// A mutable predicate evaluating `item`.
///
/// Proper implementations of `PredicateMut` may update `self` to return `false` after
/// returning `true` once for a given value of `item`.
///
/// However, the following must be observed to keep the implementation easy to reason about.
///
/// 1. When passed as a generic argument, implementations must never switch from returing
///    `false` for a particular value of `item` to `true`.
///
///    In other worde, items **excluded** by the predicate must never switch to being
///    **included**.
///
/// 2. Inclusion **is** allowed to go the other way, from being **included** to being
///    **excluded**, provided point (1) is observed.
pub trait PredicateMut<T> {
    fn eval_mut(&mut self, item: &T) -> bool;
}

/// A predicate that can be used in both an immutable and a mutable context.
///
/// This is a marker trait with no provided implementations.
///
/// Implementors must ensure that the immutable and mutable versions of the predicate "agree"
/// on element inclusion.
pub trait HybridPredicate<T>: Predicate<T> + PredicateMut<T> {}

/// A [`HybridPredicate`] using a hash-set as its implementation. The implementation of
/// [`Predicate`] checks for presence in the table while [`PredicateMut`] tries to insert the
/// item, indicates whether or not the item already existed.
pub struct NotInMut<'a, K>(&'a mut hashbrown::HashSet<K>);

impl<'a, K> NotInMut<'a, K> {
    /// Construct a new `NotInMut` around `set`.
    pub fn new(set: &'a mut hashbrown::HashSet<K>) -> Self {
        Self(set)
    }
}

impl<T> Predicate<T> for NotInMut<'_, T>
where
    T: Eq + std::hash::Hash,
{
    fn eval(&self, item: &T) -> bool {
        !self.0.contains(item)
    }
}

impl<T> PredicateMut<T> for NotInMut<'_, T>
where
    T: Clone + Eq + std::hash::Hash,
{
    fn eval_mut(&mut self, item: &T) -> bool {
        self.0.insert(item.clone())
    }
}

/// The interfaces `contains` and `insert` agree with each other.
impl<T> HybridPredicate<T> for NotInMut<'_, T> where T: Clone + Eq + std::hash::Hash {}

/// A primitive routine used by graph search. This is purposely implemented as a
/// coarse-grained operation to enable optimization opportunities by the backend.
///
/// # Description
///
/// For each `i` in `itr`, fetch the adjacency list `v_i` for `i`. For each `v_i`, then for
/// each id `j` in `v_i`, compute the distance `d` using `computer` to the data associated with
/// `j` and invoke the closure `f` with `d` and `j`, provided `pred.eval_mut(j)` evaluates to
/// `true`.
///
/// No specification is made on the traversal order of `ids`, the computation order of the
/// leaf elements, nor the order in which `pred` is evaluated.
///
/// Restriction in the implementation are as follows:
///
/// * If `pred.eval_mut()` returns `true` for an id `i`, then `on_neighbors` must be invoked
///   for that item.
///
///   If an item `i` is already passed to `on_neighbors`, the implementation is not obligated
///   to provided it again, though it **may** do so provided `pred.eval_mut()` continues to
///   return `true`.
///
/// * If `pred.eval_mut()` returns `false` for an item, then `on_neighbors` must not be
///   invoked for that item.
///
/// * `pred.eval()` may be invoked an arbitrary number of times. Proper predicate
///   implementations  will ensure that
///
///   - `pred.eval() == true` implies `pred.eval_mut() == true` if `pred.eval_mut()` is
///     invoked immediately after `pred.eval()`.
///
///   - `pred.eval() == false` implies `pred.eval_mut() == false` and vice-versa.
///
/// * `pred.eval_mut()` must be invoked at most once for all transitive items in the beam.
///
/// ## Predicate Requirements
///
/// Well behaved predicates must never return `true` (allow an id to be forwarded to
/// `on_neighbors`) if it previously returned `false`. Implementations of `ExpandBeam` are
/// allowed to assume this holds.
///
/// Additionally, the callback `on_neighbors` and the predicate have to cooperate. If the
/// callback requires unique items, the predicate must be structured such that `eval_mut`
/// correctly filters out duplicates.
///
/// # Provided Implementation
///
/// The provided implementation works on each element of `ids` sequentially, pre-filters
/// the resulting candidate list using `pred.eval()` before invoking
/// [`BuildQueryComputer::distances_unordered`].
///
/// The callback `on_neighbors` is decorated to the uses `pred.eval_mut()`.
///
/// This ensures that if `distances_unordered` errors, the predicate is not erroneously
/// updated.
///
/// ## Error Handling
///
/// Transient errors yielded by `distances_unordered` are acknowledged and not escalated.
pub trait ExpandBeam<T>: BuildQueryComputer<T> + AsNeighbor + Sized
where
    T: ?Sized,
{
    /// Submit IDs to the expansion queue.
    ///
    /// For non-pipelined providers (default), IDs are stored in an internal buffer and
    /// processed synchronously in [`expand_available`]. For pipelined providers, this
    /// submits non-blocking IO requests (e.g., io_uring reads) so that data loading
    /// overlaps with other computation.
    ///
    /// The default implementation delegates to [`expand_beam`] from within
    /// [`expand_available`], so overriding this method is only necessary for pipelined
    /// providers that need to separate submission from completion.
    fn submit_expand(&mut self, _ids: impl Iterator<Item = Self::Id> + Send) {
        // Default: no-op. IDs are passed directly to expand_beam in expand_available.
    }

    /// Expand nodes whose data is available, invoking `on_neighbors` for each discovered
    /// neighbor.
    ///
    /// For non-pipelined providers (default), this expands all the `ids` passed in
    /// synchronously via [`expand_beam`]. For pipelined providers, this polls for
    /// completed IO operations and expands only the nodes whose data has arrived,
    /// returning immediately without blocking.
    ///
    /// Returns the number of nodes that were expanded in this call.
    fn expand_available<P, F>(
        &mut self,
        ids: impl Iterator<Item = Self::Id> + Send,
        computer: &Self::QueryComputer,
        pred: P,
        on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<usize>> + Send
    where
        P: HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(f32, Self::Id) + Send,
    {
        async move {
            let id_vec: Vec<Self::Id> = ids.collect();
            let count = id_vec.len();
            self.expand_beam(id_vec.into_iter(), computer, pred, on_neighbors)
                .await?;
            Ok(count)
        }
    }

    /// Returns true if there are submitted but not-yet-expanded nodes pending.
    ///
    /// For non-pipelined providers (default), this always returns `false` since
    /// [`expand_available`] processes everything synchronously. Pipelined providers
    /// return `true` when IO operations are in-flight.
    fn has_pending(&self) -> bool {
        false
    }

    /// Returns the number of IOs currently in-flight (submitted but not completed).
    ///
    /// The search loop uses this to cap submissions at `cur_beam_width`, matching
    /// PipeSearch's behavior of not over-committing speculative reads.
    /// Default: 0 (non-pipelined providers have no in-flight IO).
    fn inflight_count(&self) -> usize {
        0
    }

    /// Block until at least one IO completes, then eagerly drain all available.
    ///
    /// Called by the search loop only when it cannot make progress: nothing was
    /// submitted (no candidates or inflight cap reached) AND nothing was expanded
    /// (no completions available). Blocking here yields the CPU thread instead of
    /// spin-polling, while the eager drain ensures we process bursts efficiently.
    ///
    /// Default: no-op (non-pipelined providers never need to wait).
    fn wait_for_io(&mut self) {}

    /// Return the IDs of nodes expanded in the most recent `expand_available` call.
    ///
    /// The search loop uses this to mark speculatively submitted nodes as visited
    /// only after they have actually been expanded. Non-pipelined providers return
    /// an empty slice (they mark visited at selection time).
    fn last_expanded_ids(&self) -> &[Self::Id] {
        &[]
    }

    /// Expand all `ids` synchronously: load data, get neighbors, compute distances.
    ///
    /// This is the original single-shot expansion method. For non-pipelined providers,
    /// the default [`expand_available`] delegates to this. Pipelined providers may
    /// override [`submit_expand`] and [`expand_available`] instead and leave this as
    /// the default.
    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        computer: &Self::QueryComputer,
        mut pred: P,
        mut on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(f32, Self::Id) + Send,
    {
        async move {
            let mut neighbors = AdjacencyList::new();
            for id in ids {
                self.get_neighbors(id, &mut neighbors).send().await?;
                neighbors.retain(|i| pred.eval(i));

                self.distances_unordered(neighbors.iter().copied(), computer, |distance, id| {
                    if pred.eval_mut(&id) {
                        on_neighbors(distance, id);
                    }
                })
                .send()
                .await
                .allow_transient("allowing transient error in beam expansion")?;
            }

            Ok(())
        }
    }
}

/// A search strategy for query objects of type `T`.
///
/// This trait should be overloaded by data providers wishing to extend
///
/// The type `O` represents the type written into the output buffer
/// during a search. This is often the same as the provider's internal ID type,
/// but it can differ depending on the use case. For example, it might represent
/// associated data or alternative identifiers.
///
/// [`crate::index::diskann_async::DiskANNIndex::search`].
pub trait SearchStrategy<Provider, T, O = <Provider as DataProvider>::InternalId>:
    Send + Sync
where
    Provider: DataProvider,
    T: ?Sized,
    O: Send,
{
    /// The computer used by the associated accessor.
    ///
    /// We could grab this type from the `SearchAccessor` associated type, but it's
    /// useful enough that we move it up here.
    type QueryComputer: for<'a, 'b> PreprocessedDistanceFunction<
            <Self::SearchAccessor<'a> as Accessor>::ElementRef<'b>,
            f32,
        > + Send
        + Sync
        + 'static;

    /// The associated [`SearchPostProcess`]or for the final results.
    type PostProcessor: for<'a> SearchPostProcess<Self::SearchAccessor<'a>, T, O> + Send + Sync;

    /// An error that can occur when getting a search_accessor.
    type SearchAccessorError: StandardError;

    /// The concrete type of the accessor that is used to access `Self` during the greedy
    /// graph search. The query will be provided to the accessor exactly once during search
    /// to construct the query computer.
    type SearchAccessor<'a>: ExpandBeam<T, QueryComputer = Self::QueryComputer, Id = Provider::InternalId>
        + SearchExt;

    /// Construct and return the search accessor.
    fn search_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError>;

    /// Construct the [`SearchPostProcess`] struct to post-process the results of search and
    /// store them into the output container.
    fn post_processor(&self) -> Self::PostProcessor;
}

/// Perform post-processing on the results of search, storing the results in an output buffer.
///
/// Simple implementations include [`CopyIds`], which simply forwards the search results
/// directly into the output buffer.
pub trait SearchPostProcess<A, T, O = <A as HasId>::Id>
where
    A: BuildQueryComputer<T>,
    T: ?Sized,
{
    type Error: StandardError;

    /// Populate `output` with the entries in `candidates`. Correct implementations must
    /// return the number of results copied into `output` on success.
    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: &T,
        computer: &<A as BuildQueryComputer<T>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized;
}

/// A [`SearchPostProcess`] base object that copies maps each `Neighbor` to a `(Id, f32)` pair
/// and writes as many as possible to the output buffer.
#[derive(Debug, Default, Clone, Copy)]
pub struct CopyIds;

impl<A, T> SearchPostProcess<A, T> for CopyIds
where
    A: BuildQueryComputer<T>,
    T: ?Sized,
{
    type Error = std::convert::Infallible;
    fn post_process<I, B>(
        &self,
        _accessor: &mut A,
        _query: &T,
        _computer: &A::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<A::Id> + Send + ?Sized,
    {
        let count = output.extend(candidates.map(|n| (n.id, n.distance)));
        std::future::ready(Ok(count))
    }
}

/// A transformation step helpful for modifying input iterators and accessor types for
/// [`SearchPostProcess`]. Multiple instances of [`SearchPostProcessStep`] can be cascaded
/// using a [`Pipeline`].
pub trait SearchPostProcessStep<A, T, O = <A as HasId>::Id>
where
    A: BuildQueryComputer<T>,
    T: ?Sized,
{
    /// A potentially modified version of the error yielded by the next state in the
    /// processing pipeline.
    type Error<NextError>: StandardError
    where
        NextError: StandardError;

    /// The accessor that will be passed to the next processing stage.
    type NextAccessor: BuildQueryComputer<T, Id = A::Id>;

    /// Perform any modification the `input`, `output`, `accessor`, or `computer` objects
    /// and invoke the [`SearchPostProcess`] routine `next` on stage.
    fn post_process_step<I, B, Next>(
        &self,
        next: &Next,
        accessor: &mut A,
        query: &T,
        computer: &<A as BuildQueryComputer<T>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error<Next::Error>>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized,
        Next: SearchPostProcess<Self::NextAccessor, T, O> + Sync;
}

/// A [`SearchPostProcessStep`] that filters out start points from the candidate stream.
#[derive(Debug, Default, Clone, Copy)]
pub struct FilterStartPoints;

impl<A, T, O> SearchPostProcessStep<A, T, O> for FilterStartPoints
where
    A: BuildQueryComputer<T> + SearchExt,
    T: Send + Sync + ?Sized,
{
    /// A this level, sub-errors are converted into [`ANNError`] to provide additional
    /// context idenfying the problem as occurring in a sub-process of filtering.
    type Error<NextError>
        = ANNError
    where
        NextError: StandardError;

    /// The accessor is unmodified.
    type NextAccessor = A;

    async fn post_process_step<I, B, Next>(
        &self,
        next: &Next,
        accessor: &mut A,
        query: &T,
        computer: &A::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> ANNResult<usize>
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized,
        Next: SearchPostProcess<A, T, O> + Sync,
    {
        let filter = accessor.is_not_start_point().await?;
        next.post_process(
            accessor,
            query,
            computer,
            candidates.filter(|n| filter(n.id)),
            output,
        )
        .await
        .map_err(|err| {
            let err = err.into();
            err.context("after filtering start points")
        })
    }
}

/// A structure for composing arbitrary [`SearchPostProcessStep`]s in front of a base
/// [`SearchPostProcess`].
///
/// This type implements [`SearchPostProcess`] when `Head: SearchPostProcessStep` and
/// `Tail: SearchPostProcess`. To compose three processing steps `A`, `B`, and `C` with a
/// final [`SearchPostProcess`] `D`, the type
/// ```text
/// Pipeline<A, Pipeline<B, Pipeline<C, D>>>
/// ```
/// can be used. The outermost type will apply the step `A` before recursing to
/// `Pipeline<B, Pipeline<C, D>>`, which will then apply `B` before invoking `Pipeline<C, D>`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Pipeline<Head, Tail> {
    head: Head,
    tail: Tail,
}

impl<Head, Tail> Pipeline<Head, Tail> {
    /// Construct a new [`Pipeline`] with the provided `Head` and `Tail` components.
    pub fn new(head: Head, tail: Tail) -> Self {
        Self { head, tail }
    }
}

impl<A, T, O, Head, Tail> SearchPostProcess<A, T, O> for Pipeline<Head, Tail>
where
    A: BuildQueryComputer<T>,
    T: ?Sized,
    Head: SearchPostProcessStep<A, T, O>,
    Tail: SearchPostProcess<Head::NextAccessor, T, O> + Sync,
{
    type Error = Head::Error<Tail::Error>;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: &T,
        computer: &<A as BuildQueryComputer<T>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized,
    {
        self.head
            .post_process_step(&self.tail, accessor, query, computer, candidates, output)
    }
}

/// A strategy for inserting elements from the data provider.
///
/// This strategy is used during the greedy search portion of index construction.
/// After the candidate list has been retrieved from greedy search, the [`PruneStrategy`]
/// is used for the rest.
pub trait InsertStrategy<Provider, T>: SearchStrategy<Provider, T> + 'static
where
    Provider: DataProvider,
    T: ?Sized,
{
    /// The pruning strategy associated with the insertion strategy.
    type PruneStrategy: PruneStrategy<Provider>;

    /// Return the prune strategy used for insertion.
    fn prune_strategy(&self) -> Self::PruneStrategy;

    /// This API is invoked during inserts to create the associated `SearchAccessor`.
    ///
    /// The provided implementation uses
    /// [`<Self as SearchStrategy<Provider, T>>::search_accessor`], but implementors of
    /// [`InsertStrategy`] can customize the implementation if the behavior of the search
    /// accessor needs to be slightly different between searches for build and regular
    /// searches.
    fn insert_search_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        self.search_accessor(provider, context)
    }
}

/// A strategy for pruning elements from the data provider.
///
/// This strategy type does not have an additional `T` parameter because there is not a
/// well-defined query-type to hook onto. Instead, `PruneStrategies` are reached via
/// other types like [`InsertStrategy`] or [`InplaceDeleteStrategy`].
pub trait PruneStrategy<Provider>: Send + Sync + 'static
where
    Provider: DataProvider,
{
    /// The distance computer used during pruning.
    ///
    /// We could grab this type from the `PruneAccessor` associated type, but it's
    /// useful enough that we move it up here.
    type DistanceComputer: for<'a, 'b, 'c, 'd> DistanceFunction<
            <Self::PruneAccessor<'a> as Accessor>::ElementRef<'b>,
            <Self::PruneAccessor<'c> as Accessor>::ElementRef<'d>,
            f32,
        > + Send
        + Sync
        + 'static;

    /// The concrete type of the accessor that is used to access `Self` during pruning.
    ///
    /// Note that the majority of interactions with the working set will go through
    /// [`FillSet`] rather than [`Accessor::get_element`].
    ///
    /// Therefore, implementations are encouraged to have [`Accessor::get_element`] return
    /// the highest-precision applicable value for a given element type.
    type PruneAccessor<'a>: Accessor<Id = Provider::InternalId>
        + BuildDistanceComputer<DistanceComputer = Self::DistanceComputer>
        + AsNeighborMut
        + FillSet;

    /// An error that can occur when getting the prune accessor.
    type PruneAccessorError: StandardError;

    /// Return the prune accessor.
    fn prune_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError>;
}

/// An extension point for filling the working set used during pruning.
///
/// A working set is needed because pruning operates computes distances between many
/// different pairs of a relatively small set of elements. For providers where element
/// retrieval is expensive, we need to cache these elements on the DiskANN layer to only
/// retrieve each element once.
///
/// The motivation for this trait is as follows:
///
/// 1. It allows two-level [`DataProvider`]s (for example, those that have a full precision
///    **and** and quantized copy of each element) to populate the working set with a
///    mixture of full-precision and quantized vectors.
///
/// 2. This also allows accessors that have more efficient bulk accesses to implement some
///    or all of the working set population using these more efficient implementations.
///
/// # Provided Implementation
///
/// The provided implementation use [`self.get_element()`] to retrieve items. This method
/// is only called if the key is not already present in `set`.
///
/// This method allows transient errors.
pub trait FillSet: Accessor {
    /// Add `self.get_element(i)` for all `i` in `itr`.
    ///
    /// Implementations should not remove elements from `set`, only add.
    fn fill_set<Itr>(
        &mut self,
        set: &mut HashMap<Self::Id, Self::Extended>,
        itr: Itr,
    ) -> impl Future<Output = Result<(), Self::GetError>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send + Sync,
    {
        use std::collections::hash_map;
        async move {
            // The dance here is a little complicated.
            //
            // If we observe a transient error, then we are obligated to continue processing
            // items, but must return a transient error to inform the caller that not all
            // the items made it through.
            let mut err: Result<(), Self::GetError> = Ok(());
            for i in itr {
                let e = set.entry(i);
                if matches!(e, hash_map::Entry::Vacant(_)) {
                    match self.get_element(i).await {
                        Ok(element) => {
                            e.insert_entry(element.into());
                        }
                        Err(e) => match e.to_ranked() {
                            RankedError::Transient(transient) => {
                                if err.is_ok() {
                                    err = Err(Self::GetError::from_transient(transient));
                                } else {
                                    transient.acknowledge(
                                        "another non-critical error was observed first",
                                    );
                                }
                            }
                            RankedError::Error(e) => return Err(Self::GetError::from_error(e)),
                        },
                    }
                }
            }
            err
        }
    }
}

/// For compatibility with [`crate::index::diskann_async::DiskANNIndex::multi_insert`],
/// we need the ability to take vectors supplied directly from the insert batch and use those
/// vectors directly in pruning.
///
/// Vectors for `multi_insert` are constrained to be slices, but `Accessor::Element`
/// is an opaque type. This trait works as a bridge from the input slices to
/// `Accessor::Element` and often can be implemented cheaply.
///
/// This trait also accepts the internal id of the vector and is guaranteed to be called
/// after [`SetElement`]. This allows the provider to simply retrieve a pre-processed
/// vector from the internal store if that is more efficient.
pub trait AsElement<T>: Accessor {
    type Error: ToRanked + std::fmt::Debug + Send + Sync;
    fn as_element(
        &mut self,
        vector: T,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::Error>> + Send;
}

/// A strategy for supporting inplace deletes.
///
/// In place deletes consist of two phases:
///
/// 1. An initial graph search using a value extracted from the index to generate a list
///    of candidates.
///
/// 2. Multiple rounds of pruning on the extracted candidates list.
///
/// Like [`PruneStrategy`], inplace-deletion does not have a query type `T` to be generic
/// over. Thus, interfaces accept an [`InplaceDeleteStrategy`] directly.
pub trait InplaceDeleteStrategy<Provider>: Send + Sync + 'static
where
    Provider: DataProvider,
{
    /// The type provided to the search portion of inplace-deletion and used to instantiate
    /// the required `SearchStrategy`.
    ///
    /// This is allowed to be different then `DeleteElementGuard` so we can support types
    /// without lifetimes.
    type DeleteElement<'a>: Send + Sync + ?Sized;

    /// Why do we need the `Lower` bound here?
    ///
    /// The answer is that search strategies are implemented to `T: ?Sized`, but the return
    /// type of an accessor can never be `?Sized`.
    ///
    /// This presents a conundrum because if we implement, for example,
    /// [`SearchStrategy<[f32]>`], we would like to be able to reuse that search strategy
    /// for the search portion of inplace deletes.
    ///
    /// This means that we need to be able to transform the value returned by
    /// [`Accessor::get_element`] to be able to decay to a non-reference type.
    ///
    /// Note that it is possible for particular `DeleteElements` to `Deref` to themselves.
    ///
    /// Additionally, the `Deref` bound allows a guard to be returned instead of the raw
    /// type while still dispatching to an unguarded search routine.
    type DeleteElementGuard: Send
        + Sync
        + for<'a> diskann_utils::reborrow::AsyncLower<'a, Self::DeleteElement<'a>>
        + 'static;

    /// Error type for accessing for search.
    type DeleteElementError: StandardError;

    /// The pruning strategy to use after the initial search is complete.
    type PruneStrategy: PruneStrategy<Provider>;

    /// The type of the search strategy to use for graph traversal.
    type SearchStrategy: for<'a> SearchStrategy<Provider, Self::DeleteElement<'a>>;

    /// Construct the prune strategy object.
    fn prune_strategy(&self) -> Self::PruneStrategy;

    /// Construct the search strategy object.
    fn search_strategy(&self) -> Self::SearchStrategy;

    /// Construct the accessor used to retrieve the item being deleted.
    fn get_delete_element<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        id: Provider::InternalId,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send;
}

/// Provides asynchronous access to an iterator over vector IDs.
///
/// This trait defines a method to asynchronously retrieve an iterator over vector IDs.
///
/// # Type Parameters
///
/// - `I`: The iterator type returned by the accessor. It must implement `Iterator` with items of type implementing `VectorId`.
///
/// # Errors
///
/// Returns an [`ANNError`] if the iterator cannot be retrieved successfully.
pub trait IdIterator<I>
where
    I: Iterator<Item: VectorId>,
{
    fn id_iterator(&mut self) -> impl std::future::Future<Output = Result<I, ANNError>>;
}

pub mod aliases {
    use super::*;

    /// A convenience alias returning the prune accessor for an insert strategy `Strategy`.
    pub type InsertPruneAccessor<'a, Strategy, Provider, T> = <<Strategy as InsertStrategy<
        Provider,
        T,
    >>::PruneStrategy as PruneStrategy<Provider>>::PruneAccessor<'a>;
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use diskann_vector::PreprocessedDistanceFunction;
    use futures_util::future;
    use thiserror::Error;

    use super::*;
    use crate::{
        ANNResult, neighbor,
        provider::{DelegateNeighbor, ExecutionContext, HasId, NeighborAccessor},
    };

    // A really simple provider that just holds floats and uses the absolute value for its
    // distances.
    struct SimpleProvider {
        items: Vec<f32>,
    }

    #[derive(Default, Clone)]
    struct CountGetVector {
        count: Arc<AtomicUsize>,
    }
    impl ExecutionContext for CountGetVector {}

    impl CountGetVector {
        fn count(&self) -> usize {
            self.count.load(Ordering::Relaxed)
        }

        fn clear(&self) {
            self.count.store(0, Ordering::Relaxed)
        }
    }

    impl DataProvider for SimpleProvider {
        type Context = CountGetVector;
        type InternalId = u32;
        type ExternalId = u32;
        type Error = ANNError;

        /// Translate an external id to its corresponding internal id.
        fn to_internal_id(
            &self,
            _context: &CountGetVector,
            gid: &Self::ExternalId,
        ) -> Result<Self::InternalId, Self::Error> {
            Ok(*gid)
        }

        /// Translate an internal id to its corresponding external id.
        fn to_external_id(
            &self,
            _context: &CountGetVector,
            id: Self::InternalId,
        ) -> Result<Self::ExternalId, Self::Error> {
            Ok(id)
        }
    }

    #[derive(Clone, Copy)]
    struct Retriever<'a> {
        provider: &'a SimpleProvider,
        count: &'a CountGetVector,
        errors_are_unrecoverable: bool,
    }

    impl SearchExt for Retriever<'_> {
        async fn starting_points(&self) -> ANNResult<Vec<u32>> {
            Ok(vec![0])
        }
    }

    impl<'a> Retriever<'a> {
        fn new(
            provider: &'a SimpleProvider,
            count: &'a CountGetVector,
            errors_are_unrecoverable: bool,
        ) -> Self {
            Self {
                provider,
                count,
                errors_are_unrecoverable,
            }
        }
    }

    impl HasId for Retriever<'_> {
        type Id = u32;
    }

    impl Accessor for Retriever<'_> {
        type Extended = f32;
        type Element<'a>
            = f32
        where
            Self: 'a;
        type ElementRef<'a> = f32;

        type GetError = InvalidVectorId;
        fn get_element(
            &mut self,
            id: Self::Id,
        ) -> impl std::future::Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send
        {
            let result = match self.provider.items.get(id as usize) {
                Some(v) => {
                    self.count.count.fetch_add(1, Ordering::Relaxed);
                    Ok(*v)
                }
                None => Err(InvalidVectorId::new(self.errors_are_unrecoverable)),
            };
            async move { result }
        }
    }

    impl NeighborAccessor for Retriever<'_> {
        fn get_neighbors(
            self,
            _id: Self::Id,
            neighbors: &mut AdjacencyList<Self::Id>,
        ) -> impl Future<Output = ANNResult<Self>> + Send {
            neighbors.clear();
            future::ok(self)
        }
    }

    #[derive(Debug, Error)]
    #[error("invalid vector id")]
    struct InvalidVectorId {
        unrecoverable: bool,
        acknowledged: bool,
    }

    impl InvalidVectorId {
        fn new(unrecoverable: bool) -> Self {
            Self {
                unrecoverable,
                acknowledged: false,
            }
        }
    }

    impl TransientError<Self> for InvalidVectorId {
        fn acknowledge<D: std::fmt::Display>(mut self, _why: D) {
            assert!(!self.unrecoverable);
            self.acknowledged = true;
        }

        fn escalate<D: std::fmt::Display>(mut self, _why: D) -> Self {
            assert!(!self.acknowledged);
            self.unrecoverable = true;
            self
        }
    }

    impl ToRanked for InvalidVectorId {
        type Transient = Self;
        type Error = Self;

        fn to_ranked(self) -> RankedError<Self, Self> {
            if self.unrecoverable {
                RankedError::Error(self)
            } else {
                RankedError::Transient(self)
            }
        }

        fn from_transient(transient: Self) -> Self {
            transient
        }

        fn from_error(error: Self) -> Self {
            error
        }
    }

    impl Drop for InvalidVectorId {
        fn drop(&mut self) {
            if !self.unrecoverable && !self.acknowledged {
                panic!("Unacknowledged recoverable error dropped!");
            }
        }
    }

    impl From<InvalidVectorId> for ANNError {
        fn from(value: InvalidVectorId) -> Self {
            assert!(value.unrecoverable);
            ANNError::log_async_error(value)
        }
    }

    struct QueryComputer;
    impl PreprocessedDistanceFunction<f32, f32> for QueryComputer {
        fn evaluate_similarity(&self, _changing: f32) -> f32 {
            panic!("this method should not be called")
        }
    }

    impl BuildQueryComputer<f32> for Retriever<'_> {
        type QueryComputerError = ANNError;
        type QueryComputer = QueryComputer;
        fn build_query_computer(&self, _from: &f32) -> Result<QueryComputer, ANNError> {
            Ok(QueryComputer)
        }
    }

    impl ExpandBeam<f32> for Retriever<'_> {}

    // This strategy explicitly does not define `post_process` so we can test the provided
    // implementation.
    struct Strategy {
        errors_are_unrecoverable: bool,
    }

    impl SearchStrategy<SimpleProvider, f32> for Strategy {
        type QueryComputer = QueryComputer;
        type PostProcessor = CopyIds;
        type SearchAccessorError = ANNError;
        type SearchAccessor<'a> = Retriever<'a>;

        fn search_accessor<'a>(
            &'a self,
            provider: &'a SimpleProvider,
            context: &'a CountGetVector,
        ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
            Ok(Retriever::new(
                provider,
                context,
                self.errors_are_unrecoverable,
            ))
        }

        fn post_processor(&self) -> Self::PostProcessor {
            Default::default()
        }
    }

    // Use the provided implementation.
    impl FillSet for Retriever<'_> {}

    #[tokio::test(flavor = "current_thread")]
    async fn test_default_post_process() {
        let ctx = CountGetVector::default();
        let strategy = Strategy {
            errors_are_unrecoverable: true,
        };

        let num_points: usize = 100;
        let provider = SimpleProvider {
            items: (0..num_points).map(|i| i as f32).collect(),
        };

        assert_eq!(provider.to_internal_id(&ctx, &10).unwrap(), 10);
        assert_eq!(provider.to_external_id(&ctx, 10).unwrap(), 10);

        let mut accessor = strategy.search_accessor(&provider, &ctx).unwrap();
        assert_eq!(accessor.starting_points().await.unwrap().as_slice(), &[0]);
        for i in 0..num_points {
            assert_eq!(accessor.get_element(i as u32).await.unwrap(), i as f32);
        }

        // Check dummy get_neighbors implmeentation for code coverage
        let mut neighbors = AdjacencyList::new();
        accessor
            .delegate_neighbor()
            .get_neighbors(0, &mut neighbors)
            .await
            .unwrap();
        assert_eq!(neighbors.len(), 0);

        // Check that the correct number of reads were emitted.
        assert_eq!(ctx.count(), num_points);
        ctx.clear();

        let query = 11.5;
        let computer = accessor.build_query_computer(&query).unwrap();

        for input_len in 0..10 {
            let input: Vec<_> = (0..input_len)
                .map(|i| Neighbor::<u32>::new(i as u32, i as f32))
                .collect();
            for output_len in 0..10 {
                let mut output = vec![Neighbor::<u32>::default(); output_len];

                let count = strategy
                    .post_processor()
                    .post_process(
                        &mut accessor,
                        &query,
                        &computer,
                        input.iter().copied(),
                        &mut neighbor::BackInserter::new(output.as_mut_slice()),
                    )
                    .await
                    .unwrap();

                assert_eq!(count, input_len.min(output_len));

                // Check that the in-range values were properly copied.
                for (i, n) in output.iter().take(count).enumerate() {
                    assert_eq!(i, n.id as usize);
                    assert_eq!(i as f32, n.distance);
                }

                // Check that out-of-range values were untouched.
                for n in output.iter().skip(count) {
                    assert_eq!(n.id, 0);
                    assert_eq!(n.distance, 0.0);
                }
            }
        }

        // Ensure that no reads were emitted.
        assert_eq!(ctx.count(), 0);
    }

    #[tokio::test]
    async fn test_default_fill_set() {
        let ctx = CountGetVector::default();
        let num_points: usize = 100;
        let provider = SimpleProvider {
            items: (0..num_points).map(|i| i as f32).collect(),
        };

        let mut s = HashMap::new();
        let mut accessor = Retriever::new(&provider, &ctx, true);

        assert_eq!(ctx.count(), 0, "default count should be 0");
        accessor
            .fill_set(&mut s, [0, 1, 2, 3].into_iter())
            .await
            .unwrap();
        assert_eq!(s.len(), 4);
        assert_eq!(*s.get(&0).unwrap(), 0.0);
        assert_eq!(*s.get(&1).unwrap(), 1.0);
        assert_eq!(*s.get(&2).unwrap(), 2.0);
        assert_eq!(*s.get(&3).unwrap(), 3.0);

        assert_eq!(ctx.count(), 4, "expected 1 load per element");
        ctx.clear();

        accessor
            .fill_set(&mut s, [2, 3, 4, 5].into_iter())
            .await
            .unwrap();
        assert_eq!(s.len(), 6);
        assert_eq!(*s.get(&0).unwrap(), 0.0);
        assert_eq!(*s.get(&1).unwrap(), 1.0);
        assert_eq!(*s.get(&2).unwrap(), 2.0);
        assert_eq!(*s.get(&3).unwrap(), 3.0);
        assert_eq!(*s.get(&4).unwrap(), 4.0);
        assert_eq!(*s.get(&5).unwrap(), 5.0);

        assert_eq!(
            ctx.count(),
            2,
            "expected 1 load per element not already in the set"
        );

        // Check error propagation.
        let _: InvalidVectorId = accessor
            .fill_set(&mut s, [8, 9, num_points as u32].into_iter())
            .await
            .unwrap_err();
    }

    #[tokio::test]
    async fn test_default_fill_set_transient_error() {
        let ctx = CountGetVector::default();
        let num_points: usize = 100;
        let provider = SimpleProvider {
            items: (0..num_points).map(|i| i as f32).collect(),
        };

        let mut s = HashMap::new();
        let mut accessor = Retriever::new(&provider, &ctx, false);

        assert_eq!(ctx.count(), 0, "default count should be 0");
        accessor
            .fill_set(&mut s, [0, 1, 2, 3].into_iter())
            .await
            .unwrap();
        assert_eq!(s.len(), 4);
        assert_eq!(*s.get(&0).unwrap(), 0.0);
        assert_eq!(*s.get(&1).unwrap(), 1.0);
        assert_eq!(*s.get(&2).unwrap(), 2.0);
        assert_eq!(*s.get(&3).unwrap(), 3.0);

        assert_eq!(ctx.count(), 4, "expected 1 load per element");
        ctx.clear();

        // Transient errors should be recorded and asknowledged, but not stop processing.
        let err = accessor
            .fill_set(
                &mut s,
                [2, 3, num_points as u32, 4, num_points as u32 + 1, 5].into_iter(),
            )
            .await
            .unwrap_err();

        assert!(!err.unrecoverable);
        err.acknowledge("acknowledging to satisfy drop logic");

        assert_eq!(s.len(), 6);
        assert_eq!(*s.get(&0).unwrap(), 0.0);
        assert_eq!(*s.get(&1).unwrap(), 1.0);
        assert_eq!(*s.get(&2).unwrap(), 2.0);
        assert_eq!(*s.get(&3).unwrap(), 3.0);
        assert_eq!(*s.get(&4).unwrap(), 4.0);
        assert_eq!(*s.get(&5).unwrap(), 5.0);

        assert_eq!(
            ctx.count(),
            2,
            "expected 1 load per element not already in the set"
        );
    }
}
