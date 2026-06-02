/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Search
//!
//! The [`SearchAccessor`] is the primary trait for implementing graph search algorithms.
//! Graph search begins at [`starting_points`](SearchAccessor::starting_points) and performs
//! several rounds of "beam expansion" via [`expand_beam`](SearchAccessor::expand_beam).
//!
//! The [`SearchAccessor`] has several duties. It must be able to retrieve adjacency list
//! information from its underlying [`DataProvider`] and can compute distances between a
//! fixed query and all elements in adjacency lists. See the documentation of
//! [`expand_beam`](SearchAccessor::expand_beam) for a more detailed description of the
//! required algorithm.
//!
//! Accessors are constructed via [`SearchStrategy::search_accessor`] and
//! [`InsertStrategy::insert_search_accessor`] where they are provided with the query.
//!
//! # Strategies
//!
//! Strategies provide the "glue" between [`DataProvider`]s and index operations like search,
//! insertion, deletion, etc.
//!
//! They do this by tying together accessors, computers, and other operations required
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
//!   1. Create the [`SearchAccessor`] defined by the strategy. This is the object that will
//!      be used to retrieve data from the [`DataProvider`].
//!
//!   2. Run greedy-search using the accessor.
//!
//!   3. After search, post-processing is run, which enables operations like filtering
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
//!   The graph search portion works by constructing a [`SearchAccessor`] with the value
//!   that was just inserted.
//!
//!   This accessor is then used for search, followed by pruning.
//!
//! * [`PruneStrategy`]: The pruning strategy is largely straightforward, consisting of
//!   an accessor and a random access [`DistanceFunction`] for performing distance
//!   calculations on the retrieved elements.
//!
//!   One subtle aspect is the use of the [`workingset::Fill`] trait. For clients with
//!   expensive vector retrieval calls, we wish to only retrieve vector IDs once for pruning.
//!
//!   This is done by using a working set to store the elements retrieved by the
//!   `PruneAccessor`. To give implementers the ability to perform a hybrid prune
//!   (consisting of a mix of full-precision and quantized vectors), **without** the index
//!   algorithm being aware of these two levels, the trait [`workingset::Fill`] is used to
//!   delegate the responsibility of populating this cache to the [`DataProvider`].
//!
//! * [`InplaceDeleteStrategy`]: This follows the trend of defining accessors and related
//!   strategies. One difference for inplace-deletion is that we use an element accessed
//!   from the [`DataProvider`] itself for search. Therefore, methods like
//!   [`crate::index::diskann_async::DiskANNIndex::inplace_delete`] also require the
//!   [`InplaceDeleteStrategy`] to implement an appropriate [`SearchStrategy`].
//!
//!   Like insertion, this trait delegates pruning to a dedicated `PruneStrategy`.

use std::{future::Future, sync::Arc};

use diskann_utils::Reborrow;
use diskann_vector::DistanceFunction;
use futures_util::FutureExt;

use crate::{
    ANNError, ANNResult,
    error::StandardError,
    graph::{SearchOutputBuffer, workingset},
    neighbor::Neighbor,
    provider::{AsNeighborMut, BuildDistanceComputer, DataProvider, HasElementRef, HasId},
};

/// The main extension point for graph search.
///
/// Search is a best-first graph traversal beginning at a collection of start points.
/// Neighbors with lower distances are prioritized over those with higher distances.
///
/// [`Self::expand_beam`] is the mechanism by which the graph is explored, expanding several
/// vertices at a time for efficiency.
///
/// **Start Point Coherence**: The various methods regarding start points are expected to
/// be coherent with one another. This means they agree on the number and IDs of the start
/// points.
pub trait SearchAccessor: HasId + Send + Sync {
    /// Return the starting points for this search.
    fn starting_points(&self)
    -> impl std::future::Future<Output = ANNResult<Vec<Self::Id>>> + Send;

    /// Compute the distance to all start points, invoking `f` with all results.
    fn start_point_distances<F>(
        &mut self,
        f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send;

    /// A primitive routine used by graph search. This is purposely implemented as a
    /// coarse-grained operation to enable optimization opportunities by the backend.
    ///
    /// # Description
    ///
    /// For each `i` in `itr`, fetch the adjacency list `v_i` for `i`. For each `v_i`, then
    /// for each id `j` in `v_i`, compute the distance `d` to the data associated with `j`
    /// and invoke the closure `f` with `d` and `j`, provided `pred.eval_mut(j)` evaluates
    /// to `true`.
    ///
    /// No specification is made on the traversal order of `ids`, the computation order of
    /// the leaf elements, nor the order in which `pred` is evaluated.
    ///
    /// Additionally, there is **no** strict requirement that all `j` discovered during the
    /// traversal need to be processed, though "dropping" too many candidates silently will
    /// adversely affect the quality of traversal.
    ///
    /// ## Implementation Notes
    ///
    /// Implementations must observe the following:
    ///
    /// * If `pred.eval_mut()` returns `true` for an id `i`, then `on_neighbors` should be
    ///   invoked for that item. Algorithms **may** choose to skip invoking `on_neighbors`
    ///   in exceptional circumstances (e.g. a transient access error occurs), though doing
    ///   this too often will degrade search quality.
    ///
    ///   If an item `i` is already passed to `on_neighbors`, the implementation is not
    ///   obligated to provide it again, though it **may** do so if `pred.eval_mut()`
    ///   continues to return `true`.
    ///
    /// * If `pred.eval_mut()` returns `false` for an item, then `on_neighbors` must not be
    ///   invoked for that item.
    ///
    /// * `pred.eval()` and `pred.eval_mut()` may be invoked multiple times for the same
    ///   item `i`.
    ///
    /// ## Predicate Requirements
    ///
    /// Well behaved predicates must never return `true` (allow an id to be forwarded to
    /// `on_neighbors`) if it previously returned `false`. Implementations of `expand_beam`
    /// are allowed to assume this holds.
    ///
    /// Additionally, the callback `on_neighbors` and the predicate have to cooperate. If the
    /// callback requires unique items, the predicate must be structured such that `eval_mut`
    /// correctly filters out duplicates.
    ///
    /// Calling `eval_mut` may change the predicate's state for an item `i`. The following
    /// hold for any pair of calls on the same `i` with no intervening predicate operations:
    ///
    /// * `eval(i) == true` implies a subsequent `eval_mut(i) == true`.
    /// * `eval(i) == false` implies a subsequent `eval_mut(i) == false`.
    /// * `eval_mut(i) == false` implies a subsequent `eval(i) == false`.
    ///
    /// # Pseudo Code
    ///
    /// ```ignore
    /// for i in ids {
    ///     // Retrieve the adjacency list IDs for node `i`.
    ///     let neighbors = self.get_neighbors_for(i);
    ///
    ///     // Loop over the adjacency list IDs, skipping IDs according to `pred`.
    ///     //
    ///     // Using `eval_mut` allows the predicate to record this visit and potentially
    ///     // exclude it from future calls.
    ///     for neighbor in neighbors.filter(|i| pred.eval_mut(i)) {
    ///         // Accessors are provided the query upon construction and are responsible
    ///         // for computing distances.
    ///         let distance = self.compute_distance_to(neighbor);
    ///         on_neighbors(neighbor, distance);
    ///     }
    /// }
    /// ```
    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        pred: P,
        on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send;

    //////////////////////
    // Provided methods //
    //////////////////////

    /// Indicate that search should terminate as soon as possible.
    ///
    /// The provided implementation always returns `false`.
    fn terminate_early(&mut self) -> bool {
        false
    }

    /// Return a closure to evaluate whether or not an ID is associated with a start point.
    ///
    /// The closure returned by the provided implementation has complexity `O(1)` and takes
    /// `O(num_starting_points)` time to construct.
    fn is_not_start_point(
        &self,
    ) -> impl std::future::Future<
        Output = ANNResult<impl Fn(Self::Id) -> bool + Send + Sync + 'static>,
    > + Send {
        async move {
            let set: std::collections::HashSet<_> =
                self.starting_points().await?.into_iter().collect();

            Ok(move |id| !set.contains(&id))
        }
    }

    /// Return the number of starting points.
    fn num_starting_points(&self) -> impl std::future::Future<Output = ANNResult<usize>> + Send {
        self.starting_points()
            .map(|result: ANNResult<_>| result.map(|v: Vec<_>| v.len()))
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

/// A search strategy for query objects of type `T`.
///
/// This trait should be overloaded by data providers wishing to extend
/// (search)[`crate::graph::DiskANNIndex::search`].
pub trait SearchStrategy<'a, Provider, T>: Send + Sync
where
    Provider: DataProvider,
    T: 'a,
{
    /// An error that can occur when getting a search_accessor.
    type SearchAccessorError: StandardError;

    /// The concrete type of the accessor that is used to access `Self` during the greedy
    /// graph search. The query will be provided to the accessor exactly once during search
    /// to construct the query computer.
    type SearchAccessor: SearchAccessor<Id = Provider::InternalId>;

    /// Construct and return the search accessor.
    fn search_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        query: T,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError>;
}

/// Opt-in trait for strategies that have a default post-processor.
///
/// Strategies implementing this trait can be used with index-level search APIs such as
/// [`crate::index::diskann_async::DiskANNIndex::search`] and
/// [`crate::index::diskann_async::DiskANNIndex::search_with`] when no explicit
/// post-processor is specified. The search infrastructure will call
/// `default_post_processor()` to obtain the processor and invoke its
/// [`SearchPostProcess::post_process`] method.
pub trait DefaultPostProcessor<'a, Provider, T, O = <Provider as DataProvider>::InternalId>:
    SearchStrategy<'a, Provider, T>
where
    Provider: DataProvider,
    O: Send,
    T: 'a,
{
    /// The default post-processor type.
    type Processor: SearchPostProcess<Self::SearchAccessor, T, O> + Send + Sync;

    /// Create the default post-processor.
    fn default_post_processor(&'a self) -> Self::Processor;
}

/// Aggregate trait for strategies that support both search access and a default post-processor.
pub trait DefaultSearchStrategy<'a, Provider, T, O = <Provider as DataProvider>::InternalId>:
    SearchStrategy<'a, Provider, T> + DefaultPostProcessor<'a, Provider, T, O>
where
    Provider: DataProvider,
    O: Send,
    T: 'a,
{
}

impl<'a, S, Provider, T, O> DefaultSearchStrategy<'a, Provider, T, O> for S
where
    S: SearchStrategy<'a, Provider, T> + DefaultPostProcessor<'a, Provider, T, O>,
    Provider: DataProvider,
    O: Send,
    T: 'a,
{
}

/// Convenience macro for implementing [`DefaultPostProcessor`] when the processor
/// is a [`Default`]-constructible type.
///
/// # Example
///
/// ```ignore
/// impl DefaultPostProcessor<MyProvider, &[f32]> for MyStrategy {
///     default_post_processor!(CopyIds);
/// }
/// ```
#[macro_export]
macro_rules! default_post_processor {
    ($Processor:ty) => {
        type Processor = $Processor;
        fn default_post_processor(&self) -> Self::Processor {
            Default::default()
        }
    };
}

/// Perform post-processing on the results of search, storing the results in an output buffer.
///
/// Simple implementations include [`CopyIds`], which simply forwards the search results
/// directly into the output buffer.
pub trait SearchPostProcess<A, T, O = <A as HasId>::Id>
where
    A: HasId,
{
    type Error: StandardError;

    /// Populate `output` with the entries in `candidates`. Correct implementations must
    /// return the number of results copied into `output` on success.
    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: T,
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
    A: HasId,
{
    type Error = std::convert::Infallible;
    fn post_process<I, B>(
        &self,
        _accessor: &mut A,
        _query: T,
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
    A: HasId,
{
    /// A potentially modified version of the error yielded by the next state in the
    /// processing pipeline.
    type Error<NextError>: StandardError
    where
        NextError: StandardError;

    /// The accessor that will be passed to the next processing stage.
    type NextAccessor: HasId<Id = A::Id>;

    /// Perform any modification the `input`, `output`, `accessor`, or `computer` objects
    /// and invoke the [`SearchPostProcess`] routine `next` on stage.
    fn post_process_step<I, B, Next>(
        &self,
        next: &Next,
        accessor: &mut A,
        query: T,
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
    A: SearchAccessor,
    T: Copy + Send + Sync,
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
        query: T,
        candidates: I,
        output: &mut B,
    ) -> ANNResult<usize>
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized,
        Next: SearchPostProcess<A, T, O> + Sync,
    {
        let filter = accessor.is_not_start_point().await?;
        next.post_process(accessor, query, candidates.filter(|n| filter(n.id)), output)
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
    A: HasId,
    Head: SearchPostProcessStep<A, T, O>,
    Tail: SearchPostProcess<Head::NextAccessor, T, O> + Sync,
{
    type Error = Head::Error<Tail::Error>;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: T,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized,
    {
        self.head
            .post_process_step(&self.tail, accessor, query, candidates, output)
    }
}

/// A strategy for inserting elements from the data provider.
///
/// This strategy is used during the greedy search portion of index construction.
/// After the candidate list has been retrieved from greedy search, the [`PruneStrategy`]
/// is used for the rest.
pub trait InsertStrategy<'a, Provider, T>: SearchStrategy<'a, Provider, T> + 'static
where
    Provider: DataProvider,
    T: 'a,
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
    fn insert_search_accessor(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        vector: T,
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        self.search_accessor(provider, context, vector)
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
    /// The [working set](crate::graph::workingset) used during pruning.
    ///
    /// For single insert this is typically an empty [`Map`](super::workingset::Map).
    /// For multi-insert it may be pre-seeded with batch elements.
    type WorkingSet: Send + Sync;

    /// The distance computer used during pruning.
    ///
    /// We could grab this type from the `PruneAccessor` associated type, but it's
    /// useful enough that we move it up here.
    type DistanceComputer<'computer>: for<'a, 'b, 'c, 'd> DistanceFunction<
            <Self::PruneAccessor<'a> as HasElementRef>::ElementRef<'b>,
            <Self::PruneAccessor<'c> as HasElementRef>::ElementRef<'d>,
            f32,
        > + Send
        + Sync;

    /// The concrete type of the accessor that is used to access `Self` during pruning.
    ///
    /// The accessor implements [`workingset::Fill`] for the strategy's
    /// [`WorkingSet`](Self::WorkingSet) type, which controls how elements are fetched and
    /// cached for distance computations.
    type PruneAccessor<'a>: HasId<Id = Provider::InternalId>
        + HasElementRef
        + BuildDistanceComputer<DistanceComputer = Self::DistanceComputer<'a>>
        + AsNeighborMut
        + workingset::Fill<Self::WorkingSet>
        + Send
        + Sync;

    /// An error that can occur when getting the prune accessor.
    type PruneAccessorError: StandardError;

    /// Create a fresh working set pre-sized for up to `capacity` elements.
    ///
    /// Argument `capacity` is an upper-bound: callers guarantee that no more than
    /// `capacity` elements will be inserted into the working set during a single
    /// [fill](workingset::Fill).
    ///
    /// Implementations may use this to pre-allocate or panic if exceeded.
    fn create_working_set(&self, capacity: usize) -> Self::WorkingSet;

    /// Return the prune accessor.
    fn prune_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError>;
}

/// Strategy for bulk insertion via [`multi_insert`](crate::graph::DiskANNIndex::multi_insert).
///
/// This trait delegates to its [`InsertStrategy`] for most selections during insert with
/// one difference, the [working set](crate::graph::workingset) provided to the insertion
/// [`PruneStrategy`] is seeded from [`Self::Seed`]. Seeding allows elements of the input
/// batch `B` to be part of the working set throughout prune, saving on vector retrievals.
pub trait MultiInsertStrategy<Provider, B>: Send + Sync
where
    Provider: DataProvider,
    B: Batch,
{
    /// The working set for the insertion [`PruneStrategy`].
    type WorkingSet: Send + Sync + 'static;

    /// The working set "seed", potentially containing `B` for faster access.
    type Seed: workingset::AsWorkingSet<Self::WorkingSet> + Send + Sync + 'static;

    /// Any critical error that occurs during [`finish`](Self::finish).
    type FinishError: Into<ANNError> + std::fmt::Debug + Send + Sync;

    /// The delegated [`InsertStrategy`] for most insertion related decisions.
    type InsertStrategy: for<'a> InsertStrategy<
            'a,
            Provider,
            B::Element<'a>,
            PruneStrategy: PruneStrategy<Provider, WorkingSet = Self::WorkingSet>,
        >;

    /// Construct the associated [`InsertStrategy`].
    fn insert_strategy(&self) -> Self::InsertStrategy;

    /// Package `batch` and the associated internal IDs into an intermediate seed.
    ///
    /// This seed type will be used to create the actual pruning working set on the various
    /// threads processing the batch insertion.
    ///
    /// Implementations may assume that `batch.len() == ids.len()`.
    fn finish<Itr>(
        &self,
        provider: &Provider,
        context: &Provider::Context,
        batch: &Arc<B>,
        ids: Itr,
    ) -> impl std::future::Future<Output = Result<Self::Seed, Self::FinishError>> + Send
    where
        Itr: ExactSizeIterator<Item = Provider::InternalId> + Send;
}

/// A batch of elements indexed positionally.
///
/// This provides random access to elements given to
/// [`multi_insert`](crate::graph::DiskANNIndex::multi_insert).
///
/// Elements are indexed in the range `[0, self.len())`.
///
/// See: [`MultiInsertStrategy`] for usage as well as
/// [`Overlay`](crate::graph::workingset::map::Overlay) for a working set seed compatible
/// with [`Batch`].
///
/// The primary implementation of this trait is [`Matrix`](diskann_utils::views::Matrix).
pub trait Batch: Send + Sync + 'static {
    /// The element type of the batch.
    type Element<'a>: Copy;

    /// The number of elements in the batch.
    fn len(&self) -> usize;

    /// Return the element at index `i`, where `i` should be in `[0, self.len())`.
    fn get(&self, i: usize) -> Self::Element<'_>;

    /// Return `true` if the batch is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Send + Sync + 'static> Batch for diskann_utils::views::Matrix<T> {
    type Element<'a> = &'a [T];

    fn len(&self) -> usize {
        self.nrows()
    }

    fn get(&self, i: usize) -> Self::Element<'_> {
        self.row(i)
    }
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
    /// With the by-value `T` parameter design, `DeleteElement<'a>` is expected to be a
    /// sized, `Copy` type (typically a reference like `&'a [f32]`).
    type DeleteElement<'a>: Copy + Send + Sync;

    /// The guard type returned by `get_delete_element`.
    ///
    /// The guard holds the retrieved element data and must be convertible to
    /// `DeleteElement<'a>` via [`Reborrow`]. This allows the guard's lifetime to scope the
    /// validity of the extracted element.
    type DeleteElementGuard: Send
        + Sync
        + for<'a> Reborrow<'a, Target = Self::DeleteElement<'a>>
        + 'static;

    /// Error type for accessing for search.
    type DeleteElementError: StandardError;

    /// The pruning strategy to use after the initial search is complete.
    type PruneStrategy: PruneStrategy<Provider>;

    /// The accessor used during the delete-search phase.
    ///
    /// This is technically redundant information as in theory, we could project through
    /// [`Self::SearchStrategy`]. However, when trying to write generic wrappers (read,
    /// the "caching" provider), rustc is unable to project all the way through the layers
    /// of associated types.
    ///
    /// Lifting the accessor all the way to the trait level makes the caching provider possible.
    type DeleteSearchAccessor<'a>: SearchAccessor<Id = Provider::InternalId>;

    /// The processor used during the delete-search phase.
    type SearchPostProcessor: for<'a> SearchPostProcess<Self::DeleteSearchAccessor<'a>, Self::DeleteElement<'a>>
        + Send
        + Sync;

    /// The type of the search strategy to use for graph traversal.
    type SearchStrategy: for<'a> SearchStrategy<
            'a,
            Provider,
            Self::DeleteElement<'a>,
            SearchAccessor = Self::DeleteSearchAccessor<'a>,
        >;

    /// Construct the prune strategy object.
    fn prune_strategy(&self) -> Self::PruneStrategy;

    /// Construct the search strategy object.
    fn search_strategy(&self) -> Self::SearchStrategy;

    /// Construct the search post-processor for the delete-search phase.
    fn search_post_processor(&self) -> Self::SearchPostProcessor;

    /// Construct the accessor used to retrieve the item being deleted.
    fn get_delete_element<'a>(
        &'a self,
        provider: &'a Provider,
        context: &'a Provider::Context,
        id: Provider::InternalId,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send;
}
