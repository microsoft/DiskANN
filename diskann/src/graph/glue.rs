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
//!   One subtle aspect is the use of the [`workingset::Fill`] trait. For clients with expensive
//!   vector retrieval calls, we wish to only retrieve vector IDs once for pruning.
//!
//!   This is done by using a working set to store the elements retrieved by the
//!   `PruneAccessor`. To give implementers the ability to perform a hybrid prune
//!   (consisting of a mix of full-precision and quantized vectors), **without** the index
//!   algorithm being aware of these two levels, the trait [`workingset::Fill`] is used to delegate
//!   the responsibility of populating this cache to the [`DataProvider`].
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
use diskann_utils::future::AssertSend;
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction};

use crate::{
    ANNError, ANNResult,
    error::{ErrorExt, StandardError},
    graph::{AdjacencyList, SearchOutputBuffer, workingset},
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
pub trait ExpandBeam<T>: BuildQueryComputer<T> + AsNeighbor + Sized {
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
/// (search)[`crate::graph::DiskANNIndex::search`].
pub trait SearchStrategy<Provider, T>: Send + Sync
where
    Provider: DataProvider,
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
}

/// Opt-in trait for strategies that have a default post-processor.
///
/// Strategies implementing this trait can be used with index-level search APIs such as
/// [`crate::index::diskann_async::DiskANNIndex::search`] and
/// [`crate::index::diskann_async::DiskANNIndex::search_with`] when no explicit
/// post-processor is specified. The search infrastructure will call
/// `default_post_processor()` to obtain the processor and invoke its
/// [`SearchPostProcess::post_process`] method.
pub trait DefaultPostProcessor<Provider, T, O = <Provider as DataProvider>::InternalId>:
    SearchStrategy<Provider, T>
where
    Provider: DataProvider,
    O: Send,
{
    /// The default post-processor type.
    type Processor: for<'a> SearchPostProcess<Self::SearchAccessor<'a>, T, O> + Send + Sync;

    /// Create the default post-processor.
    fn default_post_processor(&self) -> Self::Processor;
}

/// Aggregate trait for strategies that support both search access and a default post-processor.
pub trait DefaultSearchStrategy<Provider, T, O = <Provider as DataProvider>::InternalId>:
    SearchStrategy<Provider, T> + DefaultPostProcessor<Provider, T, O>
where
    Provider: DataProvider,
    O: Send,
{
}

impl<S, Provider, T, O> DefaultSearchStrategy<Provider, T, O> for S
where
    S: SearchStrategy<Provider, T> + DefaultPostProcessor<Provider, T, O>,
    Provider: DataProvider,
    O: Send,
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
    A: BuildQueryComputer<T>,
{
    type Error: StandardError;

    /// Populate `output` with the entries in `candidates`. Correct implementations must
    /// return the number of results copied into `output` on success.
    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: T,
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
{
    type Error = std::convert::Infallible;
    fn post_process<I, B>(
        &self,
        _accessor: &mut A,
        _query: T,
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
        query: T,
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
    Head: SearchPostProcessStep<A, T, O>,
    Tail: SearchPostProcess<Head::NextAccessor, T, O> + Sync,
{
    type Error = Head::Error<Tail::Error>;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: T,
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
    /// The [working set](crate::graph::workingset) used during pruning.
    ///
    /// For single insert this is typically an empty [`Map`](super::workingset::Map).
    /// For multi-insert it may be pre-seeded with batch elements.
    type WorkingSet: Send + Sync;

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
    /// The accessor implements [`workingset::Fill`] for the strategy's
    /// [`WorkingSet`](Self::WorkingSet) type, which controls how elements are fetched and
    /// cached for distance computations.
    ///
    /// Implementations are encouraged to have [`Accessor::get_element`] return the
    /// highest-precision applicable value for a given element type.
    type PruneAccessor<'a>: Accessor<Id = Provider::InternalId>
        + BuildDistanceComputer<DistanceComputer = Self::DistanceComputer>
        + AsNeighborMut
        + workingset::Fill<Self::WorkingSet>;

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
    type DeleteSearchAccessor<'a>: ExpandBeam<Self::DeleteElement<'a>, Id = Provider::InternalId>
        + SearchExt;

    /// The processor used during the delete-search phase.
    type SearchPostProcessor: for<'a> SearchPostProcess<Self::DeleteSearchAccessor<'a>, Self::DeleteElement<'a>>
        + Send
        + Sync;

    /// The type of the search strategy to use for graph traversal.
    type SearchStrategy: for<'a> SearchStrategy<
            Provider,
            Self::DeleteElement<'a>,
            SearchAccessor<'a> = Self::DeleteSearchAccessor<'a>,
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
        type Guard = crate::provider::NoopGuard<u32>;

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
    }

    impl SearchExt for Retriever<'_> {
        async fn starting_points(&self) -> ANNResult<Vec<u32>> {
            Ok(vec![0])
        }
    }

    impl<'a> Retriever<'a> {
        fn new(provider: &'a SimpleProvider, count: &'a CountGetVector) -> Self {
            Self { provider, count }
        }
    }

    impl HasId for Retriever<'_> {
        type Id = u32;
    }

    impl Accessor for Retriever<'_> {
        type Element<'a>
            = f32
        where
            Self: 'a;
        type ElementRef<'a> = f32;

        type GetError = ANNError;
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
                None => panic!("invalid id: {}", id),
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

    struct QueryComputer;
    impl PreprocessedDistanceFunction<f32, f32> for QueryComputer {
        fn evaluate_similarity(&self, _changing: f32) -> f32 {
            panic!("this method should not be called")
        }
    }

    impl BuildQueryComputer<f32> for Retriever<'_> {
        type QueryComputerError = ANNError;
        type QueryComputer = QueryComputer;
        fn build_query_computer(&self, _from: f32) -> Result<QueryComputer, ANNError> {
            Ok(QueryComputer)
        }
    }

    impl ExpandBeam<f32> for Retriever<'_> {}

    // This strategy explicitly does not define `post_process` so we can test the provided
    // implementation.
    struct Strategy;

    impl SearchStrategy<SimpleProvider, f32> for Strategy {
        type QueryComputer = QueryComputer;
        type SearchAccessorError = ANNError;
        type SearchAccessor<'a> = Retriever<'a>;

        fn search_accessor<'a>(
            &'a self,
            provider: &'a SimpleProvider,
            context: &'a CountGetVector,
        ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
            Ok(Retriever::new(provider, context))
        }
    }

    impl DefaultPostProcessor<SimpleProvider, f32> for Strategy {
        default_post_processor!(CopyIds);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_default_post_process() {
        let ctx = CountGetVector::default();
        let strategy = Strategy;

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
        let computer = accessor.build_query_computer(query).unwrap();

        for input_len in 0..10 {
            let input: Vec<_> = (0..input_len)
                .map(|i| Neighbor::<u32>::new(i as u32, i as f32))
                .collect();
            for output_len in 0..10 {
                let mut output = vec![Neighbor::<u32>::default(); output_len];

                let count = strategy
                    .default_post_processor()
                    .post_process(
                        &mut accessor,
                        query,
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
}
