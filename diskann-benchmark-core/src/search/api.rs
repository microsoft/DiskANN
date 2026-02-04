/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{any::Any, future::Future, num::NonZeroUsize, pin::Pin, sync::Arc};

use diskann::{ANNResult, graph, utils::async_tools};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_utils::{
    future::{AsyncFriendly, boxit},
    views::{self, Matrix},
};

use crate::{
    internal,
    search::{
        ResultIds,
        ids::{Bounded, IdAggregator, ResultIdsInner},
    },
};

/// Necessary behavior for Id aggregation. Used by [`Search::Id`].
///
/// This trait has a blanket implementation and thus needs not be implemented manually.
pub trait Id: Default + Clone + Send + Sync + 'static {}

impl<T> Id for T where T: Default + Clone + Send + Sync + 'static {}

/// Indicate whether the number of items returned from search are bounded by a fixed amount
/// or can grow to an unknown size.
#[derive(Debug, Clone, Copy)]
pub enum IdCount {
    /// The number of ids returned from search are known to be bounded.
    Fixed(NonZeroUsize),

    /// The number of ids returned from search is unknown or unbounded. A size hint can
    /// be provided that can potentially improve performance.
    Dynamic(Option<NonZeroUsize>),
}

/// The core search API for approximate nearest neighbor searches.
///
/// This uses a model where queries are stored internally and identified by their
/// index. Queries are numbered from `0` to `N-1` where `N = Search::num_queries()`
/// is the total number of queries.
///
/// This trait is used in conjunction with [`search`] and [`search_all`]. See the
/// documentation of those methods for more details.
pub trait Search: AsyncFriendly {
    /// The identifier for the type returned by search. These are canonically the
    /// unique IDs associated with indexed vectors.
    type Id: Id;

    /// Custom input search parameters.
    type Parameters: Clone + AsyncFriendly;

    /// Custom output parameters. This augments the standard metrics collected by
    /// [`search`] and allows implementation-specific data to be returned.
    type Output: AsyncFriendly;

    /// The number of queries that can be searched. The machinery in [`search`] and
    /// [`search_all`] will invoke [`Search::search`] for each index in `0..N` where
    /// `N` is the returned value of this method.
    fn num_queries(&self) -> usize;

    /// Provide a hint for the number of IDs returned for each query. This is used to
    /// optimize internal buffer allocations.
    fn id_count(&self, parameters: &Self::Parameters) -> IdCount;

    /// Perform a search for the query identified by `index` using `parameters`. The
    /// results must be written into `buffer`. Customized output is returned.
    fn search<O>(
        &self,
        parameters: &Self::Parameters,
        buffer: &mut O,
        index: usize,
    ) -> impl Future<Output = ANNResult<Self::Output>> + Send
    where
        O: graph::SearchOutputBuffer<Self::Id> + Send;
}

/// Aggregated results for a single invocation of [`search`]. This corresponds to a
/// potentially parallelized batch of queries.
///
/// # Note
///
/// In the documentation of the member functions, the term "querywise" describes that the
/// returned collection has an ordered correspondence with the original queries.
///
/// If the [`Search`] object that generated these results as `N` queries (as returned by
/// [`Search::num_queries`]), then for these returned container, entry `i` will correspond
/// to the `i`th query for `i` in `0..N`.
#[derive(Debug)]
pub struct SearchResults<I, T> {
    ids: ResultIds<I>,
    latencies: Vec<MicroSeconds>,
    output: Vec<T>,
    end_to_end_latency: MicroSeconds,
}

impl<I, T> SearchResults<I, T> {
    /// Return the number of queries in the batch.
    pub fn len(&self) -> usize {
        self.latencies.len()
    }

    /// Return `true` only if `self.len() == 0`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the wall clock time taken to process all queries in the batch.
    pub fn end_to_end_latency(&self) -> MicroSeconds {
        self.end_to_end_latency
    }

    /// Return the querywise computed IDs from search.
    pub fn ids(&self) -> &ResultIds<I> {
        &self.ids
    }

    /// Return the querywise latencies for each search. If [`Self::latencies_mut`] has been
    /// called, the return slice loses its querywise guarantee.
    pub fn latencies(&self) -> &[MicroSeconds] {
        &self.latencies
    }

    /// Return the querywise latencies for each search by mutable reference. This is for
    /// efficient use of [`diskann_benchmark_runner::utils::percentiles::compute_percentiles`].
    ///
    /// Modifying the underlying slice invalidates the querywise guarantee.
    pub fn latencies_mut(&mut self) -> &mut [MicroSeconds] {
        &mut self.latencies
    }

    /// Return the querywise customized outputs from search.
    pub fn output(&self) -> &[T] {
        &self.output
    }

    /// Consume `self`, returning the querywise customized outputs from search by value.
    pub fn take_output(self) -> Vec<T> {
        self.output
    }
}

impl<I, T> SearchResults<I, T>
where
    I: Clone + Default,
    T: Any,
{
    fn new(batch: BatchResultsInner<I>) -> Self
    where
        I: Clone + Default,
        T: Any,
    {
        // The idea here is that we use `Collector` and dynamic dispatch for the output
        // aggregation to avoid monomorphising the collection algorithm for all output
        // types `T`.
        let mut output = Vec::<T>::new();
        let mut f = |any: Box<dyn Any>| match any.downcast::<Vec<T>>() {
            Ok(outputs) => output.extend(*outputs),
            Err(_) => panic!("Bad `Any` cast during aggregation"),
        };

        let Collector {
            ids,
            latencies,
            end_to_end_latency,
        } = Collector::collect(batch, &mut f);

        Self {
            ids,
            latencies,
            output,
            end_to_end_latency,
        }
    }
}

#[derive(Debug)]
struct Collector<I> {
    ids: ResultIds<I>,
    latencies: Vec<MicroSeconds>,
    end_to_end_latency: MicroSeconds,
}

impl<I> Collector<I>
where
    I: Clone + Default,
{
    fn collect(batch: BatchResultsInner<I>, collect_any: &mut dyn FnMut(Box<dyn Any>)) -> Self {
        let mut aggregator = IdAggregator::new();
        let mut latencies = Vec::new();

        batch.task_results.into_iter().for_each(|results| {
            aggregator.push(results.ids);
            latencies.extend_from_slice(&results.latencies);
            (collect_any)(results.outputs);
        });

        Self {
            ids: aggregator.finish(),
            latencies,
            end_to_end_latency: batch.end_to_end_latency,
        }
    }
}

/// Perform a search using the provided [`Search`] object. Argument `parameters` will be
/// provided to each invocation of [`Search::search`]. The search will be parallelized into
/// `ntasks` tasks using the provided `runtime`.
///
/// The returned results will have querywise correspondence with the original queries as
/// described in the documentation of [`SearchResults`].
pub fn search<S>(
    search: Arc<S>,
    parameters: S::Parameters,
    ntasks: NonZeroUsize,
    runtime: &tokio::runtime::Runtime,
) -> anyhow::Result<SearchResults<S::Id, S::Output>>
where
    S: Search,
{
    let results = runtime.block_on(search_inner::<S::Id>(search, Arc::new(parameters), ntasks))?;
    Ok(SearchResults::new(results))
}

/// An extension of [`search`] that allows multiple runs with different parameters with
/// automatic result aggregation.
///
/// The elements of `parameters` will be executed sequentially. The element yielded from `parameters`
/// is of type [`Run`], which encapsulates both the search parameters and setup information
/// such as the number of tasks and repetitions. The returned vector will have the same length as
/// the `parameters` iterator, with each entry corresponding to the aggregated results
/// for the respective run.
///
/// The aggregation behavior is defined by `aggregator` using the [`Aggregate`] trait.
/// [`Aggregate::aggregate`] will be provided with the raw results of all repetitions of
/// a single result from `parameters`.
///
/// # Notes on Repetitions
///
/// Each run will be repeated `R` times where `R` is defined by [`Run::setup`]. Callers are
/// encouraged to use multiple repetitions to obtain more stable performance metrics. Result
/// aggregation can summarize the results across a repetition group to reduce memory consumption.
pub fn search_all<S, Itr, A>(
    object: Arc<S>,
    parameters: Itr,
    mut aggregator: A,
) -> anyhow::Result<Vec<A::Output>>
where
    S: Search,
    Itr: IntoIterator<Item = Run<S::Parameters>>,
    A: Aggregate<S::Parameters, S::Id, S::Output>,
{
    let mut output = Vec::new();
    for run in parameters {
        let runtime = crate::tokio::runtime(run.setup().threads.into())?;

        let reps: usize = run.setup().reps.into();
        let raw = (0..reps)
            .map(|_| -> anyhow::Result<_> {
                search(
                    object.clone(),
                    run.parameters().clone(),
                    run.setup().tasks,
                    &runtime,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        output.push(aggregator.aggregate(run, raw)?);
    }

    Ok(output)
}

/// High level parameters for configuring a search run using [`search_all`].
#[derive(Debug, Clone, PartialEq)]
pub struct Setup {
    /// The number of threads to spawn in the [`tokio::runtime::Runtime`].
    pub threads: NonZeroUsize,

    /// The number of search tasks into which the search will be parallelized.
    /// This is intentionally decoupled from `threads` to allow for oversubscription
    /// of truly asynchronous providers.
    pub tasks: NonZeroUsize,

    /// The number of repetitions of the search to perform.
    pub reps: NonZeroUsize,
}

/// A single run of search containing a [`Setup`] and [`Search::Parameters`].
#[derive(Debug)]
pub struct Run<P> {
    parameters: P,
    setup: Setup,
}

impl<P> Run<P> {
    /// Construct a new [`Run`] around the search parameters and setup.
    pub fn new(parameters: P, setup: Setup) -> Self {
        Self { parameters, setup }
    }

    /// Return a reference to the contained search parameters.
    pub fn parameters(&self) -> &P {
        &self.parameters
    }

    /// Return a reference to the contained setup.
    pub fn setup(&self) -> &Setup {
        &self.setup
    }
}

/// Aggregate search results from multiple repetitions of a single run in [`search_all`].
///
/// # Type Parameters
/// - `P`: The type of [`Search::Parameters`].
/// - `I`: The type of [`Search::Id`].
/// - `O`: The type of [`Search::Output`].
pub trait Aggregate<P, I, O> {
    /// The type of the aggregated result.
    type Output;

    /// Aggregate the `results` for all repetitions of `run`.
    ///
    /// The length of `results` is guaranteed to be equal to [`Run::setup().reps`](Setup::reps).
    fn aggregate(
        &mut self,
        run: Run<P>,
        results: Vec<SearchResults<I, O>>,
    ) -> anyhow::Result<Self::Output>;
}

///////////
// Inner //
///////////

/// The inner search method is only parameterized by the ID type to minimize monomorphization.
///
/// The dynamic type of `parameters` must be the same as `Search::Parameters` for the
/// concrete type of `search`.
fn search_inner<I>(
    search: Arc<dyn SearchInner<Id = I>>,
    parameters: Arc<dyn Any + Send + Sync>,
    ntasks: NonZeroUsize,
) -> impl Future<Output = anyhow::Result<BatchResultsInner<I>>> + Send
where
    I: Id,
{
    let fut = async move {
        let start = std::time::Instant::now();
        let handles: Vec<_> = async_tools::PartitionIter::new(search.num_queries(), ntasks)
            .map(|range| {
                let search_clone = search.clone();
                let parameters_clone = parameters.clone();
                tokio::spawn(
                    async move { search_clone.search_batch(&*parameters_clone, range).await },
                )
            })
            .collect();

        let mut task_results = Vec::with_capacity(ntasks.into());
        for h in handles {
            task_results.push(h.await??);
        }

        let end_to_end_latency: MicroSeconds = start.elapsed().into();

        Ok(BatchResultsInner {
            end_to_end_latency,
            task_results,
        })
    };

    boxit(fut)
}

#[derive(Debug)]
struct BatchResultsInner<I> {
    end_to_end_latency: MicroSeconds,
    task_results: Vec<SearchResultsInner<I>>,
}

/// Note: Maintain the invariant that the number of entries in all fields is the same. That
/// is, this is something approximating an array of structs with special handling for the
/// result ids.
#[derive(Debug)]
struct SearchResultsInner<I> {
    ids: ResultIdsInner<I>,
    latencies: Vec<MicroSeconds>,

    // Result belonging strictly to the device under test. The concrete type is guaranteed
    // to be `Vec<Search::Output>`.
    outputs: Box<dyn Any + Send>,
}

impl<I> SearchResultsInner<I> {
    /// A custom constructor for `SearchResultsInner` that ensures the dynamic type of the outputs.
    fn new<T>(ids: ResultIdsInner<I>, latencies: Vec<MicroSeconds>, outputs: Vec<T::Output>) -> Self
    where
        T: Search<Id = I>,
    {
        Self {
            ids,
            latencies,
            outputs: Box::new(outputs),
        }
    }
}

// General boxed futures need to be Pinned to be pollable.
type Pinned<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

trait SearchInner: AsyncFriendly {
    type Id: Id;

    fn num_queries(&self) -> usize;

    fn search_batch<'a>(
        &'a self,
        parameters: &'a dyn Any,
        range: std::ops::Range<usize>,
    ) -> Pinned<'a, ANNResult<SearchResultsInner<Self::Id>>>;
}

impl<T> SearchInner for T
where
    T: Search,
{
    type Id = <T as Search>::Id;

    fn num_queries(&self) -> usize {
        <T as Search>::num_queries(self)
    }

    fn search_batch<'a>(
        &'a self,
        parameters: &'a dyn Any,
        range: std::ops::Range<usize>,
    ) -> Pinned<'a, ANNResult<SearchResultsInner<Self::Id>>> {
        let parameters = parameters
            .downcast_ref::<T::Parameters>()
            .expect("the internal search API should always pass the correct dynamic type");

        match self.id_count(parameters) {
            IdCount::Fixed(num_ids) => boxit(search_batch_fixed(self, range, parameters, num_ids)),
            IdCount::Dynamic(hint) => boxit(search_batch_dynamic(self, range, parameters, hint)),
        }
    }
}

async fn search_batch_fixed<T>(
    search: &T,
    range: std::ops::Range<usize>,
    parameters: &T::Parameters,
    num_ids: NonZeroUsize,
) -> ANNResult<SearchResultsInner<T::Id>>
where
    T: Search,
{
    let mut lengths = Vec::with_capacity(range.len());
    let mut ids = Matrix::new(views::Init(T::Id::default), range.len(), num_ids.into());

    let mut latencies = Vec::<MicroSeconds>::with_capacity(range.len());
    let mut outputs = Vec::<T::Output>::with_capacity(range.len());

    for (ids, index) in std::iter::zip(ids.row_iter_mut(), range) {
        let mut buffer = internal::buffer::Buffer::slice(ids);

        let start = std::time::Instant::now();
        let output = search.search(parameters, &mut buffer, index).await?;
        lengths.push(buffer.current_len());

        latencies.push(start.elapsed().into());
        outputs.push(output);
    }

    Ok(SearchResultsInner::new::<T>(
        ResultIdsInner::Fixed(Bounded::new(ids, lengths)),
        latencies,
        outputs,
    ))
}

async fn search_batch_dynamic<T>(
    search: &T,
    range: std::ops::Range<usize>,
    parameters: &T::Parameters,
    hint: Option<NonZeroUsize>,
) -> ANNResult<SearchResultsInner<T::Id>>
where
    T: Search,
{
    let mut ids = Vec::with_capacity(range.len());
    let mut latencies = Vec::<MicroSeconds>::with_capacity(range.len());
    let mut outputs = Vec::<T::Output>::with_capacity(range.len());

    let hint = hint.map(|i| i.into()).unwrap_or(0);

    for index in range {
        let mut these_ids = Vec::with_capacity(hint);
        let mut buffer = internal::buffer::Buffer::vector(&mut these_ids);

        let start = std::time::Instant::now();
        let output = search.search(parameters, &mut buffer, index).await?;
        latencies.push(start.elapsed().into());

        ids.push(these_ids);
        outputs.push(output);
    }

    Ok(SearchResultsInner::new::<T>(
        ResultIdsInner::Dynamic(ids),
        latencies,
        outputs,
    ))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::hash::{self, Hash, Hasher};

    // We intentionally do not derive `Clone` to ensure that it is not needed
    // in the implementations.
    #[derive(Debug)]
    struct TestSearch {
        queries: usize,
        // A hash function to determine the number and value of returned IDs.
        hasher: fn(usize, usize) -> usize,
    }

    impl TestSearch {
        fn count(&self, index: usize, id_count: &IdCount) -> usize {
            match id_count {
                IdCount::Fixed(n) => (self.hasher)(index, index) % n.get(),
                IdCount::Dynamic(_) => (self.hasher)(index, index) % DYNAMIC_MAX,
            }
        }

        fn format(&self, index: usize, position: usize) -> String {
            (self.hasher)(index, position).to_string()
        }

        fn check(&self, id_count: &IdCount, mut results: SearchResults<String, usize>) {
            let num_queries = self.queries;

            // End-to-end latency should not be zero.
            assert_ne!(
                results.end_to_end_latency().as_seconds(),
                0.0,
                "end to end latency should be non-zero"
            );

            assert_eq!(results.latencies().len(), num_queries);
            assert_eq!(results.latencies_mut().len(), num_queries);

            let rows = results.ids().as_rows();
            assert_eq!(rows.nrows(), num_queries);
            for i in 0..num_queries {
                let row = rows.row(i);
                assert_eq!(
                    row.len(),
                    self.count(i, id_count),
                    "incorrect length for output row {}",
                    i
                );

                for (j, id) in row.iter().enumerate() {
                    assert_eq!(
                        id,
                        &self.format(i, j),
                        "mismatch for query {} at position {}",
                        i,
                        j
                    );
                }
            }

            let expected_output: Vec<_> =
                (0..num_queries).map(|i| self.count(i, id_count)).collect();

            assert_eq!(results.output(), &expected_output);

            let output = results.take_output();
            assert_eq!(output, expected_output);
        }
    }

    const DYNAMIC_MAX: usize = 5;

    impl Search for TestSearch {
        type Id = String;
        type Parameters = IdCount;
        type Output = usize;

        fn num_queries(&self) -> usize {
            self.queries
        }

        fn id_count(&self, parameters: &IdCount) -> IdCount {
            *parameters
        }

        async fn search<O>(
            &self,
            params: &IdCount,
            buffer: &mut O,
            index: usize,
        ) -> ANNResult<Self::Output>
        where
            O: graph::SearchOutputBuffer<Self::Id> + Send,
        {
            let count = self.count(index, params);
            let set = buffer.extend((0..count).map(|i| (self.format(index, i), i as f32)));
            assert_eq!(set, count);
            Ok(count)
        }
    }

    fn hash(a: usize, b: usize) -> usize {
        let mut hasher = hash::DefaultHasher::new();
        a.hash(&mut hasher);
        b.hash(&mut hasher);
        hasher.finish() as usize
    }

    // This test sweeps across a wide variety of threads, tasks, and behavior.
    //
    // We use hashing to generate deterministic but non-uniform results.
    #[test]
    fn test_search() {
        for num_queries in [3, 4, 5] {
            let searcher = Arc::new(TestSearch {
                queries: num_queries,
                hasher: hash,
            });

            for num_threads in 1..6 {
                let runtime = crate::tokio::runtime(num_threads).unwrap();

                for num_tasks in 1..6 {
                    let num_tasks = NonZeroUsize::new(num_tasks).unwrap();
                    for id_count in [
                        IdCount::Fixed(NonZeroUsize::new(3).unwrap()),
                        IdCount::Dynamic(Some(NonZeroUsize::new(4).unwrap())),
                        IdCount::Dynamic(None),
                    ] {
                        let results =
                            search(searcher.clone(), id_count, num_tasks, &runtime).unwrap();

                        searcher.check(&id_count, results);
                    }
                }
            }
        }
    }

    /// An aggregator for testing [`search_all`]. This simply invokes [`TestSearch::check`]
    /// on the inner results, verifies the number of results, and
    struct Aggregator<'a> {
        /// The searcher provided to [`search_all`].
        searcher: Arc<TestSearch>,

        /// A seed for randomizing the return values.
        seed: usize,

        /// A count for the number of times `aggregate` was called.
        called: &'a mut usize,
    }

    impl Aggregate<IdCount, String, usize> for Aggregator<'_> {
        type Output = usize;

        fn aggregate(
            &mut self,
            run: Run<IdCount>,
            results: Vec<SearchResults<String, usize>>,
        ) -> anyhow::Result<Self::Output> {
            assert_eq!(
                results.len(),
                run.setup().reps.get(),
                "the incorrect number of results was returned",
            );

            for result in results {
                self.searcher.check(run.parameters(), result);
            }

            let count = *self.called;
            *self.called += 1;
            Ok(hash(self.seed, count))
        }
    }

    #[test]
    fn test_search_all() {
        let counts = [
            IdCount::Fixed(NonZeroUsize::new(3).unwrap()),
            IdCount::Dynamic(Some(NonZeroUsize::new(4).unwrap())),
            IdCount::Dynamic(None),
        ];

        let seed = 0x2f1b462446d1f225;

        for num_queries in [3, 4, 5] {
            let searcher = Arc::new(TestSearch {
                queries: num_queries,
                hasher: hash,
            });

            let iter = itertools::iproduct!((1..6), (1..6), (2..3), counts,).map(
                |(threads, tasks, reps, parameters)| {
                    Run::new(
                        parameters,
                        Setup {
                            threads: NonZeroUsize::new(threads).unwrap(),
                            tasks: NonZeroUsize::new(tasks).unwrap(),
                            reps: NonZeroUsize::new(reps).unwrap(),
                        },
                    )
                },
            );

            let mut called = 0usize;
            let aggregator = Aggregator {
                searcher: searcher.clone(),
                seed,
                called: &mut called,
            };

            let len = iter.size_hint().0;

            let results = search_all(searcher, iter, aggregator).unwrap();

            assert_eq!(results.len(), len);
            assert_eq!(called, len);

            for (i, r) in results.into_iter().enumerate() {
                assert_eq!(r, hash(seed, i), "mismatch for result {}", i);
            }
        }
    }
}
