/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A synchronous, Rayon-based version of [`crate::search`].
//!
//! One large difference between this and the asynchronous API is the looser lifetime
//! requirements on implementations of [`Search`]. [`super::Search`] must be `'static` to be
//! passed between Tokio tasks whereas the synchronous API does not have this limitation.

use std::num::NonZeroUsize;

use diskann::{ANNResult, graph, utils::async_tools};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_utils::{
    future::AsyncFriendly,
    views::rowmajor::{self, MatrixMut},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    internal,
    search::{
        Aggregate, Id, IdCount, Run, SearchResults,
        api::{BatchResultsInner, SearchResultsInner},
        ids::{Bounded, ResultIdsInner},
    },
};

/// The core search API for approximate nearest neighbor searches.
///
/// This uses a model where queries are stored internally and identified by their
/// index. Queries are numbered from `0` to `N-1` where `N = Search::num_queries()`
/// is the total number of queries.
///
/// This trait is used in conjunction with [`search`] and [`search_all`]. See the
/// documentation of those methods for more details.
pub trait Search: Sync {
    /// The identifier for the type returned by search. These are canonically the
    /// unique IDs associated with indexed vectors.
    type Id: Id;

    /// Custom input search parameters.
    type Parameters: Sync + Clone;

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
    ) -> ANNResult<Self::Output>
    where
        O: graph::SearchOutputBuffer<Self::Id> + Send;
}

///////////////
// Interface //
///////////////

/// Perform a search using the provided [`Search`] object. Argument `parameters` will be
/// provided to each invocation of [`Search::search`]. The search will be parallelized into
/// `ntasks` tasks and run in the enclosing [`rayon::ThreadPool`]. Use
/// [`rayon::ThreadPool::install`] to select a pool, otherwise the global threadpool will be
/// used.
///
/// The returned results will have querywise correspondence with the original queries as
/// described in the documentation of [`SearchResults`].
///
/// # See Also
///
/// [`search_all`], [`search_all_with`].
pub fn search<S>(
    search: &S,
    parameters: &S::Parameters,
    ntasks: NonZeroUsize,
) -> anyhow::Result<SearchResults<S::Id, S::Output>>
where
    S: Search,
{
    let results = search_inner::<S::Id>(&Bundle::new(search, parameters), ntasks)?;
    Ok(SearchResults::new(results))
}

/// Perform [batch search](crate::search#batch-execution) on `object` with `parameters`.
/// Each search point is run sequentially in its own [`rayon::ThreadPool`].
///
/// # See Also
///
/// [`search`], [`search_all_with`].
pub fn search_all<S, Itr, A>(
    object: &S,
    parameters: Itr,
    aggregator: A,
) -> anyhow::Result<Vec<A::Output>>
where
    S: Search,
    Itr: IntoIterator<Item = Run<S::Parameters>>,
    A: Aggregate<S::Parameters, S::Id, S::Output>,
{
    search_all_with(object, parameters, aggregator, identity)
}

fn identity<T>(x: T) -> T {
    x
}

/// Perform [batch search](crate::search#batch-execution) on `object` with `parameters`.
/// Each search point is run sequentially in its own [`rayon::ThreadPool`].
///
/// Each [`rayon::ThreadPoolBuilder`] will be provided to `on_builder` before construction.
/// The thread count will already be initialized from the corresponding [`Run`].
///
/// # See Also
///
/// [`search`], [`search_all`].
pub fn search_all_with<S, Itr, A>(
    object: &S,
    parameters: Itr,
    mut aggregator: A,
    mut on_builder: impl FnMut(rayon::ThreadPoolBuilder) -> rayon::ThreadPoolBuilder,
) -> anyhow::Result<Vec<A::Output>>
where
    S: Search,
    Itr: IntoIterator<Item = Run<S::Parameters>>,
    A: Aggregate<S::Parameters, S::Id, S::Output>,
{
    let mut output = Vec::new();
    for run in parameters {
        let pool = crate::rayon::runtime_with(run.setup().threads, &mut on_builder)?;

        let reps: usize = run.setup().reps.into();
        let raw = (0..reps)
            .map(|_| -> anyhow::Result<_> {
                pool.install(|| search(object, run.parameters(), run.setup().tasks))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        output.push(aggregator.aggregate(run, raw)?);
    }

    Ok(output)
}

///////////
// Inner //
///////////

/// The inner search method is only parameterized by the ID type to minimize monomorphization.
#[expect(
    clippy::disallowed_methods,
    reason = "this is scoped behind a threadpool"
)]
fn search_inner<I>(
    search: &dyn SearchInner<Id = I>,
    ntasks: NonZeroUsize,
) -> anyhow::Result<BatchResultsInner<I>>
where
    I: Id,
{
    let start = std::time::Instant::now();
    let num_queries = search.num_queries();
    let task_results = (0..ntasks.get())
        .into_par_iter()
        .map(|task| -> anyhow::Result<_> {
            let range = async_tools::partition(num_queries, ntasks, task)?;
            Ok(search.search_batch(range)?)
        })
        .collect::<anyhow::Result<_>>()?;

    let end_to_end_latency: MicroSeconds = start.elapsed().into();
    Ok(BatchResultsInner::new(end_to_end_latency, task_results))
}

trait SearchInner: Sync {
    type Id: Id;

    fn num_queries(&self) -> usize;

    fn search_batch(
        &self,
        range: std::ops::Range<usize>,
    ) -> ANNResult<SearchResultsInner<Self::Id>>;
}

#[derive(Debug)]
struct Bundle<'a, T>
where
    T: Search,
{
    search: &'a T,
    parameters: &'a T::Parameters,
}

impl<'a, T> Bundle<'a, T>
where
    T: Search,
{
    fn new(search: &'a T, parameters: &'a T::Parameters) -> Self {
        Self { search, parameters }
    }

    fn search_batch_fixed(
        &self,
        range: std::ops::Range<usize>,
        num_ids: NonZeroUsize,
    ) -> ANNResult<SearchResultsInner<T::Id>> {
        let mut lengths = Vec::with_capacity(range.len());
        let mut ids =
            rowmajor::Owned::try_from_fn(range.len(), num_ids.into(), |_| T::Id::default())?;

        let mut latencies = Vec::<MicroSeconds>::with_capacity(range.len());
        let mut outputs = Vec::<T::Output>::with_capacity(range.len());

        for (ids, index) in std::iter::zip(ids.rows_mut(), range) {
            let mut buffer = internal::buffer::Buffer::slice(ids);

            let start = std::time::Instant::now();
            let output = self.search.search(self.parameters, &mut buffer, index)?;
            lengths.push(buffer.current_len());

            latencies.push(start.elapsed().into());
            outputs.push(output);
        }

        Ok(SearchResultsInner::new_sync::<T>(
            ResultIdsInner::Fixed(Bounded::new(ids, lengths)),
            latencies,
            outputs,
        ))
    }

    fn search_batch_dynamic(
        &self,
        range: std::ops::Range<usize>,
        hint: Option<NonZeroUsize>,
    ) -> ANNResult<SearchResultsInner<T::Id>> {
        let mut ids = Vec::with_capacity(range.len());
        let mut latencies = Vec::<MicroSeconds>::with_capacity(range.len());
        let mut outputs = Vec::<T::Output>::with_capacity(range.len());

        let hint = hint.map(|i| i.into()).unwrap_or(0);

        for index in range {
            let mut these_ids = Vec::with_capacity(hint);
            let mut buffer = internal::buffer::Buffer::vector(&mut these_ids);

            let start = std::time::Instant::now();
            let output = self.search.search(self.parameters, &mut buffer, index)?;
            latencies.push(start.elapsed().into());

            ids.push(these_ids);
            outputs.push(output);
        }

        Ok(SearchResultsInner::new_sync::<T>(
            ResultIdsInner::Dynamic(ids),
            latencies,
            outputs,
        ))
    }
}

impl<S> SearchInner for Bundle<'_, S>
where
    S: Search,
{
    type Id = S::Id;

    fn num_queries(&self) -> usize {
        self.search.num_queries()
    }

    fn search_batch(
        &self,
        range: std::ops::Range<usize>,
    ) -> ANNResult<SearchResultsInner<Self::Id>> {
        match self.search.id_count(self.parameters) {
            IdCount::Fixed(num_ids) => self.search_batch_fixed(range, num_ids),
            IdCount::Dynamic(hint) => self.search_batch_dynamic(range, hint),
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        marker::PhantomData,
        sync::{
            MutexGuard,
            atomic::{AtomicUsize, Ordering::Relaxed},
        },
    };

    use crate::search::{
        Setup,
        api::tests::{Aggregator, IdHash, TestSearch},
    };

    #[derive(Debug)]
    struct SyncNotSend<'a> {
        calls: &'a AtomicUsize,
        _not_send: PhantomData<MutexGuard<'a, ()>>,
    }

    impl<'a> SyncNotSend<'a> {
        fn new(calls: &'a AtomicUsize) -> Self {
            Self {
                calls,
                _not_send: PhantomData,
            }
        }
    }

    impl IdHash for SyncNotSend<'_> {
        fn id_hash(&self, a: usize, b: usize) -> usize {
            self.calls.fetch_add(1, Relaxed);
            crate::search::api::tests::hash(a, b)
        }
    }

    impl Search for TestSearch<SyncNotSend<'_>> {
        type Id = String;
        type Parameters = IdCount;
        type Output = usize;

        fn num_queries(&self) -> usize {
            <TestSearch<SyncNotSend<'_>>>::num_queries(self)
        }

        fn id_count(&self, parameters: &IdCount) -> IdCount {
            *parameters
        }

        fn search<O>(
            &self,
            params: &IdCount,
            buffer: &mut O,
            index: usize,
        ) -> ANNResult<Self::Output>
        where
            O: graph::SearchOutputBuffer<Self::Id> + Send,
        {
            Ok(self.search_inner(params, buffer, index))
        }
    }

    // This test sweeps across a wide variety of threads, tasks, and behavior.
    //
    // We use hashing to generate deterministic but non-uniform results.
    #[test]
    fn test_search() {
        let calls = AtomicUsize::new(0);
        for num_queries in [3, 4, 5] {
            let searcher = TestSearch::new(num_queries, SyncNotSend::new(&calls));

            for num_threads in 1..6 {
                let pool = crate::rayon::runtime(NonZeroUsize::new(num_threads).unwrap()).unwrap();

                for num_tasks in 1..6 {
                    let num_tasks = NonZeroUsize::new(num_tasks).unwrap();
                    for id_count in [
                        IdCount::Fixed(NonZeroUsize::new(3).unwrap()),
                        IdCount::Dynamic(Some(NonZeroUsize::new(4).unwrap())),
                        IdCount::Dynamic(None),
                    ] {
                        let results = pool
                            .install(|| search(&searcher, &id_count, num_tasks))
                            .unwrap();

                        searcher.check(&id_count, results);
                    }
                }
            }
        }
        assert_ne!(calls.load(Relaxed), 0);
    }

    #[test]
    fn test_search_all() {
        let counts = [
            IdCount::Fixed(NonZeroUsize::new(3).unwrap()),
            IdCount::Dynamic(Some(NonZeroUsize::new(4).unwrap())),
            IdCount::Dynamic(None),
        ];

        let seed = 0x2f1b462446d1f225;
        let calls = AtomicUsize::new(0);

        for num_queries in [3, 4, 5] {
            let searcher = TestSearch::new(num_queries, SyncNotSend::new(&calls));

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

            // `search_all`
            {
                let mut called = 0usize;
                let aggregator = Aggregator::new(&searcher, seed, &mut called);

                let len = iter.size_hint().0;
                let results = search_all(&searcher, iter.clone(), aggregator).unwrap();

                assert_eq!(results.len(), len);
                assert_eq!(called, len);

                for (i, r) in results.into_iter().enumerate() {
                    assert_eq!(
                        r,
                        crate::search::api::tests::hash(seed, i),
                        "mismatch for result {}",
                        i
                    );
                }
            }

            // `search_all_with`
            {
                let mut called = 0usize;
                let aggregator = Aggregator::new(&searcher, seed, &mut called);

                let len = iter.size_hint().0;
                let mut builder_calls = 0usize;
                let results = search_all_with(&searcher, iter, aggregator, |b| {
                    builder_calls += 1;
                    b
                })
                .unwrap();

                assert_eq!(results.len(), len);
                assert_eq!(called, len);
                assert_eq!(builder_calls, len);

                for (i, r) in results.into_iter().enumerate() {
                    assert_eq!(
                        r,
                        crate::search::api::tests::hash(seed, i),
                        "mismatch for result {}",
                        i
                    );
                }
            }
        }
        assert_ne!(calls.load(Relaxed), 0);
    }
}
