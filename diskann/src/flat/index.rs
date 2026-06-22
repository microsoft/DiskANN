/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`FlatIndex`] — the index wrapper for a [`DataProvider`](crate::provider::DataProvider)
//! over which we do flat search.
use std::num::NonZeroUsize;

use diskann_utils::future::SendFuture;

use crate::{
    ANNResult,
    error::{ErrorExt, IntoANNResult},
    flat::{DistancesUnordered, SearchStrategy},
    graph::{SearchOutputBuffer, glue::SearchPostProcess},
    neighbor::{Neighbor, NeighborPriorityQueue},
    provider::DataProvider,
};

/// Statistics collected during a flat search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// The total number of distance computations performed during the scan.
    pub cmps: u32,

    /// The total number of results written to the output buffer.
    pub result_count: u32,
}

/// A thin wrapper around a [`DataProvider`] used for flat search.
#[derive(Debug)]
pub struct FlatIndex<P: DataProvider> {
    /// The backing provider.
    provider: P,
}

impl<P: DataProvider> FlatIndex<P> {
    /// Construct a new [`FlatIndex`] around `provider`.
    pub fn new(provider: P) -> Self {
        Self { provider }
    }

    /// Borrow the underlying provider.
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Brute-force k-nearest-neighbor flat search.
    ///
    /// Streams every element produced by the strategy's visitor through the query
    /// computer, keeps the best `k` candidates in a [`NeighborPriorityQueue`], then runs
    /// `processor` over the survivors to populate `output`.
    ///
    /// The post-processor [`SearchPostProcess::post_process`] outputs the number
    /// of results that survive, which is returned as `SearchStats::result_count`.
    pub fn knn_search<S, T, O, PP, OB>(
        &self,
        k: NonZeroUsize,
        strategy: &S,
        processor: PP,
        context: &P::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<SearchStats>>
    where
        S: SearchStrategy<P, T>,
        T: Copy + Send + Sync,
        O: Send,
        PP: for<'a> SearchPostProcess<S::Visitor<'a>, T, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut visitor = strategy
                .create_visitor(&self.provider, context)
                .into_ann_result()?;

            let computer = strategy.build_query_computer(query).into_ann_result()?;

            let k = k.get();
            let mut queue = NeighborPriorityQueue::new(k);
            let mut cmps: u32 = 0;

            visitor
                .distances_unordered(&computer, |id, dist| {
                    cmps += 1;
                    queue.insert(Neighbor::new(id, dist));
                })
                .await
                .escalate("flat scan must complete to produce correct k-NN results")?;

            let result_count = processor
                .post_process(&mut visitor, query, queue.iter().take(k), output)
                .await
                .into_ann_result()? as u32;

            Ok(SearchStats { cmps, result_count })
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use crate::flat::{
        FlatIndex,
        test::{
            harness::{CopyIdsOracle, EvenIdsOnlyOracle, KnnOracleRun, OracleProcessor},
            provider::{self as flat_provider},
        },
    };
    use crate::graph::test::synthetic::Grid;

    fn fixture(grid: Grid, size: usize) -> (FlatIndex<flat_provider::Provider>, usize) {
        let provider = flat_provider::Provider::grid(grid, size).unwrap();
        let len = provider.len();
        (FlatIndex::new(provider), len)
    }

    /// `knn_search` returns a `Send` future, and a shared `&FlatIndex` can serve
    /// many concurrent searches on a multi-threaded runtime, each producing the
    /// correct output independently.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn multithreaded_knn_search() {
        use std::sync::Arc;

        let (index, len) = fixture(Grid::Two, 4);
        let index = Arc::new(index);

        // Mix of corner, axis-aligned, and off-grid queries; k spans 1..=len.
        let cases: &[(&[f32], usize)] = &[
            (&[-1.0, -1.0], 1),
            (&[1.0, 1.0], len),
            (&[-1.0, 1.0], len / 2),
            (&[1.0, -1.0], len - 1),
            (&[0.0, 0.0], 3),
            (&[3.0, 3.0], len),
            (&[-2.0, 0.5], 2),
            (&[0.5, -0.5], len),
        ];

        /// Spawn every `(query, k)` case under `oracle` onto `set`.
        fn spawn_cases<O>(
            set: &mut tokio::task::JoinSet<(Vec<f32>, usize, KnnOracleRun)>,
            index: &Arc<FlatIndex<flat_provider::Provider>>,
            oracle: O,
            cases: &[(&[f32], usize)],
        ) where
            O: OracleProcessor + Copy + Send + Sync + 'static,
        {
            for (query, k) in cases {
                let index = Arc::clone(index);
                let query: Vec<f32> = query.to_vec();
                let k = *k;
                set.spawn(async move {
                    let outcome = KnnOracleRun::run(
                        &index,
                        &flat_provider::Strategy::new(index.provider().dim()),
                        &oracle,
                        &query,
                        k,
                    )
                    .await
                    .expect("knn_search failed");
                    (query, k, outcome)
                });
            }
        }

        let mut set = tokio::task::JoinSet::new();
        spawn_cases(&mut set, &index, CopyIdsOracle, cases);
        spawn_cases(&mut set, &index, EvenIdsOnlyOracle, cases);

        while let Some(joined) = set.join_next().await {
            let (query, k, outcome) = joined.expect("task panicked");
            assert_eq!(
                outcome.top_k, outcome.ground_truth,
                "query = {query:?}, k = {k}: output must match its oracle",
            );
            assert_eq!(outcome.stats.cmps as usize, len);
            assert_eq!(
                outcome.stats.result_count as usize,
                outcome.ground_truth.len(),
            );
        }
    }

    ////////////
    // Errors //
    ////////////

    /// A transient error from the visitor's scan must escalate up through `knn_search`.
    #[test]
    fn transient_scan_error() {
        // The flat scan touches every id, so any transient id is guaranteed to be hit.
        for transient_ids in [&[0u32][..], &[3][..], &[1, 2, 5][..]] {
            let strategy =
                flat_provider::Strategy::with_transient(2, transient_ids.iter().copied());
            let (index, _) = fixture(Grid::Two, 3);
            let err = KnnOracleRun::run_sync(&index, &strategy, &CopyIdsOracle, &[1.0, 0.0], 4)
                .expect_err("transient error during full scan must escalate");

            let msg = format!("{err}");
            assert!(
                transient_ids
                    .iter()
                    .any(|id| msg.contains(&format!("id {id}"))),
                "transients = {transient_ids:?}: expected error to name one of the \
                 transient ids, got: {msg}",
            );
        }
    }

    /// Run `knn_search` via the harness, assert it fails, and check the error
    /// message contains `expected_msg`.
    fn assert_search_error(strategy: &flat_provider::Strategy, query: &[f32], expected_msg: &str) {
        let (index, _) = fixture(Grid::Two, 3);
        let err = KnnOracleRun::run_sync(&index, strategy, &CopyIdsOracle, query, 4)
            .expect_err("expected knn_search to fail");

        let msg = format!("{err}");
        assert!(
            msg.contains(expected_msg),
            "expected error containing {expected_msg:?}, got: {msg}",
        );
    }

    #[test]
    fn strategy_constructor_errors() {
        // Strategy/provider expect dim=2, query has dim=3.
        assert_search_error(
            &flat_provider::Strategy::new(2),
            &[0.0, 0.0, 0.0],
            "dimension mismatch",
        );

        // Strategy expects dim=5, provider has dim=2.
        assert_search_error(
            &flat_provider::Strategy::new(5),
            &[0.0, 0.0],
            "dimension mismatch",
        );
    }
}
