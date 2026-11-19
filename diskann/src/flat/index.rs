/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Brute-force k-nearest-neighbor search over a [`DistancesUnordered`] visitor.
use std::num::NonZeroUsize;

use diskann_utils::future::SendFuture;

use crate::{
    ANNResult,
    error::IntoANNResult,
    flat::{DistancesUnordered, topk},
    graph::{SearchOutputBuffer, glue::SearchPostProcess},
};

/// Statistics collected during a flat search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// The total number of distance computations performed during the scan.
    pub cmps: u32,

    /// The total number of results written to the output buffer.
    pub result_count: u32,
}

/// Tuning for flat-search candidate collection. Options are local to each search.
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    /// Use a reservoir when `k >= reservoir_min_k`, otherwise use a heap.
    /// `k == 1` always uses a dedicated single-best collector.
    ///
    /// The default, 100, follows Faiss's `distance_compute_min_k_reservoir`; it
    /// is a starting point for tuning, not a workload-independent optimum.
    /// Set this to 0 to force the reservoir for `k > 1`, or above the requested
    /// `k` to force the heap. This changes performance, not result ordering.
    pub reservoir_min_k: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            reservoir_min_k: 100,
        }
    }
}

/// Brute-force k-nearest-neighbor search over an initialized visitor.
///
/// Borrows `visitor` for the duration of the search, streams every distance it produces,
/// keeps the best `k` candidates, then runs `processor` over the same visitor and
/// the surviving candidates in ascending distance order to populate `output`.
/// NaN distances are ignored. Equal distances retain the most recently encountered
/// candidates first, including when choosing which candidates survive the top-k boundary.
///
/// The visitor remains owned by the caller and becomes available again after the returned
/// future completes. Whether it supports another scan is determined by its implementation.
///
/// Uses [`SearchOptions::default`]. See [`knn_search_with_options`] to tune the
/// heap/reservoir crossover for a particular workload.
///
/// # Errors
///
/// Returns an error if distance scanning or result post-processing fails. Distance-scan
/// errors are escalated because a partial flat scan cannot produce correct k-nearest-neighbor
/// results.
pub fn knn_search<V, T, O, PP, OB>(
    visitor: &mut V,
    k: NonZeroUsize,
    processor: PP,
    query: T,
    output: &mut OB,
) -> impl SendFuture<ANNResult<SearchStats>>
where
    V: DistancesUnordered,
    T: Copy + Send + Sync,
    O: Send,
    PP: SearchPostProcess<V, T, O> + Send + Sync,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
{
    knn_search_with_options(
        visitor,
        k,
        SearchOptions::default(),
        processor,
        query,
        output,
    )
}

/// Run [`knn_search`] with a per-search collector threshold.
///
/// Candidate ordering, post-processing and statistics follow [`knn_search`].
///
/// # Errors
///
/// Returns an error if distance scanning or result post-processing fails.
/// Scan errors are escalated because partial scans cannot produce correct top-k results.
pub fn knn_search_with_options<V, T, O, PP, OB>(
    visitor: &mut V,
    k: NonZeroUsize,
    options: SearchOptions,
    processor: PP,
    query: T,
    output: &mut OB,
) -> impl SendFuture<ANNResult<SearchStats>>
where
    V: DistancesUnordered,
    T: Copy + Send + Sync,
    O: Send,
    PP: SearchPostProcess<V, T, O> + Send + Sync,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
{
    async move {
        let (candidates, cmps) = topk::collect(visitor, k, options.reservoir_min_k).await?;

        let result_count = processor
            .post_process(visitor, query, candidates.into_iter(), output)
            .await
            .into_ann_result()? as u32;

        Ok(SearchStats { cmps, result_count })
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::{SearchOptions, knn_search, knn_search_with_options};
    use std::num::NonZeroUsize;

    use crate::flat::test::{
        harness::{CopyIdsOracle, EvenIdsOnlyOracle, KnnOracleRun, OracleProcessor},
        provider::{self as flat_provider},
    };
    use crate::graph::test::synthetic::Grid;

    fn fixture(grid: Grid, size: usize) -> (flat_provider::Provider, usize) {
        let provider = flat_provider::Provider::grid(grid, size).unwrap();
        let len = provider.len();
        (provider, len)
    }

    #[test]
    fn collector_switch_preserves_post_processing_and_statistics() {
        let (provider, len) = fixture(Grid::One, 256);
        for k in [1, 99, 100, 101, 1000] {
            for query in [&[-1.0][..], &[300.0][..]] {
                let result = KnnOracleRun::run_sync(&provider, &CopyIdsOracle, query, k).unwrap();
                assert_eq!(result.top_k, result.ground_truth);
                assert_eq!(result.stats.cmps as usize, len);
                assert_eq!(result.stats.result_count as usize, k.min(len));
                let filtered =
                    KnnOracleRun::run_sync(&provider, &EvenIdsOnlyOracle, query, k).unwrap();
                assert_eq!(filtered.top_k, filtered.ground_truth);
                assert_eq!(filtered.stats.cmps as usize, len);
                assert_eq!(
                    filtered.stats.result_count as usize,
                    filtered.ground_truth.len()
                );
            }
        }
    }

    async fn assert_options<O: OracleProcessor>(
        provider: &flat_provider::Provider,
        oracle: O,
        query: &[f32],
        k: NonZeroUsize,
    ) {
        let mut visitor = flat_provider::Visitor::new(provider, query).unwrap();
        let mut expected = Vec::new();
        let stats = knn_search(&mut visitor, k, oracle.processor(), query, &mut expected)
            .await
            .unwrap();
        let bits = |neighbors: &[crate::neighbor::Neighbor<u32>]| {
            neighbors
                .iter()
                .map(|n| (*n.id(), n.distance().to_bits()))
                .collect::<Vec<_>>()
        };
        for reservoir_min_k in [0, 2, 17, 100, usize::MAX] {
            let mut visitor = flat_provider::Visitor::new(provider, query).unwrap();
            let mut output = Vec::new();
            let actual_stats = knn_search_with_options(
                &mut visitor,
                k,
                SearchOptions { reservoir_min_k },
                oracle.processor(),
                query,
                &mut output,
            )
            .await
            .unwrap();
            assert_eq!(actual_stats, stats);
            assert_eq!(
                bits(&output),
                bits(&expected),
                "k={k}, threshold={reservoir_min_k}"
            );
        }
    }

    #[tokio::test]
    async fn custom_thresholds_preserve_results_and_post_processing() {
        let (provider, _) = fixture(Grid::One, 256);
        for k in [1, 2, 16, 17, 18, 99, 100, 101, 1000] {
            for query in [&[-1.0][..], &[128.0][..], &[300.0][..]] {
                let k = NonZeroUsize::new(k).unwrap();
                assert_options(&provider, CopyIdsOracle, query, k).await;
                assert_options(&provider, EvenIdsOnlyOracle, query, k).await;
            }
        }
    }

    #[tokio::test]
    async fn custom_thresholds_escalate_scan_errors_without_post_processing() {
        use crate::graph::glue::CopyIds;

        let (provider, _) = fixture(Grid::One, 32);
        for k in [1, 17] {
            for reservoir_min_k in [0, 17, 18, 100] {
                let mut visitor = flat_provider::Visitor::flaky(&provider, &[0.0], [3]).unwrap();
                let mut output = Vec::new();
                let err = knn_search_with_options(
                    &mut visitor,
                    NonZeroUsize::new(k).unwrap(),
                    SearchOptions { reservoir_min_k },
                    CopyIds,
                    (),
                    &mut output,
                )
                .await
                .expect_err("partial scans must fail before post-processing");
                assert!(format!("{err}").contains("id 3"));
                assert!(output.is_empty());
            }
        }
    }

    /// `knn_search` returns a `Send` future, and a shared provider can serve
    /// many concurrent searches on a multi-threaded runtime, each producing the
    /// correct output independently.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn multithreaded_knn_search() {
        use std::sync::Arc;

        let (provider, len) = fixture(Grid::Two, 4);
        let provider = Arc::new(provider);

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
            provider: &Arc<flat_provider::Provider>,
            oracle: O,
            cases: &[(&[f32], usize)],
        ) where
            O: OracleProcessor + Copy + Send + Sync + 'static,
        {
            for (query, k) in cases {
                let provider = Arc::clone(provider);
                let query: Vec<f32> = query.to_vec();
                let k = *k;
                set.spawn(async move {
                    let outcome = KnnOracleRun::run(&provider, &oracle, &query, k)
                        .await
                        .expect("knn_search failed");
                    (query, k, outcome)
                });
            }
        }

        let mut set = tokio::task::JoinSet::new();
        spawn_cases(&mut set, &provider, CopyIdsOracle, cases);
        spawn_cases(&mut set, &provider, EvenIdsOnlyOracle, cases);

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
        for (transient_ids, k) in [(&[0u32][..], 4), (&[3][..], 100), (&[1, 2, 5][..], 101)] {
            let (provider, _) = fixture(Grid::Two, 3);
            let query = &[1.0, 0.0];
            let visitor =
                flat_provider::Visitor::flaky(&provider, query, transient_ids.iter().copied())
                    .unwrap();
            let err =
                KnnOracleRun::run_sync_with_visitor(&provider, visitor, &CopyIdsOracle, query, k)
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
    fn assert_visitor_error(query: &[f32], expected_msg: &str) {
        let (provider, _) = fixture(Grid::Two, 3);
        let err = flat_provider::Visitor::new(&provider, query)
            .expect_err("expected visitor construction to fail");

        let msg = format!("{err}");
        assert!(
            msg.contains(expected_msg),
            "expected error containing {expected_msg:?}, got: {msg}",
        );
    }

    #[test]
    fn visitor_constructor_errors() {
        assert_visitor_error(&[0.0, 0.0, 0.0], "dimension mismatch");
    }
}
