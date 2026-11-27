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
        harness::{
            CopyIdsOracle, EvenIdsOnlyOracle, KnnOracleRun, OracleProcessor, brute_force_topk,
        },
        provider::{self as flat_provider},
    };
    use crate::graph::test::synthetic::Grid;

    fn fixture(grid: Grid, size: usize) -> (flat_provider::Provider, usize) {
        let provider = flat_provider::Provider::grid(grid, size).unwrap();
        let len = provider.len();
        (provider, len)
    }

    async fn assert_options<O: OracleProcessor>(
        provider: &flat_provider::Provider,
        oracle: O,
        query: &[f32],
        k: NonZeroUsize,
    ) {
        let expected = oracle.expected(brute_force_topk(
            provider,
            diskann_vector::distance::Metric::L2,
            query,
            k.get(),
        ));
        let mut visitor = flat_provider::Visitor::new(provider, query).unwrap();
        let mut output = Vec::new();
        let stats = knn_search(&mut visitor, k, oracle.processor(), query, &mut output)
            .await
            .unwrap();
        let bits = |neighbors: &[crate::neighbor::Neighbor<u32>]| {
            neighbors
                .iter()
                .map(|n| (*n.id(), n.distance().to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&output), bits(&expected));
        assert_eq!(stats.cmps as usize, provider.len());
        assert_eq!(stats.result_count as usize, expected.len());
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
            // Unique distances let the existing oracle check IDs as well as scores.
            // Collector tests separately exercise scan-order ties and signed zeros.
            for query in [&[-1.0][..], &[300.0][..]] {
                let k = NonZeroUsize::new(k).unwrap();
                assert_options(&provider, CopyIdsOracle, query, k).await;
                assert_options(&provider, EvenIdsOnlyOracle, query, k).await;
            }
        }
    }

    struct BatchVisitor<'a> {
        distances: &'a [f32],
        fail: bool,
        batches: usize,
        stage: bool,
        scalar_scans: usize,
    }

    impl crate::provider::HasId for BatchVisitor<'_> {
        type Id = u32;
    }

    impl crate::flat::DistancesUnordered for BatchVisitor<'_> {
        type Error = crate::ANNError;

        async fn distances_unordered<F>(&mut self, mut f: F) -> Result<(), Self::Error>
        where
            F: Send + FnMut(u32, f32),
        {
            self.scalar_scans += 1;
            for (i, &distance) in self.distances.iter().enumerate() {
                f(i as u32, distance);
                if self.fail && i == 16 {
                    return Err(crate::ANNError::new(std::io::Error::other(
                        "batch scan failed",
                    )));
                }
            }
            Ok(())
        }

        async fn distances_unordered_with<H>(&mut self, handler: &mut H) -> Result<(), Self::Error>
        where
            H: crate::flat::DistanceHandler<u32>,
        {
            if self.stage && !handler.prefers_buffered_batches() {
                return self
                    .distances_unordered(|id, distance| handler.add(id, distance))
                    .await;
            }
            handler.add_batch(&[], &[]);
            for (block, distances) in self.distances.chunks(17).enumerate() {
                let ids: Vec<_> = (0..distances.len())
                    .map(|i| (17 * block + i) as u32)
                    .collect();
                // Mix scalar submissions, SIMD groups and tails in one scan.
                handler.add(ids[0], distances[0]);
                handler.add_batch(&ids[1..], &distances[1..]);
                self.batches += 1;
                if self.fail {
                    return Err(crate::ANNError::new(std::io::Error::other(
                        "batch scan failed",
                    )));
                }
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn mixed_batches_preserve_post_processing_and_scan_errors() {
        use crate::flat::test::harness::EvenIdsOnly;
        use crate::{
            flat::SearchStats,
            neighbor::{Neighbor, NeighborPriorityQueue},
        };

        let distances: Vec<_> = (0..301)
            .map(|i| match i % 6 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => -0.0,
                3 => 0.0,
                _ => (i % 11) as f32 - 5.0,
            })
            .collect();
        for k in [1, 2, 99, 100, 101, 400] {
            let mut reference = NeighborPriorityQueue::new(k);
            for (i, &d) in distances.iter().enumerate() {
                reference.insert(Neighbor::new(i as u32, d));
            }
            let expected: Vec<_> = reference
                .iter()
                .filter(|n| *n.id() % 2 == 0)
                .map(|n| (*n.id(), n.distance().to_bits()))
                .collect();
            for reservoir_min_k in [0, 100, usize::MAX] {
                for stage in [false, true] {
                    let mut visitor = BatchVisitor {
                        distances: &distances,
                        fail: false,
                        batches: 0,
                        stage,
                        scalar_scans: 0,
                    };
                    let mut output = Vec::new();
                    let stats = knn_search_with_options(
                        &mut visitor,
                        NonZeroUsize::new(k).unwrap(),
                        SearchOptions { reservoir_min_k },
                        EvenIdsOnly,
                        (),
                        &mut output,
                    )
                    .await
                    .unwrap();
                    if stage && k == 1 {
                        assert_eq!(visitor.scalar_scans, 1);
                        assert_eq!(visitor.batches, 0);
                    } else {
                        // Native blocks must remain batched even for Top1. Other
                        // collectors also retain batching for staged scores.
                        assert_eq!(visitor.scalar_scans, 0);
                        assert!(visitor.batches > 0);
                    }
                    assert_eq!(
                        stats,
                        SearchStats {
                            cmps: distances.len() as u32,
                            result_count: expected.len() as u32
                        }
                    );
                    assert_eq!(
                        output
                            .iter()
                            .map(|n| (*n.id(), n.distance().to_bits()))
                            .collect::<Vec<_>>(),
                        expected
                    );
                    visitor.fail = true;
                    output.clear();
                    let err = knn_search_with_options(
                        &mut visitor,
                        NonZeroUsize::new(k).unwrap(),
                        SearchOptions { reservoir_min_k },
                        EvenIdsOnly,
                        (),
                        &mut output,
                    )
                    .await
                    .unwrap_err();
                    assert!(format!("{err}").contains("batch scan failed"));
                    assert!(output.is_empty());
                }
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
    #[tokio::test]
    async fn transient_scan_error() {
        use crate::graph::glue::CopyIds;

        let (provider, _) = fixture(Grid::Two, 3);
        let query = &[1.0, 0.0];
        // The flat scan touches every id, so any transient id is guaranteed to be hit.
        for transient_ids in [&[0u32][..], &[3][..], &[1, 2, 5][..]] {
            for k in [1, 4, 100, 101] {
                for threshold in [None, Some(0), Some(100), Some(usize::MAX)] {
                    let mut visitor = flat_provider::Visitor::flaky(
                        &provider,
                        query,
                        transient_ids.iter().copied(),
                    )
                    .unwrap();
                    let mut output = Vec::new();
                    let k = NonZeroUsize::new(k).unwrap();
                    let result = if let Some(reservoir_min_k) = threshold {
                        knn_search_with_options(
                            &mut visitor,
                            k,
                            SearchOptions { reservoir_min_k },
                            CopyIds,
                            (),
                            &mut output,
                        )
                        .await
                    } else {
                        knn_search(&mut visitor, k, CopyIds, (), &mut output).await
                    };
                    let err = result.expect_err("transient error during full scan must escalate");
                    let msg = format!("{err}");
                    assert!(
                        transient_ids
                            .iter()
                            .any(|id| msg.contains(&format!("id {id}"))),
                        "transients = {transient_ids:?}, k={k}, threshold={threshold:?}: {msg}",
                    );
                    assert!(
                        output.is_empty(),
                        "partial scans must not reach post-processing"
                    );
                }
            }
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
