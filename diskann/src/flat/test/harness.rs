/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Reusable execution harness for [`crate::flat::FlatIndex`] tests.
//!
//! Use [`KnnOracleRun::run`] to drive `knn_search` and pair the result with the
//! brute-force ground truth.

use std::{cmp::Ordering, num::NonZeroUsize};

use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};

use crate::{
    ANNResult,
    flat::{
        FlatIndex, SearchStats,
        test::provider::{Provider, Strategy},
    },
    neighbor::{BackInserter, Neighbor},
    test::tokio::current_thread_runtime,
    utils::VectorRepr,
};

/// Result of running [`FlatIndex::knn_search`] under the harness alongside a
/// brute-force ground-truth oracle.
#[derive(Debug, Clone)]
pub(crate) struct KnnOracleRun {
    /// Top-`k` `(id, distance)` pairs.
    /// Re-sorted from the heap output so equality checks are deterministic on ties.
    pub top_k: Vec<(u32, f32)>,
    /// `top_k.iter().map(|(_, d)| d).collect()`.
    pub top_k_distances: Vec<f32>,
    /// Statistics returned by `knn_search` (cmps, result_count).
    pub stats: SearchStats,
    /// Brute-force ground-truth top-`k` `(id, distance)` pairs in `(distance asc,
    /// id asc)` order.
    pub ground_truth: Vec<(u32, f32)>,
}

impl KnnOracleRun {
    /// Run [`FlatIndex::knn_search`] once, blocking on a fresh single-threaded
    /// runtime, and pair the result with the brute-force ground truth.
    pub fn run_sync(
        index: &FlatIndex<Provider>,
        strategy: &Strategy,
        query: &[f32],
        k: usize,
    ) -> ANNResult<Self> {
        current_thread_runtime().block_on(Self::run(index, strategy, query, k))
    }

    /// Async variant of [`KnnOracleRun::run_sync`]. Use this from tests that already
    /// have a Tokio runtime (e.g. `#[tokio::test]`) or that need to drive
    /// `knn_search` concurrently across tasks.
    pub async fn run(
        index: &FlatIndex<Provider>,
        strategy: &Strategy,
        query: &[f32],
        k: usize,
    ) -> ANNResult<Self> {
        let context = crate::flat::test::provider::Context::new();
        let mut buf = vec![Neighbor::<u32>::default(); k];

        let stats = index
            .knn_search(
                NonZeroUsize::new(k).expect("flat::test::harness requires k > 0"),
                strategy,
                &context,
                query,
                &mut BackInserter::new(buf.as_mut_slice()),
            )
            .await?;

        let top_k = top_k_sorted(&buf, stats.result_count as usize);
        let top_k_distances = top_k.iter().map(|(_, d)| *d).collect();
        let ground_truth = brute_force_topk(index.provider(), Metric::L2, query, k);

        Ok(Self {
            top_k,
            top_k_distances,
            stats,
            ground_truth,
        })
    }
}

/// Compute the brute-force top-`k` `(id, distance)` pairs over every element of
/// `provider` under `metric`. Iterates [`Provider::items`] directly and scores with
/// a fresh [`f32::query_distance`] computer, so the oracle is independent of the
/// [`crate::flat::test::provider::Visitor`] under test. Ties are broken by ascending
/// id for determinism.
pub(crate) fn brute_force_topk(
    provider: &Provider,
    metric: Metric,
    query: &[f32],
    k: usize,
) -> Vec<(u32, f32)> {
    let computer = f32::query_distance(query, metric);

    let mut neighbors: Vec<Neighbor<u32>> = provider
        .items()
        .row_iter()
        .enumerate()
        .map(|(id, element)| Neighbor::new(id as u32, computer.evaluate_similarity(element)))
        .collect();

    sort_neighbors(&mut neighbors);
    neighbors
        .into_iter()
        .take(k)
        .map(|n| n.as_tuple())
        .collect()
}

/// Take the first `result_count` neighbors and return them in `(distance asc, id asc)`
/// order.
fn top_k_sorted(buf: &[Neighbor<u32>], result_count: usize) -> Vec<(u32, f32)> {
    let mut neighbors: Vec<Neighbor<u32>> = buf.iter().copied().take(result_count).collect();
    sort_neighbors(&mut neighbors);
    neighbors.into_iter().map(|n| n.as_tuple()).collect()
}

/// Sort a slice of [`Neighbor<u32>`] by `(distance asc, id asc)`. NaN distances are
/// treated as equal (test data should not produce NaN).
fn sort_neighbors(neighbors: &mut [Neighbor<u32>]) {
    neighbors.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
}
