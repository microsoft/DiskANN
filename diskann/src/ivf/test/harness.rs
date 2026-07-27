/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Reusable execution harness for [`crate::ivf::IvfIndex`] tests.
//!
//! [`IvfOracleRun`] drives [`IvfIndex::knn_search`] and pairs the result with an *exact*
//! oracle. IVF is an approximate algorithm only because it restricts the search to the
//! members of the `nprobe` selected lists -- within that restriction it is exhaustive. The
//! oracle exploits this: it replays the same list selection (via the shared
//! [`nearest_lists`](super::provider::nearest_lists) core) and then brute-forces *only* the
//! union of those lists' members. The IVF result must therefore match the oracle exactly,
//! which lets the tests assert strict equality rather than a recall threshold.

use std::{cmp::Ordering, num::NonZeroUsize};

use diskann_vector::PreprocessedDistanceFunction;

use crate::{
    ANNResult,
    ivf::{
        IvfIndex, SearchStats,
        test::provider::{self, Id, ListId, Provider, Strategy},
    },
    neighbor::{BackInserter, Neighbor},
    test::tokio::current_thread_runtime,
    utils::VectorRepr,
};

/// Result of one [`IvfIndex::knn_search`] run alongside the exact oracle answer.
#[derive(Debug, Clone)]
pub(crate) struct IvfOracleRun {
    /// `(id, distance)` pairs returned by the search, re-sorted by `(distance asc, id asc)`
    /// so equality checks are deterministic on ties.
    pub top_k: Vec<(Id, f32)>,
    /// `top_k.iter().map(|(_, d)| d).collect()` -- convenient for distance-multiset checks.
    pub top_k_distances: Vec<f32>,
    /// Statistics returned by `knn_search`.
    pub stats: SearchStats,
    /// The exact expected top-`k`: brute force restricted to the members of the selected
    /// lists, in `(distance asc, id asc)` order.
    pub ground_truth: Vec<(Id, f32)>,
    /// The lists the oracle (and thus the search) probed for this query.
    pub probed_lists: Vec<ListId>,
    /// Total number of members across `probed_lists` -- the exact number of fine-distance
    /// computations the search must perform.
    pub probed_members: usize,
}

impl IvfOracleRun {
    /// Run [`IvfIndex::knn_search`] once, blocking on a fresh single-threaded runtime.
    pub fn run_sync(
        index: &IvfIndex<Provider>,
        strategy: &Strategy,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> ANNResult<Self> {
        current_thread_runtime().block_on(Self::run(index, strategy, query, k, nprobe))
    }

    /// Async variant of [`IvfOracleRun::run_sync`], for tests that already own a runtime
    /// (e.g. `#[tokio::test]`) or drive searches concurrently across tasks.
    pub async fn run(
        index: &IvfIndex<Provider>,
        strategy: &Strategy,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> ANNResult<Self> {
        let context = provider::Context::new();
        let mut buf = vec![Neighbor::<Id>::default(); k];

        let stats = index
            .knn_search(
                NonZeroUsize::new(k).expect("ivf::test::harness requires k > 0"),
                nprobe,
                strategy,
                &context,
                query,
                &mut BackInserter::new(buf.as_mut_slice()),
            )
            .await?;

        let mut top_k: Vec<Neighbor<Id>> = buf
            .iter()
            .copied()
            .take(stats.result_count as usize)
            .collect();
        sort_neighbors(&mut top_k);
        let top_k_distances = top_k.iter().map(|n| n.distance).collect();

        let (ground_truth, probed_lists, probed_members) =
            restricted_topk(index.provider(), query, k, nprobe);

        Ok(Self {
            top_k: top_k.into_iter().map(Neighbor::as_tuple).collect(),
            top_k_distances,
            stats,
            ground_truth: ground_truth.into_iter().map(Neighbor::as_tuple).collect(),
            probed_lists,
            probed_members,
        })
    }
}

/// Exact IVF oracle: select the same `nprobe` lists the search would, then brute-force the
/// union of their members.
///
/// Returns `(top_k, probed_lists, probed_member_count)`.
pub(crate) fn restricted_topk(
    provider: &Provider,
    query: &[f32],
    k: usize,
    nprobe: usize,
) -> (Vec<Neighbor<Id>>, Vec<ListId>, usize) {
    let metric = provider.metric();
    let probed_lists = provider::nearest_lists(provider.centroids(), query, metric, nprobe);

    let computer = f32::query_distance(query, metric);
    let mut neighbors: Vec<Neighbor<Id>> = Vec::new();
    let mut probed_members = 0usize;
    for &list in &probed_lists {
        for (id, vector) in provider.list_entries(list) {
            probed_members += 1;
            neighbors.push(Neighbor::new(id, computer.evaluate_similarity(&vector[..])));
        }
    }

    sort_neighbors(&mut neighbors);
    neighbors.truncate(k);
    (neighbors, probed_lists, probed_members)
}

/// Brute-force top-`k` over the *entire* provider -- used by tests that probe every list
/// (`nprobe == n_lists`) and therefore expect the global answer.
pub(crate) fn global_topk(provider: &Provider, query: &[f32], k: usize) -> Vec<Neighbor<Id>> {
    let metric = provider.metric();
    let computer = f32::query_distance(query, metric);
    let mut neighbors: Vec<Neighbor<Id>> = Vec::new();
    for list in 0..provider.n_lists() as ListId {
        for (id, vector) in provider.list_entries(list) {
            neighbors.push(Neighbor::new(id, computer.evaluate_similarity(&vector[..])));
        }
    }
    sort_neighbors(&mut neighbors);
    neighbors.truncate(k);
    neighbors
}

/// Sort `(distance asc, id asc)` with NaN treated as equal.
fn sort_neighbors(neighbors: &mut [Neighbor<Id>]) {
    neighbors.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
}

/// Assert two `(id, distance)` lists agree on the *distance multiset*.
///
/// The priority queue may break exact-distance ties differently from the oracle's
/// `(distance, id)` ordering, so element-wise id equality is too strict. Comparing the
/// sorted distance sequences (already sorted by the harness) plus the length is the right
/// invariant for a correct top-`k`.
pub(crate) fn assert_same_distances(actual: &[(Id, f32)], expected: &[(Id, f32)]) {
    let a: Vec<f32> = actual.iter().map(|(_, d)| *d).collect();
    let e: Vec<f32> = expected.iter().map(|(_, d)| *d).collect();
    assert_eq!(
        a.len(),
        e.len(),
        "result count mismatch: got {a:?}, expected {e:?}"
    );
    for (i, (ad, ed)) in a.iter().zip(e.iter()).enumerate() {
        assert!(
            (ad - ed).abs() <= f32::EPSILON,
            "distance mismatch at rank {i}: got {ad}, expected {ed} (got {a:?}, expected {e:?})"
        );
    }
}
