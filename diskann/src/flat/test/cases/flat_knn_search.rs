/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Baseline-cached regression sweep for [`crate::flat::FlatIndex::knn_search`].
//!
//! Bbuilds a fresh index per parameter combination, runs `knn_search` through the
//! [`crate::flat::test::harness`], snapshots the result + statistics into
//! [`FlatKnnBaseline`], and compares the entire batch against the JSON committed under
//! `diskann/test/generated/flat/test/cases/flat_knn_search/`.

use crate::{
    flat::{
        FlatIndex,
        test::{
            harness,
            provider::{self as flat_provider, Metrics, Strategy},
        },
    },
    graph::test::synthetic::Grid,
    test::{
        TestPath, TestRoot,
        cmp::{assert_eq_verbose, verbose_eq},
        get_or_save_test_results,
    },
};

fn root() -> TestRoot {
    TestRoot::new("flat/test/cases/flat_knn_search")
}

/// `k` values exercised for every `(grid, query)` combination.
const KS: [usize; 3] = [1, 4, 10];

/// One row of the baseline JSON: a single `(grid, size, query, k)` execution of
/// `FlatIndex::knn_search` plus the brute-force ground truth, search stats, and
/// per-row provider metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct FlatKnnBaseline {
    /// Free-form description of what this row exercises.
    description: String,

    /// The query vector.
    query: Vec<f32>,

    /// The dimensionality of the underlying grid.
    grid_dims: usize,

    /// The side length of the underlying grid.
    grid_size: usize,

    /// The requested `k`.
    k: usize,

    /// Sorted distance multiset of the top-`k` returned by `knn_search`.
    /// We store the distance multiset rather than `(id, distance)` pairs because
    /// the priority queue may evict different *ids* on a boundary distance tie
    /// (the queue's tie-breaking is heap-internal, not id-based) ΓÇö but the
    /// multiset of distances is invariant.
    top_k_distances: Vec<f32>,

    /// Brute-force ground-truth top-`k` `(id, distance)` (sorted by `(distance asc,
    /// id asc)`). The brute-force pass enumerates ids in ascending order, so on a
    /// tie this prefers the smaller id and gives a canonical answer for the JSON.
    ground_truth: Vec<(u32, f32)>,

    /// `cmps` reported by `knn_search`. Must equal `provider.len()`.
    comparisons: usize,

    /// `result_count` reported by `knn_search`. Must equal `min(k, provider.len())`.
    result_count: usize,

    /// Per-provider metrics observed for this row (see [`Metrics`]).
    metrics: Metrics,
}

verbose_eq!(FlatKnnBaseline {
    description,
    query,
    grid_dims,
    grid_size,
    k,
    top_k_distances,
    ground_truth,
    comparisons,
    result_count,
    metrics,
});

/// Run `knn_search` + brute-force oracle against a *shared* `index`, assert the
/// cross-row invariants, and produce the baseline row. The per-row provider metrics
/// captured into the baseline are the *delta* observed during this row, which keeps
/// the snapshot independent of how many rows preceded it.
fn run_row(
    index: &FlatIndex<flat_provider::Provider>,
    grid_dim: usize,
    grid_size: usize,
    query: &[f32],
    k: usize,
    desc: &str,
) -> FlatKnnBaseline {
    let len = index.provider().len();
    let metrics_before = index.provider().metrics();

    let outcome = harness::KnnOracleRun::run_sync(index, &Strategy::new(), query, k).unwrap();
    let stats = outcome.stats;

    assert_eq!(
        stats.cmps as usize, len,
        "flat scan must touch every element exactly once",
    );
    assert_eq!(
        stats.result_count as usize,
        k.min(len),
        "result_count must equal min(k, provider.len())",
    );

    let gt_distances: Vec<f32> = outcome.ground_truth.iter().map(|(_, d)| *d).collect();
    assert_eq!(
        outcome.top_k_distances, gt_distances,
        "flat scan top-k distance multiset must agree with brute force",
    );

    let metrics_after = index.provider().metrics();
    let metrics = Metrics {
        get_element: metrics_after.get_element - metrics_before.get_element,
    };
    // `get_element` is incremented only by the [`Visitor`] used during `knn_search`;
    // the brute-force oracle iterates `Provider::items()` directly and does not touch
    // the visitor, so we expect exactly one scan's worth of increments per row.
    assert_eq!(
        metrics.get_element, len,
        "expected exactly one scan (from knn_search) to increment get_element",
    );

    FlatKnnBaseline {
        description: desc.to_string(),
        query: query.to_vec(),
        grid_dims: grid_dim,
        grid_size,
        k,
        top_k_distances: outcome.top_k_distances,
        ground_truth: outcome.ground_truth,
        comparisons: stats.cmps as usize,
        result_count: stats.result_count as usize,
        metrics,
    }
}

/// Sweep [`KS`] ├ù `queries` for the given `(grid, size)` and snapshot the results.
fn _flat_knn_search(grid: Grid, size: usize, mut parent: TestPath<'_>) {
    let dim: usize = grid.dim().into();

    // Build the provider and index once, mirroring the production pattern where a
    // single index serves many queries.
    let provider = flat_provider::Provider::grid(grid, size);
    let len = provider.len();
    assert_eq!(
        len,
        size.pow(dim as u32),
        "flat::test::Provider::grid should produce size^dim rows",
    );
    let index = FlatIndex::new(provider);

    let queries: [(Vec<f32>, &str); 2] = [
        (
            vec![-1.0; dim],
            "All -1: nearest is the all-zeros corner; result_count = min(k, len).",
        ),
        (
            vec![(size - 1) as f32; dim],
            "All `size-1`: query coincides with the last grid corner.",
        ),
    ];

    let index_ref = &index;
    let results: Vec<FlatKnnBaseline> = queries
        .iter()
        .flat_map(|(q, desc)| {
            KS.iter()
                .map(move |&k| run_row(index_ref, dim, size, q, k, desc))
        })
        .collect();

    let name = parent.push(format!("search_{dim}_{size}"));
    let expected = get_or_save_test_results(&name, &results);
    assert_eq_verbose!(expected, results);
}

#[test]
fn flat_knn_search_1_100() {
    _flat_knn_search(Grid::One, 100, root().path());
}

#[test]
fn flat_knn_search_2_5() {
    _flat_knn_search(Grid::Two, 5, root().path());
}

#[test]
fn flat_knn_search_3_4() {
    _flat_knn_search(Grid::Three, 4, root().path());
}
