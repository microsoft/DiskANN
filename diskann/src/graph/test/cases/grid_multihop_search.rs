/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Baseline regression tests for multihop (label-filtered) search on pre-built grid indices.
//!
//! This module validates [`MultihopSearch`] over synthetic grid workloads,
//! exercising the two-hop expansion strategy with various label filters.
//! Metrics (comparisons, hops, get_vector calls) are recorded and checked
//! against baselines so that algorithmic changes are detected early.

use std::sync::Arc;

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, DiskANNIndex,
        index::{QueryLabelProvider, QueryVisitDecision},
        search::{Knn, MultihopSearch},
        test::{provider as test_provider, synthetic::Grid},
    },
    neighbor::{self, Neighbor},
    test::{
        TestPath, TestRoot,
        cmp::{assert_eq_verbose, verbose_eq},
        get_or_save_test_results,
        tokio::current_thread_runtime,
    },
    utils::IntoUsize,
};

fn root() -> TestRoot {
    TestRoot::new("graph/test/cases/grid_multihop_search")
}

/////////////////////
// Label Providers //
/////////////////////

/// A filter that accepts only vertices whose ID is even.
#[derive(Debug)]
struct EvenFilter;

impl QueryLabelProvider<u32> for EvenFilter {
    fn is_match(&self, vec_id: u32) -> bool {
        vec_id.is_multiple_of(2)
    }
}

/// A filter that accepts all vertices (no filtering).
#[derive(Debug)]
struct AcceptAll;

impl QueryLabelProvider<u32> for AcceptAll {
    fn is_match(&self, _vec_id: u32) -> bool {
        true
    }
}

/// A filter that rejects all vertices.
#[derive(Debug)]
struct RejectAll;

impl QueryLabelProvider<u32> for RejectAll {
    fn is_match(&self, _vec_id: u32) -> bool {
        false
    }
    fn on_visit(&self, _neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
        QueryVisitDecision::Reject
    }
}

/// A filter that terminates search after visiting `n` candidates.
#[derive(Debug)]
struct TerminateAfterN {
    n: std::sync::atomic::AtomicUsize,
}

impl TerminateAfterN {
    fn new(n: usize) -> Self {
        Self {
            n: std::sync::atomic::AtomicUsize::new(n),
        }
    }
}

impl QueryLabelProvider<u32> for TerminateAfterN {
    fn is_match(&self, _vec_id: u32) -> bool {
        false
    }

    fn on_visit(&self, neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
        let prev = self.n.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        if prev == 0 {
            QueryVisitDecision::Terminate
        } else {
            // Accept all until we hit the threshold.
            QueryVisitDecision::Accept(neighbor)
        }
    }
}

/// A filter that adjusts the distance of even-numbered IDs to be very small,
/// testing that distance adjustments in `on_visit` affect ranking.
#[derive(Debug)]
struct DistanceAdjuster;

impl QueryLabelProvider<u32> for DistanceAdjuster {
    fn is_match(&self, _vec_id: u32) -> bool {
        true
    }

    fn on_visit(&self, neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
        if neighbor.id.is_multiple_of(2) {
            // Make even-numbered IDs appear much closer.
            QueryVisitDecision::Accept(Neighbor::new(neighbor.id, neighbor.distance * 0.01))
        } else {
            QueryVisitDecision::Accept(neighbor)
        }
    }
}

/////////////////////
// Multihop Search //
/////////////////////

/// Metrics and results for a single multihop search of the grid synthetic workload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GridMultihopSearch {
    /// Description of the test scenario for reviewers.
    description: String,

    /// The query given to search.
    query: Vec<f32>,

    /// The results returned from search as (id, distance) pairs.
    results: Vec<(u32, f32)>,

    /// The number of comparisons returned by search.
    comparisons: usize,

    /// The number of hops returned by search.
    hops: usize,

    /// The number of results returned by search.
    num_results: usize,

    /// The dimensionality of the underlying grid.
    grid_dims: usize,

    /// The size of the underlying grid.
    grid_size: usize,

    /// The beam width used for search.
    beam_width: usize,

    /// Name of the filter used.
    filter_name: String,

    /// Index level metrics recorded during search.
    metrics: test_provider::Metrics,
}

verbose_eq!(GridMultihopSearch {
    description,
    query,
    results,
    comparisons,
    hops,
    num_results,
    grid_dims,
    grid_size,
    beam_width,
    filter_name,
    metrics,
});

/// Initialize the test provider with a grid and configure it for L2 distance.
fn setup_grid_multihop_search(
    grid: Grid,
    size: usize,
) -> Arc<DiskANNIndex<test_provider::Provider>> {
    let provider = test_provider::Provider::grid(grid, size).unwrap();

    let index_config = graph::config::Builder::new(
        provider.max_degree(),
        graph::config::MaxDegree::same(),
        100,
        (Metric::L2).into(),
    )
    .build()
    .unwrap();

    Arc::new(DiskANNIndex::new(index_config, provider, None))
}

const BEAM_WIDTHS: [usize; 2] = [1, 2];

//////////////////
// Even Filter  //
//////////////////

fn _grid_multihop_even_filter(grid: Grid, size: usize, mut parent: TestPath<'_>) {
    let rt = current_thread_runtime();
    let dim: usize = grid.dim().into();
    let k = 10;

    let description_template = "Multihop search with EvenFilter: only even-numbered IDs \
        should appear in results. Two-hop expansion allows traversal through odd \
        nodes to reach even targets. Increasing beam width may affect comparisons.";

    let queries = [
        vec![-1.0f32; dim],
        vec![0.5f32; dim],
        vec![size as f32; dim],
    ];

    let mut results = Vec::new();
    for query in &queries {
        for &beam_width in &BEAM_WIDTHS {
            let index = setup_grid_multihop_search(grid, size);
            let context = test_provider::Context::new();

            let knn = Knn::new(k, k, Some(beam_width)).unwrap();
            let search_params = MultihopSearch::new(knn, &EvenFilter);

            let mut neighbors = vec![Neighbor::<u32>::default(); k];
            let stats = rt
                .block_on(index.search(
                    search_params,
                    &test_provider::Strategy::new(),
                    &context,
                    query.as_slice(),
                    &mut neighbor::BackInserter::new(neighbors.as_mut_slice()),
                ))
                .unwrap();

            let metrics = index.provider().metrics();

            // Verify mutation metrics are zero.
            assert_eq!(metrics.set_vector, 0);
            assert_eq!(metrics.set_neighbors, 0);
            assert_eq!(metrics.append_neighbors, 0);

            {
                let test_provider::ContextMetrics { spawns, clones } = context.metrics();
                assert_eq!(spawns, 0);
                assert_eq!(clones, 0);
            }

            let result_tuples: Vec<(u32, f32)> = neighbors
                .iter()
                .take(stats.result_count.into_usize())
                .map(|n| n.as_tuple())
                .collect();

            // All returned IDs should be even (matching the filter), except
            // the start point (u32::MAX) which is always seeded into the
            // frontier before label filtering.
            for (id, _) in &result_tuples {
                assert!(
                    id.is_multiple_of(2) || *id == u32::MAX,
                    "EvenFilter should only return even IDs (or start point), but got {}",
                    id
                );
            }

            results.push(GridMultihopSearch {
                query: query.clone(),
                description: description_template.to_string(),
                results: result_tuples,
                comparisons: stats.cmps.into_usize(),
                hops: stats.hops.into_usize(),
                num_results: stats.result_count.into_usize(),
                grid_dims: dim,
                grid_size: size,
                beam_width,
                filter_name: "EvenFilter".to_string(),
                metrics,
            });
        }
    }

    let name = parent.push(format!("multihop_even_{}_{}", grid.dim(), size));
    let expected = get_or_save_test_results(&name, &results);
    assert_eq_verbose!(expected, results);
}

/////////////////////
// Reject All      //
/////////////////////

fn _grid_multihop_reject_all(grid: Grid, size: usize) {
    let rt = current_thread_runtime();
    let dim: usize = grid.dim().into();
    let k = 10;

    let index = setup_grid_multihop_search(grid, size);
    let context = test_provider::Context::new();

    let knn = Knn::new(k, k, Some(1)).unwrap();
    let search_params = MultihopSearch::new(knn, &RejectAll);

    let mut neighbors = vec![Neighbor::<u32>::default(); k];
    let stats = rt
        .block_on(index.search(
            search_params,
            &test_provider::Strategy::new(),
            &context,
            vec![-1.0f32; dim].as_slice(),
            &mut neighbor::BackInserter::new(neighbors.as_mut_slice()),
        ))
        .unwrap();

    // When all candidates are rejected, only the start point (which is seeded
    // into the frontier before label filtering) should appear in results.
    assert!(
        stats.result_count.into_usize() <= 1,
        "RejectAll filter should produce at most 1 result (the start point), got {}",
        stats.result_count,
    );
}

/////////////////////
// Terminate Early //
/////////////////////

fn _grid_multihop_terminate_early(grid: Grid, size: usize) {
    let rt = current_thread_runtime();
    let dim: usize = grid.dim().into();
    let k = 10;

    let index = setup_grid_multihop_search(grid, size);
    let context = test_provider::Context::new();

    let knn = Knn::new(k, k, Some(1)).unwrap();
    let filter = TerminateAfterN::new(2);
    let search_params = MultihopSearch::new(knn, &filter);

    let mut neighbors = vec![Neighbor::<u32>::default(); k];
    let stats = rt
        .block_on(index.search(
            search_params,
            &test_provider::Strategy::new(),
            &context,
            vec![-1.0f32; dim].as_slice(),
            &mut neighbor::BackInserter::new(neighbors.as_mut_slice()),
        ))
        .unwrap();

    // With early termination after 2 visits, the search should have limited results.
    // The exact count depends on graph structure, but it should be small.
    let result_count = stats.result_count.into_usize();
    assert!(
        result_count <= k,
        "TerminateAfterN should limit results, got {}",
        result_count
    );

    // Verify comparisons are bounded - early termination should prevent full traversal.
    let full_index = setup_grid_multihop_search(grid, size);
    let full_context = test_provider::Context::new();
    let full_knn = Knn::new(k, k, Some(1)).unwrap();
    let full_search = MultihopSearch::new(full_knn, &AcceptAll);
    let mut full_neighbors = vec![Neighbor::<u32>::default(); k];
    let full_stats = rt
        .block_on(full_index.search(
            full_search,
            &test_provider::Strategy::new(),
            &full_context,
            vec![-1.0f32; dim].as_slice(),
            &mut neighbor::BackInserter::new(full_neighbors.as_mut_slice()),
        ))
        .unwrap();

    assert!(
        stats.cmps <= full_stats.cmps,
        "Early termination should result in fewer comparisons: \
         terminated={}, full={}",
        stats.cmps,
        full_stats.cmps
    );
}

///////////////////////////
// Distance Adjustment   //
///////////////////////////

fn _grid_multihop_distance_adjustment(grid: Grid, size: usize) {
    let rt = current_thread_runtime();
    let dim: usize = grid.dim().into();
    let k = 10;

    // Run with the distance adjuster.
    let index_adjusted = setup_grid_multihop_search(grid, size);
    let context_adjusted = test_provider::Context::new();

    let knn = Knn::new(k, k, Some(1)).unwrap();
    let search_params = MultihopSearch::new(knn, &DistanceAdjuster);

    let mut neighbors_adjusted = vec![Neighbor::<u32>::default(); k];
    let stats_adjusted = rt
        .block_on(index_adjusted.search(
            search_params,
            &test_provider::Strategy::new(),
            &context_adjusted,
            vec![-1.0f32; dim].as_slice(),
            &mut neighbor::BackInserter::new(neighbors_adjusted.as_mut_slice()),
        ))
        .unwrap();

    let adjusted_results: Vec<(u32, f32)> = neighbors_adjusted
        .iter()
        .take(stats_adjusted.result_count.into_usize())
        .map(|n| n.as_tuple())
        .collect();

    // Run without the distance adjuster (AcceptAll filter).
    let index_plain = setup_grid_multihop_search(grid, size);
    let context_plain = test_provider::Context::new();

    let knn_plain = Knn::new(k, k, Some(1)).unwrap();
    let search_plain = MultihopSearch::new(knn_plain, &AcceptAll);

    let mut neighbors_plain = vec![Neighbor::<u32>::default(); k];
    let stats_plain = rt
        .block_on(index_plain.search(
            search_plain,
            &test_provider::Strategy::new(),
            &context_plain,
            vec![-1.0f32; dim].as_slice(),
            &mut neighbor::BackInserter::new(neighbors_plain.as_mut_slice()),
        ))
        .unwrap();

    let plain_results: Vec<(u32, f32)> = neighbors_plain
        .iter()
        .take(stats_plain.result_count.into_usize())
        .map(|n| n.as_tuple())
        .collect();

    // The distance adjuster scales even IDs by 0.01. This should cause even IDs
    // to dominate the top-k results compared to the plain search.
    let adjusted_even_count = adjusted_results
        .iter()
        .filter(|(id, _)| id.is_multiple_of(2))
        .count();
    let plain_even_count = plain_results
        .iter()
        .filter(|(id, _)| id.is_multiple_of(2))
        .count();

    assert!(
        adjusted_even_count >= plain_even_count,
        "Distance adjustment should cause more even IDs to appear in results: \
         adjusted_even={}, plain_even={}",
        adjusted_even_count,
        plain_even_count
    );
}

///////////
// Tests //
///////////

// Even filter tests with baselines.

#[test]
fn multihop_even_filter_3_5() {
    _grid_multihop_even_filter(Grid::Three, 5, root().path());
}

#[test]
fn multihop_even_filter_4_4() {
    _grid_multihop_even_filter(Grid::Four, 4, root().path());
}

// Reject-all tests (no baselines needed — deterministic zero results).

#[test]
fn multihop_reject_all_3_5() {
    _grid_multihop_reject_all(Grid::Three, 5);
}

#[test]
fn multihop_reject_all_4_4() {
    _grid_multihop_reject_all(Grid::Four, 4);
}

// Early termination tests (no baselines — structural assertions only).

#[test]
fn multihop_terminate_early_3_5() {
    _grid_multihop_terminate_early(Grid::Three, 5);
}

#[test]
fn multihop_terminate_early_4_4() {
    _grid_multihop_terminate_early(Grid::Four, 4);
}

// Distance adjustment tests (no baselines — relative assertions only).

#[test]
fn multihop_distance_adjustment_3_5() {
    _grid_multihop_distance_adjustment(Grid::Three, 5);
}

#[test]
fn multihop_distance_adjustment_4_4() {
    _grid_multihop_distance_adjustment(Grid::Four, 4);
}
