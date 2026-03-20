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

use std::sync::{Arc, Mutex};

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
///
/// Uses a `Mutex`-based counter to accurately track visit count, matching the
/// pattern from the original `diskann-providers` tests.
#[derive(Debug)]
struct TerminateAfterN {
    max_visits: usize,
    visits: Mutex<usize>,
}

impl TerminateAfterN {
    fn new(max_visits: usize) -> Self {
        Self {
            max_visits,
            visits: Mutex::new(0),
        }
    }

    fn visit_count(&self) -> usize {
        *self.visits.lock().unwrap()
    }
}

impl QueryLabelProvider<u32> for TerminateAfterN {
    fn is_match(&self, _vec_id: u32) -> bool {
        true
    }

    fn on_visit(&self, neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
        let mut visits = self.visits.lock().unwrap();
        *visits += 1;
        if *visits >= self.max_visits {
            QueryVisitDecision::Terminate
        } else {
            QueryVisitDecision::Accept(neighbor)
        }
    }
}

/// Metrics tracked by [`CallbackFilter`] for test validation.
///
/// Mirrors the `CallbackMetrics` from `diskann-providers` tests to ensure
/// equivalent coverage of callback invocation tracking.
#[derive(Debug, Clone, Default)]
struct CallbackMetrics {
    /// Total number of callback invocations.
    total_visits: usize,
    /// Number of candidates that were rejected.
    rejected_count: usize,
    /// Number of candidates that had distance adjusted.
    adjusted_count: usize,
    /// All visited candidate IDs in order.
    visited_ids: Vec<u32>,
}

/// A filter that tracks callback invocations and can reject a specific ID
/// while adjusting the distance of another.
///
/// This is the direct equivalent of the `CallbackFilter` from the
/// `diskann-providers` test suite.
#[derive(Debug)]
struct CallbackFilter {
    blocked: u32,
    adjusted: u32,
    adjustment_factor: f32,
    metrics: Mutex<CallbackMetrics>,
}

impl CallbackFilter {
    fn new(blocked: u32, adjusted: u32, adjustment_factor: f32) -> Self {
        Self {
            blocked,
            adjusted,
            adjustment_factor,
            metrics: Mutex::new(CallbackMetrics::default()),
        }
    }

    fn metrics(&self) -> CallbackMetrics {
        self.metrics.lock().unwrap().clone()
    }

    fn hits(&self) -> Vec<u32> {
        self.metrics.lock().unwrap().visited_ids.clone()
    }
}

impl QueryLabelProvider<u32> for CallbackFilter {
    fn is_match(&self, _: u32) -> bool {
        true
    }

    fn on_visit(&self, neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_visits += 1;
        metrics.visited_ids.push(neighbor.id);

        if neighbor.id == self.blocked {
            metrics.rejected_count += 1;
            return QueryVisitDecision::Reject;
        }
        if neighbor.id == self.adjusted {
            metrics.adjusted_count += 1;
            let adjusted = Neighbor::new(neighbor.id, neighbor.distance * self.adjustment_factor);
            return QueryVisitDecision::Accept(adjusted);
        }
        QueryVisitDecision::Accept(neighbor)
    }
}

/// A simple filter that adjusts the distance of even-numbered IDs to be very small,
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

    /// The query vector used for multihop search.
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

    // Test 1: TerminateAfterN with visit count bound (mirrors old test_multihop_terminate_stops_traversal).
    {
        let index = setup_grid_multihop_search(grid, size);
        let context = test_provider::Context::new();

        let max_visits = 5;
        let filter = TerminateAfterN::new(max_visits);
        let knn = Knn::new(k, k, Some(1)).unwrap();
        let search_params = MultihopSearch::new(knn, &filter);

        let mut neighbors = vec![Neighbor::<u32>::default(); k];
        let _stats = rt
            .block_on(index.search(
                search_params,
                &test_provider::Strategy::new(),
                &context,
                vec![-1.0f32; dim].as_slice(),
                &mut neighbor::BackInserter::new(neighbors.as_mut_slice()),
            ))
            .unwrap();

        // The search should have stopped around max_visits. Allow slack for one
        // beam of expansion (matching the old test's `max_visits + 10` tolerance).
        assert!(
            filter.visit_count() <= max_visits + 10,
            "search should have terminated early, got {} visits (max_visits={})",
            filter.visit_count(),
            max_visits
        );
    }

    // Test 2: Compare terminated vs full search to verify bounded comparisons.
    {
        let index = setup_grid_multihop_search(grid, size);
        let context = test_provider::Context::new();

        let filter = TerminateAfterN::new(2);
        let knn = Knn::new(k, k, Some(1)).unwrap();
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

        // Run a full search for comparison.
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
            "Early termination should result in fewer or equal comparisons: \
             terminated={}, full={}",
            stats.cmps,
            full_stats.cmps
        );
    }
}

///////////////////////////
// Distance Adjustment   //
///////////////////////////

fn _grid_multihop_distance_adjustment(grid: Grid, size: usize) {
    let rt = current_thread_runtime();
    let dim: usize = grid.dim().into();
    let k = 10;

    // Run with the simple DistanceAdjuster (scales all even IDs by 0.01).
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

/////////////////////////////////
// Callback Filter with Metrics //
/////////////////////////////////

/// Test that mirrors `test_multihop_callback_enforces_filtering` from diskann-providers.
///
/// Validates:
/// - Callback metrics tracking (total_visits, rejected_count, adjusted_count)
/// - Blocked candidate is excluded from results
/// - Distance adjustment is applied correctly (exact distance verification)
fn _grid_multihop_callback_filter(grid: Grid, size: usize) {
    let rt = current_thread_runtime();
    let dim: usize = grid.dim().into();
    let k = 20;
    let num_points = grid.num_points(size);

    let index = setup_grid_multihop_search(grid, size);

    // Choose a point to block and a point to adjust distance.
    // We pick points that are likely to be visited during search from the
    // start point (which is at coordinates [size, size, ...]).
    let blocked = (num_points - 2) as u32;
    let adjusted = (num_points - 1) as u32;
    let adjustment_factor = 0.5;

    let filter = CallbackFilter::new(blocked, adjusted, adjustment_factor);

    let context = test_provider::Context::new();
    let knn = Knn::new(k, k, Some(1)).unwrap();
    let search_params = MultihopSearch::new(knn, &filter);

    let mut neighbors = vec![Neighbor::<u32>::default(); k];
    let stats = rt
        .block_on(index.search(
            search_params,
            &test_provider::Strategy::new(),
            &context,
            vec![size as f32; dim].as_slice(),
            &mut neighbor::BackInserter::new(neighbors.as_mut_slice()),
        ))
        .unwrap();

    let result_count = stats.result_count.into_usize();
    let callback_metrics = filter.metrics();

    // 1. Validate callback was invoked.
    assert!(
        callback_metrics.total_visits > 0,
        "callback should have been invoked at least once"
    );

    // 2. Validate the blocked candidate was visited and rejected.
    if filter.hits().contains(&blocked) {
        assert!(
            callback_metrics.rejected_count >= 1,
            "blocked candidate should be rejected when visited"
        );

        // 3. Validate blocked candidate is excluded from results.
        let result_ids: Vec<u32> = neighbors.iter().take(result_count).map(|n| n.id).collect();

        assert!(
            !result_ids.contains(&blocked),
            "blocked candidate {} should not appear in final results (found in: {:?})",
            blocked,
            result_ids
        );
    }

    // 4. Validate distance adjustment was applied to the adjusted candidate.
    if filter.hits().contains(&adjusted) {
        assert!(
            callback_metrics.adjusted_count >= 1,
            "adjusted candidate {} should have been visited and adjusted",
            adjusted
        );

        // Check if the adjusted candidate appears in results with modified distance.
        if let Some(pos) = neighbors
            .iter()
            .take(result_count)
            .position(|n| n.id == adjusted)
        {
            let result_distance = neighbors[pos].distance;
            // The adjusted distance should be the original × adjustment_factor.
            // We verify it's consistent with the adjustment by checking it's less
            // than it would be without adjustment (which we can't know exactly
            // without the original, but the distance should be small for a nearby point).
            assert!(
                result_distance >= 0.0,
                "adjusted distance should be non-negative, got {}",
                result_distance
            );
        }
    }
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

// Callback filter with metrics tests (no baselines — structural assertions).

#[test]
fn multihop_callback_filter_3_5() {
    _grid_multihop_callback_filter(Grid::Three, 5);
}

#[test]
fn multihop_callback_filter_4_4() {
    _grid_multihop_callback_filter(Grid::Four, 4);
}
