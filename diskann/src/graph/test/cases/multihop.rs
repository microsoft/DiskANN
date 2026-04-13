/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tests for multihop search traversal behavior, including filtering, early termination,
//! and distance adjustment callbacks.
//!
//! Migrated from `diskann-providers/src/index/diskann_async.rs`. The originals used the
//! inmem provider with PQ quantization; these use `test_provider::Provider::grid()` instead.
//!
//! **Notable difference**: The inmem provider's post-processor includes `FilterStartPoints`,
//! which strips start points from search results. The test provider does not, so
//! `reject_all_returns_zero_results` asserts zero *non-start-point* results rather than
//! zero total results.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use diskann_vector::{
    PureDistanceFunction,
    distance::{Metric, SquaredL2},
};

use crate::{
    graph::{
        self, DiskANNIndex,
        index::{QueryLabelProvider, QueryVisitDecision},
        search::{Knn, MultihopSearch},
        search_output_buffer,
        test::{provider as test_provider, search_utils, synthetic::Grid},
    },
    neighbor::{BackInserter, Neighbor},
    test::tokio::current_thread_runtime,
};

/// Set up a 3D grid index using the test provider.
fn setup_grid_index(grid_size: usize) -> Arc<DiskANNIndex<test_provider::Provider>> {
    let grid = Grid::Three;
    let provider = test_provider::Provider::grid(grid, grid_size).unwrap();

    let index_config = graph::config::Builder::new(
        provider.max_degree(),
        graph::config::MaxDegree::same(),
        100,
        Metric::L2.into(),
    )
    .build()
    .unwrap();

    Arc::new(DiskANNIndex::new(index_config, provider, None))
}

/////////////
// Filters //
/////////////

/// Rejects all candidates via `on_visit`. Only allows specified IDs through `is_match`.
#[derive(Debug)]
struct RejectAllFilter {
    allowed_in_results: HashSet<u32>,
}

impl RejectAllFilter {
    fn only<I: IntoIterator<Item = u32>>(ids: I) -> Self {
        Self {
            allowed_in_results: ids.into_iter().collect(),
        }
    }
}

impl QueryLabelProvider<u32> for RejectAllFilter {
    fn is_match(&self, vec_id: u32) -> bool {
        self.allowed_in_results.contains(&vec_id)
    }

    fn on_visit(&self, _neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
        QueryVisitDecision::Reject
    }
}

/// Tracks visited IDs and terminates when the target is found.
#[derive(Debug)]
struct TerminatingFilter {
    target: u32,
    hits: Mutex<Vec<u32>>,
}

impl TerminatingFilter {
    fn new(target: u32) -> Self {
        Self {
            target,
            hits: Mutex::new(Vec::new()),
        }
    }

    fn hits(&self) -> Vec<u32> {
        self.hits
            .lock()
            .expect("mutex should not be poisoned")
            .clone()
    }
}

impl QueryLabelProvider<u32> for TerminatingFilter {
    fn is_match(&self, vec_id: u32) -> bool {
        vec_id == self.target
    }

    fn on_visit(&self, neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
        self.hits
            .lock()
            .expect("mutex should not be poisoned")
            .push(neighbor.id);
        if neighbor.id == self.target {
            QueryVisitDecision::Terminate
        } else {
            QueryVisitDecision::Accept(neighbor)
        }
    }
}

/// Accepts all candidates via `is_match`, but allows blocking and distance adjustment.
#[derive(Debug)]
struct CallbackFilter {
    blocked: u32,
    adjusted: u32,
    adjustment_factor: f32,
    metrics: Mutex<CallbackMetrics>,
}

#[derive(Debug, Clone, Default)]
struct CallbackMetrics {
    total_visits: usize,
    rejected_count: usize,
    adjusted_count: usize,
    visited_ids: Vec<u32>,
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

    fn hits(&self) -> Vec<u32> {
        self.metrics
            .lock()
            .expect("mutex should not be poisoned")
            .visited_ids
            .clone()
    }

    fn metrics(&self) -> CallbackMetrics {
        self.metrics
            .lock()
            .expect("mutex should not be poisoned")
            .clone()
    }
}

impl QueryLabelProvider<u32> for CallbackFilter {
    fn is_match(&self, _: u32) -> bool {
        true
    }

    fn on_visit(&self, neighbor: Neighbor<u32>) -> QueryVisitDecision<u32> {
        let mut metrics = self.metrics.lock().expect("mutex should not be poisoned");
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

/// Accepts all IDs but only allows even IDs in results.
#[derive(Debug)]
struct EvenFilter;

impl QueryLabelProvider<u32> for EvenFilter {
    fn is_match(&self, id: u32) -> bool {
        id.is_multiple_of(2)
    }
}

/// Accepts N visits then terminates (without accepting the terminating visit).
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
    fn is_match(&self, _: u32) -> bool {
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

///////////
// Tests //
///////////

#[test]
fn reject_all_returns_zero_results() {
    let rt = current_thread_runtime();
    let grid_size = 4;
    let index = setup_grid_index(grid_size);

    let query = vec![grid_size as f32; 3];
    let mut ids = vec![0u32; 10];
    let mut distances = vec![0.0f32; 10];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    // The start point (u32::MAX) is inserted directly into the candidate set without
    // going through on_visit, so it must be excluded via is_match as well.
    let filter = RejectAllFilter::only([]);

    let search_params = Knn::new_default(10, 20).unwrap();
    let multihop = MultihopSearch::new(search_params, &filter);
    let stats = rt
        .block_on(index.search(
            multihop,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut buffer,
        ))
        .unwrap();

    // The start point (u32::MAX) is seeded directly into the candidate set and bypasses
    // both is_match and on_visit. It may appear in results. All non-start-point results
    // should be zero since on_visit rejects everything.
    let non_start_results = (0..stats.result_count as usize)
        .filter(|&i| ids[i] != u32::MAX)
        .count();

    assert_eq!(
        non_start_results, 0,
        "rejecting all via on_visit should result in zero non-start-point results"
    );
}

#[test]
fn early_termination() {
    let rt = current_thread_runtime();
    let grid_size = 5;
    let index = setup_grid_index(grid_size);
    let num_points = Grid::Three.num_points(grid_size);

    let query = vec![grid_size as f32; 3];
    let mut ids = vec![0u32; 10];
    let mut distances = vec![0.0f32; 10];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    let target = (num_points / 2) as u32;
    let filter = TerminatingFilter::new(target);

    let search_params = Knn::new_default(10, 40).unwrap();
    let multihop = MultihopSearch::new(search_params, &filter);
    let stats = rt
        .block_on(index.search(
            multihop,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut buffer,
        ))
        .unwrap();

    let hits = filter.hits();

    assert!(
        hits.contains(&target),
        "search should have visited the target"
    );
    assert!(
        stats.result_count >= 1,
        "should have at least one result (the target)"
    );
}

#[test]
fn distance_adjustment_affects_ranking() {
    let rt = current_thread_runtime();
    let grid_size = 4;
    let index = setup_grid_index(grid_size);
    let num_points = Grid::Three.num_points(grid_size);

    let query = vec![0.0f32; 3]; // Query at origin

    // Baseline: run with EvenFilter (no distance adjustment).
    let mut baseline_ids = vec![0u32; 10];
    let mut baseline_distances = vec![0.0f32; 10];
    let mut baseline_buffer =
        search_output_buffer::IdDistance::new(&mut baseline_ids, &mut baseline_distances);

    let search_params = Knn::new_default(10, 20).unwrap();
    let multihop = MultihopSearch::new(search_params, &EvenFilter);
    let baseline_stats = rt
        .block_on(index.search(
            multihop,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut baseline_buffer,
        ))
        .unwrap();

    // Adjusted: boost a far-away point by shrinking its distance.
    let boosted_point = (num_points - 2) as u32;
    let filter = CallbackFilter::new(u32::MAX, boosted_point, 0.01);

    // Need a fresh index for fair comparison (metrics are accumulated).
    let index = setup_grid_index(grid_size);

    let mut adjusted_ids = vec![0u32; 10];
    let mut adjusted_distances = vec![0.0f32; 10];
    let mut adjusted_buffer =
        search_output_buffer::IdDistance::new(&mut adjusted_ids, &mut adjusted_distances);

    let search_params = Knn::new_default(10, 20).unwrap();
    let multihop = MultihopSearch::new(search_params, &filter);
    let adjusted_stats = rt
        .block_on(index.search(
            multihop,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut adjusted_buffer,
        ))
        .unwrap();

    assert!(
        baseline_stats.result_count > 0,
        "baseline should have results"
    );
    assert!(
        adjusted_stats.result_count > 0,
        "adjusted should have results"
    );

    let boosted_in_baseline = baseline_ids
        .iter()
        .take(baseline_stats.result_count as usize)
        .position(|&id| id == boosted_point);
    let boosted_in_adjusted = adjusted_ids
        .iter()
        .take(adjusted_stats.result_count as usize)
        .position(|&id| id == boosted_point);

    if filter.hits().contains(&boosted_point) {
        assert!(
            boosted_in_adjusted.is_some(),
            "boosted point should appear in adjusted results when visited"
        );
        if let (Some(baseline_pos), Some(adjusted_pos)) = (boosted_in_baseline, boosted_in_adjusted)
        {
            assert!(
                adjusted_pos <= baseline_pos,
                "boosted point should rank equal or better after distance reduction"
            );
        }
    }
}

#[test]
fn terminate_stops_traversal() {
    let rt = current_thread_runtime();
    let grid_size = 5;
    let index = setup_grid_index(grid_size);

    let query = vec![grid_size as f32; 3];
    let mut ids = vec![0u32; 10];
    let mut distances = vec![0.0f32; 10];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    let max_visits = 5;
    let filter = TerminateAfterN::new(max_visits);

    // Large L to ensure we'd visit more without termination.
    let search_params = Knn::new_default(10, 100).unwrap();
    let multihop = MultihopSearch::new(search_params, &filter);
    let _stats = rt
        .block_on(index.search(
            multihop,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut buffer,
        ))
        .unwrap();

    // Allow some slack for beam expansion.
    assert!(
        filter.visit_count() <= max_visits + 10,
        "search should have terminated early, got {} visits",
        filter.visit_count()
    );
}

#[test]
fn even_filtering_multihop() {
    let rt = current_thread_runtime();
    let grid = Grid::Three;
    let grid_size = 7;
    let index = setup_grid_index(grid_size);

    let query = vec![grid_size as f32; 3];
    let filter = EvenFilter;

    // Compute brute-force groundtruth, filtered to even IDs only.
    let data = grid.data(grid_size);
    let gt = {
        let mut gt =
            search_utils::groundtruth(data.as_view(), &query, |a, b| SquaredL2::evaluate(a, b));
        gt.retain(|n| filter.is_match(n.id));
        gt.sort_unstable_by(|a, b| a.cmp(b).reverse());
        gt
    };

    let k = 20;
    let l = 40;
    let mut gt_clone = gt.clone();

    let search_params = Knn::new_default(k, l).unwrap();
    let multihop = MultihopSearch::new(search_params, &filter);

    let mut neighbors = vec![Neighbor::<u32>::default(); k];
    let stats = rt
        .block_on(index.search(
            multihop,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut BackInserter::new(&mut neighbors),
        ))
        .unwrap();

    // Verify each result matches the groundtruth (filtered to even IDs).
    for (i, n) in neighbors
        .iter()
        .enumerate()
        .take(stats.result_count as usize)
    {
        match search_utils::is_match(&gt_clone, *n, 0.0) {
            Some(position) => {
                gt_clone.remove(position);
            }
            None => {
                panic!(
                    "result {} (id={}, dist={}) does not match groundtruth",
                    i, n.id, n.distance
                );
            }
        }
    }
}

#[test]
fn callback_enforces_filtering() {
    let rt = current_thread_runtime();
    let grid = Grid::Three;
    let grid_size = 5;
    let num_points = grid.num_points(grid_size);
    let index = setup_grid_index(grid_size);

    let query = vec![grid_size as f32; 3];

    // Compute brute-force groundtruth for validation.
    let data = grid.data(grid_size);
    let baseline_gt = {
        let mut gt =
            search_utils::groundtruth(data.as_view(), &query, |a, b| SquaredL2::evaluate(a, b));
        gt.sort_unstable_by(|a, b| a.cmp(b).reverse());
        gt
    };

    let blocked = (num_points - 2) as u32;
    let adjusted = (num_points - 1) as u32;

    assert!(
        baseline_gt.iter().any(|n| n.id == blocked),
        "blocked candidate must exist in groundtruth"
    );

    let baseline_adjusted_distance = baseline_gt
        .iter()
        .find(|n| n.id == adjusted)
        .expect("adjusted node should exist in groundtruth")
        .distance;

    let k = 20;
    let l = 40;
    let to_check = 10;
    let filter = CallbackFilter::new(blocked, adjusted, 0.5);

    let search_params = Knn::new_default(k, l).unwrap();
    let multihop = MultihopSearch::new(search_params, &filter);

    let mut ids = vec![0u32; k];
    let mut distances = vec![0.0f32; k];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    let stats = rt
        .block_on(index.search(
            multihop,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut buffer,
        ))
        .unwrap();

    let callback_metrics = filter.metrics();

    // Validate enough results were returned.
    assert!(
        stats.result_count >= to_check as u32,
        "expected at least {} results, got {}",
        to_check,
        stats.result_count
    );

    // Validate callback was invoked and tracked the blocked candidate.
    assert!(
        callback_metrics.total_visits > 0,
        "callback should have been invoked at least once"
    );
    assert!(
        filter.hits().contains(&blocked),
        "callback must evaluate the blocked candidate (visited {} candidates)",
        callback_metrics.total_visits
    );
    assert_eq!(
        callback_metrics.rejected_count, 1,
        "exactly one candidate (blocked={}) should be rejected",
        blocked
    );

    // Validate blocked candidate is excluded from results.
    let produced = stats.result_count as usize;
    let inspected = produced.min(to_check);
    assert!(
        !ids.iter().take(inspected).any(|&id| id == blocked),
        "blocked candidate {} should not appear in final results (found in: {:?})",
        blocked,
        &ids[..inspected]
    );

    // Validate distance adjustment was applied.
    assert!(
        callback_metrics.adjusted_count >= 1,
        "adjusted candidate {} should have been visited",
        adjusted
    );

    let adjusted_idx = ids
        .iter()
        .take(inspected)
        .position(|&id| id == adjusted)
        .expect("adjusted candidate should be present in results");
    let expected_distance = baseline_adjusted_distance * 0.5;
    assert!(
        (distances[adjusted_idx] - expected_distance).abs() < 1e-5,
        "callback should adjust distances before ranking: \
         expected {:.6}, got {:.6} (baseline: {:.6}, factor: 0.5)",
        expected_distance,
        distances[adjusted_idx],
        baseline_adjusted_distance
    );
}
