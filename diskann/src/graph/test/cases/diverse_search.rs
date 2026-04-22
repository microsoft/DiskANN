/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Unit tests for diverse search at the `diskann` crate level.
//!
//! These tests exercise the [`Diverse`] search type through the
//! [`DiskANNIndex::search`] path using the in-memory test provider.
//!
//! Strict per-attribute cardinality guarantees (`diverse_results_k`) are tested
//! at the queue level in `diverse_priority_queue_test`. At the search level we
//! verify that:
//!
//! * Diverse search completes without error.
//! * Distances are non-negative and sorted.
//! * Results span multiple attribute groups (i.e. diversity is happening).
//! * The parameter accessors on [`Diverse`] and [`DiverseSearchParams`] work.

use std::{collections::HashMap, num::NonZeroUsize, sync::Arc};

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, DiskANNIndex, DiverseSearchParams,
        search::{Diverse, Knn},
        test::{provider as test_provider, synthetic::Grid},
    },
    neighbor::{AttributeValueProvider, BackInserter, Neighbor},
    test::tokio::current_thread_runtime,
};

/// A simple attribute provider that assigns attributes to vector IDs via a HashMap.
#[derive(Debug, Clone)]
struct TestAttributeProvider {
    attributes: HashMap<u32, u32>,
}

impl TestAttributeProvider {
    fn new() -> Self {
        Self {
            attributes: HashMap::new(),
        }
    }

    fn insert(&mut self, id: u32, attribute: u32) {
        self.attributes.insert(id, attribute);
    }
}

impl crate::provider::HasId for TestAttributeProvider {
    type Id = u32;
}

impl AttributeValueProvider for TestAttributeProvider {
    type Value = u32;

    fn get(&self, id: Self::Id) -> Option<Self::Value> {
        self.attributes.get(&id).copied()
    }
}

/// Build a grid-based index and an attribute provider that cyclically assigns
/// `num_labels` distinct labels to the grid points.
///
/// Returns `(index, attribute_provider)`.
fn setup_diverse_grid(
    grid: Grid,
    size: usize,
    num_labels: u32,
) -> (
    Arc<DiskANNIndex<test_provider::Provider>>,
    Arc<TestAttributeProvider>,
) {
    let provider = test_provider::Provider::grid(grid, size).unwrap();
    let num_points = grid.num_points(size) as u32;

    // Assign cyclic attributes: id % num_labels.
    let mut attr = TestAttributeProvider::new();
    for id in 0..num_points {
        attr.insert(id, id % num_labels);
    }
    // Provider::grid uses u32::MAX as the start point ID.
    attr.insert(u32::MAX, 0);

    let index_config = graph::config::Builder::new(
        provider.max_degree(),
        graph::config::MaxDegree::same(),
        100,
        Metric::L2.into(),
    )
    .build()
    .unwrap();

    let index = Arc::new(DiskANNIndex::new(index_config, provider, None));
    (index, Arc::new(attr))
}

/// Run a diverse search and return `(result_count, Vec<(id, distance)>)`.
fn run_diverse_search(
    index: &DiskANNIndex<test_provider::Provider>,
    query: &[f32],
    k: usize,
    l: usize,
    diverse_results_k: usize,
    attribute_provider: Arc<TestAttributeProvider>,
) -> (usize, Vec<(u32, f32)>) {
    run_diverse_search_with_beam(index, query, k, l, diverse_results_k, None, attribute_provider)
}

/// Run a diverse search with explicit beam width control.
fn run_diverse_search_with_beam(
    index: &DiskANNIndex<test_provider::Provider>,
    query: &[f32],
    k: usize,
    l: usize,
    diverse_results_k: usize,
    beam_width: Option<usize>,
    attribute_provider: Arc<TestAttributeProvider>,
) -> (usize, Vec<(u32, f32)>) {
    let rt = current_thread_runtime();

    let knn = Knn::new(k, l, beam_width).unwrap();
    let diverse_params = DiverseSearchParams::new(0, diverse_results_k, attribute_provider);
    let diverse = Diverse::new(knn, diverse_params);

    let mut neighbors = vec![Neighbor::<u32>::default(); k];
    let context = test_provider::Context::new();

    let stats = rt
        .block_on(index.search(
            diverse,
            &test_provider::Strategy::new(),
            &context,
            query,
            &mut BackInserter::new(&mut neighbors),
        ))
        .unwrap();

    let result_count = stats.result_count as usize;
    let results: Vec<_> = neighbors[..result_count]
        .iter()
        .map(|n| (n.id, n.distance))
        .collect();

    (result_count, results)
}

/// Count attribute occurrences in search results.
fn count_attributes(
    results: &[(u32, f32)],
    attr: &TestAttributeProvider,
) -> HashMap<u32, usize> {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for (id, _) in results {
        if let Some(val) = attr.get(*id) {
            *counts.entry(val).or_default() += 1;
        }
    }
    counts
}

// ─── Diverse parameter accessors ─────────────────────────────────────────────

#[test]
fn diverse_params_accessors() {
    let attr = Arc::new(TestAttributeProvider::new());
    let knn = Knn::new(10, 20, Some(2)).unwrap();
    let diverse_params = DiverseSearchParams::new(42, 3, attr.clone());
    let diverse = Diverse::new(knn, diverse_params);

    assert_eq!(diverse.inner().k_value(), NonZeroUsize::new(10).unwrap());
    assert_eq!(diverse.inner().l_value().get(), 20);
    assert_eq!(diverse.diverse_params().diverse_attribute_id, 42);
    assert_eq!(diverse.diverse_params().diverse_results_k, 3);
}

// ─── Basic diverse search ────────────────────────────────────────────────────

#[test]
fn diverse_search_returns_results() {
    let (index, attr) = setup_diverse_grid(Grid::Two, 4, 3);

    let query = vec![-0.7f32, 0.3];
    let (count, results) = run_diverse_search(&index, &query, 10, 20, 5, attr);

    assert!(count > 0, "diverse search should return at least one result");

    // Distances should be non-negative and sorted.
    for (_, dist) in &results {
        assert!(*dist >= 0.0, "distance should be non-negative, got {dist}");
    }
    for window in results.windows(2) {
        assert!(
            window[0].1 <= window[1].1,
            "distances should be sorted: {} > {}",
            window[0].1,
            window[1].1,
        );
    }
}

// ─── Diversity improves label spread ─────────────────────────────────────────

#[test]
fn diverse_search_produces_multiple_labels_2d() {
    // With 5 labels and a central query, results should span multiple labels.
    let (index, attr) = setup_diverse_grid(Grid::Two, 6, 5);

    let query = vec![3.1f32, 2.9];
    let (count, results) = run_diverse_search(&index, &query, 10, 30, 2, attr.clone());

    assert!(count > 0);

    let unique_labels: std::collections::HashSet<_> = results
        .iter()
        .filter_map(|(id, _)| attr.get(*id))
        .collect();

    assert!(
        unique_labels.len() >= 2,
        "expected at least 2 distinct labels in results, got {}",
        unique_labels.len(),
    );
}

#[test]
fn diverse_search_produces_multiple_labels_3d() {
    let (index, attr) = setup_diverse_grid(Grid::Three, 4, 4);

    let query = vec![1.3f32, 0.7, 2.1];
    let (count, results) = run_diverse_search(&index, &query, 10, 30, 1, attr.clone());

    assert!(count > 0, "3D diverse search should return results");

    let unique_labels: std::collections::HashSet<_> = results
        .iter()
        .filter_map(|(id, _)| attr.get(*id))
        .collect();

    assert!(
        unique_labels.len() >= 2,
        "3D: expected at least 2 distinct labels, got {}",
        unique_labels.len(),
    );
}

// ─── Diversity reduces per-label concentration ───────────────────────────────

#[test]
fn diverse_search_limits_per_label_count() {
    // With diverse_results_k = 2 and 5 labels on a large grid, the post-process
    // should reduce per-label concentration compared to the request size.
    let (index, attr) = setup_diverse_grid(Grid::Two, 8, 5);

    let query = vec![3.7f32, 4.1];
    let k = 10;
    let (count, results) = run_diverse_search(&index, &query, k, 30, 2, attr.clone());

    assert!(count > 0);

    let attr_counts = count_attributes(&results, &attr);

    // In a non-diverse search of k=10 with 5 labels, some labels would dominate
    // (all k nearest might be from 1-2 labels). With diversity, no single label
    // should take ALL the result slots.
    let max_per_label = attr_counts.values().copied().max().unwrap_or(0);
    assert!(
        max_per_label < count,
        "with diversity, no single label should monopolize all {} results (max was {})",
        count,
        max_per_label,
    );
}

// ─── Distances sorted across configurations ──────────────────────────────────

#[test]
fn diverse_search_sorted_distances_various_params() {
    let (index, attr) = setup_diverse_grid(Grid::Two, 6, 4);

    let configs = [
        (vec![0.1f32, 0.2], 5, 15, 1),
        (vec![3.3, 2.7], 10, 20, 2),
        (vec![5.5, 5.5], 8, 25, 3),
    ];

    for (query, k, l, div_k) in &configs {
        let (count, results) = run_diverse_search(&index, query, *k, *l, *div_k, attr.clone());

        assert!(
            count > 0,
            "search with k={k}, l={l}, div_k={div_k} returned no results",
        );

        for window in results.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "distances not sorted with k={k}, l={l}, div_k={div_k}: {} > {}",
                window[0].1,
                window[1].1,
            );
        }
    }
}

// ─── Beam width variation ────────────────────────────────────────────────────

#[test]
fn diverse_search_beam_width_variations() {
    let (index, attr) = setup_diverse_grid(Grid::Two, 5, 3);
    let query = vec![2.3f32, 1.7];

    // Exercise the same diverse search with beam widths 1, 2, 4
    // (mirrors the BEAM_WIDTHS pattern from grid_search.rs).
    for beam_width in [1, 2, 4] {
        let (count, results) = run_diverse_search_with_beam(
            &index,
            &query,
            10,
            20,
            2,
            Some(beam_width),
            attr.clone(),
        );

        assert!(
            count > 0,
            "beam_width={beam_width}: expected results",
        );

        // Result count should never exceed k.
        assert!(
            count <= 10,
            "beam_width={beam_width}: result_count {} exceeds k=10",
            count,
        );

        // Distances should remain sorted regardless of beam width.
        for window in results.windows(2) {
            assert!(
                window[0].1 <= window[1].1,
                "beam_width={beam_width}: distances not sorted: {} > {}",
                window[0].1,
                window[1].1,
            );
        }
    }
}

// ─── Empty attribute provider (all None) ─────────────────────────────────────

#[test]
fn diverse_search_all_attributes_none() {
    // When no vector has an attribute, every insert into the diverse queue
    // is skipped. The search should complete without error and return 0 results.
    let grid = Grid::Two;
    let size = 4;
    let provider = test_provider::Provider::grid(grid, size).unwrap();

    // Empty attribute provider — get() always returns None.
    let attr = Arc::new(TestAttributeProvider::new());

    let index_config = graph::config::Builder::new(
        provider.max_degree(),
        graph::config::MaxDegree::same(),
        100,
        Metric::L2.into(),
    )
    .build()
    .unwrap();

    let index = DiskANNIndex::new(index_config, provider, None);

    let query = vec![0.0f32; 2];
    let (count, _results) = run_diverse_search(&index, &query, 10, 20, 5, attr);

    assert_eq!(count, 0, "with no attributes, no results should be returned");
}

// ─── diverse_results_l boundary: div_l = 0 panics ────────────────────────────

#[test]
#[should_panic]
fn diverse_search_div_l_zero_panics() {
    // When diverse_results_k * l_value / k_value rounds to 0, local queues
    // have capacity 0. The first insert with an attribute triggers Case 2
    // (local queue "full" at size 0 == capacity 0), which indexes at
    // diverse_results_l - 1 = usize::MAX. This is an unchecked edge case.
    let (index, attr) = setup_diverse_grid(Grid::Two, 4, 3);

    let query = vec![0.0f32; 2];
    // k=10, l=5, div_k=1 → diverse_results_l = 1 * 5 / 10 = 0
    let _result = run_diverse_search(&index, &query, 10, 10, 0, attr);
}

// ─── diverse_results_l = 1: tightest non-degenerate local queues ─────────────

#[test]
fn diverse_search_div_l_one() {
    // k = l = 10, diverse_results_k = 1 → diverse_results_l = 1.
    // Each local queue holds exactly 1 item. This is the tightest non-degenerate
    // configuration and exercises Case 2 (local full) on every second insert
    // for a given attribute.
    let (index, attr) = setup_diverse_grid(Grid::Two, 5, 3);

    let query = vec![1.3f32, 0.7];
    let (count, results) = run_diverse_search(&index, &query, 10, 10, 1, attr.clone());

    assert!(count > 0, "div_l=1 search should still return results");
    assert!(count <= 10, "result_count should not exceed k");

    for window in results.windows(2) {
        assert!(
            window[0].1 <= window[1].1,
            "div_l=1: distances not sorted: {} > {}",
            window[0].1,
            window[1].1,
        );
    }
}

// ─── diverse_results_k > k: over-provisioned diversity ───────────────────────

#[test]
fn diverse_search_div_k_exceeds_k() {
    // diverse_results_k = 20 with k = 5. The diversity cap is wider than the
    // result count, so post_process should effectively be a no-op (no local
    // queue exceeds 20). This exercises the "no trimming needed" path through
    // post_process.
    let (index, attr) = setup_diverse_grid(Grid::Two, 5, 3);

    let query = vec![2.1f32, 1.9];
    let (count, results) = run_diverse_search(&index, &query, 5, 20, 20, attr.clone());

    assert!(count > 0);
    assert!(count <= 5, "result_count should not exceed k=5");

    for window in results.windows(2) {
        assert!(
            window[0].1 <= window[1].1,
            "div_k>k: distances not sorted: {} > {}",
            window[0].1,
            window[1].1,
        );
    }
}

// ─── result_count <= k invariant across many configurations ──────────────────

#[test]
fn diverse_search_result_count_bounded_by_k() {
    let (index, attr) = setup_diverse_grid(Grid::Two, 6, 4);

    let configs: &[(usize, usize, usize)] = &[
        // (k, l, div_k)
        (5, 15, 1),
        (5, 15, 5),
        (10, 10, 1),
        (10, 30, 3),
        (15, 15, 10),
        (3, 30, 1),
    ];

    let query = vec![2.5f32, 3.1];
    for &(k, l, div_k) in configs {
        let (count, _) = run_diverse_search(&index, &query, k, l, div_k, attr.clone());

        assert!(
            count <= k,
            "k={k}, l={l}, div_k={div_k}: result_count {count} exceeds k",
        );
    }
}
