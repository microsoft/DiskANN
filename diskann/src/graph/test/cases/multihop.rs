/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tests for multihop search traversal behavior.
//!
//! Organized into two layers:
//! - **Unit tests** call `multihop_search_internal` directly on small hand-constructed
//!   graphs to test each decision path (Accept, Reject+two-hop, Terminate) in isolation.
//! - **Integration tests** go through `index.search(MultihopSearch{...})` end-to-end
//!   with baselines for regression protection.

use std::sync::Arc;

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, AdjacencyList, DiskANNIndex,
        ext::labeled,
        search::{
            Knn, MultihopSearch,
            record::NoopSearchRecord,
            scratch::{PriorityQueueConfiguration, SearchScratch},
        },
        search_output_buffer,
        test::provider as test_provider,
    },
    neighbor::Neighbor,
    test::{
        TestRoot,
        cmp::{assert_eq_verbose, verbose_eq},
        get_or_save_test_results,
        tokio::current_thread_runtime,
    },
};

fn root() -> TestRoot {
    TestRoot::new("graph/test/cases/multihop")
}

/////////////
// Filters //
/////////////

/// Accepts all candidates unconditionally.
#[derive(Debug)]
struct AcceptAll;

impl labeled::QueryLabelProvider<u32> for AcceptAll {
    fn is_match(&self, _: u32) -> bool {
        true
    }
}

/// Accepts all IDs but only allows even IDs in results.
#[derive(Debug)]
struct EvenFilter;

impl labeled::QueryLabelProvider<u32> for EvenFilter {
    fn is_match(&self, id: u32) -> bool {
        id.is_multiple_of(2)
    }
}

/// Rejects all candidates via `on_visit`.
#[derive(Debug)]
struct RejectAll;

impl labeled::QueryLabelProvider<u32> for RejectAll {
    fn is_match(&self, _: u32) -> bool {
        false
    }
}

////////////////////////////////////
// Shared helpers for small graphs //
////////////////////////////////////

/// Build a 1D provider with the given points and adjacency lists.
///
/// `start_pos` is the 1D position of the start node (id = `start_id`).
fn build_1d_provider(
    start_id: u32,
    start_pos: f32,
    start_neighbors: AdjacencyList<u32>,
    points: Vec<(u32, Vec<f32>, AdjacencyList<u32>)>,
    max_degree: usize,
) -> test_provider::Provider {
    let config = test_provider::Config::new(
        Metric::L2,
        max_degree,
        test_provider::StartPoint::new(start_id, vec![start_pos]),
    )
    .unwrap();

    test_provider::Provider::new_from(config, std::iter::once((start_id, start_neighbors)), points)
        .unwrap()
}

/// Call `multihop_search_internal` directly on a provider, bypassing the Search trait.
///
/// Returns (internal_stats, best_neighbors) where best_neighbors is the contents
/// of the scratch priority queue sorted by distance (nearest first).
fn run_internal(
    provider: &test_provider::Provider,
    query: &[f32],
    k: usize,
    l: usize,
    max_degree: usize,
    filter: &dyn labeled::QueryLabelProvider<u32>,
) -> (graph::index::InternalSearchStats, Vec<Neighbor<u32>>) {
    let rt = current_thread_runtime();
    rt.block_on(async {
        let mut accessor = labeled::FilteredAccessor::new(
            test_provider::Accessor::new(provider, query).unwrap(),
            filter,
        );

        let mut scratch = SearchScratch::new(PriorityQueueConfiguration::Fixed(l), Some(l));

        let stats = crate::graph::search::multihop_search::multihop_search_internal(
            max_degree,
            &Knn::new_default(k, l).unwrap(),
            &mut accessor,
            &mut scratch,
            &mut NoopSearchRecord::new(),
        )
        .await
        .unwrap();

        let mut results: Vec<_> = scratch.best.iter().collect();
        results.sort_unstable_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        (stats, results)
    })
}

//////////////////////////////////////////
// Unit tests: multihop_search_internal //
//////////////////////////////////////////

/// Graph: start(10) → 0 → 1 → 2, all matching (AcceptAll).
/// Query at 1.5 — should find all three nodes via normal one-hop expansion.
#[test]
fn accept_all_finds_all_nodes() {
    let start_id = 10u32;
    let provider = build_1d_provider(
        start_id,
        5.0,
        AdjacencyList::from_iter_untrusted([0, 1, 2]),
        vec![
            (
                0,
                vec![0.0],
                AdjacencyList::from_iter_untrusted([1, start_id]),
            ),
            (1, vec![1.0], AdjacencyList::from_iter_untrusted([0, 2])),
            (2, vec![2.0], AdjacencyList::from_iter_untrusted([1])),
        ],
        3,
    );

    let (stats, results) = run_internal(&provider, &[1.5], 3, 10, 3, &AcceptAll);

    let ids: Vec<u32> = results.iter().map(|n| n.id).collect();
    assert!(ids.contains(&0), "node 0 should be found");
    assert!(ids.contains(&1), "node 1 should be found");
    assert!(ids.contains(&2), "node 2 should be found");
    assert!(stats.cmps > 0, "should have computed distances");
}

/// Graph: start(10) → 1(odd) → 2(even), start → 3(odd) → 4(even), start → 0(even).
/// EvenFilter rejects odds via two-hop. Nodes 2 and 4 are only reachable through odds.
#[test]
fn reject_triggers_two_hop_expansion() {
    let start_id = 10u32;
    let provider = build_1d_provider(
        start_id,
        5.0,
        AdjacencyList::from_iter_untrusted([0, 1, 3]),
        vec![
            (
                0,
                vec![0.0],
                AdjacencyList::from_iter_untrusted([1, start_id]),
            ),
            (
                1,
                vec![1.0],
                AdjacencyList::from_iter_untrusted([0, 2, start_id]),
            ),
            (2, vec![2.0], AdjacencyList::from_iter_untrusted([1, 3])),
            (
                3,
                vec![3.0],
                AdjacencyList::from_iter_untrusted([0, 4, start_id]),
            ),
            (4, vec![4.0], AdjacencyList::from_iter_untrusted([3, 2])),
        ],
        4,
    );

    let filter = EvenFilter;
    let (stats, results) = run_internal(&provider, &[2.0], 5, 20, 4, &filter);

    let ids: Vec<u32> = results.iter().map(|n| n.id).collect();

    // Even nodes reachable only via two-hop through odd nodes.
    assert!(
        ids.contains(&2),
        "node 2 should be found via two-hop through node 1"
    );
    assert!(
        ids.contains(&4),
        "node 4 should be found via two-hop through node 3"
    );
    assert!(ids.contains(&0), "node 0 should be found directly");

    // All results in the best set should be even (matching).
    for n in &results {
        if n.id == start_id {
            continue;
        }
        assert!(
            n.id.is_multiple_of(2),
            "non-matching node {} should not be in best set",
            n.id
        );
    }

    assert!(stats.hops > 0, "should have expanded at least one hop");
}

#[test]
fn reject_all_yields_nothing() {
    let start_id = 10u32;
    let provider = build_1d_provider(
        start_id,
        0.0,
        AdjacencyList::from_iter_untrusted([0, 1]),
        vec![
            (
                0,
                vec![1.0],
                AdjacencyList::from_iter_untrusted([1, start_id]),
            ),
            (1, vec![2.0], AdjacencyList::from_iter_untrusted([0])),
        ],
        2,
    );

    let (_stats, results) = run_internal(&provider, &[0.5], 5, 10, 2, &RejectAll);

    // Nothing should be present in the result, not even the start point since it does not
    // satisfy the predicate.
    //
    // This mainly checks that search didn't explode.
    assert!(results.is_empty(), "all points are rejected");
}

///////////////////////////////
// Integration tests (E2E)   //
///////////////////////////////

/// Baseline struct for end-to-end multihop search results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MultihopBaseline {
    grid_size: usize,
    query: Vec<f32>,
    k: usize,
    l: usize,
    results: Vec<(u32, f32)>,
    comparisons: usize,
    hops: usize,
}

verbose_eq!(MultihopBaseline {
    grid_size,
    query,
    k,
    l,
    results,
    comparisons,
    hops,
});

/// Set up a 3D grid index using the test provider.
fn setup_grid_index(grid_size: usize) -> Arc<DiskANNIndex<test_provider::Provider>> {
    use crate::graph::test::synthetic::Grid;

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

/// Two-hop reachability through non-matching nodes, end-to-end with baseline.
///
/// Uses the same hand-constructed 1D graph as the unit test, but goes through
/// `index.search(MultihopSearch{...})` to also exercise post-processing.
#[test]
fn two_hop_reaches_through_non_matching() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("two_hop_reaches_through_non_matching");

    let start_id = 10u32;
    let provider = build_1d_provider(
        start_id,
        5.0,
        AdjacencyList::from_iter_untrusted([0, 1, 3]),
        vec![
            (
                0,
                vec![0.0],
                AdjacencyList::from_iter_untrusted([1, start_id]),
            ),
            (
                1,
                vec![1.0],
                AdjacencyList::from_iter_untrusted([0, 2, start_id]),
            ),
            (2, vec![2.0], AdjacencyList::from_iter_untrusted([1, 3])),
            (
                3,
                vec![3.0],
                AdjacencyList::from_iter_untrusted([0, 4, start_id]),
            ),
            (4, vec![4.0], AdjacencyList::from_iter_untrusted([3, 2])),
        ],
        4,
    );

    let index_config =
        graph::config::Builder::new(4, graph::config::MaxDegree::same(), 100, Metric::L2.into())
            .build()
            .unwrap();

    let index = Arc::new(DiskANNIndex::new(index_config, provider, None));
    let filter = EvenFilter;
    let query = vec![2.0f32];
    let k = 5;
    let l = 20;

    let search_params = Knn::new_default(k, l).unwrap();
    let multihop = MultihopSearch::new(search_params);

    let mut ids = vec![0u32; k];
    let mut distances = vec![0.0f32; k];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    let stats = rt
        .block_on(index.search(
            multihop,
            &labeled::Filtered::new(test_provider::Strategy::new(), &filter),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut buffer,
        ))
        .unwrap();

    let result_count = stats.result_count as usize;
    let baseline = MultihopBaseline {
        grid_size: 0, // hand-constructed, not grid-based
        query: query.clone(),
        k,
        l,
        results: ids[..result_count]
            .iter()
            .zip(distances[..result_count].iter())
            .map(|(&id, &d)| (id, d))
            .collect(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
    };

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    // Invariants that must hold regardless of baseline.
    let result_ids: Vec<u32> = baseline.results.iter().map(|(id, _)| *id).collect();
    assert!(
        result_ids.contains(&2),
        "node 2 must be found via two-hop through node 1"
    );
    assert!(
        result_ids.contains(&4),
        "node 4 must be found via two-hop through node 3"
    );
    for &(id, _) in &baseline.results {
        assert!(
            id.is_multiple_of(2),
            "all results must match the even filter, got id {}",
            id
        );
    }
}

/// Even-filtered multihop search on a 3D grid with baseline.
#[test]
fn even_filtering_grid() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("even_filtering_grid");

    let grid_size = 7;
    let index = setup_grid_index(grid_size);
    let query = vec![grid_size as f32; 3];
    let filter = EvenFilter;

    let k = 20;
    let l = 40;
    let search_params = Knn::new_default(k, l).unwrap();
    let multihop = MultihopSearch::new(search_params);

    let mut ids = vec![0u32; k];
    let mut distances = vec![0.0f32; k];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    let stats = rt
        .block_on(index.search(
            multihop,
            &labeled::Filtered::new(test_provider::Strategy::new(), &filter),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut buffer,
        ))
        .unwrap();

    let result_count = stats.result_count as usize;
    let baseline = MultihopBaseline {
        grid_size,
        query: query.clone(),
        k,
        l,
        results: ids[..result_count]
            .iter()
            .zip(distances[..result_count].iter())
            .map(|(&id, &d)| (id, d))
            .collect(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
    };

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    // Invariant: all returned IDs must be even.
    for &(id, _) in &baseline.results {
        assert!(
            id.is_multiple_of(2),
            "all results must match the even filter, got id {}",
            id
        );
    }
}
