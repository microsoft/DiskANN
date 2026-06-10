/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tests for inline filter search traversal behavior.
//!
//! These tests exercise end-to-end `index.search(InlineSearch { .. })` execution.

use diskann_vector::distance::Metric;
use std::collections::HashSet;
use std::sync::Arc;

use crate::{
    graph::{
        self, AdjacencyList,
        index::QueryLabelProvider,
        search::{AdaptiveL, InlineFilterSearch, Knn},
        search_output_buffer,
        test::provider as test_provider,
        test::synthetic::Grid,
    },
    test::{
        TestRoot,
        cmp::{assert_eq_verbose, verbose_eq},
        get_or_save_test_results,
        tokio::current_thread_runtime,
    },
};

use super::multihop::{BlockAndAdjust, EvenFilter, build_1d_provider, setup_grid_index};

fn root() -> TestRoot {
    TestRoot::new("graph/test/cases/inline")
}

/// Baseline struct for end-to-end inline search results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct InlineBaseline {
    query: Vec<f32>,
    k: usize,
    l: usize,
    result_count: usize,
    results: Vec<(u32, f32)>,
    comparisons: usize,
    hops: usize,
}

verbose_eq!(InlineBaseline {
    query,
    k,
    l,
    result_count,
    results,
    comparisons,
    hops,
});

// Topology (3 levels below the start):
//                    0(start)                    level 0, coord 0.0, label 0
//                 /            \
//                1              2               level 1, coord 0.0, label 0
//              /   \          /   \
//             3     4        5     6            level 2, coord 1.0, label 0
//            / \   / \      / \   / \
//           7  8  9  10    11 12 13 14          level 3, coord 2.0, label 1
fn build_three_level_labeled_provider() -> test_provider::Provider {
    let max_degree = 3;
    let start_id = 0u32;

    let config = test_provider::Config::new(
        Metric::L2,
        max_degree,
        test_provider::StartPoint::new(start_id, vec![0.0]),
    )
    .unwrap();

    let start_neighbors = std::iter::once((start_id, AdjacencyList::from_iter_untrusted([1, 2])));

    let points = vec![
        // level 1: coord 0.0, label 0
        (1, vec![0.0], AdjacencyList::from_iter_untrusted([0, 3, 4])),
        (2, vec![0.0], AdjacencyList::from_iter_untrusted([0, 5, 6])),
        // level 2: coord 1.0, label 0
        (3, vec![1.0], AdjacencyList::from_iter_untrusted([1, 7, 8])),
        (4, vec![1.0], AdjacencyList::from_iter_untrusted([1, 9, 10])),
        (
            5,
            vec![1.0],
            AdjacencyList::from_iter_untrusted([2, 11, 12]),
        ),
        (
            6,
            vec![1.0],
            AdjacencyList::from_iter_untrusted([2, 13, 14]),
        ),
        // level 3 (final): coord 2.0, label 1
        (7, vec![2.0], AdjacencyList::from_iter_untrusted([3])),
        (8, vec![2.0], AdjacencyList::from_iter_untrusted([3])),
        (9, vec![2.0], AdjacencyList::from_iter_untrusted([4])),
        (10, vec![2.0], AdjacencyList::from_iter_untrusted([4])),
        (11, vec![2.0], AdjacencyList::from_iter_untrusted([5])),
        (12, vec![2.0], AdjacencyList::from_iter_untrusted([5])),
        (13, vec![2.0], AdjacencyList::from_iter_untrusted([6])),
        (14, vec![2.0], AdjacencyList::from_iter_untrusted([6])),
    ];

    test_provider::Provider::new_from(config, start_neighbors, points).unwrap()
}

#[derive(Debug)]
struct LevelLabelProvider;

impl LevelLabelProvider {
    fn new() -> Self {
        Self
    }

    fn label_of(id: u32) -> u8 {
        match id {
            0..=6 => 0,  // start + non-final levels
            7..=14 => 1, // final level only
            _ => 255,    // unknown id
        }
    }
}

impl QueryLabelProvider<u32> for LevelLabelProvider {
    fn is_match(&self, id: u32) -> bool {
        Self::label_of(id) == 1
    }
}

#[derive(Debug)]
struct EvenlyDistributedIdFilter {
    matching_ids: HashSet<u32>,
}

impl EvenlyDistributedIdFilter {
    fn new(total_points: usize, matching_points: usize) -> Self {
        assert!(matching_points > 0, "matching_points must be > 0");
        assert!(
            matching_points <= total_points,
            "matching_points must be <= total_points"
        );

        if matching_points == total_points {
            return Self {
                matching_ids: (0..total_points as u32).collect(),
            };
        }

        // Spread selected IDs across the full ID range using midpoint bucket sampling.
        let mut matching_ids = HashSet::with_capacity(matching_points);
        for i in 0..matching_points {
            let id = ((2 * i + 1) * total_points / (2 * matching_points)) as u32;
            matching_ids.insert(id.min((total_points - 1) as u32));
        }

        Self { matching_ids }
    }
}

impl QueryLabelProvider<u32> for EvenlyDistributedIdFilter {
    fn is_match(&self, id: u32) -> bool {
        self.matching_ids.contains(&id)
    }
}

fn build_three_level_index() -> std::sync::Arc<graph::DiskANNIndex<test_provider::Provider>> {
    let provider = build_three_level_labeled_provider();
    let index_config =
        graph::config::Builder::new(3, graph::config::MaxDegree::same(), 32, Metric::L2.into())
            .build()
            .unwrap();
    std::sync::Arc::new(graph::DiskANNIndex::new(index_config, provider, None))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct InlineFilterBaseline {
    grid_size: usize,
    matching_points: usize,
    query: Vec<f32>,
    k: usize,
    l: usize,
    result_count: usize,
    result_ids: Vec<u32>,
    result_distances: Vec<f32>,
    comparisons: usize,
    hops: usize,
}

verbose_eq!(InlineFilterBaseline {
    grid_size,
    matching_points,
    query,
    k,
    l,
    result_count,
    result_ids,
    result_distances,
    comparisons,
    hops,
});

#[allow(clippy::too_many_arguments)]
fn run_inline_on_grid(
    index: &Arc<graph::DiskANNIndex<test_provider::Provider>>,
    filter: &dyn QueryLabelProvider<u32>,
    grid_size: usize,
    matching_points: usize,
    query: &[f32],
    k: usize,
    l: usize,
    adaptive_l: Option<AdaptiveL>,
) -> InlineFilterBaseline {
    let rt = current_thread_runtime();
    let inline = InlineFilterSearch::new(Knn::new_default(k, l).unwrap(), filter, adaptive_l);

    let mut ids = vec![0u32; k];
    let mut distances = vec![0.0f32; k];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    let stats = rt
        .block_on(index.search(
            inline,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query,
            &mut buffer,
        ))
        .unwrap();

    let result_count = stats.result_count as usize;
    InlineFilterBaseline {
        grid_size,
        matching_points,
        query: query.to_vec(),
        k,
        l,
        result_count,
        result_ids: ids[..result_count].to_vec(),
        result_distances: distances[..result_count].to_vec(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
    }
}

fn assert_inline_large_grid_with_even_filter(
    test_name: &str,
    matching_points: usize,
    adaptive_l: Option<AdaptiveL>,
) {
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push(test_name);

    let grid_size = 10; // 10^3 = 1000 points
    let total_points = Grid::Three.num_points(grid_size);
    assert_eq!(total_points, 1000, "expected a 1000-point 3D grid");

    let index = setup_grid_index(grid_size);
    let filter = EvenlyDistributedIdFilter::new(total_points, matching_points);

    let query = [10.0f32, 10.0, 10.0];

    // Keep l_search tiny so non-adaptive search is strongly constrained.
    // Knn requires l >= k, so we keep both at 1.
    let k = 1;
    let l = 1;

    let baseline = run_inline_on_grid(
        &index,
        &filter,
        grid_size,
        matching_points,
        &query,
        k,
        l,
        adaptive_l,
    );

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    for id in baseline.result_ids {
        assert!(
            filter.is_match(id),
            "returned id {} must satisfy the evenly-distributed filter",
            id
        );
    }
}

#[test]
fn inline_search_returns_only_final_level_matches() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("inline_search_returns_only_final_level_matches");

    let index = build_three_level_index();

    let filter = LevelLabelProvider::new();
    let k = 8;
    let l = 32;
    let inline = InlineFilterSearch::new(Knn::new_default(k, l).unwrap(), &filter, None);

    let mut ids = vec![0u32; k];
    let mut distances = vec![0.0f32; k];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    let stats = rt
        .block_on(index.search(
            inline,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            [2.0f32].as_slice(),
            &mut buffer,
        ))
        .unwrap();

    let result_count = stats.result_count as usize;
    let baseline = InlineFilterBaseline {
        grid_size: 0,
        matching_points: 0,
        query: vec![2.0f32],
        k,
        l,
        result_count,
        result_ids: ids[..result_count].to_vec(),
        result_distances: distances[..result_count].to_vec(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
    };

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    let results = ids[..stats.result_count as usize].iter().copied();

    assert!(stats.result_count > 0, "should return final-level matches");
    for id in results {
        assert!(
            (7..=14).contains(&id),
            "inline search should only return final-level nodes, got {}",
            id
        );
    }
}

#[test]
fn inline_search_three_level_no_adaptive_l_with_l1_finds_no_matches() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("inline_search_three_level_no_adaptive_l_with_l1_finds_no_matches");

    let index = build_three_level_index();

    let filter = LevelLabelProvider::new();
    let k = 1;
    let l = 1;
    let inline = InlineFilterSearch::new(Knn::new_default(k, l).unwrap(), &filter, None);

    let mut ids = vec![0u32; k];
    let mut distances = vec![0.0f32; k];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    let stats = rt
        .block_on(index.search(
            inline,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            [0.0f32].as_slice(),
            &mut buffer,
        ))
        .unwrap();

    let result_count = stats.result_count as usize;
    let baseline = InlineFilterBaseline {
        grid_size: 0,
        matching_points: 0,
        query: vec![0.0f32],
        k,
        l,
        result_count,
        result_ids: ids[..result_count].to_vec(),
        result_distances: distances[..result_count].to_vec(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
    };

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert_eq!(
        stats.result_count, 0,
        "with l_search=1 and no adaptive L, search should not reach final-level matches"
    );
}

#[test]
fn inline_search_three_level_adaptive_l_with_l1_finds_matches() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("inline_search_three_level_adaptive_l_with_l1_finds_matches");

    let index = build_three_level_index();

    let filter = LevelLabelProvider::new();
    let k = 1;
    let l = 1;
    let adaptive_l = AdaptiveL::new(1, 16.0).unwrap();
    let inline =
        InlineFilterSearch::new(Knn::new_default(k, l).unwrap(), &filter, Some(adaptive_l));

    let mut ids = vec![0u32; k];
    let mut distances = vec![0.0f32; k];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    let stats = rt
        .block_on(index.search(
            inline,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            [0.0f32].as_slice(),
            &mut buffer,
        ))
        .unwrap();

    let result_count = stats.result_count as usize;
    let baseline = InlineFilterBaseline {
        grid_size: 0,
        matching_points: 0,
        query: vec![0.0f32],
        k,
        l,
        result_count,
        result_ids: ids[..result_count].to_vec(),
        result_distances: distances[..result_count].to_vec(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
    };

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert!(
        stats.result_count > 0,
        "adaptive L should expand search enough to find final-level matches"
    );

    let results = ids[..stats.result_count as usize].iter().copied();
    for id in results {
        assert!(
            (7..=14).contains(&id),
            "adaptive inline search should only return final-level nodes, got {}",
            id
        );
    }
}

#[test]
fn inline_adaptive_l_large_grid_1_matching_point() {
    assert_inline_large_grid_with_even_filter(
        "inline_adaptive_l_large_grid_1_matching_point",
        1,
        Some(AdaptiveL::new(100, 16.0).unwrap()),
    );
}

#[test]
fn inline_adaptive_l_large_grid_10_matching_points() {
    assert_inline_large_grid_with_even_filter(
        "inline_adaptive_l_large_grid_10_matching_points",
        10,
        Some(AdaptiveL::new(100, 16.0).unwrap()),
    );
}

#[test]
fn inline_adaptive_l_large_grid_100_matching_points() {
    assert_inline_large_grid_with_even_filter(
        "inline_adaptive_l_large_grid_100_matching_points",
        100,
        Some(AdaptiveL::new(100, 16.0).unwrap()),
    );
}

#[test]
fn inline_large_grid_1_matching_point() {
    assert_inline_large_grid_with_even_filter("inline_large_grid_1_matching_point", 1, None);
}

#[test]
fn inline_large_grid_10_matching_points() {
    assert_inline_large_grid_with_even_filter("inline_large_grid_10_matching_points", 10, None);
}

#[test]
fn inline_large_grid_100_matching_points() {
    assert_inline_large_grid_with_even_filter("inline_large_grid_100_matching_points", 100, None);
}

#[test]
fn inline_search_reaches_matches_through_non_matching_nodes() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("inline_search_reaches_matches_through_non_matching_nodes");

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

    let index = std::sync::Arc::new(graph::DiskANNIndex::new(index_config, provider, None));
    let filter = EvenFilter;

    let k = 5;
    let l = 20;
    let search_params = Knn::new_default(k, l).unwrap();
    let inline = InlineFilterSearch::new(search_params, &filter, None);

    let mut ids = vec![0u32; k];
    let mut distances = vec![0.0f32; k];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    let stats = rt
        .block_on(index.search(
            inline,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            [2.0f32].as_slice(),
            &mut buffer,
        ))
        .unwrap();

    let result_count = stats.result_count as usize;
    let baseline = InlineBaseline {
        query: vec![2.0f32],
        k,
        l,
        result_count,
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

    let result_ids: Vec<u32> = ids[..stats.result_count as usize].to_vec();
    assert!(result_ids.contains(&2), "node 2 should be discoverable");
    assert!(result_ids.contains(&4), "node 4 should be discoverable");
    for id in result_ids {
        assert_eq!(id % 2, 0, "all inline results must match filter");
    }
}

#[test]
fn inline_callback_filtering_grid() {
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct InlineCallbackBaseline {
        grid_size: usize,
        query: Vec<f32>,
        k: usize,
        l: usize,
        blocked: u32,
        adjusted: u32,
        factor: f32,
        result_count: usize,
        results: Vec<(u32, f32)>,
        comparisons: usize,
        hops: usize,
        metrics: super::multihop::BlockAndAdjustMetrics,
    }

    verbose_eq!(InlineCallbackBaseline {
        grid_size,
        query,
        k,
        l,
        blocked,
        adjusted,
        factor,
        result_count,
        results,
        comparisons,
        hops,
        metrics,
    });

    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("inline_callback_filtering_grid");

    let grid_size = 5;
    let num_points = Grid::Three.num_points(grid_size);
    let index = setup_grid_index(grid_size);
    let query = vec![grid_size as f32; 3];

    let blocked = (num_points - 2) as u32;
    let adjusted = (num_points - 1) as u32;
    let filter = BlockAndAdjust::new(blocked, adjusted, 0.5);

    let k = 20;
    let l = 40;
    let search_params = Knn::new_default(k, l).unwrap();
    let inline = InlineFilterSearch::new(search_params, &filter, None);

    let mut ids = vec![0u32; k];
    let mut distances = vec![0.0f32; k];
    let mut buffer = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

    let stats = rt
        .block_on(index.search(
            inline,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut buffer,
        ))
        .unwrap();

    let result_count = stats.result_count as usize;
    let metrics = filter.metrics();
    let baseline = InlineCallbackBaseline {
        grid_size,
        query: query.clone(),
        k,
        l,
        blocked,
        adjusted,
        factor: 0.5,
        result_count,
        results: ids[..result_count]
            .iter()
            .zip(distances[..result_count].iter())
            .map(|(&id, &d)| (id, d))
            .collect(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
        metrics: metrics.clone(),
    };

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    let result_ids: Vec<u32> = ids[..stats.result_count as usize].to_vec();
    assert!(
        !result_ids.contains(&blocked),
        "blocked node {} must not appear in inline results",
        blocked
    );

    assert_eq!(metrics.rejected_count, 1, "exactly one rejection expected");
    assert!(
        metrics.adjusted_count >= 1,
        "adjusted node should have been visited"
    );
}
