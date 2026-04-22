/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tests for range-based search within a distance radius.
//!
//! Covers basic range search, inner radius filtering, two-round expansion,
//! and empty result handling. Integration tests use baselines for regression
//! protection.

use std::sync::Arc;

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, DiskANNIndex,
        search::Range,
        test::{provider as test_provider, synthetic::Grid},
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
    TestRoot::new("graph/test/cases/range_search")
}

fn setup_grid_index(grid_size: usize, dims: Grid) -> Arc<DiskANNIndex<test_provider::Provider>> {
    let provider = test_provider::Provider::grid(dims, grid_size).unwrap();

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

fn setup_grid_index_and_default_query(
    grid_size: usize,
    dims: Grid,
) -> (Arc<DiskANNIndex<test_provider::Provider>>, Vec<f32>) {
    let index = setup_grid_index(grid_size, dims);
    let query = vec![grid_size as f32; dims.dim().into()];
    (index, query)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RangeSearchBaseline {
    grid_size: usize,
    query: Vec<f32>,
    radius: f32,
    inner_radius: Option<f32>,
    starting_l: usize,
    results: Vec<(u32, f32)>,
    comparisons: usize,
    hops: usize,
    result_count: usize,
    range_search_second_round: bool,
}

verbose_eq!(RangeSearchBaseline {
    grid_size,
    query,
    radius,
    inner_radius,
    starting_l,
    results,
    comparisons,
    hops,
    result_count,
    range_search_second_round,
});

fn assert_no_duplicates(results: &[Neighbor<u32>]) {
    let mut seen = std::collections::HashSet::new();
    for n in results {
        assert!(seen.insert(n.id), "duplicate result id {}", n.id);
    }
}

fn assert_range_invariants(results: &[Neighbor<u32>], radius: f32, inner_radius: Option<f32>) {
    for n in results {
        assert!(
            n.distance <= radius,
            "result {} distance {} exceeds radius {}",
            n.id,
            n.distance,
            radius
        );
        if let Some(inner) = inner_radius {
            assert!(
                n.distance > inner,
                "result {} distance {} is within inner radius {}",
                n.id,
                n.distance,
                inner
            );
        }
    }
}

#[test]
fn basic_range_search() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("basic_range_search");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 12.0;
    let starting_l = 32;

    let range_search = Range::new(starting_l, radius).unwrap();
    let mut results: Vec<Neighbor<u32>> = Vec::new();

    let stats = rt
        .block_on(index.search(
            range_search,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut results,
        ))
        .unwrap();

    let baseline = RangeSearchBaseline {
        grid_size,
        query: query.clone(),
        radius,
        inner_radius: None,
        starting_l,
        results: results.iter().map(|n| (n.id, n.distance)).collect(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
        result_count: results.len(),
        range_search_second_round: stats.range_search_second_round,
    };

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert_range_invariants(&results, radius, None);
    assert_no_duplicates(&results);
}

#[test]
fn inner_radius_filtering() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("inner_radius_filtering");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 20.0;
    let inner_radius = 6.0; // exclude closest neighbors
    let starting_l = 32;

    let range_search =
        Range::with_options(None, starting_l, None, radius, Some(inner_radius), 1.0, 1.0).unwrap();
    let mut results: Vec<Neighbor<u32>> = Vec::new();

    let stats = rt
        .block_on(index.search(
            range_search,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut results,
        ))
        .unwrap();

    let baseline = RangeSearchBaseline {
        grid_size,
        query: query.clone(),
        radius,
        inner_radius: Some(inner_radius),
        starting_l,
        results: results.iter().map(|n| (n.id, n.distance)).collect(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
        result_count: results.len(),
        range_search_second_round: stats.range_search_second_round,
    };

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert_range_invariants(&results, radius, Some(inner_radius));
    assert_no_duplicates(&results);
}

#[test]
fn two_round_search() {
    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("two_round_search");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 50.0; // every point will be in range with this radius
    let starting_l = 4; // small set to trigger multiple rounds

    let range_search = Range::new(starting_l, radius).unwrap();
    let mut results: Vec<Neighbor<u32>> = Vec::new();

    let stats = rt
        .block_on(index.search(
            range_search,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut results,
        ))
        .unwrap();

    let baseline = RangeSearchBaseline {
        grid_size,
        query: query.clone(),
        radius,
        inner_radius: None,
        starting_l,
        results: results.iter().map(|n| (n.id, n.distance)).collect(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
        result_count: results.len(),
        range_search_second_round: stats.range_search_second_round,
    };

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert!(
        stats.range_search_second_round,
        "low starting_l with large radius should trigger a second round"
    );
    assert_range_invariants(&results, radius, None);
    assert_no_duplicates(&results);
}

#[test]
fn empty_results() {
    let rt = current_thread_runtime();

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 0.01; // too small and won't match any points on the grid
    let starting_l = 32;

    let range_search = Range::new(starting_l, radius).unwrap();
    let mut results: Vec<Neighbor<u32>> = Vec::new();

    let stats = rt
        .block_on(index.search(
            range_search,
            &test_provider::Strategy::new(),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut results,
        ))
        .unwrap();

    assert!(
        results.is_empty(),
        "no points should be within the radius {}",
        radius
    );
    assert!(
        !stats.range_search_second_round,
        "empty results shouldn't trigger a second round"
    );
}
