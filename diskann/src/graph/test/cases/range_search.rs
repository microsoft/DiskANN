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
        index::SearchStats,
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct RangeSearchBaseline {
    /// A description of what to expect, what trends to observe, and anything else
    /// a reviewer may need to either understand why this test is checked in or to validate
    /// any changes that occur in the checked-in file.
    pub(super) description: String,
    pub(super) grid_dims: u8,
    pub(super) grid_size: usize,
    pub(super) query: Vec<f32>,
    pub(super) radius: f32,
    pub(super) inner_radius: Option<f32>,
    pub(super) starting_l: usize,
    pub(super) results: Vec<(u32, f32)>,
    pub(super) comparisons: usize,
    pub(super) hops: usize,
    pub(super) result_count: usize,
    pub(super) range_search_second_round: bool,
}

impl RangeSearchBaseline {
    pub(super) fn new(
        range: &Range,
        results: &[Neighbor<u32>],
        stats: SearchStats,
        grid_dims: Grid,
        grid_size: usize,
        description: impl Into<String>,
        query: Vec<f32>,
    ) -> Self {
        Self {
            description: description.into(),
            grid_dims: grid_dims.dim(),
            grid_size,
            query,
            radius: range.radius(),
            inner_radius: range.inner_radius(),
            starting_l: range.starting_l().get(),
            results: results.iter().map(|n| (*n.id(), *n.distance())).collect(),
            comparisons: stats.cmps as usize,
            hops: stats.hops as usize,
            result_count: stats.result_count as usize,
            range_search_second_round: stats.range_search_second_round,
        }
    }
}

verbose_eq!(RangeSearchBaseline {
    description,
    grid_dims,
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

fn root() -> TestRoot {
    TestRoot::new("graph/test/cases/range_search")
}

pub(super) fn setup_grid_index(
    grid_size: usize,
    dims: Grid,
) -> Arc<DiskANNIndex<test_provider::Provider>> {
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

pub(super) fn setup_grid_index_and_default_query(
    grid_size: usize,
    dims: Grid,
) -> (Arc<DiskANNIndex<test_provider::Provider>>, Vec<f32>) {
    let index = setup_grid_index(grid_size, dims);
    let query = vec![grid_size as f32; dims.dim().into()];
    (index, query)
}

pub(super) fn assert_no_duplicates(results: &[Neighbor<u32>]) {
    let mut seen = std::collections::HashSet::new();
    for n in results {
        assert!(seen.insert(*n.id()), "duplicate result id {}", n.id());
    }
}

pub(super) fn assert_range_invariants(
    results: &[Neighbor<u32>],
    radius: f32,
    inner_radius: Option<f32>,
) {
    for n in results {
        assert!(
            *n.distance() <= radius,
            "result {} distance {} exceeds radius {}",
            n.id(),
            n.distance(),
            radius
        );
        if let Some(inner) = inner_radius {
            assert!(
                *n.distance() > inner,
                "result {} distance {} is within inner radius {}",
                n.id(),
                n.distance(),
                inner
            );
        }
    }
}

#[test]
fn basic_range_search() {
    let description = "Basic range search test to validate that the range \
     search returns results within the specified radius and that there are \
     no duplicate results.";

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
        description: description.to_string(),
        grid_dims: Grid::Three.dim(),
        grid_size,
        query: query.clone(),
        radius,
        inner_radius: None,
        starting_l,
        results: results.iter().map(|n| n.as_tuple()).collect(),
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
    let description = "Inner radius filtering test to validate that the \
    range search correctly excludes neighbors within the inner radius.";

    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("inner_radius_filtering");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 20.0;
    let inner_radius = 6.0; // exclude closest neighbors
    let starting_l = 32;

    let range_search = Range::builder(starting_l, radius)
        .inner_radius(Some(inner_radius))
        .build()
        .unwrap();
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
        description: description.to_string(),
        grid_dims: Grid::Three.dim(),
        grid_size,
        query: query.clone(),
        radius,
        inner_radius: Some(inner_radius),
        starting_l,
        results: results.iter().map(|n| n.as_tuple()).collect(),
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
    let description = "Two round search test to validate that a \
    low starting L with a large radius triggers a second round \
    of range search.";

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
        description: description.to_string(),
        grid_dims: Grid::Three.dim(),
        grid_size,
        query: query.clone(),
        radius,
        inner_radius: None,
        starting_l,
        results: results.iter().map(|n| n.as_tuple()).collect(),
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

#[test]
fn max_results_respected_and_second_round_triggered() {
    let description = "Two round search test to validate that max_results > \
    starting_l means a second round is triggered.";

    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("max_results_respected_and_second_round_triggered");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 1.0e9; // every point will be in range with this radius
    let starting_l = 4; // small set to trigger multiple rounds
    let max_results = 5; // max_returned greater than starting_l, so second round should be triggered

    let range_search = Range::builder(starting_l, radius)
        .max_returned(Some(max_results))
        .build()
        .unwrap();

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
        description: description.to_string(),
        grid_dims: Grid::Three.dim(),
        grid_size,
        query: query.clone(),
        radius,
        inner_radius: None,
        starting_l,
        results: results.iter().map(|n| n.as_tuple()).collect(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
        result_count: results.len(),
        range_search_second_round: stats.range_search_second_round,
    };

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert!(
        results.len() <= max_results,
        "result count {} exceeds max_results {}",
        results.len(),
        max_results
    );

    assert!(
        stats.range_search_second_round,
        "If max_results is respected, a second round should be triggered"
    );

    assert_range_invariants(&results, radius, None);
    assert_no_duplicates(&results);
}

/// The next two tests use a 5 by 5 grid indexed from zero where the start point and
/// query are [5,5,5]. The closest point to the query is thus [4,4,4] with
/// a squared distance of 3.0 from the query. Setting the radius to 3 will then
/// include just the start point and the closest point [4,4,4] in the in-range
/// results at the end of the first round of search. Thus, a second round can
/// only be triggered if the initial slack is .5 or less

#[test]
fn initial_slack_low_triggers_second_round() {
    let description = "Grid setup where radius of 3.0 includes exactly two points \
    in the in-range results at the end of the second round of search. Thus, initial \
    slack of .5 or lower should trigger a second round.";

    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("initial_slack_low_triggers_second_round");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 3.0;
    let starting_l = 4;
    let low_slack = 0.5;

    let range_search = Range::builder(starting_l, radius)
        .initial_slack(low_slack)
        .build()
        .unwrap();
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
        description: description.to_string(),
        grid_dims: Grid::Three.dim(),
        grid_size,
        query: query.clone(),
        radius,
        inner_radius: None,
        starting_l,
        results: results.iter().map(|n| n.as_tuple()).collect(),
        comparisons: stats.cmps as usize,
        hops: stats.hops as usize,
        result_count: results.len(),
        range_search_second_round: stats.range_search_second_round,
    };

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert!(
        stats.range_search_second_round,
        "low initial_slack ({}) should trigger second round",
        low_slack
    );

    assert_range_invariants(&results, radius, None);
    assert_no_duplicates(&results);
}

#[test]
fn initial_slack_high_avoids_second_round() {
    let description = "Grid setup where radius of 3.0 includes exactly two points \
    in the in-range results at the end of the second round of search. Thus, initial \
    slack of .51 or higher should avoid triggering a second round.";

    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("initial_slack_high_avoids_second_round");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 3.0;
    let starting_l = 4;
    let high_slack = 0.51;

    let range_search = Range::builder(starting_l, radius)
        .initial_slack(high_slack)
        .build()
        .unwrap();
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
        description: description.to_string(),
        grid_dims: Grid::Three.dim(),
        grid_size,
        query: query.clone(),
        radius,
        inner_radius: None,
        starting_l,
        results: results.iter().map(|n| n.as_tuple()).collect(),
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
fn higher_range_slack_finds_more_results() {
    use crate::graph::AdjacencyList;

    let mut test_root = root();
    let mut path = test_root.path();
    let mut path = path.push("higher_range_slack_finds_more_results");

    // Use grid coordinates, but connect them as start -> 0 <-> 1 <-> 3 <-> 2.
    // With query [0, 0], squared distances for IDs 0, 1, 2, 3 are 0, 1, 1, 2.
    // ID 2 is in range, but can only be reached through out-of-range ID 3.
    let index = super::helpers::setup_2d_square(
        vec![
            AdjacencyList::from_iter_untrusted([1]),
            AdjacencyList::from_iter_untrusted([0, 3]),
            AdjacencyList::from_iter_untrusted([3]),
            AdjacencyList::from_iter_untrusted([1, 2]),
            AdjacencyList::from_iter_untrusted([0]),
        ],
        2,
    );
    let rt = current_thread_runtime();
    let query = [0.0, 0.0];
    let radius = 1.0;
    // Keep the initial search too small to expand ID 3. The start point at
    // [0.5, 0.5] is in range and counts toward the second-round threshold.
    let starting_l = 2;

    let mut search = |case: &str, range_slack| {
        let name = path.push(case);
        let range = Range::builder(starting_l, radius)
            .range_slack(range_slack)
            .build()
            .unwrap();
        let mut results: Vec<Neighbor<u32>> = Vec::new();
        let stats = rt
            .block_on(index.search(
                range,
                &test_provider::Strategy::new(),
                &test_provider::Context::new(),
                query.as_slice(),
                &mut results,
            ))
            .unwrap();

        let baseline = RangeSearchBaseline {
            description: format!(
                "Range slack comparison on a 2 by 2 grid connected as \
                start -> 0 <-> 1 <-> 3 <-> 2. With range_slack={range_slack}, \
                slack 1.0 finds IDs 0 and 1, while slack 2.0 also finds ID 2 \
                by expanding out-of-range ID 3 without returning it."
            ),
            grid_dims: Grid::Two.dim(),
            grid_size: 2,
            query: query.to_vec(),
            radius,
            inner_radius: None,
            starting_l,
            results: results.iter().map(|n| n.as_tuple()).collect(),
            comparisons: stats.cmps as usize,
            hops: stats.hops as usize,
            result_count: results.len(),
            range_search_second_round: stats.range_search_second_round,
        };

        let expected = get_or_save_test_results(&name, &baseline);
        assert_eq_verbose!(expected, baseline);

        assert!(
            stats.range_search_second_round,
            "range_slack={range_slack} should exercise second-round expansion"
        );
        assert_range_invariants(&results, radius, None);
        assert_no_duplicates(&results);
        let mut results: Vec<_> = results.iter().map(|n| n.as_tuple()).collect();
        results.sort_unstable_by_key(|n| n.0);
        results
    };

    let low_slack_results = search("low_range_slack", 1.0);
    let high_slack_results = search("high_range_slack", 2.0);

    assert!(
        high_slack_results.len() > low_slack_results.len(),
        "higher range slack should find more in-range results"
    );
}
