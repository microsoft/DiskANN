/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tests for filtered range search using an always-true filter.
//!
//! These cover the filtered-range cases directly and validate the filtered-range
//! behavior against its own baselines.

use std::sync::Arc;

use super::range_search::{
    RangeSearchBaseline, assert_no_duplicates, assert_range_invariants,
    setup_grid_index_and_default_query,
};
use crate::{
    graph::{
        self, DiskANNIndex,
        ext::labeled,
        search::FilteredRange,
        test::{provider as test_provider, synthetic::Grid},
    },
    neighbor::Neighbor,
    test::{
        TestRoot, cmp::assert_eq_verbose, get_or_save_test_results, tokio::current_thread_runtime,
    },
};

fn root() -> TestRoot {
    TestRoot::new("graph/test/cases/filtered_range_search")
}

#[derive(Debug)]
struct AlwaysTrueFilter;

impl labeled::QueryLabelProvider<u32> for AlwaysTrueFilter {
    fn is_match(&self, _: u32) -> bool {
        true
    }
}

fn run_filtered_range_search(
    index: &Arc<DiskANNIndex<test_provider::Provider>>,
    query: &[f32],
    filtered_range: FilteredRange,
    filter: &impl labeled::QueryLabelProvider<u32>,
) -> (graph::index::SearchStats, Vec<Neighbor<u32>>) {
    let rt = current_thread_runtime();
    let mut results: Vec<Neighbor<u32>> = Vec::new();

    let stats = rt
        .block_on(index.search(
            filtered_range,
            &labeled::Filtered::new(test_provider::Strategy::new(), filter),
            &test_provider::Context::new(),
            query,
            &mut results,
        ))
        .unwrap();

    (stats, results)
}

#[derive(Debug)]
struct DivisibleByFourFilter;

impl labeled::QueryLabelProvider<u32> for DivisibleByFourFilter {
    fn is_match(&self, id: u32) -> bool {
        id.is_multiple_of(4)
    }
}

fn assert_divisible_by_four(results: &[Neighbor<u32>]) {
    for n in results {
        assert_eq!(
            n.id % 4,
            0,
            "result id {} does not satisfy divisible-by-4 filter",
            n.id
        );
    }
}

///////////////////////////////////////////////////////
// Tests with an always-true filter that validate    //
// that filtered range search gives correct answers  //
// and respects parameters such as inner_radius and  //
// max_results. Note that identical behavior to the  //
// unfiltered range search is NOT expected even when //
// using an always-true filter, because the filtered //
// initial search returns every predicate-satisfying //
// point it finds as opposed to only those in        //
// `scratch.best`.                                   //
///////////////////////////////////////////////////////

#[test]
fn basic_range_search() {
    let description = "Basic range search test with an always-true filter. Validates /
     that the filtered range search returns correct results without any duplicates.";

    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("basic_range_search");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 12.0;
    let starting_l = 32;
    let filter = AlwaysTrueFilter;
    let filtered_range = FilteredRange::new(starting_l, radius).unwrap();

    let (filtered_stats, filtered_results) =
        run_filtered_range_search(&index, query.as_slice(), filtered_range, &filter);

    let baseline = RangeSearchBaseline::new(
        &filtered_range.range(),
        &filtered_results,
        filtered_stats,
        grid_size,
        description,
        query.clone(),
    );

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert_range_invariants(&filtered_results, radius, None);
    assert_no_duplicates(&filtered_results);
}

#[test]
fn inner_radius_filtering() {
    let description = "Inner radius filtering test to validate that the \
    range search correctly excludes neighbors within the inner radius.";

    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("inner_radius_filtering");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 20.0;
    let inner_radius = 6.0;
    let starting_l = 32;
    let filter = AlwaysTrueFilter;

    let filtered_range = FilteredRange::builder(starting_l, radius)
        .inner_radius(Some(inner_radius))
        .build_filtered()
        .unwrap();

    let (filtered_stats, filtered_results) =
        run_filtered_range_search(&index, query.as_slice(), filtered_range, &filter);

    let baseline = RangeSearchBaseline::new(
        &filtered_range.range(),
        &filtered_results,
        filtered_stats,
        grid_size,
        description,
        query.clone(),
    );

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert_range_invariants(&filtered_results, radius, Some(inner_radius));
    assert_no_duplicates(&filtered_results);
}

#[test]
fn two_round_search() {
    let description = "Two round search test to validate that a /
    low starting L with a large radius triggers a second round /
    of range search.";

    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("two_round_search");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 50.0;
    let starting_l = 4;
    let filter = AlwaysTrueFilter;

    let filtered_range = FilteredRange::new(starting_l, radius).unwrap();

    let (filtered_stats, filtered_results) =
        run_filtered_range_search(&index, query.as_slice(), filtered_range, &filter);

    let baseline = RangeSearchBaseline::new(
        &filtered_range.range(),
        &filtered_results,
        filtered_stats,
        grid_size,
        description,
        query.clone(),
    );

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert!(
        filtered_stats.range_search_second_round,
        "low starting_l with large radius should trigger a second round"
    );
    assert_range_invariants(&filtered_results, radius, None);
    assert_no_duplicates(&filtered_results);
}

#[test]
fn empty_results() {
    let rt = current_thread_runtime();

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 0.01;
    let starting_l = 32;
    let filter = AlwaysTrueFilter;

    let filtered_range = FilteredRange::new(starting_l, radius).unwrap();
    let mut filtered_results: Vec<Neighbor<u32>> = Vec::new();

    let filtered_stats = rt
        .block_on(index.search(
            filtered_range,
            &labeled::Filtered::new(test_provider::Strategy::new(), &filter),
            &test_provider::Context::new(),
            query.as_slice(),
            &mut filtered_results,
        ))
        .unwrap();

    assert!(
        filtered_results.is_empty(),
        "no points should be within the radius {}",
        radius
    );
    assert!(
        !filtered_stats.range_search_second_round,
        "empty results shouldn't trigger a second round"
    );
}

#[test]
fn max_results_respected_means_no_second_round() {
    let description = "Test of `max_results` that sets `initial_l_search` equal /
     to `max_results`, with a permissive radius so that `max_results` is met / 
     without needing a second round.";
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("max_results_respected_means_no_second_round");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 50.0;
    let starting_l = 4;
    let max_results = 4;
    let filter = AlwaysTrueFilter;

    let filtered_range = FilteredRange::builder(starting_l, radius)
        .max_returned(Some(max_results))
        .build_filtered()
        .unwrap();

    let (filtered_stats, filtered_results) =
        run_filtered_range_search(&index, query.as_slice(), filtered_range, &filter);

    let baseline = RangeSearchBaseline::new(
        &filtered_range.range(),
        &filtered_results,
        filtered_stats,
        grid_size,
        description,
        query.clone(),
    );

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert!(
        filtered_results.len() <= max_results,
        "result count {} exceeds max_results {}",
        filtered_results.len(),
        max_results
    );
    assert!(
        !filtered_stats.range_search_second_round,
        "If max_results is respected, a second round should not be triggered"
    );
    assert_range_invariants(&filtered_results, radius, None);
    assert_no_duplicates(&filtered_results);
}

#[test]
fn max_results_respected_and_second_round_triggered() {
    let description = "Test of `max_results` that sets `initial_l_search` /
     below `max_results` so that a second round of search is triggered. Note that /
     the result set size is expected to be below `max_results` due to filtering /
     out start points during `post_process`.";

    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("max_results_respected_and_second_round_triggered");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 50.0;
    let starting_l = 4;
    let max_results = 200;
    let filter = AlwaysTrueFilter;

    let filtered_range = FilteredRange::builder(starting_l, radius)
        .max_returned(Some(max_results))
        .build_filtered()
        .unwrap();

    let (filtered_stats, filtered_results) =
        run_filtered_range_search(&index, query.as_slice(), filtered_range, &filter);

    let baseline = RangeSearchBaseline::new(
        &filtered_range.range(),
        &filtered_results,
        filtered_stats,
        grid_size,
        description,
        query.clone(),
    );

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert!(
        filtered_results.len() <= max_results,
        "result count {} exceeds max_results {}",
        filtered_results.len(),
        max_results
    );
    assert!(
        filtered_stats.range_search_second_round,
        "If max_results is respected, a second round should be triggered"
    );
    assert_range_invariants(&filtered_results, radius, None);
    assert_no_duplicates(&filtered_results);
}

//////////////////////////////////////////////////////////////
// Next, some tests with a somewhat more complex filter.    //
// We check that second round behavior is as expected, and  //
// that the filter is applied correctly. We also check that //
// putting a cap on the returned results correctly uses the //
// filter-satisfying results, not all results.              //
//////////////////////////////////////////////////////////////

#[test]
fn divisible_by_four_filter_second_round_triggered() {
    let description = "Test that a small starting L triggers a second /
     round of search when using the divisible-by-4 filter. Tests for /
     no duplicates, respect for the radius, and respect for the filter.";

    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("divisible_by_four_filter_second_round_triggered");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 50.0;
    let starting_l = 4;
    let filter = DivisibleByFourFilter;

    let filtered_range = FilteredRange::new(starting_l, radius).unwrap();

    let (filtered_stats, filtered_results) =
        run_filtered_range_search(&index, query.as_slice(), filtered_range, &filter);

    let baseline = RangeSearchBaseline::new(
        &filtered_range.range(),
        &filtered_results,
        filtered_stats,
        grid_size,
        description,
        query.clone(),
    );

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert!(
        filtered_stats.range_search_second_round,
        "small starting_l should trigger a second round with divisible-by-4 filter"
    );
    assert_range_invariants(&filtered_results, radius, None);
    assert_no_duplicates(&filtered_results);
    assert_divisible_by_four(&filtered_results);
}

#[test]
fn divisible_by_four_filter_no_second_round_from_l_search() {
    let description = "Test of divisible-by-4 filter with a larger starting L. \
     Since only 1/4 of discovered points will satisfy the filter, the second \
     round of search should not be triggered.";

    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("divisible_by_four_filter_no_second_round_from_l_search");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 12.0;
    let starting_l = 32;
    let filter = DivisibleByFourFilter;

    let filtered_range = FilteredRange::new(starting_l, radius).unwrap();

    let (filtered_stats, filtered_results) =
        run_filtered_range_search(&index, query.as_slice(), filtered_range, &filter);

    let baseline = RangeSearchBaseline::new(
        &filtered_range.range(),
        &filtered_results,
        filtered_stats,
        grid_size,
        description,
        query.clone(),
    );

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert!(
        !filtered_stats.range_search_second_round,
        "larger starting_l should avoid a second round with divisible-by-4 filter"
    );
    assert_range_invariants(&filtered_results, radius, None);
    assert_no_duplicates(&filtered_results);
    assert_divisible_by_four(&filtered_results);
}

#[test]
fn divisible_by_four_filter_no_second_round_from_max_results() {
    let description = "A test of the divisible-by-4 filter with `max_results` /
    set to starting_l / 4. The second round of search should not be triggered /
    since the correct number of results should be discovered.";
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("divisible_by_four_filter_no_second_round_from_max_results");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 50.0;
    let starting_l = 16;
    let max_results = starting_l / 4;
    let filter = DivisibleByFourFilter;

    let filtered_range = FilteredRange::builder(starting_l, radius)
        .max_returned(Some(max_results))
        .build_filtered()
        .unwrap();

    let (filtered_stats, filtered_results) =
        run_filtered_range_search(&index, query.as_slice(), filtered_range, &filter);

    let baseline = RangeSearchBaseline::new(
        &filtered_range.range(),
        &filtered_results,
        filtered_stats,
        grid_size,
        description,
        query.clone(),
    );

    let expected = get_or_save_test_results(&name, &baseline);
    assert_eq_verbose!(expected, baseline);

    assert!(
        !filtered_stats.range_search_second_round,
        "max_results = starting_l / 4 should prevent a second round"
    );
    assert_range_invariants(&filtered_results, radius, None);
    assert_no_duplicates(&filtered_results);
    assert_divisible_by_four(&filtered_results);
}
