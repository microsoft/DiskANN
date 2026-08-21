/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tests for range-based search within a distance radius.
//!
//! Covers basic range search, inner radius filtering, two-round expansion,
//! and empty result handling. Integration tests use baselines for regression
//! protection.

use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, DiskANNIndex, SearchOutputBuffer,
        glue::SearchPostProcess,
        index::SearchStats,
        search::Range,
        test::{provider as test_provider, synthetic::Grid},
    },
    neighbor::Neighbor,
    provider::HasId,
    test::{
        TestRoot,
        cmp::{assert_eq_verbose, verbose_eq},
        get_or_save_test_results,
        tokio::current_thread_runtime,
    },
};

#[derive(Clone)]
struct RecordingCopyIds {
    candidate_count: Arc<AtomicUsize>,
    saw_start_point: Arc<AtomicBool>,
}

impl<A, T> SearchPostProcess<A, T> for RecordingCopyIds
where
    A: HasId<Id = u32>,
{
    type Error = Infallible;

    fn post_process<I, B>(
        &self,
        _accessor: &mut A,
        _query: T,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<u32> + Send + ?Sized,
    {
        let candidates: Vec<_> = candidates.collect();
        self.candidate_count
            .store(candidates.len(), Ordering::Relaxed);
        self.saw_start_point.store(
            candidates
                .iter()
                .any(|candidate| *candidate.id() == u32::MAX),
            Ordering::Relaxed,
        );
        let count = output.extend(candidates);
        std::future::ready(Ok(count))
    }
}

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

#[test]
fn max_results_caps_non_start_candidates_after_range_collection() {
    let rt = current_thread_runtime();
    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let starting_l = 4;
    let max_results = 5;
    let range_search = Range::builder(starting_l, f32::MAX)
        .inner_radius(Some(0.0))
        .max_returned(Some(max_results))
        .build()
        .unwrap();

    let candidate_count = Arc::new(AtomicUsize::new(0));
    let saw_start_point = Arc::new(AtomicBool::new(false));
    let processor = RecordingCopyIds {
        candidate_count: Arc::clone(&candidate_count),
        saw_start_point: Arc::clone(&saw_start_point),
    };
    let mut results = Vec::<Neighbor<u32>>::new();

    let stats = rt
        .block_on(index.search_with(
            range_search,
            &test_provider::Strategy::new(),
            processor,
            &test_provider::Context::new(),
            query.as_slice(),
            &mut results,
        ))
        .unwrap();

    assert_eq!(candidate_count.load(Ordering::Relaxed), max_results + 1);
    assert!(!saw_start_point.load(Ordering::Relaxed));
    assert_eq!(results.len(), max_results);
    assert_eq!(stats.result_count as usize, max_results);
    assert!(stats.range_search_second_round);
    assert!(results.iter().all(|result| *result.id() != u32::MAX));
    assert_range_invariants(&results, f32::MAX, Some(0.0));
    assert_no_duplicates(&results);
}

#[test]
fn initial_slack_low_triggers_second_round() {
    let _description = "Test that low initial_slack triggers second round. \
    With initial_slack=0.5 and starting_l=4, the threshold is 2, so any \
    outer_range_len >= 2 will trigger the second round.";

    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("initial_slack_low_triggers_second_round");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 50.0;
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
        description: "Low initial_slack triggers second round search.".to_string(),
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
    let _description = "Test that high initial_slack avoids second round. \
    With initial_slack=1.0 and starting_l=4, the threshold is 4, making it \
    harder to trigger the second round.";

    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("initial_slack_high_avoids_second_round");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 50.0;
    let starting_l = 4;
    let high_slack = 1.0;

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
        description: "High initial_slack avoids second round search.".to_string(),
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
fn range_slack_low_constrains_frontier() {
    let _description = "Test that low range_slack constrains frontier expansion. \
    With range_slack=1.0, the frontier only expands to nodes within radius, \
    resulting in fewer total results found.";

    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("range_slack_low_constrains_frontier");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 30.0;
    let starting_l = 4;
    let initial_slack = 0.5;
    let low_range_slack = 1.0;

    let range_search = Range::builder(starting_l, radius)
        .initial_slack(initial_slack)
        .range_slack(low_range_slack)
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
        description: "Low range_slack constrains frontier expansion.".to_string(),
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
        "low initial_slack should trigger second round"
    );

    assert_range_invariants(&results, radius, None);
    assert_no_duplicates(&results);
}

#[test]
fn range_slack_high_expands_frontier() {
    let _description = "Test that high range_slack expands frontier exploration. \
    With range_slack=1.3, the frontier expands to nodes up to radius * 1.3, \
    potentially finding more results than lower range_slack.";

    let rt = current_thread_runtime();
    let mut test_root = root();
    let mut path = test_root.path();
    let name = path.push("range_slack_high_expands_frontier");

    let grid_size = 5;
    let (index, query) = setup_grid_index_and_default_query(grid_size, Grid::Three);
    let radius = 30.0;
    let starting_l = 4;
    let initial_slack = 0.5;
    let high_range_slack = 1.3;

    let range_search = Range::builder(starting_l, radius)
        .initial_slack(initial_slack)
        .range_slack(high_range_slack)
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
        description: "High range_slack expands frontier exploration.".to_string(),
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
        "low initial_slack should trigger second round"
    );

    assert_range_invariants(&results, radius, None);
    assert_no_duplicates(&results);
}
