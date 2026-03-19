/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Baseline regression tests for range search on pre-built grid indices.
//!
//! This module validates [`Range`] search over synthetic grid workloads,
//! exercising both the initial k-NN phase and the range-expansion phase.
//! Metrics (comparisons, hops, get_vector calls) are recorded and checked
//! against baselines so that algorithmic changes are detected early.

use std::sync::Arc;

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, DiskANNIndex,
        search::Range,
        test::{provider as test_provider, synthetic::Grid},
    },
    test::{
        TestPath, TestRoot,
        cmp::{assert_eq_verbose, verbose_eq},
        get_or_save_test_results,
        tokio::current_thread_runtime,
    },
    utils::IntoUsize,
};

fn root() -> TestRoot {
    TestRoot::new("graph/test/cases/grid_range_search")
}

//////////////////
// Range Search //
//////////////////

/// Metrics and results for a single range search of the grid synthetic workload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GridRangeSearch {
    /// Description of the test scenario for reviewers.
    description: String,

    /// The query given to search.
    query: Vec<f32>,

    /// The search radius.
    radius: f32,

    /// The inner radius (if any).
    inner_radius: Option<f32>,

    /// The IDs returned within the range.
    result_ids: Vec<u32>,

    /// The distances corresponding to each result.
    result_distances: Vec<f32>,

    /// The number of results returned.
    num_results: usize,

    /// Whether the range search second round was activated.
    range_search_second_round: bool,

    /// The number of comparisons returned by search.
    comparisons: usize,

    /// The number of hops returned by search.
    hops: usize,

    /// The dimensionality of the underlying grid.
    grid_dims: usize,

    /// The size of the underlying grid.
    grid_size: usize,

    /// Index level metrics recorded during search.
    metrics: test_provider::Metrics,
}

verbose_eq!(GridRangeSearch {
    description,
    query,
    radius,
    inner_radius,
    result_ids,
    result_distances,
    num_results,
    range_search_second_round,
    comparisons,
    hops,
    grid_dims,
    grid_size,
    metrics,
});

/// Initialize the test provider with a grid and configure it for L2 distance.
fn setup_grid_range_search(grid: Grid, size: usize) -> Arc<DiskANNIndex<test_provider::Provider>> {
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

/// Range search test parameters.
struct RangeTestCase {
    query: Vec<f32>,
    radius: f32,
    inner_radius: Option<f32>,
    description: &'static str,
}

fn _grid_range_search(grid: Grid, size: usize, mut parent: TestPath<'_>) {
    let rt = current_thread_runtime();

    let dim: usize = grid.dim().into();

    // For a grid with L2 distance, the distance between adjacent points is 1.0.
    // A query at the origin [-1, -1, ...] has distance `dim` to point (0,0,...,0).
    // We choose radii to exercise different result-set sizes.
    let cases = vec![
        RangeTestCase {
            query: vec![-1.0f32; dim],
            radius: (dim as f32).sqrt() + 0.1,
            inner_radius: None,
            description: "With a query at all -1s, a radius just over sqrt(dim) should \
                capture point (0,...,0) and no others since L2 squared distances \
                for the next ring of neighbors are at least dim+1.",
        },
        RangeTestCase {
            query: vec![0.5f32; dim],
            radius: (dim as f32) + 0.1,
            inner_radius: None,
            description: "With a query at all 0.5s, a radius of dim+0.1 captures a small \
                neighborhood around the center of the grid. We expect multiple results.",
        },
        RangeTestCase {
            query: vec![0.0f32; dim],
            radius: (dim as f32) * 2.0 + 0.1,
            inner_radius: Some((dim as f32) + 0.1),
            description: "With a query at the origin, inner_radius excludes the closest \
                points and outer_radius captures the next ring. Validates annular \
                range search.",
        },
    ];

    let mut results = Vec::new();
    for case in &cases {
        // Fresh index per search to get accurate provider counters.
        let index = setup_grid_range_search(grid, size);

        let range = match case.inner_radius {
            None => Range::new(100, case.radius).unwrap(),
            Some(inner) => {
                Range::with_options(None, 100, None, case.radius, Some(inner), 1.0, 1.0).unwrap()
            }
        };

        let context = test_provider::Context::new();

        let output = rt
            .block_on(index.search(
                range,
                &test_provider::Strategy::new(),
                &context,
                case.query.as_slice(),
                &mut (),
            ))
            .unwrap();

        let metrics = index.provider().metrics();

        // Verify that mutation metrics are zero.
        assert_eq!(metrics.set_vector, 0);
        assert_eq!(metrics.set_neighbors, 0);
        assert_eq!(metrics.append_neighbors, 0);

        // Verify that comparisons and hops correspond to provider counters.
        assert_eq!(
            metrics.get_neighbors,
            output.stats.hops.into_usize(),
            "recorded hops should have a one-to-one correspondence with `get_neighbors`",
        );
        assert_eq!(
            metrics.get_vector,
            output.stats.cmps.into_usize(),
            "recorded comparisons should have a one-to-one correspondence with `get_vector`",
        );

        {
            let test_provider::ContextMetrics { spawns, clones } = context.metrics();
            assert_eq!(spawns, 0);
            assert_eq!(clones, 0);
        }

        // Verify result consistency.
        assert_eq!(
            output.ids.len(),
            output.distances.len(),
            "ids and distances must have the same length"
        );
        assert_eq!(
            output.ids.len(),
            output.stats.result_count.into_usize(),
            "result_count should match ids length"
        );

        // All returned distances should be within [inner_radius, radius].
        for (i, &dist) in output.distances.iter().enumerate() {
            assert!(
                dist <= case.radius,
                "result {i}: distance {dist} exceeds radius {}",
                case.radius
            );
            if let Some(inner) = case.inner_radius {
                assert!(
                    dist > inner,
                    "result {i}: distance {dist} is within inner_radius {inner}",
                );
            }
        }

        results.push(GridRangeSearch {
            query: case.query.clone(),
            description: case.description.to_string(),
            radius: case.radius,
            inner_radius: case.inner_radius,
            result_ids: output.ids,
            result_distances: output.distances,
            num_results: output.stats.result_count.into_usize(),
            range_search_second_round: output.stats.range_search_second_round,
            comparisons: output.stats.cmps.into_usize(),
            hops: output.stats.hops.into_usize(),
            grid_dims: dim,
            grid_size: size,
            metrics,
        });
    }

    let name = parent.push(format!("range_search_{}_{}", grid.dim(), size));
    let expected = get_or_save_test_results(&name, &results);
    assert_eq_verbose!(expected, results);
}

#[test]
fn grid_range_search_1_100() {
    _grid_range_search(Grid::One, 100, root().path());
}

#[test]
fn grid_range_search_3_5() {
    _grid_range_search(Grid::Three, 5, root().path());
}

#[test]
fn grid_range_search_4_4() {
    _grid_range_search(Grid::Four, 4, root().path());
}
