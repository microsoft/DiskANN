/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use crate::{
    graph::{
        self,
        search::Knn,
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

// The root directory for tests located in this module.
fn root() -> TestRoot {
    TestRoot::new("graph/test/cases/grid_search")
}

////////////
// Search //
////////////

/// Metrics and results for a single search of the grid synthetic workload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct GridSearch {
    /// A description of what to expect, what trends to observe, and anything else
    /// a reviewer may need to either understand why this test is checked in or to validate
    /// any changes that occur in the checked-in file.
    pub(super) description: String,

    /// The query given to search.
    pub(super) query: Vec<f32>,

    /// The results returned from search.
    pub(super) results: Vec<(u32, f32)>,

    /// The number of comparisons returned by search.
    pub(super) comparisons: usize,

    /// The number of hops returned by search.
    pub(super) hops: usize,

    /// The number of results returned by search.
    pub(super) num_results: usize,

    /// The dimesionality of the underlying grid.
    pub(super) grid_dims: usize,

    /// The size of the underlying grid.
    pub(super) grid_size: usize,

    /// The beam width used for search.
    pub(super) beam_width: usize,

    /// Index level metrics recorded during search.
    pub(super) metrics: test_provider::Metrics,
}

verbose_eq!(GridSearch {
    query,
    description,
    results,
    comparisons,
    hops,
    num_results,
    grid_size,
    grid_dims,
    beam_width,
    metrics,
});

/// Initialize the test provider with a grid of `grid.dim()` dimensions and a side length
/// of `size`.
///
/// The provider will be configured with the L2 metric.
fn setup_grid_search(grid: Grid, size: usize) -> Arc<test_provider::Provider> {
    Arc::new(test_provider::Provider::grid(grid, size).unwrap())
}

const BEAM_WIDTHS: [usize; 3] = [1, 2, 4];

fn _grid_search(grid: Grid, size: usize, mut parent: TestPath<'_>) {
    let rt = current_thread_runtime();

    let description_0 = "With a query of all -1s, we expect the neighbor with all zeros to be\
        the closest. Due to how the grid is generated, this will be coordinate 0. \
        Next, there should be `dim` neighbors that are one further away. \
        Increasing the beam width should increase the number of comparisons.";

    let description_1 = "With a query of all `size`s, the start point is filtered by default \
        and should not appear in results.";

    let query_desc = [
        (vec![-1.0f32; grid.dim().into()], description_0),
        (vec![size as f32; grid.dim().into()], description_1),
    ];

    let mut results = Vec::new();
    for (query, desc) in query_desc {
        for beam_width in BEAM_WIDTHS {
            // Make sure we start with a new index every time so the internal counters
            // are correct.
            let index = setup_grid_search(grid, size);

            let k_value = 10;
            let params = Knn::new(10, Some(beam_width)).unwrap();

            let (neighbors, stats) = rt
                .block_on(index.search(params, query.as_slice(), NonZeroUsize::new(k_value)))
                .unwrap();

            let graph::algorithms::SearchStats {
                cmps,
                hops,
                result_count,
                range_search_second_round,
            } = stats;

            assert!(
                result_count.into_usize() <= k_value,
                "grid search should not return more than the requested number of neighbors",
            );

            assert!(
                !range_search_second_round,
                "range search should not activate for k-nearest-neighbors",
            );

            let metrics = index.metrics();

            // Check that the mutation metrics are zero.
            assert_eq!(metrics.set_vector, 0);
            assert_eq!(metrics.set_neighbors, 0);
            assert_eq!(metrics.append_neighbors, 0);

            // Check that the number of hops and was computed successfully.
            assert_eq!(
                metrics.get_neighbors,
                hops.into_usize(),
                "recorded hops should have a one-to-one correspondence with `get_neighbors`",
            );

            assert_eq!(
                metrics.get_vector,
                cmps.into_usize(),
                "recorded comparisons should have a one-to-one correspondence with `get_vector`",
            );

            results.push(GridSearch {
                query: query.clone(),
                description: desc.to_string(),
                results: neighbors.into_iter().map(|i| i.as_tuple()).collect(),
                comparisons: cmps.into_usize(),
                hops: hops.into_usize(),
                num_results: result_count.into_usize(),
                grid_dims: grid.dim().into(),
                grid_size: size,
                beam_width,
                metrics,
            });
        }
    }
    // Mangle the test parameters.
    let name = parent.push(format!("search_{}_{}", grid.dim(), size,));
    let expected = get_or_save_test_results(&name, &results);
    assert_eq_verbose!(expected, results);
}

#[test]
fn grid_search_1_100() {
    _grid_search(Grid::One, 100, root().path());
}

#[test]
fn grid_search_3_5() {
    _grid_search(Grid::Three, 5, root().path());
}

#[test]
fn grid_search_4_4() {
    _grid_search(Grid::Four, 4, root().path());
}
