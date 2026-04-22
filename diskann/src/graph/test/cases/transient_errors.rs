/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tests that transient (non-critical) errors during search and build are
//! gracefully tolerated rather than propagated.
//!
//! These exercise the `allow_transient` sites in the core algorithm:
//! - Beam expansion during search (glue.rs `ExpandBeam`)
//! - Prune accessor during insert (robust_prune_list)

use std::sync::Arc;

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, DiskANNIndex,
        search::Knn,
        test::{provider as test_provider, synthetic::Grid},
    },
    neighbor::{BackInserter, Neighbor},
    test::tokio::current_thread_runtime,
};

/// Build a pre-populated grid index for search tests.
fn setup_grid_index(grid: Grid, size: usize) -> Arc<DiskANNIndex<test_provider::Provider>> {
    let provider = test_provider::Provider::grid(grid, size).unwrap();

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

/// Search with transient errors on some IDs during beam expansion.
///
/// When `get_element` fails transiently for certain vector IDs, the search
/// should still complete successfully — those nodes are simply skipped during
/// distance computation. Results may differ from a non-flaky search, but
/// the operation must not return an error.
#[test]
fn search_tolerates_transient_errors() {
    let rt = current_thread_runtime();
    let grid = Grid::Three;
    let size = 5;
    let index = setup_grid_index(grid, size);

    // Make a few interior points flaky. These are valid IDs in the grid
    // (grid has size^dim = 125 points, IDs 0..124, start at u32::MAX).
    let flaky_strategy = test_provider::Strategy::with_transient(
        true, // working_set_reuse
        [10, 20, 30, 40, 50],
    );

    let query = vec![-1.0f32; grid.dim().into()];
    let params = Knn::new(10, 10, None).unwrap();
    let ctx = test_provider::Context::new();
    let mut neighbors = vec![Neighbor::<u32>::default(); params.k_value().get()];

    let stats = rt
        .block_on(index.search(
            params,
            &flaky_strategy,
            &ctx,
            query.as_slice(),
            &mut BackInserter::new(neighbors.as_mut_slice()),
        ))
        .unwrap();

    // Search should return some results. The exact count may be less than
    // k if flaky nodes were on the only path, but it must not be zero on
    // a well-connected grid.
    assert!(
        stats.result_count > 0,
        "search should return results despite transient errors"
    );
}

/// Build an index via single-insert with transient errors during pruning.
///
/// When `get_element` fails transiently for some IDs during `robust_prune_list`,
/// the insert should still complete. The resulting graph may not be identical to
/// a non-flaky build, but every insert must succeed.
#[test]
fn insert_tolerates_transient_prune_errors() {
    let rt = current_thread_runtime();
    let grid = Grid::Three;
    let size = 5;

    let max_degree = (grid.dim() as usize) * 2;
    let start_vector = grid.start_point(size);

    let config = test_provider::Config::new(
        Metric::L2,
        max_degree,
        test_provider::StartPoint::new(u32::MAX, start_vector),
    )
    .unwrap();

    let provider = test_provider::Provider::new(config);

    let index_config = graph::config::Builder::new(
        max_degree.saturating_sub(2).max(2),
        graph::config::MaxDegree::new(max_degree),
        100,
        Metric::L2.into(),
    )
    .build()
    .unwrap();

    let index = Arc::new(DiskANNIndex::new(index_config, provider, None));

    // Make a handful of IDs flaky during pruning so `get_element` can fail
    // transiently for some existing IDs. This exercises that insert/prune
    // tolerates those transient read failures, both during an ID's own insert
    // and when later inserts encounter that ID during prune.
    let flaky_strategy = test_provider::Strategy::with_transient(true, [3, 7, 12, 18]);

    let ctx = test_provider::Context::new();
    let data = grid.data(size);

    // Insert all grid points one-by-one. Each insert should succeed even when
    // pruning encounters transient errors on the flaky IDs.
    for (id, vector) in data.as_view().row_iter().enumerate() {
        rt.block_on(index.insert(flaky_strategy.clone(), &ctx, &(id as u32), vector))
            .unwrap();
    }

    // Verify the index is searchable after building with flaky prune.
    let params = Knn::new(5, 10, None).unwrap();
    let search_ctx = test_provider::Context::new();
    let mut neighbors = vec![Neighbor::<u32>::default(); params.k_value().get()];

    let stats = rt
        .block_on(index.search(
            params,
            &test_provider::Strategy::new(),
            &search_ctx,
            &vec![-1.0f32; grid.dim().into()],
            &mut BackInserter::new(neighbors.as_mut_slice()),
        ))
        .unwrap();

    assert!(
        stats.result_count > 0,
        "index built with flaky prune should still be searchable"
    );
}
