/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tests for `consolidate_vector` covering all three code paths:
//!
//! 1. Consolidating a deleted vertex → `ConsolidateKind::Deleted`
//! 2. Consolidating a vertex with nothing to repair → `ConsolidateKind::Complete` (no-op)
//! 3. Consolidating a vertex with deleted neighbors → prune and repair edges
//! 4. Transient failure during vector retrieval → `ConsolidateKind::FailedVectorRetrieval`

use std::{iter, sync::Arc};

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, AdjacencyList, ConsolidateKind, DiskANNIndex,
        test::provider::{self as test_provider, Provider, Strategy},
    },
    provider::Delete,
    test::tokio::current_thread_runtime,
};

use super::helpers::{assert_neighbors, create_2d_unit_square, setup_2d_square};

/// Build a small index with explicit vectors and adjacency lists for consolidation testing.
///
/// The provider's `max_degree` is set higher than the index's `pruned_degree` so that
/// adjacency lists can exceed the index limit, forcing `consolidate_vector` to prune.
fn setup_consolidation_index(
    vectors: Vec<Vec<f32>>,
    adjacency_lists: Vec<AdjacencyList<u32>>,
) -> Arc<DiskANNIndex<Provider>> {
    let num_points = vectors.len();
    let dim = vectors[0].len();
    let start_id = num_points as u32;

    // The provider accepts up to 5 neighbors per node, but the index only targets 4.
    // This mismatch lets us populate adjacency lists that consolidate_vector must prune.
    let provider_max_degree = 5;
    let pruned_degree = 4;

    let provider_config = test_provider::Config::new(
        Metric::L2,
        provider_max_degree,
        test_provider::StartPoint::new(start_id, vec![0.5; dim]),
    )
    .unwrap();

    let start_neighbors =
        AdjacencyList::from_iter_untrusted((0..num_points as u32).take(provider_max_degree));

    let points = vectors
        .into_iter()
        .zip(adjacency_lists)
        .enumerate()
        .map(|(id, (vec, adj))| (id as u32, vec, adj));

    let provider = Provider::new_from(
        provider_config,
        iter::once((start_id, start_neighbors)),
        points,
    )
    .unwrap();

    let index_config = graph::config::Builder::new(
        pruned_degree,
        graph::config::MaxDegree::same(),
        10,
        Metric::L2.into(),
    )
    .build()
    .unwrap();

    Arc::new(DiskANNIndex::new(index_config, provider, None))
}

/// When `consolidate_vector` encounters a transient error during `fill` (vector retrieval),
/// it should gracefully return `ConsolidateKind::FailedVectorRetrieval` rather than
/// propagating the error.
#[test]
fn flaky_consolidate_returns_failed_retrieval() {
    let rt = current_thread_runtime();

    // Build a graph where node 0 has 5 neighbors (exceeds max_degree=4),
    // forcing consolidate_vector to attempt pruning.
    let vectors = vec![
        vec![0.0, 0.0], // point 0
        vec![0.0, 1.0], // point 1
        vec![1.0, 0.0], // point 2
        vec![1.0, 1.0], // point 3
        vec![2.0, 2.0], // point 4
        vec![0.0, 2.0], // point 5
        vec![2.0, 0.0], // point 6
    ];
    let adjacency_lists = vec![
        AdjacencyList::from_iter_untrusted([1, 2, 3, 4, 5]), // point 0: 5 neighbors > max_degree
        AdjacencyList::from_iter_untrusted([0, 3, 4]),
        AdjacencyList::from_iter_untrusted([0, 3, 4]),
        AdjacencyList::from_iter_untrusted([1, 2, 4]),
        AdjacencyList::from_iter_untrusted([0, 1, 2, 3]),
        AdjacencyList::from_iter_untrusted([0, 1, 2]),
        AdjacencyList::from_iter_untrusted([0, 1, 2, 3, 5]),
    ];

    let index = setup_consolidation_index(vectors, adjacency_lists);
    let ctx = test_provider::Context::new();

    // Make only the consolidated node (0) transient. During robust_prune_list, fill()
    // requests node 0's vector first — the transient error causes view.get(0) to return
    // None, triggering the FailedVectorRetrieval path.
    let flaky_strategy = Strategy::with_transient(
        true, // working_set_reuse
        [0],  // transient ids
    );

    let result = rt
        .block_on(index.consolidate_vector(&flaky_strategy, &ctx, 0))
        .unwrap();

    assert_eq!(
        result,
        ConsolidateKind::FailedVectorRetrieval,
        "consolidate should handle transient errors gracefully"
    );
}

/// Consolidating a deleted vertex returns `ConsolidateKind::Deleted`.
#[test]
fn consolidate_deleted_vertex_returns_deleted() {
    let rt = current_thread_runtime();

    let adjacency_lists = vec![
        AdjacencyList::from_iter_untrusted([1, 2, 4]),
        AdjacencyList::from_iter_untrusted([0, 3, 4]),
        AdjacencyList::from_iter_untrusted([0, 3, 4]),
        AdjacencyList::from_iter_untrusted([1, 2, 4]),
        AdjacencyList::from_iter_untrusted([0, 1, 2, 3]),
    ];

    let index = setup_2d_square(create_2d_unit_square(), adjacency_lists, 4);
    let ctx = test_provider::Context::new();
    let strategy = Strategy::new();

    // Delete vertex 3, then try to consolidate it.
    rt.block_on(index.data_provider.delete(&ctx, &3)).unwrap();

    let result = rt
        .block_on(index.consolidate_vector(&strategy, &ctx, 3))
        .unwrap();

    assert_eq!(result, ConsolidateKind::Deleted);
}

/// Consolidating a vertex with no deleted neighbors and degree within limits is a no-op.
#[test]
fn consolidate_nothing_to_do_returns_complete() {
    let rt = current_thread_runtime();

    let adjacency_lists = vec![
        AdjacencyList::from_iter_untrusted([1, 2, 4]),
        AdjacencyList::from_iter_untrusted([0, 3, 4]),
        AdjacencyList::from_iter_untrusted([0, 3, 4]),
        AdjacencyList::from_iter_untrusted([1, 2, 4]),
        AdjacencyList::from_iter_untrusted([0, 1, 2, 3]),
    ];

    let index = setup_2d_square(create_2d_unit_square(), adjacency_lists, 4);
    let ctx = test_provider::Context::new();
    let strategy = Strategy::new();

    // No deletions, all degrees within limit → nothing to do.
    let result = rt
        .block_on(index.consolidate_vector(&strategy, &ctx, 0))
        .unwrap();

    assert_eq!(result, ConsolidateKind::Complete);
}

/// After deleting vertex 3 and consolidating all vertices, edges to vertex 3 are removed
/// and replacement edges are added.
#[test]
fn consolidate_repairs_after_deletion() {
    let rt = current_thread_runtime();

    let adjacency_lists = vec![
        AdjacencyList::from_iter_untrusted([1, 2, 4]),
        AdjacencyList::from_iter_untrusted([0, 3, 4]),
        AdjacencyList::from_iter_untrusted([0, 3, 4]),
        AdjacencyList::from_iter_untrusted([1, 2, 4]),
        AdjacencyList::from_iter_untrusted([0, 1, 2, 3]),
    ];

    let index = setup_2d_square(create_2d_unit_square(), adjacency_lists, 4);
    let ctx = test_provider::Context::new();
    let strategy = Strategy::new();

    // Delete vertex 3.
    rt.block_on(index.data_provider.delete(&ctx, &3)).unwrap();

    // Consolidate all vertices.
    for id in 0..5u32 {
        let result = rt
            .block_on(index.consolidate_vector(&strategy, &ctx, id))
            .unwrap();

        if id == 3 {
            assert_eq!(result, ConsolidateKind::Deleted);
        } else {
            assert_ne!(result, ConsolidateKind::FailedVectorRetrieval);
        }
    }

    // Expected outcome:
    // - vertex 0: unchanged → [1, 2, 4]
    // - vertex 1: edge to 3 replaced with edge to 2 → [0, 2, 4]
    // - vertex 2: edge to 3 replaced with edge to 1 → [0, 1, 4]
    // - vertex 4 (start): edge to 3 removed → [0, 1, 2]
    assert_neighbors(&rt, &index, 0, &[1, 2, 4]);
    assert_neighbors(&rt, &index, 1, &[0, 2, 4]);
    assert_neighbors(&rt, &index, 2, &[0, 1, 4]);
    assert_neighbors(&rt, &index, 4, &[0, 1, 2]);
}
