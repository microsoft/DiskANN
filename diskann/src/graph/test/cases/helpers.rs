/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared test helpers for graph index test cases.

use std::{iter, sync::Arc};

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, AdjacencyList, DiskANNIndex,
        test::provider::{self as test_provider, Provider},
    },
    provider::NeighborAccessor,
};

pub(super) fn create_2d_unit_square() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ]
}

/// Build a 2D square index: 4 corners + start point at (0.5, 0.5) with ID 4.
///
/// The `pruned_degree` controls the index's target degree. The provider's max degree
/// is set to the largest adjacency list size to allow pre-populating graphs that
/// may exceed the index limit (useful for consolidation tests).
pub(super) fn setup_2d_square(
    vectors: Vec<Vec<f32>>,
    adjacency_lists: Vec<AdjacencyList<u32>>,
    pruned_degree: usize,
) -> Arc<DiskANNIndex<Provider>> {
    let num_points = vectors.len();
    let start_id = num_points as u32;
    let provider_max_degree = adjacency_lists
        .iter()
        .map(|a| a.len())
        .max()
        .map(|m| m.max(pruned_degree))
        .unwrap_or(pruned_degree);

    let provider_config = test_provider::Config::new(
        Metric::L2,
        provider_max_degree,
        test_provider::StartPoint::new(start_id, vec![0.5, 0.5]),
    )
    .unwrap();

    let start_adj = adjacency_lists
        .get(num_points)
        .cloned()
        .unwrap_or_else(|| AdjacencyList::from_iter_untrusted(0..num_points as u32));

    let points = vectors
        .into_iter()
        .zip(adjacency_lists.into_iter().take(num_points))
        .enumerate()
        .map(|(id, (vec, adj))| (id as u32, vec, adj));

    let provider =
        Provider::new_from(provider_config, iter::once((start_id, start_adj)), points).unwrap();

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

pub(super) fn generate_2d_square_adjacency_list() -> Vec<AdjacencyList<u32>> {
    vec![
        AdjacencyList::from_iter_untrusted([1, 4]),
        AdjacencyList::from_iter_untrusted([0, 4]),
        AdjacencyList::from_iter_untrusted([3, 4]),
        AdjacencyList::from_iter_untrusted([2, 4]),
        AdjacencyList::from_iter_untrusted([0, 1, 2, 3]),
    ]
}

/// Assert that `id`'s sorted neighbor list equals `expected`.
pub(super) fn assert_neighbors(
    rt: &tokio::runtime::Runtime,
    index: &DiskANNIndex<Provider>,
    id: u32,
    expected: &[u32],
) {
    let mut list = AdjacencyList::new();
    rt.block_on(index.provider().neighbors().get_neighbors(id, &mut list))
        .expect("get_neighbors failed");
    list.sort();
    assert_eq!(&*list, expected, "neighbors of node {id}");
}
