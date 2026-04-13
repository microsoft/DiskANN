/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tests for `consolidate_vector` behavior under transient (flaky) errors.
//!
//! Migrated from `diskann-providers/src/index/diskann_async.rs`. The original used
//! the inmem provider's `FlakyAccessor` which fails every Nth `get_element` call.
//! This version uses `test_provider::Accessor::flaky()` which fails for specific IDs.

use std::{collections::HashSet, iter, sync::Arc};

use diskann_vector::distance::Metric;

use crate::{
    error::Infallible,
    graph::{
        self, AdjacencyList, ConsolidateKind, DiskANNIndex,
        glue::PruneStrategy,
        test::provider::{self as test_provider, Accessor, Provider},
        workingset,
    },
    test::tokio::current_thread_runtime,
    utils::VectorRepr,
};

/// A [`PruneStrategy`] that produces a flaky accessor returning transient errors
/// for the specified IDs. Everything else delegates to the default [`test_provider::Strategy`].
struct FlakyPruneStrategy {
    transient_ids: HashSet<u32>,
}

impl FlakyPruneStrategy {
    fn new(transient_ids: impl IntoIterator<Item = u32>) -> Self {
        Self {
            transient_ids: transient_ids.into_iter().collect(),
        }
    }
}

impl PruneStrategy<Provider> for FlakyPruneStrategy {
    type WorkingSet = workingset::Map<u32, Box<[f32]>, workingset::map::Ref<[f32]>>;
    type DistanceComputer = <f32 as VectorRepr>::Distance;
    type PruneAccessor<'a> = Accessor<'a>;
    type PruneAccessorError = Infallible;

    fn create_working_set(&self, capacity: usize) -> Self::WorkingSet {
        workingset::map::Builder::new(workingset::map::Capacity::None).build(capacity)
    }

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a Provider,
        _context: &'a test_provider::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(Accessor::flaky(provider, self.transient_ids.clone()))
    }
}

/// Build a small index with explicit vectors and adjacency lists for consolidation testing.
///
/// The provider's `max_degree` is set higher than the index's `pruned_degree` so that
/// adjacency lists can exceed the index limit, forcing `consolidate_vector` to prune.
fn setup_consolidation_index(
    vectors: &[Vec<f32>],
    adjacency_lists: &[AdjacencyList<u32>],
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

    let points = vectors.iter().enumerate().map(|(id, vec)| {
        (id as u32, vec.clone(), adjacency_lists[id].clone())
    });

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
    let adjacency_lists = [
        AdjacencyList::from_iter_untrusted([1, 2, 3, 4, 5]), // point 0: 5 neighbors > max_degree
        AdjacencyList::from_iter_untrusted([0, 3, 4]),
        AdjacencyList::from_iter_untrusted([0, 3, 4]),
        AdjacencyList::from_iter_untrusted([1, 2, 4]),
        AdjacencyList::from_iter_untrusted([0, 1, 2, 3]),
        AdjacencyList::from_iter_untrusted([0, 1, 2]),
        AdjacencyList::from_iter_untrusted([0, 1, 2, 3, 5]),
    ];

    let index = setup_consolidation_index(&vectors, &adjacency_lists);
    let ctx = test_provider::Context::new();

    // Make ALL non-start-point IDs transient — this ensures that fill() will encounter
    // a transient error when trying to retrieve any data point's vector.
    let transient_ids: HashSet<u32> = (0..vectors.len() as u32).collect();
    let flaky_strategy = FlakyPruneStrategy::new(transient_ids);

    let result = rt
        .block_on(index.consolidate_vector(&flaky_strategy, &ctx, 0))
        .unwrap();

    assert_eq!(
        result,
        ConsolidateKind::FailedVectorRetrieval,
        "consolidate should handle transient errors gracefully"
    );
}
