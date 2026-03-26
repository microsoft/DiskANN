/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod knn;
pub mod multihop;
pub mod range;

pub mod strategy;

pub use knn::KNN;
pub use multihop::MultiHop;
pub use range::Range;

pub use strategy::Strategy;

////////////////
// Test Utils //
////////////////

#[cfg(test)]
fn test_grid_provider()
-> std::sync::Arc<diskann::graph::DiskANNIndex<diskann::graph::test::provider::Provider>> {
    use diskann::graph::{
        self,
        test::{provider::Provider, synthetic},
    };

    let grid = synthetic::Grid::Four;
    let provider = Provider::grid(grid, 4).unwrap();
    let config = diskann::graph::config::Builder::new(
        provider.max_degree(),
        graph::config::MaxDegree::same(),
        10,
        provider.distance_metric().into(),
    )
    .build()
    .unwrap();

    std::sync::Arc::new(graph::DiskANNIndex::new(config, provider, None))
}
