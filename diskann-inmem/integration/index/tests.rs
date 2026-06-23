/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::graph::search::Knn;
use diskann_benchmark_core::recall::Rows;

use super::Index;
use crate::support::datatype::DatasetView;

pub(super) fn insert(
    index: &dyn Index,
    dataset: DatasetView<'_>,
    rt: &tokio::runtime::Handle,
) -> anyhow::Result<()> {
    for i in 0..dataset.nrows() {
        rt.block_on(index.insert(dataset.row(i).unwrap(), i as u64))?;
    }
    Ok(())
}

fn knn(
    index: &dyn Index,
    knn: Knn,
    queries: DatasetView<'_>,
    groundtruth: &dyn Rows<u64>,
    rt: &tokio::runtime::Handle,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        queries.nrows() == groundtruth.nrows(),
        "number of queries ({}) must match number of groundtruth entries ({})",
        queries.nrows(),
        groundtruth.nrows(),
    );

    for i in 0..queries.nrows() {
        let mut neighbors = Vec::new();
        rt.block_on(index.search(queries.row(i).unwrap(), knn, &mut neighbors))?;
    }
    Ok(())
}
