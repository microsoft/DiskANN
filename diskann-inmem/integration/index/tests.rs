/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::graph::search::Knn;
use diskann_benchmark_core::recall::{RecallMetrics, Rows};
use diskann_benchmark_runner::utils::fmt::KeyValue;
use diskann_utils::views::Matrix;
use serde::{Deserialize, Serialize};

use crate::{
    index::{Counters, Index, KnnSearch},
    support::datatype::DatasetView,
};

pub(super) fn insert(
    index: &dyn Index,
    dataset: DatasetView<'_>,
    rt: &tokio::runtime::Handle,
) -> anyhow::Result<Counters> {
    let before = index.counters();
    for i in 0..dataset.nrows() {
        rt.block_on(index.insert(dataset.row(i).unwrap(), i as u64))?;
    }
    Ok(before.delta(&index.counters())?)
}

pub(super) fn knn(
    index: &dyn Index,
    knn: Knn,
    queries: DatasetView<'_>,
    groundtruth: &dyn Rows<u64>,
    rt: &tokio::runtime::Handle,
) -> anyhow::Result<KnnStats> {
    anyhow::ensure!(
        queries.nrows() == groundtruth.nrows(),
        "number of queries ({}) must match number of groundtruth entries ({})",
        queries.nrows(),
        groundtruth.nrows(),
    );

    let mut ids = Matrix::new(u64::MAX, queries.nrows(), knn.k_value().get());

    let before = index.counters();
    let mut misc = KnnSearch::new();
    let mut neighbors = Vec::new();
    for (i, out) in ids.row_iter_mut().enumerate() {
        neighbors.clear();

        let stats = rt.block_on(index.search(queries.row(i).unwrap(), knn, &mut neighbors))?;
        misc += stats;

        std::iter::zip(out.iter_mut(), neighbors.iter()).for_each(|(d, s)| *d = s.id);
    }
    let counters = before.delta(&index.counters())?;

    let recall = diskann_benchmark_core::recall::knn(
        groundtruth,
        None,
        &ids.as_view(),
        knn.k_value().get(),
        knn.k_value().get(),
        diskann_benchmark_core::recall::GroundTruthMode::Fixed,
    )?;

    Ok(KnnStats {
        counters,
        recall: recall.into(),
        misc,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct KnnRecall {
    recall_k: usize,
    recall_n: usize,
    num_queries: usize,
    average: f64,
}

impl From<RecallMetrics> for KnnRecall {
    fn from(metrics: RecallMetrics) -> Self {
        Self {
            recall_k: metrics.recall_k,
            recall_n: metrics.recall_n,
            num_queries: metrics.num_queries,
            average: metrics.average,
        }
    }
}

impl std::fmt::Display for KnnRecall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "recall = {:.4}, recall_k = {}, recall_n = {}, num_queries = {}",
            self.average, self.recall_k, self.recall_n, self.num_queries
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct KnnStats {
    recall: KnnRecall,
    counters: Counters,
    misc: KnnSearch,
}

impl std::fmt::Display for KnnStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("counters", &self.counters);
        kv.push("recall", &self.recall);
        kv.push("misc", &self.misc);
        kv.render(f)
    }
}
