/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::graph::search::Knn;
use diskann_benchmark_core::recall::{RecallMetrics, Rows};
use diskann_benchmark_runner::{
    report::{
        Report,
        tolerance::{FmtSpec, MaxAbsoluteChange},
    },
    utils::fmt::KeyValue,
};
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use serde::{Deserialize, Serialize};

use crate::{
    index::{CheckCounters, Counters, Index, KnnSearch},
    support::datatype::DatasetView,
};

pub(super) fn insert(
    index: &dyn Index,
    dataset: DatasetView<'_>,
    rt: &tokio::runtime::Handle,
) -> anyhow::Result<Counters> {
    let before = index.counters();
    for (i, r) in dataset.iter().enumerate() {
        rt.block_on(index.insert(r, i as u64))?;
    }
    before.delta(&index.counters())
}

pub(super) fn knn(
    index: &dyn Index,
    k: usize,
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

    let mut ids = rowmajor::Owned::from_element(queries.nrows(), k, u64::MAX);

    let before = index.counters();
    let mut misc = KnnSearch::new();
    let mut neighbors = Vec::new();
    for (out, query) in std::iter::zip(ids.rows_mut(), queries.iter()) {
        neighbors.clear();

        let stats = rt.block_on(index.search(query, knn, &mut neighbors))?;
        misc += stats;

        std::iter::zip(out.iter_mut(), neighbors.iter()).for_each(|(d, s)| *d = *s.id());
    }
    let counters = before.delta(&index.counters())?;

    let recall = diskann_benchmark_core::recall::knn(
        groundtruth,
        None,
        &ids.as_view(),
        k,
        k,
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
struct CheckKnnRecall {
    recall: MaxAbsoluteChange,
}

impl CheckKnnRecall {
    fn example() -> Self {
        const RECALL: MaxAbsoluteChange = MaxAbsoluteChange::try_from_f64(0.001).unwrap();
        Self { recall: RECALL }
    }

    fn check(&self, old: &KnnRecall, new: &KnnRecall) -> Report {
        let Self {
            recall: check_recall,
        } = self;
        let KnnRecall {
            recall_k,
            recall_n,
            num_queries,
            average,
        } = old;

        let fmt = FmtSpec::new().with_precision(4);
        let mut builder = Report::builder();
        builder.push("recall_k", Report::equal(*recall_k, new.recall_k));
        builder.push("recall_n", Report::equal(*recall_n, new.recall_n));
        builder.push("num_queries", Report::equal(*num_queries, new.num_queries));
        builder.push(
            "average",
            Report::equal_with_tolerance(check_recall, *average, new.average, fmt),
        );
        builder.finish()
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
        write!(f, "{}", kv)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct KnnDiff {
    recall: CheckKnnRecall,
    counters: CheckCounters,
}

impl KnnDiff {
    pub(crate) fn example() -> Self {
        Self {
            recall: CheckKnnRecall::example(),
            counters: CheckCounters::example(),
        }
    }

    pub(crate) fn check(&self, old: &KnnStats, new: &KnnStats) -> Report {
        let Self {
            recall: check_recall,
            counters: check_counters,
        } = self;
        let KnnStats {
            recall,
            counters,
            misc,
        } = old;

        let mut builder = Report::builder();
        builder.push("recall", check_recall.check(recall, &new.recall));
        builder.push("counters", check_counters.check(counters, &new.counters));
        builder.push("misc", check_counters.check_hops_and_cmps(misc, &new.misc));
        builder.finish()
    }
}
