/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::utils::IntoUsize;
use diskann_benchmark_core::recall::RecallMetrics;
use diskann_benchmark_runner::{
    report::{
        Report,
        tolerance::{FmtSpec, MaxAbsoluteChange, MaxRelativeChange},
    },
    utils::fmt::KeyValue,
};
use diskann_inmem::integration;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct KnnSearch {
    hops: usize,
    cmps: usize,
}

impl KnnSearch {
    pub(crate) fn new() -> Self {
        Self { hops: 0, cmps: 0 }
    }
}

impl From<diskann::graph::index::SearchStats> for KnnSearch {
    fn from(stats: diskann::graph::index::SearchStats) -> Self {
        Self {
            hops: stats.hops.into_usize(),
            cmps: stats.cmps.into_usize(),
        }
    }
}

impl std::ops::AddAssign for KnnSearch {
    fn add_assign(&mut self, rhs: Self) {
        self.hops = self.hops.wrapping_add(rhs.hops);
        self.cmps = self.cmps.wrapping_add(rhs.cmps);
    }
}

impl std::fmt::Display for KnnSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hops = {}, cmps = {}", self.hops, self.cmps)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Counters {
    query_distance: u64,
    distance: u64,
    get_vector: u64,
    set_vector: u64,
    get_neighbors: u64,
    set_neighbors: u64,
    append_neighbors: u64,
}

impl Counters {
    #[cfg(test)]
    pub(super) fn test_fixture() -> Self {
        Self {
            query_distance: 10_000,
            distance: 0,
            get_vector: 0,
            set_vector: 0,
            get_neighbors: 0,
            set_neighbors: 0,
            append_neighbors: 0,
        }
    }

    pub(crate) fn delta(&self, after: &Counters) -> anyhow::Result<Self> {
        #[derive(Debug, Error)]
        #[error(
            "counter \"{}\" non-monotonically increasing from {} to {}",
            self.0,
            self.1,
            self.2
        )]
        struct NonMonotonic(&'static str, u64, u64);

        fn check(before: u64, after: u64, field: &'static str) -> Result<u64, NonMonotonic> {
            after
                .checked_sub(before)
                .ok_or(NonMonotonic(field, before, after))
        }

        let delta = Self {
            query_distance: check(self.query_distance, after.query_distance, "query_distance")?,
            distance: check(self.distance, after.distance, "distance")?,
            get_vector: check(self.get_vector, after.get_vector, "get_vector")?,
            set_vector: check(self.set_vector, after.set_vector, "set_vector")?,
            get_neighbors: check(self.get_neighbors, after.get_neighbors, "get_neighbors")?,
            set_neighbors: check(self.set_neighbors, after.set_neighbors, "set_neighbors")?,
            append_neighbors: check(
                self.append_neighbors,
                after.append_neighbors,
                "append_neighbors",
            )?,
        };

        Ok(delta)
    }
}

impl std::fmt::Display for Counters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("query_distance", &self.query_distance);
        kv.push("distance", &self.distance);
        kv.push("get_vector", &self.get_vector);
        kv.push("set_vector", &self.set_vector);
        kv.push("get_neighbors", &self.get_neighbors);
        kv.push("set_neighbors", &self.set_neighbors);
        kv.push("append_neighbors", &self.append_neighbors);
        write!(f, "{}", kv)
    }
}

impl From<integration::counters::CounterSnapshot> for Counters {
    fn from(snapshot: integration::counters::CounterSnapshot) -> Self {
        Self {
            query_distance: snapshot.query_distance,
            distance: snapshot.distance,
            get_vector: snapshot.get_vector,
            set_vector: snapshot.set_vector,
            get_neighbors: snapshot.get_neighbors,
            set_neighbors: snapshot.set_neighbors,
            append_neighbors: snapshot.append_neighbors,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CheckCounters(MaxRelativeChange);

impl CheckCounters {
    pub(crate) fn example() -> Self {
        const TOL: MaxRelativeChange = MaxRelativeChange::new(0.0001, 0.0001).unwrap();
        Self(TOL)
    }

    pub(crate) fn check_hops_and_cmps(&self, old: &KnnSearch, new: &KnnSearch) -> Report {
        let Self(tol) = self;
        let KnnSearch { hops, cmps } = old;

        let fmt = FmtSpec::new();
        let mut builder = Report::builder();
        builder.push(
            "hops",
            Report::equal_with_tolerance(tol, *hops, new.hops, fmt),
        );
        builder.push(
            "cmps",
            Report::equal_with_tolerance(tol, *cmps, new.cmps, fmt),
        );
        builder.finish()
    }

    pub(crate) fn check(&self, old: &Counters, new: &Counters) -> Report {
        let Self(tol) = self;
        let Counters {
            query_distance,
            distance,
            get_vector,
            set_vector,
            get_neighbors,
            set_neighbors,
            append_neighbors,
        } = old;

        let mut builder = Report::builder();
        let fmt = FmtSpec::new().with_precision(3);

        builder.push(
            "query_distance",
            Report::equal_with_tolerance(tol, *query_distance, new.query_distance, fmt),
        );
        builder.push(
            "distance",
            Report::equal_with_tolerance(tol, *distance, new.distance, fmt),
        );
        builder.push(
            "get_vector",
            Report::equal_with_tolerance(tol, *get_vector, new.get_vector, fmt),
        );
        builder.push(
            "set_vector",
            Report::equal_with_tolerance(tol, *set_vector, new.set_vector, fmt),
        );
        builder.push(
            "get_neighbors",
            Report::equal_with_tolerance(tol, *get_neighbors, new.get_neighbors, fmt),
        );
        builder.push(
            "set_neighbors",
            Report::equal_with_tolerance(tol, *set_neighbors, new.set_neighbors, fmt),
        );
        builder.push(
            "append_neighbors",
            Report::equal_with_tolerance(tol, *append_neighbors, new.append_neighbors, fmt),
        );
        builder.finish()
    }
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
    pub(super) recall: KnnRecall,
    pub(super) counters: Counters,
    pub(super) misc: KnnSearch,
}

#[cfg(test)]
impl KnnStats {
    pub(super) fn test_fixture() -> Self {
        Self {
            recall: KnnRecall {
                recall_k: 10,
                recall_n: 10,
                num_queries: 100,
                average: 0.75,
            },
            counters: Counters::test_fixture(),
            misc: KnnSearch::new(),
        }
    }
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
pub(crate) struct CheckKnnStats {
    recall: CheckKnnRecall,
    counters: CheckCounters,
}

impl CheckKnnStats {
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_benchmark_runner::report::Status;

    #[test]
    fn check_counters_rejects_exceeded_tolerance() {
        let tolerance = CheckCounters::example();
        let old = Counters::test_fixture();
        for (query_distance, expected) in [
            (10_000, Status::Pass),
            (10_001, Status::Pass),
            (10_002, Status::Fail),
        ] {
            let new = Counters {
                query_distance,
                ..Counters::test_fixture()
            };
            assert_eq!(
                tolerance.check(&old, &new).status(),
                expected,
                "query_distance = {query_distance}"
            );
        }
    }

    #[test]
    fn check_knn_stats_rejects_recall_changes() {
        let tolerance = CheckKnnStats::example();
        let old = KnnStats::test_fixture();
        for (recall_k, average, expected) in [
            (10, 0.75, Status::Pass),
            (11, 0.75, Status::Fail),
            (10, 0.74, Status::Fail),
        ] {
            let mut new = KnnStats::test_fixture();
            new.recall.recall_k = recall_k;
            new.recall.average = average;
            assert_eq!(
                tolerance.check(&old, &new).status(),
                expected,
                "recall_k = {recall_k}, average = {average}"
            );
        }
    }
}
