/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::utils::IntoUsize;
use diskann_benchmark_core::recall::{RangeMetrics, RecallMetrics};
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

///////////
// Stats //
///////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Stats {
    pub(super) quality: Quality,
    pub(super) counters: Counters,
    pub(super) misc: MiscStats,
}

#[cfg(test)]
impl Stats {
    pub(super) fn test_fixture() -> Self {
        Self {
            quality: Quality::Recall(KnnRecall {
                recall_k: 10,
                recall_n: 10,
                num_queries: 100,
                average: 0.75,
            }),
            counters: Counters::test_fixture(),
            misc: MiscStats::new(),
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            quality,
            counters,
            misc,
        } = self;

        let mut kv = KeyValue::new();
        kv.push("counters", counters);
        kv.push("quality", quality);
        kv.push("misc", misc);
        write!(f, "{}", kv)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CheckStats {
    quality: CheckQuality,
    counters: CheckCounters,
}

impl CheckStats {
    pub(crate) fn example() -> Self {
        Self {
            quality: CheckQuality::example(),
            counters: CheckCounters::example(),
        }
    }

    pub(crate) fn check(&self, old: &Stats, new: &Stats) -> Report {
        let Self {
            quality: check_quality,
            counters: check_counters,
        } = self;
        let Stats {
            quality,
            counters,
            misc,
        } = old;

        let mut builder = Report::builder();
        builder.push("quality", check_quality.check(quality, &new.quality));
        builder.push("counters", check_counters.check(counters, &new.counters));
        builder.push("misc", check_counters.check_hops_and_cmps(misc, &new.misc));
        builder.finish()
    }
}

//---------------//
// Hops and Cmps //
//---------------//

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MiscStats {
    hops: usize,
    cmps: usize,
}

impl MiscStats {
    pub(crate) fn new() -> Self {
        Self { hops: 0, cmps: 0 }
    }
}

impl From<diskann::graph::index::SearchStats> for MiscStats {
    fn from(stats: diskann::graph::index::SearchStats) -> Self {
        Self {
            hops: stats.hops.into_usize(),
            cmps: stats.cmps.into_usize(),
        }
    }
}

impl std::ops::AddAssign for MiscStats {
    fn add_assign(&mut self, rhs: Self) {
        self.hops = self.hops.wrapping_add(rhs.hops);
        self.cmps = self.cmps.wrapping_add(rhs.cmps);
    }
}

impl std::fmt::Display for MiscStats {
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

    pub(crate) fn check_hops_and_cmps(&self, old: &MiscStats, new: &MiscStats) -> Report {
        let Self(tol) = self;
        let MiscStats { hops, cmps } = old;

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

//-----------//
// KnnRecall //
//-----------//

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

//---------------//
// Range Quality //
//---------------//

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RangeQuality {
    num_queries: usize,
    pooled: PooledCounts,
    mean_precision: QueryMean,
    mean_recall: QueryMean,
    empty_groundtruth: EmptyGroundtruth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PooledCounts {
    groundtruth: usize,
    returned: usize,
    matches: usize,
    precision: f64,
    recall: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueryMean {
    value: Option<f64>,
    num_queries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EmptyGroundtruth {
    num_queries: usize,
    queries_with_results: usize,
    returned: usize,
    false_positive_rate: Option<f64>,
}

impl From<RangeMetrics> for RangeQuality {
    fn from(metrics: RangeMetrics) -> Self {
        Self {
            num_queries: metrics.num_queries,
            pooled: PooledCounts {
                groundtruth: metrics.pooled.groundtruth,
                returned: metrics.pooled.returned,
                matches: metrics.pooled.matches,
                precision: metrics.pooled.precision(),
                recall: metrics.pooled.recall(),
            },
            mean_precision: QueryMean {
                value: metrics.mean_precision.value,
                num_queries: metrics.mean_precision.num_queries,
            },
            mean_recall: QueryMean {
                value: metrics.mean_recall.value,
                num_queries: metrics.mean_recall.num_queries,
            },
            empty_groundtruth: EmptyGroundtruth {
                num_queries: metrics.empty_groundtruth.num_queries,
                queries_with_results: metrics.empty_groundtruth.queries_with_results,
                returned: metrics.empty_groundtruth.returned,
                false_positive_rate: metrics.empty_groundtruth.false_positive_rate(),
            },
        }
    }
}

impl std::fmt::Display for RangeQuality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "precision = {:.4}, recall = {:.4}, num_queries = {}, num_groundtruth_values = {}, num_result_values = {}, num_matches = {}",
            self.pooled.precision,
            self.pooled.recall,
            self.num_queries,
            self.pooled.groundtruth,
            self.pooled.returned,
            self.pooled.matches
        )
    }
}

//---------//
// Quality //
//---------//

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Quality {
    Recall(KnnRecall),
    Range(RangeQuality),
}

impl From<KnnRecall> for Quality {
    fn from(recall: KnnRecall) -> Self {
        Self::Recall(recall)
    }
}

impl From<RangeQuality> for Quality {
    fn from(metrics: RangeQuality) -> Self {
        Self::Range(metrics)
    }
}

impl std::fmt::Display for Quality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut kv = KeyValue::new();
        match self {
            Self::Recall(recall) => kv.push_eager_nested("knn-recall", recall),
            Self::Range(metrics) => kv.push_eager_nested("range", metrics),
        }
        kv.fmt(f)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CheckQuality {
    change: MaxAbsoluteChange,
}

impl CheckQuality {
    fn example() -> Self {
        const CHANGE: MaxAbsoluteChange = MaxAbsoluteChange::try_from_f64(0.001).unwrap();
        Self { change: CHANGE }
    }

    fn check(&self, old: &Quality, new: &Quality) -> Report {
        let Self { change } = self;

        let fmt = FmtSpec::new().with_precision(4);
        match (old, new) {
            (Quality::Recall(old), Quality::Recall(new)) => {
                let KnnRecall {
                    recall_k,
                    recall_n,
                    num_queries,
                    average,
                } = old;

                let mut builder = Report::builder();
                builder.push("recall_k", Report::equal(*recall_k, new.recall_k));
                builder.push("recall_n", Report::equal(*recall_n, new.recall_n));
                builder.push("num_queries", Report::equal(*num_queries, new.num_queries));
                builder.push(
                    "average",
                    Report::equal_with_tolerance(change, *average, new.average, fmt),
                );
                builder.finish()
            }
            (Quality::Range(old), Quality::Range(new)) => {
                let RangeQuality {
                    num_queries,
                    pooled,
                    mean_precision,
                    mean_recall,
                    empty_groundtruth,
                } = old;

                let mut builder = Report::builder();
                builder.push("num_queries", Report::equal(*num_queries, new.num_queries));
                builder.push(
                    "num_groundtruth_values",
                    Report::equal(pooled.groundtruth, new.pooled.groundtruth),
                );
                builder.push(
                    "precision",
                    Report::equal_with_tolerance(
                        change,
                        pooled.precision,
                        new.pooled.precision,
                        fmt,
                    ),
                );
                builder.push(
                    "recall",
                    Report::equal_with_tolerance(change, pooled.recall, new.pooled.recall, fmt),
                );
                let check_optional = |old: Option<f64>, new: Option<f64>| match (old, new) {
                    (Some(old), Some(new)) => Report::equal_with_tolerance(change, old, new, fmt),
                    _ => Report::equal(old.is_some(), new.is_some()),
                };
                builder.push(
                    "mean_precision",
                    check_optional(mean_precision.value, new.mean_precision.value),
                );
                builder.push(
                    "mean_recall",
                    check_optional(mean_recall.value, new.mean_recall.value),
                );
                builder.push(
                    "num_empty_groundtruth_queries",
                    Report::equal(
                        empty_groundtruth.num_queries,
                        new.empty_groundtruth.num_queries,
                    ),
                );
                builder.push(
                    "empty_groundtruth_false_positive_rate",
                    check_optional(
                        empty_groundtruth.false_positive_rate,
                        new.empty_groundtruth.false_positive_rate,
                    ),
                );
                builder.finish()
            }
            (Quality::Recall(_), Quality::Range(_)) => {
                Report::error("old stats were KNN-recall but new were range precision/recall")
            }
            (Quality::Range(_), Quality::Recall(_)) => {
                Report::error("old stats were range precision/recall but new were KNN-recall")
            }
        }
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
    fn check_range_quality() {
        let tolerance = CheckQuality::example();
        let fixture = || RangeQuality {
            num_queries: 2,
            pooled: PooledCounts {
                groundtruth: 4,
                returned: 5,
                matches: 3,
                precision: 0.6,
                recall: 0.75,
            },
            mean_precision: QueryMean {
                value: Some(2.0 / 3.0),
                num_queries: 2,
            },
            mean_recall: QueryMean {
                value: Some(0.75),
                num_queries: 2,
            },
            empty_groundtruth: EmptyGroundtruth {
                num_queries: 0,
                queries_with_results: 0,
                returned: 0,
                false_positive_rate: None,
            },
        };
        let old = Quality::Range(fixture());
        for (precision, recall, expected) in [
            (0.6, 0.75, Status::Pass),
            (0.59, 0.75, Status::Fail),
            (0.6, 0.74, Status::Fail),
        ] {
            let new = Quality::Range(RangeQuality {
                pooled: PooledCounts {
                    precision,
                    recall,
                    ..fixture().pooled
                },
                ..fixture()
            });
            assert_eq!(tolerance.check(&old, &new).status(), expected);
        }

        let new = Quality::Range(RangeQuality {
            pooled: PooledCounts {
                groundtruth: 5,
                ..fixture().pooled
            },
            ..fixture()
        });
        assert_eq!(tolerance.check(&old, &new).status(), Status::Fail);

        for (mean_precision, mean_recall, empty_rate, expected) in [
            (Some(2.0 / 3.0), Some(0.75), None, Status::Pass),
            (Some(0.65), Some(0.75), None, Status::Fail),
            (Some(2.0 / 3.0), Some(0.74), None, Status::Fail),
            (None, Some(0.75), None, Status::Fail),
            (Some(2.0 / 3.0), None, None, Status::Fail),
            (Some(2.0 / 3.0), Some(0.75), Some(0.0), Status::Fail),
        ] {
            let new = Quality::Range(RangeQuality {
                mean_precision: QueryMean {
                    value: mean_precision,
                    ..fixture().mean_precision
                },
                mean_recall: QueryMean {
                    value: mean_recall,
                    ..fixture().mean_recall
                },
                empty_groundtruth: EmptyGroundtruth {
                    false_positive_rate: empty_rate,
                    ..fixture().empty_groundtruth
                },
                ..fixture()
            });
            assert_eq!(tolerance.check(&old, &new).status(), expected);
        }

        let empty_fixture = || RangeQuality {
            num_queries: 12,
            pooled: PooledCounts {
                returned: 8,
                precision: 3.0 / 8.0,
                ..fixture().pooled
            },
            mean_precision: QueryMean {
                value: Some(4.0 / 9.0),
                num_queries: 3,
            },
            empty_groundtruth: EmptyGroundtruth {
                num_queries: 10,
                queries_with_results: 1,
                returned: 3,
                false_positive_rate: Some(0.1),
            },
            ..fixture()
        };
        let old_empty = Quality::Range(empty_fixture());
        for rate in [Some(0.2), None] {
            let new = Quality::Range(RangeQuality {
                empty_groundtruth: EmptyGroundtruth {
                    false_positive_rate: rate,
                    ..empty_fixture().empty_groundtruth
                },
                ..empty_fixture()
            });
            assert_eq!(tolerance.check(&old_empty, &new).status(), Status::Fail);
        }
        let new = Quality::Range(RangeQuality {
            empty_groundtruth: EmptyGroundtruth {
                num_queries: 11,
                ..empty_fixture().empty_groundtruth
            },
            ..empty_fixture()
        });
        assert_eq!(tolerance.check(&old_empty, &new).status(), Status::Fail);

        let json = serde_json::to_value(&old).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "range": {
                    "num_queries": 2,
                    "pooled": {
                        "groundtruth": 4,
                        "returned": 5,
                        "matches": 3,
                        "precision": 0.6,
                        "recall": 0.75
                    },
                    "mean_precision": { "value": 2.0 / 3.0, "num_queries": 2 },
                    "mean_recall": { "value": 0.75, "num_queries": 2 },
                    "empty_groundtruth": {
                        "num_queries": 0,
                        "queries_with_results": 0,
                        "returned": 0,
                        "false_positive_rate": null
                    }
                }
            })
        );
        let decoded: Quality = serde_json::from_value(json).unwrap();
        assert_eq!(tolerance.check(&old, &decoded).status(), Status::Pass);
    }

    #[test]
    fn range_quality_from_metrics() {
        let metrics = diskann_benchmark_core::recall::range_metrics(
            &vec![vec![1, 8], vec![3], vec![], vec![7, 7, 8], vec![]],
            &vec![vec![1, 2], vec![3, 4, 5, 6], vec![], vec![], vec![9]],
        )
        .unwrap();
        let quality = RangeQuality::from(metrics);
        assert_eq!(
            serde_json::to_value(&quality).unwrap(),
            serde_json::json!({
                "num_queries": 5,
                "pooled": {
                    "groundtruth": 7, "returned": 5, "matches": 2,
                    "precision": 2.0 / 5.0, "recall": 2.0 / 7.0
                },
                "mean_precision": { "value": 0.5, "num_queries": 3 },
                "mean_recall": { "value": 0.25, "num_queries": 3 },
                "empty_groundtruth": {
                    "num_queries": 2, "queries_with_results": 1,
                    "returned": 2, "false_positive_rate": 0.5
                }
            })
        );
        assert_eq!(
            quality.to_string(),
            "precision = 0.4000, recall = 0.2857, num_queries = 5, num_groundtruth_values = 7, num_result_values = 5, num_matches = 2"
        );
    }

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

    // #[test]
    // fn check_knn_stats_rejects_recall_changes() {
    //     let tolerance = CheckStats::example();
    //     let old = Stats::test_fixture();
    //     for (recall_k, average, expected) in [
    //         (10, 0.75, Status::Pass),
    //         (11, 0.75, Status::Fail),
    //         (10, 0.74, Status::Fail),
    //     ] {
    //         let mut new = Stats::test_fixture();
    //         new.recall.recall_k = recall_k;
    //         new.recall.average = average;
    //         assert_eq!(
    //             tolerance.check(&old, &new).status(),
    //             expected,
    //             "recall_k = {recall_k}, average = {average}"
    //         );
    //     }
    // }
}
