/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};
use std::{fmt, io::Write};

use diskann::utils::VectorRepr;
use diskann_benchmark_runner::{
    benchmark::{PassFail, Regression},
    dispatcher::{DispatchRule, FailureScore, MatchScore},
    output::Output,
    utils::{
        datatype::{DataType, Type},
        fmt::Table,
        num::{relative_change, NonNegativeFinite},
    },
    Any, Benchmark, CheckDeserialization, Checker, Checkpoint, Input,
};
use diskann_providers::storage::FileStorageProvider;
use half::f16;

use crate::{
    backend::disk_index::{
        build::{build_disk_index, DiskBuildStats},
        search::{search_disk_index, DiskSearchStats},
    },
    inputs::disk::{DiskIndexLoad, DiskIndexOperation, DiskIndexSource},
};

/// Disk Index
struct DiskIndex<'a, T> {
    input: &'a DiskIndexOperation,
    _vector_type: std::marker::PhantomData<T>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct DiskIndexStats {
    pub(super) build: Option<DiskBuildStats>,
    pub(super) search: DiskSearchStats,
}

impl<'a, T> DiskIndex<'a, T>
where
    T: VectorRepr,
{
    fn new(input: &'a DiskIndexOperation) -> Self {
        Self {
            input,
            _vector_type: std::marker::PhantomData,
        }
    }

    fn run(
        &self,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> Result<DiskIndexStats, anyhow::Error> {
        writeln!(output, "{}", self.input.source)?;
        let (build_stats, index_load) = match &self.input.source {
            DiskIndexSource::Load(load) => Ok((None, (*load).clone())),
            DiskIndexSource::Build(build) => build_disk_index::<T, _>(&FileStorageProvider, build)
                .map(|stats| {
                    (
                        Some(stats),
                        DiskIndexLoad {
                            data_type: build.data_type,
                            load_path: build.save_path.clone(),
                        },
                    )
                }),
        }?;
        if let Some(build_stats) = &build_stats {
            writeln!(output, "{}", build_stats)?;
        }

        writeln!(output, "{}", self.input.search_phase)?;
        let search_stats =
            search_disk_index::<T, _>(&index_load, &self.input.search_phase, &FileStorageProvider)?;
        writeln!(output, "{}", search_stats)?;

        Ok(DiskIndexStats {
            build: build_stats,
            search: search_stats,
        })
    }
}

impl<T> Benchmark for DiskIndex<'static, T>
where
    T: VectorRepr + 'static,
    Type<T>: DispatchRule<DataType>,
{
    type Input = DiskIndexOperation;
    type Output = DiskIndexStats;

    fn try_match(input: &DiskIndexOperation) -> Result<MatchScore, FailureScore> {
        match &input.source {
            DiskIndexSource::Load(load) => Type::<T>::try_match(&load.data_type),
            DiskIndexSource::Build(build) => Type::<T>::try_match(&build.data_type),
        }
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&DiskIndexOperation>,
    ) -> std::fmt::Result {
        match input {
            Some(arg) => match &arg.source {
                DiskIndexSource::Load(load) => Type::<T>::description(f, Some(&load.data_type)),
                DiskIndexSource::Build(build) => Type::<T>::description(f, Some(&build.data_type)),
            },
            None => Type::<T>::description(f, None::<&DataType>),
        }
    }

    fn run(
        input: &DiskIndexOperation,
        checkpoint: Checkpoint<'_>,
        output: &mut dyn Output,
    ) -> anyhow::Result<DiskIndexStats> {
        DiskIndex::<T>::new(input).run(checkpoint, output)
    }
}

////////////////////////////
// Benchmark Registration //
////////////////////////////

pub(super) fn register_benchmarks(benchmarks: &mut diskann_benchmark_runner::registry::Benchmarks) {
    benchmarks.register_regression::<DiskIndex<'static, f32>>("disk-index-f32");
    benchmarks.register_regression::<DiskIndex<'static, f16>>("disk-index-f16");
    benchmarks.register_regression::<DiskIndex<'static, u8>>("disk-index-u8");
    benchmarks.register_regression::<DiskIndex<'static, i8>>("disk-index-i8");
}

/////////////////////////
// Regression Checking //
/////////////////////////

/// Tolerance thresholds for disk-index regression checks.
///
/// Each field specifies the maximum allowed relative increase (for "lower is better" metrics)
/// or decrease (for "higher is better" metrics) before a regression is flagged.
///
/// For example, `recall_regression: 0.01` means recall must not drop by more than 1%.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct DiskIndexTolerance {
    /// Max allowed relative increase in build time (e.g., 0.10 = 10%).
    build_time_regression: NonNegativeFinite,
    /// Max allowed relative decrease in QPS (e.g., 0.10 = 10%).
    qps_regression: NonNegativeFinite,
    /// Max allowed relative decrease in recall (e.g., 0.01 = 1%).
    recall_regression: NonNegativeFinite,
    /// Max allowed relative increase in mean I/Os.
    mean_ios_regression: NonNegativeFinite,
    /// Max allowed relative increase in mean comparisons.
    mean_comps_regression: NonNegativeFinite,
    /// Max allowed relative increase in mean latency.
    mean_latency_regression: NonNegativeFinite,
    /// Max allowed relative increase in p95 latency.
    p95_latency_regression: NonNegativeFinite,
}

impl DiskIndexTolerance {
    const fn tag() -> &'static str {
        "disk-index-tolerance"
    }
}

impl CheckDeserialization for DiskIndexTolerance {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

impl Input for DiskIndexTolerance {
    fn tag() -> &'static str {
        Self::tag()
    }

    fn try_deserialize(
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(Self::deserialize(serialized)?)
    }

    fn example() -> anyhow::Result<serde_json::Value> {
        const DEFAULT: NonNegativeFinite = match NonNegativeFinite::new(0.10) {
            Ok(v) => v,
            Err(_) => panic!("use a non-negative finite value"),
        };
        const RECALL: NonNegativeFinite = match NonNegativeFinite::new(0.01) {
            Ok(v) => v,
            Err(_) => panic!("use a non-negative finite value"),
        };

        Ok(serde_json::to_value(DiskIndexTolerance {
            build_time_regression: DEFAULT,
            qps_regression: DEFAULT,
            recall_regression: RECALL,
            mean_ios_regression: DEFAULT,
            mean_comps_regression: DEFAULT,
            mean_latency_regression: DEFAULT,
            p95_latency_regression: DEFAULT,
        })?)
    }
}

/// Whether a metric improves when its value goes down or up.
#[derive(Clone, Copy)]
enum Direction {
    LowerIsBetter,
    HigherIsBetter,
}

/// A single metric comparison in the regression check.
#[derive(Debug, Serialize)]
struct MetricComparison {
    metric: String,
    before: f64,
    after: f64,
    change_pct: String,
    tolerance_pct: f64,
    passed: bool,
    remark: String,
}

/// Aggregated result of a disk-index regression check.
#[derive(Debug, Serialize)]
struct DiskIndexCheckResult {
    comparisons: Vec<MetricComparison>,
}

impl fmt::Display for DiskIndexCheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header = ["Metric", "Before", "After", "Change", "Tolerance", "Remark"];
        let mut table = Table::new(header, self.comparisons.len());

        for (i, c) in self.comparisons.iter().enumerate() {
            let mut row = table.row(i);
            row.insert(c.metric.clone(), 0);
            row.insert(format!("{:.3}", c.before), 1);
            row.insert(format!("{:.3}", c.after), 2);
            row.insert(c.change_pct.clone(), 3);
            row.insert(format!("{:.1}%", c.tolerance_pct * 100.0), 4);
            if !c.remark.is_empty() {
                row.insert(c.remark.clone(), 5);
            }
        }

        table.fmt(f)
    }
}

/// Check a metric for regression.
///
/// For `LowerIsBetter` metrics (latency, IOs), regression = value increased beyond tolerance.
/// For `HigherIsBetter` metrics (QPS, recall), regression = value decreased beyond tolerance.
fn check_metric(
    name: String,
    direction: Direction,
    before: f64,
    after: f64,
    tolerance: NonNegativeFinite,
    passed: &mut bool,
) -> MetricComparison {
    let (change_pct, remark, metric_passed) = match relative_change(before, after) {
        Ok(change) => {
            let ok = match direction {
                Direction::LowerIsBetter => change <= tolerance.get(),
                Direction::HigherIsBetter => -change <= tolerance.get(),
            };
            if !ok {
                *passed = false;
            }
            (
                format!("{:.3}%", change * 100.0),
                if ok {
                    String::new()
                } else {
                    "REGRESSION".to_string()
                },
                ok,
            )
        }
        Err(e) => {
            *passed = false;
            ("invalid".to_string(), e.to_string(), false)
        }
    };
    MetricComparison {
        metric: name,
        before,
        after,
        change_pct,
        tolerance_pct: tolerance.get(),
        passed: metric_passed,
        remark,
    }
}

impl<T> Regression for DiskIndex<'static, T>
where
    T: VectorRepr + 'static,
    Type<T>: DispatchRule<DataType>,
{
    type Tolerances = DiskIndexTolerance;
    type Pass = DiskIndexCheckResult;
    type Fail = DiskIndexCheckResult;

    fn check(
        tolerances: &DiskIndexTolerance,
        _input: &DiskIndexOperation,
        before: &DiskIndexStats,
        after: &DiskIndexStats,
    ) -> anyhow::Result<PassFail<DiskIndexCheckResult, DiskIndexCheckResult>> {
        use Direction::{HigherIsBetter, LowerIsBetter};

        let mut passed = true;
        let mut comparisons = Vec::new();

        // Check build time if both sides have it
        if let (Some(b_build), Some(a_build)) = (&before.build, &after.build) {
            comparisons.push(check_metric(
                "build_time".to_string(),
                LowerIsBetter,
                b_build.build_time_seconds(),
                a_build.build_time_seconds(),
                tolerances.build_time_regression,
                &mut passed,
            ));
        }

        // Check search metrics for each matching search_l
        anyhow::ensure!(
            before.search.search_results_per_l.len() == after.search.search_results_per_l.len(),
            "before has {} search_l entries but after has {}",
            before.search.search_results_per_l.len(),
            after.search.search_results_per_l.len(),
        );

        for (b_sr, a_sr) in before
            .search
            .search_results_per_l
            .iter()
            .zip(after.search.search_results_per_l.iter())
        {
            anyhow::ensure!(
                b_sr.search_l == a_sr.search_l,
                "search_l mismatch: before={} after={}",
                b_sr.search_l,
                a_sr.search_l,
            );

            // Prefix metric names with L value when multiple search_l entries exist.
            let prefix = if before.search.search_results_per_l.len() > 1 {
                format!("L{}:", b_sr.search_l)
            } else {
                String::new()
            };

            comparisons.push(check_metric(
                format!("{prefix}qps"),
                HigherIsBetter,
                b_sr.qps as f64,
                a_sr.qps as f64,
                tolerances.qps_regression,
                &mut passed,
            ));
            comparisons.push(check_metric(
                format!("{prefix}recall"),
                HigherIsBetter,
                b_sr.recall as f64,
                a_sr.recall as f64,
                tolerances.recall_regression,
                &mut passed,
            ));
            comparisons.push(check_metric(
                format!("{prefix}mean_latency"),
                LowerIsBetter,
                b_sr.mean_latency,
                a_sr.mean_latency,
                tolerances.mean_latency_regression,
                &mut passed,
            ));
            comparisons.push(check_metric(
                format!("{prefix}p95_latency"),
                LowerIsBetter,
                b_sr.p95_latency.as_f64(),
                a_sr.p95_latency.as_f64(),
                tolerances.p95_latency_regression,
                &mut passed,
            ));
            comparisons.push(check_metric(
                format!("{prefix}mean_ios"),
                LowerIsBetter,
                b_sr.mean_ios,
                a_sr.mean_ios,
                tolerances.mean_ios_regression,
                &mut passed,
            ));
            comparisons.push(check_metric(
                format!("{prefix}mean_comparisons"),
                LowerIsBetter,
                b_sr.mean_comparisons,
                a_sr.mean_comparisons,
                tolerances.mean_comps_regression,
                &mut passed,
            ));
        }

        let result = DiskIndexCheckResult { comparisons };

        if passed {
            Ok(PassFail::Pass(result))
        } else {
            Ok(PassFail::Fail(result))
        }
    }
}
