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

/// A single metric comparison in the regression check.
#[derive(Debug, Serialize)]
struct MetricComparison {
    metric: &'static str,
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
    search_l: u32,
    comparisons: Vec<MetricComparison>,
}

impl fmt::Display for DiskIndexCheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "  Search L={}: {:>15} {:>15} {:>12} {:>12}   {}",
            self.search_l, "Before", "After", "Change", "Tolerance", "Remark"
        )?;
        writeln!(
            f,
            "  {}",
            "=".repeat(90)
        )?;
        for c in &self.comparisons {
            writeln!(
                f,
                "  {:>20}, {:>14.3}, {:>14.3}, {:>11}, {:>11.1}%,   {}",
                c.metric, c.before, c.after, c.change_pct, c.tolerance_pct * 100.0, c.remark
            )?;
        }
        Ok(())
    }
}

/// Check a "lower is better" metric (latency, IOs, comparisons).
/// Regression = value increased beyond tolerance.
fn check_lower_is_better(
    name: &'static str,
    before: f64,
    after: f64,
    tolerance: NonNegativeFinite,
    passed: &mut bool,
) -> MetricComparison {
    let (change_pct, remark, metric_passed) = match relative_change(before, after) {
        Ok(change) => {
            let ok = change <= tolerance.get();
            if !ok {
                *passed = false;
            }
            (
                format!("{:.3}%", change * 100.0),
                if ok { String::new() } else { "REGRESSION".to_string() },
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

/// Check a "higher is better" metric (QPS, recall).
/// Regression = value decreased beyond tolerance.
fn check_higher_is_better(
    name: &'static str,
    before: f64,
    after: f64,
    tolerance: NonNegativeFinite,
    passed: &mut bool,
) -> MetricComparison {
    // Flip before/after so that a decrease becomes a positive relative_change
    let (change_pct, remark, metric_passed) = match relative_change(before, after) {
        Ok(change) => {
            // For higher-is-better, a negative change is a regression
            let ok = -change <= tolerance.get();
            if !ok {
                *passed = false;
            }
            (
                format!("{:.3}%", change * 100.0),
                if ok { String::new() } else { "REGRESSION".to_string() },
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
        let mut passed = true;
        let mut comparisons = Vec::new();

        // Check build time if both sides have it
        if let (Some(b_build), Some(a_build)) = (&before.build, &after.build) {
            let b_time = b_build.build_time_seconds();
            let a_time = a_build.build_time_seconds();
            comparisons.push(check_lower_is_better(
                "build_time",
                b_time,
                a_time,
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

            comparisons.push(check_higher_is_better(
                "qps",
                b_sr.qps as f64,
                a_sr.qps as f64,
                tolerances.qps_regression,
                &mut passed,
            ));
            comparisons.push(check_higher_is_better(
                "recall",
                b_sr.recall as f64,
                a_sr.recall as f64,
                tolerances.recall_regression,
                &mut passed,
            ));
            comparisons.push(check_lower_is_better(
                "mean_latency",
                b_sr.mean_latency,
                a_sr.mean_latency,
                tolerances.mean_latency_regression,
                &mut passed,
            ));
            comparisons.push(check_lower_is_better(
                "p95_latency",
                b_sr.p95_latency.as_f64(),
                a_sr.p95_latency.as_f64(),
                tolerances.p95_latency_regression,
                &mut passed,
            ));
            comparisons.push(check_lower_is_better(
                "mean_ios",
                b_sr.mean_ios,
                a_sr.mean_ios,
                tolerances.mean_ios_regression,
                &mut passed,
            ));
            comparisons.push(check_lower_is_better(
                "mean_comparisons",
                b_sr.mean_comparisons,
                a_sr.mean_comparisons,
                tolerances.mean_comps_regression,
                &mut passed,
            ));
        }

        let search_l = before
            .search
            .search_results_per_l
            .first()
            .map(|s| s.search_l)
            .unwrap_or(0);
        let result = DiskIndexCheckResult {
            search_l,
            comparisons,
        };

        if passed {
            Ok(PassFail::Pass(result))
        } else {
            Ok(PassFail::Fail(result))
        }
    }
}
