/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};
use std::{fmt, io::Write};

use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore, PassFail, Regression},
    output::Output,
    utils::{
        fmt::Table,
        num::{relative_change, NonNegativeFinite},
    },
    Benchmark, Checker, Checkpoint, Input, Registry,
};

use crate::{
    ivf::{
        build::{build_ivf_index, IvfBuildStats},
        search::{search_ivf_index, IvfSearchStats},
    },
    inputs::ivf::{IvfLoad, IvfOperation, IvfSource},
};

/// IVF benchmark dispatcher.
struct Ivf;

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct IvfStats {
    pub(super) build: Option<IvfBuildStats>,
    pub(super) search: IvfSearchStats,
}

impl Benchmark for Ivf {
    type Input = IvfOperation;
    type Output = IvfStats;

    fn try_match(&self, _input: &IvfOperation) -> Result<MatchScore, FailureScore> {
        // IVF only supports f32 for now; always match.
        Ok(MatchScore(0))
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&IvfOperation>,
    ) -> std::fmt::Result {
        match input {
            Some(arg) => {
                let mode = match &arg.source {
                    IvfSource::Load(_) => "load",
                    IvfSource::Build(_) => "build",
                };
                write!(f, "IVF-f32 ({})", mode)
            }
            None => write!(f, "IVF-f32"),
        }
    }

    fn run(
        &self,
        input: &IvfOperation,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<IvfStats> {
        writeln!(output, "{}", input.source)?;

        let (build_stats, index_load) = match &input.source {
            IvfSource::Load(load) => Ok((None, (*load).clone())),
            IvfSource::Build(build) => build_ivf_index(build).map(|stats| {
                (
                    Some(stats),
                    IvfLoad {
                        data_type: build.data_type,
                        load_path: build.save_path.clone(),
                    },
                )
            }),
        }?;

        if let Some(build_stats) = &build_stats {
            writeln!(output, "{}", build_stats)?;
        }

        writeln!(output, "{}", input.search_phase)?;
        let search_stats = search_ivf_index(&index_load, &input.search_phase)?;
        writeln!(output, "{}", search_stats)?;

        Ok(IvfStats {
            build: build_stats,
            search: search_stats,
        })
    }
}

////////////////////////////
// Benchmark Registration //
////////////////////////////

pub(super) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register_regression("ivf-f32", Ivf)?;
    Ok(())
}

/////////////////////////
// Regression Checking //
/////////////////////////

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct IvfTolerance {
    build_time_regression: NonNegativeFinite,
    qps_regression: NonNegativeFinite,
    recall_regression: NonNegativeFinite,
    mean_ios_regression: NonNegativeFinite,
    mean_comps_regression: NonNegativeFinite,
    mean_latency_regression: NonNegativeFinite,
    p95_latency_regression: NonNegativeFinite,
}

impl IvfTolerance {
    const fn tag() -> &'static str {
        "ivf-tolerance"
    }
}

impl Input for IvfTolerance {
    type Raw = Self;

    fn tag() -> &'static str {
        Self::tag()
    }

    fn from_raw(raw: Self::Raw, _checker: &mut Checker) -> anyhow::Result<Self> {
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self {
        const DEFAULT: NonNegativeFinite = match NonNegativeFinite::new(0.10) {
            Ok(v) => v,
            Err(_) => panic!("use a non-negative finite value"),
        };
        const RECALL: NonNegativeFinite = match NonNegativeFinite::new(0.01) {
            Ok(v) => v,
            Err(_) => panic!("use a non-negative finite value"),
        };

        IvfTolerance {
            build_time_regression: DEFAULT,
            qps_regression: DEFAULT,
            recall_regression: RECALL,
            mean_ios_regression: DEFAULT,
            mean_comps_regression: DEFAULT,
            mean_latency_regression: DEFAULT,
            p95_latency_regression: DEFAULT,
        }
    }
}

#[derive(Clone, Copy)]
enum Direction {
    LowerIsBetter,
    HigherIsBetter,
}

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

#[derive(Debug, Serialize)]
struct IvfCheckResult {
    comparisons: Vec<MetricComparison>,
}

impl fmt::Display for IvfCheckResult {
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

impl Regression for Ivf {
    type Tolerances = IvfTolerance;
    type Pass = IvfCheckResult;
    type Fail = IvfCheckResult;

    fn check(
        &self,
        tolerances: &IvfTolerance,
        _input: &IvfOperation,
        before: &IvfStats,
        after: &IvfStats,
    ) -> anyhow::Result<PassFail<IvfCheckResult, IvfCheckResult>> {
        use Direction::{HigherIsBetter, LowerIsBetter};

        let mut passed = true;
        let mut comparisons = Vec::new();

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

        anyhow::ensure!(
            before.search.search_results_per_nprobe.len()
                == after.search.search_results_per_nprobe.len(),
            "before has {} nprobe entries but after has {}",
            before.search.search_results_per_nprobe.len(),
            after.search.search_results_per_nprobe.len(),
        );

        for (b_sr, a_sr) in before
            .search
            .search_results_per_nprobe
            .iter()
            .zip(after.search.search_results_per_nprobe.iter())
        {
            anyhow::ensure!(
                b_sr.search_l == a_sr.search_l,
                "nprobe mismatch: before={} after={}",
                b_sr.search_l,
                a_sr.search_l,
            );

            let prefix = if before.search.search_results_per_nprobe.len() > 1 {
                format!("np{}:", b_sr.search_l)
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

        let result = IvfCheckResult { comparisons };

        if passed {
            Ok(PassFail::Pass(result))
        } else {
            Ok(PassFail::Fail(result))
        }
    }
}
