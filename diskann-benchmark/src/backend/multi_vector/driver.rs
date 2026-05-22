/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Timing harness, data fixtures, and result types for multi-vector benchmarks.

use diskann_benchmark_runner::{
    utils::{
        fmt::Table,
        num::{relative_change, NonNegativeFinite},
        percentiles, MicroSeconds,
    },
    Checker, Input,
};
use diskann_quantization::multi_vector::{Mat, MatRef, MaxSimKernel, Standard};
use rand::{
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    SeedableRng,
};
use serde::{Deserialize, Serialize};

use crate::inputs::multi_vector::Run;
use crate::utils::DisplayWrapper;

//////////////////////
// Tolerance        //
//////////////////////

/// Maximum allowed relative regression in min latency.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct MultiVectorTolerance {
    pub(super) min_time_regression: NonNegativeFinite,
}

impl Input for MultiVectorTolerance {
    type Raw = Self;

    fn tag() -> &'static str {
        "multi-vector-tolerance"
    }

    fn from_raw(raw: Self::Raw, _checker: &mut Checker) -> anyhow::Result<Self> {
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self {
        const EXAMPLE: NonNegativeFinite = match NonNegativeFinite::new(0.05) {
            Ok(v) => v,
            Err(_) => panic!("use a non-negative finite please"),
        };

        MultiVectorTolerance {
            min_time_regression: EXAMPLE,
        }
    }
}

///////////////////
// Data fixtures //
///////////////////

/// Random query / doc fixture for a single benchmark run.
pub(super) struct Data<T: Copy> {
    pub(super) queries: Mat<Standard<T>>,
    pub(super) docs: Mat<Standard<T>>,
}

impl<T: Copy> Data<T>
where
    StandardUniform: Distribution<T>,
{
    pub(super) fn new(run: &Run) -> Self {
        let mut rng = StdRng::seed_from_u64(0x12345);
        let queries = Mat::from_fn(
            Standard::new(run.num_query_vectors.get(), run.dim.get()).unwrap(),
            || StandardUniform.sample(&mut rng),
        );
        let docs = Mat::from_fn(
            Standard::new(run.num_doc_vectors.get(), run.dim.get()).unwrap(),
            || StandardUniform.sample(&mut rng),
        );
        Self { queries, docs }
    }
}

//////////////////////
// Distance trait   //
//////////////////////

pub(super) trait Distance<T: Copy> {
    fn max_sim(&self, doc: MatRef<'_, Standard<T>>, scores: &mut [f32]);
}

pub(super) struct BoxedKernel<T: Copy>(pub(super) Box<dyn MaxSimKernel<T>>);

impl<T: Copy> Distance<T> for BoxedKernel<T> {
    fn max_sim(&self, doc: MatRef<'_, Standard<T>>, scores: &mut [f32]) {
        let nq = self.0.nrows();
        assert_eq!(
            scores.len(),
            nq,
            "scores buffer not right size: {} != {}",
            scores.len(),
            nq
        );
        if doc.num_vectors() == 0 {
            return;
        }
        self.0.compute_max_sim(doc, scores);
    }
}

//////////////////////
// Timing harness   //
//////////////////////

fn run_loops(run: &Run, body: &mut dyn FnMut()) -> RunResult {
    let mut latencies = Vec::with_capacity(run.num_measurements.get());

    for _ in 0..run.num_measurements.get() {
        let start = std::time::Instant::now();
        for _ in 0..run.loops_per_measurement.get() {
            body();
        }
        latencies.push(start.elapsed().into());
    }

    let percentiles = percentiles::compute_percentiles(&mut latencies).unwrap();
    RunResult {
        run: run.clone(),
        latencies,
        percentiles,
    }
}

pub(super) fn run_with_distance<T: Copy>(
    run: &Run,
    doc: MatRef<'_, Standard<T>>,
    dist: &dyn Distance<T>,
) -> RunResult {
    let mut scores = vec![0.0f32; run.num_query_vectors.get()];
    run_loops(run, &mut || {
        dist.max_sim(doc, &mut scores);
        std::hint::black_box(&mut scores);
    })
}

//////////////////////
// Result types     //
//////////////////////

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct RunResult {
    pub(super) run: Run,
    pub(super) latencies: Vec<MicroSeconds>,
    pub(super) percentiles: percentiles::Percentiles<MicroSeconds>,
}

impl RunResult {
    pub(super) fn computations_per_latency(&self) -> usize {
        self.run.num_query_vectors.get()
            * self.run.num_doc_vectors.get()
            * self.run.loops_per_measurement.get()
    }
}

impl std::fmt::Display for DisplayWrapper<'_, [RunResult]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return Ok(());
        }

        writeln!(
            f,
            "ns/IP = time per (query, doc) inner-product call (~ linear in Dim)"
        )?;

        let header = [
            "Q",
            "D",
            "Dim",
            "Min Time (ns/IP @ Dim)",
            "Mean Time (ns/IP @ Dim)",
            "Loops",
            "Measurements",
        ];

        let mut table = Table::new(header, self.len());

        self.iter().enumerate().for_each(|(row, r)| {
            let mut row = table.row(row);

            let min_latency = r
                .latencies
                .iter()
                .min()
                .copied()
                .unwrap_or(MicroSeconds::new(u64::MAX));
            let mean_latency = r.percentiles.mean;

            let computations_per_latency = r.computations_per_latency() as f64;
            let min_time = min_latency.as_f64() / computations_per_latency * 1000.0;
            let mean_time = mean_latency / computations_per_latency * 1000.0;

            row.insert(r.run.num_query_vectors, 0);
            row.insert(r.run.num_doc_vectors, 1);
            row.insert(r.run.dim, 2);
            row.insert(format!("{:.3}", min_time), 3);
            row.insert(format!("{:.3}", mean_time), 4);
            row.insert(r.run.loops_per_measurement, 5);
            row.insert(r.run.num_measurements, 6);
        });

        table.fmt(f)
    }
}

//////////////////////
// Regression Check //
//////////////////////

#[derive(Debug, Serialize)]
pub(super) struct Comparison {
    pub(super) run: Run,
    pub(super) tolerance: MultiVectorTolerance,
    pub(super) before_min: f64,
    pub(super) after_min: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct CheckResult {
    pub(super) checks: Vec<Comparison>,
}

impl std::fmt::Display for CheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = [
            "Q",
            "D",
            "Dim",
            "Min Before (ns/IP @ Dim)",
            "Min After (ns/IP @ Dim)",
            "Change (%)",
            "Remark",
        ];

        let mut table = Table::new(header, self.checks.len());

        for (i, c) in self.checks.iter().enumerate() {
            let mut row = table.row(i);
            let change = relative_change(c.before_min, c.after_min);

            row.insert(c.run.num_query_vectors, 0);
            row.insert(c.run.num_doc_vectors, 1);
            row.insert(c.run.dim, 2);
            row.insert(format!("{:.3}", c.before_min), 3);
            row.insert(format!("{:.3}", c.after_min), 4);
            match change {
                Ok(change) => {
                    row.insert(format!("{:.3} %", change * 100.0), 5);
                    if change > c.tolerance.min_time_regression.get() {
                        row.insert("FAIL", 6);
                    }
                }
                Err(err) => {
                    row.insert("invalid", 5);
                    row.insert(err, 6);
                }
            }
        }

        table.fmt(f)
    }
}
