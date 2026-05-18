/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared benchmark infrastructure for multi-vector kernels.
//!
//! Houses the timing harness ([`run_loops`]), data fixtures ([`Data`]), result
//! types ([`RunResult`], [`Comparison`], [`CheckResult`]), and the trait-object
//! [`Distance<T>`] boundary the driver dispatches through. None of the
//! contents are kernel-aware.

use diskann_benchmark_runner::{
    utils::{
        fmt::Table,
        num::{relative_change, NonNegativeFinite},
        percentiles, MicroSeconds,
    },
    Any, CheckDeserialization, Checker, Input,
};
use diskann_quantization::multi_vector::{Mat, MatRef, MaxSimKernel, Standard};
use rand::{
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    SeedableRng,
};
use serde::{Deserialize, Serialize};

use crate::inputs::multi_vector::Run;

//////////////////////
// Tolerance        //
//////////////////////

/// Tolerance thresholds for multi-vector benchmark regression detection.
///
/// Each field specifies the maximum allowed relative increase in the corresponding metric.
/// For example, a value of `0.05` means a 5% increase is tolerated.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct MultiVectorTolerance {
    pub(super) min_time_regression: NonNegativeFinite,
}

impl CheckDeserialization for MultiVectorTolerance {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

impl Input for MultiVectorTolerance {
    fn tag() -> &'static str {
        "multi-vector-tolerance"
    }

    fn try_deserialize(
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(Self::deserialize(serialized)?)
    }

    fn example() -> anyhow::Result<serde_json::Value> {
        const EXAMPLE: NonNegativeFinite = match NonNegativeFinite::new(0.05) {
            Ok(v) => v,
            Err(_) => panic!("use a non-negative finite please"),
        };

        Ok(serde_json::to_value(MultiVectorTolerance {
            min_time_regression: EXAMPLE,
        })?)
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

/// Object-safe distance executor. The library factory's `Erase` visitor
/// already produces a `Box<dyn MaxSimKernel<T>>`, but the driver wants its
/// own narrow trait so the kernel + its assertions are tucked inside one
/// vtable boundary. Simpler than threading `Box<dyn MaxSimKernel<T>>`
/// generically through the timing harness.
pub(super) trait Distance<T: Copy> {
    fn max_sim(&self, doc: MatRef<'_, Standard<T>>, scores: &mut [f32]);
}

/// Distance executor wrapping a boxed `MaxSimKernel<T>` from the library
/// factory. One vtable hop in the hot loop.
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

fn run_loops<F>(run: &Run, mut body: F) -> RunResult
where
    F: FnMut(),
{
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

/// Shared loop nest. The trait-object dispatch happens once per outer iteration
/// of `run_loops`; the work inside each `max_sim` call is O(Q·D·dim), so the
/// vtable hop is in the noise.
pub(super) fn run_with_distance<T: Copy>(
    run: &Run,
    doc: MatRef<'_, Standard<T>>,
    dist: &dyn Distance<T>,
) -> RunResult {
    let mut scores = vec![0.0f32; run.num_query_vectors.get()];
    run_loops(run, || {
        dist.max_sim(doc, &mut scores);
        std::hint::black_box(&mut scores);
    })
}

//////////////////////
// Result types     //
//////////////////////

#[derive(Debug, Clone, Copy)]
pub(super) struct DisplayWrapper<'a, T: ?Sized>(pub(super) &'a T);

impl<T: ?Sized> std::ops::Deref for DisplayWrapper<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct RunResult {
    /// The configuration for this run.
    pub(super) run: Run,
    /// Per-measurement latencies (over `loops_per_measurement` calls).
    pub(super) latencies: Vec<MicroSeconds>,
    /// Latency percentiles.
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

/// Per-run comparison result showing before/after percentile differences.
#[derive(Debug, Serialize)]
pub(super) struct Comparison {
    pub(super) run: Run,
    pub(super) tolerance: MultiVectorTolerance,
    pub(super) before_min: f64,
    pub(super) after_min: f64,
}

/// Aggregated result of the regression check across all runs.
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
