/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Multi-vector distance benchmarks with regression detection.

use std::{io::Write, num::NonZeroUsize};

use diskann_quantization::multi_vector::{Chamfer, MatRef, MaxSim, QueryComputer, Standard};
use diskann_vector::distance::InnerProduct;
use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
use half::f16;
use rand::{
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    SeedableRng,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use diskann_benchmark_runner::{
    benchmark::{PassFail, Regression},
    dispatcher::{Description, DispatchRule, FailureScore, MatchScore},
    utils::{
        datatype::{self, DataType},
        num::{relative_change, NonNegativeFinite},
        percentiles, MicroSeconds,
    },
    Any, Benchmark, CheckDeserialization, Checker, Input,
};

////////////////
// Public API //
////////////////

/// Register all multi-vector benchmarks with the runner's dispatcher.
pub fn register(dispatcher: &mut diskann_benchmark_runner::registry::Benchmarks) {
    register_benchmarks_impl(dispatcher)
}

///////////
// Utils //
///////////

#[derive(Debug, Clone, Copy)]
struct DisplayWrapper<'a, T: ?Sized>(&'a T);

impl<T: ?Sized> std::ops::Deref for DisplayWrapper<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0
    }
}

////////////
// Inputs //
////////////

/// The two distance operations exposed by [`QueryComputer`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Chamfer,
    MaxSim,
}

impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = match self {
            Self::Chamfer => "chamfer",
            Self::MaxSim => "max_sim",
        };
        write!(f, "{}", st)
    }
}

/// Which implementation tier to benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Implementation {
    Optimized,
    Reference,
}

impl std::fmt::Display for Implementation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = match self {
            Self::Optimized => "optimized",
            Self::Reference => "reference",
        };
        write!(f, "{}", st)
    }
}

/// One benchmark configuration: a single (operation, shape) measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Run {
    operation: Operation,
    num_query_vectors: NonZeroUsize,
    num_doc_vectors: NonZeroUsize,
    dim: NonZeroUsize,
    loops_per_measurement: NonZeroUsize,
    num_measurements: NonZeroUsize,
}

/// A complete multi-vector benchmark job.
#[derive(Debug, Serialize, Deserialize)]
pub struct MultiVectorOp {
    element_type: DataType,
    implementation: Implementation,
    runs: Vec<Run>,
}

impl CheckDeserialization for MultiVectorOp {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f, "{:>18}: {}", $field, $($expr)*)
    }
}

impl MultiVectorOp {
    fn summarize_fields(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_field!(f, "element type", self.element_type)?;
        write_field!(f, "implementation", self.implementation)?;
        write_field!(f, "number of runs", self.runs.len())?;
        Ok(())
    }
}

impl std::fmt::Display for MultiVectorOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Multi-Vector Operation\n")?;
        write_field!(f, "tag", Self::tag())?;
        self.summarize_fields(f)
    }
}

impl Input for MultiVectorOp {
    fn tag() -> &'static str {
        "multi-vector-op"
    }

    fn try_deserialize(
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(Self::deserialize(serialized)?)
    }

    fn example() -> anyhow::Result<serde_json::Value> {
        const NUM_QUERY_VECTORS: NonZeroUsize = NonZeroUsize::new(32).unwrap();
        const NUM_DOC_VECTORS: NonZeroUsize = NonZeroUsize::new(64).unwrap();
        const DIM: NonZeroUsize = NonZeroUsize::new(128).unwrap();
        const LOOPS_PER_MEASUREMENT: NonZeroUsize = NonZeroUsize::new(200).unwrap();
        const NUM_MEASUREMENTS: NonZeroUsize = NonZeroUsize::new(100).unwrap();

        let runs = vec![
            Run {
                operation: Operation::Chamfer,
                num_query_vectors: NUM_QUERY_VECTORS,
                num_doc_vectors: NUM_DOC_VECTORS,
                dim: DIM,
                loops_per_measurement: LOOPS_PER_MEASUREMENT,
                num_measurements: NUM_MEASUREMENTS,
            },
            Run {
                operation: Operation::MaxSim,
                num_query_vectors: NUM_QUERY_VECTORS,
                num_doc_vectors: NUM_DOC_VECTORS,
                dim: DIM,
                loops_per_measurement: LOOPS_PER_MEASUREMENT,
                num_measurements: NUM_MEASUREMENTS,
            },
        ];

        Ok(serde_json::to_value(&Self {
            element_type: DataType::Float32,
            implementation: Implementation::Optimized,
            runs,
        })?)
    }
}

//////////////////////
// Regression Check //
//////////////////////

/// Tolerance thresholds for multi-vector benchmark regression detection.
///
/// Each field specifies the maximum allowed relative increase in the corresponding metric.
/// For example, a value of `0.05` means a 5% increase is tolerated.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct MultiVectorTolerance {
    min_time_regression: NonNegativeFinite,
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

/// Per-run comparison result showing before/after percentile differences.
#[derive(Debug, Serialize)]
struct Comparison {
    run: Run,
    tolerance: MultiVectorTolerance,
    before_min: f64,
    after_min: f64,
}

/// Aggregated result of the regression check across all runs.
#[derive(Debug, Serialize)]
struct CheckResult {
    checks: Vec<Comparison>,
}

impl std::fmt::Display for CheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let header = [
            "Operation",
            "Q",
            "D",
            "Dim",
            "Min Before (ns/IP @ Dim)",
            "Min After (ns/IP @ Dim)",
            "Change (%)",
            "Remark",
        ];

        let mut table = diskann_benchmark_runner::utils::fmt::Table::new(header, self.checks.len());

        for (i, c) in self.checks.iter().enumerate() {
            let mut row = table.row(i);
            let change = relative_change(c.before_min, c.after_min);

            row.insert(c.run.operation, 0);
            row.insert(c.run.num_query_vectors, 1);
            row.insert(c.run.num_doc_vectors, 2);
            row.insert(c.run.dim, 3);
            row.insert(format!("{:.3}", c.before_min), 4);
            row.insert(format!("{:.3}", c.after_min), 5);
            match change {
                Ok(change) => {
                    row.insert(format!("{:.3} %", change * 100.0), 6);
                    if change > c.tolerance.min_time_regression.get() {
                        row.insert("FAIL", 7);
                    }
                }
                Err(err) => {
                    row.insert("invalid", 6);
                    row.insert(err, 7);
                }
            }
        }

        table.fmt(f)
    }
}

////////////////////////////
// Benchmark Registration //
////////////////////////////

fn register_benchmarks_impl(dispatcher: &mut diskann_benchmark_runner::registry::Benchmarks) {
    macro_rules! register {
        ($impl:ident, $t:ty, $tag:literal) => {
            dispatcher.register_regression($tag, Kernel::<$impl, $t>::new());
        };
    }

    // Optimized (architecture-dispatched QueryComputer).
    register!(Optimized, f32, "multi-vector-op-f32-optimized");
    register!(Optimized, f16, "multi-vector-op-f16-optimized");

    // Reference (Chamfer / MaxSim fallback path).
    register!(Reference, f32, "multi-vector-op-f32-reference");
    register!(Reference, f16, "multi-vector-op-f16-reference");
}

//////////////
// Dispatch //
//////////////

/// Dispatch marker for the [`QueryComputer`] implementation.
#[derive(Debug)]
struct Optimized;

/// Dispatch marker for the [`Chamfer`] / [`MaxSim`] fallback.
#[derive(Debug)]
struct Reference;

/// A multi-vector benchmark.
struct Kernel<I, T> {
    _type: std::marker::PhantomData<(I, T)>,
}

impl<I, T> Kernel<I, T> {
    fn new() -> Self {
        Self {
            _type: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Error)]
#[error("this kernel handles a different implementation than {0}")]
pub(crate) struct ImplementationMismatch(Implementation);

macro_rules! impl_dispatch_rule {
    ($marker:ident, $variant:ident, $description:literal) => {
        impl DispatchRule<Implementation> for $marker {
            type Error = ImplementationMismatch;

            fn try_match(from: &Implementation) -> Result<MatchScore, FailureScore> {
                if *from == Implementation::$variant {
                    Ok(MatchScore(0))
                } else {
                    Err(FailureScore(1))
                }
            }

            fn convert(from: Implementation) -> Result<Self, Self::Error> {
                if from == Implementation::$variant {
                    Ok($marker)
                } else {
                    Err(ImplementationMismatch(from))
                }
            }

            fn description(
                f: &mut std::fmt::Formatter<'_>,
                from: Option<&Implementation>,
            ) -> std::fmt::Result {
                match from {
                    None => write!(f, $description),
                    Some(impl_) => {
                        if Self::try_match(impl_).is_ok() {
                            write!(f, "matched {}", impl_)
                        } else {
                            write!(f, "expected {}, got {}", Implementation::$variant, impl_)
                        }
                    }
                }
            }
        }
    };
}

impl_dispatch_rule!(
    Optimized,
    Optimized,
    "QueryComputer (architecture-dispatched)"
);
impl_dispatch_rule!(Reference, Reference, "Chamfer / MaxSim fallback");

impl<I, T> Benchmark for Kernel<I, T>
where
    datatype::Type<T>: DispatchRule<datatype::DataType>,
    I: DispatchRule<Implementation, Error = ImplementationMismatch> + 'static,
    Kernel<I, T>: RunBenchmark<I>,
    T: 'static,
{
    type Input = MultiVectorOp;
    type Output = Vec<RunResult>;

    fn try_match(&self, from: &MultiVectorOp) -> Result<MatchScore, FailureScore> {
        let mut failscore: Option<u32> = None;
        if datatype::Type::<T>::try_match(&from.element_type).is_err() {
            *failscore.get_or_insert(0) += 10;
        }
        if let Err(FailureScore(score)) = I::try_match(&from.implementation) {
            *failscore.get_or_insert(0) += 2 + score;
        }

        match failscore {
            None => Ok(MatchScore(0)),
            Some(score) => Err(FailureScore(score)),
        }
    }

    fn run(
        &self,
        input: &MultiVectorOp,
        _: diskann_benchmark_runner::Checkpoint<'_>,
        mut output: &mut dyn diskann_benchmark_runner::Output,
    ) -> anyhow::Result<Self::Output> {
        // The dispatcher only invokes `run` after `try_match` has already accepted
        // the input, so a failure here would indicate a dispatcher bug.
        I::convert(input.implementation).expect("try_match accepted the input");
        writeln!(output, "{}", input)?;
        let results = self.run_benchmark(input)?;
        writeln!(output, "\n\n{}", DisplayWrapper(&*results))?;
        Ok(results)
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&MultiVectorOp>,
    ) -> std::fmt::Result {
        match input {
            None => {
                writeln!(
                    f,
                    "- Element Type: {}",
                    Description::<datatype::DataType, datatype::Type<T>>::new()
                )?;
                writeln!(
                    f,
                    "- Implementation: {}",
                    Description::<Implementation, I>::new()
                )?;
            }
            Some(input) => {
                if let Err(err) = datatype::Type::<T>::try_match_verbose(&input.element_type) {
                    writeln!(f, "\n    - Mismatched element type: {}", err)?;
                }
                if let Err(err) = I::try_match_verbose(&input.implementation) {
                    writeln!(f, "\n    - Mismatched implementation: {}", err)?;
                }
            }
        }
        Ok(())
    }
}

impl<I, T> Regression for Kernel<I, T>
where
    datatype::Type<T>: DispatchRule<datatype::DataType>,
    I: DispatchRule<Implementation, Error = ImplementationMismatch> + 'static,
    Kernel<I, T>: RunBenchmark<I>,
    T: 'static,
{
    type Tolerances = MultiVectorTolerance;
    type Pass = CheckResult;
    type Fail = CheckResult;

    fn check(
        &self,
        tolerance: &MultiVectorTolerance,
        _input: &MultiVectorOp,
        before: &Vec<RunResult>,
        after: &Vec<RunResult>,
    ) -> anyhow::Result<PassFail<CheckResult, CheckResult>> {
        anyhow::ensure!(
            before.len() == after.len(),
            "before has {} runs but after has {}",
            before.len(),
            after.len(),
        );

        let mut passed = true;
        let checks: Vec<Comparison> = std::iter::zip(before.iter(), after.iter())
            .enumerate()
            .map(|(i, (b, a))| {
                anyhow::ensure!(b.run == a.run, "run {i} mismatched");

                let computations_per_latency = b.computations_per_latency() as f64;

                let before_min = b.percentiles.minimum.as_f64() * 1000.0 / computations_per_latency;
                let after_min = a.percentiles.minimum.as_f64() * 1000.0 / computations_per_latency;

                let comparison = Comparison {
                    run: b.run.clone(),
                    tolerance: *tolerance,
                    before_min,
                    after_min,
                };

                match relative_change(before_min, after_min) {
                    Ok(change) => {
                        if change > tolerance.min_time_regression.get() {
                            passed = false;
                        }
                    }
                    Err(_) => passed = false,
                };

                Ok(comparison)
            })
            .collect::<anyhow::Result<Vec<Comparison>>>()?;

        let check = CheckResult { checks };

        if passed {
            Ok(PassFail::Pass(check))
        } else {
            Ok(PassFail::Fail(check))
        }
    }
}

///////////////
// Benchmark //
///////////////

trait RunBenchmark<I> {
    fn run_benchmark(&self, input: &MultiVectorOp) -> Result<Vec<RunResult>, anyhow::Error>;
}

#[derive(Debug, Serialize, Deserialize)]
struct RunResult {
    /// The configuration for this run.
    run: Run,
    /// Per-measurement latencies (over `loops_per_measurement` calls).
    latencies: Vec<MicroSeconds>,
    /// Latency percentiles.
    percentiles: percentiles::Percentiles<MicroSeconds>,
}

impl RunResult {
    fn computations_per_latency(&self) -> usize {
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

        // ns/IP is normalized as `min_latency_us * 1000 / (Q * D * loops)` and is
        // approximately linear in `dim`. Compare across rows with the same `Dim`;
        // divide further by `Dim` to recover ns per scalar multiply.
        writeln!(
            f,
            "ns/IP = time per (query, doc) inner-product call (~ linear in Dim)"
        )?;

        let header = [
            "Operation",
            "Q",
            "D",
            "Dim",
            "Min Time (ns/IP @ Dim)",
            "Mean Time (ns/IP @ Dim)",
            "Loops",
            "Measurements",
        ];

        let mut table = diskann_benchmark_runner::utils::fmt::Table::new(header, self.len());

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

            // Convert time from micro-seconds to nano-seconds per inner-product call
            // (one (query, doc) pair, ~ linear in dim).
            let min_time = min_latency.as_f64() / computations_per_latency * 1000.0;
            let mean_time = mean_latency / computations_per_latency * 1000.0;

            row.insert(r.run.operation, 0);
            row.insert(r.run.num_query_vectors, 1);
            row.insert(r.run.num_doc_vectors, 2);
            row.insert(r.run.dim, 3);
            row.insert(format!("{:.3}", min_time), 4);
            row.insert(format!("{:.3}", mean_time), 5);
            row.insert(r.run.loops_per_measurement, 6);
            row.insert(r.run.num_measurements, 7);
        });

        table.fmt(f)
    }
}

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

///////////////////
// Data fixtures //
///////////////////

const RNG_SEED: u64 = 0x12345;

struct Data<T> {
    query_data: Box<[T]>,
    doc_data: Box<[T]>,
}

impl<T: Copy> Data<T>
where
    StandardUniform: Distribution<T>,
{
    fn new(run: &Run) -> Self {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let query_data: Box<[T]> = (0..run.num_query_vectors.get() * run.dim.get())
            .map(|_| StandardUniform.sample(&mut rng))
            .collect();
        let doc_data: Box<[T]> = (0..run.num_doc_vectors.get() * run.dim.get())
            .map(|_| StandardUniform.sample(&mut rng))
            .collect();

        Self {
            query_data,
            doc_data,
        }
    }

    fn query(&self, run: &Run) -> MatRef<'_, Standard<T>> {
        MatRef::new(
            Standard::new(run.num_query_vectors.get(), run.dim.get()).unwrap(),
            &self.query_data,
        )
        .unwrap()
    }

    fn doc(&self, run: &Run) -> MatRef<'_, Standard<T>> {
        MatRef::new(
            Standard::new(run.num_doc_vectors.get(), run.dim.get()).unwrap(),
            &self.doc_data,
        )
        .unwrap()
    }
}

/////////////////////
// Implementations //
/////////////////////

fn run_optimized<T>(input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>>
where
    T: Copy,
    StandardUniform: Distribution<T>,
    QueryComputer<T>: NewFromMatRef<T>,
{
    let mut results = Vec::with_capacity(input.runs.len());
    for run in input.runs.iter() {
        let data = Data::<T>::new(run);
        // `QueryComputer` performs query-side precomputation that is intentionally
        // amortized across many `chamfer` / `max_sim` calls; construct it once per
        // shape, outside the timed loop.
        let computer = <QueryComputer<T> as NewFromMatRef<T>>::new_from(data.query(run));
        let doc = data.doc(run);

        let result = match run.operation {
            Operation::Chamfer => run_loops(run, || {
                let v = computer.chamfer(doc);
                std::hint::black_box(v);
            }),
            Operation::MaxSim => {
                let mut scores = vec![0.0f32; run.num_query_vectors.get()];
                run_loops(run, || {
                    computer.max_sim(doc, &mut scores);
                    std::hint::black_box(&mut scores);
                })
            }
        };
        results.push(result);
    }
    Ok(results)
}

/// Drive the [`Chamfer`] / [`MaxSim`] fallback path.
fn run_reference<T>(input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>>
where
    T: Copy,
    StandardUniform: Distribution<T>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
{
    let mut results = Vec::with_capacity(input.runs.len());
    for run in input.runs.iter() {
        let data = Data::<T>::new(run);
        let doc = data.doc(run);
        // Hoist out of the timed loop to mirror the optimized path's
        // per-shape precomputation.
        let query: diskann_quantization::multi_vector::distance::QueryMatRef<'_, _> =
            data.query(run).into();

        let result = match run.operation {
            Operation::Chamfer => run_loops(run, || {
                let v = Chamfer::evaluate(query, doc);
                std::hint::black_box(v);
            }),
            Operation::MaxSim => {
                let mut scores = vec![0.0f32; run.num_query_vectors.get()];
                let mut max_sim = MaxSim::new(&mut scores).unwrap();
                run_loops(run, || {
                    let _ = max_sim.evaluate(query, doc);
                    std::hint::black_box(max_sim.scores_mut());
                })
            }
        };
        results.push(result);
    }
    Ok(results)
}

/// Element-type-erasing constructor for [`QueryComputer`].
trait NewFromMatRef<T: Copy> {
    fn new_from(query: MatRef<'_, Standard<T>>) -> QueryComputer<T>;
}

macro_rules! impl_kernel_for {
    ($t:ty) => {
        impl NewFromMatRef<$t> for QueryComputer<$t> {
            fn new_from(query: MatRef<'_, Standard<$t>>) -> QueryComputer<$t> {
                QueryComputer::<$t>::new(query)
            }
        }

        impl RunBenchmark<Optimized> for Kernel<Optimized, $t> {
            fn run_benchmark(
                &self,
                input: &MultiVectorOp,
            ) -> Result<Vec<RunResult>, anyhow::Error> {
                run_optimized::<$t>(input)
            }
        }

        impl RunBenchmark<Reference> for Kernel<Reference, $t> {
            fn run_benchmark(
                &self,
                input: &MultiVectorOp,
            ) -> Result<Vec<RunResult>, anyhow::Error> {
                run_reference::<$t>(input)
            }
        }
    };
}

impl_kernel_for!(f32);
impl_kernel_for!(f16);

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_benchmark_runner::{
        benchmark::{PassFail, Regression},
        utils::percentiles::compute_percentiles,
    };

    fn tiny_run(operation: Operation) -> Run {
        Run {
            operation,
            num_query_vectors: NonZeroUsize::new(2).unwrap(),
            num_doc_vectors: NonZeroUsize::new(2).unwrap(),
            dim: NonZeroUsize::new(4).unwrap(),
            loops_per_measurement: NonZeroUsize::new(1).unwrap(),
            num_measurements: NonZeroUsize::new(1).unwrap(),
        }
    }

    fn tiny_op() -> MultiVectorOp {
        MultiVectorOp {
            element_type: DataType::Float32,
            implementation: Implementation::Optimized,
            runs: vec![tiny_run(Operation::Chamfer)],
        }
    }

    fn tiny_result(operation: Operation, minimum: u64) -> RunResult {
        let run = tiny_run(operation);
        let minimum = MicroSeconds::new(minimum);
        let mut latencies = vec![minimum];
        let percentiles = compute_percentiles(&mut latencies).unwrap();
        RunResult {
            run,
            latencies,
            percentiles,
        }
    }

    fn tolerance(limit: f64) -> MultiVectorTolerance {
        MultiVectorTolerance {
            min_time_regression: NonNegativeFinite::new(limit).unwrap(),
        }
    }

    #[test]
    fn check_rejects_mismatched_runs() {
        let kernel = Kernel::<Optimized, f32>::new();

        let err = kernel
            .check(
                &tolerance(0.0),
                &tiny_op(),
                &vec![tiny_result(Operation::Chamfer, 100)],
                &vec![tiny_result(Operation::MaxSim, 100)],
            )
            .unwrap_err();

        assert_eq!(err.to_string(), "run 0 mismatched");
    }

    #[test]
    fn check_allows_negative_relative_change() {
        let kernel = Kernel::<Optimized, f32>::new();

        let result = kernel
            .check(
                &tolerance(0.0),
                &tiny_op(),
                &vec![tiny_result(Operation::Chamfer, 100)],
                &vec![tiny_result(Operation::Chamfer, 95)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Pass(_)));
    }

    #[test]
    fn check_passes_on_tolerance_boundary() {
        let kernel = Kernel::<Optimized, f32>::new();

        let result = kernel
            .check(
                &tolerance(0.05),
                &tiny_op(),
                &vec![tiny_result(Operation::Chamfer, 100)],
                &vec![tiny_result(Operation::Chamfer, 105)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Pass(_)));
    }

    #[test]
    fn check_fails_above_tolerance_boundary() {
        let kernel = Kernel::<Optimized, f32>::new();

        let result = kernel
            .check(
                &tolerance(0.05),
                &tiny_op(),
                &vec![tiny_result(Operation::Chamfer, 100)],
                &vec![tiny_result(Operation::Chamfer, 106)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Fail(_)));
    }

    #[test]
    fn check_result_display_includes_failure_details() {
        let check = CheckResult {
            checks: vec![Comparison {
                run: tiny_run(Operation::Chamfer),
                tolerance: tolerance(0.05),
                before_min: 100.0,
                after_min: 106.0,
            }],
        };

        let rendered = check.to_string();
        assert!(rendered.contains("Operation"), "rendered = {rendered}");
        assert!(rendered.contains("chamfer"), "rendered = {rendered}");
        assert!(rendered.contains("100.000"), "rendered = {rendered}");
        assert!(rendered.contains("106.000"), "rendered = {rendered}");
        assert!(rendered.contains("6.000 %"), "rendered = {rendered}");
        assert!(rendered.contains("FAIL"), "rendered = {rendered}");
    }

    /// A "before" value of 0 means the measurement was too fast to obtain a
    /// reliable signal, so we *could* be letting a regression through. We
    /// require at least a non-zero value.
    #[test]
    fn zero_values_rejected() {
        let kernel = Kernel::<Optimized, f32>::new();

        let result = kernel
            .check(
                &tolerance(0.05),
                &tiny_op(),
                &vec![tiny_result(Operation::Chamfer, 0)],
                &vec![tiny_result(Operation::Chamfer, 0)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Fail(_)));
    }
}
