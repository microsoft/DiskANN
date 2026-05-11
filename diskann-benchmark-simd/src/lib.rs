/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! SIMD distance kernel benchmarks with regression detection.

use std::{io::Write, num::NonZeroUsize};

use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::simd;
use diskann_wide::Architecture;
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimilarityMeasure {
    SquaredL2,
    InnerProduct,
    Cosine,
}

impl std::fmt::Display for SimilarityMeasure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = match self {
            Self::SquaredL2 => "squared_l2",
            Self::InnerProduct => "inner_product",
            Self::Cosine => "cosine",
        };
        write!(f, "{}", st)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Arch {
    #[serde(rename = "x86-64-v4")]
    #[allow(non_camel_case_types)]
    X86_64_V4,
    #[serde(rename = "x86-64-v3")]
    #[allow(non_camel_case_types)]
    X86_64_V3,
    Neon,
    Scalar,
    Reference,
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = match self {
            Self::X86_64_V4 => "x86-64-v4",
            Self::X86_64_V3 => "x86-64-v3",
            Self::Neon => "neon",
            Self::Scalar => "scalar",
            Self::Reference => "reference",
        };
        write!(f, "{}", st)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Run {
    distance: SimilarityMeasure,
    dim: NonZeroUsize,
    num_points: NonZeroUsize,
    loops_per_measurement: NonZeroUsize,
    num_measurements: NonZeroUsize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SimdOp {
    query_type: DataType,
    data_type: DataType,
    arch: Arch,
    runs: Vec<Run>,
}

impl CheckDeserialization for SimdOp {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f, "{:>18}: {}", $field, $($expr)*)
    }
}

impl SimdOp {
    fn summarize_fields(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_field!(f, "query type", self.query_type)?;
        write_field!(f, "data type", self.data_type)?;
        write_field!(f, "arch", self.arch)?;
        write_field!(f, "number of runs", self.runs.len())?;
        Ok(())
    }
}

impl std::fmt::Display for SimdOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SIMD Operation\n")?;
        write_field!(f, "tag", Self::tag())?;
        self.summarize_fields(f)
    }
}

impl Input for SimdOp {
    fn tag() -> &'static str {
        "simd-op"
    }

    fn try_deserialize(
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(Self::deserialize(serialized)?)
    }

    fn example() -> anyhow::Result<serde_json::Value> {
        const DIM: [NonZeroUsize; 2] = [
            NonZeroUsize::new(128).unwrap(),
            NonZeroUsize::new(150).unwrap(),
        ];

        const NUM_POINTS: [NonZeroUsize; 2] = [
            NonZeroUsize::new(1000).unwrap(),
            NonZeroUsize::new(800).unwrap(),
        ];

        const LOOPS_PER_MEASUREMENT: NonZeroUsize = NonZeroUsize::new(100).unwrap();
        const NUM_MEASUREMENTS: NonZeroUsize = NonZeroUsize::new(100).unwrap();

        let runs = vec![
            Run {
                distance: SimilarityMeasure::SquaredL2,
                dim: DIM[0],
                num_points: NUM_POINTS[0],
                loops_per_measurement: LOOPS_PER_MEASUREMENT,
                num_measurements: NUM_MEASUREMENTS,
            },
            Run {
                distance: SimilarityMeasure::InnerProduct,
                dim: DIM[1],
                num_points: NUM_POINTS[1],
                loops_per_measurement: LOOPS_PER_MEASUREMENT,
                num_measurements: NUM_MEASUREMENTS,
            },
        ];

        Ok(serde_json::to_value(&Self {
            query_type: DataType::Float32,
            data_type: DataType::Float32,
            arch: Arch::X86_64_V3,
            runs,
        })?)
    }
}

//////////////////////
// Regression Check //
//////////////////////

/// Tolerance thresholds for SIMD benchmark regression detection.
///
/// Each field specifies the maximum allowed relative increase in the corresponding metric.
/// For example, a value of `0.10` means a 10% increase is tolerated.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct SimdTolerance {
    min_time_regression: NonNegativeFinite,
}

impl CheckDeserialization for SimdTolerance {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

impl Input for SimdTolerance {
    fn tag() -> &'static str {
        "simd-tolerance"
    }

    fn try_deserialize(
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(Self::deserialize(serialized)?)
    }

    fn example() -> anyhow::Result<serde_json::Value> {
        const EXAMPLE: NonNegativeFinite = match NonNegativeFinite::new(0.10) {
            Ok(v) => v,
            Err(_) => panic!("use a non-negative finite please"),
        };

        Ok(serde_json::to_value(SimdTolerance {
            min_time_regression: EXAMPLE,
        })?)
    }
}

/// Per-run comparison result showing before/after percentile differences.
#[derive(Debug, Serialize)]
struct Comparison {
    run: Run,
    tolerance: SimdTolerance,
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
            "Distance",
            "Dim",
            "Min Before (ns)",
            "Min After (ns)",
            "Change (%)",
            "Remark",
        ];

        let mut table = diskann_benchmark_runner::utils::fmt::Table::new(header, self.checks.len());

        for (i, c) in self.checks.iter().enumerate() {
            let mut row = table.row(i);
            let change = relative_change(c.before_min, c.after_min);

            row.insert(c.run.distance, 0);
            row.insert(c.run.dim, 1);
            row.insert(format!("{:.3}", c.before_min), 2);
            row.insert(format!("{:.3}", c.after_min), 3);
            match change {
                Ok(change) => {
                    row.insert(format!("{:.3} %", change * 100.0), 4);
                    if change > c.tolerance.min_time_regression.get() {
                        row.insert("FAIL", 5);
                    }
                }
                Err(err) => {
                    row.insert("invalid", 4);
                    row.insert(err, 5);
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
    // x86-64-v4
    #[cfg(target_arch = "x86_64")]
    {
        dispatcher.register_regression(
            "simd-op-f32xf32-x86_64_V4",
            Kernel::<diskann_wide::arch::x86_64::V4, f32, f32>::new(),
        );
        dispatcher.register_regression(
            "simd-op-f16xf16-x86_64_V4",
            Kernel::<diskann_wide::arch::x86_64::V4, f16, f16>::new(),
        );
        dispatcher.register_regression(
            "simd-op-u8xu8-x86_64_V4",
            Kernel::<diskann_wide::arch::x86_64::V4, u8, u8>::new(),
        );
        dispatcher.register_regression(
            "simd-op-i8xi8-x86_64_V4",
            Kernel::<diskann_wide::arch::x86_64::V4, i8, i8>::new(),
        );
    }

    // x86-64-v3
    #[cfg(target_arch = "x86_64")]
    {
        dispatcher.register_regression(
            "simd-op-f32xf32-x86_64_V3",
            Kernel::<diskann_wide::arch::x86_64::V3, f32, f32>::new(),
        );
        dispatcher.register_regression(
            "simd-op-f16xf16-x86_64_V3",
            Kernel::<diskann_wide::arch::x86_64::V3, f16, f16>::new(),
        );
        dispatcher.register_regression(
            "simd-op-u8xu8-x86_64_V3",
            Kernel::<diskann_wide::arch::x86_64::V3, u8, u8>::new(),
        );
        dispatcher.register_regression(
            "simd-op-i8xi8-x86_64_V3",
            Kernel::<diskann_wide::arch::x86_64::V3, i8, i8>::new(),
        );
    }

    // aarch64-neon
    #[cfg(target_arch = "aarch64")]
    {
        dispatcher.register_regression(
            "simd-op-f32xf32-aarch64_neon",
            Kernel::<diskann_wide::arch::aarch64::Neon, f32, f32>::new(),
        );
        dispatcher.register_regression(
            "simd-op-f16xf16-aarch64_neon",
            Kernel::<diskann_wide::arch::aarch64::Neon, f16, f16>::new(),
        );
        dispatcher.register_regression(
            "simd-op-u8xu8-aarch64_neon",
            Kernel::<diskann_wide::arch::aarch64::Neon, u8, u8>::new(),
        );
        dispatcher.register_regression(
            "simd-op-i8xi8-aarch64_neon",
            Kernel::<diskann_wide::arch::aarch64::Neon, i8, i8>::new(),
        );
    }

    // scalar
    dispatcher.register_regression(
        "simd-op-f32xf32-scalar",
        Kernel::<diskann_wide::arch::Scalar, f32, f32>::new(),
    );
    dispatcher.register_regression(
        "simd-op-f16xf16-scalar",
        Kernel::<diskann_wide::arch::Scalar, f16, f16>::new(),
    );
    dispatcher.register_regression(
        "simd-op-u8xu8-scalar",
        Kernel::<diskann_wide::arch::Scalar, u8, u8>::new(),
    );
    dispatcher.register_regression(
        "simd-op-i8xi8-scalar",
        Kernel::<diskann_wide::arch::Scalar, i8, i8>::new(),
    );

    // reference
    dispatcher.register_regression(
        "simd-op-f32xf32-reference",
        Kernel::<Reference, f32, f32>::new(),
    );
    dispatcher.register_regression(
        "simd-op-f16xf16-reference",
        Kernel::<Reference, f16, f16>::new(),
    );
    dispatcher.register_regression(
        "simd-op-u8xu8-reference",
        Kernel::<Reference, u8, u8>::new(),
    );
    dispatcher.register_regression(
        "simd-op-i8xi8-reference",
        Kernel::<Reference, i8, i8>::new(),
    );
}

//////////////
// Dispatch //
//////////////

/// Dispatch receiver for the reference implementations.
struct Reference;

/// A dispatch mapper for `wide` types.
#[derive(Debug)]
struct Identity<T>(T);

struct Kernel<A, Q, D> {
    _type: std::marker::PhantomData<(A, Q, D)>,
}

impl<A, Q, D> Kernel<A, Q, D> {
    fn new() -> Self {
        Self {
            _type: std::marker::PhantomData,
        }
    }
}

// Map Architectures to the enum.
#[derive(Debug, Error)]
#[error("architecture {0} is not supported by this CPU")]
pub(crate) struct ArchNotSupported(Arch);

impl DispatchRule<Arch> for Identity<Reference> {
    type Error = ArchNotSupported;

    fn try_match(from: &Arch) -> Result<MatchScore, FailureScore> {
        if *from == Arch::Reference {
            Ok(MatchScore(0))
        } else {
            Err(FailureScore(1))
        }
    }

    fn convert(from: Arch) -> Result<Self, Self::Error> {
        assert_eq!(from, Arch::Reference);
        Ok(Identity(Reference))
    }

    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&Arch>) -> std::fmt::Result {
        match from {
            None => write!(f, "loop based"),
            Some(arch) => {
                if Self::try_match(arch).is_ok() {
                    write!(f, "matched {}", arch)
                } else {
                    write!(f, "expected {}, got {}", Arch::Reference, arch)
                }
            }
        }
    }
}

impl DispatchRule<Arch> for Identity<diskann_wide::arch::Scalar> {
    type Error = ArchNotSupported;

    fn try_match(from: &Arch) -> Result<MatchScore, FailureScore> {
        if *from == Arch::Scalar {
            Ok(MatchScore(0))
        } else {
            Err(FailureScore(1))
        }
    }

    fn convert(from: Arch) -> Result<Self, Self::Error> {
        assert_eq!(from, Arch::Scalar);
        Ok(Identity(diskann_wide::arch::Scalar))
    }

    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&Arch>) -> std::fmt::Result {
        match from {
            None => write!(f, "scalar (compilation target CPU)"),
            Some(arch) => {
                if Self::try_match(arch).is_ok() {
                    write!(f, "matched {}", arch)
                } else {
                    write!(f, "expected {}, got {}", Arch::Scalar, arch)
                }
            }
        }
    }
}

macro_rules! match_arch {
    ($target_arch:literal, $arch:path, $enum:ident) => {
        #[cfg(target_arch = $target_arch)]
        impl DispatchRule<Arch> for Identity<$arch> {
            type Error = ArchNotSupported;

            fn try_match(from: &Arch) -> Result<MatchScore, FailureScore> {
                let available = <$arch>::new_checked().is_some();
                if available && *from == Arch::$enum {
                    Ok(MatchScore(0))
                } else if !available && *from == Arch::$enum {
                    Err(FailureScore(0))
                } else {
                    Err(FailureScore(1))
                }
            }

            fn convert(from: Arch) -> Result<Self, Self::Error> {
                assert_eq!(from, Arch::$enum);
                <$arch>::new_checked()
                    .ok_or(ArchNotSupported(from))
                    .map(Identity)
            }

            fn description(
                f: &mut std::fmt::Formatter<'_>,
                from: Option<&Arch>,
            ) -> std::fmt::Result {
                let available = <$arch>::new_checked().is_some();
                match from {
                    None => write!(f, "{}", Arch::$enum),
                    Some(arch) => {
                        if Self::try_match(arch).is_ok() {
                            write!(f, "matched {}", arch)
                        } else if !available && *arch == Arch::$enum {
                            write!(f, "matched {} but unsupported by this CPU", Arch::$enum)
                        } else {
                            write!(f, "expected {}, got {}", Arch::$enum, arch)
                        }
                    }
                }
            }
        }
    };
}

match_arch!("x86_64", diskann_wide::arch::x86_64::V4, X86_64_V4);
match_arch!("x86_64", diskann_wide::arch::x86_64::V3, X86_64_V3);
match_arch!("aarch64", diskann_wide::arch::aarch64::Neon, Neon);

impl<A, Q, D> Benchmark for Kernel<A, Q, D>
where
    datatype::Type<Q>: DispatchRule<datatype::DataType>,
    datatype::Type<D>: DispatchRule<datatype::DataType>,
    Identity<A>: DispatchRule<Arch, Error = ArchNotSupported>,
    Kernel<A, Q, D>: RunBenchmark<A>,
    A: 'static,
    Q: 'static,
    D: 'static,
{
    type Input = SimdOp;
    type Output = Vec<RunResult>;

    // Matching simply requires that we match the inner type.
    fn try_match(&self, from: &SimdOp) -> Result<MatchScore, FailureScore> {
        let mut failscore: Option<u32> = None;
        if datatype::Type::<Q>::try_match(&from.query_type).is_err() {
            *failscore.get_or_insert(0) += 10;
        }
        if datatype::Type::<D>::try_match(&from.data_type).is_err() {
            *failscore.get_or_insert(0) += 10;
        }
        if let Err(FailureScore(score)) = Identity::<A>::try_match(&from.arch) {
            *failscore.get_or_insert(0) += 2 + score;
        }

        match failscore {
            None => Ok(MatchScore(0)),
            Some(score) => Err(FailureScore(score)),
        }
    }

    fn run(
        &self,
        input: &SimdOp,
        _: diskann_benchmark_runner::Checkpoint<'_>,
        mut output: &mut dyn diskann_benchmark_runner::Output,
    ) -> anyhow::Result<Self::Output> {
        let arch = Identity::<A>::convert(input.arch)?.0;
        writeln!(output, "{}", input)?;
        let results = self.run_benchmark(input, arch)?;
        writeln!(output, "\n\n{}", DisplayWrapper(&*results))?;
        Ok(results)
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&SimdOp>,
    ) -> std::fmt::Result {
        match input {
            None => {
                writeln!(
                    f,
                    "- Query Type: {}",
                    Description::<datatype::DataType, datatype::Type<Q>>::new()
                )?;
                writeln!(
                    f,
                    "- Data Type: {}",
                    Description::<datatype::DataType, datatype::Type<D>>::new()
                )?;
                writeln!(
                    f,
                    "- Implementation: {}",
                    Description::<Arch, Identity<A>>::new()
                )?;
            }
            Some(input) => {
                if let Err(err) = datatype::Type::<Q>::try_match_verbose(&input.query_type) {
                    writeln!(f, "\n    - Mismatched query type: {}", err)?;
                }
                if let Err(err) = datatype::Type::<D>::try_match_verbose(&input.data_type) {
                    writeln!(f, "\n    - Mismatched data type: {}", err)?;
                }
                if let Err(err) = Identity::<A>::try_match_verbose(&input.arch) {
                    writeln!(f, "\n    - Mismatched architecture: {}", err)?;
                }
            }
        }
        Ok(())
    }
}

impl<A, Q, D> Regression for Kernel<A, Q, D>
where
    datatype::Type<Q>: DispatchRule<datatype::DataType>,
    datatype::Type<D>: DispatchRule<datatype::DataType>,
    Identity<A>: DispatchRule<Arch, Error = ArchNotSupported>,
    Kernel<A, Q, D>: RunBenchmark<A>,
    A: 'static,
    Q: 'static,
    D: 'static,
{
    type Tolerances = SimdTolerance;
    type Pass = CheckResult;
    type Fail = CheckResult;

    fn check(
        &self,
        tolerance: &SimdTolerance,
        _input: &SimdOp,
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

                // Determine whether or not we pass.
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

trait RunBenchmark<A> {
    fn run_benchmark(&self, input: &SimdOp, arch: A) -> Result<Vec<RunResult>, anyhow::Error>;
}

#[derive(Debug, Serialize, Deserialize)]
struct RunResult {
    /// The configuration for this run.
    run: Run,
    /// The latencies of individual runs.
    latencies: Vec<MicroSeconds>,
    /// Latency percentiles.
    percentiles: percentiles::Percentiles<MicroSeconds>,
}

impl RunResult {
    fn computations_per_latency(&self) -> usize {
        self.run.num_points.get() * self.run.loops_per_measurement.get()
    }
}

impl std::fmt::Display for DisplayWrapper<'_, [RunResult]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return Ok(());
        }

        let header = [
            "Distance",
            "Dim",
            "Min Time (ns)",
            "Mean Time (ns)",
            "Points",
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

            // Convert time from micro-seconds to nano-seconds.
            let min_time = min_latency.as_f64() / computations_per_latency * 1000.0;
            let mean_time = mean_latency / computations_per_latency * 1000.0;

            row.insert(r.run.distance, 0);
            row.insert(r.run.dim, 1);
            row.insert(format!("{:.3}", min_time), 2);
            row.insert(format!("{:.3}", mean_time), 3);
            row.insert(r.run.num_points, 4);
            row.insert(r.run.loops_per_measurement, 5);
            row.insert(r.run.num_measurements, 6);
        });

        table.fmt(f)
    }
}

fn run_loops<Q, D, F>(query: &[Q], data: MatrixView<D>, run: &Run, f: F) -> RunResult
where
    F: Fn(&[Q], &[D]) -> f32,
{
    let mut latencies = Vec::with_capacity(run.num_measurements.get());
    let mut dst = vec![0.0; data.nrows()];

    for _ in 0..run.num_measurements.get() {
        let start = std::time::Instant::now();
        for _ in 0..run.loops_per_measurement.get() {
            std::iter::zip(dst.iter_mut(), data.row_iter()).for_each(|(d, r)| {
                *d = f(query, r);
            });
            std::hint::black_box(&mut dst);
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

struct Data<Q, D> {
    query: Box<[Q]>,
    data: Matrix<D>,
}

impl<Q, D> Data<Q, D> {
    fn new(run: &Run) -> Self
    where
        StandardUniform: Distribution<Q>,
        StandardUniform: Distribution<D>,
    {
        let mut rng = StdRng::seed_from_u64(0x12345);
        let query: Box<[Q]> = (0..run.dim.get())
            .map(|_| StandardUniform.sample(&mut rng))
            .collect();
        let data = Matrix::<D>::new(
            diskann_utils::views::Init(|| StandardUniform.sample(&mut rng)),
            run.num_points.get(),
            run.dim.get(),
        );

        Self { query, data }
    }

    fn run<F>(&self, run: &Run, f: F) -> RunResult
    where
        F: Fn(&[Q], &[D]) -> f32,
    {
        run_loops(&self.query, self.data.as_view(), run, f)
    }
}

/////////////////////
// Implementations //
/////////////////////

macro_rules! stamp {
    (reference, $Q:ty, $D:ty, $f_l2:ident, $f_ip:ident, $f_cosine:ident) => {
        impl RunBenchmark<Reference> for Kernel<Reference, $Q, $D> {
            fn run_benchmark(
                &self,
                input: &SimdOp,
                _arch: Reference,
            ) -> Result<Vec<RunResult>, anyhow::Error> {
                let mut results = Vec::new();
                for run in input.runs.iter() {
                    let data = Data::<$Q, $D>::new(run);
                    let result = match run.distance {
                        SimilarityMeasure::SquaredL2 => data.run(run, reference::$f_l2),
                        SimilarityMeasure::InnerProduct => data.run(run, reference::$f_ip),
                        SimilarityMeasure::Cosine => data.run(run, reference::$f_cosine),
                    };
                    results.push(result);
                }
                Ok(results)
            }
        }
    };
    ($arch:path, $Q:ty, $D:ty) => {
        impl RunBenchmark<$arch> for Kernel<$arch, $Q, $D> {
            fn run_benchmark(
                &self,
                input: &SimdOp,
                arch: $arch,
            ) -> Result<Vec<RunResult>, anyhow::Error> {
                let mut results = Vec::new();

                let l2 = &simd::L2 {};
                let ip = &simd::IP {};
                let cosine = &simd::CosineStateless {};

                for run in input.runs.iter() {
                    let data = Data::<$Q, $D>::new(run);
                    // For each kernel, we need to do a two-step wrapping of closures so
                    // the inner-most closure is executed by the architecture.
                    //
                    // This is required for the implementation of `simd_op` to be inlined
                    // into the architecture run function so it can properly inherit
                    // target features.
                    let result = match run.distance {
                        SimilarityMeasure::SquaredL2 => data.run(run, |q, d| {
                            arch.run2(|q, d| simd::simd_op(l2, arch, q, d), q, d)
                        }),
                        SimilarityMeasure::InnerProduct => data.run(run, |q, d| {
                            arch.run2(|q, d| simd::simd_op(ip, arch, q, d), q, d)
                        }),
                        SimilarityMeasure::Cosine => data.run(run, |q, d| {
                            arch.run2(|q, d| simd::simd_op(cosine, arch, q, d), q, d)
                        }),
                    };
                    results.push(result)
                }
                Ok(results)
            }
        }
    };
    ($target_arch:literal, $arch:path, $Q:ty, $D:ty) => {
        #[cfg(target_arch = $target_arch)]
        stamp!($arch, $Q, $D);
    };
}

stamp!("x86_64", diskann_wide::arch::x86_64::V4, f32, f32);
stamp!("x86_64", diskann_wide::arch::x86_64::V4, f16, f16);
stamp!("x86_64", diskann_wide::arch::x86_64::V4, u8, u8);
stamp!("x86_64", diskann_wide::arch::x86_64::V4, i8, i8);

stamp!("x86_64", diskann_wide::arch::x86_64::V3, f32, f32);
stamp!("x86_64", diskann_wide::arch::x86_64::V3, f16, f16);
stamp!("x86_64", diskann_wide::arch::x86_64::V3, u8, u8);
stamp!("x86_64", diskann_wide::arch::x86_64::V3, i8, i8);

stamp!("aarch64", diskann_wide::arch::aarch64::Neon, f32, f32);
stamp!("aarch64", diskann_wide::arch::aarch64::Neon, f16, f16);
stamp!("aarch64", diskann_wide::arch::aarch64::Neon, u8, u8);
stamp!("aarch64", diskann_wide::arch::aarch64::Neon, i8, i8);

stamp!(diskann_wide::arch::Scalar, f32, f32);
stamp!(diskann_wide::arch::Scalar, f16, f16);
stamp!(diskann_wide::arch::Scalar, u8, u8);
stamp!(diskann_wide::arch::Scalar, i8, i8);

stamp!(
    reference,
    f32,
    f32,
    squared_l2_f32,
    inner_product_f32,
    cosine_f32
);
stamp!(
    reference,
    f16,
    f16,
    squared_l2_f16,
    inner_product_f16,
    cosine_f16
);
stamp!(
    reference,
    u8,
    u8,
    squared_l2_u8,
    inner_product_u8,
    cosine_u8
);
stamp!(
    reference,
    i8,
    i8,
    squared_l2_i8,
    inner_product_i8,
    cosine_i8
);

///////////////
// Reference //
///////////////

// These are largely copied from the implementations in vector, with a tweak that we don't
// use FMA when the current architecture is scalar.
mod reference {
    use diskann_wide::ARCH;
    use half::f16;

    trait MaybeFMA {
        // Perform `a*b + c` using FMA when a hardware instruction is guaranteed to be
        // available, otherwise decompose into a multiply and add.
        fn maybe_fma(self, a: f32, b: f32, c: f32) -> f32;
    }

    impl MaybeFMA for diskann_wide::arch::Scalar {
        fn maybe_fma(self, a: f32, b: f32, c: f32) -> f32 {
            a * b + c
        }
    }

    #[cfg(target_arch = "x86_64")]
    impl MaybeFMA for diskann_wide::arch::x86_64::V3 {
        fn maybe_fma(self, a: f32, b: f32, c: f32) -> f32 {
            a.mul_add(b, c)
        }
    }

    #[cfg(target_arch = "x86_64")]
    impl MaybeFMA for diskann_wide::arch::x86_64::V4 {
        fn maybe_fma(self, a: f32, b: f32, c: f32) -> f32 {
            a.mul_add(b, c)
        }
    }

    #[cfg(target_arch = "aarch64")]
    impl MaybeFMA for diskann_wide::arch::aarch64::Neon {
        fn maybe_fma(self, a: f32, b: f32, c: f32) -> f32 {
            a.mul_add(b, c)
        }
    }

    //------------//
    // Squared L2 //
    //------------//

    pub(super) fn squared_l2_i8(x: &[i8], y: &[i8]) -> f32 {
        assert_eq!(x.len(), y.len());
        std::iter::zip(x.iter(), y.iter())
            .map(|(&a, &b)| {
                let a: i32 = a.into();
                let b: i32 = b.into();
                let diff = a - b;
                diff * diff
            })
            .sum::<i32>() as f32
    }

    pub(super) fn squared_l2_u8(x: &[u8], y: &[u8]) -> f32 {
        assert_eq!(x.len(), y.len());
        std::iter::zip(x.iter(), y.iter())
            .map(|(&a, &b)| {
                let a: i32 = a.into();
                let b: i32 = b.into();
                let diff = a - b;
                diff * diff
            })
            .sum::<i32>() as f32
    }

    pub(super) fn squared_l2_f16(x: &[f16], y: &[f16]) -> f32 {
        assert_eq!(x.len(), y.len());
        std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| {
            let a: f32 = a.into();
            let b: f32 = b.into();
            let diff = a - b;
            ARCH.maybe_fma(diff, diff, acc)
        })
    }

    pub(super) fn squared_l2_f32(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| {
            let diff = a - b;
            ARCH.maybe_fma(diff, diff, acc)
        })
    }

    //---------------//
    // Inner Product //
    //---------------//

    pub(super) fn inner_product_i8(x: &[i8], y: &[i8]) -> f32 {
        assert_eq!(x.len(), y.len());
        std::iter::zip(x.iter(), y.iter())
            .map(|(&a, &b)| {
                let a: i32 = a.into();
                let b: i32 = b.into();
                a * b
            })
            .sum::<i32>() as f32
    }

    pub(super) fn inner_product_u8(x: &[u8], y: &[u8]) -> f32 {
        assert_eq!(x.len(), y.len());
        std::iter::zip(x.iter(), y.iter())
            .map(|(&a, &b)| {
                let a: i32 = a.into();
                let b: i32 = b.into();
                a * b
            })
            .sum::<i32>() as f32
    }

    pub(super) fn inner_product_f16(x: &[f16], y: &[f16]) -> f32 {
        assert_eq!(x.len(), y.len());
        std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| {
            let a: f32 = a.into();
            let b: f32 = b.into();
            ARCH.maybe_fma(a, b, acc)
        })
    }

    pub(super) fn inner_product_f32(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        std::iter::zip(x.iter(), y.iter()).fold(0.0f32, |acc, (&a, &b)| ARCH.maybe_fma(a, b, acc))
    }

    //--------//
    // Cosine //
    //--------//

    #[derive(Default)]
    struct XY<T> {
        xnorm: T,
        ynorm: T,
        xy: T,
    }

    pub(super) fn cosine_i8(x: &[i8], y: &[i8]) -> f32 {
        assert_eq!(x.len(), y.len());
        let r: XY<i32> =
            std::iter::zip(x.iter(), y.iter()).fold(XY::<i32>::default(), |acc, (&vx, &vy)| {
                let vx: i32 = vx.into();
                let vy: i32 = vy.into();
                XY {
                    xnorm: acc.xnorm + vx * vx,
                    ynorm: acc.ynorm + vy * vy,
                    xy: acc.xy + vx * vy,
                }
            });

        if r.xnorm == 0 || r.ynorm == 0 {
            return 0.0;
        }

        (r.xy as f32 / ((r.xnorm as f32).sqrt() * (r.ynorm as f32).sqrt())).clamp(-1.0, 1.0)
    }

    pub(super) fn cosine_u8(x: &[u8], y: &[u8]) -> f32 {
        assert_eq!(x.len(), y.len());
        let r: XY<i32> =
            std::iter::zip(x.iter(), y.iter()).fold(XY::<i32>::default(), |acc, (&vx, &vy)| {
                let vx: i32 = vx.into();
                let vy: i32 = vy.into();
                XY {
                    xnorm: acc.xnorm + vx * vx,
                    ynorm: acc.ynorm + vy * vy,
                    xy: acc.xy + vx * vy,
                }
            });

        if r.xnorm == 0 || r.ynorm == 0 {
            return 0.0;
        }

        (r.xy as f32 / ((r.xnorm as f32).sqrt() * (r.ynorm as f32).sqrt())).clamp(-1.0, 1.0)
    }

    pub(super) fn cosine_f16(x: &[f16], y: &[f16]) -> f32 {
        assert_eq!(x.len(), y.len());
        let r: XY<f32> =
            std::iter::zip(x.iter(), y.iter()).fold(XY::<f32>::default(), |acc, (&vx, &vy)| {
                let vx: f32 = vx.into();
                let vy: f32 = vy.into();
                XY {
                    xnorm: ARCH.maybe_fma(vx, vx, acc.xnorm),
                    ynorm: ARCH.maybe_fma(vy, vy, acc.ynorm),
                    xy: ARCH.maybe_fma(vx, vy, acc.xy),
                }
            });

        if r.xnorm < f32::EPSILON || r.ynorm < f32::EPSILON {
            return 0.0;
        }

        (r.xy / (r.xnorm.sqrt() * r.ynorm.sqrt())).clamp(-1.0, 1.0)
    }

    pub(super) fn cosine_f32(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        let r: XY<f32> =
            std::iter::zip(x.iter(), y.iter()).fold(XY::<f32>::default(), |acc, (&vx, &vy)| XY {
                xnorm: vx.mul_add(vx, acc.xnorm),
                ynorm: vy.mul_add(vy, acc.ynorm),
                xy: vx.mul_add(vy, acc.xy),
            });

        if r.xnorm < f32::EPSILON || r.ynorm < f32::EPSILON {
            return 0.0;
        }

        (r.xy / (r.xnorm.sqrt() * r.ynorm.sqrt())).clamp(-1.0, 1.0)
    }
}

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

    fn tiny_run(distance: SimilarityMeasure) -> Run {
        Run {
            distance,
            dim: NonZeroUsize::new(8).unwrap(),
            num_points: NonZeroUsize::new(1).unwrap(),
            loops_per_measurement: NonZeroUsize::new(1).unwrap(),
            num_measurements: NonZeroUsize::new(1).unwrap(),
        }
    }

    fn tiny_op() -> SimdOp {
        SimdOp {
            query_type: DataType::Float32,
            data_type: DataType::Float32,
            arch: Arch::Scalar,
            runs: vec![tiny_run(SimilarityMeasure::SquaredL2)],
        }
    }

    fn tiny_result(distance: SimilarityMeasure, minimum: u64) -> RunResult {
        let run = tiny_run(distance);
        let minimum = MicroSeconds::new(minimum);
        let mut latencies = vec![minimum];
        let percentiles = compute_percentiles(&mut latencies).unwrap();
        RunResult {
            run,
            latencies,
            percentiles,
        }
    }

    fn tolerance(limit: f64) -> SimdTolerance {
        SimdTolerance {
            min_time_regression: NonNegativeFinite::new(limit).unwrap(),
        }
    }

    #[test]
    fn check_rejects_mismatched_runs() {
        let kernel = Kernel::<diskann_wide::arch::Scalar, f32, f32>::new();

        let err = kernel
            .check(
                &tolerance(0.0),
                &tiny_op(),
                &vec![tiny_result(SimilarityMeasure::SquaredL2, 100)],
                &vec![tiny_result(SimilarityMeasure::Cosine, 100)],
            )
            .unwrap_err();

        assert_eq!(err.to_string(), "run 0 mismatched");
    }

    #[test]
    fn check_allows_negative_relative_change() {
        let kernel = Kernel::<diskann_wide::arch::Scalar, f32, f32>::new();

        let result = kernel
            .check(
                &tolerance(0.0),
                &tiny_op(),
                &vec![tiny_result(SimilarityMeasure::SquaredL2, 100)],
                &vec![tiny_result(SimilarityMeasure::SquaredL2, 95)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Pass(_)));
    }

    #[test]
    fn check_passes_on_tolerance_boundary() {
        let kernel = Kernel::<diskann_wide::arch::Scalar, f32, f32>::new();

        let result = kernel
            .check(
                &tolerance(0.05),
                &tiny_op(),
                &vec![tiny_result(SimilarityMeasure::SquaredL2, 100)],
                &vec![tiny_result(SimilarityMeasure::SquaredL2, 105)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Pass(_)));
    }

    #[test]
    fn check_fails_above_tolerance_boundary() {
        let kernel = Kernel::<diskann_wide::arch::Scalar, f32, f32>::new();

        let result = kernel
            .check(
                &tolerance(0.05),
                &tiny_op(),
                &vec![tiny_result(SimilarityMeasure::SquaredL2, 100)],
                &vec![tiny_result(SimilarityMeasure::SquaredL2, 106)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Fail(_)));
    }

    #[test]
    fn check_result_display_includes_failure_details() {
        let check = CheckResult {
            checks: vec![Comparison {
                run: tiny_run(SimilarityMeasure::SquaredL2),
                tolerance: tolerance(0.05),
                before_min: 100.0,
                after_min: 106.0,
            }],
        };

        let rendered = check.to_string();
        assert!(rendered.contains("Distance"), "rendered = {rendered}");
        assert!(rendered.contains("squared_l2"), "rendered = {rendered}");
        assert!(rendered.contains("100.000"), "rendered = {rendered}");
        assert!(rendered.contains("106.000"), "rendered = {rendered}");
        assert!(rendered.contains("6.000 %"), "rendered = {rendered}");
        assert!(rendered.contains("FAIL"), "rendered = {rendered}");
    }

    // If a "before" value is 0, we should fail with an error because this means the
    // measurement was too fast for us to obtain a reliable signal, so we *could* be letting
    // a regression through.
    //
    // We require at least a non-zero value.
    #[test]
    fn zero_values_rejected() {
        let kernel = Kernel::<diskann_wide::arch::Scalar, f32, f32>::new();

        let result = kernel
            .check(
                &tolerance(0.05),
                &tiny_op(),
                &vec![tiny_result(SimilarityMeasure::SquaredL2, 0)],
                &vec![tiny_result(SimilarityMeasure::SquaredL2, 0)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Fail(_)));
    }
}
