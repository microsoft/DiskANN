/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

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
    describeln,
    dispatcher::{self, DispatchRule, FailureScore, MatchScore},
    utils::{
        datatype::{self, DataType},
        percentiles, MicroSeconds,
    },
    Any, CheckDeserialization, Checker,
};

////////////////
// Public API //
////////////////

#[derive(Debug)]
pub struct SimdInput;

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
pub(crate) enum Arch {
    #[serde(rename = "x86-64-v4")]
    #[allow(non_camel_case_types)]
    X86_64_V4,
    #[serde(rename = "x86-64-v3")]
    #[allow(non_camel_case_types)]
    X86_64_V3,
    Scalar,
    Reference,
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = match self {
            Self::X86_64_V4 => "x86-64-v4",
            Self::X86_64_V3 => "x86-64-v3",
            Self::Scalar => "scalar",
            Self::Reference => "reference",
        };
        write!(f, "{}", st)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Run {
    pub(crate) distance: SimilarityMeasure,
    pub(crate) dim: NonZeroUsize,
    pub(crate) num_points: NonZeroUsize,
    pub(crate) loops_per_measurement: NonZeroUsize,
    pub(crate) num_measurements: NonZeroUsize,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SimdOp {
    pub(crate) query_type: DataType,
    pub(crate) data_type: DataType,
    pub(crate) arch: Arch,
    pub(crate) runs: Vec<Run>,
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
    pub(crate) const fn tag() -> &'static str {
        "simd-op"
    }

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

impl diskann_benchmark_runner::Input for SimdInput {
    fn tag(&self) -> &'static str {
        "simd-op"
    }

    fn try_deserialize(
        &self,
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(SimdOp::deserialize(serialized)?)
    }

    fn example(&self) -> anyhow::Result<serde_json::Value> {
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

        Ok(serde_json::to_value(&SimdOp {
            query_type: DataType::Float32,
            data_type: DataType::Float32,
            arch: Arch::X86_64_V3,
            runs,
        })?)
    }
}

////////////////////////////
// Benchmark Registration //
////////////////////////////

macro_rules! register {
    ($arch:literal, $dispatcher:ident, $name:literal, $($kernel:tt)*) => {
        #[cfg(target_arch = $arch)]
        $dispatcher.register::<$($kernel)*>(
            $name,
            run_benchmark,
        )
    };
    ($dispatcher:ident, $name:literal, $($kernel:tt)*) => {
        $dispatcher.register::<$($kernel)*>(
            $name,
            run_benchmark,
        )
    };
}

fn register_benchmarks_impl(dispatcher: &mut diskann_benchmark_runner::registry::Benchmarks) {
    // x86-64-v4
    register!(
        "x86_64",
        dispatcher,
        "simd-op-f32xf32-x86_64_V4",
        Kernel<'static, diskann_wide::arch::x86_64::V4, f32, f32>
    );
    register!(
        "x86_64",
        dispatcher,
        "simd-op-f16xf16-x86_64_V4",
        Kernel<'static, diskann_wide::arch::x86_64::V4, f16, f16>
    );
    register!(
        "x86_64",
        dispatcher,
        "simd-op-u8xu8-x86_64_V4",
        Kernel<'static, diskann_wide::arch::x86_64::V4, u8, u8>
    );
    register!(
        "x86_64",
        dispatcher,
        "simd-op-i8xi8-x86_64_V4",
        Kernel<'static, diskann_wide::arch::x86_64::V4, i8, i8>
    );

    // x86-64-v3
    register!(
        "x86_64",
        dispatcher,
        "simd-op-f32xf32-x86_64_V3",
        Kernel<'static, diskann_wide::arch::x86_64::V3, f32, f32>
    );
    register!(
        "x86_64",
        dispatcher,
        "simd-op-f16xf16-x86_64_V3",
        Kernel<'static, diskann_wide::arch::x86_64::V3, f16, f16>
    );
    register!(
        "x86_64",
        dispatcher,
        "simd-op-u8xu8-x86_64_V3",
        Kernel<'static, diskann_wide::arch::x86_64::V3, u8, u8>
    );
    register!(
        "x86_64",
        dispatcher,
        "simd-op-i8xi8-x86_64_V3",
        Kernel<'static, diskann_wide::arch::x86_64::V3, i8, i8>
    );

    // scalar
    register!(
        dispatcher,
        "simd-op-f32xf32-scalar",
        Kernel<'static, diskann_wide::arch::Scalar, f32, f32>
    );
    register!(
        dispatcher,
        "simd-op-f16xf16-scalar",
        Kernel<'static, diskann_wide::arch::Scalar, f16, f16>
    );
    register!(
        dispatcher,
        "simd-op-u8xu8-scalar",
        Kernel<'static, diskann_wide::arch::Scalar, u8, u8>
    );
    register!(
        dispatcher,
        "simd-op-i8xi8-scalar",
        Kernel<'static, diskann_wide::arch::Scalar, i8, i8>
    );

    // reference
    register!(
        dispatcher,
        "simd-op-f32xf32-reference",
        Kernel<'static, Reference, f32, f32>
    );
    register!(
        dispatcher,
        "simd-op-f16xf16-reference",
        Kernel<'static, Reference, f16, f16>
    );
    register!(
        dispatcher,
        "simd-op-u8xu8-reference",
        Kernel<'static, Reference, u8, u8>
    );
    register!(
        dispatcher,
        "simd-op-i8xi8-reference",
        Kernel<'static, Reference, i8, i8>
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

impl<T> dispatcher::Map for Identity<T>
where
    T: 'static,
{
    type Type<'a> = T;
}

struct Kernel<'a, A, Q, D> {
    input: &'a SimdOp,
    arch: A,
    _type: std::marker::PhantomData<(A, Q, D)>,
}

impl<'a, A, Q, D> Kernel<'a, A, Q, D> {
    fn new(input: &'a SimdOp, arch: A) -> Self {
        Self {
            input,
            arch,
            _type: std::marker::PhantomData,
        }
    }
}

impl<A, Q, D> dispatcher::Map for Kernel<'static, A, Q, D>
where
    A: 'static,
    Q: 'static,
    D: 'static,
{
    type Type<'a> = Kernel<'a, A, Q, D>;
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
            Err(FailureScore(0))
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
            Err(FailureScore(0))
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

#[cfg(target_arch = "x86_64")]
impl DispatchRule<Arch> for Identity<diskann_wide::arch::x86_64::V4> {
    type Error = ArchNotSupported;

    fn try_match(from: &Arch) -> Result<MatchScore, FailureScore> {
        if *from == Arch::X86_64_V4 {
            Ok(MatchScore(0))
        } else {
            Err(FailureScore(0))
        }
    }

    fn convert(from: Arch) -> Result<Self, Self::Error> {
        assert_eq!(from, Arch::X86_64_V4);
        diskann_wide::arch::x86_64::V4::new_checked()
            .ok_or(ArchNotSupported(from))
            .map(Identity)
    }

    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&Arch>) -> std::fmt::Result {
        match from {
            None => write!(f, "x86-64-v4"),
            Some(arch) => {
                if Self::try_match(arch).is_ok() {
                    write!(f, "matched {}", arch)
                } else {
                    write!(f, "expected {}, got {}", Arch::X86_64_V4, arch)
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl DispatchRule<Arch> for Identity<diskann_wide::arch::x86_64::V3> {
    type Error = ArchNotSupported;

    fn try_match(from: &Arch) -> Result<MatchScore, FailureScore> {
        if *from == Arch::X86_64_V3 {
            Ok(MatchScore(0))
        } else {
            Err(FailureScore(0))
        }
    }

    fn convert(from: Arch) -> Result<Self, Self::Error> {
        assert_eq!(from, Arch::X86_64_V3);
        diskann_wide::arch::x86_64::V3::new_checked()
            .ok_or(ArchNotSupported(from))
            .map(Identity)
    }

    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&Arch>) -> std::fmt::Result {
        match from {
            None => write!(f, "x86-64-v3"),
            Some(arch) => {
                if Self::try_match(arch).is_ok() {
                    write!(f, "matched {}", arch)
                } else {
                    write!(f, "expected {}, got {}", Arch::X86_64_V3, arch)
                }
            }
        }
    }
}

impl<'a, A, Q, D> DispatchRule<&'a SimdOp> for Kernel<'a, A, Q, D>
where
    datatype::Type<Q>: DispatchRule<datatype::DataType>,
    datatype::Type<D>: DispatchRule<datatype::DataType>,
    Identity<A>: DispatchRule<Arch, Error = ArchNotSupported>,
{
    type Error = ArchNotSupported;

    // Matching simply requires that we match the inner type.
    fn try_match(from: &&'a SimdOp) -> Result<MatchScore, FailureScore> {
        let mut failscore: Option<u32> = None;
        if datatype::Type::<Q>::try_match(&from.query_type).is_err() {
            *failscore.get_or_insert(0) += 10;
        }
        if datatype::Type::<D>::try_match(&from.data_type).is_err() {
            *failscore.get_or_insert(0) += 10;
        }
        if Identity::<A>::try_match(&from.arch).is_err() {
            *failscore.get_or_insert(0) += 2;
        }
        match failscore {
            None => Ok(MatchScore(0)),
            Some(score) => Err(FailureScore(score)),
        }
    }

    fn convert(from: &'a SimdOp) -> Result<Self, Self::Error> {
        assert!(Self::try_match(&from).is_ok());
        let arch = Identity::<A>::convert(from.arch)?.0;
        Ok(Self::new(from, arch))
    }

    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&&'a SimdOp>) -> std::fmt::Result {
        match from {
            None => {
                describeln!(
                    f,
                    "- Query Type: {}",
                    dispatcher::Description::<datatype::DataType, datatype::Type<Q>>::new()
                )?;
                describeln!(
                    f,
                    "- Data Type: {}",
                    dispatcher::Description::<datatype::DataType, datatype::Type<D>>::new()
                )?;
                describeln!(
                    f,
                    "- Implementation: {}",
                    dispatcher::Description::<Arch, Identity<A>>::new()
                )?;
            }
            Some(input) => {
                if let Err(err) = datatype::Type::<Q>::try_match_verbose(&input.query_type) {
                    describeln!(f, "- Mismatched query type: {}", err)?;
                }
                if let Err(err) = datatype::Type::<D>::try_match_verbose(&input.data_type) {
                    describeln!(f, "- Mismatched data type: {}", err)?;
                }
                if let Err(err) = Identity::<A>::try_match_verbose(&input.arch) {
                    describeln!(f, "- Mismatched architecture: {}", err)?;
                }
            }
        }
        Ok(())
    }
}

impl<'a, A, Q, D> DispatchRule<&'a diskann_benchmark_runner::Any> for Kernel<'a, A, Q, D>
where
    Kernel<'a, A, Q, D>: DispatchRule<&'a SimdOp>,
    <Kernel<'a, A, Q, D> as DispatchRule<&'a SimdOp>>::Error:
        std::error::Error + Send + Sync + 'static,
{
    type Error = anyhow::Error;

    fn try_match(from: &&'a diskann_benchmark_runner::Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<SimdOp, Self>()
    }

    fn convert(from: &'a diskann_benchmark_runner::Any) -> Result<Self, Self::Error> {
        from.convert::<SimdOp, Self>()
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        from: Option<&&'a diskann_benchmark_runner::Any>,
    ) -> std::fmt::Result {
        Any::description::<SimdOp, Self>(f, from, SimdOp::tag())
    }
}

///////////////
// Benchmark //
///////////////

fn run_benchmark<A, Q, D>(
    kernel: Kernel<'_, A, Q, D>,
    _: diskann_benchmark_runner::Checkpoint<'_>,
    mut output: &mut dyn diskann_benchmark_runner::Output,
) -> Result<serde_json::Value, anyhow::Error>
where
    for<'a> Kernel<'a, A, Q, D>: RunBenchmark,
{
    writeln!(output, "{}", kernel.input)?;
    let results = kernel.run()?;
    writeln!(output, "\n\n{}", DisplayWrapper(&*results))?;
    Ok(serde_json::to_value(results)?)
}

trait RunBenchmark {
    fn run(self) -> Result<Vec<RunResult>, anyhow::Error>;
}

#[derive(Debug, Serialize)]
struct RunResult {
    /// The setup
    run: Run,
    /// The latencies of individual runs.
    latencies: Vec<MicroSeconds>,
    /// Latency percentiles.
    percentiles: percentiles::Percentiles<MicroSeconds>,
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

            let computations_per_latency: f64 =
                (r.run.num_points.get() * r.run.loops_per_measurement.get()) as f64;

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
        impl RunBenchmark for Kernel<'_, Reference, $Q, $D> {
            fn run(self) -> Result<Vec<RunResult>, anyhow::Error> {
                let mut results = Vec::new();
                for run in self.input.runs.iter() {
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
        impl RunBenchmark for Kernel<'_, $arch, $Q, $D> {
            fn run(self) -> Result<Vec<RunResult>, anyhow::Error> {
                let mut results = Vec::new();

                let l2 = &simd::L2 {};
                let ip = &simd::IP {};
                let cosine = &simd::CosineStateless {};

                for run in self.input.runs.iter() {
                    let data = Data::<$Q, $D>::new(run);
                    // For each kernel, we need to do a two-step wrapping of closures so
                    // the inner-most closure is executed by the architecture.
                    //
                    // This is required for the implementation of `simd_op` to be inlined
                    // into the architecture run function so it can properly inherit
                    // target features.
                    let result = match run.distance {
                        SimilarityMeasure::SquaredL2 => data.run(run, |q, d| {
                            self.arch
                                .run2(|q, d| simd::simd_op(l2, self.arch, q, d), q, d)
                        }),
                        SimilarityMeasure::InnerProduct => data.run(run, |q, d| {
                            self.arch
                                .run2(|q, d| simd::simd_op(ip, self.arch, q, d), q, d)
                        }),
                        SimilarityMeasure::Cosine => data.run(run, |q, d| {
                            self.arch
                                .run2(|q, d| simd::simd_op(cosine, self.arch, q, d), q, d)
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
