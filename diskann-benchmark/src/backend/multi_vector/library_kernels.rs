/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Library kernel registrations and arch-dispatch machinery.
//!
//! Mirrors the structure of `diskann-benchmark-simd`: a `Kernel<A, T>`
//! PhantomData carrier carries the (arch × element type) pair through the
//! benchmark registry, [`DispatchRule<Arch>`] maps the JSON-facing `Arch`
//! enum to a concrete arch token, and the `stamp!` / `match_arch!` macros
//! generate the repetitive `RunBenchmark<A>` / `DispatchRule` impls.
//!
//! Library kernels registered here:
//! - `multi-vector-op-{f32,f16}-auto` — `QueryComputer::new` (auto-dispatch)
//! - `multi-vector-op-{f32,f16}-scalar` — `from_arch(Scalar)`
//! - `multi-vector-op-{f32,f16}-x86_64_V3` — `from_arch(V3)` (x86_64 only)
//! - `multi-vector-op-{f32,f16}-x86_64_V4` — `from_arch(V4)` (x86_64 only)
//! - `multi-vector-op-{f32,f16}-aarch64_neon` — `from_arch(Neon)` (aarch64 only)
//! - `multi-vector-op-{f32,f16}-reference` — `MaxSim` fallback

use std::io::Write;
use std::marker::PhantomData;

use diskann_benchmark_runner::{
    benchmark::{PassFail, Regression},
    dispatcher::{Description, DispatchRule, FailureScore, MatchScore},
    utils::{datatype, num::relative_change},
    Benchmark, Checkpoint, Output,
};
use diskann_quantization::multi_vector::{MatRef, QueryComputer, Standard};
use diskann_vector::distance::InnerProduct;
use diskann_vector::PureDistanceFunction;
#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};
use diskann_wide::arch::Scalar;
use diskann_wide::Architecture;
use rand::distr::{Distribution, StandardUniform};

use super::driver::{
    run_with_distance, CheckResult, Comparison, Data, DisplayWrapper, OptimizedDistance,
    ReferenceDistance, RunResult,
};
use crate::inputs::multi_vector::{Arch, MultiVectorOp, MultiVectorTolerance};

/// PhantomData carrier for one (arch, element-type) entry in the benchmark
/// registry. The arch parameter `A` is either a real arch token (`Scalar`,
/// `V3`, `V4`, `Neon`) or one of the marker types [`Auto`] / [`Reference`].
pub(super) struct Kernel<A, T> {
    _type: PhantomData<(A, T)>,
}

impl<A, T> Kernel<A, T> {
    pub(super) fn new() -> Self {
        Self { _type: PhantomData }
    }
}

/// Marker for the auto-dispatched (CPU-detected) kernel — `QueryComputer::new`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Auto;

/// Marker for the reference (`MaxSim` fallback) kernel.
#[derive(Debug, Clone, Copy)]
pub(super) struct Reference;

/// Wrapper around an arch token (real or marker) that implements
/// [`DispatchRule<Arch>`] for the JSON-facing [`Arch`] enum.
pub(super) struct Identity<A>(pub(super) A);

/// Returned by `Identity::<A>::convert` when the host CPU doesn't support the
/// requested ISA. The dispatcher converts this into a friendly error message.
#[derive(Debug, Clone, Copy)]
pub(super) struct ArchNotSupported(pub(super) Arch);

impl std::fmt::Display for ArchNotSupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} not supported on this CPU", self.0)
    }
}

impl std::error::Error for ArchNotSupported {}

//////////////////////
// Dispatch rules   //
//////////////////////

/// Generates a [`DispatchRule<Arch>`] for a real arch token. `try_match` returns:
/// - `Ok(MatchScore(0))` when the input names this arch AND the host CPU supports it
/// - `Err(FailureScore(0))` when the input names this arch but the CPU doesn't support it
///   (this surfaces in the dispatcher's near-miss diagnostic)
/// - `Err(FailureScore(1))` when the input names a different arch
macro_rules! match_arch_x86_64 {
    ($arch:path, $enum:ident) => {
        #[cfg(target_arch = "x86_64")]
        impl DispatchRule<Arch> for Identity<$arch> {
            type Error = ArchNotSupported;
            fn try_match(from: &Arch) -> Result<MatchScore, FailureScore> {
                if *from != Arch::$enum {
                    return Err(FailureScore(1));
                }
                if <$arch>::new_checked().is_some() {
                    Ok(MatchScore(0))
                } else {
                    Err(FailureScore(0))
                }
            }
            fn convert(from: Arch) -> Result<Self, Self::Error> {
                <$arch>::new_checked()
                    .ok_or(ArchNotSupported(from))
                    .map(Identity)
            }
        }
    };
}

match_arch_x86_64!(V3, X86_64_V3);
match_arch_x86_64!(V4, X86_64_V4);

#[cfg(target_arch = "aarch64")]
impl DispatchRule<Arch> for Identity<Neon> {
    type Error = ArchNotSupported;
    fn try_match(from: &Arch) -> Result<MatchScore, FailureScore> {
        if *from != Arch::Neon {
            return Err(FailureScore(1));
        }
        if Neon::new_checked().is_some() {
            Ok(MatchScore(0))
        } else {
            Err(FailureScore(0))
        }
    }
    fn convert(from: Arch) -> Result<Self, Self::Error> {
        Neon::new_checked()
            .ok_or(ArchNotSupported(from))
            .map(Identity)
    }
}

// Scalar is always available; no CPU check needed.
impl DispatchRule<Arch> for Identity<Scalar> {
    type Error = ArchNotSupported;
    fn try_match(from: &Arch) -> Result<MatchScore, FailureScore> {
        if *from == Arch::Scalar {
            Ok(MatchScore(0))
        } else {
            Err(FailureScore(1))
        }
    }
    fn convert(_from: Arch) -> Result<Self, Self::Error> {
        Ok(Identity(Scalar::new()))
    }
}

impl DispatchRule<Arch> for Identity<Auto> {
    type Error = ArchNotSupported;
    fn try_match(from: &Arch) -> Result<MatchScore, FailureScore> {
        if *from == Arch::Auto {
            Ok(MatchScore(0))
        } else {
            Err(FailureScore(1))
        }
    }
    fn convert(_from: Arch) -> Result<Self, Self::Error> {
        Ok(Identity(Auto))
    }
}

impl DispatchRule<Arch> for Identity<Reference> {
    type Error = ArchNotSupported;
    fn try_match(from: &Arch) -> Result<MatchScore, FailureScore> {
        if *from == Arch::Reference {
            Ok(MatchScore(0))
        } else {
            Err(FailureScore(1))
        }
    }
    fn convert(_from: Arch) -> Result<Self, Self::Error> {
        Ok(Identity(Reference))
    }
}

//////////////////////
// Benchmark trait  //
//////////////////////

/// Per-arch run trait. The `stamp!` macro generates impls for real arch tokens;
/// `Auto` and `Reference` get hand-written impls.
pub(super) trait RunBenchmark<A> {
    fn run_benchmark(&self, input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>>;
}

impl<A, T> Benchmark for Kernel<A, T>
where
    datatype::Type<T>: DispatchRule<datatype::DataType>,
    Identity<A>: DispatchRule<Arch, Error = ArchNotSupported>,
    Kernel<A, T>: RunBenchmark<A>,
    A: 'static,
    T: 'static,
{
    type Input = MultiVectorOp;
    type Output = Vec<RunResult>;

    fn try_match(&self, from: &MultiVectorOp) -> Result<MatchScore, FailureScore> {
        let mut failscore: Option<u32> = None;
        if datatype::Type::<T>::try_match(&from.element_type).is_err() {
            *failscore.get_or_insert(0) += 10;
        }
        match Identity::<A>::try_match(&from.arch) {
            Ok(MatchScore(_)) => (),
            Err(FailureScore(score)) => {
                *failscore.get_or_insert(0) += score;
            }
        }
        match failscore {
            None => Ok(MatchScore(0)),
            Some(score) => Err(FailureScore(score)),
        }
    }

    fn run(
        &self,
        input: &MultiVectorOp,
        _: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
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
                writeln!(f, "- Arch: {}", Description::<Arch, Identity<A>>::new())?;
            }
            Some(input) => {
                if let Err(err) = datatype::Type::<T>::try_match_verbose(&input.element_type) {
                    writeln!(f, "\n    - Mismatched element type: {}", err)?;
                }
                if Identity::<A>::try_match(&input.arch).is_err() {
                    writeln!(f, "\n    - Wrong or unsupported arch: {}", input.arch)?;
                }
            }
        }
        Ok(())
    }
}

impl<A, T> Regression for Kernel<A, T>
where
    datatype::Type<T>: DispatchRule<datatype::DataType>,
    Identity<A>: DispatchRule<Arch, Error = ArchNotSupported>,
    Kernel<A, T>: RunBenchmark<A>,
    A: 'static,
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
        Ok(if passed {
            PassFail::Pass(check)
        } else {
            PassFail::Fail(check)
        })
    }
}

//////////////////////
// RunBenchmark impls
//////////////////////

/// Element-type-erasing constructor for [`QueryComputer<T>`]. `QueryComputer`'s
/// `new` / `from_arch` are inherent methods on the concrete `QueryComputer<f32>`
/// and `QueryComputer<half::f16>` types, so generic code needs this shim.
pub(super) trait BuildArchQc<T: Copy> {
    /// Build a `QueryComputer<T>` pinned to the host's auto-dispatched arch.
    fn build_auto(query: MatRef<'_, Standard<T>>) -> QueryComputer<T>;
}

impl BuildArchQc<f32> for f32 {
    fn build_auto(query: MatRef<'_, Standard<f32>>) -> QueryComputer<f32> {
        QueryComputer::<f32>::new(query)
    }
}

impl BuildArchQc<half::f16> for half::f16 {
    fn build_auto(query: MatRef<'_, Standard<half::f16>>) -> QueryComputer<half::f16> {
        QueryComputer::<half::f16>::new(query)
    }
}

/// Per-(arch, T) constructor for `QueryComputer::from_arch`. Same idea as
/// [`BuildArchQc::build_auto`] but pinned to a specific arch token.
pub(super) trait BuildPinnedQc<A: Architecture, T: Copy> {
    fn build_pinned(query: MatRef<'_, Standard<T>>, arch: A) -> QueryComputer<T>;
}

macro_rules! impl_build_pinned {
    ($arch:path, $T:ty) => {
        impl BuildPinnedQc<$arch, $T> for $T {
            fn build_pinned(query: MatRef<'_, Standard<$T>>, arch: $arch) -> QueryComputer<$T> {
                QueryComputer::<$T>::from_arch(query, arch)
            }
        }
    };
}

impl_build_pinned!(Scalar, f32);
impl_build_pinned!(Scalar, half::f16);
#[cfg(target_arch = "x86_64")]
impl_build_pinned!(V3, f32);
#[cfg(target_arch = "x86_64")]
impl_build_pinned!(V3, half::f16);
#[cfg(target_arch = "x86_64")]
impl_build_pinned!(V4, f32);
#[cfg(target_arch = "x86_64")]
impl_build_pinned!(V4, half::f16);
#[cfg(target_arch = "aarch64")]
impl_build_pinned!(Neon, f32);
#[cfg(target_arch = "aarch64")]
impl_build_pinned!(Neon, half::f16);

/// Stamp out `RunBenchmark<$arch>` for `Kernel<$arch, $T>` using
/// `QueryComputer::<T>::from_arch($arch_token)`.
macro_rules! stamp {
    ($arch:path, $T:ty) => {
        impl RunBenchmark<$arch> for Kernel<$arch, $T>
        where
            StandardUniform: Distribution<$T>,
            $T: BuildPinnedQc<$arch, $T>,
        {
            fn run_benchmark(&self, input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>> {
                let arch = Identity::<$arch>::convert(input.arch)?.0;
                let mut results = Vec::with_capacity(input.runs.len());
                for run in input.runs.iter() {
                    let data = Data::<$T>::new(run);
                    // `QueryComputer` performs query-side precomputation that is
                    // intentionally amortized across many `max_sim` calls;
                    // construct it once per shape, outside the timed loop.
                    let qc = <$T as BuildPinnedQc<$arch, $T>>::build_pinned(
                        data.queries.as_view(),
                        arch,
                    );
                    let dist = OptimizedDistance(qc);
                    results.push(run_with_distance(run, data.docs.as_view(), &dist));
                }
                Ok(results)
            }
        }
    };
    ($target_arch:literal, $arch:path, $T:ty) => {
        #[cfg(target_arch = $target_arch)]
        stamp!($arch, $T);
    };
}

stamp!(Scalar, f32);
stamp!(Scalar, half::f16);
stamp!("x86_64", V3, f32);
stamp!("x86_64", V3, half::f16);
stamp!("x86_64", V4, f32);
stamp!("x86_64", V4, half::f16);
stamp!("aarch64", Neon, f32);
stamp!("aarch64", Neon, half::f16);

// Auto and Reference get hand-written impls (different construction paths).

impl<T> RunBenchmark<Auto> for Kernel<Auto, T>
where
    T: Copy + 'static + BuildArchQc<T>,
    StandardUniform: Distribution<T>,
{
    fn run_benchmark(&self, input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>> {
        let mut results = Vec::with_capacity(input.runs.len());
        for run in input.runs.iter() {
            let data = Data::<T>::new(run);
            let qc = <T as BuildArchQc<T>>::build_auto(data.queries.as_view());
            let dist = OptimizedDistance(qc);
            results.push(run_with_distance(run, data.docs.as_view(), &dist));
        }
        Ok(results)
    }
}

impl<T> RunBenchmark<Reference> for Kernel<Reference, T>
where
    T: Copy + 'static,
    StandardUniform: Distribution<T>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
    for<'a> ReferenceDistance<'a, T>: super::driver::Distance<T>,
{
    fn run_benchmark(&self, input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>> {
        let mut results = Vec::with_capacity(input.runs.len());
        for run in input.runs.iter() {
            let data = Data::<T>::new(run);
            let dist = ReferenceDistance(data.queries.as_view().into());
            results.push(run_with_distance(run, data.docs.as_view(), &dist));
        }
        Ok(results)
    }
}

//////////////////////
// Registration     //
//////////////////////

pub(super) fn register(benchmarks: &mut diskann_benchmark_runner::registry::Benchmarks) {
    benchmarks.register_regression("multi-vector-op-f32-auto", Kernel::<Auto, f32>::new());
    benchmarks.register_regression("multi-vector-op-f16-auto", Kernel::<Auto, half::f16>::new());

    benchmarks.register_regression("multi-vector-op-f32-scalar", Kernel::<Scalar, f32>::new());
    benchmarks.register_regression(
        "multi-vector-op-f16-scalar",
        Kernel::<Scalar, half::f16>::new(),
    );

    benchmarks.register_regression(
        "multi-vector-op-f32-reference",
        Kernel::<Reference, f32>::new(),
    );
    benchmarks.register_regression(
        "multi-vector-op-f16-reference",
        Kernel::<Reference, half::f16>::new(),
    );

    #[cfg(target_arch = "x86_64")]
    {
        benchmarks.register_regression("multi-vector-op-f32-x86_64_V3", Kernel::<V3, f32>::new());
        benchmarks.register_regression(
            "multi-vector-op-f16-x86_64_V3",
            Kernel::<V3, half::f16>::new(),
        );
        benchmarks.register_regression("multi-vector-op-f32-x86_64_V4", Kernel::<V4, f32>::new());
        benchmarks.register_regression(
            "multi-vector-op-f16-x86_64_V4",
            Kernel::<V4, half::f16>::new(),
        );
    }

    #[cfg(target_arch = "aarch64")]
    {
        benchmarks.register_regression(
            "multi-vector-op-f32-aarch64_neon",
            Kernel::<Neon, f32>::new(),
        );
        benchmarks.register_regression(
            "multi-vector-op-f16-aarch64_neon",
            Kernel::<Neon, half::f16>::new(),
        );
    }
}
