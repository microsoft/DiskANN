/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `Benchmark` impls for the multi-vector MaxSim factory.
//!
//! One entry per element type. Each `try_match` checks `element_type` only;
//! the `isa` field is passed to the library factory at run time. ISA
//! unavailability surfaces as `NotSupported`, which becomes a job-level
//! error.

use std::io::Write;

use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore, PassFail, Regression},
    utils::{datatype::AsDataType, num::relative_change},
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_quantization::multi_vector::{
    build_max_sim_f16, build_max_sim_f32, BoxErase, MaxSimKernel,
};
use rand::distr::{Distribution, StandardUniform};

use super::driver::{
    run_with_distance, BoxedKernel, CheckResult, Comparison, Data, DisplayWrapper,
    MultiVectorTolerance, RunResult,
};
use crate::inputs::multi_vector::MultiVectorOp;

// ─────────────────────────────────────────────────────────────────────────
//  Per-element-type `Benchmark` carriers.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(super) struct KernelF32;

#[derive(Debug)]
pub(super) struct KernelF16;

/// Per-element-type bridge: factory entry name + factory call.
///
/// Data-type matching (`DATA_TYPE`, `is_match`, `describe`) comes from the
/// framework's [`AsDataType`] trait, which is already implemented for `f32`,
/// `half::f16`, etc.
trait ElementType: AsDataType + Copy {
    const ENTRY_NAME: &'static str;
    fn build(
        isa: diskann_quantization::multi_vector::MaxSimIsa,
        query: diskann_quantization::multi_vector::MatRef<
            '_,
            diskann_quantization::multi_vector::Standard<Self>,
        >,
    ) -> Result<Box<dyn MaxSimKernel<Self>>, diskann_quantization::multi_vector::NotSupported>;
}

impl ElementType for f32 {
    const ENTRY_NAME: &'static str = "multi-vector-op-f32";
    fn build(
        isa: diskann_quantization::multi_vector::MaxSimIsa,
        query: diskann_quantization::multi_vector::MatRef<
            '_,
            diskann_quantization::multi_vector::Standard<f32>,
        >,
    ) -> Result<Box<dyn MaxSimKernel<f32>>, diskann_quantization::multi_vector::NotSupported> {
        build_max_sim_f32(isa, query, BoxErase)
    }
}

impl ElementType for half::f16 {
    const ENTRY_NAME: &'static str = "multi-vector-op-f16";
    fn build(
        isa: diskann_quantization::multi_vector::MaxSimIsa,
        query: diskann_quantization::multi_vector::MatRef<
            '_,
            diskann_quantization::multi_vector::Standard<half::f16>,
        >,
    ) -> Result<Box<dyn MaxSimKernel<half::f16>>, diskann_quantization::multi_vector::NotSupported>
    {
        build_max_sim_f16(isa, query, BoxErase)
    }
}

fn run_benchmark<T: ElementType>(input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>>
where
    StandardUniform: Distribution<T>,
{
    let mut results = Vec::with_capacity(input.runs.len());
    for run in input.runs.iter() {
        let data = Data::<T>::new(run);
        let kernel = T::build(input.isa.into(), data.queries.as_view())?;
        let dist = BoxedKernel(kernel);
        results.push(run_with_distance(run, data.docs.as_view(), &dist));
    }
    Ok(results)
}

// ─────────────────────────────────────────────────────────────────────────
//  Benchmark + Regression impls.
// ─────────────────────────────────────────────────────────────────────────

macro_rules! impl_benchmark {
    ($ty:ident, $T:ty) => {
        impl Benchmark for $ty
        where
            StandardUniform: Distribution<$T>,
        {
            type Input = MultiVectorOp;
            type Output = Vec<RunResult>;

            fn try_match(&self, from: &MultiVectorOp) -> Result<MatchScore, FailureScore> {
                crate::utils::match_data_type::<$T>(from.element_type)
            }

            fn run(
                &self,
                input: &MultiVectorOp,
                _: Checkpoint<'_>,
                mut output: &mut dyn Output,
            ) -> anyhow::Result<Self::Output> {
                writeln!(output, "{}", input)?;
                let results = run_benchmark::<$T>(input)?;
                writeln!(output, "\n\n{}", DisplayWrapper(&*results))?;
                Ok(results)
            }

            fn description(
                &self,
                f: &mut std::fmt::Formatter<'_>,
                input: Option<&MultiVectorOp>,
            ) -> std::fmt::Result {
                match input {
                    None => writeln!(f, "- Element Type: {}", <$T as AsDataType>::DATA_TYPE)?,
                    Some(input) => {
                        let desc = <$T as AsDataType>::describe(input.element_type);
                        if !desc.is_match() {
                            writeln!(f, "\n    - Mismatched element type: {}", desc)?;
                        }
                    }
                }
                Ok(())
            }
        }

        impl Regression for $ty
        where
            StandardUniform: Distribution<$T>,
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
                        let before_min =
                            b.percentiles.minimum.as_f64() * 1000.0 / computations_per_latency;
                        let after_min =
                            a.percentiles.minimum.as_f64() * 1000.0 / computations_per_latency;

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

                Ok(if passed {
                    PassFail::Pass(CheckResult { checks })
                } else {
                    PassFail::Fail(CheckResult { checks })
                })
            }
        }
    };
}

impl_benchmark!(KernelF32, f32);
impl_benchmark!(KernelF16, half::f16);

// ─────────────────────────────────────────────────────────────────────────
//  Registration.
// ─────────────────────────────────────────────────────────────────────────

pub(super) fn register(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register_regression(<f32 as ElementType>::ENTRY_NAME, KernelF32)?;
    registry.register_regression(<half::f16 as ElementType>::ENTRY_NAME, KernelF16)?;
    Ok(())
}
