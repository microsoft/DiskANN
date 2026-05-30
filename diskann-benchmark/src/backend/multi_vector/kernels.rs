/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `Benchmark` and `Regression` impls for the multi-vector MaxSim factory.
//!
//! A single generic [`Kernel<T>`] carrier covers every element type accepted
//! by [`MaxSimElement`]; `try_match` also rejects ISAs unavailable on the
//! host so unsupported jobs fail at job-selection rather than mid-run.

use std::io::Write;
use std::marker::PhantomData;

use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore, PassFail, Regression},
    utils::{datatype::AsDataType, num::relative_change},
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_quantization::multi_vector::{build_max_sim, BoxErase, MaxSimElement, MaxSimIsa};
use rand::distr::{Distribution, StandardUniform};

use super::driver::{
    run_with_kernel, CheckResult, Comparison, Data, MultiVectorTolerance, RunResult,
};
use crate::inputs::multi_vector::MultiVectorOp;
use crate::utils::DisplayWrapper;

// ─────────────────────────────────────────────────────────────────────────
//  Kernel<T> — generic carrier registered once per element type.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(super) struct Kernel<T>(PhantomData<T>);

impl<T> Kernel<T> {
    pub(super) const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Benchmark for Kernel<T>
where
    T: MaxSimElement + AsDataType,
    StandardUniform: Distribution<T>,
{
    type Input = MultiVectorOp;
    type Output = Vec<RunResult>;

    fn try_match(&self, from: &MultiVectorOp) -> Result<MatchScore, FailureScore> {
        let mut failscore: Option<u32> = None;
        if crate::utils::match_data_type::<T>(from.element_type).is_err() {
            *failscore.get_or_insert(0) += 1;
        }
        let isa: MaxSimIsa = from.isa.into();
        if !isa.is_available() {
            *failscore.get_or_insert(0) += 1;
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
        let mut results = Vec::with_capacity(input.runs.len());
        for run in input.runs.iter() {
            let data = Data::<T>::new(run)?;
            let kernel = build_max_sim::<T, _>(input.isa.into(), data.queries.as_view(), BoxErase)?;
            results.push(run_with_kernel(run, data.docs.as_view(), &*kernel));
        }
        writeln!(output, "\n\n{}", DisplayWrapper(&*results))?;
        Ok(results)
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&MultiVectorOp>,
    ) -> std::fmt::Result {
        match input {
            None => writeln!(f, "- Element Type: {}", <T as AsDataType>::DATA_TYPE)?,
            Some(input) => {
                let desc = <T as AsDataType>::describe(input.element_type);
                if !desc.is_match() {
                    writeln!(f, "\n    - Mismatched element type: {}", desc)?;
                }
                let isa: MaxSimIsa = input.isa.into();
                if !isa.is_available() {
                    writeln!(f, "\n    - ISA unavailable on this CPU: {}", isa)?;
                }
            }
        }
        Ok(())
    }
}

impl<T> Regression for Kernel<T>
where
    T: MaxSimElement + AsDataType,
    StandardUniform: Distribution<T>,
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

        Ok(if passed {
            PassFail::Pass(CheckResult { checks })
        } else {
            PassFail::Fail(CheckResult { checks })
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Registration.
// ─────────────────────────────────────────────────────────────────────────

pub(super) fn register(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register_regression("multi-vector-op-f32", Kernel::<f32>::new())?;
    registry.register_regression("multi-vector-op-f16", Kernel::<half::f16>::new())?;
    Ok(())
}
