/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `Benchmark` and `Regression` impls for the multi-vector MaxSim factory.
//!
//! All element types share job selection, timing results and regression checks.

use std::io::Write;

use diskann_benchmark_runner::{
    benchmark::{MatchContext, PassFail, Regression, Score},
    utils::num::relative_change,
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_quantization::multi_vector::{build_max_sim, BoxErase, MaxSimElement, MaxSimIsa};
use rand::distr::{Distribution, StandardUniform};

use super::driver::{
    run_with_kernel, CheckResult, Comparison, Data, MultiVectorTolerance, RunResult,
};
use crate::inputs::multi_vector::{ElementType, MultiVectorOp, Run};
use crate::utils::DisplayWrapper;

#[derive(Debug)]
pub(super) struct Kernel;

impl Benchmark for Kernel {
    type Input = MultiVectorOp;
    type Output = Vec<RunResult>;

    fn try_match(&self, from: &MultiVectorOp, context: &MatchContext) -> Score {
        let mut score = context.success(0);
        let isa: MaxSimIsa = from.isa.into();
        if !isa.is_available() {
            score.fail(1, &format_args!("ISA unavailable on this CPU: {}", isa));
        }
        score
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
            let isa = input.isa.into();
            let result = match input.element_type {
                ElementType::Float32 => run_standard::<f32>(run, isa),
                ElementType::Float16 => run_standard::<half::f16>(run, isa),
                ElementType::Int8 => run_standard::<i8>(run, isa),
                ElementType::MinMax {
                    query_bits,
                    doc_bits,
                } => super::minmax::run(run, isa, query_bits, doc_bits),
            }?;
            results.push(result);
        }
        writeln!(output, "\n\n{}", DisplayWrapper(&*results))?;
        Ok(results)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "- Element Types: float32, float16, int8, minmax")
    }
}

fn run_standard<T>(run: &Run, isa: MaxSimIsa) -> anyhow::Result<RunResult>
where
    T: MaxSimElement,
    StandardUniform: Distribution<T>,
{
    let data = Data::<T>::new(run)?;
    let kernel = build_max_sim::<T, _>(isa, data.queries.as_view(), BoxErase)?;
    let mut scores = vec![T::Score::default(); kernel.nrows()];
    Ok(run_with_kernel(run, &mut scores, |scores| {
        kernel.compute_max_sim(data.docs.as_view(), scores)
    })?)
}

impl Regression for Kernel {
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
    registry.register_regression("multi-vector-op", Kernel)?;
    Ok(())
}
