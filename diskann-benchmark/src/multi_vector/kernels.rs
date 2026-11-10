/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `Benchmark` and `Regression` impls for the multi-vector MaxSim factory.
//!
//! Representations have separate registrations but share timing results and regression checks.

use std::io::Write;

use diskann_benchmark_runner::{
    benchmark::{MatchContext, PassFail, Regression, Score},
    utils::{
        datatype::{AsDataType, DataType},
        num::relative_change,
    },
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_quantization::multi_vector::{build_max_sim, BoxErase, MaxSimElement, MaxSimIsa};
use rand::distr::{Distribution, StandardUniform};

use super::driver::{
    run_with_kernel, CheckResult, Comparison, Data, MultiVectorTolerance, RunResult,
};
use crate::inputs::multi_vector::{ElementType, MultiVectorOp, Run};
use crate::utils::DisplayWrapper;

/// Keep registry and serialization adapters monomorphic while allowing each
/// representation to be registered independently. Dispatch is outside the timed loop.
#[derive(Debug)]
pub(super) struct Kernel {
    element_type: &'static str,
    match_element: fn(ElementType, &mut Score),
    run: fn(&MultiVectorOp, &mut dyn Output) -> anyhow::Result<Vec<RunResult>>,
}

impl Kernel {
    pub(super) fn standard<T>() -> Self
    where
        T: MaxSimElement + AsDataType,
        StandardUniform: Distribution<T>,
    {
        Self {
            element_type: T::DATA_TYPE.as_str(),
            match_element: match_standard::<T>,
            run: |input, output| {
                run_job(input, output, &mut |run| {
                    run_standard::<T>(run, input.isa.into())
                })
            },
        }
    }

    pub(super) fn minmax() -> Self {
        Self {
            element_type: "minmax",
            match_element: |element_type, score| {
                if !matches!(element_type, ElementType::MinMax { .. }) {
                    score.fail(1000, &"expected a MinMax element type");
                }
            },
            run: run_minmax,
        }
    }

    pub(super) fn run(
        &self,
        input: &MultiVectorOp,
        output: &mut dyn Output,
    ) -> anyhow::Result<Vec<RunResult>> {
        (self.run)(input, output)
    }
}

impl Benchmark for Kernel {
    type Input = MultiVectorOp;
    type Output = Vec<RunResult>;

    fn try_match(&self, from: &MultiVectorOp, context: &MatchContext) -> Score {
        let mut score = match_isa(from.isa.into(), context);
        (self.match_element)(from.element_type, &mut score);
        score
    }

    fn run(
        &self,
        input: &MultiVectorOp,
        _: Checkpoint<'_>,
        output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        self.run(input, output)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "- Element Type: {}", self.element_type)
    }
}

fn match_standard<T: AsDataType>(element_type: ElementType, score: &mut Score) {
    let data_type = match element_type {
        ElementType::Float32 => Some(DataType::Float32),
        ElementType::Float16 => Some(DataType::Float16),
        ElementType::Int8 => Some(DataType::Int8),
        ElementType::MinMax { .. } => None,
    };
    if let Some(data_type) = data_type {
        crate::utils::match_data_type::<T>(score, data_type);
    } else {
        score.fail(1000, &"expected a primitive element type");
    }
}

fn match_isa(isa: MaxSimIsa, context: &MatchContext) -> Score {
    let mut score = context.success(0);
    if !isa.is_available() {
        score.fail(1, &format_args!("ISA unavailable on this CPU: {}", isa));
    }
    score
}

fn run_job(
    input: &MultiVectorOp,
    mut output: &mut dyn Output,
    run: &mut dyn FnMut(&Run) -> anyhow::Result<RunResult>,
) -> anyhow::Result<Vec<RunResult>> {
    writeln!(output, "{}", input)?;
    let mut results = Vec::with_capacity(input.runs.len());
    for config in &input.runs {
        results.push(run(config)?);
    }
    writeln!(output, "\n\n{}", DisplayWrapper(&*results))?;
    Ok(results)
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

fn run_minmax(input: &MultiVectorOp, output: &mut dyn Output) -> anyhow::Result<Vec<RunResult>> {
    let ElementType::MinMax {
        query_bits,
        doc_bits,
    } = input.element_type
    else {
        anyhow::bail!("expected a MinMax element type, got {}", input.element_type);
    };
    run_job(input, output, &mut |run| {
        super::minmax::run(run, input.isa.into(), query_bits, doc_bits)
    })
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
        check_results(tolerance, before, after)
    }
}

fn check_results(
    tolerance: &MultiVectorTolerance,
    before: &[RunResult],
    after: &[RunResult],
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

// ─────────────────────────────────────────────────────────────────────────
//  Registration.
// ─────────────────────────────────────────────────────────────────────────

pub(super) fn register(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register_regression("multi-vector-op-f32", Kernel::standard::<f32>())?;
    registry.register_regression("multi-vector-op-f16", Kernel::standard::<half::f16>())?;
    registry.register_regression("multi-vector-op-i8", Kernel::standard::<i8>())?;
    registry.register_regression("multi-vector-op-minmax", Kernel::minmax())?;
    Ok(())
}
