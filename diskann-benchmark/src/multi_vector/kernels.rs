/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `Benchmark` and `Regression` impls for the multi-vector MaxSim factory.
//!
//! A single generic [`Kernel<T>`] carrier covers every element type accepted
//! by [`MaxSimElement`]; `try_match` also rejects ISAs unavailable on the
//! host so unsupported jobs fail at job-selection rather than mid-run.
//! MinMax8/MinMax4 uses a separate carrier with the same input and result format.

use std::io::Write;
use std::marker::PhantomData;

use diskann_benchmark_runner::{
    benchmark::{MatchContext, PassFail, Regression, Score},
    utils::{
        datatype::{AsDataType, DataType},
        num::relative_change,
    },
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_quantization::minmax::build_minmax_max_sim;
use diskann_quantization::multi_vector::{build_max_sim, BoxErase, MaxSimElement, MaxSimIsa};
use rand::distr::{Distribution, StandardUniform};

use super::driver::{
    run_with_compute, run_with_kernel, CheckResult, Comparison, Data, MinMax8Data,
    MultiVectorTolerance, RunResult,
};
use crate::inputs::multi_vector::{ElementType, MultiVectorOp};
use crate::utils::{DisplayWrapper, DATA_TYPE_MISMATCH};

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

    fn try_match(&self, from: &MultiVectorOp, context: &MatchContext) -> Score {
        let data_type = match from.element_type {
            ElementType::Float32 => DataType::Float32,
            ElementType::Float16 => DataType::Float16,
            ElementType::Int8 => DataType::Int8,
            ElementType::MinMax8 => {
                return context.fail(DATA_TYPE_MISMATCH, &"MinMax8 requires the quantized kernel");
            }
        };
        let mut score = context.success(0);
        crate::utils::match_data_type::<T>(&mut score, data_type);
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
            let data = Data::<T>::new(run)?;
            let kernel = build_max_sim::<T, _>(input.isa.into(), data.queries.as_view(), BoxErase)?;
            results.push(run_with_kernel(run, data.docs.as_view(), &*kernel)?);
        }
        writeln!(output, "\n\n{}", DisplayWrapper(&*results))?;
        Ok(results)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "- Element Type: {}", <T as AsDataType>::DATA_TYPE)
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
        CheckResult::compare(tolerance, before, after)
    }
}

impl CheckResult {
    fn compare(
        tolerance: &MultiVectorTolerance,
        before: &[RunResult],
        after: &[RunResult],
    ) -> anyhow::Result<PassFail<Self, Self>> {
        anyhow::ensure!(
            before.len() == after.len(),
            "before has {} runs but after has {}",
            before.len(),
            after.len(),
        );

        let mut passed = true;
        let checks = std::iter::zip(before, after)
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
            PassFail::Pass(Self { checks })
        } else {
            PassFail::Fail(Self { checks })
        })
    }
}

#[derive(Debug)]
pub(super) struct MinMax8Kernel;

impl Benchmark for MinMax8Kernel {
    type Input = MultiVectorOp;
    type Output = Vec<RunResult>;

    fn try_match(&self, from: &MultiVectorOp, context: &MatchContext) -> Score {
        let mut score = context.success(0);
        if from.element_type != ElementType::MinMax8 {
            score.fail(DATA_TYPE_MISMATCH, &"expected minmax8 element type");
        }
        let isa: MaxSimIsa = from.isa.into();
        if isa == MaxSimIsa::Reference {
            score.fail(1, &"MinMax8 has no reference kernel; use scalar");
        } else if !isa.is_available() {
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
        for run in &input.runs {
            let data = MinMax8Data::new(run)?;
            let kernel = build_minmax_max_sim(input.isa.into(), data.queries.as_view(), BoxErase)?;
            let mut scores = vec![0.0; run.num_query_vectors.get()];
            kernel.compute_max_sim(data.docs.as_view(), &mut scores)?;
            std::hint::black_box(&mut scores);
            results.push(run_with_compute(run, || {
                kernel.compute_max_sim(data.docs.as_view(), &mut scores)?;
                std::hint::black_box(&mut scores);
                Ok(())
            })?);
        }
        writeln!(output, "\n\n{}", DisplayWrapper(&*results))?;
        Ok(results)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "- Query: MinMax8; Document: MinMax4")
    }
}

impl Regression for MinMax8Kernel {
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
        CheckResult::compare(tolerance, before, after)
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Registration.
// ─────────────────────────────────────────────────────────────────────────

pub(super) fn register(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register_regression("multi-vector-op-f32", Kernel::<f32>::new())?;
    registry.register_regression("multi-vector-op-f16", Kernel::<half::f16>::new())?;
    registry.register_regression("multi-vector-op-i8", Kernel::<i8>::new())?;
    registry.register_regression("multi-vector-op-minmax8", MinMax8Kernel)?;
    Ok(())
}
