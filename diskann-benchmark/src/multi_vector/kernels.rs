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
    utils::datatype::AsDataType,
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_quantization::minmax::build_minmax_max_sim;
use diskann_quantization::multi_vector::{build_max_sim, BoxErase, MaxSimElement, MaxSimIsa};
use rand::distr::{Distribution, StandardUniform};

use super::driver::{
    run_with_compute, run_with_kernel, CheckResult, Data, MinMax8Data, MultiVectorTolerance,
    RunResult,
};
use crate::inputs::multi_vector::{MultiVectorFormat, MultiVectorOp};
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

    fn try_match(&self, from: &MultiVectorOp, context: &MatchContext) -> Score {
        let mut score = context.success(0);
        if from.format != MultiVectorFormat::Dense {
            score.fail(1, &"expected dense vector format");
        }
        crate::utils::match_data_type::<T>(&mut score, from.element_type);
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

#[derive(Debug)]
pub(super) struct MinMax8Kernel;

impl Benchmark for MinMax8Kernel {
    type Input = MultiVectorOp;
    type Output = Vec<RunResult>;

    fn try_match(&self, from: &MultiVectorOp, context: &MatchContext) -> Score {
        let mut score = context.success(0);
        if from.format != MultiVectorFormat::MinMax8 {
            score.fail(1, &"expected minmax8 vector format");
        }
        crate::utils::match_data_type::<f32>(&mut score, from.element_type);
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
