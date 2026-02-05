// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Benchmarking utilities for multi-vector distance computations.
//!
//! This module provides a diskann-benchmark-runner based benchmark framework for measuring
//! the performance of multi-vector similarity computations using both naive
//! and SIMD-accelerated approaches.

mod input;
mod runner;

use std::io::Write;

use diskann_benchmark_runner::{
    describeln,
    dispatcher::{self, DispatchRule, FailureScore, MatchScore},
    Any,
};

use crate::distance::{
    NaiveApproach, QueryTransposedWithTilingApproach, SgemmApproach, SimdApproach,
    TransposedApproach, TransposedWithTilingApproach,
};

pub use input::{Approach, MultiVectorInput, MultiVectorOp, Run};
use runner::{
    run_benchmark_with_approach, run_benchmark_with_query_transposed_approach,
    run_benchmark_with_sgemm_approach, run_benchmark_with_transposed_approach, DisplayWrapper,
};

////////////////
// Public API //
////////////////

/// Register multi-vector benchmarks.
pub fn register(dispatcher: &mut diskann_benchmark_runner::registry::Benchmarks) {
    dispatcher.register::<MultiVectorKernel<'static>>("multivec-op", run_benchmark);
}

//////////////
// Dispatch //
//////////////

/// Kernel for multi-vector benchmarks.
struct MultiVectorKernel<'a> {
    input: &'a MultiVectorOp,
}

impl<'a> MultiVectorKernel<'a> {
    fn new(input: &'a MultiVectorOp) -> Self {
        Self { input }
    }
}

impl dispatcher::Map for MultiVectorKernel<'static> {
    type Type<'a> = MultiVectorKernel<'a>;
}

impl<'a> DispatchRule<&'a MultiVectorOp> for MultiVectorKernel<'a> {
    type Error = std::convert::Infallible;

    fn try_match(_from: &&'a MultiVectorOp) -> Result<MatchScore, FailureScore> {
        Ok(MatchScore(0))
    }

    fn convert(from: &'a MultiVectorOp) -> Result<Self, Self::Error> {
        Ok(Self::new(from))
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        from: Option<&&'a MultiVectorOp>,
    ) -> std::fmt::Result {
        match from {
            None => describeln!(f, "- Multi-vector benchmark (naive or simd)"),
            Some(input) => describeln!(f, "- Approach: {}", input.approach),
        }
    }
}

impl<'a> DispatchRule<&'a diskann_benchmark_runner::Any> for MultiVectorKernel<'a> {
    type Error = anyhow::Error;

    fn try_match(from: &&'a diskann_benchmark_runner::Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<MultiVectorOp, Self>()
    }

    fn convert(from: &'a diskann_benchmark_runner::Any) -> Result<Self, Self::Error> {
        from.convert::<MultiVectorOp, Self>()
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        from: Option<&&'a diskann_benchmark_runner::Any>,
    ) -> std::fmt::Result {
        Any::description::<MultiVectorOp, Self>(f, from, MultiVectorOp::tag())
    }
}

///////////////
// Benchmark //
///////////////

fn run_benchmark(
    kernel: MultiVectorKernel<'_>,
    _: diskann_benchmark_runner::Checkpoint<'_>,
    mut output: &mut dyn diskann_benchmark_runner::Output,
) -> Result<serde_json::Value, anyhow::Error> {
    writeln!(output, "{}", kernel.input)?;

    let results = match kernel.input.approach {
        Approach::Naive => {
            writeln!(output, "Running with Naive (scalar) approach...\n")?;
            run_benchmark_with_approach::<NaiveApproach>(kernel.input, kernel.input.verify, output)?
        }
        Approach::Simd => {
            writeln!(output, "Running with SIMD (vectorized) approach...\n")?;
            run_benchmark_with_approach::<SimdApproach>(kernel.input, kernel.input.verify, output)?
        }
        Approach::TransposedSimd => {
            writeln!(output, "Running with Transposed SIMD approach...\n")?;
            run_benchmark_with_transposed_approach::<TransposedApproach>(
                kernel.input,
                kernel.input.verify,
                output,
            )?
        }
        Approach::TransposedWithTiling => {
            writeln!(
                output,
                "Running with Transposed SIMD + Tiling approach...\n"
            )?;
            run_benchmark_with_transposed_approach::<TransposedWithTilingApproach>(
                kernel.input,
                kernel.input.verify,
                output,
            )?
        }
        Approach::QueryTransposedWithTiling => {
            writeln!(
                output,
                "Running with Query-Transposed SIMD + Tiling approach...\n"
            )?;
            run_benchmark_with_query_transposed_approach::<QueryTransposedWithTilingApproach>(
                kernel.input,
                kernel.input.verify,
                output,
            )?
        }
        Approach::Sgemm => {
            writeln!(
                output,
                "Running with SGEMM (BLAS matrix multiplication) approach...\n"
            )?;
            run_benchmark_with_sgemm_approach::<SgemmApproach>(
                kernel.input,
                kernel.input.verify,
                output,
            )?
        }
    };

    writeln!(output, "\n{}", DisplayWrapper(&*results))?;
    Ok(serde_json::to_value(results)?)
}
