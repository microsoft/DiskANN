/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Gated Tests
//!
//! Each gated benchmark registered by [`super::register_benchmarks`] has a "real" counterpart
//! here. When the controlling feature set is enabled in the [`super::TestConfig`], the real
//! benchmark (and its input, if it too was gated) is registered instead of the gated stub. This
//! mirrors how a downstream crate uses `cfg_if!` to swap a gated placeholder for the compiled
//! benchmark once the relevant Cargo feature is turned on.

use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::{
    benchmark::MatchContext, benchmark::Score, Benchmark, Checker, Checkpoint, Input, Output,
};

use super::{dim::DimInput, typed::TypeInput};

/////////////////////
// Partially-gated //
/////////////////////

// The real counterpart of the `gated-bench` placeholder. Its input (`TypeInput`) is always
// compiled, so only the benchmark itself is toggled.
#[derive(Debug)]
pub(super) struct GatedBench;

impl Benchmark for GatedBench {
    type Input = TypeInput;
    type Output = String;

    fn try_match(&self, _input: &TypeInput, context: &MatchContext) -> Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("A gated benchmark")
    }

    fn run(
        &self,
        input: &TypeInput,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        write!(output, "gated bench: {:?}", input)?;
        Ok(format!("{:?}", input))
    }
}

// The real counterpart of the `another-gated-bench` placeholder, gated behind a conjunction of
// features.
#[derive(Debug)]
pub(super) struct AnotherGatedBench;

impl Benchmark for AnotherGatedBench {
    type Input = DimInput;
    type Output = usize;

    fn try_match(&self, _input: &DimInput, context: &MatchContext) -> Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Another gated benchmark")
    }

    fn run(
        &self,
        input: &DimInput,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        write!(output, "another gated bench: {:?}", input)?;
        Ok(0)
    }
}

//////////////////////////////////////////////////
// Partially Gated with Input Always Registered //
//////////////////////////////////////////////////

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct SampleInput {
    value: String,
}

impl Input for SampleInput {
    type Raw = Self;

    fn tag() -> &'static str {
        "sample-input"
    }

    fn from_raw(raw: Self::Raw, _checker: &mut Checker) -> anyhow::Result<Self> {
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self::Raw {
        Self {
            value: "hello world".into(),
        }
    }
}

#[derive(Debug)]
pub(super) struct GatedWithIndependentInput;

impl Benchmark for GatedWithIndependentInput {
    type Input = SampleInput;
    type Output = String;

    fn try_match(&self, _input: &SampleInput, context: &MatchContext) -> Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("A gated benchmark that always registers its input")
    }

    fn run(
        &self,
        input: &SampleInput,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        write!(output, "partially-gated bench: {:?}", input)?;
        Ok(input.value.clone())
    }
}

/////////////////////////////////////
// Fully-gated benchmark (real)     //
/////////////////////////////////////

// The input backing the fully-gated benchmark. This is only compiled and registered when the
// controlling feature set is enabled, standing in for an input whose validation would otherwise
// pull in a heavy optional dependency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct PhantomInput {
    value: usize,
}

impl Input for PhantomInput {
    type Raw = Self;

    fn tag() -> &'static str {
        "phantom-input"
    }

    fn from_raw(raw: Self::Raw, _checker: &mut Checker) -> anyhow::Result<Self> {
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self::Raw {
        Self { value: 42 }
    }
}

// The real counterpart of the `fully-gated-bench` placeholder. Both this benchmark and its
// `PhantomInput` are toggled together.
#[derive(Debug)]
pub(super) struct FullyGatedBench;

impl Benchmark for FullyGatedBench {
    type Input = PhantomInput;
    type Output = usize;

    fn try_match(&self, _input: &PhantomInput, context: &MatchContext) -> Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("A fully gated benchmark")
    }

    fn run(
        &self,
        input: &PhantomInput,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        write!(output, "fully gated bench: {:?}", input)?;
        Ok(input.value)
    }
}
