/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::{
    benchmark::{MatchContext, PassFail, Regression, Score},
    Benchmark, Checker, Checkpoint, Input, Output,
};

///////////
// Input //
///////////

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct DimInput {
    dim: Option<usize>,
}

impl DimInput {
    fn new(dim: Option<usize>) -> Self {
        Self { dim }
    }

    fn run(&self) -> usize {
        self.dim.unwrap_or(usize::MAX)
    }
}

impl Input for DimInput {
    type Raw = Self;

    fn tag() -> &'static str {
        "test-input-dim"
    }

    fn from_raw(raw: Self::Raw, _checker: &mut Checker) -> anyhow::Result<Self> {
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self::Raw {
        DimInput::new(Some(128))
    }
}

///////////////
// Tolerance //
///////////////

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Tolerance {
    succeed: bool,
    error_in_check: bool,
}

impl Input for Tolerance {
    type Raw = Self;

    fn tag() -> &'static str {
        "test-input-dim-tolerance"
    }

    fn from_raw(raw: Self::Raw, _checker: &mut Checker) -> anyhow::Result<Self> {
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self::Raw {
        Self {
            succeed: true,
            error_in_check: false,
        }
    }
}

////////////////
// Benchmarks //
////////////////

// A simple benchmark that doesn't implement [`Regression`] and only matches `None` variants
// of `DimInput`.
#[derive(Debug)]
pub(super) struct SimpleBench;

impl Benchmark for SimpleBench {
    type Input = DimInput;
    type Output = usize;

    fn try_match(&self, input: &DimInput, context: &MatchContext) -> Score {
        if input.dim.is_none() {
            context.success(0)
        } else {
            context.fail(1000, &"expected dim=None")
        }
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("dim=None only")
    }

    fn run(
        &self,
        input: &DimInput,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        write!(output, "simple bench: {:?}", input.dim)?;
        Ok(input.run())
    }
}

// A more general version of `SimpleBench` that matches all flavors of `dim`.
#[derive(Debug)]
pub(super) struct DimBench;

impl Benchmark for DimBench {
    type Input = DimInput;
    type Output = usize;

    fn try_match(&self, _input: &DimInput, context: &MatchContext) -> Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("matches all")
    }

    fn run(
        &self,
        input: &DimInput,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        write!(output, "dim bench: {:?}", input.dim)?;
        Ok(input.run())
    }
}

impl Regression for DimBench {
    type Tolerances = Tolerance;
    type Pass = &'static str;
    type Fail = &'static str;

    fn check(
        &self,
        tolerance: &Tolerance,
        input: &DimInput,
        before: &usize,
        after: &usize,
    ) -> anyhow::Result<PassFail<Self::Pass, Self::Fail>> {
        let Tolerance {
            succeed,
            error_in_check,
        } = tolerance;
        if *error_in_check {
            anyhow::bail!("simulated check error");
        }

        // This check here mainly serves to verify that the before and after results were
        // propagated correctly.
        //
        // Really, this is a unit test masquerading behind an integration test.
        let expected = input.run();
        assert_eq!(*before, expected);
        assert_eq!(*after, expected);

        // The success or failure of the benchmark depends on the configuration of the
        // tolerance.
        if *succeed {
            Ok(PassFail::Pass("we did it!"))
        } else {
            Ok(PassFail::Fail("we didn't do it!"))
        }
    }
}
