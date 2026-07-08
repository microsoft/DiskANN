/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::{
    benchmark::{MatchContext, PassFail, Regression, Score},
    utils::datatype::{AsDataType, DataType},
    Benchmark, Checker, Checkpoint, Input, Output,
};

///////////
// Input //
///////////

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub(crate) struct TypeInput {
    pub(super) data_type: DataType,
    pub(super) dim: usize,
    error_when_checked: bool,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct TypeInputRaw {
    data_type: DataType,
    dim: usize,
    // Should we return an error when deserializing?
    error_when_checked: bool,
}

impl TypeInput {
    pub(crate) fn new(data_type: DataType, dim: usize) -> Self {
        Self {
            data_type,
            dim,
            error_when_checked: false,
        }
    }

    fn run(&self) -> &'static str {
        self.data_type.as_str()
    }
}

impl Input for TypeInput {
    type Raw = TypeInputRaw;

    fn tag() -> &'static str {
        "test-input-types"
    }

    fn from_raw(raw: Self::Raw, _checker: &mut Checker) -> anyhow::Result<Self> {
        if raw.error_when_checked {
            Err(anyhow::anyhow!("test input erroring when checked"))
        } else {
            Ok(Self::new(raw.data_type, raw.dim))
        }
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self::Raw {
        TypeInputRaw {
            data_type: DataType::Float32,
            dim: 128,
            error_when_checked: false,
        }
    }
}

///////////////
// Tolerance //
///////////////

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Tolerance {
    // Should we return an error when `from_raw` is called?
    pub(super) error_when_checked: bool,
}

impl Input for Tolerance {
    type Raw = Self;

    fn tag() -> &'static str {
        "test-input-types-tolerance"
    }

    fn from_raw(raw: Self::Raw, _checker: &mut Checker) -> anyhow::Result<Self> {
        if raw.error_when_checked {
            Err(anyhow::anyhow!("test input erroring when checked"))
        } else {
            Ok(raw)
        }
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self::Raw {
        Self {
            error_when_checked: false,
        }
    }
}

////////////////
// Benchmarks //
////////////////

#[derive(Debug)]
pub(super) struct TypeBench<T>(std::marker::PhantomData<T>);

impl<T> TypeBench<T> {
    pub(super) fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T> Benchmark for TypeBench<T>
where
    T: AsDataType,
{
    type Input = TypeInput;
    type Output = String;

    fn try_match(&self, input: &TypeInput, context: &MatchContext) -> Score {
        // Try to match based on data type.
        // Add a small penalty so `ExactTypeBench` can be more specific if it hits.
        if T::is_match(input.data_type) {
            context.success(10)
        } else {
            context.fail(1000, &T::describe(input.data_type))
        }
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", T::DATA_TYPE)
    }

    fn run(
        &self,
        input: &TypeInput,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        let result = input.run().to_string();
        write!(output, "hello: {}", result)?;
        checkpoint.checkpoint(&result)?;
        Ok(result)
    }
}

impl<T> Regression for TypeBench<T>
where
    T: AsDataType,
{
    type Tolerances = Tolerance;
    type Pass = DataType;
    type Fail = DataType;

    fn check(
        &self,
        _tolerance: &Tolerance,
        input: &TypeInput,
        before: &String,
        after: &String,
    ) -> anyhow::Result<PassFail<Self::Pass, Self::Fail>> {
        // This check here mainly serves to verify that the before and after results were
        // propagated correctly.
        //
        // Really, this is a unit test masquerading behind an integration test.
        let expected = input.run();
        assert_eq!(*before, expected);
        assert_eq!(*after, expected);

        Ok(PassFail::Pass(input.data_type))
    }
}

#[derive(Debug)]
pub(super) struct ExactTypeBench<T, const N: usize>(std::marker::PhantomData<T>);

impl<T, const N: usize> ExactTypeBench<T, N> {
    pub(super) fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T, const N: usize> Benchmark for ExactTypeBench<T, N>
where
    T: AsDataType,
{
    type Input = TypeInput;
    type Output = String;

    fn try_match(&self, input: &TypeInput, context: &MatchContext) -> Score {
        let mut score = if T::is_match(input.data_type) {
            context.success(0)
        } else {
            context.fail(1000, &T::describe(input.data_type))
        };

        if input.dim != N {
            score.fail(
                10,
                &format_args!("expected dim={}, but found dim={}", N, input.dim),
            );
        }

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}, dim={}", T::DATA_TYPE, N)
    }

    fn run(
        &self,
        input: &TypeInput,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        let s = format!("hello<{}>: {}", N, input.data_type.as_str());
        write!(output, "{}", s)?;
        checkpoint.checkpoint(&s)?;
        Ok(s)
    }
}

impl<T, const N: usize> Regression for ExactTypeBench<T, N>
where
    T: AsDataType,
{
    type Tolerances = Tolerance;
    type Pass = String;
    type Fail = String;

    fn check(
        &self,
        _tolerance: &Tolerance,
        input: &TypeInput,
        before: &String,
        after: &String,
    ) -> anyhow::Result<PassFail<Self::Pass, Self::Fail>> {
        // Verify correct dispatch: ExactTypeBench produces a different output format than
        // TypeBench. If the wrong benchmark was dispatched, the assertion below will catch
        // it.
        let expected = format!("hello<{}>: {}", N, input.data_type.as_str());
        assert_eq!(*before, expected);
        assert_eq!(*after, expected);

        Ok(PassFail::Pass(format!(
            "exact match dim={} type={}",
            N, input.data_type
        )))
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::benchmark::TestScore;

    #[test]
    fn test_try_match() {
        let bench = ExactTypeBench::<f32, 1000>::new();
        match MatchContext::test(&bench, &TypeInput::new(DataType::Float32, 1000)) {
            TestScore::Success(v) => assert_eq!(v, 0),
            _ => panic!("assumed success"),
        };

        match MatchContext::test(&bench, &TypeInput::new(DataType::Float16, 1000)) {
            TestScore::Failure { score, reasons } => {
                assert_eq!(score, 1000);
                assert_eq!(reasons.unwrap().len(), 1);
            }
            _ => panic!("assumed failure"),
        };
    }
}
