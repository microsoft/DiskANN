/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::{
    benchmark::{PassFail, Regression},
    dispatcher::{Description, DispatchRule, FailureScore, MatchScore},
    utils::datatype::{DataType, Type},
    Any, Benchmark, CheckDeserialization, Checker, Checkpoint, Input, Output,
};

///////////
// Input //
///////////

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct TypeInput {
    pub(super) data_type: DataType,
    pub(super) dim: usize,
    // Should we return an error when `check_deserialization` is called?
    pub(super) error_when_checked: bool,
    // A flag to verify that [`CheckDeserialization`] has run.
    #[serde(skip)]
    pub(crate) checked: bool,
}

impl TypeInput {
    pub(crate) fn new(data_type: DataType, dim: usize, error_when_checked: bool) -> Self {
        Self {
            data_type,
            dim,
            error_when_checked,
            checked: false,
        }
    }

    fn run(&self) -> &'static str {
        self.data_type.as_str()
    }
}

impl Input for TypeInput {
    fn tag() -> &'static str {
        "test-input-types"
    }

    fn try_deserialize(
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(TypeInput::deserialize(serialized)?)
    }

    fn example() -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(TypeInput::new(
            DataType::Float32,
            128,
            false,
        ))?)
    }
}

impl CheckDeserialization for TypeInput {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> anyhow::Result<()> {
        if self.error_when_checked {
            Err(anyhow::anyhow!("test input erroring when checked"))
        } else {
            self.checked = true;
            Ok(())
        }
    }
}

///////////////
// Tolerance //
///////////////

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Tolerance {
    // Should we return an error when `check_deserialization` is called?
    pub(super) error_when_checked: bool,

    // A flag to verify that [`CheckDeserialization`] has run.
    #[serde(skip)]
    pub(crate) checked: bool,
}

impl Input for Tolerance {
    fn tag() -> &'static str {
        "test-input-types-tolerance"
    }

    fn try_deserialize(
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(Self::deserialize(serialized)?)
    }

    fn example() -> anyhow::Result<serde_json::Value> {
        let this = Self {
            error_when_checked: false,
            checked: false,
        };
        Ok(serde_json::to_value(this)?)
    }
}

impl CheckDeserialization for Tolerance {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> anyhow::Result<()> {
        if self.error_when_checked {
            Err(anyhow::anyhow!("test input erroring when checked"))
        } else {
            self.checked = true;
            Ok(())
        }
    }
}

////////////////
// Benchmarks //
////////////////

#[derive(Debug)]
pub(super) struct TypeBench<T>(std::marker::PhantomData<T>);

impl<T> Benchmark for TypeBench<T>
where
    T: 'static,
    Type<T>: DispatchRule<DataType, Error: std::error::Error + Send + Sync + 'static>,
{
    type Input = TypeInput;
    type Output = String;

    fn try_match(input: &TypeInput) -> Result<MatchScore, FailureScore> {
        // Try to match based on data type.
        // Add a small penalty so `ExactTypeBench` can be more specific if it hits.
        Type::<T>::try_match(&input.data_type).map(|m| MatchScore(m.0 + 10))
    }

    fn description(f: &mut std::fmt::Formatter<'_>, input: Option<&TypeInput>) -> std::fmt::Result {
        Type::<T>::description(f, input.map(|i| &i.data_type))
    }

    fn run(
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
    T: 'static,
    Type<T>: DispatchRule<DataType, Error: std::error::Error + Send + Sync + 'static>,
{
    type Tolerances = Tolerance;
    type Pass = DataType;
    type Fail = DataType;

    fn check(
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

impl<T, const N: usize> Benchmark for ExactTypeBench<T, N>
where
    T: 'static,
    Type<T>: DispatchRule<DataType, Error: std::error::Error + Send + Sync + 'static>,
{
    type Input = TypeInput;
    type Output = String;

    fn try_match(input: &TypeInput) -> Result<MatchScore, FailureScore> {
        if input.dim == N {
            Type::<T>::try_match(&input.data_type)
        } else {
            Err(FailureScore(1000))
        }
    }

    fn description(f: &mut std::fmt::Formatter<'_>, input: Option<&TypeInput>) -> std::fmt::Result {
        match input {
            None => {
                write!(f, "{}, dim={}", Description::<DataType, Type<T>>::new(), N)
            }
            Some(input) => {
                let type_result = Type::<T>::try_match_verbose(&input.data_type);
                let dim_ok = input.dim == N;
                match (type_result, dim_ok) {
                    (Ok(_), true) => write!(f, "successful match"),
                    (Err(err), true) => write!(f, "{}", err),
                    (Ok(_), false) => {
                        write!(f, "expected dim={}, but found dim={}", N, input.dim)
                    }
                    (Err(err), false) => {
                        write!(
                            f,
                            "{}; expected dim={}, but found dim={}",
                            err, N, input.dim
                        )
                    }
                }
            }
        }
    }

    fn run(
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
    T: 'static,
    Type<T>: DispatchRule<DataType, Error: std::error::Error + Send + Sync + 'static>,
{
    type Tolerances = Tolerance;
    type Pass = String;
    type Fail = String;

    fn check(
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
