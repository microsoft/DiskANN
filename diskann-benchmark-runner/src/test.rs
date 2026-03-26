/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::{
    dispatcher::{Description, DispatchRule, FailureScore, MatchScore},
    registry,
    utils::datatype::{DataType, Type},
    Any, Benchmark, CheckDeserialization, Checker, Checkpoint, Input, Output,
};

/////////
// API //
/////////

pub fn register_inputs(inputs: &mut registry::Inputs) -> anyhow::Result<()> {
    inputs.register::<TypeInput>()?;
    inputs.register::<DimInput>()?;
    Ok(())
}

pub fn register_benchmarks(benchmarks: &mut registry::Benchmarks) {
    benchmarks.register::<TypeBench<f32>>("type-bench-f32");
    benchmarks.register::<TypeBench<i8>>("type-bench-i8");
    benchmarks.register::<ExactTypeBench<f32, 1000>>("exact-type-bench-f32-1000");
    benchmarks.register::<DimBench>("dim-bench");
}

////////////
// Inputs //
////////////

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct TypeInput {
    pub(crate) data_type: DataType,
    pub(crate) dim: usize,
    // Should we return an error when `check_deserialization` is called?
    pub(crate) error_when_checked: bool,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DimInput {
    pub(crate) dim: Option<usize>,
}

impl DimInput {
    pub(crate) fn new(dim: Option<usize>) -> Self {
        Self { dim }
    }
}

impl Input for DimInput {
    fn tag() -> &'static str {
        "test-input-dim"
    }

    fn try_deserialize(
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(DimInput::deserialize(serialized)?)
    }

    fn example() -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(DimInput::new(Some(128)))?)
    }
}

impl CheckDeserialization for DimInput {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> anyhow::Result<()> {
        Ok(())
    }
}

////////////////
// Benchmarks //
////////////////

#[derive(Debug)]
struct TypeBench<T>(std::marker::PhantomData<T>);

impl<T> Benchmark for TypeBench<T>
where
    T: 'static,
    Type<T>: DispatchRule<DataType, Error: std::error::Error + Send + Sync + 'static>,
{
    type Input = TypeInput;
    type Output = &'static str;

    fn try_match(input: &TypeInput) -> Result<MatchScore, FailureScore> {
        // Try to match based on data type.
        // Add a small penalty to `ExactTypeBench` can be more specific if it hits.
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
        write!(output, "hello: {}", input.data_type.as_str())?;
        checkpoint.checkpoint(input.data_type.as_str())?;
        Ok(input.data_type.as_str())
    }
}

#[derive(Debug)]
struct ExactTypeBench<T, const N: usize>(std::marker::PhantomData<T>);

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

#[derive(Debug)]
struct DimBench;

impl Benchmark for DimBench {
    type Input = DimInput;
    type Output = usize;

    fn try_match(_input: &DimInput) -> Result<MatchScore, FailureScore> {
        Ok(MatchScore(0))
    }

    fn description(f: &mut std::fmt::Formatter<'_>, input: Option<&DimInput>) -> std::fmt::Result {
        if input.is_some() {
            write!(f, "perfect match")
        } else {
            write!(f, "matches all")
        }
    }

    fn run(
        input: &DimInput,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        write!(output, "dim bench: {:?}", input.dim)?;
        Ok(input.dim.unwrap_or(usize::MAX))
    }
}
