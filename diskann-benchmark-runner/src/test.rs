/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use serde::{Deserialize, Serialize};

use crate::{
    dispatcher::{self, DispatchRule, FailureScore, MatchScore},
    registry,
    utils::datatype::{DataType, Type},
    Any, CheckDeserialization, Checker, Checkpoint, Input, Output,
};

/////////
// API //
/////////

pub fn register_inputs(inputs: &mut registry::Inputs) -> anyhow::Result<()> {
    inputs.register(AsTypeInput)?;
    inputs.register(AsDimInput)?;
    Ok(())
}

pub fn register_benchmarks(benchmarks: &mut registry::Benchmarks) {
    benchmarks
        .register::<TypeBench<'static, f32>>("type-bench-f32", TypeBench::<'static, f32>::run);
    benchmarks.register::<TypeBench<'static, i8>>("type-bench-i8", TypeBench::<'static, i8>::run);

    benchmarks.register::<DimBench>("dim-bench", DimBench::run);
}

////////////
// Inputs //
////////////

#[derive(Debug)]
struct AsTypeInput;

impl Input for AsTypeInput {
    fn tag(&self) -> &'static str {
        TypeInput::tag()
    }

    fn try_deserialize(
        &self,
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(TypeInput::deserialize(serialized)?)
    }

    fn example(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(TypeInput::new(
            DataType::Float32,
            128,
            false,
        ))?)
    }
}

#[derive(Debug)]
struct AsDimInput;

impl Input for AsDimInput {
    fn tag(&self) -> &'static str {
        "test-input-dim"
    }

    fn try_deserialize(
        &self,
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(DimInput::deserialize(serialized)?)
    }

    fn example(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(DimInput::new(Some(128)))?)
    }
}

/////////////////////////
// Deserialized Inputs //
/////////////////////////

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

    const fn tag() -> &'static str {
        "test-input-types"
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

impl CheckDeserialization for DimInput {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> anyhow::Result<()> {
        Ok(())
    }
}

////////////////
// Benchmarks //
////////////////

#[derive(Debug)]
struct TypeBench<'a, T> {
    input: &'a TypeInput,
    _type: Type<T>,
}

impl<T> TypeBench<'static, T> {
    fn run(
        this: TypeBench<'_, T>,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<serde_json::Value> {
        write!(output, "hello: {}", this.input.data_type.as_str())?;
        checkpoint.checkpoint(this.input.data_type.as_str())?;
        Ok(serde_json::Value::String(
            this.input.data_type.as_str().into(),
        ))
    }
}

impl<T> dispatcher::Map for TypeBench<'static, T>
where
    T: 'static,
{
    type Type<'a> = TypeBench<'a, T>;
}

impl<'a, T> DispatchRule<&'a TypeInput> for TypeBench<'a, T>
where
    Type<T>: DispatchRule<DataType, Error: std::error::Error + Send + Sync + 'static>,
{
    type Error = anyhow::Error;

    fn try_match(from: &&'a TypeInput) -> Result<MatchScore, FailureScore> {
        Type::<T>::try_match(&from.data_type)
    }
    fn convert(from: &'a TypeInput) -> Result<Self, Self::Error> {
        Ok(Self {
            input: from,
            _type: Type::<T>::convert(from.data_type)?,
        })
    }
    fn description(
        f: &mut std::fmt::Formatter<'_>,
        from: Option<&&'a TypeInput>,
    ) -> std::fmt::Result {
        match from {
            Some(v) => Type::<T>::description(f, Some(&v.data_type)),
            None => Type::<T>::description(f, None::<&DataType>),
        }
    }
}

impl<'a, T> DispatchRule<&'a Any> for TypeBench<'a, T>
where
    Self: DispatchRule<&'a TypeInput, Error = anyhow::Error>,
{
    type Error = anyhow::Error;

    fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<TypeInput, Self>()
    }
    fn convert(from: &'a Any) -> Result<Self, Self::Error> {
        from.convert::<TypeInput, Self>()
    }
    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&&'a Any>) -> std::fmt::Result {
        Any::description::<TypeInput, Self>(f, from, (AsTypeInput).tag())
    }
}

#[derive(Debug)]
struct DimBench {
    dim: Option<usize>,
}

impl DimBench {
    fn run(
        self,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<serde_json::Value> {
        write!(output, "dim bench: {:?}", self.dim)?;
        Ok(serde_json::Value::from(self.dim.unwrap_or(usize::MAX)))
    }
}

crate::self_map!(DimBench);

impl DispatchRule<&DimInput> for DimBench {
    type Error = std::convert::Infallible;

    fn try_match(_: &&DimInput) -> Result<MatchScore, FailureScore> {
        Ok(MatchScore(0))
    }
    fn convert(from: &DimInput) -> Result<Self, Self::Error> {
        Ok(Self { dim: from.dim })
    }
    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&&DimInput>) -> std::fmt::Result {
        if from.is_some() {
            write!(f, "perfect match")
        } else {
            write!(f, "matches all")
        }
    }
}

impl DispatchRule<&Any> for DimBench {
    type Error = anyhow::Error;

    fn try_match(from: &&Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<DimInput, Self>()
    }
    fn convert(from: &Any) -> Result<Self, Self::Error> {
        from.convert::<DimInput, Self>()
    }
    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&&Any>) -> std::fmt::Result {
        Any::description::<DimInput, Self>(f, from, (AsDimInput).tag())
    }
}
