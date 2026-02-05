/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) mod async_;
pub(crate) mod multi;
pub(crate) mod disk;
pub(crate) mod exhaustive;
pub(crate) mod filters;
pub(crate) mod save_and_load;

pub(crate) fn register_inputs(
    registry: &mut diskann_benchmark_runner::registry::Inputs,
) -> anyhow::Result<()> {
    async_::register_inputs(registry)?;
    exhaustive::register_inputs(registry)?;
    disk::register_inputs(registry)?;
    filters::register_inputs(registry)?;
    multi::register_inputs(registry)?;
    Ok(())
}

/// A helper type for implementing `diskann_benchmark_runner::Input` for benchmark types.
pub(crate) struct Input<T> {
    _type: std::marker::PhantomData<T>,
}

impl<T> Input<T> {
    pub(crate) fn new() -> Self {
        Self {
            _type: std::marker::PhantomData,
        }
    }
}

/// Construct an example input of type `Self`.
pub(crate) trait Example {
    fn example() -> Self;
}

macro_rules! as_input {
    ($T:ty) => {
        impl diskann_benchmark_runner::Input for $crate::inputs::Input<$T> {
            fn tag(&self) -> &'static str {
                <$T>::tag()
            }

            fn try_deserialize(
                &self,
                serialized: &serde_json::Value,
                checker: &mut diskann_benchmark_runner::Checker,
            ) -> anyhow::Result<diskann_benchmark_runner::Any> {
                checker.any(<$T as serde::Deserialize>::deserialize(serialized)?)
            }

            fn example(&self) -> anyhow::Result<serde_json::Value> {
                Ok(serde_json::to_value(
                    <$T as $crate::inputs::Example>::example(),
                )?)
            }
        }
    };
}

use as_input;
