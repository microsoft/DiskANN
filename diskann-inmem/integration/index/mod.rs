/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod factory;
mod object;
mod ops;
mod parse;
mod runner;
mod stats;

use diskann_benchmark_runner::{Registry, RegistryError};

pub(super) fn register(registry: &mut Registry) -> Result<(), RegistryError> {
    runner::register(registry)?;
    Ok(())
}
