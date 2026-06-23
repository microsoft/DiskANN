/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod index;
mod runner;
mod tests;

use index::Index;

use diskann_benchmark_runner::{Registry, RegistryError};

pub(super) fn register(registry: &mut Registry) -> Result<(), RegistryError> {
    runner::register(registry)?;
    Ok(())
}
