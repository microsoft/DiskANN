/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod object;
mod runner;
mod tests;

use object::{CheckCounters, Counters, Index, KnnSearch};

use diskann_benchmark_runner::{Registry, RegistryError};

pub(super) fn register(registry: &mut Registry) -> Result<(), RegistryError> {
    runner::register(registry)?;
    Ok(())
}
