/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::registry;

// submodules
mod dim;
mod typed;

pub(crate) use typed::TypeInput;

/////////
// API //
/////////

pub fn register_benchmarks(
    registry: &mut registry::Registry,
) -> Result<(), registry::RegistryError> {
    registry.register_regression("type-bench-f32", typed::TypeBench::<f32>::new())?;
    registry.register_regression("type-bench-i8", typed::TypeBench::<i8>::new())?;
    registry.register_regression(
        "exact-type-bench-f32-1000",
        typed::ExactTypeBench::<f32, 1000>::new(),
    )?;

    registry.register("simple-bench", dim::SimpleBench)?;
    registry.register_regression("dim-bench", dim::DimBench)?;
    Ok(())
}
