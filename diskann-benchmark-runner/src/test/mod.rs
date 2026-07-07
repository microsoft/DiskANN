/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{Features, registry};

mod dim;
mod typed;

pub(crate) use typed::TypeInput;

//////////////////
// Registration //
//////////////////

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

    // Gated
    registry.register_partially_gated::<TypeInput>(
        "gated-bench",
        Features::new("gated-feature"),
        "A gated benchmark",
    )?;

    registry.register_partially_gated::<dim::DimInput>(
        "another-gated-bench",
        Features::all(["gated-feature", "super-special-gated-feature"]),
        "Another gated benchmark",
    )?;

    registry.register_gated(
        "phantom-input",
        "fully-gated-bench",
        Features::any(["fully-gated-feature", "something-else"]),
        "A fully gated benchmark",
    )?;

    Ok(())
}
