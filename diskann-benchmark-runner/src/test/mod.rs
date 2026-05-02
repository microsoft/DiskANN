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

pub fn register_inputs(inputs: &mut registry::Inputs) -> anyhow::Result<()> {
    inputs.register::<typed::TypeInput>()?;
    inputs.register::<dim::DimInput>()?;
    Ok(())
}

pub fn register_benchmarks(benchmarks: &mut registry::Benchmarks) {
    benchmarks.register_regression("type-bench-f32", typed::TypeBench::<f32>::new());
    benchmarks.register_regression("type-bench-i8", typed::TypeBench::<i8>::new());
    benchmarks.register_regression(
        "exact-type-bench-f32-1000",
        typed::ExactTypeBench::<f32, 1000>::new(),
    );

    benchmarks.register("simple-bench", dim::SimpleBench);
    benchmarks.register_regression("dim-bench", dim::DimBench);
}
