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
    benchmarks.register_regression::<typed::TypeBench<f32>>("type-bench-f32");
    benchmarks.register_regression::<typed::TypeBench<i8>>("type-bench-i8");
    benchmarks.register_regression::<typed::ExactTypeBench<f32, 1000>>("exact-type-bench-f32-1000");

    benchmarks.register::<dim::SimpleBench>("simple-bench");
    benchmarks.register_regression::<dim::DimBench>("dim-bench");
}
