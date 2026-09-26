/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Development CLI for exercising the benchmark runner with its test registry.

use diskann_benchmark_runner::{reflect, Reflection};

fn main() -> anyhow::Result<()> {
    // println!("{}", Reflection::new::<reflect::Test>().render());

    // println!("{}", Reflection::new::<reflect::Test2>().render());

    // println!("{}", Reflection::new::<reflect::Metric>().render());

    Ok(())
}
