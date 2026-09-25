/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Development CLI for exercising the benchmark runner with its test registry.

use diskann_benchmark_runner::reflect;

fn main() -> anyhow::Result<()> {
    println!("{}", reflect::reflect::<reflect::Test>().render());

    println!("{}", reflect::reflect::<reflect::Test2>().render());

    println!("{}", reflect::reflect::<reflect::Metric>().render());

    Ok(())
}
