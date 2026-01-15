/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{app::App, output, registry};

fn main() -> anyhow::Result<()> {
    // Parse the command line options.
    let app = App::parse();

    // Gather the test inputs and outputs.
    let mut inputs = registry::Inputs::new();
    diskann_benchmark_runner::test::register_inputs(&mut inputs)?;

    let mut benchmarks = registry::Benchmarks::new();
    diskann_benchmark_runner::test::register_benchmarks(&mut benchmarks);

    app.run(&inputs, &benchmarks, &mut output::default())
}
