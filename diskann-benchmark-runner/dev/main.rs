/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{app::App, output, registry};

fn main() -> anyhow::Result<()> {
    // Parse the command line options.
    let app = App::parse();

    let mut registry = registry::Benchmarks::new();
    diskann_benchmark_runner::test::register_benchmarks(&mut registry);

    app.run(&registry, &mut output::default())
}
