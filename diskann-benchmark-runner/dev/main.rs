/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{output, App, Registry};

fn main() -> anyhow::Result<()> {
    // Parse the command line options.
    let app = App::parse();

    let mut registry = Registry::new();
    diskann_benchmark_runner::test::register_benchmarks(&mut registry)?;

    app.run(&registry, &mut output::default())
}
