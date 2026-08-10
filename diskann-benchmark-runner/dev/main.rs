/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use clap::Parser;
use diskann_benchmark_runner::{output, test, App, Registry};

fn main() -> anyhow::Result<()> {
    // Parse the command line options.
    let cli = Cli::parse();

    let mut registry = Registry::new();
    let config = test::TestConfig::with_features(cli.features);
    test::register_benchmarks(&mut registry, &config)?;

    cli.app.run(&registry, &mut output::default())
}

/// Top level CLI.
///
/// In addition to the `App`, we accept a list of features for the `TestConfig`.
#[derive(Debug, clap::Parser)]
struct Cli {
    /// Emulate enabling various features for feature gated functionality.
    #[arg(long, value_delimiter = ',', num_args = 0..)]
    features: Vec<String>,

    /// The actual application.
    #[command(flatten)]
    app: App,
}
