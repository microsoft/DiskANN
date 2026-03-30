// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Multi-vector benchmark binary.
//!
//! This binary provides a CLI for benchmarking multi-vector distance computations
//! using the diskann-benchmark-runner framework.
//!
//! # Usage
//!
//! ```bash
//! # List available inputs
//! multivec-bench inputs
//!
//! # List available benchmarks
//! multivec-bench benchmarks
//!
//! # Generate example input JSON
//! multivec-bench skeleton
//!
//! # Run benchmarks
//! multivec-bench run --input examples/bench.json --output results.json
//! ```

use diskann_benchmark_runner::{output, registry, App, Output};
use experimental_multi_vector_bench::bench::{register, MultiVectorInput};

pub fn main() -> anyhow::Result<()> {
    let app = App::parse();
    main_inner(&app, &mut output::default())
}

fn main_inner(app: &App, output: &mut dyn Output) -> anyhow::Result<()> {
    // Register inputs
    let mut inputs = registry::Inputs::new();
    inputs.register(MultiVectorInput)?;

    // Register benchmarks
    let mut benchmarks = registry::Benchmarks::new();
    register(&mut benchmarks);

    // Run the application
    app.run(&inputs, &benchmarks, output)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::{Path, PathBuf};

    use diskann_benchmark_runner::app::Commands;

    fn run_integration_test(input_file: &Path, output_file: &Path) {
        let commands = Commands::Run {
            input_file: input_file.to_str().unwrap().into(),
            output_file: output_file.to_str().unwrap().into(),
            dry_run: false,
        };

        let app = App::from_commands(commands);

        let mut output = output::Memory::new();
        main_inner(&app, &mut output).unwrap();
        println!(
            "output = {}",
            String::from_utf8(output.into_inner()).unwrap()
        );

        assert!(output_file.exists());
    }

    #[test]
    fn integration_test() {
        let input_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("bench.json");

        let tempdir = tempfile::tempdir().unwrap();
        let output_path = tempdir.path().join("output.json");

        run_integration_test(&input_path, &output_path);
    }
}
