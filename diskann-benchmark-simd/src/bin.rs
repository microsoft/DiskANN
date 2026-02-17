/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{output, registry, App, Output};
use diskann_benchmark_simd::{register, SimdInput};

pub fn main() -> anyhow::Result<()> {
    // Create the pocket bench application.
    let app = App::parse();
    main_inner(&app, &mut output::default())
}

fn main_inner(app: &App, output: &mut dyn Output) -> anyhow::Result<()> {
    // Register inputs and benchmarks.
    let mut inputs = registry::Inputs::new();
    inputs.register(SimdInput)?;

    let mut benchmarks = registry::Benchmarks::new();
    register(&mut benchmarks);

    // Here we go!
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
        // Use architecture-specific test file
        // On macOS (and other non-x86_64 platforms), use test_mac.json with scalar architecture
        // On x86_64, use test.json with x86-64-v3 architecture
        #[cfg(target_arch = "x86_64")]
        let test_file = "test.json";
        #[cfg(not(target_arch = "x86_64"))]
        let test_file = "test_mac.json";

        let input_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join(test_file);

        let tempdir = tempfile::tempdir().unwrap();
        let output_path = tempdir.path().join("output.json");

        run_integration_test(&input_path, &output_path);
    }
}
