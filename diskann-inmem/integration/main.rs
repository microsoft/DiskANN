/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod index;
mod store;
mod support;

use diskann_benchmark_runner::{App, Registry, output};

/// Build a [`Registry`] with all integration benchmarks registered.
fn registry() -> anyhow::Result<Registry> {
    let mut registry = Registry::new();
    registry.register("store-stress", store::StoreStress)?;
    index::register(&mut registry)?;
    Ok(registry)
}

fn main() -> anyhow::Result<()> {
    let app = App::parse();
    app.run(&registry()?, &mut output::default())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_benchmark_runner::{app::Commands, output::Memory};

    // The directory containing the committed example input files.
    fn example_directory() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("integration")
            .join("example")
    }

    // Drive the named example through the full runner flow: load the JSON input file,
    // dispatch through the registry, run the benchmark, and write results to disk.
    fn run_example(name: &str) {
        let input_file = example_directory().join(name);
        assert!(input_file.exists(), "missing example file: {input_file:?}");

        let tempdir = tempfile::tempdir().unwrap();
        let output_file = tempdir.path().join("output.json");

        let command = Commands::Run {
            input_file,
            output_file: output_file.clone(),
            dry_run: false,
            // Unit tests are a debug build; bypass the runner's debug-mode guard.
            allow_debug: true,
        };
        let app = App::from_commands(command);

        let mut output = Memory::new();
        // A benchmark error (e.g. an invariant violation) propagates here and fails the test.
        app.run(&registry().unwrap(), &mut output).unwrap();

        assert!(output_file.exists(), "results file was not written");
    }

    #[test]
    fn store_stress_integration() {
        run_example("store-stress-test.json");
    }
}
