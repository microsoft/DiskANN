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

    use std::path::Path;

    use diskann_benchmark_runner::{
        app::{Check, Commands},
        output::Memory,
    };
    use diskann_utils::test_data_root;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    // Environment variable used to regenerate committed regression baselines.
    const DISKANN_TEST_ENV: &str = "DISKANN_TEST";

    // Return `true` if `DISKANN_TEST=overwrite` is set, instructing regression tests to
    // overwrite their committed baselines instead of checking against them.
    //
    // If `DISKANN_TEST` is set to anything other than `overwrite`, panic.
    fn overwrite_baselines() -> bool {
        match std::env::var(DISKANN_TEST_ENV) {
            Ok(v) if v == "overwrite" => true,
            Ok(v) => {
                panic!("unknown value for {DISKANN_TEST_ENV}: \"{v}\". Expected \"overwrite\"")
            }
            Err(std::env::VarError::NotPresent) => false,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("value for {DISKANN_TEST_ENV} is not unicode")
            }
        }
    }

    // The directory containing the committed example input files.
    fn example_directory() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("integration")
            .join("jsons")
    }

    // TODO: add first class `diskann-benchmark-runner` support for this.
    fn load_from_file<T>(path: &std::path::Path) -> T
    where
        T: for<'a> Deserialize<'a>,
    {
        let file = std::fs::File::open(path).unwrap();
        let reader = std::io::BufReader::new(file);
        serde_json::from_reader(reader).unwrap()
    }

    fn value_from_file(path: &std::path::Path) -> serde_json::Value {
        load_from_file(path)
    }

    fn save_to_file<T>(path: &std::path::Path, value: &T, force: bool)
    where
        T: Serialize + ?Sized,
    {
        if path.exists() && !force {
            panic!("path {} already exists!", path.display());
        }
        let buffer = std::fs::File::create(path).unwrap();
        serde_json::to_writer_pretty(buffer, value).unwrap();
    }

    fn prefix_search_directories(raw: &mut serde_json::Value, root: &std::path::Path) {
        let key = "search_directories";
        if let serde_json::Value::Object(obj) = raw {
            let value = obj
                .get_mut(key)
                .expect("key \"search_directories\" should exist");
            if let serde_json::Value::Array(directories) = value {
                for value in directories.iter_mut() {
                    if let serde_json::Value::String(dir) = value {
                        *dir = root.join(&dir).to_str().unwrap().into();
                    }
                }
            } else {
                panic!("Expected an Array - got {}", raw);
            }
        } else {
            panic!("Expected an Object - got {}", raw);
        }
    }

    fn prepend(input: &Path, output: &Path, root: &Path) {
        let mut v = value_from_file(input);
        prefix_search_directories(&mut v, root);
        save_to_file(output, &v, false);
    }

    // Drive the named example through the full runner flow: load the JSON input file,
    // dispatch through the registry, run the benchmark, and write results to disk.
    fn run_example(name: &str) {
        let input_file = example_directory().join(name);
        assert!(input_file.exists(), "missing example file: {input_file:?}");

        let tempdir = tempfile::tempdir().unwrap();
        let modified_input_file = tempdir.path().join("input.json");
        let output_file = tempdir.path().join("output.json");

        prepend(&input_file, &modified_input_file, &test_data_root());

        let command = Commands::Run {
            input_file: modified_input_file,
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

    // Drive the named example through the runner, then run a regression check comparing the
    // freshly produced results against a committed baseline.
    //
    // By default this fails the test if the regression check reports a negative result. When
    // `DISKANN_TEST=overwrite` is set, the committed baseline is instead overwritten with the
    // freshly produced results (enabling future migrations) and no check is performed.
    fn run_regression_example(input_name: &str, tolerances_name: &str, baseline_name: &str) {
        let input_file = example_directory().join(input_name);
        let tolerances_file = example_directory().join(tolerances_name);
        let baseline_file = example_directory().join(baseline_name);
        assert!(input_file.exists(), "missing example file: {input_file:?}");
        assert!(
            tolerances_file.exists(),
            "missing tolerances file: {tolerances_file:?}"
        );

        let tempdir = tempfile::tempdir().unwrap();
        let modified_input_file = tempdir.path().join("input.json");
        let output_file = tempdir.path().join("output.json");

        prepend(&input_file, &modified_input_file, &test_data_root());

        // Run the benchmark to produce the "after" results.
        let command = Commands::Run {
            input_file: modified_input_file.clone(),
            output_file: output_file.clone(),
            dry_run: false,
            // Unit tests are a debug build; bypass the runner's debug-mode guard.
            allow_debug: true,
        };
        let mut output = Memory::new();
        App::from_commands(command)
            .run(&registry().unwrap(), &mut output)
            .unwrap();
        assert!(output_file.exists(), "results file was not written");

        // In overwrite mode, replace the committed baseline and skip the check.
        if overwrite_baselines() {
            // When over-writing, we need to scrub the file paths of the test directory.
            //
            // Otherwise, we end up with absolute paths in the baselines.
            let mut v = value_from_file(&output_file);
            scrub(&mut v, &test_data_root());
            save_to_file(&baseline_file, &v, true);

            return;
        }

        assert!(
            baseline_file.exists(),
            "missing baseline {baseline_file:?}; regenerate it with {DISKANN_TEST_ENV}=overwrite"
        );

        // Run the regression check. A negative result (or any error) propagates here and
        // fails the test.
        let command = Commands::Check(Check::Run {
            tolerances: tolerances_file,
            input_file: modified_input_file,
            before: baseline_file,
            after: output_file,
            output_file: None,
        });
        let mut output = Memory::new();

        if let Err(err) = App::from_commands(command).run(&registry().unwrap(), &mut output) {
            panic!(
                "Regression check failed:\n\n{}\n\n{}",
                err,
                String::from_utf8(output.into_inner()).unwrap()
            );
        }
    }

    fn scrub(value: &mut Value, root: &Path) {
        let mut values = vec![value];
        while let Some(value) = values.pop() {
            match value {
                Value::Null | Value::Bool(_) | Value::Number(_) => {}
                Value::String(s) => {
                    *s = diskann_benchmark_runner::ux::scrub_path(s.clone(), root, "");
                }
                Value::Array(v) => v.iter_mut().for_each(|v| values.push(v)),
                Value::Object(m) => m.values_mut().for_each(|v| values.push(v)),
            }
        }
    }

    #[test]
    fn store_stress_integration() {
        run_example("store-stress-test.json");
    }

    #[test]
    #[cfg(not(miri))]
    fn graph_index() {
        run_regression_example(
            "integration.json",
            "checks.json",
            "integration-baseline.json",
        );
    }
}
