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

    use diskann_benchmark_runner::{app::Commands, output::Memory};
    use diskann_utils::test_data_root;
    use serde::{Serialize, Deserialize};

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

    fn save_to_file<T>(path: &std::path::Path, value: &T)
    where
        T: Serialize + ?Sized,
    {
        if path.exists() {
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
                .expect("key \"search-directories\" should exist");
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

    fn prepend(input: &Path, output: &Path) {
        let mut v = value_from_file(input);
        prefix_search_directories(&mut v, &test_data_root());
        save_to_file(output, &v);
    }

    // Drive the named example through the full runner flow: load the JSON input file,
    // dispatch through the registry, run the benchmark, and write results to disk.
    fn run_example(name: &str) {
        let input_file = example_directory().join(name);
        assert!(input_file.exists(), "missing example file: {input_file:?}");

        let tempdir = tempfile::tempdir().unwrap();
        let modified_input_file = tempdir.path().join("input.json");
        let output_file = tempdir.path().join("output.json");

        prepend(&input_file, &modified_input_file);

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

    #[test]
    fn store_stress_integration() {
        run_example("store-stress-test.json");
    }

    #[test]
    fn graph_index() {
        run_example("integration.json");
    }
}
