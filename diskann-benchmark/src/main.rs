/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod backend;
mod inputs;
mod utils;

fn main() -> Result<(), anyhow::Error> {
    run_program(
        &diskann_benchmark_runner::App::parse(),
        &mut diskann_benchmark_runner::output::default(),
    )
}

fn run_program(
    app: &diskann_benchmark_runner::App,
    output: &mut dyn diskann_benchmark_runner::Output,
) -> anyhow::Result<()> {
    // Collect inputs.
    let mut inputs = diskann_benchmark_runner::registry::Inputs::new();
    inputs::register_inputs(&mut inputs)?;

    // Collect benchmarks.
    let mut benchmarks = diskann_benchmark_runner::registry::Benchmarks::new();
    backend::register_benchmarks(&mut benchmarks);

    app.run(&inputs, &benchmarks, output)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    use super::*;

    use diskann_benchmark_runner::{app::Commands, output::Memory, App};
    use diskann_providers::storage::FileStorageProvider;
    use diskann_tools::utils::{compute_ground_truth_from_datafiles, GraphDataF32Vector};
    use diskann_vector::distance::Metric;

    // Add these structs to deserialize the benchmark results
    #[derive(Debug, Deserialize)]
    struct BenchmarkResult {
        results: ResultsContainer,
    }

    #[derive(Debug, Deserialize)]
    struct ResultsContainer {
        search: SearchContainer,
    }

    #[derive(Debug, Deserialize)]
    struct SearchContainer {
        #[serde(rename = "Topk")]
        topk: Vec<SearchResultItem>,
    }

    #[derive(Debug, Deserialize)]
    struct SearchResultItem {
        recall: RecallMetrics,
    }

    #[derive(Debug, Deserialize)]
    struct RecallMetrics {
        average: f64,
    }

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

    // The directory containing the benchmark executable.
    fn project_directory() -> std::path::PathBuf {
        env!("CARGO_MANIFEST_DIR").into()
    }

    // The directory containing example inputs.
    fn example_directory() -> std::path::PathBuf {
        project_directory().join("example")
    }

    // The directories containing input data in the example inputs.
    fn root_directory() -> std::path::PathBuf {
        project_directory().parent().unwrap().to_path_buf()
    }

    // This operats on a raw JSON like
    // ```json
    // {
    //     "search_directories": [
    //         "string",
    //         "string",
    //     ],
    //     // the rest
    // }
    // ```
    // and prefixes the `root` path to all string entries in `search_directories`.
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

    // Retrieve the number of jobs in the raw input JSON.
    //
    // The format is
    // ```json
    // {
    //     "jobs": [
    //        // jobs
    //     ],
    //     // other keys
    // }
    // ```
    // Panics if the value is not in the expected format.
    fn num_jobs(raw: &serde_json::Value) -> usize {
        let key = "jobs";
        if let serde_json::Value::Object(object) = raw {
            let value = object.get(key).expect("key \"jobs\" should exist");
            if let serde_json::Value::Array(array) = value {
                array.len()
            } else {
                panic!("Expected an Array - got {}", raw);
            }
        } else {
            panic!("Expected an Object - got {}", raw);
        }
    }

    // Async Build Integration Test.
    #[test]
    fn async_integration() {
        // First, parse and modify the input file to establish paths relative to the
        // directory building the dispatcher.
        let mut raw = value_from_file(&example_directory().join("async.json"));
        prefix_search_directories(&mut raw, &root_directory());

        let tempdir = tempfile::tempdir().unwrap();

        let input_path = tempdir.path().join("async.json");
        save_to_file(&input_path, &raw);

        let output_path = tempdir.path().join("output.json");
        assert!(!output_path.exists());

        // Run the example program.
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };
        let app = App::from_commands(command);
        let mut output = Memory::new();

        run_program(&app, &mut output).unwrap();
        println!(
            "output = {}",
            String::from_utf8(output.into_inner()).unwrap()
        );

        // Check that the results file is generated.
        assert!(output_path.exists());

        let results: Vec<Value> = load_from_file(&output_path);
        assert_eq!(results.len(), num_jobs(&raw));
    }

    ////////////////////////////
    //  MinMax Quantization   //
    ////////////////////////////

    #[test]
    fn minmax_quantization_integration() {
        let path = example_directory().join("minmax-exhaustive.json");
        let tempdir = tempfile::tempdir().unwrap();
        let output_path = tempdir.path().join("output.json");
        assert!(!output_path.exists());

        let modified_input_path = tempdir.path().join("input.json");

        let mut raw = value_from_file(&path);
        prefix_search_directories(&mut raw, &root_directory());
        save_to_file(&modified_input_path, &raw);

        run_minmax_integration(&modified_input_path, &output_path)
    }

    #[cfg(feature = "minmax-quantization")]
    fn run_minmax_integration(input_path: &std::path::Path, output_path: &std::path::Path) {
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };

        let app = App::from_commands(command);
        let mut output = Memory::new();

        run_program(&app, &mut output).unwrap();
        println!(
            "output = {}",
            String::from_utf8(output.into_inner()).unwrap()
        );

        // Check that the results file is generated.
        assert!(output_path.exists());
    }

    #[cfg(not(feature = "minmax-quantization"))]
    fn run_minmax_integration(input_path: &std::path::Path, output_path: &std::path::Path) {
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };
        let app = App::from_commands(command);
        let mut output = Memory::new();

        let err = run_program(&app, &mut output).unwrap_err();
        println!("err = {:?}", err);

        let output = String::from_utf8(output.into_inner()).unwrap();
        assert!(output.contains("\"minmax-quantization\" feature"));
        println!("output = {}", output);

        // The output file should not have been created because we failed the test.
        assert!(!output_path.exists());
    }

    /////////////////////////
    // Scalar Quantization //
    /////////////////////////

    #[test]
    fn scalar_quantization_intergration() {
        let input_paths = [example_directory().join("scalar.json")];

        for input_path in input_paths {
            let tempdir = tempfile::tempdir().unwrap();
            let output_path = tempdir.path().join("output.json");
            assert!(!output_path.exists());

            let modified_input_path = tempdir.path().join("input.json");

            let mut raw = value_from_file(&input_path);
            prefix_search_directories(&mut raw, &root_directory());
            save_to_file(&modified_input_path, &raw);

            run_scalar_integration(&modified_input_path, &output_path)
        }
    }

    #[cfg(feature = "scalar-quantization")]
    fn run_scalar_integration(input_path: &std::path::Path, output_path: &std::path::Path) {
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };

        let app = App::from_commands(command);
        let mut output = Memory::new();

        run_program(&app, &mut output).unwrap();
        println!(
            "output = {}",
            String::from_utf8(output.into_inner()).unwrap()
        );

        // Check that the results file is generated.
        assert!(output_path.exists());
    }

    #[cfg(not(feature = "scalar-quantization"))]
    fn run_scalar_integration(input_path: &std::path::Path, output_path: &std::path::Path) {
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };
        let app = App::from_commands(command);
        let mut output = Memory::new();

        let err = run_program(&app, &mut output).unwrap_err();
        println!("err = {:?}", err);

        let output = String::from_utf8(output.into_inner()).unwrap();
        assert!(output.contains("\"scalar-quantization\" feature"));
        println!("output = {}", output);

        // The output file should not have been created because we failed the test.
        assert!(!output_path.exists());
    }

    ////////////////////////////
    // Spherical Quantization //
    ////////////////////////////

    #[test]
    fn spherical_quantization_intergration() {
        let input_paths = [
            example_directory().join("spherical.json"),
            example_directory().join("spherical-exhaustive.json"),
        ];

        for input_path in input_paths {
            let tempdir = tempfile::tempdir().unwrap();
            let output_path = tempdir.path().join("output.json");
            assert!(!output_path.exists());

            let modified_input_path = tempdir.path().join("input.json");

            let mut raw = value_from_file(&input_path);
            prefix_search_directories(&mut raw, &root_directory());
            save_to_file(&modified_input_path, &raw);

            run_spherical_integration(&modified_input_path, &output_path)
        }
    }

    #[cfg(feature = "spherical-quantization")]
    fn run_spherical_integration(input_path: &std::path::Path, output_path: &std::path::Path) {
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };

        let app = App::from_commands(command);
        let mut output = Memory::new();

        run_program(&app, &mut output).unwrap();
        println!(
            "output = {}",
            String::from_utf8(output.into_inner()).unwrap()
        );

        // Check that the results file is generated.
        assert!(output_path.exists());
    }

    #[cfg(not(feature = "spherical-quantization"))]
    fn run_spherical_integration(input_path: &std::path::Path, output_path: &std::path::Path) {
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };
        let app = App::from_commands(command);
        let mut output = Memory::new();

        let err = run_program(&app, &mut output).unwrap_err();
        println!("err = {:?}", err);

        let output = String::from_utf8(output.into_inner()).unwrap();
        println!("output = {}", output);
        assert!(output.contains("\"spherical-quantization\" feature"));

        // The output file should not have been created because we failed the test.
        assert!(!output_path.exists());
    }

    ///////////////////
    // Filter Search //
    ///////////////////

    #[test]
    fn label_index_integration() {
        // First, parse and modify the input file to establish paths relative to the
        // directory building the dispatcher.
        let mut raw = value_from_file(&example_directory().join("metadata-index.json"));
        prefix_search_directories(&mut raw, &root_directory());

        let tempdir = tempfile::tempdir().unwrap();

        let input_path = tempdir.path().join("metadata-index.json");
        save_to_file(&input_path, &raw);

        let output_path = tempdir.path().join("output.json");
        assert!(!output_path.exists());

        // Run the example program.
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };
        let app = App::from_commands(command);
        let mut output = Memory::new();

        run_program(&app, &mut output).unwrap();
        println!(
            "output = {}",
            String::from_utf8(output.into_inner()).unwrap()
        );

        // Check that the results file is generated.
        assert!(output_path.exists());

        let results: Vec<Value> = load_from_file(&output_path);
        assert_eq!(results.len(), num_jobs(&raw));
    }

    #[test]
    fn spherical_filter_search_integration() {
        let input_path = example_directory().join("spherical-filter.json");

        let tempdir = tempfile::tempdir().unwrap();
        let output_path = tempdir.path().join("output.json");
        assert!(!output_path.exists());

        let modified_input_path = tempdir.path().join("input.json");

        let mut raw = value_from_file(&input_path);
        prefix_search_directories(&mut raw, &root_directory());
        save_to_file(&modified_input_path, &raw);

        run_spherical_integration(&modified_input_path, &output_path)
    }

    #[test]
    fn async_filter_integration() {
        // First, parse and modify the input file to establish paths relative to the
        // directory building the dispatcher.
        let mut raw = value_from_file(&example_directory().join("async-filter.json"));
        prefix_search_directories(&mut raw, &root_directory());

        let tempdir = tempfile::tempdir().unwrap();

        let input_path = tempdir.path().join("async-filter.json");
        save_to_file(&input_path, &raw);

        let output_path = tempdir.path().join("output.json");
        assert!(!output_path.exists());

        // Run the example program.
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };
        let app = App::from_commands(command);
        let mut output = Memory::new();

        run_program(&app, &mut output).unwrap();
        println!(
            "output = {}",
            String::from_utf8(output.into_inner()).unwrap()
        );

        // Check that the results file is generated.
        assert!(output_path.exists());

        let results: Vec<Value> = load_from_file(&output_path);
        assert_eq!(results.len(), num_jobs(&raw));
    }

    #[test]
    fn async_filter_integration_with_gt_compute() {
        let storage_provider = FileStorageProvider;

        let disk_index_search_path = root_directory().join("test_data/disk_index_search");

        let result = compute_ground_truth_from_datafiles::<GraphDataF32Vector, FileStorageProvider>(
            &storage_provider,
            Metric::L2, // distance function
            disk_index_search_path
                .join("disk_index_siftsmall_learn_256pts_data.fbin")
                .to_str()
                .unwrap(), // base_file
            disk_index_search_path
                .join("disk_index_sample_query_10pts.fbin")
                .to_str()
                .unwrap(), // query_file
            disk_index_search_path
                .join("gt_small_filter.bin")
                .to_str()
                .unwrap(), // ground_truth_file
            None,       // vector_filters_file
            10,         // default recall_at value
            None,       // insert_file
            None,       // skip_base
            None,       // associated_data_file
            Some(
                disk_index_search_path
                    .join("data.256.label.jsonl")
                    .to_str()
                    .unwrap(),
            ), // base_file_labels
            Some(
                disk_index_search_path
                    .join("query.10.label.jsonl")
                    .to_str()
                    .unwrap(),
            ), // query_file_labels
        );

        match result {
            Ok(_) => {
                println!("Compute ground-truth completed successfully");
            }
            Err(err) => {
                panic!("Error: {:?}", err);
            }
        };

        let mut raw =
            value_from_file(&example_directory().join("async-filter-ground-truth-small.json"));
        prefix_search_directories(&mut raw, &root_directory());

        let tempdir = tempfile::tempdir().unwrap();

        let input_path = tempdir.path().join("async-filter-ground-truth-small.json");
        save_to_file(&input_path, &raw);

        let output_path = tempdir.path().join("output.json");
        assert!(!output_path.exists());

        // Run the example program.
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };
        let app = App::from_commands(command);
        let mut output = Memory::new();

        run_program(&app, &mut output).unwrap();
        println!(
            "output = {}",
            String::from_utf8(output.into_inner()).unwrap()
        );

        // Check that the results file is generated.
        assert!(output_path.exists());

        let results: Vec<Value> = load_from_file(&output_path);
        assert_eq!(results.len(), num_jobs(&raw));

        for (job_idx, result) in results.iter().enumerate() {
            let benchmark_result = BenchmarkResult::deserialize(result).unwrap_or_else(|e| {
                panic!(
                    "Failed to deserialize result for job {}: {}\nResult: {:#?}",
                    job_idx, e, result
                )
            });

            for (search_idx, search_result) in
                benchmark_result.results.search.topk.iter().enumerate()
            {
                let recall_avg = search_result.recall.average;
                println!(
                    "Job {}, Search config {}: recall average = {}",
                    job_idx, search_idx, recall_avg
                );
                assert_eq!(
                    recall_avg, 1.0,
                    "Expected recall average of 1.0 for job {} search config {}, got {}",
                    job_idx, search_idx, recall_avg
                );
            }
        }
    }

    //////////////////////////
    // Product Quantization //
    //////////////////////////

    #[test]
    fn product_quantization_intergration() {
        let input_paths = [
            example_directory().join("product-exhaustive.json"),
            example_directory().join("product.json"),
        ];

        for input_path in input_paths {
            let tempdir = tempfile::tempdir().unwrap();
            let output_path = tempdir.path().join("output.json");
            assert!(!output_path.exists());

            let modified_input_path = tempdir.path().join("input.json");

            let mut raw = value_from_file(&input_path);
            prefix_search_directories(&mut raw, &root_directory());
            save_to_file(&modified_input_path, &raw);

            run_product_integration(&modified_input_path, &output_path)
        }
    }

    #[cfg(feature = "product-quantization")]
    fn run_product_integration(input_path: &std::path::Path, output_path: &std::path::Path) {
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };

        let app = App::from_commands(command);
        let mut output = Memory::new();

        run_program(&app, &mut output).unwrap();
        println!(
            "output = {}",
            String::from_utf8(output.into_inner()).unwrap()
        );

        // Check that the results file is generated.
        assert!(output_path.exists());
    }

    #[cfg(not(feature = "product-quantization"))]
    fn run_product_integration(input_path: &std::path::Path, output_path: &std::path::Path) {
        let command = Commands::Run {
            input_file: input_path.to_owned(),
            output_file: output_path.to_owned(),
            dry_run: false,
        };
        let app = App::from_commands(command);
        let mut output = Memory::new();

        let err = run_program(&app, &mut output).unwrap_err();
        println!("err = {:?}", err);

        let output = String::from_utf8(output.into_inner()).unwrap();
        assert!(output.contains("\"product-quantization\" feature"));
        println!("output = {}", output);

        // The output file should not have been created because we failed the test.
        assert!(!output_path.exists());
    }
}
