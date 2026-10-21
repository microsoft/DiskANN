/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fs::File, num::NonZeroUsize};

use diskann_benchmark_runner::{app::Commands, output::Memory};
use serde_json::{json, Value};

use super::*;

fn small_input() -> Value {
    json!({
        "search_directories": [],
        "jobs": [{
            "type": "graph-ivf-build",
            "content": {
                "dataset": {"kind": "synthetic", "vectors": 64, "queries": 8, "dimensions": 8, "seed": 42},
                "build": {"initial_centroids": 2, "batch_size": 8, "split_threshold": 8, "two_means_iterations": 3, "seed": 7, "repetitions": 2},
                "graph": {"max_degree": 16, "l_build": 64},
                "search": {"k": 4, "nprobe": [1, 64], "l_search": 64, "warmup_repetitions": 1, "repetitions": 2}
            }
        }]
    })
}

fn run_cli(input: &Value) -> anyhow::Result<Value> {
    let directory = tempfile::tempdir()?;
    let input_path = directory.path().join("input.json");
    let output_path = directory.path().join("output.json");
    serde_json::to_writer(File::create(&input_path)?, input)?;
    crate::Cli::from_commands(
        Commands::Run {
            input_file: input_path,
            output_file: output_path.clone(),
            dry_run: false,
            allow_debug: true,
        },
        true,
    )
    .run(&mut Memory::new())?;
    Ok(serde_json::from_reader(File::open(output_path)?)?)
}

#[test]
fn graph_ivf_cli_reports_build_and_real_graph_search() {
    let output = run_cli(&small_input()).unwrap();
    let results = &output[0]["results"];
    let builds = results["builds"].as_array().unwrap();
    assert_eq!(builds.len(), 2);
    for build in builds {
        assert_eq!(build["inserted"], 64);
        let splits = build["splits"].as_u64().unwrap();
        assert!(splits > 0);
        assert_eq!(build["lists"].as_u64().unwrap(), 2 + splits);
    }
    let searches = results["searches"].as_array().unwrap();
    assert_eq!(searches.len(), 8);
    for search in searches {
        let scanned = search["mean_points_scanned"].as_f64().unwrap();
        if search["requested_nprobe"] == 64 {
            assert_eq!(search["recall_at_k"], 1.0);
            assert_eq!(scanned, 64.0);
            assert_eq!(search["mean_results_returned"], 4.0);
        } else {
            assert!(
                scanned < 64.0,
                "nprobe=1 must not silently scan the whole corpus"
            );
        }
    }
}

#[test]
fn graph_ivf_split_preserves_points_and_full_probe_matches_oracle() {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(async {
            let input: GraphIvfBuild =
                serde_json::from_value(small_input()["jobs"][0]["content"].clone()).unwrap();
            let (data, queries) = load_data(&input).unwrap();
            let queries = queries.unwrap();
            let truth = load_groundtruth(&input, data.as_view(), queries.as_view(), 4).unwrap();
            let (index, graph, result) = build(&input, data.as_view()).await.unwrap();
            assert!(result.splits > 0);
            index.provider().validate_membership(64).unwrap();
            for (id, vector) in data.rows().enumerate() {
                assert_eq!(index.provider().vector(id as u32).unwrap(), vector);
            }
            for (query, expected) in queries.rows().zip(truth.rows()) {
                let result = graph
                    .search(
                        index.provider(),
                        query,
                        NonZeroUsize::new(4).unwrap(),
                        64,
                        64,
                    )
                    .await
                    .unwrap();
                assert_eq!(result.ids, expected);
                assert_eq!(result.scanned, 64);
            }
        });
}

#[test]
fn graph_ivf_recall_does_not_reward_duplicates_or_short_results() {
    let expected = [3, 8, 5, 2];
    assert_eq!(recall_hits(&[3, 3, 3, 3], &expected, 4), 1);
    assert_eq!(recall_hits(&[3], &expected, 4), 1);
    assert_eq!(recall_hits(&[9, 10, 11, 12, 3], &expected, 4), 0);
    assert_eq!(recall_hits(&[], &expected, 4), 0);
}

#[test]
fn graph_ivf_rejects_invalid_measurement_inputs() {
    for (field, value) in [
        ("split_threshold", 1),
        ("batch_size", 0),
        ("initial_centroids", 65),
    ] {
        let mut input = small_input();
        input["jobs"][0]["content"]["build"][field] = json!(value);
        assert!(run_cli(&input).is_err(), "accepted invalid {field}");
    }
    let mut input = small_input();
    input["jobs"][0]["content"]["search"]["l_search"] = json!(4);
    assert!(run_cli(&input).is_err());
    input["jobs"][0]["content"]["search"]["l_search"] = json!(64);
    input["jobs"][0]["content"]["search"]["k"] = json!(65);
    assert!(run_cli(&input).is_err());
    let invalid = rowmajor::Owned::try_from_data(vec![f32::NAN].into_boxed_slice(), 1, 1).unwrap();
    assert!(validate_vectors(invalid.as_view(), "dataset").is_err());
    let duplicate = rowmajor::Owned::try_from_data(vec![0, 0].into_boxed_slice(), 1, 2).unwrap();
    assert!(validate_groundtruth(duplicate.as_view(), 1, 2, 2).is_err());
}

#[test]
fn graph_ivf_files_and_build_only_use_same_pipeline() {
    let directory = tempfile::tempdir().unwrap();
    let input: GraphIvfBuild =
        serde_json::from_value(small_input()["jobs"][0]["content"].clone()).unwrap();
    let (data, queries) = load_data(&input).unwrap();
    let queries = queries.unwrap();
    let truth = load_groundtruth(&input, data.as_view(), queries.as_view(), 4).unwrap();
    let data_path = directory.path().join("data.bin");
    let query_path = directory.path().join("queries.bin");
    let truth_path = directory.path().join("truth.bin");
    for (path, rows, cols, bytes) in [
        (
            &data_path,
            data.nrows(),
            data.ncols(),
            bytemuck::cast_slice(data.as_slice()),
        ),
        (
            &query_path,
            queries.nrows(),
            queries.ncols(),
            bytemuck::cast_slice(queries.as_slice()),
        ),
        (
            &truth_path,
            truth.nrows(),
            truth.ncols(),
            bytemuck::cast_slice(truth.as_slice()),
        ),
    ] {
        let mut file = File::create(path).unwrap();
        file.write_all(&(rows as u32).to_le_bytes()).unwrap();
        file.write_all(&(cols as u32).to_le_bytes()).unwrap();
        file.write_all(bytes).unwrap();
    }
    let mut job = small_input();
    job["jobs"][0]["content"]["dataset"] = json!({"kind": "files", "data": data_path, "queries": query_path, "groundtruth": truth_path});
    job["jobs"][0]["content"]["build"]["repetitions"] = json!(1);
    let output = run_cli(&job).unwrap();
    assert_eq!(output[0]["results"]["searches"][2]["recall_at_k"], 1.0);
    job["jobs"][0]["content"]["dataset"] = json!({"kind": "files", "data": data_path});
    job["jobs"][0]["content"]["search"] = Value::Null;
    let output = run_cli(&job).unwrap();
    assert_eq!(output[0]["results"]["builds"][0]["inserted"], 64);
    assert!(output[0]["results"]["searches"]
        .as_array()
        .unwrap()
        .is_empty());
}
