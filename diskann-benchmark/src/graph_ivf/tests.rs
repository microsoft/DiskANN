/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fs::File, io::Write, num::NonZeroUsize};

use diskann_benchmark_runner::{app::Commands, output::Memory};
use diskann_utils::views::rowmajor::{self, Matrix};
use serde_json::{json, Value};

use super::{
    build::build,
    data::{load_data, load_groundtruth, load_vectors, validate_groundtruth, validate_vectors},
};
use crate::inputs::graph_ivf::{FileType, GraphIvfBuild};

fn small_input() -> Value {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("test_data")
        .join("disk_index_search");
    json!({
        "search_directories": [],
        "jobs": [{
            "type": "graph-ivf-build",
            "content": {
                "dataset": {
                    "data": directory.join("disk_index_siftsmall_learn_256pts_data.fbin"),
                    "queries": directory.join("disk_index_sample_query_10pts.fbin")
                },
                "build": {"initial_centroids": 2, "batch_size": 32, "split_threshold": 32, "two_means_iterations": 3, "seed": 7, "repetitions": 2},
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
fn graph_ivf_repository_workloads_run_with_existing_datasets() {
    use crate::inputs::Example;

    let example: Value =
        serde_json::from_str(include_str!("../../example/graph-ivf.json")).unwrap();
    assert_eq!(
        serde_json::to_value(
            serde_json::from_value::<GraphIvfBuild>(example["jobs"][0]["content"].clone()).unwrap()
        )
        .unwrap(),
        serde_json::to_value(GraphIvfBuild::example()).unwrap()
    );
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    for (text, vectors, dimensions, queries) in [
        (include_str!("../../example/graph-ivf.json"), 256, 128, 10),
        (
            include_str!("../../perf_test_inputs/graph-ivf-stages-build.json"),
            10000,
            192,
            0,
        ),
        (
            include_str!("../../perf_test_inputs/graph-ivf-stages-search.json"),
            10000,
            192,
            100,
        ),
    ] {
        let mut input: Value = serde_json::from_str(text).unwrap();
        for directory in input["search_directories"].as_array_mut().unwrap() {
            *directory = json!(root.join(directory.as_str().unwrap()));
        }
        let output = run_cli(&input).unwrap();
        let results = &output[0]["results"];
        assert_eq!(results["vectors"], vectors);
        assert_eq!(results["dimensions"], dimensions);
        for build in results["builds"].as_array().unwrap() {
            assert_eq!(build["inserted"], vectors);
            assert!(build["splits"].as_u64().unwrap() > 0);
        }
        let searches = results["searches"].as_array().unwrap();
        assert_eq!(searches.is_empty(), queries == 0);
        for search in searches {
            assert_eq!(search["queries"], queries);
            assert!(search["recall_at_k"].as_f64().unwrap() > 0.0);
            assert_eq!(search["mean_results_returned"], 10.0);
        }
    }
}

#[test]
fn graph_ivf_cli_reports_build_and_real_graph_search() {
    let output = run_cli(&small_input()).unwrap();
    let results = &output[0]["results"];
    let builds = results["builds"].as_array().unwrap();
    assert_eq!(builds.len(), 2);
    for build in builds {
        assert_eq!(build["inserted"], 256);
        let splits = build["splits"].as_u64().unwrap();
        assert!(splits > 0);
        assert_eq!(build["lists"].as_u64().unwrap(), 2 + splits);
        let two_means = build["two_means_seconds"].as_f64().unwrap();
        let reassignment = build["reassignment_seconds"].as_f64().unwrap();
        let other = build["other_ivf_seconds"].as_f64().unwrap();
        let ivf = build["ivf_seconds"].as_f64().unwrap();
        assert!(two_means >= 0.0 && reassignment >= 0.0 && other >= 0.0);
        assert!((two_means + reassignment + other - ivf).abs() < 1e-9);
        assert!(ivf <= build["total_seconds"].as_f64().unwrap());
        let histogram = build["split_parent_histogram"].as_object().unwrap();
        assert_eq!(
            histogram.values().map(|n| n.as_u64().unwrap()).sum::<u64>(),
            8
        );
        assert_eq!(
            histogram
                .iter()
                .map(|(parents, count)| {
                    parents.parse::<u64>().unwrap() * count.as_u64().unwrap()
                })
                .sum::<u64>(),
            splits
        );
        // Every split parent had more than 32 points; reassignment may revisit
        // points on subsequent batches, so this is not a distinct-point count.
        assert!(build["split_points"].as_u64().unwrap() >= 33 * splits);
    }
    let searches = results["searches"].as_array().unwrap();
    assert_eq!(searches.len(), 8);
    for search in searches {
        let scanned = search["mean_points_scanned"].as_f64().unwrap();
        if search["requested_nprobe"] == 64 {
            assert_eq!(search["recall_at_k"], 1.0);
            assert_eq!(scanned, 256.0);
            assert_eq!(search["mean_results_returned"], 4.0);
        } else {
            assert!(
                scanned < 256.0,
                "nprobe=1 must not silently scan the whole corpus"
            );
        }
    }
}

#[test]
fn graph_ivf_unsplit_build_reports_zero_split_work() {
    let mut input = small_input();
    input["jobs"][0]["content"]["build"]["split_threshold"] = json!(256);
    input["jobs"][0]["content"]["build"]["batch_size"] = json!(60);
    input["jobs"][0]["content"]["search"] = Value::Null;
    let output = run_cli(&input).unwrap();
    for build in output[0]["results"]["builds"].as_array().unwrap() {
        assert_eq!(build["splits"], 0);
        assert_eq!(build["split_points"], 0);
        assert_eq!(build["two_means_seconds"], 0.0);
        assert_eq!(build["reassignment_seconds"], 0.0);
        assert_eq!(build["other_ivf_seconds"], build["ivf_seconds"]);
        // Four full batches and a 16-point tail are each counted once.
        assert_eq!(build["split_parent_histogram"], json!({"0": 5}));
        assert_eq!(build["inserted"], 256);
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
            index.provider().validate_membership(256).unwrap();
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
                assert_eq!(result.scanned, 256);
            }
        });
}

#[test]
fn graph_ivf_recall_keeps_full_denominator_for_short_results() {
    let mut input = small_input();
    input["jobs"][0]["content"]["search"]["k"] = json!(256);
    input["jobs"][0]["content"]["search"]["nprobe"] = json!([1]);
    let output = run_cli(&input).unwrap();
    for search in output[0]["results"]["searches"].as_array().unwrap() {
        let returned = search["mean_results_returned"].as_f64().unwrap();
        assert!(returned > 0.0 && returned < 256.0);
        let recall = search["recall_at_k"].as_f64().unwrap();
        assert!((recall - returned / 256.0).abs() < 1e-12);
    }
}

#[test]
fn graph_ivf_rejects_invalid_measurement_inputs() {
    for (field, value) in [
        ("split_threshold", 1),
        ("batch_size", 0),
        ("initial_centroids", 257),
    ] {
        let mut input = small_input();
        input["jobs"][0]["content"]["build"][field] = json!(value);
        assert!(run_cli(&input).is_err(), "accepted invalid {field}");
    }
    let mut input = small_input();
    input["jobs"][0]["content"]["search"]["l_search"] = json!(4);
    assert!(run_cli(&input).is_err());
    input["jobs"][0]["content"]["search"]["l_search"] = json!(64);
    input["jobs"][0]["content"]["search"]["k"] = json!(257);
    assert!(run_cli(&input).is_err());
    let mut input = small_input();
    input["jobs"][0]["content"]["dataset"]["queries"] = Value::Null;
    assert!(run_cli(&input).is_err());
    let mut input = small_input();
    input["jobs"][0]["content"]["dataset"] =
        json!({"kind": "synthetic", "vectors": 64, "queries": 8, "dimensions": 8, "seed": 42});
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
    job["jobs"][0]["content"]["dataset"] =
        json!({"data": data_path, "queries": query_path, "groundtruth": truth_path});
    job["jobs"][0]["content"]["build"]["repetitions"] = json!(1);
    let output = run_cli(&job).unwrap();
    assert_eq!(output[0]["results"]["searches"][2]["recall_at_k"], 1.0);
    job["jobs"][0]["content"]["dataset"] = json!({"data": data_path});
    job["jobs"][0]["content"]["search"] = Value::Null;
    let output = run_cli(&job).unwrap();
    assert_eq!(output[0]["results"]["builds"][0]["inserted"], 256);
    assert!(output[0]["results"]["searches"]
        .as_array()
        .unwrap()
        .is_empty());
}

#[test]
fn fp16_input_is_promoted_exactly_before_timing() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("half.bin");
    let values = [
        half::f16::from_f32(1.5),
        half::f16::from_f32(-0.5),
        half::f16::from_f32(2.0),
        half::f16::from_f32(0.25),
    ];
    let mut file = File::create(&path).unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap();
    file.write_all(bytemuck::cast_slice(&values)).unwrap();
    drop(file);
    let loaded = load_vectors(&path, FileType::Float16).unwrap();
    assert_eq!(loaded.as_slice(), [1.5, -0.5, 2.0, 0.25]);
    assert!(load_vectors(&path, FileType::Float32).is_err());
}

#[cfg(feature = "ivf-two-means")]
#[test]
fn exact_strategy_preserves_centroid_bits_and_ordered_members() {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(async {
            let mut input: GraphIvfBuild =
                serde_json::from_value(small_input()["jobs"][0]["content"].clone()).unwrap();
            let (data, _) = load_data(&input).unwrap();
            let (a, _, _) = build(&input, data.as_view()).await.unwrap();
            input.build.two_means_strategy =
                crate::inputs::graph_ivf_two_means::TwoMeansStrategy::TiledExact;
            let (b, _, _) = build(&input, data.as_view()).await.unwrap();
            let a = a.provider();
            let b = b.provider();
            a.validate_membership(data.nrows()).unwrap();
            b.validate_membership(data.nrows()).unwrap();
            assert_eq!(a.lists.len(), b.lists.len());
            for (id, a) in &a.lists {
                let b = &b.lists[id];
                assert_eq!(a.members, b.members);
                assert_eq!(
                    a.centroid.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.centroid.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        });
}

#[cfg(feature = "ivf-two-means")]
#[test]
fn experimental_strategies_use_real_ivf_build_and_search() {
    let mut raw = small_input();
    raw["jobs"][0]["content"]["build"]["repetitions"] = json!(1);
    let reference = run_cli(&raw).unwrap();
    for strategy in crate::inputs::graph_ivf_two_means::TwoMeansStrategy::ALL {
        raw["jobs"][0]["content"]["build"]["two_means_strategy"] = json!(strategy.name());
        let result = run_cli(&raw).unwrap();
        let build = &result[0]["results"]["builds"][0];
        assert_eq!(build["inserted"], 256);
        assert_eq!(result[0]["results"]["searches"][2]["recall_at_k"], 1.0);
        {
            assert_eq!(
                build["index_fingerprint"],
                reference[0]["results"]["builds"][0]["index_fingerprint"]
            );
            for (a, b) in result[0]["results"]["searches"]
                .as_array()
                .unwrap()
                .iter()
                .zip(reference[0]["results"]["searches"].as_array().unwrap())
            {
                for field in [
                    "recall_at_k",
                    "result_fingerprint",
                    "mean_points_scanned",
                    "mean_lists_scanned",
                    "mean_centroid_comparisons",
                ] {
                    assert_eq!(a[field], b[field]);
                }
            }
        }
    }
    for removed in [
        "not-a-strategy",
        "separate",
        "packed_paired",
        "batched_4",
        "batched_8",
        "sgemm",
        "center_difference",
        "transposed_lloyd_reference",
        "transposed_l2_16",
        "transposed_dot_16",
    ] {
        raw["jobs"][0]["content"]["build"]["two_means_strategy"] = json!(removed);
        assert!(run_cli(&raw).is_err());
    }
}

#[test]
#[cfg(feature = "ivf-two-means")]
fn two_means_strategy_cli_covers_all_candidates_and_layouts() {
    let input = json!({
        "search_directories": [],
        "jobs": [{
            "type": "graph-ivf-two-means",
            "content": {
                "bucket_sizes": [17],
                "dimensions": [9],
                "iterations": [3],
                "layouts": ["contiguous", "scattered"],
                "seed": 7,
                "repetitions": 2,
                "warmup_repetitions": 0
            }
        }]
    });
    let output = run_cli(&input).unwrap();
    let workloads = output[0]["results"]["workloads"].as_array().unwrap();
    assert_eq!(workloads.len(), 2);
    for workload in workloads {
        let strategies = workload["strategies"].as_array().unwrap();
        assert_eq!(strategies.len(), 2);
        assert_eq!(strategies[0]["strategy"], "paired_point");
        assert!(strategies.iter().all(|strategy| strategy["samples"] == 2));
        assert_eq!(workload["invocation_order"].as_array().unwrap().len(), 4);
        for strategy in strategies {
            assert_eq!(strategy["status"], "complete");
            assert_eq!(
                strategy["diagnostics"]["steps"].as_array().unwrap().len(),
                3
            );
        }
    }
}

#[cfg(feature = "ivf-two-means")]
#[test]
fn two_means_abba_records_every_position_and_rejects_invalid_pairs() {
    let mut input = json!({"search_directories": [], "jobs": [{"type": "graph-ivf-two-means", "content": {
        "bucket_sizes": [17], "dimensions": [9], "iterations": [3], "layouts": ["contiguous"],
        "seed": 7, "repetitions": 2, "warmup_repetitions": 1, "order": "abba",
        "strategies": ["paired_point", "tiled_exact"]
    }}]});
    let result = run_cli(&input).unwrap();
    let workload = &result[0]["results"]["workloads"][0];
    let order = workload["invocation_order"].as_array().unwrap();
    assert_eq!(order.len(), 12);
    for (round, block) in order.chunks_exact(4).enumerate() {
        for (position, expected) in ["paired_point", "tiled_exact", "tiled_exact", "paired_point"]
            .iter()
            .enumerate()
        {
            assert_eq!(block[position]["strategy"], *expected);
            assert_eq!(block[position]["round"], round);
            assert_eq!(block[position]["warmup"], round == 0);
        }
    }
    assert!(workload["strategies"]
        .as_array()
        .unwrap()
        .iter()
        .all(|s| s["samples"] == 4));
    for pair in [
        json!([]),
        json!(["paired_point", "paired_point"]),
        json!(["tiled_exact", "paired_point"]),
    ] {
        input["jobs"][0]["content"]["strategies"] = pair;
        assert!(run_cli(&input).is_err());
    }
}

#[cfg(feature = "ivf-two-means")]
#[test]
fn two_means_file_sampling_preserves_logical_rows_and_checks_shape() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("points.bin");
    let mut file = File::create(&path).unwrap();
    file.write_all(&19u32.to_le_bytes()).unwrap();
    file.write_all(&9u32.to_le_bytes()).unwrap();
    for i in 0..19 * 9 {
        file.write_all(&half::f16::from_f32((i % 11) as f32 - 5.0).to_le_bytes())
            .unwrap();
    }
    drop(file);
    let mut input = json!({
        "search_directories": [],
        "jobs": [{
            "type": "graph-ivf-two-means",
            "content": {
                "data": path, "data_type": "float16",
                "bucket_sizes": [17], "dimensions": [9], "iterations": [3],
                "layouts": ["contiguous", "scattered"], "seed": 7,
                "repetitions": 2, "warmup_repetitions": 1
            }
        }]
    });
    let result = run_cli(&input).unwrap();
    assert_eq!(result[0]["results"]["source"], "file_sample");
    let workloads = result[0]["results"]["workloads"].as_array().unwrap();
    for (a, b) in workloads[0]["strategies"]
        .as_array()
        .unwrap()
        .iter()
        .zip(workloads[1]["strategies"].as_array().unwrap())
    {
        assert_eq!(
            a["diagnostics"], b["diagnostics"],
            "layout changed logical result"
        );
        assert_eq!(a["total_samples_ns"].as_array().unwrap().len(), 2);
    }
    input["jobs"][0]["content"]["dimensions"] = json!([8]);
    assert!(run_cli(&input).is_err());
    input["jobs"][0]["content"]["dimensions"] = json!([9]);
    input["jobs"][0]["content"]["bucket_sizes"] = json!([20]);
    assert!(run_cli(&input).is_err());
    input["jobs"][0]["content"]["bucket_sizes"] = json!([1]);
    assert!(run_cli(&input).is_err());
    input["jobs"][0]["content"]["bucket_sizes"] = json!([]);
    assert!(run_cli(&input).is_err());
}

#[cfg(feature = "ivf-two-means")]
#[test]
fn parent_replay_preserves_member_order_and_records_missing_ordinals() {
    let directory = tempfile::tempdir().unwrap();
    let mut index = small_input()["jobs"][0]["content"].clone();
    index["search"] = Value::Null;
    index["build"]["repetitions"] = json!(1);
    let input = json!({"search_directories": [], "jobs": [{"type": "graph-ivf-parent-replay", "content": {
        "index": index, "directory": directory.path().join("captured"),
        "parent_ordinals": [0, 1, 9999], "seed": 7, "repetitions": 1, "warmup_repetitions": 0
    }}]});
    let output = run_cli(&input).unwrap();
    let result = &output[0]["results"];
    assert_eq!(result["missing_ordinals"], json!([9999]));
    assert_eq!(result["parents"].as_array().unwrap().len(), 2);
    for parent in result["parents"].as_array().unwrap() {
        let snapshot = &parent["snapshot"];
        let ids: Vec<usize> =
            serde_json::from_reader(File::open(snapshot["ids_file"].as_str().unwrap()).unwrap())
                .unwrap();
        let points = load_vectors(
            std::path::Path::new(snapshot["coordinates"].as_str().unwrap()),
            FileType::Float32,
        )
        .unwrap();
        let source = load_vectors(
            std::path::Path::new(index["dataset"]["data"].as_str().unwrap()),
            FileType::Float32,
        )
        .unwrap();
        for (&id, row) in ids.iter().zip(points.rows()) {
            assert_eq!(row, source.row(id));
        }
        for (a, b) in parent["workloads"][0]["strategies"]
            .as_array()
            .unwrap()
            .iter()
            .zip(parent["workloads"][1]["strategies"].as_array().unwrap())
        {
            assert_eq!(a["diagnostics"], b["diagnostics"]);
            assert_eq!(a["status"], "complete");
        }
    }
    assert!(
        run_cli(&input).is_err(),
        "existing capture directory must not be overwritten"
    );
}
