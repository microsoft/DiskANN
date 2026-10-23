/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;
use crate::inputs::graph_ivf_two_means::TwoMeansStrategy;

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
            input.build.two_means_strategy = TwoMeansStrategy::TiledExact;
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

#[test]
fn experimental_strategies_use_real_ivf_build_and_search() {
    let mut raw = small_input();
    raw["jobs"][0]["content"]["build"]["repetitions"] = json!(1);
    let reference = run_cli(&raw).unwrap();
    for strategy in TwoMeansStrategy::ALL {
        raw["jobs"][0]["content"]["build"]["two_means_strategy"] = json!(strategy.name());
        let result = run_cli(&raw).unwrap();
        let build = &result[0]["results"]["builds"][0];
        assert_eq!(build["inserted"], 256);
        assert_eq!(result[0]["results"]["searches"][2]["recall_at_k"], 1.0);
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
