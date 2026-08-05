/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![cfg(feature = "pipnn")]
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "deterministic test fixture construction must abort on invalid setup"
)]

use diskann::graph::config::{self, MaxDegree};
use diskann::graph::pipnn::{PiPNNBuildContext, PiPNNConfig, build_graph};
use diskann_utils::views::MatrixView;
use diskann_vector::distance::Metric;
use half::f16;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn pipnn_config() -> PiPNNConfig {
    PiPNNConfig {
        c_max: 4,
        c_min: 1,
        p_samp: 0.5,
        fanout: vec![2],
        k: 1,
        replicas: 1,
    }
}

fn graph_config(metric: Metric, degree: usize) -> diskann::graph::Config {
    config::Builder::new_with(degree, MaxDegree::same(), 8, metric.into(), |builder| {
        builder.alpha(1.2);
    })
    .build()
    .unwrap()
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

fn rows(graph: Vec<diskann::graph::AdjacencyList<u32>>) -> Vec<Vec<u32>> {
    graph.into_iter().map(Vec::from).collect()
}

fn assert_graph_invariants(
    graph: &[diskann::graph::AdjacencyList<u32>],
    points: usize,
    degree: usize,
) {
    assert_eq!(graph.len(), points);
    for (source, row) in graph.iter().enumerate() {
        assert!(row.len() <= degree);
        let mut sorted = row.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), row.len());
        assert!(
            row.iter()
                .all(|&id| (id as usize) < points && id as usize != source)
        );
    }
}

#[test]
fn builds_a_single_leaf_graph_for_real_dataset_ids() {
    let data = [0.0_f32, 1.0, 2.0, 3.0];
    let data = MatrixView::try_from(&data[..], 4, 1).unwrap();
    let graph = graph_config(Metric::L2, 2);
    let pool = pool(2);
    let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

    let actual = build_graph(data, &context).unwrap();

    assert_eq!(rows(actual), [vec![1], vec![0, 2], vec![1, 3], vec![2]]);

    let graph = graph_config(Metric::L2, 1);
    let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

    let pruned = build_graph(data, &context).unwrap();

    assert_graph_invariants(&pruned, 4, 1);
    for (source, neighbors) in pruned.iter().enumerate() {
        assert_eq!(source.abs_diff(neighbors[0] as usize), 1);
    }
}

#[test]
fn prunes_complete_single_leaf_candidates_to_the_graph_degree() {
    let data = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
    let data = MatrixView::try_from(&data[..], 5, 1).unwrap();
    let graph = graph_config(Metric::L2, 1);
    let pool = pool(2);
    let config = PiPNNConfig {
        c_max: 5,
        c_min: 1,
        p_samp: 0.5,
        fanout: vec![2],
        k: 4,
        replicas: 1,
    };
    let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

    let actual = build_graph(data, &context).unwrap();

    assert_graph_invariants(&actual, 5, 1);
    assert!(actual.iter().all(|row| row.len() == 1));
}

#[test]
fn rejects_empty_dataset_dimensions_at_the_public_boundary() {
    let graph = graph_config(Metric::L2, 2);
    let pool = pool(1);
    let context = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();

    let no_rows = MatrixView::try_from(&[] as &[f32], 0, 4).unwrap();
    let no_columns = MatrixView::try_from(&[] as &[f32], 4, 0).unwrap();

    assert!(build_graph(no_rows, &context).is_err());
    assert!(build_graph(no_columns, &context).is_err());
}

#[test]
fn supports_every_source_type_and_metric() {
    fn build<T: diskann::utils::VectorRepr + Send + Sync + 'static>(values: &[T], metric: Metric) {
        let data = MatrixView::try_from(values, 6, 2).unwrap();
        let graph = graph_config(metric, 2);
        let pool = pool(2);
        let context = PiPNNBuildContext::new(pipnn_config(), &graph, metric, &pool).unwrap();
        let actual = build_graph(data, &context).unwrap();
        assert_graph_invariants(&actual, 6, 2);
    }

    let values = [
        1.0_f32, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.5, 0.5, -0.5, -0.5,
    ];
    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        build(&values, metric);
    }
    build(&values.map(f16::from_f32), Metric::L2);
    build(&[1_u8, 0, 0, 1, 2, 0, 0, 2, 1, 1, 2, 2], Metric::L2);
    build(&[1_i8, 0, 0, 1, -1, 0, 0, -1, 1, 1, -1, -1], Metric::L2);
}

#[test]
fn integer_normalized_cosine_matches_cosine() {
    fn assert_match<T: diskann::utils::VectorRepr + Send + Sync + 'static>(values: &[T]) {
        let data = MatrixView::try_from(values, 8, 2).unwrap();
        let pool = pool(2);
        let build = |metric| {
            let graph = graph_config(metric, 2);
            let config = PiPNNConfig {
                c_max: 8,
                c_min: 1,
                p_samp: 0.5,
                fanout: vec![2],
                k: 1,
                replicas: 1,
            };
            let context = PiPNNBuildContext::new(config, &graph, metric, &pool).unwrap();
            rows(build_graph(data, &context).unwrap())
        };
        assert_eq!(build(Metric::CosineNormalized), build(Metric::Cosine));
    }

    assert_match(&[1_u8, 0, 100, 1, 2, 0, 0, 1, 1, 1, 200, 2, 2, 1, 1, 2]);
    assert_match(&[1_i8, 0, 100, 1, 2, 0, 0, 1, 1, 1, 120, 2, 2, 1, 1, 2]);
}

#[test]
fn is_deterministic_for_a_fixed_pool_size() {
    let data: Vec<f32> = (0..96 * 4)
        .map(|value| ((value * 17 + 3) % 101) as f32)
        .collect();
    let data = MatrixView::try_from(&data[..], 96, 4).unwrap();
    let graph = graph_config(Metric::L2, 8);
    let pool = pool(4);
    let config = PiPNNConfig {
        c_max: 16,
        c_min: 4,
        p_samp: 0.25,
        fanout: vec![3, 2],
        k: 3,
        replicas: 2,
    };
    let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

    let first = build_graph(data, &context).unwrap();
    let second = build_graph(data, &context).unwrap();

    assert_eq!(first, second);
    assert_graph_invariants(&first, 96, 8);
}

#[test]
fn fixed_seed_randomized_sweeps_preserve_graph_invariants() {
    let mut rng = StdRng::seed_from_u64(0x857a_d38b_44c2_0f11);
    for case in 0..24 {
        let points = rng.random_range(4..=32);
        let dimensions = rng.random_range(1..=8);
        let c_max = rng.random_range(4..=points.min(12));
        let c_min = rng.random_range(1..=c_max);
        let degree = rng.random_range(1..=points.min(8));
        let values: Vec<f32> = (0..points * dimensions)
            .map(|_| rng.random_range(-10.0..10.0))
            .collect();
        let data = MatrixView::try_from(&values[..], points, dimensions).unwrap();
        let graph = graph_config(Metric::L2, degree);
        let pool = pool(2);
        let config = PiPNNConfig {
            c_max,
            c_min,
            p_samp: 0.5,
            fanout: vec![2],
            k: rng.random_range(1..=3),
            replicas: rng.random_range(1..=2),
        };
        let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();

        let actual = build_graph(data, &context)
            .unwrap_or_else(|error| panic!("randomized case {case} failed: {error}"));
        assert_graph_invariants(&actual, points, degree);
    }
}
