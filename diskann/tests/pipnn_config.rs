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
use diskann::graph::pipnn::{PiPNNBuildContext, PiPNNConfig};
use diskann_vector::distance::Metric;

fn pipnn_config() -> PiPNNConfig {
    PiPNNConfig {
        c_max: 512,
        c_min: 64,
        p_samp: 0.01,
        fanout: vec![10, 3],
        k: 2,
        replicas: 1,
    }
}

fn graph_config(metric: Metric, alpha: f32) -> diskann::graph::Config {
    config::Builder::new_with(64, MaxDegree::same(), 72, metric.into(), |builder| {
        builder.alpha(alpha);
    })
    .build()
    .unwrap()
}

fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
}

#[test]
fn rejects_each_invalid_algorithm_parameter() {
    let graph = graph_config(Metric::L2, 1.2);
    let pool = pool();
    let mut cases = [
        PiPNNConfig {
            c_max: 0,
            ..pipnn_config()
        },
        PiPNNConfig {
            c_min: 0,
            ..pipnn_config()
        },
        PiPNNConfig {
            c_min: 513,
            ..pipnn_config()
        },
        PiPNNConfig {
            p_samp: 0.0,
            ..pipnn_config()
        },
        PiPNNConfig {
            p_samp: -0.01,
            ..pipnn_config()
        },
        PiPNNConfig {
            p_samp: 1.01,
            ..pipnn_config()
        },
        PiPNNConfig {
            p_samp: f64::NAN,
            ..pipnn_config()
        },
        PiPNNConfig {
            fanout: Vec::new(),
            ..pipnn_config()
        },
        PiPNNConfig {
            fanout: vec![1, 0],
            ..pipnn_config()
        },
        PiPNNConfig {
            fanout: vec![17],
            ..pipnn_config()
        },
        PiPNNConfig {
            k: 0,
            ..pipnn_config()
        },
        PiPNNConfig {
            replicas: 0,
            ..pipnn_config()
        },
    ];

    for config in &mut cases {
        let error = PiPNNBuildContext::new(config.clone(), &graph, Metric::L2, &pool)
            .expect_err("invalid PiPNN config must be rejected");
        assert_eq!(error.kind(), diskann::ANNErrorKind::IndexConfigError);
    }
}

#[test]
fn rejects_graph_policy_for_a_different_metric() {
    let graph = graph_config(Metric::InnerProduct, 1.2);
    let pool = pool();

    let error = PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap_err();

    assert_eq!(error.kind(), diskann::ANNErrorKind::IndexConfigError);
    assert!(error.to_string().contains("prune kind"));
}

#[test]
fn does_not_add_alpha_validation_beyond_graph_config() {
    let pool = pool();
    for alpha in [0.9, f32::NAN, f32::INFINITY] {
        let graph = graph_config(Metric::L2, alpha);
        PiPNNBuildContext::new(pipnn_config(), &graph, Metric::L2, &pool).unwrap();
    }
}
