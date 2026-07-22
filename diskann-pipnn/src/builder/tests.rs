/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_vector::distance::Metric;
use rand::{Rng, SeedableRng};

use super::*;
use crate::PiPNNConfig;

fn random_data(points: usize, dimensions: usize) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..points * dimensions)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect()
}

fn context(metric: Metric, final_prune: bool, threads: usize) -> PiPNNBuildContext {
    let config = PiPNNConfig {
        c_max: 32,
        c_min: 4,
        k: 3,
        replicas: 1,
        l_max: 32,
        final_prune,
        ..PiPNNConfig::default()
    };
    PiPNNBuildContext::new(config, NonZeroUsize::new(16).unwrap(), 1.2, metric, threads).unwrap()
}

#[test]
fn build_returns_bounded_adjacency() {
    let (points, dimensions) = (100, 8);
    let graph = build_typed(
        &random_data(points, dimensions),
        points,
        dimensions,
        &context(Metric::L2, true, 0),
    )
    .unwrap();

    assert_eq!(graph.len(), points);
    assert!(graph.iter().all(|neighbors| neighbors.len() <= 16));
    assert!(graph.iter().any(|neighbors| !neighbors.is_empty()));
}

#[test]
fn build_rejects_invalid_dataset_shapes() {
    let context = context(Metric::L2, true, 0);

    assert!(build_typed::<f32>(&[], 0, 8, &context).is_err());
    assert!(build_typed::<f32>(&[], 8, 0, &context).is_err());
    let error = build_typed(&[0.0f32; 7], 2, 4, &context).unwrap_err();
    assert!(error.to_string().contains("expected 8 elements"));
}

#[test]
fn build_handles_tiny_datasets() {
    let context = context(Metric::L2, true, 0);

    let one = build_typed(&[1.0f32, 2.0], 1, 2, &context).unwrap();
    assert_eq!(one, [Vec::<u32>::new()]);

    let two = build_typed(&[0.0f32, 0.0, 1.0, 0.0], 2, 2, &context).unwrap();
    assert_eq!(two.len(), 2);
    assert!(two.iter().flatten().any(|_| true));
}

#[test]
fn build_handles_duplicate_points() {
    let graph = build_typed(&[1.0f32; 20 * 4], 20, 4, &context(Metric::L2, true, 0)).unwrap();

    assert_eq!(graph.len(), 20);
}

#[test]
fn build_supports_every_metric() {
    let (points, dimensions) = (64, 8);
    let data = random_data(points, dimensions);

    for metric in [
        Metric::L2,
        Metric::Cosine,
        Metric::CosineNormalized,
        Metric::InnerProduct,
    ] {
        let graph = build_typed(&data, points, dimensions, &context(metric, true, 0)).unwrap();
        assert_eq!(graph.len(), points, "metric {metric:?}");
        assert!(
            graph.iter().any(|neighbors| !neighbors.is_empty()),
            "metric {metric:?}"
        );
    }
}

#[test]
fn build_supports_hashprune_only_output() {
    let (points, dimensions) = (64, 8);
    let graph = build_typed(
        &random_data(points, dimensions),
        points,
        dimensions,
        &context(Metric::L2, false, 0),
    )
    .unwrap();

    assert_eq!(graph.len(), points);
    assert!(graph.iter().all(|neighbors| neighbors.len() <= 16));
}

#[test]
fn build_supports_direct_deduplicated_candidates() {
    let (points, dimensions) = (100, 8);
    let data = random_data(points, dimensions);
    let config = PiPNNConfig {
        c_max: 32,
        c_min: 4,
        k: 3,
        replicas: 1,
        l_max: 0,
        num_hash_planes: 0,
        final_prune: true,
        skip_hash_prune: true,
        ..PiPNNConfig::default()
    };
    let context =
        PiPNNBuildContext::new(config, NonZeroUsize::new(16).unwrap(), 1.2, Metric::L2, 2).unwrap();

    let graph = build_typed(&data, points, dimensions, &context).unwrap();

    assert_eq!(graph.len(), points);
    assert!(graph.iter().any(|neighbors| !neighbors.is_empty()));
    for neighbors in graph {
        assert!(neighbors.len() <= 16);
        let mut unique = neighbors.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(neighbors.len(), unique.len());
    }
}

#[test]
fn build_supports_explicit_thread_count() {
    let (points, dimensions) = (64, 8);
    let graph = build_typed(
        &random_data(points, dimensions),
        points,
        dimensions,
        &context(Metric::L2, true, 2),
    )
    .unwrap();

    assert_eq!(graph.len(), points);
}
