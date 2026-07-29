/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use diskann::graph::config::{self, MaxDegree};
use diskann_pipnn::{build_graph, HashPruneConfig, PiPNNBuildContext, PiPNNConfig};
use diskann_utils::views::MatrixView;
use diskann_vector::distance::Metric;

const POINTS: usize = 2_048;
const DIMENSIONS: usize = 128;

fn data() -> Vec<f32> {
    (0..POINTS * DIMENSIONS)
        .map(|index| {
            ((index.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) % 2_003) as f32 - 1_001.0)
                / 1_001.0
        })
        .collect()
}

fn benchmark_candidate_merge(c: &mut Criterion) {
    let values = data();
    let view = MatrixView::try_from(values.as_slice(), POINTS, DIMENSIONS).unwrap();
    let graph =
        config::Builder::new_with(64, MaxDegree::same(), 72, Metric::L2.into(), |builder| {
            builder.alpha(1.2);
        })
        .build()
        .unwrap();
    let config = PiPNNConfig {
        c_max: 256,
        c_min: 64,
        p_samp: 0.01,
        fanout: vec![4, 2],
        k: 3,
        replicas: 1,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let direct = PiPNNBuildContext::new(config.clone(), &graph, Metric::L2, &pool).unwrap();
    let pruned = PiPNNBuildContext::new(config.clone(), &graph, Metric::L2, &pool)
        .unwrap()
        .with_hash_prune(HashPruneConfig {
            num_hash_planes: 12,
            l_max: 64,
            final_prune: true,
        })
        .unwrap();
    let reservoir_only = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool)
        .unwrap()
        .with_hash_prune(HashPruneConfig {
            num_hash_planes: 12,
            l_max: 64,
            final_prune: false,
        })
        .unwrap();

    let mut group = c.benchmark_group("pipnn/candidate-merge");
    group.throughput(Throughput::Elements(POINTS as u64));
    for (name, context) in [
        ("direct", direct),
        ("hash-prune+robust-prune", pruned),
        ("hash-prune-only", reservoir_only),
    ] {
        group.bench_function(name, |bencher| {
            bencher.iter(|| black_box(build_graph(view, &context).unwrap()));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    targets = benchmark_candidate_merge
}
criterion_main!(benches);
