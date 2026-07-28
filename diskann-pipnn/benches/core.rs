/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{convert::Infallible, hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use diskann::{
    graph::{
        config::{self, MaxDegree},
        prune::{self, Policy},
        Config,
    },
    neighbor::Neighbor,
};
use diskann_pipnn::{build_graph, PiPNNBuildContext, PiPNNConfig};
use diskann_utils::views::MatrixView;
use diskann_vector::{distance::Metric, DistanceFunction};

const DIMENSIONS: usize = 128;
const POINTS: usize = 1_024;
const DEGREE: usize = 64;

fn fixed_data(rows: usize, columns: usize) -> Vec<f32> {
    (0..rows * columns)
        .map(|index| {
            ((index.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) % 2_003) as f32 - 1_001.0)
                / 1_001.0
        })
        .collect()
}

fn graph_config(degree: usize) -> Config {
    config::Builder::new_with(
        degree,
        MaxDegree::same(),
        72,
        Metric::L2.into(),
        |builder| {
            builder.alpha(1.2);
        },
    )
    .build()
    .unwrap()
}

fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
}

fn benchmark_stage_focused_builds(c: &mut Criterion) {
    let data = fixed_data(POINTS, DIMENSIONS);
    let view = MatrixView::try_from(data.as_slice(), POINTS, DIMENSIONS).unwrap();
    let pool = pool();
    let mut group = c.benchmark_group("pipnn/core");
    group.throughput(Throughput::Elements(POINTS as u64));

    // These use the public build boundary so the benchmark does not widen the
    // production API. Each workload suppresses unrelated work where possible.
    let scenarios = [
        (
            "partition-heavy-full-build",
            PiPNNConfig {
                c_max: 64,
                c_min: 16,
                p_samp: 0.1,
                fanout: vec![3, 2],
                k: 1,
                replicas: 1,
            },
            POINTS,
        ),
        (
            "single-leaf-candidates-full-build",
            PiPNNConfig {
                c_max: POINTS,
                c_min: 1,
                p_samp: 0.01,
                fanout: vec![1],
                k: 2,
                replicas: 1,
            },
            POINTS,
        ),
        (
            "overfull-finalization-full-build",
            PiPNNConfig {
                c_max: POINTS,
                c_min: 1,
                p_samp: 0.01,
                fanout: vec![1],
                k: 96,
                replicas: 1,
            },
            DEGREE,
        ),
    ];

    for (name, config, degree) in scenarios {
        let graph = graph_config(degree);
        let context = PiPNNBuildContext::new(config, &graph, Metric::L2, &pool).unwrap();
        group.bench_function(name, |bencher| {
            bencher.iter(|| black_box(build_graph(view, &context).unwrap()));
        });
    }
    group.finish();
}

fn benchmark_repeated_robust_prune(c: &mut Criterion) {
    let data = fixed_data(128, DIMENSIONS);
    let view = MatrixView::try_from(data.as_slice(), 128, DIMENSIONS).unwrap();
    let distance = <f32 as diskann_vector::distance::DistanceProvider<f32>>::distance_comparer(
        Metric::L2,
        Some(DIMENSIONS),
    );
    let source = view.row(0);
    let mut candidates = (1..128_u32)
        .map(|id| {
            Neighbor::new(
                id,
                distance.evaluate_similarity(source, view.row(id as usize)),
            )
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    let policy = Policy::new(DEGREE, 1.2, Metric::L2.into(), false);
    let mut scratch = prune::Scratch::new();
    let mut cache = Vec::new();

    let mut group = c.benchmark_group("vamana/robust-prune");
    group.throughput(Throughput::Elements(candidates.len() as u64));
    group.bench_function("127-to-64/reused-scratch", |bencher| {
        bencher.iter_batched_ref(
            || candidates.clone(),
            |candidates| {
                scratch.candidates_mut().clear();
                scratch.candidates_mut().append(candidates);
                let candidate_count = scratch.candidates_mut().len();
                let mut context = scratch.as_context(candidate_count);
                prune::robust_prune(
                    &mut context,
                    policy,
                    &mut cache,
                    Some,
                    |left, right| {
                        Ok::<_, Infallible>(distance.evaluate_similarity(
                            view.row(*left as usize),
                            view.row(*right as usize),
                        ))
                    },
                    |_| false,
                )
                .unwrap();
                black_box(scratch.neighbors());
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = benchmark_stage_focused_builds, benchmark_repeated_robust_prune
}
criterion_main!(benches);
