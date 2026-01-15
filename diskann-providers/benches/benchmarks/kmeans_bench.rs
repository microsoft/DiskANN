/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use criterion::Criterion;
use diskann_providers::utils::{
    RayonThreadPool, compute_vecs_l2sq, create_thread_pool_for_bench, k_means_clustering,
};
use rand::Rng;

const NUM_POINTS: usize = 100000;
const DIM: usize = 4;
const NUM_CENTERS: usize = 256;
const MAX_KMEANS_REPS: usize = 12;

pub fn benchmark_kmeans(c: &mut Criterion) {
    let rng = &mut diskann_providers::utils::create_rnd_from_seed(42);
    let pool = create_thread_pool_for_bench();
    let data: Vec<f32> = (0..NUM_POINTS * DIM)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let centers: Vec<f32> = vec![0.0; NUM_CENTERS * DIM];

    let mut group = c.benchmark_group("kmeans-computation");
    group.sample_size(50);

    group.bench_function("K-Means Rust Run", |f| {
        f.iter(|| {
            let data_copy = data.clone();
            let mut centers_copy = centers.clone();
            k_means_clustering(
                &data_copy,
                NUM_POINTS,
                DIM,
                &mut centers_copy,
                NUM_CENTERS,
                MAX_KMEANS_REPS,
                rng,
                &mut false,
                &pool,
            )
        })
    });

    group.bench_function("Snrm2 Rust Run", |f| {
        f.iter(|| {
            let data_copy = data.clone();
            snrm2_benchmark_rust(&data_copy, NUM_POINTS, DIM, &pool);
        })
    });
}

/// compute_vecs_l2sq benchmark
fn snrm2_benchmark_rust(data: &[f32], num_points: usize, dim: usize, pool: &RayonThreadPool) {
    let mut docs_l2sq = vec![0.0; num_points];
    compute_vecs_l2sq(&mut docs_l2sq, data, num_points, dim, pool).unwrap();
}
