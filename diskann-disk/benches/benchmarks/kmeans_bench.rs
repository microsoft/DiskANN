/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use criterion::Criterion;
use diskann_providers::utils::{
    RayonThreadPool, compute_vecs_l2sq, create_thread_pool, create_thread_pool_for_bench, k_means_clustering
};
use diskann_quantization::algorithms::kmeans::{lloyds::lloyds, plusplus::kmeans_plusplus_into};
use diskann_utils::views::MatrixBase;
use rand::Rng;

const NUM_POINTS: usize = 100000;
const NUM_CENTERS: usize = 256;
const MAX_KMEANS_REPS: usize = 12;

fn benchmark_kmeans_with_params(c: &mut Criterion, dim: usize, num_threads: usize) {
    let rng = &mut diskann_providers::utils::create_rnd_from_seed(42);
    let pool = create_thread_pool(num_threads).unwrap();
    let data: Vec<f32> = (0..NUM_POINTS * dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let centers: Vec<f32> = vec![0.0; NUM_CENTERS * dim];

    let mut group = c.benchmark_group("kmeans-computation");
    group.sample_size(50);

    group.bench_function(
        format!("K-Means from diskann-providers (dim={}, threads={})", dim, num_threads),
        |f| {
            f.iter(|| {
                let data_copy = data.clone();
                let mut centers_copy = centers.clone();
                k_means_clustering(
                    &data_copy,
                    NUM_POINTS,
                    dim,
                    &mut centers_copy,
                    NUM_CENTERS,
                    MAX_KMEANS_REPS,
                    rng,
                    &mut false,
                    &pool,
                )
            })
        },
    );

    group.finish();
}

pub fn benchmark_kmeans(c: &mut Criterion) {
    // Benchmark with varying num_threads (DIM=768)
    for num_threads in [2, 4] {
        benchmark_kmeans_with_params(c, 768, num_threads);
    }
}

/// compute_vecs_l2sq benchmark
fn snrm2_benchmark_rust(data: &[f32], num_points: usize, dim: usize, pool: &RayonThreadPool) {
    let mut docs_l2sq = vec![0.0; num_points];
    compute_vecs_l2sq(&mut docs_l2sq, data, num_points, dim, pool).unwrap();
}
