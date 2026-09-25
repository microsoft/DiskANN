/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::hint::black_box;

use diskann_disk::utils::{compute_vecs_l2sq, k_means_clustering};
use diskann_providers::utils::{create_thread_pool_for_bench, RayonThreadPool};
use rand::Rng;

const NUM_POINTS: usize = 100000;
const DIM: usize = 4;
const NUM_CENTERS: usize = 256;
const MAX_KMEANS_REPS: usize = 12;

iai_callgrind::library_benchmark_group!(
    name = kmeans_bench_iai;
    benchmarks = benchmark_kmeans_iai, snrm2_benchmark_rust_iai,
);

fn setup_data() -> (Vec<f32>, RayonThreadPool) {
    let rng = &mut diskann_providers::utils::create_rnd_from_seed(42);
    let data: Vec<f32> = (0..NUM_POINTS * DIM)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let pool = create_thread_pool_for_bench();

    (data, pool)
}
#[iai_callgrind::library_benchmark(setup = setup_data)]
pub fn benchmark_kmeans_iai((data, pool): (Vec<f32>, RayonThreadPool)) {
    let mut centers: Vec<f32> = vec![0.0; NUM_CENTERS * DIM];
    let rng = &mut diskann_providers::utils::create_rnd_from_seed(42);
    k_means_clustering(
        black_box(&data),
        NUM_POINTS,
        DIM,
        black_box(&mut centers),
        NUM_CENTERS,
        MAX_KMEANS_REPS,
        rng,
        &mut false,
        pool.as_ref(),
    )
    .expect("k_means_clustering call failed");
    black_box(centers);
}

#[iai_callgrind::library_benchmark(setup = setup_data)]
pub fn snrm2_benchmark_rust_iai((data, pool): (Vec<f32>, RayonThreadPool)) {
    let mut docs_l2sq = vec![0.0; NUM_POINTS];
    compute_vecs_l2sq(&mut docs_l2sq, black_box(&data), DIM, pool.as_ref())
        .expect("compute_vecs_l2sq call failed");
    black_box(docs_l2sq);
}
