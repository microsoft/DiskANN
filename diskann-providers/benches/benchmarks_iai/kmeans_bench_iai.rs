/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_providers::utils::{compute_vecs_l2sq, create_thread_pool_for_bench};
use diskann_quantization::algorithms::kmeans::{lloyds::lloyds, plusplus::kmeans_plusplus_into};
use diskann_utils::views::MatrixBase;
use rand::Rng;

const NUM_POINTS: usize = 100000;
const DIM: usize = 4;
const NUM_CENTERS: usize = 256;
const MAX_KMEANS_REPS: usize = 12;

iai_callgrind::library_benchmark_group!(
    name = kmeans_bench_iai;
    benchmarks = benchmark_kmeans_iai, snrm2_benchmark_rust_iai,
);

fn setup_data() -> Vec<f32> {
    let rng = &mut diskann_providers::utils::create_rnd_from_seed(42);
    let data: Vec<f32> = (0..NUM_POINTS * DIM)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();

    data
}
#[iai_callgrind::library_benchmark(setup = setup_data)]
pub fn benchmark_kmeans_iai(data: Vec<f32>) {
    let rng = &mut diskann_providers::utils::create_rnd_from_seed(42);

    let data_copy = data.clone();
    let mut centers_copy: Vec<f32> = vec![0.0; NUM_CENTERS * DIM];
    let data_view = MatrixBase::try_from(data_copy.as_slice(), NUM_POINTS, DIM).unwrap();
    let mut centers_view =
        MatrixBase::try_from(centers_copy.as_mut_slice(), NUM_CENTERS, DIM).unwrap();
    let _ = kmeans_plusplus_into(centers_view.as_mut_view(), data_view, rng);
    lloyds(data_view, centers_view.as_mut_view(), MAX_KMEANS_REPS);

    let data_copy = data.clone();
    snrm2_benchmark_rust(&data_copy, NUM_POINTS, DIM);
}

#[iai_callgrind::library_benchmark(setup = setup_data)]
pub fn snrm2_benchmark_rust_iai(data: Vec<f32>) {
    let data_copy = data.clone();
    snrm2_benchmark_rust(&data_copy, NUM_POINTS, DIM);
}

/// compute_vecs_l2sq benchmark
pub fn snrm2_benchmark_rust(data: &[f32], num_points: usize, dim: usize) {
    let mut docs_l2sq = vec![0.0; num_points];
    let pool = create_thread_pool_for_bench();
    compute_vecs_l2sq(&mut docs_l2sq, data, num_points, dim, &pool).unwrap();
}
