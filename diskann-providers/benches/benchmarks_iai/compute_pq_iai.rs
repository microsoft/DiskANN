/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use diskann_providers::{
    common::AlignedBoxWithSlice,
    model::{NUM_PQ_CENTROIDS, compute_pq_distance},
    utils::{ParallelIteratorInPool, create_thread_pool_for_bench},
};
use rand::Rng;
use rayon::{prelude::IndexedParallelIterator, slice::ParallelSliceMut};

const NUM_POINTS: usize = 100000;
const MAX_DEGREE: usize = 64;
const NUM_PQ_CHUNKS: usize = 128;

iai_callgrind::library_benchmark_group!(
    name = compute_pq_bench_iai;
    benchmarks = benchmark_compute_pq_iai,
);

#[iai_callgrind::library_benchmark(setup = generate_benchmark_data)]
pub fn benchmark_compute_pq_iai(data: (Vec<u32>, AlignedBoxWithSlice<f32>, Vec<u8>)) {
    let (neighbor_vector_ids, query_centroid_l2_distance, pq_data) = data;
    let mut pq_distance_scratch: Vec<f32> = vec![0.0; MAX_DEGREE];
    let mut pq_coordinate_scratch: Vec<u8> = vec![0; NUM_PQ_CHUNKS * neighbor_vector_ids.len()];

    // Call the function being tested
    compute_pq_distance(
        &neighbor_vector_ids,
        NUM_PQ_CHUNKS,
        &query_centroid_l2_distance,
        &pq_data,
        &mut pq_coordinate_scratch,
        &mut pq_distance_scratch,
    )
    .unwrap();
}

fn generate_benchmark_data() -> (Vec<u32>, AlignedBoxWithSlice<f32>, Vec<u8>) {
    let n_pts = NUM_POINTS;
    let n_nbrs = MAX_DEGREE;
    let rng = &mut diskann_providers::utils::create_rnd_from_seed(42);
    // Generate mock input data using thread_rng
    let neighbor_vector_ids: Vec<u32> = (0..n_nbrs)
        .map(|_| rng.random_range(0..n_pts) as u32)
        .collect();

    let mut query_centroid_l2_distance =
        AlignedBoxWithSlice::new(NUM_PQ_CENTROIDS * NUM_PQ_CHUNKS, 256).unwrap();
    let vec_256 = (0..NUM_PQ_CENTROIDS)
        .map(|i| i as f32)
        .collect::<Vec<f32>>();
    // mock query_centroid_l2_distance, distance from query to each centroid i = (i) for each chunk, just for simple calculation.
    let pool = create_thread_pool_for_bench();
    query_centroid_l2_distance[0..NUM_PQ_CHUNKS * NUM_PQ_CENTROIDS]
        .par_chunks_mut(NUM_PQ_CENTROIDS)
        .enumerate()
        .for_each_in_pool(&pool, |(_, chunk)| chunk.copy_from_slice(&vec_256));

    let pq_data: Vec<u8> = (0..NUM_PQ_CHUNKS * n_pts)
        .map(|_| rng.random_range(0..256) as u8)
        .collect();

    (neighbor_vector_ids, query_centroid_l2_distance, pq_data)
}
