/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::hint::black_box;

use diskann::{ANNError, ANNResult};
use diskann_disk::utils::{compute_vecs_l2sq, k_means_clustering};
use diskann_providers::utils::{
    create_thread_pool_for_bench, ParallelIteratorInPool, RayonThreadPool, RayonThreadPoolRef,
};
use rand::Rng;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefMutIterator},
    slice::ParallelSlice,
};

const NUM_POINTS: usize = 100000;
const DIM: usize = 4;
const NUM_CENTERS: usize = 256;
const MAX_KMEANS_REPS: usize = 12;

iai_callgrind::library_benchmark_group!(
    name = kmeans_bench_iai;
    benchmarks = benchmark_kmeans_iai, snrm2_benchmark_rust_iai, snrm2_benchmark_scalar_iai,
);

/// Original scalar implementation for comparison.
fn compute_vecs_l2sq_scalar(
    vecs_l2sq: &mut [f32],
    data: &[f32],
    dim: usize,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<()> {
    if dim == 0 {
        return Err(ANNError::log_index_error(format_args!(
            "dim must be non-zero"
        )));
    }

    let expected_data_len = vecs_l2sq.len().checked_mul(dim).ok_or_else(|| {
        ANNError::log_index_error(format_args!(
            "vecs_l2sq.len() * dim overflowed: vecs_l2sq.len() ({}) * dim ({})",
            vecs_l2sq.len(),
            dim
        ))
    })?;

    if data.len() != expected_data_len {
        return Err(ANNError::log_index_error(format_args!(
            "data.len() ({}) should be vecs_l2sq.len() ({}) * dim ({})",
            data.len(),
            vecs_l2sq.len(),
            dim
        )));
    }

    if dim < 5 {
        for (vec_l2sq, chunk) in vecs_l2sq.iter_mut().zip(data.chunks_exact(dim)) {
            *vec_l2sq = chunk.iter().map(|v| v * v).sum();
        }
    } else {
        vecs_l2sq
            .par_iter_mut()
            .zip(data.par_chunks_exact(dim))
            .for_each_in_pool(pool, |(vec_l2sq, chunk)| {
                *vec_l2sq = chunk.iter().map(|v| v * v).sum();
            });
    }
    Ok(())
}

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
    let centers: Vec<f32> = vec![0.0; NUM_CENTERS * DIM];
    let rng = &mut diskann_providers::utils::create_rnd_from_seed(42);

    let data_copy = data.clone();
    let mut centers_copy = centers.clone();
    k_means_clustering(
        black_box(&data_copy),
        NUM_POINTS,
        DIM,
        black_box(&mut centers_copy),
        NUM_CENTERS,
        MAX_KMEANS_REPS,
        rng,
        &mut false,
        pool.as_ref(),
    )
    .unwrap();
    black_box(centers_copy);
}

#[iai_callgrind::library_benchmark(setup = setup_data)]
pub fn snrm2_benchmark_rust_iai((data, pool): (Vec<f32>, RayonThreadPool)) {
    let mut docs_l2sq = vec![0.0; NUM_POINTS];
    compute_vecs_l2sq(&mut docs_l2sq, black_box(&data), DIM, pool.as_ref()).unwrap();
    black_box(docs_l2sq);
}

#[iai_callgrind::library_benchmark(setup = setup_data)]
pub fn snrm2_benchmark_scalar_iai((data, pool): (Vec<f32>, RayonThreadPool)) {
    let mut docs_l2sq = vec![0.0; NUM_POINTS];
    compute_vecs_l2sq_scalar(&mut docs_l2sq, black_box(&data), DIM, pool.as_ref()).unwrap();
    black_box(docs_l2sq);
}
