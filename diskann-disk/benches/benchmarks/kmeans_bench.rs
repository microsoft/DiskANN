/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use criterion::{BatchSize, Criterion};
use diskann::{ANNError, ANNResult};
use diskann_disk::utils::{compute_vecs_l2sq, k_means_clustering};
use diskann_providers::utils::{create_thread_pool_for_bench, RayonThreadPoolRef};
use rand::Rng;
use rayon::prelude::*;

use diskann_providers::utils::ParallelIteratorInPool;

const NUM_POINTS: usize = 100000;
const DIM: usize = 4;
const NUM_CENTERS: usize = 256;
const MAX_KMEANS_REPS: usize = 12;

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
        f.iter_batched(
            || (data.clone(), centers.clone()),
            |(data_copy, mut centers_copy)| {
                k_means_clustering(
                    &data_copy,
                    NUM_POINTS,
                    DIM,
                    &mut centers_copy,
                    NUM_CENTERS,
                    MAX_KMEANS_REPS,
                    rng,
                    &mut false,
                    pool.as_ref(),
                )
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("Snrm2 Scalar", |f| {
        f.iter_batched(
            || (data.clone(), vec![0.0f32; NUM_POINTS]),
            |(data_copy, mut docs_l2sq)| {
                compute_vecs_l2sq_scalar(&mut docs_l2sq, &data_copy, DIM, pool.as_ref()).unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("Snrm2 Rust Run", |f| {
        f.iter_batched(
            || (data.clone(), vec![0.0f32; NUM_POINTS]),
            |(data_copy, mut docs_l2sq)| {
                compute_vecs_l2sq(&mut docs_l2sq, &data_copy, DIM, pool.as_ref()).unwrap();
            },
            BatchSize::SmallInput,
        )
    });
}
