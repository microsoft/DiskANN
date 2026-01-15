/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_linalg::{self, Transpose};
use itertools::Itertools;

fn mock_blas_call(chunk_size: i32) {
    let centers = vec![1.0; 256];
    let data = vec![1.0; 256_000];
    data.chunks(chunk_size as usize).for_each(|chunk| {
        let mut dist_matrix = vec![0.0; chunk.len() * 256];
        let ones_a = vec![1.0; chunk.len()];
        let ones_b = vec![1.0; 256];

        diskann_linalg::sgemm(
            Transpose::None,
            Transpose::Ordinary,
            chunk.len(),
            256,
            1,
            1.0,
            chunk,
            &ones_b,
            None,
            &mut dist_matrix,
        );
        diskann_linalg::sgemm(
            Transpose::None,
            Transpose::Ordinary,
            chunk.len(),
            256,
            1,
            1.0,
            &ones_a,
            &centers,
            Some(1.0),
            &mut dist_matrix,
        );
        diskann_linalg::sgemm(
            Transpose::None,
            Transpose::Ordinary,
            chunk.len(),
            256,
            1,
            -2.0,
            chunk,
            &centers,
            Some(1.0),
            &mut dist_matrix,
        );
    })
}

fn generate_chunk_size() -> impl Iterator<Item = i32> {
    vec![1, 64, 256, 1024, 4096, 16384].into_iter()
}

#[iai_callgrind::library_benchmark]
pub fn benchmark_chunking_size_closest_centers_performance_iai() {
    let nums = generate_chunk_size();

    for chunk_size in nums.dropping(1) {
        mock_blas_call(chunk_size);
    }
}

iai_callgrind::library_benchmark_group!(
    name = benchmark_chunking_size_closest_centers_performance_bench_iai;
    benchmarks = benchmark_chunking_size_closest_centers_performance_iai,
);
