/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::iter::successors;

use criterion::{BenchmarkId, Criterion, black_box};
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
fn fib() -> impl Iterator<Item = i32> {
    successors(Some((1, 1)), |&(a, b)| Some((b, a + b))).map(|(a, _)| a)
}
pub fn benchmark_chunking_size_closest_centers_performance(c: &mut Criterion) {
    let fib = fib();
    let mut group = c.benchmark_group("test_chunk_size_fib");
    for chunk_size in fib.dropping(1) {
        if chunk_size > 256_000 {
            group.bench_with_input(
                BenchmarkId::from_parameter(chunk_size),
                &chunk_size,
                |b, &size| {
                    b.iter(|| mock_blas_call(black_box(size)));
                },
            );
            break;
        }
        // group.throughput(Throughput::Bytes(chunk_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &chunk_size,
            |b, &size| {
                b.iter(|| mock_blas_call(black_box(size)));
            },
        );
    }
    group.finish();
}
