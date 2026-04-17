/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks::aligned_file_reader_bench::benchmark_aligned_file_reader;
use benchmarks::kmeans_bench::benchmark_kmeans;
use criterion::{criterion_group, criterion_main};

mod benchmarks;

criterion_group!(benches, benchmark_aligned_file_reader, benchmark_kmeans);

criterion_main!(benches);
