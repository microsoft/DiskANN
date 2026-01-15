/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks::aligned_file_reader_bench::benchmark_aligned_file_reader;
use criterion::{criterion_group, criterion_main};

mod benchmarks;

criterion_group!(benches, benchmark_aligned_file_reader);

criterion_main!(benches);
