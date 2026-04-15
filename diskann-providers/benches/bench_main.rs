/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks::{
    compute_pq_bench::benchmark_compute_pq,
    copy_aligned_data_bench::benchmark_copy_aligned_data,
    diskann_bench::benchmark_diskann_insert,
    neighbor_bench::{
        benchmark_priority_queue_has_notvisited_node, benchmark_priority_queue_insert,
    },
};
use criterion::{criterion_group, criterion_main};
mod benchmarks;

criterion_group!(
    benches,
    benchmark_priority_queue_insert,
    benchmark_compute_pq,
    benchmark_diskann_insert,
    benchmark_priority_queue_has_notvisited_node,
    benchmark_copy_aligned_data,
);

criterion_main!(benches);
