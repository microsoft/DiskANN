/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks_iai::{
    compute_pq_iai::compute_pq_bench_iai,
    copy_aligned_data_bench_iai::benchmark_copy_aligned_data_bench_iai,
    diskann_iai::diskann_insert_bench_iai, neighbor_bench_iai::priority_queue_insert_bench_iai,
};
use iai_callgrind::{EventKind, LibraryBenchmarkConfig, RegressionConfig, main};
mod benchmarks_iai;

main!(
    config = LibraryBenchmarkConfig::default()
        .regression(
            RegressionConfig::default()
                .limits([(EventKind::Ir, 5.0), (EventKind::EstimatedCycles, 5.0)])
        );
    library_benchmark_groups =
        benchmark_copy_aligned_data_bench_iai,
        compute_pq_bench_iai,
        priority_queue_insert_bench_iai,
        diskann_insert_bench_iai,
);
