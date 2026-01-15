/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks_iai::aligned_file_reader_bench_iai::aligned_file_reader_bench_iai;
use iai_callgrind::{main, EventKind, LibraryBenchmarkConfig, RegressionConfig};

mod benchmarks_iai;

main!(
    config = LibraryBenchmarkConfig::default()
        .regression(
            RegressionConfig::default()
                .limits([(EventKind::Ir, 5.0), (EventKind::EstimatedCycles, 5.0)])
        );
    library_benchmark_groups =
        aligned_file_reader_bench_iai,
);
