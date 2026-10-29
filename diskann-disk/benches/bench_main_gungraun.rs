/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks_gungraun::aligned_file_reader_bench_gungraun::aligned_file_reader_bench_gungraun;
use benchmarks_gungraun::kmeans_bench_gungraun::kmeans_bench_gungraun;
use gungraun::{main, Callgrind, EventKind, LibraryBenchmarkConfig};

mod benchmarks_gungraun;

main!(
    config = LibraryBenchmarkConfig::default()
        .tool(Callgrind::default().soft_limits([(EventKind::Ir, 5.0), (EventKind::EstimatedCycles, 5.0)]));
    library_benchmark_groups =
        aligned_file_reader_bench_gungraun,
        kmeans_bench_gungraun,
);
