/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks_iai::pipnn_kernels::pipnn_kernels;
use iai_callgrind::{EventKind, LibraryBenchmarkConfig, RegressionConfig, main};

mod benchmarks_iai;

main!(
    config = LibraryBenchmarkConfig::default()
        .regression(
            RegressionConfig::default().limits([
                (EventKind::Ir, 5.0),
                (EventKind::EstimatedCycles, 5.0),
                (EventKind::TotalRW, 5.0),
                (EventKind::L1hits, 5.0),
            ])
        );
    library_benchmark_groups = pipnn_kernels,
);
