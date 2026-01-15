/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks_iai::{contains_bench_iai::*, cosine_iai::*, cosine_normalized_iai::*, l2_iai::*};
use iai_callgrind::{main, EventKind, LibraryBenchmarkConfig, RegressionConfig};
mod benchmarks_iai;
pub(crate) mod utils;

main!(
    config = LibraryBenchmarkConfig::default()
        .regression(
            RegressionConfig::default()
                .limits([(EventKind::Ir, 5.0), (EventKind::EstimatedCycles, 5.0), (EventKind::TotalRW, 5.0), (EventKind::L1hits, 5.0)])
        );
    library_benchmark_groups =
        benchmark_contains_bench_iai,
        cosine_f16_iai,
        cosine_f32_iai,
        cosine_i8_iai,
        cosine_u8_iai,
        cosine_normalized_f32_iai,
        cosine_normalized_f16_iai,
        l2_f16_iai,
        l2_f32_iai,
        l2_i8_iai,
        l2_u8_iai,
);
