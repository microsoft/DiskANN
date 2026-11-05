/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use benchmarks_gungraun::{
    contains_bench_gungraun::*, cosine_gungraun::*, cosine_normalized_gungraun::*, l2_gungraun::*,
};
use gungraun::{main, Callgrind, EventKind, LibraryBenchmarkConfig};
mod benchmarks_gungraun;
pub(crate) mod utils;

main!(
    config = LibraryBenchmarkConfig::default()
        .tool(Callgrind::default().soft_limits([(EventKind::Ir, 5.0), (EventKind::EstimatedCycles, 5.0), (EventKind::TotalRW, 5.0), (EventKind::L1hits, 5.0)]));

    library_benchmark_groups =
        benchmark_contains_bench_gungraun,
        cosine_f16_gungraun,
        cosine_f32_gungraun,
        cosine_i8_gungraun,
        cosine_u8_gungraun,
        cosine_normalized_f32_gungraun,
        cosine_normalized_f16_gungraun,
        l2_f16_gungraun,
        l2_f32_gungraun,
        l2_i8_gungraun,
        l2_u8_gungraun,
);
