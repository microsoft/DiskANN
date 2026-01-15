/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
        use benchmarks::contains_bench::benchmark_contains;

        use benchmarks::cosine;
        use benchmarks::cosine_normalized;
        use benchmarks::l2;

        use criterion::{criterion_group, criterion_main, Criterion};
        use std::time::Duration;

        pub(crate) mod utils;

        mod benchmarks;

        criterion_group!(
            name = benches;
            config = Criterion::default()
                .sample_size(3500)
                .warm_up_time(Duration::from_secs(2))
                .measurement_time(Duration::from_secs(5))
                .nresamples(200_000);
            targets =
                benchmark_contains,
                cosine::benchmark_f32,
                cosine::benchmark_f16,
                cosine::benchmark_i8,
                cosine::benchmark_u8,
                cosine_normalized::benchmark_f32,
                cosine_normalized::benchmark_f16,
                l2::benchmark_f32,
                l2::benchmark_f16,
                l2::benchmark_i8,
                l2::benchmark_u8,
        );
        criterion_main!(benches);
    } else {
        fn main() {} // no benches for non-SIMD hardware
    }
}
