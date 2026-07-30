/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

cfg_if::cfg_if! {
    if #[cfg(feature = "kmeans-comparison")] {
        mod comparison;

        pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            comparison::register(registry)
        }
    } else {
        pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            registry.register_partially_gated::<crate::inputs::kmeans::KmeansComparison>(
                "kmeans-comparison",
                diskann_benchmark_runner::Features::new("kmeans-comparison"),
                "Legacy disk and quantization K-means comparison",
            )?;
            Ok(())
        }
    }
}
