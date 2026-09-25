/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

cfg_if::cfg_if! {
    if #[cfg(feature = "disk-index")] {
        mod benchmarks;
        mod build;
        mod search;
        mod json_spancollector;

        /// Register disk index benchmarks when the `disk-index` feature is enabled.
        pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            benchmarks::register_benchmarks(registry)
        }
    } else {
        /// Register a stub that guides users to enable the `disk-index` feature.
        pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            registry.register_partially_gated::<crate::inputs::disk::DiskIndexOperation>(
                "disk-index",
                diskann_benchmark_runner::Features::new("disk-index"),
                "Disk index build and search",
            )?;

            Ok(())
        }
    }
}
