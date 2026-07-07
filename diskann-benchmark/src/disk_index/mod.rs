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
        pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            registry.register_partially_gated::<crate::inputs::disk::DiskIndexOperation>(
                "disk-index",
                "disk-index",
                "Disk Index build and search benchmarks"
            )?;

            Ok(())
        }
    }
}
