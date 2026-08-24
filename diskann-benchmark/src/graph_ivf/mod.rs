/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

cfg_if::cfg_if! {
    if #[cfg(feature = "graph-ivf")] {
        mod benchmarks;
        mod build;
        mod element;
        mod online;
        mod search;
        mod streaming;

        /// Register graph-IVF benchmarks when the `graph-ivf` feature is enabled.
        pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            benchmarks::register_benchmarks(registry)
        }
    } else {
        /// Register a stub that guides users to enable the `graph-ivf` feature.
        pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            registry.register_partially_gated::<crate::inputs::graph_ivf::GraphIvfOperation>(
                "graph-ivf",
                diskann_benchmark_runner::Features::new("graph-ivf"),
                "Graph-IVF build and search",
            )?;

            Ok(())
        }
    }
}
