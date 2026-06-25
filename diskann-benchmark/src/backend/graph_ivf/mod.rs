/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

cfg_if::cfg_if! {
    if #[cfg(feature = "graph-ivf")] {
        mod benchmarks;
        mod build;
        mod search;

        /// Register graph-IVF benchmarks when the `graph-ivf` feature is enabled.
        pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            benchmarks::register_benchmarks(registry)
        }
    } else {
        crate::utils::stub_impl!(
            "graph-ivf",
            inputs::graph_ivf::GraphIvfOperation
        );

        /// Register a stub that guides users to enable the `graph-ivf` feature.
        pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            imp::register("graph-ivf", registry)
        }
    }
}
