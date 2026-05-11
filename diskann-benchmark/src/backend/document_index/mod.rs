/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Backend benchmark implementation for document index with label filters.
//!
//! This benchmark tests the DocumentInsertStrategy which enables inserting
//! Document objects (vector + attributes) into a DiskANN index.

use diskann_benchmark_runner::registry::Benchmarks;

cfg_if::cfg_if! {
    if #[cfg(feature = "document-index")] {
        mod benchmark;

        /// Register document index benchmarks when the `document-index` feature is enabled.
        pub(crate) fn register_benchmarks(registry: &mut Benchmarks) {
            benchmark::register_benchmarks(registry);
        }
    } else {
        crate::utils::stub_impl!(
            "document-index",
            inputs::document_index::DocumentIndexBuild
        );

        /// Register a stub that guides users to enable the `document-index` feature.
        pub(crate) fn register_benchmarks(registry: &mut Benchmarks) {
            imp::register("document-index", registry);
        }
    }
}
