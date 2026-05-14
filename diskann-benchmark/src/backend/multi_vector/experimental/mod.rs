/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Researcher-authored experimental multi-vector kernels.
//!
//! See [`template`] for the full kernel-author workflow (writing a `Kernel<A>`
//! impl, adapting it via `DynQueryComputer<T>`, wiring up dispatch and
//! registration, and validating under Miri).
//!
//! New experimental kernels live in their own module file in this directory.
//! Their registration goes in [`register`] below.

use diskann_benchmark_runner::registry::Benchmarks;

mod template;

pub(super) fn register(_benchmarks: &mut Benchmarks) {
    // No experimental kernels registered by default.
    // Add `benchmarks.register_regression(...)` calls here when authoring
    // new experimental kernels.
}
