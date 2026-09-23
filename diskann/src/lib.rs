/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Generic indexing algorithms.

// The async indexing paths spawn tasks; a runtime backend must be selected.
// Keep the failure actionable instead of erroring on unresolved imports.
#[cfg(not(any(feature = "tokio", feature = "compio")))]
compile_error!(
    "diskann requires an async runtime backend: enable the `tokio` feature \
     (on by default) or the `compio` feature."
);

pub mod error;
pub mod neighbor;
pub mod provider;
pub mod tracing;
pub mod utils;

// Internals
pub(crate) mod internal;
#[cfg(any(feature = "tokio", feature = "compio"))]
pub(crate) mod runtime;

// Index Implementations
pub mod flat;
pub mod graph;

// Top level exports.
pub use error::ann_error::{ANNError, ANNResult};

/// Returns the version of the DiskANN crate.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod test;
