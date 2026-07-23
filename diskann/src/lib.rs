/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod error;
pub mod neighbor;
pub mod provider;
pub mod tracing;
pub mod utils;

// Internals
pub(crate) mod internal;

// Index Implementations
pub mod flat;
pub mod graph;
pub mod ivf;

// Top level exports.
pub use error::ann_error::{ANNError, ANNErrorKind, ANNResult};

/// Returns the version of the DiskANN crate.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod test;
