/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Incremental IVF construction and basic IVF-Flat search.
//!
//! The index routes points, splits oversized lists, and assigns their members to
//! replacement centroids. Providers supply centroid selection, logical list
//! operations, and borrowed working sets through [`traits::BuildAccessor`].
//!
//! Appends and replacements remain private to one build operation until it
//! finishes. This lets clustering treat prepared and stored points identically.
//! Providers choose their own storage, publication, and failure-recovery mechanisms.
//!
//! This first implementation supports initialization and batch insertion with
//! two-means splitting and read-only L2 search. Neighborhood reassignment,
//! deletion, and colocation are not implemented.

mod clustering;
mod index;
mod metrics;
pub mod pending;
mod search;
pub mod traits;
pub mod workingset;

pub use index::{Config, ConfigError, IVFIndex, InsertStats};
pub use metrics::BuildMetrics;
pub use search::{SearchProvider, SearchResult};

#[cfg(test)]
mod tests;
