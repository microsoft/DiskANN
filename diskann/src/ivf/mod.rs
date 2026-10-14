/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Incremental IVF construction.
//!
//! The index routes points, splits oversized lists, and assigns their members to
//! replacement centroids. Providers supply routing and operation-local appends
//! through [`traits::BuildAccessor`], borrowed inputs through [`regionset::RegionView`],
//! and regional writes through [`regionset::RegionUpdate`].
//!
//! Clustering finishes reading its inputs before recording child creation and
//! parent retirement through the separate update object. The region's input
//! borrows end before the accessor finalizes regional writes and ordinary appends
//! together. Providers may buffer writes or apply them inside a private operation
//! context.
//! Providers choose their own storage, publication, and failure-recovery mechanisms.
//!
//! This first implementation supports batch insertion into an already initialized
//! provider, with two-means splitting of the explicitly requested parents.
//! Initialization, neighborhood reassignment, search, deletion, and colocation are
//! not implemented.

mod clustering;
mod index;
pub mod partition;
pub mod pending;
pub mod region;
pub mod traits;

pub use index::{Config, ConfigError, IVFIndex, InsertStats};

#[cfg(test)]
mod tests;
