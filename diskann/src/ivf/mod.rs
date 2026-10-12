/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Incremental IVF construction.
//!
//! The index routes points, splits oversized lists, and assigns their members to
//! replacement centroids. Providers supply routing and operation-local appends
//! through [`traits::BuildAccessor`], and borrowed inputs and private regional
//! edits through [`workingset::RegionSet`].
//!
//! Clustering finishes reading its inputs before recording child creation and
//! parent retirement. Consuming the region releases its input borrows and yields
//! an owned update; the accessor publishes it together with ordinary appends once.
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
pub mod traits;
pub mod workingset;

pub use index::{Config, ConfigError, IVFIndex, InsertStats};

#[cfg(test)]
mod tests;
