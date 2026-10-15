/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Incremental IVF construction.
//!
//! The index routes points, splits oversized lists, and assigns their members to
//! replacement centroids. Providers supply routing and operation-local appends
//! through [`traits::BuildAccessor`], borrowed inputs through [`region::RegionView`],
//! and regional writes through [`region::RegionUpdate`].
//!
//! Neighbor selection uses [`traits::RouterAndNearest`] before filling the unique
//! union of parents and retained neighbors. Children are fitted only from parent
//! points, then all gathered points are reassigned across every child and retained
//! neighbor centroid. This permits movement between retained neighbors as well.
//! Parent centroids are not assignment candidates; retained centroids are unchanged.
//!
//! Clustering finishes reading its inputs before recording child creation and
//! parent retirement and neighbor membership replacement through the separate
//! update object. The region's input
//! borrows end before the accessor finalizes regional writes and ordinary appends
//! together. Providers may buffer writes or apply them inside a private operation
//! context.
//! Providers choose their own storage, publication, and failure-recovery mechanisms.
//!
//! This first implementation supports batch insertion into an already initialized
//! provider, with two-means splitting and optional neighborhood reassignment.
//! Setting [`Config::reassign_count`] to [`ReassignCount::None`] preserves
//! parent-only splitting.
//! Each touched parent is split once; neighbors that become oversized are not
//! recursively split. Initialization, search, deletion, and colocation are not
//! implemented.

mod clustering;
mod index;
pub mod partition;
pub mod pending;
pub mod region;
pub mod traits;

pub use index::{Config, ConfigError, IVFIndex, InsertStats, ReassignCount};

#[cfg(test)]
mod tests;
