/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Incremental IVF construction and search.
//!
//! The index routes points, splits oversized lists, and assigns their members to
//! replacement centroids. Providers supply routing and operation-local appends
//! through [`traits::BuildAccessor`], borrowed inputs through [`region::View`],
//! and regional writes through [`region::Update`].
//!
//! Neighbor selection uses [`traits::Nearest`] to build a split plan that records
//! each parent's neighborhood, then fills the unique union of parents and retained
//! neighbors. Children are fitted only from parent points. Each parent's points
//! are then reassigned to the nearest of its children and its linked neighbors, and
//! each neighbor's points stay or move to the nearest child of a parent linked to it.
//! Parent centroids are not assignment candidates; retained centroids are unchanged.
//! Regional splitting owns fitting, assignment, and updates as one operation.
//! The built-in two-means implementation fits each parent independently, then assigns
//! each point within its list's neighborhood. Child centroids are owned;
//! retained centroids and source vectors remain borrowed.
//!
//! Clustering finishes reading its inputs before recording child creation and
//! parent retirement and neighbor membership deltas through the separate
//! update object. The region's input
//! borrows end before the accessor finalizes regional writes and ordinary appends
//! together. Providers may buffer writes or apply them inside a private operation
//! context.
//! Providers choose their own storage, publication, and failure-recovery mechanisms.
//!
//! A query-bound [`traits::SearchAccessor`] selects and scans lists, controlling its
//! own scheduling. It can reduce independent [`traits::Accumulator`] states before
//! returning. The index writes the final top-k only after the scan succeeds.
//!
//! This first implementation supports batch insertion into an already initialized
//! provider, with two-means splitting and optional neighborhood reassignment.
//! Setting [`Config::reassign_count`] to [`ReassignCount::None`] preserves
//! parent-only splitting.
//! Each touched parent is split once; neighbors that become oversized are not
//! recursively split.
//! Initialization, deletion, and colocation are not implemented.

pub mod index;
pub mod region;
pub mod split;
pub mod traits;

pub use index::{Config, ConfigError, IVFIndex, InsertStats, ReassignCount};

#[cfg(test)]
mod tests;
