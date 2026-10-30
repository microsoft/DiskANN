/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Incremental IVF construction and search.
//!
//! The index routes points, splits oversized lists, and reassigns points between nearby
//! lists. Providers supply routing and operation-local appends through
//! [`traits::BuildAccessor`], borrowed inputs through [`region::View`], and regional
//! writes through [`region::Update`].
//!
//! Insertion runs in up to two stages. Each stage fills the lists it reads, records its
//! changes through an update, and finishes that update before the next stage begins:
//!
//! 1. Splitting replaces every oversized parent with `k` children fitted only from the
//!    parent's points, and assigns each of those points to one of its own children.
//! 2. Reassignment uses [`traits::Nearest`] to link each child to its nearest lists,
//!    other than its siblings, and lets every point stay or move to a list linked to its
//!    own. No centroid changes. Setting [`Config::reassign_count`] to
//!    [`ReassignCount::None`] skips this stage.
//!
//! Policies borrow the filled centroids and vectors; only fitted children are allocated.
//! A finished update is visible to later fills and lookups on the same accessor, so
//! children can be filled and queried, and retired parents are never returned. The
//! accessor publishes both stages together with its ordinary appends. Providers may
//! buffer writes or apply them inside a private operation context, and choose their own
//! storage, publication, and failure-recovery mechanisms.
//!
//! A query-bound [`traits::SearchAccessor`] selects and scans lists, controlling its
//! own scheduling. It can reduce independent [`traits::Accumulator`] states before
//! returning. The index writes the final top-k only after the scan succeeds.
//!
//! This first implementation supports batch insertion into an already initialized
//! provider. Each touched parent is split once; lists that become oversized during
//! reassignment are not split again. Initialization, deletion, and colocation are not
//! implemented.

pub mod index;
pub mod region;
pub mod split;
pub mod traits;

pub use index::{Config, ConfigError, IVFIndex, InsertStats, ReassignCount};

#[cfg(test)]
mod tests;
