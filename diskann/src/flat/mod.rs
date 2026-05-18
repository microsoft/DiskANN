/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Sequential ("flat") search.
//!
//! This module is the streaming counterpart to the random-access
//! [`crate::provider::Accessor`] family. It is designed for backends whose natural access
//! pattern is a one-pass scan over their data -- for example append-only buffered stores or
//! on-disk shards streamed via I/O.
//!
//! # Architecture
//!
//! The module mirrors the layering used by graph search:
//!
//! | Graph (random access)                       | Flat (sequential)                          |  Shared?  |
//! | :------------------------------------       | :----------------------------------------- |:--------- |
//! | [`crate::provider::DataProvider`]           | [`crate::provider::DataProvider`]          | Yes       |
//! | [`crate::graph::DiskANNIndex`]              | [`FlatIndex`]                              | No        |
//! | [`crate::graph::glue::ExpandBeam`]          | [`DistancesUnordered`]                     | No        |
//! | [`crate::graph::glue::SearchStrategy`]      | [`SearchStrategy`]                         | No        |
//! | [`crate::graph::Search`]                    | [`FlatIndex::knn_search`]                  | No        |
//!
//! The flat surface is intentionally narrower than graph search: there is no shared
//! post-processing trait, and the [`SearchStrategy`] (not the visitor) owns the
//! `QueryComputer`. See [`FlatIndex::knn_search`] for the canonical brute-force k-NN
//! algorithm built on these primitives.

pub mod index;
pub mod iterator;
pub mod strategy;

pub use index::{FlatIndex, SearchStats};
pub use iterator::{DistancesUnordered, FlatIterator, Iterated};
pub use strategy::SearchStrategy;

#[cfg(test)]
mod test;
