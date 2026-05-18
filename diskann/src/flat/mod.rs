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
pub mod index;
pub mod iterator;
pub mod strategy;

pub use index::{FlatIndex, SearchStats};
pub use iterator::{FlatIterator, Iterated};
pub use strategy::{DistancesUnordered, SearchStrategy};

#[cfg(test)]
mod test;
