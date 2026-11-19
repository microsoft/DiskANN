/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Sequential ("flat") search.
//!
//! This module is the streaming counterpart to the random-access
//! [`crate::graph::glue::SearchAccessor`] family. It is designed for backends whose natural access
//! pattern is a one-pass scan over their data -- for example append-only buffered stores or
//! on-disk shards streamed via I/O.
//!
//! # Architecture
//!
//! The search algorithm operates directly on an initialized visitor:
//!
//! | Graph (random access)                       | Flat (sequential)                          |  Shared?  |
//! | :------------------------------------       | :----------------------------------------- |:--------- |
//! | [`crate::graph::glue::SearchAccessor`]      | [`DistancesUnordered`]                     | No        |
//! | [`crate::graph::Search`]                    | [`knn_search`]                             | No        |
//! | [`crate::graph::glue::SearchPostProcess`]   | [`crate::graph::glue::SearchPostProcess`]  | Yes       |
//!
pub mod index;
pub mod strategy;
mod topk;

pub use index::{SearchOptions, SearchStats, knn_search, knn_search_with_options};
pub use strategy::DistancesUnordered;

#[cfg(test)]
mod test;
