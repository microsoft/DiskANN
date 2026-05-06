/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Sequential ("flat") search infrastructure.
//!
//! This module is the streaming counterpart to the random-access [`crate::provider::Accessor`]
//! family. It is designed for backends whose natural access pattern is a one-pass scan over
//! their data — for example append-only buffered stores, on-disk shards streamed via I/O,
//! or any provider where random access is significantly more expensive than sequential.
//!
//! # Architecture
//!
//! The module mirrors the layering used by graph search:
//!
//! | Graph (random access)                     | Flat (sequential)                 |  Shared?  |
//! | :------------------------------------     | :-------------------------------- |:--------- |
//! | [`crate::provider::DataProvider`]         | [`crate::provider::DataProvider`] | Yes       |
//! | [`crate::graph::DiskANNIndex`]            | [`FlatIndex`]                     | No        |
//! | [`crate::provider::Accessor`]             | [`FlatIterator`]                  | No        |
//! | [`crate::graph::glue::SearchStrategy`]    | [`SearchStrategy`]                | No        |
//! | [`crate::graph::glue::SearchPostProcess`] | [`crate::graph::glue::SearchPostProcess`] | Yes |
//! | [`crate::graph::Search`]                  | [`FlatIndex::knn_search`]         | No        |
//!
//! # Hot loop
//!
//! Algorithms drive the scan via [`FlatIterator::next`] (lending iterator) or override
//! [`FlatIterator::on_elements_unordered`] when batching/prefetching wins. The default
//! implementation of `on_elements_unordered` simply loops over `next`.
//!
//! See [`FlatIndex::knn_search`] for the canonical brute-force k-NN algorithm built on these
//! primitives.

pub mod index;
pub mod iterator;
pub mod strategy;

pub use index::{FlatIndex, SearchStats};
pub use iterator::{DistancesUnordered, FlatIterator, Iterated, OnElementsUnordered};
pub use strategy::SearchStrategy;
