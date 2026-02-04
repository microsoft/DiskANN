/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Benchmark Tools for Build Operations
//!
//! Implementations integrate with build infrastructure via the [`Build`] trait. This trait
//! is sufficient to use the [`build`] and [`build_tracked`] functions, which perform parallelized
//! builds with aggregated results.
//!
//! In contrast to search operations, the results of build ([`BuildResults`] and [`BatchResult`])
//! contain more information about the parallel task on which they were run. Less aggregation happens
//! automatically because build operations often display variance as the build progresses. For example,
//! inserts later in the cycle often take longer than those occurring earlier.
//!
//! Since builds are often long-running operations, progress reporting is supported via [`build_tracked`] with
//! the [`Progress`] and [`AsProgress`] traits.
//!
//! ## Customization Points
//!
//! The main customization point for build operations is the [`Build`] trait itself and its custom output.
//! This output is collected on a per-batch basis and will be made available in the final [`BuildResults`].
//!
//! ## Example
//!
//! The example below shows a toy implementation of [`Build`].
//!
//! ```rust
//! use std::{sync::Arc, num::NonZeroUsize};
//! use diskann_benchmark_core::build;
//!
//! /// A simple example implementation of the `Build` trait.
//! #[derive(Debug)]
//! struct Example {
//!     /// Implementations are expected to store data internally with the `Build`
//!     /// orchestrating the work distribution.
//!     num_items: usize,
//! }
//!
//! /// Example `Build::Output`.
//! #[derive(Debug, PartialEq)]
//! struct Output {
//!     /// The number of items built in this batch.
//!     num_built: usize,
//! }
//!
//! impl Output {
//!     fn new(num_built: usize) -> Self {
//!         Self { num_built }
//!     }
//! }
//!
//! impl build::Build for Example {
//!     type Output = Output;
//!     fn num_data(&self) -> usize {
//!         self.num_items
//!     }
//!
//!     async fn build(&self, range: std::ops::Range<usize>) -> diskann::ANNResult<Self::Output> {
//!         // Simulate building the items in the given range.
//!         Ok(Output::new(range.len()))
//!     }
//! }
//!
//! // Run the build.
//! let runtime = diskann_benchmark_core::tokio::runtime(1).unwrap();
//!
//! // Run a build with 100 objects, using a sequential insertion with
//! // a batchsize of 20.
//! let results = build::build(
//!     Arc::new(Example { num_items: 100 }),
//!     build::Parallelism::sequential(NonZeroUsize::new(20).unwrap()),
//!     &runtime,
//! ).unwrap();
//!
//! assert_eq!(results.output().len(), 5, "expected 5 batches of 20 items");
//! assert_eq!(results.output().iter().map(|o| o.output.num_built).sum::<usize>(), 100, "expected 100 items built");
//! ```
//!
//! # Built-In Runners
//!
//! ## Graph Index
//!
//! * [`graph::SingleInsert`] - A builder for inserting items into a [`diskann::graph::DiskANNIndex`]
//!   using [`diskann::graph::DiskANNIndex::insert`].
//!
//! * [`graph::MultiInsert`] - A builder for inserting items into a [`diskann::graph::DiskANNIndex`]
//!   using [`diskann::graph::DiskANNIndex::multi_insert`].

mod api;
pub use api::{
    AsProgress, BatchResult, Build, BuildResults, Parallelism, Progress, build, build_tracked,
};

pub mod graph;
pub mod ids;
