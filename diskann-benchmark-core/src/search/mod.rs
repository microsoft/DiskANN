/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Benchmark Tools for Search Operations.
//!
//! Implementations integrate with search infrastructure via the [`Search`] trait, which
//! defines a loose collection of input parameters and output results. An implementation of
//! [`Search`] is sufficient to use the [`search`] function, which performs a parallelized
//! search with aggregated results.
//!
//! When benchmarking search operations, it can be useful to run the batch search multiple
//! times to obtain a distribution of performance results. This is facilitated by [`search_all`]
//! which automatically runs the batch search for a configurable number of iterations and
//! returns an aggregated result. Custom result aggregation is supported via the [`Aggregate`]
//! trait.
//!
//! ## Customization Points
//!
//! The search API provides several opportunities for customization while still using parts
//! of the predefined infrastructure. The list below summarizes these customization points
//! from the lowest level (with the most control and least reuse) to the highest level.
//!
//! * [`Search`]: The exact mechanics of query search are implementation defined. If
//!   additional output metrics are desired, these can be captured in [`Search::Output`],
//!   which can be just about any arbitrary type.
//!
//! * [`Aggregate`]: When using [`search_all`], custom aggregators are allowed independently
//!   from the [`Search`] implementation. This provides some flexibility in result collection
//!   while reusing a specific [`Search`] implementation.
//!
//!   On the other hand, if [`Search`] is defined with custom output, then a new aggregator
//!   will likely be required.
//!
//! ## Example
//!
//! The example below shows a toy implementation of [`Search`].
//!
//! ```rust
//! use std::{sync::Arc, num::NonZeroUsize};
//! use diskann_benchmark_core::search;
//!
//! /// A simple example implementation of the `Search` trait.
//! #[derive(Debug)]
//! struct Example {
//!     /// Implementations are expected to store queries internally with the `Search`
//!     /// infrastructure requesting the IDs to search over.
//!     num_queries: usize,
//! }
//!
//! /// Example `Search::Parameters`.
//! #[derive(Debug, Clone)]
//! struct Parameters {
//!     /// The number of IDs to return per-search. This is meant solely for example purposes.
//!     num_ids: usize,
//! }
//!
//! /// Example `Search::Output`.
//! #[derive(Debug, PartialEq)]
//! struct Output {
//!     /// The input for the search. This is meant solely for example purposes.
//!     index: usize,
//! }
//!
//! impl Output {
//!     fn new(index: usize) -> Self {
//!         Self { index }
//!     }
//! }
//!
//! impl search::Search for Example {
//!     type Id = usize;
//!     type Parameters = Parameters;
//!     type Output = Output;
//!
//!     /// Return the number of queries contained in `self`. The search infrastructure will
//!     /// generate search requests for all indices in `0..self.num_queries()`.
//!     fn num_queries(&self) -> usize {
//!         self.num_queries
//!     }
//!
//!     fn id_count(&self, parameters: &Self::Parameters) -> search::IdCount {
//!         search::IdCount::Dynamic(NonZeroUsize::new(parameters.num_ids))
//!     }
//!
//!    async fn search<O>(
//!        &self,
//!        parameters: &Self::Parameters,
//!        buffer: &mut O,
//!        index: usize,
//!    ) -> diskann::ANNResult<Self::Output>
//!    where
//!        O: diskann::graph::SearchOutputBuffer<Self::Id> + Send
//!    {
//!        use diskann::graph::SearchOutputBuffer;
//!
//!        // Fill the buffer with `index`.
//!        buffer.extend((0..parameters.num_ids).map(|_| (index, 0.0f32)));
//!        Ok(Output::new(index))
//!    }
//! }
//!
//! // Run Search
//! let runtime = diskann_benchmark_core::tokio::runtime(1).unwrap();
//!
//! // Search over `Example` that contains 4 queries.
//! let results = search::search(
//!     Arc::new(Example { num_queries: 4 }),
//!     Parameters { num_ids: 3 },
//!     NonZeroUsize::new(1).unwrap(),
//!     &runtime
//! ).unwrap();
//!
//! // Number of results is equal to the number of queries.
//! assert_eq!(results.len(), 4);
//! assert_eq!(&*results.output(), &[0, 1, 2, 3].map(Output::new));
//! ```
//!
//! # Built-in Runners
//!
//! ## Graph Index
//!
//! * [`graph::KNN`]: K-nearest neighbors search for [`diskann::graph::DiskANNIndex`].
//! * [`graph::Range`]: Range search for [`diskann::graph::DiskANNIndex`].
//! * [`graph::MultiHop`]: Multi-hop filtered search for [`diskann::graph::DiskANNIndex`].

pub(crate) mod ids;
pub use ids::ResultIds;

mod api;
pub use api::{Aggregate, Id, IdCount, Run, Search, SearchResults, Setup, search, search_all};

pub mod graph;
