/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Unified search execution framework.
//!
//! This module provides the primary search interface for DiskANN. All search types
//! are represented as parameter structs that implement [`Search`], which
//! contains the complete search logic.
//!
//! # Usage
//!
//! ```ignore
//! use diskann::graph::{search::{Knn, Range, MultihopSearch}, Search};
//!
//! // Standard k-NN search
//! let params = Knn::new(10, 100, None)?;
//! let stats = index.search(params, &strategy, &context, &query, &mut output).await?;
//!
//! // Range search
//! let params = Range::new(100, 0.5)?;
//! let result = index.search(params, &strategy, &context, &query, &mut ()).await?;
//! println!("Found {} points within radius", result.ids.len());
//! ```

use diskann_utils::future::SendFuture;

use crate::{ANNResult, graph::index::DiskANNIndex, provider::DataProvider};

mod knn_search;
mod multihop_search;
mod range_search;

pub mod record;
pub(crate) mod scratch;

/// Trait for search parameter types that execute their own search logic.
///
/// This trait defines the interface for different search modes in DiskANN.
/// Each implementation encapsulates its own algorithm and parameter handling.
///
/// # Implementations
///
/// See the specific search types for detailed documentation:
/// - [`Knn`] - Standard k-nearest neighbor search
/// - [`Range`] - Range-based search within a distance radius
/// - [`Diverse`] - Diversity-aware search (feature-gated)
/// - [`MultihopSearch`] - Label-filtered search with multi-hop expansion
/// - [`RecordedKnn`] - K-NN search with path recording for debugging
pub trait Search<DP, S, T: ?Sized, O, OB: ?Sized>
where
    DP: DataProvider,
{
    /// The result type returned by this search.
    type Output;

    /// Execute the search operation with full search logic.
    ///
    /// This method executes a search using the provided `strategy` to access and process elements.
    /// It computes the similarity between the query vector and the elements in the index,
    /// finding nearest neighbors according to the search parameters.
    ///
    /// # Arguments
    ///
    /// * `index` - The DiskANN index to search.
    /// * `strategy` - The search strategy to use for accessing and processing elements.
    /// * `context` - The context to pass through to providers.
    /// * `query` - The query vector for which nearest neighbors are sought.
    /// * `output` - A mutable buffer to store the search results. Must be pre-allocated by the caller.
    ///
    /// # Returns
    ///
    /// Returns `Self::Output` which varies by search type (e.g., [`SearchStats`](super::index::SearchStats)
    /// for k-NN, [`RangeSearchOutput`] for range search).
    ///
    /// # Errors
    ///
    /// Returns an error if there is a failure accessing elements or computing distances.
    fn search(
        self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>;
}

// Re-export search parameter types.
pub use knn_search::{Knn, KnnSearchError, RecordedKnn};
pub use multihop_search::MultihopSearch;
pub use range_search::{Range, RangeSearchError, RangeSearchOutput};

// Feature-gated diverse search.
#[cfg(feature = "experimental_diversity_search")]
mod diverse_search;

#[cfg(feature = "experimental_diversity_search")]
pub use diverse_search::Diverse;
