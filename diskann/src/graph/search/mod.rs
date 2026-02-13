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
//! use diskann::graph::{KnnSearch, RangeSearch, MultihopSearch, Search};
//!
//! // Standard k-NN search
//! let mut params = KnnSearch::new(10, 100, None)?;;
//! let stats = index.search(&strategy, &context, &query, &mut params, &mut output).await?;
//!
//! // Range search
//! let mut params = RangeSearch::new(100, 0.5)?;
//! let result = index.search(&strategy, &context, &query, &mut params, &mut ()).await?;
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
/// Each search type (graph search, range search, etc.) implements this trait
/// to define its complete search behavior. The [`DiskANNIndex::search`] method
/// delegates to the `dispatch` method.
pub trait Search<DP, S, T: ?Sized, O, OB: ?Sized>
where
    DP: DataProvider,
{
    /// The result type returned by this search.
    type Output;

    /// Execute the search operation with full search logic.
    fn dispatch<'a>(
        &'a mut self,
        index: &'a DiskANNIndex<DP>,
        strategy: &'a S,
        context: &'a DP::Context,
        query: &'a T,
        output: &'a mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>;
}

// Re-export search parameter types.
pub use knn_search::{KnnSearch, KnnSearchError, RecordedKnnSearch};
pub use multihop_search::MultihopSearch;
pub use range_search::{RangeSearch, RangeSearchError, RangeSearchOutput};

// Feature-gated diverse search.
#[cfg(feature = "experimental_diversity_search")]
mod diverse_search;

#[cfg(feature = "experimental_diversity_search")]
pub use diverse_search::DiverseSearch;
