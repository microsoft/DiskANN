/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! RAG (Retrieval-Augmented Generation) search.
//!
//! This module provides a post-process-only RAG search that composes a standard k-NN
//! graph search with RAG-specific reranking in post-processing. The reranking uses
//! greedy orthogonalization to maximize diversity among the returned results.
//!
//! # Design
//!
//! `RagSearch` is a thin wrapper around [`KnnWith<RagSearchParams>`]. The core
//! graph traversal is delegated entirely to [`Knn`]'s `search_core`, while the
//! RAG-specific reranking (greedy orthogonalization / ridge-aware log-det) is applied
//! purely in the post-processing phase.
//!
//! Strategy implementations must implement
//! `PostProcess<RagSearchParams, Provider, T, O>` to bridge their accessor
//! to the RAG reranking logic.
//!
//! # Example
//!
//! ```ignore
//! use diskann::graph::search::RagSearch;
//! use diskann::graph::search::Knn;
//!
//! let rag = RagSearch::new(Knn::new(10, 100, None)?, 0.01, 2.0);
//! let stats = index.search(rag, &strategy, &context, &query, &mut output).await?;
//! ```

use std::num::NonZeroUsize;

use diskann_utils::future::SendFuture;

use super::{Knn, KnnWith, Search};
use crate::{
    ANNResult,
    graph::{
        glue::PostProcess,
        index::{DiskANNIndex, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    provider::DataProvider,
};

/// Parameters for RAG post-processing.
///
/// `RagSearchParams` applies diversity-maximizing reranking purely in post-processing
/// using greedy orthogonalization. This means it can be composed with **any** search algorithm
/// (e.g., [`Knn`] via [`KnnWith`]) without duplicating search logic.
///
/// The reranking algorithm scales candidate vectors by their similarity to the query raised
/// to `rag_power`, then greedily selects vectors that maximize the determinant of the
/// Gram matrix (i.e., maximize volume/diversity).
///
/// # Parameters
///
/// - `rag_eta` - Ridge regularization parameter. Use 0.0 for pure greedy orthogonalization.
/// - `rag_power` - Power to raise similarity scores to before scaling vectors.
///
/// # Usage
///
/// Strategy implementations should implement
/// `PostProcess<RagSearchParams, Provider, T, O>` to bridge their accessor
/// to the RAG reranking logic.
///
/// ```ignore
/// use diskann::graph::search::{Knn, KnnWith, RagSearchParams};
///
/// let rag_pp = RagSearchParams::new(0.01, 2.0);
/// let search = KnnWith::new(Knn::new(10, 100, None)?, rag_pp);
/// let stats = index.search(search, &strategy, &context, &query, &mut output).await?;
/// ```
#[derive(Debug, Clone)]
pub struct RagSearchParams {
    /// Ridge regularization parameter. Use 0.0 for pure greedy orthogonalization.
    pub rag_eta: f64,
    /// Power to raise similarity scores to before scaling vectors.
    pub rag_power: f64,
}

impl RagSearchParams {
    /// Create new RAG search parameters.
    pub fn new(rag_eta: f64, rag_power: f64) -> Self {
        Self { rag_eta, rag_power }
    }
}

/// RAG (Retrieval-Augmented Generation) search.
///
/// Performs a standard k-NN graph search followed by RAG-specific post-processing
/// that reranks results for diversity using greedy orthogonalization.
///
/// # Example
///
/// ```ignore
/// use diskann::graph::search::{Knn, RagSearch};
///
/// let knn = Knn::new(10, 100, None)?;
/// let rag = RagSearch::new(knn, 0.01, 2.0);
/// let stats = index.search(rag, &strategy, &context, &query, &mut output).await?;
/// ```
#[derive(Debug, Clone)]
pub struct RagSearch {
    /// The inner `KnnWith` that composes k-NN search with RAG post-processing.
    inner: KnnWith<RagSearchParams>,
}

impl RagSearch {
    /// Create new RAG search parameters.
    ///
    /// # Arguments
    ///
    /// * `knn` - Base k-NN search parameters (k, l, beam_width).
    /// * `rag_eta` - Ridge regularization parameter. Use 0.0 for pure greedy orthogonalization.
    /// * `rag_power` - Power to raise similarity scores to before scaling vectors.
    pub fn new(knn: Knn, rag_eta: f64, rag_power: f64) -> Self {
        let post_processor = RagSearchParams::new(rag_eta, rag_power);
        Self {
            inner: KnnWith::new(knn, post_processor),
        }
    }

    /// Returns a reference to the inner k-NN parameters.
    #[inline]
    pub fn knn(&self) -> &Knn {
        self.inner.inner()
    }

    /// Returns the RAG eta (ridge regularization) parameter.
    #[inline]
    pub fn rag_eta(&self) -> f64 {
        self.inner.post_processor().rag_eta
    }

    /// Returns the RAG power parameter.
    #[inline]
    pub fn rag_power(&self) -> f64 {
        self.inner.post_processor().rag_power
    }

    /// Returns the number of results to return (k in k-NN).
    #[inline]
    pub fn k_value(&self) -> NonZeroUsize {
        self.inner.inner().k_value()
    }

    /// Returns the search list size.
    #[inline]
    pub fn l_value(&self) -> NonZeroUsize {
        self.inner.inner().l_value()
    }
}

impl<DP, S, T, O, OB> Search<DP, S, T, O, OB> for RagSearch
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: PostProcess<RagSearchParams, DP, T, O>,
    O: Send,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
{
    type Output = SearchStats;

    /// Execute the RAG search on the given index.
    ///
    /// This performs a standard k-NN graph traversal followed by RAG-specific
    /// post-processing that reranks results for diversity.
    ///
    /// # Arguments
    ///
    /// * `index` - The DiskANN index to search.
    /// * `strategy` - The search strategy (must implement `PostProcess<RagSearchParams>`).
    /// * `context` - The context to pass through to providers.
    /// * `query` - The query vector for which nearest neighbors are sought.
    /// * `output` - A mutable buffer to store the search results.
    ///
    /// # Returns
    ///
    /// Returns [`SearchStats`] containing distance computations, hops, and timing.
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
    ) -> impl SendFuture<ANNResult<Self::Output>> {
        self.inner.search(index, strategy, context, query, output)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::search::knn_search::KnnSearchError;

    #[test]
    fn test_rag_search_creation() {
        let knn = Knn::new(10, 100, None).unwrap();
        let rag = RagSearch::new(knn, 0.01, 2.0);

        assert_eq!(rag.k_value().get(), 10);
        assert_eq!(rag.l_value().get(), 100);
        assert_eq!(rag.rag_eta(), 0.01);
        assert_eq!(rag.rag_power(), 2.0);
    }

    #[test]
    fn test_rag_search_zero_eta() {
        let knn = Knn::new(5, 50, Some(4)).unwrap();
        let rag = RagSearch::new(knn, 0.0, 1.0);

        assert_eq!(rag.k_value().get(), 5);
        assert_eq!(rag.l_value().get(), 50);
        assert_eq!(rag.rag_eta(), 0.0);
        assert_eq!(rag.rag_power(), 1.0);
    }

    #[test]
    fn test_rag_search_invalid_knn() {
        // k > l should fail at the Knn level
        assert!(matches!(
            Knn::new(100, 10, None),
            Err(KnnSearchError::LLessThanK { .. })
        ));
    }
}
