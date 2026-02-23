/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Standard k-NN (k-nearest neighbor) graph-based search.

use std::{fmt::Debug, num::NonZeroUsize};

use diskann_utils::future::SendFuture;
use thiserror::Error;

use super::Search;
use crate::{
    ANNError, ANNErrorKind, ANNResult,
    error::IntoANNResult,
    graph::{
        glue::{DefaultPostProcess, PostProcess, SearchExt},
        index::{DiskANNIndex, SearchStats},
        search::record::{NoopSearchRecord, SearchRecord},
        search_output_buffer::SearchOutputBuffer,
    },
    provider::{BuildQueryComputer, DataProvider},
    utils::IntoUsize,
};

/// Error type for [`Knn`] parameter validation.
#[derive(Debug, Error)]
pub enum KnnSearchError {
    #[error("l_value ({l_value}) cannot be less than k_value ({k_value})")]
    LLessThanK { l_value: usize, k_value: usize },
    #[error("beam width cannot be zero")]
    BeamWidthZero,
    #[error("k_value cannot be zero")]
    KZero,
    #[error("l_value cannot be zero")]
    LZero,
}

impl From<KnnSearchError> for ANNError {
    #[track_caller]
    fn from(err: KnnSearchError) -> Self {
        Self::new(ANNErrorKind::IndexError, err)
    }
}

/// Standard k-NN (k-nearest neighbor) graph-based search parameters.
///
/// This is the primary search type for approximate nearest neighbor queries. It performs
/// a greedy beam search over the graph, maintaining a priority queue of the best candidates
/// found so far. The search explores neighbors of promising candidates until convergence.
///
/// # Algorithm
///
/// 1. Initialize with starting points
/// 2. Compute distances from query to starting points
/// 3. Greedily expand the most promising unexplored candidate
/// 4. Add the candidate's neighbors to the frontier
/// 5. Repeat until no unexplored candidates remain within the search list
/// 6. Return the top-k results from the best candidates found
///
/// # Parameters
///
/// - `k_value`: Number of nearest neighbors to return
/// - `l_value`: Search list size (larger values improve recall at cost of latency)
/// - `beam_width`: Optional parallel exploration width
///
/// # Example
///
/// ```ignore
/// use diskann::graph::{search::Knn, Search};
///
/// let params = Knn::new(10, 100, None)?;
/// let stats = index.search(params, &strategy, &context, &query, &mut output).await?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Knn {
    /// Number of results to return (k in k-NN).
    k_value: NonZeroUsize,
    /// Search list size - controls accuracy vs speed tradeoff.
    l_value: NonZeroUsize,
    /// Beam width for parallel graph exploration (defaults to 1).
    beam_width: NonZeroUsize,
}

impl Knn {
    /// Create new k-NN search parameters.
    ///
    /// If `beam_width` is `None`, it defaults to 1.
    ///
    /// # Errors
    ///
    /// Returns an error if `k_value` is zero, `l_value` is zero,
    /// `l_value < k_value`, or if `beam_width` is `Some(0)`.
    pub fn new(
        k_value: usize,
        l_value: usize,
        beam_width: Option<usize>,
    ) -> Result<Self, KnnSearchError> {
        let k_value = NonZeroUsize::new(k_value).ok_or(KnnSearchError::KZero)?;
        let l_value = NonZeroUsize::new(l_value).ok_or(KnnSearchError::LZero)?;
        if k_value > l_value {
            return Err(KnnSearchError::LLessThanK {
                l_value: l_value.get(),
                k_value: k_value.get(),
            });
        }

        const ONE: NonZeroUsize = NonZeroUsize::new(1).unwrap();
        let beam_width = match beam_width {
            Some(bw) => NonZeroUsize::new(bw).ok_or(KnnSearchError::BeamWidthZero)?,
            None => ONE,
        };

        Ok(Self {
            k_value,
            l_value,
            beam_width,
        })
    }

    /// Create parameters with default beam width.
    pub fn new_default(k_value: usize, l_value: usize) -> Result<Self, KnnSearchError> {
        Self::new(k_value, l_value, None)
    }

    /// Returns the number of results to return (k in k-NN).
    #[inline]
    pub fn k_value(&self) -> NonZeroUsize {
        self.k_value
    }

    /// Returns the search list size.
    #[inline]
    pub fn l_value(&self) -> NonZeroUsize {
        self.l_value
    }

    /// Returns the beam width for parallel graph exploration.
    #[inline]
    pub fn beam_width(&self) -> NonZeroUsize {
        self.beam_width
    }

    /// Core search implementation shared by [`Knn`], [`RecordedKnn`], and [`KnnWith`].
    ///
    /// All k-NN search variants follow the same algorithm: create an accessor, build a
    /// query computer, run `search_internal`, then post-process. The only axes of
    /// variation are:
    ///
    /// * `recorder` — controls whether the traversal path is recorded.
    /// * `post_processor` — selects which post-processing pipeline to apply.
    async fn search_core<DP, S, T, O, OB, SR, PP>(
        &self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        output: &mut OB,
        recorder: &mut SR,
        post_processor: &PP,
    ) -> ANNResult<SearchStats>
    where
        DP: DataProvider,
        T: Sync + ?Sized,
        S: PostProcess<PP, DP, T, O>,
        O: Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
        SR: SearchRecord<DP::InternalId> + ?Sized,
        PP: Send + Sync,
    {
        let mut accessor = strategy
            .search_accessor(&index.data_provider, context)
            .into_ann_result()?;

        let computer = accessor.build_query_computer(query).into_ann_result()?;
        let start_ids = accessor.starting_points().await?;

        let mut scratch = index.search_scratch(self.l_value.get(), start_ids.len());

        let stats = index
            .search_internal(
                Some(self.beam_width.get()),
                &start_ids,
                &mut accessor,
                &computer,
                &mut scratch,
                recorder,
            )
            .await?;

        let result_count = strategy
            .post_process_with(
                post_processor,
                &mut accessor,
                query,
                &computer,
                scratch.best.iter().take(self.l_value.get().into_usize()),
                output,
            )
            .await?;

        Ok(stats.finish(result_count as u32))
    }
}

impl<DP, S, T, O, OB> Search<DP, S, T, O, OB> for Knn
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: PostProcess<DefaultPostProcess, DP, T, O>,
    O: Send,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
{
    type Output = SearchStats;

    /// Execute the k-NN search on the given index.
    ///
    /// This method executes a search using the provided `strategy` to access and process elements.
    /// It computes the similarity between the query vector and the elements in the index, traversing
    /// the graph towards the nearest neighbors according to the search parameters.
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
    /// Returns [`SearchStats`] containing:
    /// - The number of distance computations performed.
    /// - The number of hops (graph traversal steps).
    /// - Timing information for the search operation.
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
        async move {
            let mut recorder = NoopSearchRecord::new();
            self.search_core(
                index,
                strategy,
                context,
                query,
                output,
                &mut recorder,
                &DefaultPostProcess,
            )
            .await
        }
    }
}

////////////////////////
// Recorded Knn //
////////////////////////

/// K-NN search with traversal path recording.
///
/// Records the path taken during search for debugging or analysis.
#[derive(Debug)]
pub struct RecordedKnn<'r, SR: ?Sized> {
    /// Base k-NN search parameters.
    pub inner: Knn,
    /// The recorder to capture search path.
    pub recorder: &'r mut SR,
}

impl<'r, SR: ?Sized> RecordedKnn<'r, SR> {
    /// Create new recorded search parameters.
    pub fn new(inner: Knn, recorder: &'r mut SR) -> Self {
        Self { inner, recorder }
    }
}

impl<'r, DP, S, T, O, OB, SR> Search<DP, S, T, O, OB> for RecordedKnn<'r, SR>
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: PostProcess<DefaultPostProcess, DP, T, O>,
    O: Send,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
    SR: SearchRecord<DP::InternalId> + ?Sized,
{
    type Output = SearchStats;

    fn search(
        self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>> {
        async move {
            self.inner
                .search_core(
                    index,
                    strategy,
                    context,
                    query,
                    output,
                    self.recorder,
                    &DefaultPostProcess,
                )
                .await
        }
    }
}

///////////////////////
// KnnWith (generic) //
///////////////////////

/// K-NN search with a custom post-processor.
///
/// This type composes a standard [`Knn`] search with an arbitrary post-processing
/// strategy `PP`. The strategy `S` must implement `PostProcess<PP, ...>` to provide
/// the post-processing logic.
///
/// This eliminates the need to duplicate the core search loop for each post-processing
/// variant. For example, pairing `KnnWith<RagSearchParams>` gives
/// RAG-reranked results without rewriting the search algorithm.
///
/// # Example
///
/// ```ignore
/// use diskann::graph::search::{Knn, KnnWith, RagSearchParams};
///
/// let rag_pp = RagSearchParams::new(0.01, 2.0);
/// let search = KnnWith::new(Knn::new(10, 100, None)?, rag_pp);
/// let stats = index.search(search, &strategy, &context, &query, &mut output).await?;
/// ```
#[derive(Debug, Clone)]
pub struct KnnWith<PP> {
    /// Base k-NN search parameters.
    inner: Knn,
    /// The post-processor to apply after the core search.
    post_processor: PP,
}

impl<PP> KnnWith<PP> {
    /// Create a new k-NN search with custom post-processing.
    pub fn new(inner: Knn, post_processor: PP) -> Self {
        Self {
            inner,
            post_processor,
        }
    }

    /// Returns a reference to the inner k-NN parameters.
    #[inline]
    pub fn inner(&self) -> &Knn {
        &self.inner
    }

    /// Returns a reference to the post-processor.
    #[inline]
    pub fn post_processor(&self) -> &PP {
        &self.post_processor
    }
}

impl<DP, S, T, O, OB, PP> Search<DP, S, T, O, OB> for KnnWith<PP>
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: PostProcess<PP, DP, T, O>,
    O: Send,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
    PP: Send + Sync,
{
    type Output = SearchStats;

    fn search(
        self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>> {
        async move {
            let mut recorder = NoopSearchRecord::new();
            self.inner
                .search_core(
                    index,
                    strategy,
                    context,
                    query,
                    output,
                    &mut recorder,
                    &self.post_processor,
                )
                .await
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_search_validation() {
        // Valid
        assert!(Knn::new(10, 100, None).is_ok());
        assert!(Knn::new(10, 100, Some(4)).is_ok());
        assert!(Knn::new(10, 10, None).is_ok()); // k == l is valid

        // Invalid: k = 0
        assert!(matches!(Knn::new(0, 100, None), Err(KnnSearchError::KZero)));

        // Invalid: l = 0
        assert!(matches!(Knn::new(10, 0, None), Err(KnnSearchError::LZero)));

        // Invalid: l < k
        assert!(matches!(
            Knn::new(100, 10, None),
            Err(KnnSearchError::LLessThanK { .. })
        ));

        // Invalid: zero beam_width
        assert!(matches!(
            Knn::new(10, 100, Some(0)),
            Err(KnnSearchError::BeamWidthZero)
        ));
    }
}
