/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Standard k-NN (k-nearest neighbor) graph-based search.

use std::{fmt::Debug, marker::PhantomData, num::NonZeroUsize};

use diskann_utils::future::SendFuture;
use thiserror::Error;

use super::{Search, SearchFor};
use crate::{
    ANNResult, convert_error,
    error::IntoANNResult,
    graph::{
        algorithms::{self, SearchStats},
        glue::{SearchAccessor, SearchPostProcess},
        search::record::NoopSearchRecord,
        search_output_buffer::SearchOutputBuffer,
    },
    utils::VectorId,
};

/// Error type for [`Knn`] parameter validation.
#[derive(Debug, Error)]
pub enum KnnSearchError {
    #[error("beam width cannot be zero")]
    BeamWidthZero,
    #[error("l_value cannot be zero")]
    LZero,
}

convert_error!(KnnSearchError);

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
/// - `l_value`: Search list size (larger values improve recall at cost of latency)
/// - `beam_width`: Optional parallel exploration width
///
/// # Example
///
/// ```ignore
/// use diskann::graph::{search::Knn, Search};
///
/// let params = Knn::new(100, None)?;
/// let stats = index.search(params, &strategy, &context, &query, &mut output).await?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Knn {
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
    /// Returns an error if `l_value` is zero,
    /// or if `beam_width` is `Some(0)`.
    pub fn new(l_value: usize, beam_width: Option<usize>) -> Result<Self, KnnSearchError> {
        let l_value = NonZeroUsize::new(l_value).ok_or(KnnSearchError::LZero)?;

        const ONE: NonZeroUsize = NonZeroUsize::new(1).unwrap();
        let beam_width = match beam_width {
            Some(bw) => NonZeroUsize::new(bw).ok_or(KnnSearchError::BeamWidthZero)?,
            None => ONE,
        };

        Ok(Self {
            l_value,
            beam_width,
        })
    }

    /// Create parameters with default beam width.
    pub fn new_default(l_value: usize) -> Result<Self, KnnSearchError> {
        Self::new(l_value, None)
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

    pub(crate) fn new_infallible(l_value: NonZeroUsize, beam_width: NonZeroUsize) -> Self {
        Self {
            l_value,
            beam_width,
        }
    }
}

impl Search for Knn {
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
    /// * `processor` - The post-processor to apply to the search results.
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
    fn search<A, O, PP, OB>(
        self,
        accessor: &mut A,
        processor: PP,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        A: SearchAccessor,
        O: Send,
        PP: SearchPostProcess<A, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let num_start_ids = accessor.num_starting_points().await?;
            let max_degree = algorithms::max_degree_from_hint(accessor.max_degree_hint());
            let mut scratch =
                algorithms::search_scratch_2(max_degree, self.l_value.get(), num_start_ids);

            let stats = algorithms::search(
                accessor,
                self.beam_width(),
                &mut scratch,
                &mut NoopSearchRecord::new(),
            )
            .await?;

            let result_count = processor
                .post_process(accessor, scratch.best.iter(), output)
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}

//////////////////
// Recorded Knn //
//////////////////

/// K-NN search with traversal path recording.
///
/// Records the path taken during search for debugging or analysis.
#[derive(Debug)]
pub struct RecordedKnn<'r, I, SR: ?Sized> {
    /// Base k-NN search parameters.
    pub inner: Knn,
    /// The recorder to capture search path.
    pub recorder: &'r mut SR,
    /// The ID type expected.
    _id_type: PhantomData<I>,
}

impl<'r, I, SR: ?Sized> RecordedKnn<'r, I, SR> {
    /// Create new recorded search parameters.
    pub fn new(inner: Knn, recorder: &'r mut SR) -> Self {
        Self {
            inner,
            recorder,
            _id_type: PhantomData,
        }
    }
}

impl<I, SR> SearchFor<I> for RecordedKnn<'_, I, SR>
where
    I: VectorId,
    SR: super::record::SearchRecord<I> + ?Sized,
{
    type Output = SearchStats;

    fn search_for<A, O, PP, OB>(
        self,
        accessor: &mut A,
        processor: PP,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        A: SearchAccessor<Id = I>,
        O: Send,
        PP: SearchPostProcess<A, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let num_start_ids = accessor.num_starting_points().await?;
            let max_degree = algorithms::max_degree_from_hint(accessor.max_degree_hint());
            let mut scratch =
                algorithms::search_scratch_2(max_degree, self.inner.l_value.get(), num_start_ids);

            let stats = algorithms::search(
                accessor,
                self.inner.beam_width(),
                &mut scratch,
                self.recorder,
            )
            .await?;

            let result_count = processor
                .post_process(accessor, scratch.best.iter(), output)
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
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
        assert!(Knn::new(100, None).is_ok());
        assert!(Knn::new(100, Some(4)).is_ok());

        // Invalid: l = 0
        assert!(matches!(Knn::new(0, None), Err(KnnSearchError::LZero)));

        // Invalid: zero beam_width
        assert!(matches!(
            Knn::new(100, Some(0)),
            Err(KnnSearchError::BeamWidthZero)
        ));
    }
}
