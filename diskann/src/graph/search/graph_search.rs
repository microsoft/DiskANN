/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Standard graph-based ANN search.

use std::fmt::Debug;

use diskann_utils::future::{AssertSend, SendFuture};

use super::dispatch::SearchDispatch;
use crate::{
    ANNResult,
    error::IntoANNResult,
    graph::{
        glue::{SearchExt, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, SearchStats},
        search::record::NoopSearchRecord,
        search_output_buffer::SearchOutputBuffer,
    },
    provider::{BuildQueryComputer, DataProvider},
    utils::IntoUsize,
};

/// Parameters for standard graph-based ANN search.
///
/// This is the primary search mode, using the Vamana graph structure for efficient
/// approximate nearest neighbor traversal.
#[derive(Debug, Clone, Copy)]
pub struct GraphSearch {
    /// Number of results to return (k in k-NN).
    pub k: usize,
    /// Search list size - controls accuracy vs speed tradeoff.
    pub l: usize,
    /// Optional beam width for parallel graph exploration.
    pub beam_width: Option<usize>,
}

impl GraphSearch {
    /// Create new graph search parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if `l < k` or if any value is zero.
    pub fn new(
        k: usize,
        l: usize,
        beam_width: Option<usize>,
    ) -> Result<Self, super::super::SearchParamsError> {
        use super::super::SearchParamsError;

        if k > l {
            return Err(SearchParamsError::LLessThanK { l_value: l, k_value: k });
        }
        if let Some(bw) = beam_width {
            if bw == 0 {
                return Err(SearchParamsError::BeamWidthZero);
            }
        }
        if k == 0 {
            return Err(SearchParamsError::KZero);
        }
        if l == 0 {
            return Err(SearchParamsError::LZero);
        }

        Ok(Self { k, l, beam_width })
    }

    /// Create parameters with default beam width.
    pub fn new_default(k: usize, l: usize) -> Result<Self, super::super::SearchParamsError> {
        Self::new(k, l, None)
    }
}

impl From<super::super::SearchParams> for GraphSearch {
    fn from(params: super::super::SearchParams) -> Self {
        Self {
            k: params.k_value,
            l: params.l_value,
            beam_width: params.beam_width,
        }
    }
}

/// Implement SearchDispatch for SearchParams to provide backwards compatibility.
/// This treats SearchParams as an alias for GraphSearch.
impl<DP, S, T, O, OB> SearchDispatch<DP, S, T, O, OB> for super::super::SearchParams
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: SearchStrategy<DP, T, O>,
    O: Send,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
{
    type Output = SearchStats;

    fn dispatch<'a>(
        &'a mut self,
        index: &'a DiskANNIndex<DP>,
        strategy: &'a S,
        context: &'a DP::Context,
        query: &'a T,
        output: &'a mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>> {
        async move {
            let mut graph_search = GraphSearch::from(*self);
            graph_search.dispatch(index, strategy, context, query, output).await
        }
    }
}

impl<DP, S, T, O, OB> SearchDispatch<DP, S, T, O, OB> for GraphSearch
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: SearchStrategy<DP, T, O>,
    O: Send,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
{
    type Output = SearchStats;

    fn dispatch<'a>(
        &'a mut self,
        index: &'a DiskANNIndex<DP>,
        strategy: &'a S,
        context: &'a DP::Context,
        query: &'a T,
        output: &'a mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>> {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context)
                .into_ann_result()?;

            let computer = accessor.build_query_computer(query).into_ann_result()?;
            let start_ids = accessor.starting_points().await?;

            let mut scratch = index.search_scratch(self.l, start_ids.len());

            let stats = index
                .search_internal(
                    self.beam_width,
                    &start_ids,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                    &mut NoopSearchRecord::new(),
                )
                .await?;

            let result_count = strategy
                .post_processor()
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    scratch.best.iter().take(self.l.into_usize()),
                    output,
                )
                .send()
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}

//=============================================================================
// Recorded Graph Search
//=============================================================================

/// Graph search with traversal path recording.
///
/// Records the path taken during search for debugging or analysis.
pub struct RecordedGraphSearch<'r, SR: ?Sized> {
    /// Base graph search parameters.
    pub inner: GraphSearch,
    /// The recorder to capture search path.
    pub recorder: &'r mut SR,
}

impl<'r, SR: ?Sized> RecordedGraphSearch<'r, SR> {
    /// Create new recorded search parameters.
    pub fn new(inner: GraphSearch, recorder: &'r mut SR) -> Self {
        Self { inner, recorder }
    }
}

impl<'r, SR: Debug + ?Sized> Debug for RecordedGraphSearch<'r, SR> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordedGraphSearch")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl<'r, DP, S, T, O, OB, SR> SearchDispatch<DP, S, T, O, OB> for RecordedGraphSearch<'r, SR>
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: SearchStrategy<DP, T, O>,
    O: Send,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
    SR: super::record::SearchRecord<DP::InternalId> + ?Sized,
{
    type Output = SearchStats;

    fn dispatch<'a>(
        &'a mut self,
        index: &'a DiskANNIndex<DP>,
        strategy: &'a S,
        context: &'a DP::Context,
        query: &'a T,
        output: &'a mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>> {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context)
                .into_ann_result()?;

            let computer = accessor.build_query_computer(query).into_ann_result()?;
            let start_ids = accessor.starting_points().await?;

            let mut scratch = index.search_scratch(self.inner.l, start_ids.len());

            let stats = index
                .search_internal(
                    self.inner.beam_width,
                    &start_ids,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                    self.recorder,
                )
                .await?;

            let result_count = strategy
                .post_processor()
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    scratch.best.iter().take(self.inner.l.into_usize()),
                    output,
                )
                .send()
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}

//=============================================================================
// Tests
//=============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_search_validation() {
        // Valid
        assert!(GraphSearch::new(10, 100, None).is_ok());
        assert!(GraphSearch::new(10, 100, Some(4)).is_ok());
        assert!(GraphSearch::new(10, 10, None).is_ok()); // k == l is valid

        // Invalid: l < k
        assert!(GraphSearch::new(100, 10, None).is_err());

        // Invalid: zero values
        assert!(GraphSearch::new(0, 100, None).is_err());
        assert!(GraphSearch::new(10, 0, None).is_err());
        assert!(GraphSearch::new(10, 100, Some(0)).is_err());
    }
}
