/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Standard k-NN (k-nearest neighbor) graph-based search.

use std::{fmt::Debug, num::NonZeroUsize};

use diskann_utils::future::{AssertSend, SendFuture};
use thiserror::Error;

use super::Search;
use crate::{
    ANNError, ANNErrorKind, ANNResult,
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

/// Error type for [`KnnSearch`] parameter validation.
#[derive(Debug, Error)]
pub enum KnnSearchError {
    #[error("l_value ({l_value}) cannot be less than k_value ({k_value})")]
    LLessThanK { l_value: usize, k_value: usize },
    #[error("beam width cannot be zero")]
    BeamWidthZero,
}

impl From<KnnSearchError> for ANNError {
    fn from(err: KnnSearchError) -> Self {
        Self::new(ANNErrorKind::IndexError, err)
    }
}

/// Parameters for standard k-NN (k-nearest neighbor) graph-based search.
///
/// This is the primary search mode, using the Vamana graph structure for efficient
/// approximate nearest neighbor traversal.
#[derive(Debug, Clone, Copy)]
pub struct KnnSearch {
    /// Number of results to return (k in k-NN).
    k_value: NonZeroUsize,
    /// Search list size - controls accuracy vs speed tradeoff.
    l_value: NonZeroUsize,
    /// Optional beam width for parallel graph exploration.
    beam_width: Option<usize>,
}

impl KnnSearch {
    /// Create new k-NN search parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if `l_value < k_value` or if beam_width is zero.
    pub fn new(
        k_value: NonZeroUsize,
        l_value: NonZeroUsize,
        beam_width: Option<usize>,
    ) -> Result<Self, KnnSearchError> {
        if k_value > l_value {
            return Err(KnnSearchError::LLessThanK {
                l_value: l_value.get(),
                k_value: k_value.get(),
            });
        }
        if let Some(bw) = beam_width
            && bw == 0
        {
            return Err(KnnSearchError::BeamWidthZero);
        }

        Ok(Self {
            k_value,
            l_value,
            beam_width,
        })
    }

    /// Create parameters with default beam width.
    pub fn new_default(
        k_value: NonZeroUsize,
        l_value: NonZeroUsize,
    ) -> Result<Self, KnnSearchError> {
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

    /// Returns the optional beam width for parallel graph exploration.
    #[inline]
    pub fn beam_width(&self) -> Option<usize> {
        self.beam_width
    }
}

impl<DP, S, T, O, OB> Search<DP, S, T, O, OB> for KnnSearch
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

            let mut scratch = index.search_scratch(self.l_value.get(), start_ids.len());

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
                    scratch.best.iter().take(self.l_value.get().into_usize()),
                    output,
                )
                .send()
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}

////////////////////////
// Recorded KnnSearch //
////////////////////////

/// K-NN search with traversal path recording.
///
/// Records the path taken during search for debugging or analysis.
#[derive(Debug)]
pub struct RecordedKnnSearch<'r, SR: ?Sized> {
    /// Base k-NN search parameters.
    pub inner: KnnSearch,
    /// The recorder to capture search path.
    pub recorder: &'r mut SR,
}

impl<'r, SR: ?Sized> RecordedKnnSearch<'r, SR> {
    /// Create new recorded search parameters.
    pub fn new(inner: KnnSearch, recorder: &'r mut SR) -> Self {
        Self { inner, recorder }
    }
}

impl<'r, DP, S, T, O, OB, SR> Search<DP, S, T, O, OB> for RecordedKnnSearch<'r, SR>
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

            let mut scratch = index.search_scratch(self.inner.l_value.get(), start_ids.len());

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
                    scratch
                        .best
                        .iter()
                        .take(self.inner.l_value.get().into_usize()),
                    output,
                )
                .send()
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
        assert!(
            KnnSearch::new(
                NonZeroUsize::new(10).unwrap(),
                NonZeroUsize::new(100).unwrap(),
                None
            )
            .is_ok()
        );
        assert!(
            KnnSearch::new(
                NonZeroUsize::new(10).unwrap(),
                NonZeroUsize::new(100).unwrap(),
                Some(4)
            )
            .is_ok()
        );
        assert!(
            KnnSearch::new(
                NonZeroUsize::new(10).unwrap(),
                NonZeroUsize::new(10).unwrap(),
                None
            )
            .is_ok()
        ); // k == l is valid

        // Invalid: l < k
        assert!(
            KnnSearch::new(
                NonZeroUsize::new(100).unwrap(),
                NonZeroUsize::new(10).unwrap(),
                None
            )
            .is_err()
        );

        // Invalid: zero beam_width
        assert!(
            KnnSearch::new(
                NonZeroUsize::new(10).unwrap(),
                NonZeroUsize::new(100).unwrap(),
                Some(0)
            )
            .is_err()
        );
    }
}
