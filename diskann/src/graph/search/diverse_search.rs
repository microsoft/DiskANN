/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Diversity-aware search.

use diskann_utils::future::SendFuture;
use hashbrown::HashSet;
use std::num::NonZeroUsize;
use thiserror::Error;

use super::{Knn, Search, record::NoopSearchRecord, scratch::SearchScratch};
use crate::{
    ANNError, ANNResult,
    error::IntoANNResult,
    graph::{
        glue::{SearchAccessor, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::{AttributeValueProvider, DiverseNeighborQueue, NeighborQueue},
    provider::DataProvider,
};

/// Error type for [`DiverseSearchParams`] parameter validation.
#[derive(Debug, Error)]
pub enum DiverseSearchError {
    #[error("total k_value cannot be zero")]
    TotalKZero,
    #[error("diverse k_value cannot be zero")]
    DiverseKZero,
    #[error("diverse k_value ({diverse_results_k}) cannot exceed total k_value ({total_k_value})")]
    DiverseKGreaterThanTotalK {
        diverse_results_k: usize,
        total_k_value: usize,
    },
}

impl From<DiverseSearchError> for ANNError {
    #[track_caller]
    fn from(err: DiverseSearchError) -> Self {
        Self::new(err)
    }
}

/// Error type for [`Diverse`] parameter validation.
#[derive(Debug, Error)]
pub enum DiverseError {
    #[error("l_value ({l_value}) must be greater than or equal to total_k_value ({total_k_value})")]
    LValueTooSmall {
        l_value: usize,
        total_k_value: usize,
    },
}

impl From<DiverseError> for ANNError {
    #[track_caller]
    fn from(err: DiverseError) -> Self {
        Self::new(err)
    }
}

// Parameters for diverse search
#[derive(Clone, Debug)]
pub struct DiverseSearchParams<P>
where
    P: crate::neighbor::AttributeValueProvider,
{
    pub diverse_attribute_id: usize,
    pub diverse_results_k: NonZeroUsize,
    pub total_k_value: NonZeroUsize,
    pub attribute_provider: std::sync::Arc<P>,
}

impl<P> DiverseSearchParams<P>
where
    P: crate::neighbor::AttributeValueProvider,
{
    pub fn new(
        diverse_attribute_id: usize,
        diverse_results_k: usize,
        total_k_value: usize,
        attribute_provider: std::sync::Arc<P>,
    ) -> Result<Self, DiverseSearchError> {
        let diverse_results_k =
            NonZeroUsize::new(diverse_results_k).ok_or(DiverseSearchError::DiverseKZero)?;
        let total_k_value =
            NonZeroUsize::new(total_k_value).ok_or(DiverseSearchError::TotalKZero)?;

        Ok(Self {
            diverse_attribute_id,
            diverse_results_k,
            total_k_value,
            attribute_provider,
        })
    }
}

/// Parameters for diversity-aware search.
///
/// Returns results that are diverse across a specified attribute.
#[derive(Debug)]
pub struct Diverse<P>
where
    P: AttributeValueProvider,
{
    /// Base k-NN search parameters.
    inner: Knn,
    /// Diversity-specific parameters.
    diverse_params: DiverseSearchParams<P>,
}

impl<P> Diverse<P>
where
    P: AttributeValueProvider,
{
    /// Create new diverse search parameters.
    pub fn new(inner: Knn, diverse_params: DiverseSearchParams<P>) -> Result<Self, DiverseError> {
        let l_value = inner.l_value().get();
        let total_k_value = diverse_params.total_k_value.get();

        if l_value < total_k_value {
            return Err(DiverseError::LValueTooSmall {
                l_value,
                total_k_value,
            });
        }

        Ok(Self {
            inner,
            diverse_params,
        })
    }

    /// Returns a reference to the inner k-NN search parameters.
    #[inline]
    pub fn inner(&self) -> &Knn {
        &self.inner
    }

    /// Returns a reference to the diversity-specific parameters.
    #[inline]
    pub fn diverse_params(&self) -> &DiverseSearchParams<P> {
        &self.diverse_params
    }

    /// Create search scratch with DiverseNeighborQueue for this search.
    fn create_scratch<DP>(
        &self,
        index: &DiskANNIndex<DP>,
    ) -> SearchScratch<DP::InternalId, DiverseNeighborQueue<P>>
    where
        DP: DataProvider,
        P: AttributeValueProvider<Id = DP::InternalId>,
    {
        let attribute_provider = self.diverse_params.attribute_provider.clone();
        let diverse_queue = DiverseNeighborQueue::new(
            self.inner.l_value().get(),
            self.diverse_params.total_k_value,
            self.diverse_params.diverse_results_k.get(),
            attribute_provider,
        );

        SearchScratch {
            best: diverse_queue,
            visited: HashSet::with_capacity(
                index.estimate_visited_set_capacity(Some(self.inner.l_value().get())),
            ),
            id_scratch: Vec::with_capacity(index.max_degree_with_slack()),
            beam_nodes: Vec::with_capacity(self.inner.beam_width().get()),
            hops: 0,
            cmps: 0,
        }
    }
}

impl<'a, DP, S, T, P> Search<'a, DP, S, T> for Diverse<P>
where
    DP: DataProvider,
    T: Copy + Send + Sync,
    S: SearchStrategy<'a, DP, T, SearchAccessor: SearchAccessor>,
    P: AttributeValueProvider<Id = DP::InternalId>,
{
    type Output = SearchStats;

    fn search<O, PP, OB>(
        self,
        index: &'a DiskANNIndex<DP>,
        strategy: &'a S,
        processor: PP,
        context: &'a DP::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        O: Send,
        PP: SearchPostProcess<S::SearchAccessor, T, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context, query)
                .into_ann_result()?;

            let mut diverse_scratch = self.create_scratch(index);

            let stats = index
                .search_internal(
                    Some(self.inner.beam_width().get()),
                    &mut accessor,
                    &mut diverse_scratch,
                    &mut NoopSearchRecord::new(),
                )
                .await?;

            // Post-process diverse results
            diverse_scratch.best.post_process();

            let result_count = processor
                .post_process(
                    &mut accessor,
                    query,
                    diverse_scratch.best.iter().take(self.inner.l_value().get()),
                    output,
                )
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}
