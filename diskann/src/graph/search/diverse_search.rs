/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Diversity-aware search.

use hashbrown::HashSet;
use std::num::NonZeroUsize;
use thiserror::Error;

use super::{Knn, Search, record::NoopSearchRecord, scratch::SearchScratch};
use crate::{
    ANNError, ANNResult,
    error::IntoANNResult,
    graph::{
        algorithms::{self, SearchStats},
        glue::{SearchAccessor, SearchPostProcess},
        search::scratch,
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::{AttributeValueProvider, DiverseNeighborQueue, NeighborQueue},
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
    fn create_scratch(
        &self,
        max_degree_hint: NonZeroUsize,
    ) -> SearchScratch<P::Id, DiverseNeighborQueue<P>> {
        let attribute_provider = self.diverse_params.attribute_provider.clone();
        let diverse_queue = DiverseNeighborQueue::new(
            self.inner.l_value().get(),
            self.diverse_params.total_k_value,
            self.diverse_params.diverse_results_k.get(),
            attribute_provider,
        );

        SearchScratch {
            best: diverse_queue,
            visited: HashSet::with_capacity(scratch::estimate_node_visited_set_size(
                max_degree_hint.get(),
                self.inner.l_value().get(),
            )),
            id_scratch: Vec::with_capacity(max_degree_hint.get()),
            beam_nodes: Vec::with_capacity(self.inner.beam_width().get()),
            hops: 0,
            cmps: 0,
        }
    }
}

impl<P> Search<P::Id> for Diverse<P>
where
    P: AttributeValueProvider,
{
    type Output = SearchStats;

    async fn search<A, O, PP, OB>(
        self,
        accessor: &mut A,
        processor: PP,
        output: &mut OB,
    ) -> ANNResult<Self::Output>
    where
        A: SearchAccessor<Id = P::Id>,
        O: Send,
        PP: SearchPostProcess<A, O> + Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        let mut diverse_scratch =
            self.create_scratch(algorithms::max_degree_from_hint(accessor.max_degree_hint()));

        let stats = algorithms::search(
            accessor,
            self.inner.beam_width(),
            &mut diverse_scratch,
            &mut NoopSearchRecord::new(),
        )
        .await?;

        // Post-process diverse results
        diverse_scratch.best.post_process();

        let result_count = processor
            .post_process(
                accessor,
                diverse_scratch.best.iter().take(self.inner.l_value().get()),
                output,
            )
            .await
            .into_ann_result()?;

        Ok(stats.finish(result_count as u32))
    }
}
