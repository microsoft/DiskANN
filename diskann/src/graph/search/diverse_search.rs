/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Diversity-aware search.

use diskann_utils::future::SendFuture;
use hashbrown::HashSet;

use super::{Knn, Search, record::NoopSearchRecord, scratch::SearchScratch};
use crate::{
    ANNResult,
    error::IntoANNResult,
    graph::{
        DiverseSearchParams,
        glue::{SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::{AttributeValueProvider, DiverseNeighborQueue, NeighborQueue},
    provider::DataProvider,
};

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
    pub fn new(inner: Knn, diverse_params: DiverseSearchParams<P>) -> Self {
        Self {
            inner,
            diverse_params,
        }
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
            self.inner.k_value(),
            self.diverse_params.diverse_results_k,
            attribute_provider,
        );

        SearchScratch {
            best: diverse_queue,
            visited: HashSet::with_capacity(
                index.estimate_visited_set_capacity(Some(self.inner.l_value().get())),
            ),
            id_scratch: Vec::with_capacity(index.max_degree_with_slack()),
            beam_nodes: Vec::with_capacity(self.inner.beam_width().get()),
            range_frontier: std::collections::VecDeque::new(),
            in_range: Vec::new(),
            hops: 0,
            cmps: 0,
        }
    }
}

impl<'a, DP, S, T, P> Search<'a, DP, S, T> for Diverse<P>
where
    DP: DataProvider,
    T: Copy + Send + Sync,
    S: SearchStrategy<'a, DP, T>,
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
