/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Diversity-aware search.

use diskann_utils::future::{AssertSend, SendFuture};
use hashbrown::HashSet;

use super::{Knn, Search, record::NoopSearchRecord, scratch::SearchScratch};
use crate::{
    ANNResult,
    error::IntoANNResult,
    graph::{
        DiverseSearchParams,
        glue::{SearchExt, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::{AttributeValueProvider, DiverseNeighborQueue, NeighborQueue},
    provider::{BuildQueryComputer, DataProvider},
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
            beam_nodes: Vec::with_capacity(self.inner.beam_width().map_or(1, |nz| nz.get())),
            range_frontier: std::collections::VecDeque::new(),
            in_range: Vec::new(),
            hops: 0,
            cmps: 0,
        }
    }
}

impl<DP, S, T, O, OB, P> Search<DP, S, T, O, OB> for Diverse<P>
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: SearchStrategy<DP, T, O>,
    O: Send,
    OB: SearchOutputBuffer<O> + Send,
    P: AttributeValueProvider<Id = DP::InternalId>,
{
    type Output = SearchStats;

    fn search(
        &mut self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        context: &DP::Context,
        query: &T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>> {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context)
                .into_ann_result()?;

            let computer = accessor.build_query_computer(query).into_ann_result()?;
            let start_ids = accessor.starting_points().await?;

            let mut diverse_scratch = self.create_scratch(index);

            let stats = index
                .search_internal(
                    self.inner.beam_width().map(|nz| nz.get()),
                    &start_ids,
                    &mut accessor,
                    &computer,
                    &mut diverse_scratch,
                    &mut NoopSearchRecord::new(),
                )
                .await?;

            // Post-process diverse results
            diverse_scratch.best.post_process();

            let result_count = strategy
                .post_processor()
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    diverse_scratch.best.iter().take(self.inner.l_value().get()),
                    output,
                )
                .send()
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}
