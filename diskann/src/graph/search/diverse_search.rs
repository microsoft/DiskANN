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
        glue::{SearchAccessor, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::{Attribute, DiverseAttributeQueue, NeighborQueue},
    provider::DataProvider,
};

/// Parameters for diversity-aware search.
///
/// Returns results that are diverse across a specified attribute. The attribute
/// for each candidate id is obtained via the `attribute_of` lookup supplied at
/// construction; ids without an attribute are excluded from the diverse results.
pub struct Diverse<F> {
    /// Base k-NN search parameters.
    inner: Knn,
    /// Diversity-specific parameters.
    diverse_params: DiverseSearchParams,
    /// Maps a raw internal id to its diversity attribute (or `None`).
    attribute_of: F,
}

impl<F> Diverse<F> {
    /// Create new diverse search parameters.
    ///
    /// `attribute_of` maps an internal vector id to its diversity attribute, or
    /// `None` if the id has no attribute (in which case it is excluded from the
    /// diverse results).
    pub fn new(inner: Knn, diverse_params: DiverseSearchParams, attribute_of: F) -> Self {
        Self {
            inner,
            diverse_params,
            attribute_of,
        }
    }

    /// Returns a reference to the inner k-NN search parameters.
    #[inline]
    pub fn inner(&self) -> &Knn {
        &self.inner
    }

    /// Returns a reference to the diversity-specific parameters.
    #[inline]
    pub fn diverse_params(&self) -> &DiverseSearchParams {
        &self.diverse_params
    }

    /// Create search scratch with a [`DiverseAttributeQueue`] for this search.
    fn create_scratch<DP, A>(
        &self,
        index: &DiskANNIndex<DP>,
    ) -> SearchScratch<DP::InternalId, DiverseAttributeQueue<DP::InternalId, A, F>>
    where
        DP: DataProvider,
        A: Attribute,
        F: Fn(DP::InternalId) -> Option<A> + Clone + Send + Sync,
    {
        let diverse_queue = DiverseAttributeQueue::new(
            self.inner.l_value().get(),
            self.inner.k_value(),
            self.diverse_params.diverse_results_k,
            self.attribute_of.clone(),
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

impl<'a, DP, S, T, A, F> Search<'a, DP, S, T> for Diverse<F>
where
    DP: DataProvider,
    A: Attribute,
    F: Fn(DP::InternalId) -> Option<A> + Clone + Send + Sync,
    T: Copy + Send + Sync,
    S: SearchStrategy<'a, DP, T, SearchAccessor: SearchAccessor>,
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

            let mut diverse_scratch = self.create_scratch::<DP, A>(index);

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
