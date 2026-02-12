/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Diversity-aware search (feature-gated).

#![cfg(feature = "experimental_diversity_search")]

use std::num::NonZeroUsize;

use diskann_utils::future::{AssertSend, SendFuture};
use hashbrown::HashSet;

use super::{dispatch::SearchDispatch, graph_search::GraphSearch, record::NoopSearchRecord, scratch::SearchScratch};
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
    utils::IntoUsize,
};

/// Parameters for diversity-aware search.
///
/// Returns results that are diverse across a specified attribute.
#[derive(Debug)]
pub struct DiverseSearch<P>
where
    P: AttributeValueProvider,
{
    /// Base graph search parameters.
    pub inner: GraphSearch,
    /// Diversity-specific parameters.
    pub diverse_params: DiverseSearchParams<P>,
}

impl<P> DiverseSearch<P>
where
    P: AttributeValueProvider,
{
    /// Create new diverse search parameters.
    pub fn new(inner: GraphSearch, diverse_params: DiverseSearchParams<P>) -> Self {
        Self { inner, diverse_params }
    }
}

impl<DP, S, T, O, OB, P> SearchDispatch<DP, S, T, O, OB> for DiverseSearch<P>
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: SearchStrategy<DP, T, O>,
    O: Send,
    OB: SearchOutputBuffer<O> + Send,
    P: AttributeValueProvider<Id = DP::InternalId>,
{
    type Output = SearchStats;

    fn dispatch<'a>(
        &'a self,
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

            let mut diverse_scratch = create_diverse_scratch(
                index,
                self.inner.l,
                self.inner.beam_width,
                &self.diverse_params,
                self.inner.k,
            );

            let stats = index
                .search_internal(
                    self.inner.beam_width,
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
                    diverse_scratch.best.iter().take(self.inner.l.into_usize()),
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
// Internal Implementation
//=============================================================================

/// Create a diverse search scratch with DiverseNeighborQueue.
///
/// # Arguments
///
/// * `index` - The DiskANN index for capacity estimation
/// * `l_value` - Search list size
/// * `beam_width` - Optional beam width for parallel exploration
/// * `diverse_params` - Diversity-specific parameters
/// * `k_value` - Number of results to return
pub(crate) fn create_diverse_scratch<DP, P>(
    index: &DiskANNIndex<DP>,
    l_value: usize,
    beam_width: Option<usize>,
    diverse_params: &DiverseSearchParams<P>,
    k_value: usize,
) -> SearchScratch<DP::InternalId, DiverseNeighborQueue<P>>
where
    DP: DataProvider,
    P: AttributeValueProvider<Id = DP::InternalId>,
{
    let attribute_provider = diverse_params.attribute_provider.clone();
    let diverse_queue = DiverseNeighborQueue::new(
        l_value,
        // SAFETY: k_value is guaranteed to be non-zero by SearchParams validation by caller
        #[allow(clippy::expect_used)]
        NonZeroUsize::new(k_value).expect("k_value must be non-zero"),
        diverse_params.diverse_results_k,
        attribute_provider,
    );

    SearchScratch {
        best: diverse_queue,
        visited: HashSet::with_capacity(index.estimate_visited_set_capacity(Some(l_value))),
        id_scratch: Vec::with_capacity(index.max_degree_with_slack()),
        beam_nodes: Vec::with_capacity(beam_width.unwrap_or(1)),
        range_frontier: std::collections::VecDeque::new(),
        in_range: Vec::new(),
        hops: 0,
        cmps: 0,
    }
}
