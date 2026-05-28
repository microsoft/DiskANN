/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Pure greedy filtered search.
//!
//! All nodes (matched and unmatched) guide navigation in `scratch.best`.
//! Matched results are tracked separately in `matched_results`.
//! No two-hop expansion. The `QueryLabelProvider` controls early termination
//! via `on_visit()` returning `Terminate`.

use diskann_utils::Reborrow;
use diskann_utils::future::SendFuture;
use diskann_vector::PreprocessedDistanceFunction;

use super::{Knn, Search, record::SearchRecord, scratch::SearchScratch};
use crate::{
    ANNResult,
    error::{ErrorExt, IntoANNResult},
    graph::{
        glue::{self, ExpandBeam, SearchExt, SearchPostProcess, SearchStrategy},
        index::{
            DiskANNIndex, InternalSearchStats, QueryLabelProvider, QueryVisitDecision, SearchStats,
        },
        search::record::NoopSearchRecord,
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::{Neighbor, NeighborPriorityQueue},
    provider::{BuildQueryComputer, DataProvider},
    utils::VectorId,
};

/// Parameters for pure greedy filtered search.
///
/// All nodes participate in greedy navigation regardless of filter match.
/// Matched results are tracked separately and returned as final output.
/// Early termination is controlled by the `QueryLabelProvider` callback.
#[derive(Debug)]
pub struct InlineFilterSearch<'q, InternalId> {
    /// Base graph search parameters.
    pub inner: Knn,
    /// Label evaluator for determining node matches and early termination.
    pub label_evaluator: &'q dyn QueryLabelProvider<InternalId>,
}

impl<'q, InternalId> InlineFilterSearch<'q, InternalId> {
    /// Create new greedy filter search parameters.
    pub fn new(inner: Knn, label_evaluator: &'q dyn QueryLabelProvider<InternalId>) -> Self {
        Self {
            inner,
            label_evaluator,
        }
    }
}

impl<'q, DP, S, T> Search<DP, S, T> for InlineFilterSearch<'q, DP::InternalId>
where
    DP: DataProvider,
    S: SearchStrategy<DP, T>,
    T: Copy + Send + Sync,
{
    type Output = SearchStats;

    fn search<O, PP, OB>(
        self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        processor: PP,
        context: &DP::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        O: Send,
        PP: for<'a> SearchPostProcess<S::SearchAccessor<'a>, T, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context)
                .into_ann_result()?;
            let computer = accessor.build_query_computer(query).into_ann_result()?;

            let start_ids = accessor.starting_points().await?;

            let mut scratch = index.search_scratch(self.inner.l_value().get(), start_ids.len());

            let stats = inline_filter_search_internal(
                index.max_degree_with_slack(),
                &self.inner,
                &mut accessor,
                &computer,
                &mut scratch,
                &mut NoopSearchRecord::new(),
                self.label_evaluator,
            )
            .await?;

            let result_count = processor
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    scratch.best.iter().take(self.inner.l_value().get()),
                    output,
                )
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}

/// Internal inline filter search implementation
/// It's a normal search implementation with an extra step
/// of saving any nodes we encounter that match the filter,
/// and returning those as the final output
pub(crate) async fn inline_filter_search_internal<I, A, T, SR>(
    max_degree_with_slack: usize,
    search_params: &Knn,
    accessor: &mut A,
    computer: &A::QueryComputer,
    scratch: &mut SearchScratch<I>,
    search_record: &mut SR,
    query_label_evaluator: &dyn QueryLabelProvider<I>,
) -> ANNResult<InternalSearchStats>
where
    I: VectorId,
    A: ExpandBeam<T, Id = I> + SearchExt,
    SR: SearchRecord<I> + ?Sized,
{
    let beam_width = search_params.beam_width().get();
    let l_search = search_params.l_value().get();

    // Helper to build the final stats from scratch state.
    let make_stats = |scratch: &SearchScratch<I>| InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: false,
    };

    // Initialize search state if not already initialized.
    // This allows paged search to call this function multiple times
    if scratch.visited.is_empty() {
        let start_ids = accessor.starting_points().await?;

        for id in start_ids {
            scratch.visited.insert(id);
            let element = accessor
                .get_element(id)
                .await
                .escalate("start point retrieval must succeed")?;
            let dist = computer.evaluate_similarity(element.reborrow());
            scratch.best.insert(Neighbor::new(id, dist));
        }
    }

    // Pre-allocate with good capacity to avoid repeated allocations
    let mut one_hop_neighbors = Vec::with_capacity(max_degree_with_slack);

    // Matched results tracked separately — scratch.best contains all nodes
    // for greedy navigation, matched_results contains only filter-matching nodes.
    // Stored as a vector and sorted once at the end for efficiency
    let mut matched_results = Vec::with_capacity(l_search * max_degree_with_slack);

    loop {
        // Check termination conditions
        if accessor.terminate_early() {
            break;
        }

        scratch.beam_nodes.clear();
        one_hop_neighbors.clear();

        // Fill beam from scratch.best (all nodes participate in navigation)
        while scratch.beam_nodes.len() < beam_width {
            let Some(closest_node) = scratch.best.closest_notvisited() else {
                break;
            };
            search_record.record(closest_node, scratch.hops, scratch.cmps);
            scratch.beam_nodes.push(closest_node.id);
        }

        // Exit if no nodes to process
        if scratch.beam_nodes.is_empty() {
            break;
        }

        // compute distances from query to one-hop neighbors, and mark them visited
        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                computer,
                glue::NotInMut::new(&mut scratch.visited),
                |distance, id| one_hop_neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        // Process one-hop neighbors based on on_visit() decision
        for neighbor in one_hop_neighbors.iter().copied() {
            let decision = query_label_evaluator.on_visit(neighbor);

            match decision {
                QueryVisitDecision::Accept(accepted) => {
                    // All nodes go into scratch.best for navigation,
                    // matched nodes also go into matched_results for final output.
                    scratch.best.insert(neighbor);
                    matched_results.push(accepted);
                }
                QueryVisitDecision::Reject => {
                    // Unmatched nodes still guide navigation
                    scratch.best.insert(neighbor);
                }
                QueryVisitDecision::Terminate => {
                    scratch.cmps += one_hop_neighbors.len() as u32;
                    scratch.hops += scratch.beam_nodes.len() as u32;
                    matched_results.sort_unstable_by(|a, b| {
                        a.distance
                            .partial_cmp(&b.distance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    matched_results.truncate(l_search);
                    let mut best = NeighborPriorityQueue::new(l_search);
                    for nbr in matched_results {
                        best.insert(nbr);
                    }
                    scratch.best = best;
                    return Ok(make_stats(scratch));
                }
            }
        }

        scratch.cmps += one_hop_neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;
    }

    // Replace scratch.best with only the matched results
    // so that post_process returns the right candidates.
    matched_results.sort_unstable_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    matched_results.truncate(l_search);
    let mut best = NeighborPriorityQueue::new(l_search);
    for nbr in matched_results {
        best.insert(nbr);
    }
    scratch.best = best;

    Ok(make_stats(scratch))
}
