/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-queue filtered search with convergence detection and support for max effort tradeoff.
//!
//! This search uses the standard `ExpandBeam` graph traversal but maintains
//! two separate queues:
//! - `scratch.best` (resizable): all discovered neighbors for graph exploration
//! - `scratch.filtered_results`: only neighbors passing the filter predicate
//!
//! Convergence occurs when enough filtered results are found and the closest
//! unexplored candidate is worse than the worst filtered result.

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
        search::{
            record::NoopSearchRecord,
            scratch::PriorityQueueConfiguration,
        },
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::{BuildQueryComputer, DataProvider},
    utils::VectorId,
};

/// Parameters for two-queue filtered search.
///
/// This search extends standard graph search by maintaining a separate results
/// queue for filter-passing nodes. All neighbors are explored regardless of
/// filter status, but only matching nodes contribute to convergence.
#[derive(Debug)]
pub struct TwoQueueSearch<'q, InternalId> {
    /// Base graph search parameters (k, ef/l_value, beam_width).
    pub inner: Knn,
    /// Filter evaluator for determining node matches.
    pub filter: &'q dyn QueryLabelProvider<InternalId>,
    /// Maximum number of hops before stopping search.
    pub max_candidates: usize,
}

/// Describes why the two-queue search terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoQueueTermination {
    /// The search explored all reachable candidates without hitting any limit.
    Exhausted,
    /// The search reached the `max_candidates` hop limit.
    MaxCandidates,
    /// Enough filtered results were found and the closest unexplored candidate
    /// was worse than the worst filtered result.
    Converged,
    /// The filter returned [`QueryVisitDecision::Terminate`].
    FilterTerminated,
}

/// Statistics returned by [`TwoQueueSearch`], wrapping the standard [`SearchStats`]
/// with the termination reason.
#[derive(Debug, Clone, Copy)]
pub struct TwoQueueStats {
    /// Standard search statistics (cmps, hops, result_count).
    pub stats: SearchStats,
    /// Why the search stopped.
    pub termination: TwoQueueTermination,
}

impl<'q, InternalId> TwoQueueSearch<'q, InternalId> {
    /// Create new two-queue search parameters.
    pub fn new(
        inner: Knn,
        filter: &'q dyn QueryLabelProvider<InternalId>,
        max_candidates: usize,
    ) -> Self {
        Self {
            inner,
            filter,
            max_candidates,
        }
    }
}

impl<'q, DP, S, T> Search<DP, S, T> for TwoQueueSearch<'q, DP::InternalId>
where
    DP: DataProvider,
    S: SearchStrategy<DP, T>,
    T: Copy + Send + Sync,
{
    type Output = TwoQueueStats;

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

            // Use resizable queue so exploration is unbounded — filtered search
            // may need to visit many nodes before finding enough matches.
            let mut scratch = SearchScratch::new(
                PriorityQueueConfiguration::Resizable(self.inner.l_value().get()),
                None,
            );

            let (stats, termination) = two_queue_search_internal(
                index.max_degree_with_slack(),
                &self,
                &mut accessor,
                &computer,
                &mut scratch,
                &mut NoopSearchRecord::new(),
            )
            .await?;

            // Post-process from filtered_results rather than scratch.best.
            // into_sorted_vec() returns ascending order (smallest distance first).
            let k = self.inner.k_value().get();
            let sorted_results = scratch.filtered_results.into_sorted_vec();
            let result_count = processor
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    sorted_results.iter().copied().take(k),
                    output,
                )
                .await
                .into_ann_result()?;

            Ok(TwoQueueStats {
                stats: stats.finish(result_count as u32),
                termination,
            })
        }
    }
}

/////////////////////////////
// Internal Implementation //
/////////////////////////////

/// Internal two-queue search implementation.
///
/// Performs filtered search by exploring all neighbors for graph traversal
/// but maintaining a separate sorted results list for filter-passing nodes.
/// Convergence is based on the quality of filtered results.
pub(crate) async fn two_queue_search_internal<I, A, T, SR>(
    max_degree_with_slack: usize,
    search_params: &TwoQueueSearch<'_, I>,
    accessor: &mut A,
    computer: &A::QueryComputer,
    scratch: &mut SearchScratch<I>,
    search_record: &mut SR,
) -> ANNResult<(InternalSearchStats, TwoQueueTermination)>
where
    I: VectorId,
    A: ExpandBeam<T, Id = I> + SearchExt,
    SR: SearchRecord<I> + ?Sized,
{
    let beam_width = search_params.inner.beam_width().get();
    let k = search_params.inner.k_value().get();
    let explore_ef = search_params.inner.l_value().get();
    let result_cap = k * 10;
    let max_candidates = search_params.max_candidates;
    let filter = search_params.filter;

    let make_stats = |scratch: &SearchScratch<I>| InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: false,
    };

    // Initialize search state with starting points.
    if scratch.visited.is_empty() {
        let start_ids = accessor.starting_points().await?;

        for id in start_ids {
            scratch.visited.insert(id);
            let element = accessor
                .get_element(id)
                .await
                .escalate("start point retrieval must succeed")?;
            let dist = computer.evaluate_similarity(element.reborrow());
            let neighbor = Neighbor::new(id, dist);
            scratch.best.insert(neighbor);
            scratch.cmps += 1;

            // Check filter on start point
            if let QueryVisitDecision::Accept(n) = filter.on_visit(neighbor) {
                scratch.filtered_results.push(n);
            }
        }
    }

    let mut neighbors = Vec::with_capacity(max_degree_with_slack);
    let mut beam_dists = Vec::with_capacity(beam_width);

    let mut termination = TwoQueueTermination::Exhausted;

    while scratch.best.has_notvisited_node_unbounded() {
        if scratch.hops as usize >= max_candidates {
            termination = TwoQueueTermination::MaxCandidates;
            break;
        }

        // Pop closest unvisited nodes (beam)
        scratch.beam_nodes.clear();
        beam_dists.clear();
        while scratch.beam_nodes.len() < beam_width
            && let Some(closest_node) = scratch.best.closest_notvisited_unbounded()
        {
            search_record.record(closest_node, scratch.hops, scratch.cmps);
            scratch.beam_nodes.push(closest_node.id);
            beam_dists.push(closest_node.distance);
        }

        // Convergence: enough filtered results and closest candidate is worse than worst
        if scratch.filtered_results.len() >= result_cap {
            let closest_dist = beam_dists[0];
            if closest_dist > scratch.filtered_results.peek().map_or(f32::MAX, |n| n.distance) {
                termination = TwoQueueTermination::Converged;
                break;
            }
        }

        // Expand via ExpandBeam
        neighbors.clear();
        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                computer,
                glue::NotInMut::new(&mut scratch.visited),
                |distance, id| neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        // Insert neighbors into explore queue, guarded by worst filtered result.
        // Only add if distance is better than worst result or explore queue is still small.
        let worst_dist = &scratch.filtered_results.peek().map_or(f32::MAX, |n| n.distance);
        for &neighbor in &neighbors {
            if neighbor.distance < *worst_dist || scratch.best.size() < explore_ef {
                scratch.best.insert(neighbor);
            }
        }

        // Apply filter to each neighbor
        for &neighbor in &neighbors {
            match filter.on_visit(neighbor) {
                QueryVisitDecision::Accept(n) => {
                    scratch.filtered_results.push(n);
                }
                QueryVisitDecision::Reject => {}
                QueryVisitDecision::Terminate => {
                    scratch.cmps += neighbors.len() as u32;
                    scratch.hops += scratch.beam_nodes.len() as u32;
                    return Ok((make_stats(scratch), TwoQueueTermination::FilterTerminated));
                }
            }
        }

        // Prune filtered_results: pop worst until at capacity
        while scratch.filtered_results.len() > result_cap {
            scratch.filtered_results.pop();
        }

        scratch.cmps += neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;
    }

    Ok((make_stats(scratch), termination))
}
