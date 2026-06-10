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
pub struct AdaptiveLGreedySearch<'q, InternalId> {
    /// Base graph search parameters.
    pub inner: Knn,
    /// Label evaluator for determining node matches and early termination.
    pub label_evaluator: &'q dyn QueryLabelProvider<InternalId>,
}

impl<'q, InternalId> AdaptiveLGreedySearch<'q, InternalId> {
    /// Create new greedy filter search parameters.
    pub fn new(inner: Knn, label_evaluator: &'q dyn QueryLabelProvider<InternalId>) -> Self {
        Self {
            inner,
            label_evaluator,
        }
    }
}

impl<'q, DP, S, T> Search<DP, S, T> for AdaptiveLGreedySearch<'q, DP::InternalId>
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

            let stats = greedy_filter_search_internal(
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

/// Internal greedy filter search implementation.
///
/// Pure greedy mode: all nodes (matched + unmatched) enter `scratch.best`
/// for navigation. Matched results are tracked separately in `matched_results`.
/// No two-hop expansion — unmatched nodes naturally participate in subsequent
/// navigation rounds.
///
/// Adaptive L: after visiting `ADAPTIVE_L_SAMPLE_COUNT` nodes, the match rate
/// is estimated and L is scaled up for low match rates:
///   match_rate ≥ 50%    → 1× L (no change, most nodes match)
///   10% ≤ match_rate < 50% → 2× L
///   match_rate < 10%    → log-scale: 2^(-log10(match_rate))
///     match_rate = 10%  (100/1000) → 2× L
///     match_rate = 1%   (10/1000)  → 4× L
///     match_rate = 0.1% (1/1000)   → 8× L
///   0 matches in sample → 16× L (maximum expansion)
///
/// With 1000 samples, the minimum observable non-zero match rate is 0.1% (1/1000),
/// so the effective multiplier range is [1×, 8×] for non-zero matches
/// and 16× for zero matches.
pub(crate) async fn greedy_filter_search_internal<I, A, T, SR>(
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
    let mut matched_results = NeighborPriorityQueue::<I>::new(l_search);

    // Adaptive L constants — defined at function scope, also used by compute_adaptive_l
    const ADAPTIVE_L_SAMPLE_COUNT: u32 = 1000;
    const ADAPTIVE_L_MAX_MULTIPLIER: f64 = 16.0;
    let mut sample_visited: u32 = 0;
    let mut sample_matched: u32 = 0;
    let mut l_adjusted = false;

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
                    matched_results.insert(accepted);
                    sample_matched += 1;
                }
                QueryVisitDecision::Reject => {
                    // Unmatched nodes still guide navigation
                    scratch.best.insert(neighbor);
                }
                QueryVisitDecision::Terminate => {
                    scratch.cmps += one_hop_neighbors.len() as u32;
                    scratch.hops += scratch.beam_nodes.len() as u32;
                    scratch.best = matched_results;
                    return Ok(make_stats(scratch));
                }
            }
            sample_visited += 1;
        }

        scratch.cmps += one_hop_neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;

        // Adaptive L: after enough samples, estimate match rate and scale L
        if !l_adjusted && sample_visited >= ADAPTIVE_L_SAMPLE_COUNT {
            l_adjusted = true;
            let new_l = compute_adaptive_l(
                l_search,
                sample_visited,
                sample_matched,
                ADAPTIVE_L_MAX_MULTIPLIER,
            );
            if new_l > l_search {
                scratch.resize(new_l);
            }
        }
    }

    // Replace scratch.best with only the matched results
    // so that post_process returns the right candidates.
    scratch.best = matched_results;

    Ok(make_stats(scratch))
}

/// Compute adaptive L based on observed match rate.
///
/// Piecewise scaling:
///   match_rate ≥ 50%   → 1× L (no change, most nodes match)
///   10% ≤ match_rate < 50%  → 2× L
///   match_rate < 10%   → log-scale: 2^(1 - log10(match_rate))
///     match_rate = 0.1  (10%)   → 2× L
///     match_rate = 0.01 (1%)    → 4× L
///     match_rate = 0.001 (0.1%) → 8× L
///   0 matches in sample → 16× L (maximum expansion)
///
/// Clamped to [1×, max_multiplier] range.
fn compute_adaptive_l(base_l: usize, visited: u32, matched: u32, max_multiplier: f64) -> usize {
    if matched == 0 || visited == 0 {
        // No matches at all — use maximum multiplier
        return (base_l as f64 * max_multiplier) as usize;
    }

    let match_rate = matched as f64 / visited as f64;

    let multiplier = if match_rate >= 0.5 {
        // ≥50% match rate: no scaling needed
        1.0
    } else if match_rate >= 0.1 {
        // 10%-50%: use 2× L
        2.0
    } else {
        // Below 10%: log-scale from 2× upward
        // match_rate=0.1 → 2×, match_rate=0.01 → 4×, match_rate=0.001 → 8×
        let neg_log10 = -match_rate.log10(); // 0.1→1, 0.01→2, 0.001→3
        2.0_f64.powf(neg_log10) // 2^1=2, 2^2=4, 2^3=8
    };

    let multiplier = multiplier.clamp(1.0, max_multiplier);
    (base_l as f64 * multiplier) as usize
}
