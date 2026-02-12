/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Label-filtered search using multi-hop expansion.

use std::fmt::Debug;

use diskann_utils::future::{AssertSend, SendFuture};
use diskann_utils::Reborrow;
use diskann_vector::PreprocessedDistanceFunction;
use hashbrown::HashSet;

use super::{dispatch::SearchDispatch, record::SearchRecord, scratch::SearchScratch};
use crate::{
    ANNResult,
    error::{ErrorExt, IntoANNResult},
    graph::{
        SearchParams,
        glue::{self, ExpandBeam, HybridPredicate, Predicate, PredicateMut, SearchExt, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, InternalSearchStats, QueryLabelProvider, QueryVisitDecision, SearchStats},
        search::record::NoopSearchRecord,
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::{BuildQueryComputer, DataProvider},
    utils::{IntoUsize, VectorId},
};

use super::graph_search::GraphSearch;

/// Parameters for label-filtered search using multi-hop expansion.
///
/// This search extends standard graph search by expanding through non-matching
/// nodes to find matching neighbors. More efficient than flat search when the
/// matching subset is reasonably large.
pub struct MultihopSearch<'q, InternalId> {
    /// Base graph search parameters.
    pub inner: GraphSearch,
    /// Label evaluator for determining node matches.
    pub label_evaluator: &'q dyn QueryLabelProvider<InternalId>,
}

impl<InternalId: Debug> Debug for MultihopSearch<'_, InternalId> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultihopSearch")
            .field("inner", &self.inner)
            .field("label_evaluator", self.label_evaluator)
            .finish()
    }
}

impl<'q, InternalId> MultihopSearch<'q, InternalId> {
    /// Create new multihop search parameters.
    pub fn new(
        inner: GraphSearch,
        label_evaluator: &'q dyn QueryLabelProvider<InternalId>,
    ) -> Self {
        Self { inner, label_evaluator }
    }
}

impl<'q, DP, S, T, O, OB> SearchDispatch<DP, S, T, O, OB> for MultihopSearch<'q, DP::InternalId>
where
    DP: DataProvider,
    T: Sync + ?Sized,
    S: SearchStrategy<DP, T, O>,
    O: Send,
    OB: SearchOutputBuffer<O> + Send,
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
        let params = SearchParams {
            k_value: self.inner.k,
            l_value: self.inner.l,
            beam_width: self.inner.beam_width,
        };
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context)
                .into_ann_result()?;
            let computer = accessor.build_query_computer(query).into_ann_result()?;

            let start_ids = accessor.starting_points().await?;

            let mut scratch = index.search_scratch(params.l_value, start_ids.len());

            let stats = multihop_search_internal(
                    index.max_degree_with_slack(),
                    &params,
                    &mut accessor,
                    &computer,
                    &mut scratch,
                    &mut NoopSearchRecord::new(),
                    self.label_evaluator,
                )
                .await?;

            let result_count = strategy
                .post_processor()
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    scratch.best.iter().take(params.l_value.into_usize()),
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

/// A predicate that checks if an item is not in the visited set AND matches the label filter.
///
/// Used during two-hop expansion to filter neighbors based on both visitation
/// status and label matching criteria.
pub struct NotInMutWithLabelCheck<'a, K>
where
    K: VectorId,
{
    visited_set: &'a mut HashSet<K>,
    query_label_evaluator: &'a dyn QueryLabelProvider<K>,
}

impl<'a, K> NotInMutWithLabelCheck<'a, K>
where
    K: VectorId,
{
    /// Construct a new `NotInMutWithLabelCheck` around `visited_set`.
    pub fn new(
        visited_set: &'a mut HashSet<K>,
        query_label_evaluator: &'a dyn QueryLabelProvider<K>,
    ) -> Self {
        Self {
            visited_set,
            query_label_evaluator,
        }
    }
}

impl<K> Predicate<K> for NotInMutWithLabelCheck<'_, K>
where
    K: VectorId,
{
    fn eval(&self, item: &K) -> bool {
        !self.visited_set.contains(item) && self.query_label_evaluator.is_match(*item)
    }
}

impl<K> PredicateMut<K> for NotInMutWithLabelCheck<'_, K>
where
    K: VectorId,
{
    fn eval_mut(&mut self, item: &K) -> bool {
        if self.query_label_evaluator.is_match(*item) {
            return self.visited_set.insert(*item);
        }
        false
    }
}

impl<K> HybridPredicate<K> for NotInMutWithLabelCheck<'_, K> where K: VectorId {}

/// Internal multihop search implementation.
///
/// Performs label-filtered search by expanding through non-matching nodes
/// to find matching neighbors within two hops.
pub(crate) async fn multihop_search_internal<I, A, T, SR>(
    max_degree_with_slack: usize,
    search_params: &SearchParams,
    accessor: &mut A,
    computer: &A::QueryComputer,
    scratch: &mut SearchScratch<I>,
    search_record: &mut SR,
    query_label_evaluator: &dyn QueryLabelProvider<I>,
) -> ANNResult<InternalSearchStats>
where
    I: VectorId,
    A: ExpandBeam<T, Id = I> + SearchExt,
    T: ?Sized,
    SR: SearchRecord<I> + ?Sized,
{
    let beam_width = search_params.beam_width.unwrap_or(1);

    // Helper to build the final stats from scratch state.
    let make_stats = |scratch: &SearchScratch<I>| InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: false,
    };

    // Initialize search state if not already initialized.
    // This allows paged search to call multihop_search_internal multiple times
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
    let mut two_hop_neighbors = Vec::with_capacity(max_degree_with_slack);
    let mut candidates_two_hop_expansion = Vec::with_capacity(max_degree_with_slack);

    while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
        scratch.beam_nodes.clear();
        one_hop_neighbors.clear();
        candidates_two_hop_expansion.clear();
        two_hop_neighbors.clear();

        // In this loop we are going to find the beam_width number of nodes that are closest to the query.
        // Each of these nodes will be a frontier node.
        while scratch.best.has_notvisited_node() && scratch.beam_nodes.len() < beam_width {
            let closest_node = scratch.best.closest_notvisited();
            search_record.record(closest_node, scratch.hops, scratch.cmps);
            scratch.beam_nodes.push(closest_node.id);
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
            match query_label_evaluator.on_visit(neighbor) {
                QueryVisitDecision::Accept(accepted) => {
                    scratch.best.insert(accepted);
                }
                QueryVisitDecision::Reject => {
                    // Rejected nodes: still add to two-hop expansion so we can traverse through them
                    candidates_two_hop_expansion.push(neighbor);
                }
                QueryVisitDecision::Terminate => {
                    scratch.cmps += one_hop_neighbors.len() as u32;
                    scratch.hops += scratch.beam_nodes.len() as u32;
                    return Ok(make_stats(scratch));
                }
            }
        }

        scratch.cmps += one_hop_neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;

        // sort the candidates for two-hop expansion by distance to query point
        candidates_two_hop_expansion.sort_unstable_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // limit the number of two-hop candidates to avoid too many expansions
        candidates_two_hop_expansion.truncate(max_degree_with_slack / 2);

        // Expand each two-hop candidate: if its neighbor is a match, compute its distance
        // to the query and insert into `scratch.visited`
        // If it is not a match, do nothing
        let two_hop_expansion_candidate_ids: Vec<I> =
            candidates_two_hop_expansion.iter().map(|n| n.id).collect();

        accessor
            .expand_beam(
                two_hop_expansion_candidate_ids.iter().copied(),
                computer,
                NotInMutWithLabelCheck::new(&mut scratch.visited, query_label_evaluator),
                |distance, id| {
                    two_hop_neighbors.push(Neighbor::new(id, distance));
                },
            )
            .await?;

        // Next, insert the new matches into `scratch.best` and increment stats counters
        two_hop_neighbors
            .iter()
            .for_each(|neighbor| scratch.best.insert(*neighbor));

        scratch.cmps += two_hop_neighbors.len() as u32;
        scratch.hops += two_hop_expansion_candidate_ids.len() as u32;
    }

    Ok(make_stats(scratch))
}
