/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Filtered search using multi-hop expansion.

use std::{collections::HashSet, num::NonZeroUsize};

use diskann_utils::future::SendFuture;

use super::{FilteredSearch, Knn, record::SearchRecord, scratch::SearchScratch};
use crate::{
    ANNResult,
    error::IntoANNResult,
    graph::{
        algorithms::{self, SearchStats},
        glue::{self, FilteredAccessor, SearchPostProcess},
        search::record::NoopSearchRecord,
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::{self, Neighbor},
    utils::VectorId,
};

/// Parameters for Filtered search using multi-hop expansion.
///
/// This search extends standard graph search by expanding through non-matching
/// nodes to find matching neighbors. More efficient than flat search when the
/// matching subset is reasonably large.
#[derive(Debug, Clone)]
pub struct MultihopFilterSearch {
    /// Base graph search parameters.
    inner: Knn,
    /// The maximum number of two hop candidates considered for each iteration.
    max_two_hop_candidates: Option<NonZeroUsize>,
}

impl MultihopFilterSearch {
    /// Create new multihop search parameters.
    ///
    /// This is parameterized by:
    ///
    /// * `knn`: Control the beam width and search-L for one-hop candidate aggregation.
    /// * `max_two_hop_candidates`: The maximum number of additional two-hop candidates to
    ///   search for additional filter-satifying nodes.
    ///
    ///   The number of nodes accessed during this phase is bounded by the graph's maximum
    ///   degree `M` and may access roughly `M^2` nodes.
    ///
    ///   As such, tuning this parameter can vary performance dramatically.
    ///
    ///   This algorithm was originally written with this parameter set to `M / 2` and that
    ///   is the recommended setting.
    ///
    ///   Setting this value to `None` puts no limit on the number of candidates.
    pub fn new(inner: Knn, max_two_hop_candidates: Option<NonZeroUsize>) -> Self {
        Self {
            inner,
            max_two_hop_candidates,
        }
    }

    /// Return the [`Knn`] setting for multi-hop filtered search.
    pub fn knn(&self) -> &Knn {
        &self.inner
    }

    /// Return the maximum number of two-hop candidates used for additional exploration.
    pub fn max_two_hop_candidates(&self) -> Option<NonZeroUsize> {
        self.max_two_hop_candidates
    }
}

impl<I> FilteredSearch<I> for MultihopFilterSearch
where
    I: VectorId,
{
    type Output = SearchStats;

    fn filtered_search<A, O, PP, OB>(
        self,
        accessor: &mut A,
        processor: PP,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        A: FilteredAccessor<Id = I>,
        O: Send,
        PP: SearchPostProcess<A, O> + Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let num_starting_points = accessor.num_starting_points().await?;
            let max_degree = algorithms::max_degree_from_hint(accessor.max_degree_hint());
            let mut scratch = algorithms::search_scratch_2(
                max_degree,
                self.knn().l_value().get(),
                num_starting_points,
            );

            let ret = multihop_search_internal(
                max_degree.get(),
                &self,
                accessor,
                &mut scratch,
                &mut NoopSearchRecord::new(),
            )
            .await?;

            // NOTE: The start point filter here filters out start points that were
            // explicitly rejected by the FilteredAccessor - not start points in general.
            let result_count = processor
                .post_process(
                    accessor,
                    scratch
                        .best
                        .iter()
                        .filter(|n| !ret.rejected_start_points.contains(n.id()))
                        .take(self.inner.l_value().get()),
                    output,
                )
                .await
                .into_ann_result()?;

            let stats = SearchStats {
                cmps: ret.cmps,
                hops: ret.hops,
                result_count: result_count as u32,
                range_search_second_round: false,
            };

            Ok(stats)
        }
    }
}

/////////////////////////////
// Internal Implementation //
/////////////////////////////

#[derive(Debug)]
struct Ret<I> {
    cmps: u32,
    hops: u32,
    rejected_start_points: HashSet<I>,
}

/// Internal multihop search implementation.
///
/// Performs filtered search by expanding through non-matching nodes to find matching
/// neighbors within two hops.
async fn multihop_search_internal<I, A, SR>(
    max_degree: usize,
    search_params: &MultihopFilterSearch,
    accessor: &mut A,
    scratch: &mut SearchScratch<I>,
    search_record: &mut SR,
) -> ANNResult<Ret<I>>
where
    I: VectorId,
    A: FilteredAccessor<Id = I>,
    SR: SearchRecord<I> + ?Sized,
{
    let beam_width = search_params.knn().beam_width().get();

    // It's possible for start points to be rejected by the `FilteredAccessor`. To deal with
    // this, we still use them as standard graph entry points, but track the rejection in
    // this hash set for later filtering.
    let mut rejected_start_points = HashSet::new();

    accessor
        .start_point_distances(|id, distance| {
            if id.is_reject() {
                rejected_start_points.insert(id.into_inner());
            }

            scratch.visited.insert(id.into_inner());
            scratch
                .best
                .insert(Neighbor::new(id.into_inner(), distance));
        })
        .await?;

    // Pre-allocate with good capacity to avoid repeated allocations
    let mut one_hop_neighbors = Vec::with_capacity(max_degree);
    let mut two_hop_neighbors = Vec::with_capacity(max_degree);
    let mut candidates_two_hop_expansion = Vec::with_capacity(max_degree);

    while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
        scratch.beam_nodes.clear();
        one_hop_neighbors.clear();
        candidates_two_hop_expansion.clear();
        two_hop_neighbors.clear();

        // In this loop we are going to find the beam_width number of nodes that are closest
        // to the query. Each of these nodes will be a frontier node.
        while scratch.beam_nodes.len() < beam_width
            && let Some(closest_node) = scratch.best.closest_notvisited()
        {
            search_record.record(closest_node, scratch.hops, scratch.cmps);
            scratch.beam_nodes.push(*closest_node.id());
        }

        // compute distances from query to one-hop neighbors, and mark them visited
        accessor
            .expand_beam_filtered(
                scratch.beam_nodes.iter().copied(),
                glue::NotInMut::new(&mut scratch.visited),
                |id, distance| one_hop_neighbors.push((id, distance)),
            )
            .await?;

        // Process one-hop neighbors based on on_visit() decision
        for (choice, distance) in one_hop_neighbors.iter().copied() {
            match choice {
                glue::Decision::Accept(id) => {
                    scratch
                        .best
                        .insert(Neighbor::new(id.into_inner(), distance));
                }
                glue::Decision::Reject(id) => {
                    candidates_two_hop_expansion.push(Neighbor::new(id.into_inner(), distance));
                }
            }
        }

        scratch.cmps += one_hop_neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;

        // sort the candidates for two-hop expansion by distance to query point
        candidates_two_hop_expansion.sort_unstable_by(neighbor::ord::fast_distance);

        // limit the number of two-hop candidates to avoid too many expansions
        if let Some(max_two_hop_candidates) = search_params.max_two_hop_candidates {
            candidates_two_hop_expansion.truncate(max_two_hop_candidates.get());
        }

        // Expand each two-hop candidate: if its neighbor is a match, compute its distance
        // to the query and insert into `scratch.visited`
        // If it is not a match, do nothing
        let two_hop_expansion_candidate_ids: Vec<I> = candidates_two_hop_expansion
            .iter()
            .map(|n| *n.id())
            .collect();

        accessor
            .expand_beam_accept_only(
                two_hop_expansion_candidate_ids.iter().copied(),
                glue::NotInMut::new(&mut scratch.visited),
                |id, distance| {
                    two_hop_neighbors.push(Neighbor::new(id.into_inner(), distance));
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

    Ok(Ret {
        cmps: scratch.cmps,
        hops: scratch.hops,
        rejected_start_points,
    })
}
