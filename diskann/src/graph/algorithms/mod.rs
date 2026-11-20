/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, num::NonZeroUsize};

use super::{
    Config,
    glue::{self, SearchAccessor},
    search::{
        record::SearchRecord,
        scratch::{self, PriorityQueueConfiguration, SearchScratch},
    },
};

use crate::{
    ANNResult,
    neighbor::{Neighbor, NeighborQueue},
    utils::VectorId,
};

//-------------//
// Sub-modules //
//-------------//

pub mod insert;
pub mod measure;

pub mod consolidate;
pub mod inplace_delete;

pub mod batch;

/// Statistics collected during a search operation.
///
/// This struct provides detailed metrics about the search process, including
/// the number of nodes visited, the number of distance computations performed,
/// the number of hops taken during the search, and the total number of results returned.
#[derive(Debug, Clone, Copy)]
pub struct SearchStats {
    /// The total number of distance computations performed during the search.
    pub cmps: u32,

    /// The number of hops (iterations) taken during the search traversal.
    pub hops: u32,

    /// The total number of results returned by the search.
    pub result_count: u32,

    /// Whether the range search continued to the second round
    pub range_search_second_round: bool,
}

/// Return the maximum degree from [`SearchAccessor::max_degree_hint`] or an approximate
/// default. This should be used as a sizing hint only.
pub(crate) fn max_degree_from_hint(hint: Option<NonZeroUsize>) -> NonZeroUsize {
    const FALLBACK: NonZeroUsize = NonZeroUsize::new(32).unwrap();
    hint.unwrap_or(FALLBACK)
}

// FIXME: Can we do better?
pub(crate) fn search_scratch<I>(config: &Config, l: usize, additional: usize) -> SearchScratch<I>
where
    I: VectorId,
{
    SearchScratch::new(
        PriorityQueueConfiguration::Fixed(l + additional),
        Some(scratch::estimate_node_visited_set_size(
            config.max_degree().get(),
            l,
        )),
    )
}

pub(crate) fn search_scratch_2<I>(
    degree: NonZeroUsize,
    l: usize,
    additional: usize,
) -> SearchScratch<I>
where
    I: VectorId,
{
    SearchScratch::new(
        PriorityQueueConfiguration::Fixed(l + additional),
        Some(scratch::estimate_node_visited_set_size(degree.get(), l)),
    )
}

pub(crate) fn estimate_visited_set_capacity(config: &Config, search_l: Option<usize>) -> usize {
    let effective_l = search_l.unwrap_or(config.l_build().get());
    scratch::estimate_node_visited_set_size(config.pruned_degree().get(), effective_l)
}

//--------//
// Search //
//--------//

/// Internal struct for recording search statistics.
pub(crate) struct InternalSearchStats {
    pub(crate) cmps: u32,
    pub(crate) hops: u32,
    pub(crate) range_search_second_round: bool,
}

impl InternalSearchStats {
    pub(crate) fn finish(self, result_count: u32) -> SearchStats {
        SearchStats {
            cmps: self.cmps,
            hops: self.hops,
            result_count,
            range_search_second_round: self.range_search_second_round,
        }
    }
}

pub(crate) async fn search<A, SR, Q>(
    accessor: &mut A,
    beam_width: NonZeroUsize,
    scratch: &mut SearchScratch<A::Id, Q>,
    search_record: &mut SR,
) -> ANNResult<InternalSearchStats>
where
    A: SearchAccessor,
    SR: SearchRecord<A::Id> + ?Sized,
    Q: NeighborQueue<A::Id>,
{
    // paged search can call search_internal multiple times, we only need to initialize
    // state if not already initialized.
    if scratch.visited.is_empty() {
        accessor
            .start_point_distances(|id, distance| {
                scratch.visited.insert(id);
                scratch.best.insert(Neighbor::new(id, distance));
                scratch.cmps += 1;
            })
            .await?;
    }

    let mut neighbors =
        Vec::with_capacity(accessor.max_degree_hint().map(|v| v.get()).unwrap_or(32));

    while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
        scratch.beam_nodes.clear();

        while scratch.beam_nodes.len() < beam_width.get()
            && let Some(closest_node) = scratch.best.closest_notvisited()
        {
            search_record.record(closest_node, scratch.hops, scratch.cmps);
            scratch.beam_nodes.push(*closest_node.id());
        }

        neighbors.clear();
        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                glue::NotInMut::new(&mut scratch.visited),
                |id, distance| neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        // The predicate ensures that the contents of `neighbors` are unique.
        //
        // We insert into the priority queue outside of the expansion for
        // code-locality purposes.
        neighbors
            .iter()
            .for_each(|neighbor| scratch.best.insert(*neighbor));

        scratch.cmps += neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;
    }

    Ok(InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: false,
    })
}
