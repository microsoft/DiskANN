/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![warn(missing_debug_implementations, missing_docs)]

//! Scratch space for in-memory index based search

use std::collections::VecDeque;

use crate::{
    neighbor::{Neighbor, NeighborPriorityQueue},
    utils::{VectorId, object_pool::AsPooled},
};
use hashbrown::HashSet;

/// In-mem index related limits
pub const GRAPH_SLACK_FACTOR: f64 = 1.3_f64;

/// Scratch space used during graph search.
///
/// This struct contains three important members used by both the sync and async indexes:
/// `query`, `best`, and `visited`.
///
/// The member `id_scratch` is only used by the sync index.
///
/// Members `labels` and `beta` are used by the async index for beta-filtered search.
#[derive(Debug)]
pub struct SearchScratch<I, Q = NeighborPriorityQueue<I>>
where
    I: VectorId,
{
    /// A priority queue of the best candidates seen during search. This data structure is
    /// also responsible for determining the best unvisited candidate.
    ///
    /// Used by both sync and async.
    ///
    /// When used in a paged search context, this queue is unbounded.
    pub best: Q,

    /// A record of all ids visited during a search.
    ///
    /// Used by both sync and async.
    ///
    /// This is used to prevent multiple requests to the same `id` from the vector providers.
    pub visited: HashSet<I>,

    /// A buffer for adjacency lists.
    ///
    /// Only used by sync.
    ///
    /// Adjacency lists in the sync provider are guarded by read/write locks. The
    /// `id_scratch` is used to copy out the contents of an adjacency list to minimize the
    /// duration the lock is held.
    pub id_scratch: Vec<I>,

    /// A list of beam search nodes used during search. This is used when beam search is enabled
    /// to temporarily hold beam of nodes in each hop.
    pub beam_nodes: Vec<I>,

    /// A queue of nodes to visit during range search
    /// Does not need to be ordered by distance
    pub range_frontier: VecDeque<I>,

    /// A list of nodes that are in range of the query
    /// Only used during range search
    pub in_range: Vec<Neighbor<I>>,

    /// A tracker for how many hops we have taken during the current search
    pub hops: u32,

    /// A tracker for how many comparisons we have made during the current search
    pub cmps: u32,
}

/// The priority queue in `SearchScratch` operates in two modes:
///
/// * Fixed: The queue has a fixed capacity and discards the worst members. This is used by
///   the standard nearest-neighbor search methods.
///
/// * Resizable: Allows the priority queue to be unbounded in size. This is used by paged
///   search to allow multiple rounds of searching.
#[derive(Debug, Clone, Copy)]
pub enum PriorityQueueConfiguration {
    /// Configure the priority queue in fixed-L mode with the provided `search_l`.
    Fixed(usize),
    /// Configure the priority queue in resizeable mode with the provided `search_l`.
    Resizable(usize),
}

impl<I> SearchScratch<I, NeighborPriorityQueue<I>>
where
    I: VectorId,
{
    /// Create a new `SearchScratch` with an uninitializd `dim`-dimensional query.
    ///
    /// This method is used when pre-allocating many scratch spaces and should not be used
    /// for general searching (use [`Self::new`] instead.
    ///
    /// # Parameters
    ///
    /// * `dim`: The number of dimensions to allocate for the internal query vector.
    /// * `nbest`: The number of best candidates to track.
    /// * `size_hint`: If provided, hints at the capacity for preallocating the `visited` set.
    ///
    /// # Note
    ///
    /// This method does not enable the beta-filtering feature.
    pub fn new(nbest: PriorityQueueConfiguration, size_hint: Option<usize>) -> Self {
        let visited = match size_hint {
            Some(size_hint) => HashSet::with_capacity(size_hint),
            None => HashSet::new(),
        };

        let best = match nbest {
            PriorityQueueConfiguration::Fixed(capacity) => NeighborPriorityQueue::new(capacity),
            PriorityQueueConfiguration::Resizable(capacity) => {
                NeighborPriorityQueue::auto_resizable_with_search_param_l(capacity)
            }
        };

        Self {
            best,
            visited,
            id_scratch: Vec::new(),
            beam_nodes: Vec::new(),
            in_range: Vec::new(),
            range_frontier: VecDeque::new(),
            hops: 0,
            cmps: 0,
        }
    }

    /// Reconfigures the queue lengths of `self` for a longer or shorter search.
    ///
    /// # Parameters
    ///
    /// * `nbest`: The new number of candidates to track.
    pub fn resize(&mut self, nbest: usize) {
        self.best.reconfigure(nbest);
    }

    /// Clear internal data structures.
    ///
    /// This allows `self` to be used for another search.
    pub fn clear(&mut self) {
        self.best.clear();
        self.visited.clear();
        self.id_scratch.clear();
        self.beam_nodes.clear();
        self.in_range.clear();
        self.range_frontier.clear();

        self.hops = 0;
        self.cmps = 0;
    }

    /// Return the currently configured `search_l`: the number of best candidates to track.
    pub fn search_l(&self) -> usize {
        self.best.search_l()
    }
}

/// Estimate the size of the node visited set. This needs to be upper bound of the number of
/// nodes visited during search. The formula is based on initial theoretical analysis and
/// adjusted with MARGIN_FACTOR obtained from empirical data.
///
/// The formula is:
/// ```math
/// MARGIN_FACTOR * max_degree * GRAPH_SLACK_FACTOR * search_list_size
/// ```
///
/// ## Explanation
///
/// * `visited_nodes <= number_of_hops * max_degree`.
/// * `number_of_hops` is generally `MARGIN_FACTOR * L`.
/// * `MAX_DEGREE` is adjusted with `GRAPH_SLACK_FACTOR`.
///
/// In future, this could be more precisely computed by either of two approaches:
///
/// 1. Analyzing data in index build time based on theory
/// 2. At run time by analyzing about 1000 queries and taking maximum size
pub(crate) fn estimate_node_visited_set_size(max_degree: usize, search_list_size: usize) -> usize {
    // The `MARGIN_FACTOR` is obtained by running search queries on the Sift 1M and
    // OpenAI datasets, checking maximum visited nodes, and comparing it with the formula.
    const MARGIN_FACTOR: f64 = 1.1;

    (MARGIN_FACTOR * max_degree as f64 * GRAPH_SLACK_FACTOR * search_list_size as f64).ceil()
        as usize
}

impl<I> AsPooled<&SearchScratchParams> for SearchScratch<I, NeighborPriorityQueue<I>>
where
    I: VectorId,
{
    fn create(param: &SearchScratchParams) -> Self {
        SearchScratch::new(
            PriorityQueueConfiguration::Fixed(param.l_value + param.num_frozen_pts),
            Some(estimate_node_visited_set_size(
                param.max_degree,
                param.l_value,
            )),
        )
    }

    fn modify(&mut self, param: &SearchScratchParams) {
        self.clear();
        if self.best.is_resizable() {
            // Scratch is used only in Fixed configuration. If it is resizable, then convert to fixed.
            self.best = NeighborPriorityQueue::new(param.l_value + param.num_frozen_pts);
        } else {
            self.best.reconfigure(param.l_value + param.num_frozen_pts);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchScratchParams {
    pub l_value: usize,
    pub max_degree: usize,
    pub num_frozen_pts: usize,
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_new() {
        {
            let x = SearchScratch::<u64, NeighborPriorityQueue<u64>>::new(
                PriorityQueueConfiguration::Fixed(10),
                None,
            );

            assert!(!x.best.is_resizable());
            assert_eq!(x.search_l(), 10);
            assert_eq!(x.best.search_l(), 10);
            assert_eq!(x.visited.capacity(), 0);

            assert!(x.visited.is_empty());
            assert!(x.id_scratch.is_empty());

            assert!(x.hops == 0);
            assert!(x.cmps == 0);
        }

        {
            let x = SearchScratch::<u64, NeighborPriorityQueue<u64>>::new(
                PriorityQueueConfiguration::Resizable(10),
                None,
            );

            assert!(x.best.is_resizable());
            assert_eq!(x.search_l(), 10);
            assert_eq!(x.best.search_l(), 10);
            assert_eq!(x.visited.capacity(), 0);

            assert!(x.visited.is_empty());
            assert!(x.id_scratch.is_empty());

            assert!(x.hops == 0);
            assert!(x.cmps == 0);
        }
    }

    #[test]
    pub fn test_resize() {
        let mut x = SearchScratch::<u64, _>::new(PriorityQueueConfiguration::Fixed(5), None);
        assert_eq!(x.search_l(), 5);
        x.resize(10);
        assert_eq!(x.search_l(), 10);
    }

    #[test]
    pub fn test_reconfigure() {
        let mut x = SearchScratch::<u64, NeighborPriorityQueue<u64>>::new(
            PriorityQueueConfiguration::Fixed(5),
            None,
        );
        assert_eq!(x.search_l(), 5);

        x.resize(10);
        assert_eq!(x.search_l(), 10);
    }

    #[test]
    pub fn test_clear() {
        let mut x = SearchScratch::<u64, NeighborPriorityQueue<u64>>::new(
            PriorityQueueConfiguration::Fixed(5),
            None,
        );

        x.visited.insert(1);
        x.visited.insert(10);

        x.id_scratch.push(1);
        x.id_scratch.push(10);

        x.best.insert(Neighbor::new(1, 1.0));
        x.best.insert(Neighbor::new(10, 2.0));
        assert_eq!(x.best.size(), 2);

        // Do the clear.
        x.clear();
        assert!(x.visited.is_empty());
        assert!(x.id_scratch.is_empty());
        assert_eq!(x.best.size(), 0);

        assert!(x.hops == 0);
        assert!(x.cmps == 0);
    }
}
