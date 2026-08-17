/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

use crate::{
    ANNError, error,
    graph::{AdjacencyList, config::PruneKind, internal::SortedNeighbors},
    neighbor::Neighbor,
    utils::{IntoUsize, VectorId},
};

/// Options provided to prune. See the field-level documentation for more details.
///
/// This struct should be kept cheap to construct.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Options {
    /// Force adjacency list saturation.
    ///
    /// Adjacency list saturation expands the post-pruning candidate list up to the
    /// maximum degree by greedily adding skipped neighbors from the original candidate
    /// pool.
    pub(in crate::graph) force_saturate: bool,
}

/// An aggregate of scratch space used by the pruning algorithm for allocation.
///
/// The actual object passed to the pruning algorithms is [`Context`], which allows
/// sub-fields to be over-written as needed with local state if that is available instead.
#[derive(Debug)]
pub(crate) struct Scratch<I>
where
    I: VectorId,
{
    pub(in crate::graph) pool: Vec<Neighbor<I>>,
    pub(in crate::graph) states: Vec<State>,
    pub(in crate::graph) neighbors: AdjacencyList<I>,
}

impl<I> Scratch<I>
where
    I: VectorId,
{
    /// Create a new empty scratch space.
    ///
    /// This function should not allocate.
    pub(in crate::graph) fn new() -> Self {
        Self {
            pool: Vec::new(),
            states: Vec::new(),
            neighbors: AdjacencyList::new(),
        }
    }

    /// Convert `self` into a `Context`, truncating the internal `pool` list to a length of
    /// `max_candidates`.
    pub(in crate::graph) fn as_context(&mut self, max_candidates: usize) -> Context<'_, I> {
        Context {
            pool: SortedNeighbors::new(&mut self.pool, max_candidates),
            states: &mut self.states,
            neighbors: &mut self.neighbors,
        }
    }
}

/// Arguments passed to the lowest-level pruning algorithm.
#[derive(Debug)]
pub(crate) struct Context<'ctx, I>
where
    I: VectorId,
{
    /// Input: The list of candidates to prune.
    pub(in crate::graph) pool: SortedNeighbors<'ctx, I>,
    /// Scratch: State tracking for prune.
    pub(in crate::graph) states: &'ctx mut Vec<State>,
    /// Output: The pruned candidates list.
    pub(in crate::graph) neighbors: &'ctx mut AdjacencyList<I>,
}

/// Position-wise state tracking.
///
/// Refer to the inline documentation in [`robust_prune`] for documentation on the use
/// of these fields.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct State {
    /// The occlude factor for the pool item at the corresponding index.
    pub(in crate::graph) occlude_factor: f32,
    /// The index of the last checked neighbor.
    pub(in crate::graph) last_checked: u16,
    /// The candidate index of this neighbor.
    pub(in crate::graph) neighbor: u16,
}

/// Select a degree-bounded neighbor set with Vamana RobustPrune.
///
/// `cache` stores each source distance and candidate vector in ascending
/// source-distance order. `None` excludes that candidate without changing
/// positional alignment. `states` has one entry for each candidate position.
///
/// The function writes selected candidate indexes to `states[..result]` and
/// returns `result`. The caller converts those indexes to graph IDs.
pub(in crate::graph) fn robust_prune<V, D>(
    cache: SortedNeighbors<'_, Option<V>>,
    states: &mut [State],
    degree: usize,
    alpha: f32,
    prune_kind: PruneKind,
    mut compute_distance: D,
) -> usize
where
    D: FnMut(&V, &V) -> f32,
{
    let mut current_alpha = 1.0f32;
    let increment_factor = alpha.min(1.2);

    // For an alpha value `A`, a candidate `i` is promoted to a neighbor if for all
    // ```
    // max{j < i | j is a neighbor}(occlude_factor(i, j))
    // ```
    // This process happens with multiple values of `A`.
    //
    // We can compute this efficiently using the following rules:
    //
    // 1. For a candidate `i`, start scanning `j < i`, computing occlude factors.
    // 2. If we find an occlude factor greater than `A`, record that `i` has visited
    //    `j`, stop computing occlude factors, and move on to `i + 1`.
    // 3. If we reach `j == i - 1` with the maximum occlude factor less than `A`, then
    //    `i` gets promoted to a neighbor.
    //
    // On the implementation side, we use `states` in the following way:
    //
    // * `states[n].neighbor` is the **index** in `cache` of the `n`th **neighbor**.
    //   Note that a "neighbor" is a candidate that passes pruning.
    //
    //   Very important: to get the index `j` in the above description, we need to
    //   check `cache[states[n].neighbor]`.
    //
    //   This indexing naturally skips candidates `j` that have not been promoted to
    //   neighbors.
    //
    // * `states[i].occlude_factor` is the maximum occlude factor found for a candidate
    //   `i`. This gets set to `f32::MAX` when `i` is promoted to a neighbor which
    //   excludes it from future consideration.
    //
    // * `states[i].last_checked` is the highest value of `n` against which the
    //   occlude factor for `j = cache[states[n].neighbor]` has been checked.
    //
    //   The maximum value this should reach is `i`.
    //
    // Note that we use `states` for both "candidate" and "neighbor" tracking.
    let mut found = 0;
    while found < degree {
        for (i, neighbor) in cache.iter().enumerate() {
            if found >= degree {
                break;
            }

            // The tracking states for candidate `i`.
            let State {
                mut occlude_factor,
                mut last_checked,
                ..
            } = states[i];

            // If the occlusion factor for this neighbor is too high, skip it.
            if occlude_factor > current_alpha {
                continue;
            }

            // Retrieval from the cache might not be perfect.
            //
            // This neighbor did not end up in the cache, then just skip it.
            let neighbor_distance = neighbor.distance();
            let neighbor = match neighbor.id() {
                Some(n) => n,
                None => {
                    debug_assert!(states.get(i).is_some(), "index {i} is out of bounds");
                    // SAFETY: We've already checked `states[i]`.
                    unsafe { states.get_unchecked_mut(i) }.occlude_factor = f32::MAX;
                    continue;
                }
            };

            // Increment `position` until we've compared with all current entries in
            // `result`.
            //
            // When the list is empty, the loop is skipped allowing the first undeleted
            // element to be added.
            while last_checked as usize != found {
                let result_position = states[last_checked as usize].neighbor.into_usize();
                last_checked += 1;

                // If the position of this result in `cache` is greater than or equal
                // to the current working position, then skip this candidate.
                if result_position >= i {
                    debug_assert!(states.get(i).is_some(), "index {i} is out of bounds");
                    // SAFETY: We've already checked `states[i]`.
                    unsafe { states.get_unchecked_mut(i) }.last_checked = last_checked;
                    continue;
                }

                // Otherwise, compute the distance between the result and this neighbor
                // and update the occlude factor.
                let distance = match cache[result_position].id() {
                    Some(v) => compute_distance(neighbor, v),
                    None => f32::MAX,
                };

                // Update occlude factor
                occlude_factor = prune_kind.update_occlude_factor(
                    *neighbor_distance,
                    distance,
                    occlude_factor,
                    current_alpha,
                );

                // Check if the most recent update to the occlusion factor removes this
                // neighbor from consideration.
                if occlude_factor > current_alpha {
                    break;
                }
            }

            debug_assert!(states.get(i).is_some(), "index {i} is out of bounds");
            // SAFETY: We've already checked `states[i]`.
            let state = unsafe { states.get_unchecked_mut(i) };

            state.last_checked = last_checked;
            if occlude_factor > current_alpha {
                state.occlude_factor = occlude_factor;
                continue;
            }

            // This neighbor has passed all the requirements of being a candidate.
            state.occlude_factor = f32::MAX;

            // This conversion should always succeed.
            states[found].neighbor = i as u16;
            found += 1;
        }

        // Exit if we completed the final iteration.
        if current_alpha == alpha {
            break;
        }
        // Update current alpha for the next iteration.
        current_alpha = (current_alpha * increment_factor).min(alpha);
    }

    found
}

#[derive(Debug, Clone, Copy, Error)]
#[error("retrieval of main vector id {} failed during prune aggregation", self.0)]
pub(crate) struct FailedVectorRetrieval<I>(I)
where
    I: VectorId;

impl<I> error::TransientError<ANNError> for FailedVectorRetrieval<I>
where
    I: VectorId,
{
    fn acknowledge<D>(self, _why: D)
    where
        D: std::fmt::Display,
    {
    }

    #[track_caller]
    #[inline(never)]
    fn escalate<D>(self, why: D) -> ANNError
    where
        D: std::fmt::Display,
    {
        ANNError::new(self).context(why.to_string())
    }
}

/// Failure condition for [`DiskANNIndex::robust_prune_list`].
///
/// It's currently possible for retrieval of the id being pruned to fail due to a transient
/// error. We do not always want to escalate this as a hard error, and thus provide an
/// option for transient error handling.
#[derive(Debug)]
pub(crate) enum ListError<I>
where
    I: VectorId,
{
    /// A potentially transient error.
    FailedVectorRetrieval(FailedVectorRetrieval<I>),
    /// A critical error.
    Other(ANNError),
}

impl<I> ListError<I>
where
    I: VectorId,
{
    pub(in crate::graph) fn failed_retrieval(id: I) -> Self {
        Self::FailedVectorRetrieval(FailedVectorRetrieval(id))
    }
}

impl<I> From<ANNError> for ListError<I>
where
    I: VectorId,
{
    fn from(err: ANNError) -> Self {
        Self::Other(err)
    }
}

impl<I> error::ToRanked for ListError<I>
where
    I: VectorId,
{
    type Transient = FailedVectorRetrieval<I>;
    type Error = ANNError;

    fn to_ranked(self) -> error::RankedError<Self::Transient, Self::Error> {
        match self {
            Self::FailedVectorRetrieval(err) => error::RankedError::Transient(err),
            Self::Other(err) => error::RankedError::Error(err),
        }
    }

    fn from_transient(transient: Self::Transient) -> Self {
        Self::FailedVectorRetrieval(transient)
    }

    fn from_error(error: Self::Error) -> Self {
        Self::Other(error)
    }
}
