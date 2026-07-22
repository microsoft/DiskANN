/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

use super::{config::PruneKind, internal::SortedNeighbors};

use crate::{
    ANNError, ANNErrorKind, error,
    graph::AdjacencyList,
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
pub struct Scratch<I>
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
    pub fn new() -> Self {
        Self {
            pool: Vec::new(),
            states: Vec::new(),
            neighbors: AdjacencyList::new(),
        }
    }

    /// Convert `self` into a `Context`, truncating the internal `pool` list to a length of
    /// `max_candidates`.
    pub fn as_context(&mut self, max_candidates: usize) -> Context<'_, I> {
        Context {
            pool: SortedNeighbors::new(&mut self.pool, max_candidates),
            states: &mut self.states,
            neighbors: &mut self.neighbors,
        }
    }

    /// Candidate buffer used by callers before pruning.
    pub fn candidates_mut(&mut self) -> &mut Vec<Neighbor<I>> {
        &mut self.pool
    }

    /// The most recent pruned adjacency list.
    pub fn neighbors(&self) -> &AdjacencyList<I> {
        &self.neighbors
    }
}

impl<I> Default for Scratch<I>
where
    I: VectorId,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Arguments passed to the lowest-level pruning algorithm.
#[derive(Debug)]
pub struct Context<'ctx, I>
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
/// Refer to the inline documentation in [`DiskANNIndex::occlude_list`] for documentation
/// on the use of these fields.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct State {
    /// The occlude factor for the pool item at the corresponding index.
    pub(in crate::graph) occlude_factor: f32,
    /// The index of the last checked neighbor.
    pub(in crate::graph) last_checked: u16,
    /// The candidate index of this neighbor.
    pub(in crate::graph) neighbor: u16,
}

/// Provider-independent policy for the Vamana robust-prune algorithm.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    degree: usize,
    alpha: f32,
    prune_kind: PruneKind,
    saturate: bool,
}

impl Policy {
    pub fn new(degree: usize, alpha: f32, prune_kind: PruneKind, saturate: bool) -> Self {
        Self {
            degree,
            alpha,
            prune_kind,
            saturate,
        }
    }
}

/// Failure returned by [`robust_prune`].
#[derive(Debug, Error)]
pub enum RobustPruneError<E = std::convert::Infallible> {
    #[error("robust prune alpha must be finite and >= 1.0, got {0}")]
    InvalidAlpha(f32),
    #[error("robust prune supports at most {max} candidates, got {actual}")]
    TooManyCandidates { actual: usize, max: usize },
    #[error("failed to reserve robust-prune workspace: {0}")]
    Allocation(#[source] std::collections::TryReserveError),
    #[error("distance computation failed: {0}")]
    Distance(E),
}

/// Run Vamana's robust-prune state machine over an already sorted candidate pool.
///
/// Data access and distance evaluation stay with the caller so providers may fetch
/// asynchronously before entering this synchronous kernel, while in-memory builders
/// can use contiguous vector storage and specialized distance functions.
pub fn robust_prune<I, V, E, L, D, X>(
    context: &mut Context<'_, I>,
    policy: Policy,
    cache: &mut Vec<(f32, Option<V>)>,
    mut lookup: L,
    mut distance: D,
    exclude: X,
) -> Result<(), RobustPruneError<E>>
where
    I: VectorId,
    L: FnMut(I) -> Option<V>,
    D: FnMut(&V, &V) -> Result<f32, E>,
    X: Fn(I) -> bool,
{
    if !policy.alpha.is_finite() || policy.alpha < 1.0 {
        return Err(RobustPruneError::InvalidAlpha(policy.alpha));
    }
    if context.pool.len() > u16::MAX as usize {
        return Err(RobustPruneError::TooManyCandidates {
            actual: context.pool.len(),
            max: u16::MAX as usize,
        });
    }

    let Context {
        pool,
        states,
        neighbors,
    } = context;

    cache.clear();
    if pool.is_empty() {
        neighbors.clear();
        return Ok(());
    }

    states
        .try_reserve(pool.len().saturating_sub(states.len()))
        .map_err(RobustPruneError::Allocation)?;
    cache
        .try_reserve(pool.len().saturating_sub(cache.len()))
        .map_err(RobustPruneError::Allocation)?;
    states.clear();
    states.resize(pool.len(), State::default());

    let mut current_alpha = 1.0f32;
    let increment_factor = policy.alpha.min(1.2);

    cache.extend(pool.iter().map(|neighbor| {
        if exclude(neighbor.id) {
            (neighbor.distance, None)
        } else {
            (neighbor.distance, lookup(neighbor.id))
        }
    }));

    // This is Vamana's existing lazy occlusion state machine. Keep the candidate and
    // neighbor positions in one `State` array so retries at larger alpha values resume
    // instead of recomputing comparisons.
    let mut found = 0;
    while found < policy.degree {
        for (i, (neighbor_distance, neighbor)) in cache.iter().enumerate() {
            if found >= policy.degree {
                break;
            }

            let State {
                mut occlude_factor,
                mut last_checked,
                ..
            } = states[i];

            if occlude_factor > current_alpha {
                continue;
            }

            let neighbor = match neighbor {
                Some(n) => n,
                None => {
                    debug_assert!(states.get(i).is_some(), "index {i} is out of bounds");
                    // SAFETY: `i` comes from iterating `cache`, which has the same length
                    // as `states`.
                    unsafe { states.get_unchecked_mut(i) }.occlude_factor = f32::MAX;
                    continue;
                }
            };

            while last_checked as usize != found {
                let result_position = states[last_checked as usize].neighbor.into_usize();
                last_checked += 1;

                if result_position >= i {
                    debug_assert!(states.get(i).is_some(), "index {i} is out of bounds");
                    // SAFETY: `i` comes from iterating `cache`, which has the same length
                    // as `states`.
                    unsafe { states.get_unchecked_mut(i) }.last_checked = last_checked;
                    continue;
                }

                let pair_distance = match &cache[result_position] {
                    (_, Some(value)) => {
                        distance(neighbor, value).map_err(RobustPruneError::Distance)?
                    }
                    (_, None) => f32::MAX,
                };

                occlude_factor = policy.prune_kind.update_occlude_factor(
                    *neighbor_distance,
                    pair_distance,
                    occlude_factor,
                    current_alpha,
                );

                if occlude_factor > current_alpha {
                    break;
                }
            }

            debug_assert!(states.get(i).is_some(), "index {i} is out of bounds");
            // SAFETY: `i` comes from iterating `cache`, which has the same length as
            // `states`.
            let state = unsafe { states.get_unchecked_mut(i) };

            state.last_checked = last_checked;
            if occlude_factor > current_alpha {
                state.occlude_factor = occlude_factor;
                continue;
            }

            state.occlude_factor = f32::MAX;
            states[found].neighbor = i as u16;
            found += 1;
        }

        if current_alpha == policy.alpha {
            break;
        }
        current_alpha = (current_alpha * increment_factor).min(policy.alpha);
    }

    let mut guard = neighbors.resize(found);
    std::iter::zip(guard.iter_mut(), states.iter()).for_each(|(destination, state)| {
        *destination = pool[state.neighbor.into_usize()].id;
    });
    guard.finish(found);

    debug_assert!(
        neighbors.len() <= policy.degree,
        "max degree bound violated"
    );

    if policy.saturate {
        for neighbor in pool.iter() {
            if neighbors.len() >= policy.degree {
                break;
            }
            if !exclude(neighbor.id) {
                neighbors.push(neighbor.id);
            }
        }
    }

    Ok(())
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
        ANNError::new(ANNErrorKind::IndexError, self).context(why.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbor::Neighbor;

    #[derive(Debug, PartialEq)]
    struct DistanceFailure;

    #[test]
    fn robust_prune_propagates_distance_failure() {
        let mut scratch = Scratch::new();
        scratch
            .candidates_mut()
            .extend([Neighbor::new(1_u32, 1.0), Neighbor::new(2_u32, 2.0)]);
        let mut context = scratch.as_context(2);
        let mut cache = Vec::new();

        let error = robust_prune(
            &mut context,
            Policy::new(2, 1.2, PruneKind::TriangleInequality, false),
            &mut cache,
            Some,
            |_, _| Err(DistanceFailure),
            |_| false,
        )
        .unwrap_err();

        assert!(matches!(error, RobustPruneError::Distance(DistanceFailure)));
    }

    #[test]
    fn robust_prune_rejects_invalid_alpha() {
        let mut scratch = Scratch::<u32>::new();
        let mut context = scratch.as_context(0);
        let mut cache = Vec::<(f32, Option<u32>)>::new();

        let error = robust_prune(
            &mut context,
            Policy::new(1, f32::NAN, PruneKind::TriangleInequality, false),
            &mut cache,
            Some,
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
            |_| false,
        )
        .unwrap_err();

        assert!(matches!(error, RobustPruneError::InvalidAlpha(value) if value.is_nan()));
    }

    #[test]
    fn robust_prune_does_not_allocate_output_for_an_empty_pool() {
        let mut scratch = Scratch::<u32>::new();
        let mut context = scratch.as_context(0);
        let mut cache = Vec::<(f32, Option<u32>)>::new();

        robust_prune(
            &mut context,
            Policy::new(usize::MAX, 1.2, PruneKind::TriangleInequality, false),
            &mut cache,
            Some,
            |_, _| Ok::<_, std::convert::Infallible>(0.0),
            |_| false,
        )
        .unwrap();

        assert!(scratch.neighbors().is_empty());
    }
}
