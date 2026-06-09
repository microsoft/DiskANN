/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

use super::SortedNeighbors;

use crate::{
    ANNError, ANNErrorKind, error, error::TransientError, graph::AdjacencyList, neighbor::Neighbor,
    utils::VectorId,
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

impl<I> From<ListError<I>> for ANNError
where
    I: VectorId,
{
    #[track_caller]
    fn from(err: ListError<I>) -> Self {
        match err {
            ListError::FailedVectorRetrieval(fvr) => {
                fvr.escalate("unexpected transient error during prune")
            }
            ListError::Other(e) => e,
        }
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
