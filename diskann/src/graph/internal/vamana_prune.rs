/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Vamana-owned RobustPrune scratch and provider error state.

use thiserror::Error;

use super::{SortedNeighbors, robust_prune::State};
use crate::{
    ANNError, ANNErrorKind, error, graph::AdjacencyList, neighbor::Neighbor, utils::VectorId,
};

#[derive(Debug, Clone, Copy)]
pub(in crate::graph) struct Options {
    pub(in crate::graph) force_saturate: bool,
}

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
    pub(in crate::graph) fn new() -> Self {
        Self {
            pool: Vec::new(),
            states: Vec::new(),
            neighbors: AdjacencyList::new(),
        }
    }

    pub(in crate::graph) fn as_context(&mut self, max_candidates: usize) -> Context<'_, I> {
        Context {
            pool: SortedNeighbors::new(&mut self.pool, max_candidates),
            states: &mut self.states,
            neighbors: &mut self.neighbors,
        }
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

#[derive(Debug)]
pub(in crate::graph) struct Context<'a, I>
where
    I: VectorId,
{
    pub(in crate::graph) pool: SortedNeighbors<'a, I>,
    pub(in crate::graph) states: &'a mut Vec<State>,
    pub(in crate::graph) neighbors: &'a mut AdjacencyList<I>,
}

#[derive(Debug, Clone, Copy, Error)]
#[error("retrieval of main vector id {} failed during prune aggregation", self.0)]
pub(in crate::graph) struct FailedVectorRetrieval<I>(I)
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

#[derive(Debug)]
pub(in crate::graph) enum ListError<I>
where
    I: VectorId,
{
    FailedVectorRetrieval(FailedVectorRetrieval<I>),
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
    fn from(error: ANNError) -> Self {
        Self::Other(error)
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
            Self::FailedVectorRetrieval(error) => error::RankedError::Transient(error),
            Self::Other(error) => error::RankedError::Error(error),
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
mod tests;
