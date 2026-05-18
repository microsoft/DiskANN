/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Self-contained test provider for the flat-search module.

use std::{
    borrow::Cow,
    collections::HashSet,
    fmt::{self, Debug},
    future::Future,
    sync::Arc,
};

use diskann_utils::{future::SendFuture, views::Matrix};
use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};
use thiserror::Error;

use crate::{
    ANNError, always_escalate,
    error::{RankedError, ToRanked, TransientError},
    flat::{DistancesUnordered, SearchStrategy},
    graph::test::synthetic::Grid,
    internal::counter::{Counter, LocalCounter},
    provider::{self, ExecutionContext, HasId, NoopGuard},
    utils::VectorRepr,
};

/// Error conditions for [`Provider::new`].
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("flat::test::Provider needs at least one item")]
    Empty,
    #[error("flat::test::Provider items must have non-zero dimension")]
    ZeroDimension,
}

impl From<ProviderError> for ANNError {
    #[track_caller]
    fn from(err: ProviderError) -> ANNError {
        ANNError::opaque(err)
    }
}

//////////////
// Provider //
//////////////

/// In-memory test provider for flat search.
#[derive(Debug)]
pub struct Provider {
    items: Matrix<f32>,
    get_element: Counter,
}

impl Provider {
    /// Construct a provider from a matrix of vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is empty or has zero-width columns.
    pub fn new(items: Matrix<f32>) -> Result<Self, ProviderError> {
        if items.nrows() == 0 {
            return Err(ProviderError::Empty);
        }
        if items.ncols() == 0 {
            return Err(ProviderError::ZeroDimension);
        }
        Ok(Self {
            items,
            get_element: Counter::new(),
        })
    }

    /// Build a provider over the row vectors of [`Grid::data`]. IDs are `0..n` in
    /// row-major order (last coordinate varies fastest).
    ///
    /// Unlike the graph-side `Provider::grid`, this does *not* add a separate
    /// start-point row — flat search has no notion of one.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::new`].
    pub fn grid(grid: Grid, size: usize) -> Result<Self, ProviderError> {
        Self::new(grid.data(size))
    }

    /// Number of vectors in the provider.
    pub fn len(&self) -> usize {
        self.items.nrows()
    }

    /// Dimension of each vector in the provider.
    pub fn dim(&self) -> usize {
        self.items.ncols()
    }

    /// Snapshot of the per-provider counters.
    pub fn metrics(&self) -> Metrics {
        Metrics {
            get_element: self.get_element.value(),
        }
    }

    /// Expose the items for brute force.
    pub fn items(&self) -> &Matrix<f32> {
        &self.items
    }
}

/// Counters tracked by [`Provider`].
#[derive(Debug, Clone)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct Metrics {
    /// The number of times any [`Visitor`] yielded an element.
    pub get_element: usize,
}

#[cfg(test)]
crate::test::cmp::verbose_eq!(Metrics { get_element });

/////////////
// Context //
/////////////

/// Per-search execution context. No spawn/clone tracking — flat search runs on
/// the calling task and never spawns.
#[derive(Debug, Clone, Default)]
pub struct Context;

impl Context {
    pub fn new() -> Self {
        Self
    }
}

impl ExecutionContext for Context {
    fn wrap_spawn<F, T>(&self, f: F) -> impl Future<Output = T> + Send + 'static
    where
        F: Future<Output = T> + Send + 'static,
    {
        f
    }
}

/////////////////////
// Errors / Guards //
/////////////////////

/// Critical id-validation error: the requested id is out of range.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
#[error("flat::test::Provider has no id {0}")]
pub struct InvalidId(pub u32);

always_escalate!(InvalidId);

impl From<InvalidId> for ANNError {
    #[track_caller]
    fn from(err: InvalidId) -> ANNError {
        ANNError::opaque(err)
    }
}

/// Transient access error injected by [`Visitor::flaky`].
///
/// Matches the shape of `graph::test::TransientAccessError`: panics in `Drop` if it
/// is dropped without being acknowledged or escalated. This guards against accidental
/// silent suppression of the error in the test code itself.
#[must_use]
#[derive(Debug)]
pub struct TransientGetError {
    id: u32,
    handled: bool,
}

impl TransientGetError {
    fn new(id: u32) -> Self {
        Self { id, handled: false }
    }
}

impl fmt::Display for TransientGetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "transient failure retrieving id {}", self.id)
    }
}

impl std::error::Error for TransientGetError {}

impl Drop for TransientGetError {
    fn drop(&mut self) {
        assert!(
            self.handled,
            "dropped an unhandled TransientGetError for id {}",
            self.id,
        );
    }
}

impl TransientError<InvalidId> for TransientGetError {
    fn acknowledge<D>(mut self, _why: D)
    where
        D: fmt::Display,
    {
        self.handled = true;
    }

    fn escalate<D>(mut self, _why: D) -> InvalidId
    where
        D: fmt::Display,
    {
        self.handled = true;
        InvalidId(self.id)
    }
}

/// Two-tier error for [`Visitor::distances_unordered`]: a critical [`InvalidId`]
/// or a recoverable [`TransientGetError`].
#[derive(Debug)]
pub enum AccessError {
    InvalidId(InvalidId),
    Transient(TransientGetError),
}

impl fmt::Display for AccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId(e) => fmt::Display::fmt(e, f),
            Self::Transient(e) => fmt::Display::fmt(e, f),
        }
    }
}

impl std::error::Error for AccessError {}

impl ToRanked for AccessError {
    type Transient = TransientGetError;
    type Error = InvalidId;

    fn to_ranked(self) -> RankedError<TransientGetError, InvalidId> {
        match self {
            Self::InvalidId(e) => RankedError::Error(e),
            Self::Transient(e) => RankedError::Transient(e),
        }
    }

    fn from_transient(transient: TransientGetError) -> Self {
        Self::Transient(transient)
    }

    fn from_error(error: InvalidId) -> Self {
        Self::InvalidId(error)
    }
}

//////////////////
// DataProvider //
//////////////////

impl provider::DataProvider for Provider {
    type Context = Context;
    type InternalId = u32;
    type ExternalId = u32;
    type Error = InvalidId;
    type Guard = NoopGuard<u32>;

    fn to_internal_id(&self, _ctx: &Context, gid: &u32) -> Result<u32, InvalidId> {
        if (*gid as usize) < self.items.nrows() {
            Ok(*gid)
        } else {
            Err(InvalidId(*gid))
        }
    }

    fn to_external_id(&self, _ctx: &Context, id: u32) -> Result<u32, InvalidId> {
        if (id as usize) < self.items.nrows() {
            Ok(id)
        } else {
            Err(InvalidId(id))
        }
    }
}

/////////////
// Visitor //
/////////////

/// Per-search visitor over a [`Provider`]. Analog of `graph::test::Accessor`: holds
/// the `'a` borrow of the provider, accumulates a local `get_element` counter that
/// flushes back on drop, and optionally injects transient errors for a configurable
/// set of ids.
pub struct Visitor<'a> {
    provider: &'a Provider,
    transient_ids: Option<Cow<'a, HashSet<u32>>>,
    get_element: LocalCounter<'a>,
}

impl<'a> Visitor<'a> {
    /// Construct a visitor with no fault injection.
    pub fn new(provider: &'a Provider) -> Self {
        Self {
            provider,
            transient_ids: None,
            get_element: provider.get_element.local(),
        }
    }

    /// Construct a visitor that returns a [`TransientGetError`] for any id in
    /// `transient_ids`. Other ids behave normally.
    pub fn flaky(provider: &'a Provider, transient_ids: Cow<'a, HashSet<u32>>) -> Self {
        Self {
            provider,
            transient_ids: Some(transient_ids),
            get_element: provider.get_element.local(),
        }
    }
}

impl Debug for Visitor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Visitor")
            .field("provider", &self.provider)
            .field("transient_ids", &self.transient_ids)
            .finish_non_exhaustive()
    }
}

impl HasId for Visitor<'_> {
    type Id = u32;
}

impl DistancesUnordered<<f32 as VectorRepr>::QueryDistance> for Visitor<'_> {
    type ElementRef<'a> = &'a [f32];
    type Error = AccessError;

    fn distances_unordered<F>(
        &mut self,
        computer: &<f32 as VectorRepr>::QueryDistance,
        mut f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + FnMut(Self::Id, f32),
    {
        async move {
            for (i, vector) in self.provider.items.row_iter().enumerate() {
                let id = i as u32;
                if let Some(ids) = &self.transient_ids
                    && ids.contains(&id)
                {
                    return Err(AccessError::Transient(TransientGetError::new(id)));
                }
                self.get_element.increment();
                let dist = computer.evaluate_similarity(vector);
                f(id, dist);
            }
            Ok(())
        }
    }
}

//////////////
// Strategy //
//////////////

/// Error from [`Strategy::create_visitor`] or [`Strategy::build_query_computer`]
/// when dimensions don't match.
#[derive(Debug, Clone, Error)]
#[error("dimension mismatch: strategy expects {expected}, got {actual}")]
pub struct StrategyError {
    pub expected: usize,
    pub actual: usize,
}

impl From<StrategyError> for ANNError {
    #[track_caller]
    fn from(err: StrategyError) -> ANNError {
        ANNError::opaque(err)
    }
}

/// Factory of [`Visitor`]s that validates dimensions and optionally injects
/// transient errors into the scan.
#[derive(Clone, Debug)]
pub struct Strategy {
    dim: usize,
    transient_ids: Option<Arc<HashSet<u32>>>,
}

impl Strategy {
    /// Construct a strategy expecting vectors of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            transient_ids: None,
        }
    }

    /// Construct a strategy whose visitors return a transient error on `get_element`
    /// for every id in `transient_ids`.
    pub fn with_transient(dim: usize, transient_ids: impl IntoIterator<Item = u32>) -> Self {
        Self {
            dim,
            transient_ids: Some(Arc::new(transient_ids.into_iter().collect())),
        }
    }
}

impl SearchStrategy<Provider, &[f32]> for Strategy {
    type ElementRef<'a> = &'a [f32];
    type QueryComputer = <f32 as VectorRepr>::QueryDistance;
    type QueryComputerError = StrategyError;
    type Visitor<'a> = Visitor<'a>;
    type Error = StrategyError;

    fn create_visitor<'a>(
        &'a self,
        provider: &'a Provider,
        _context: &'a Context,
    ) -> Result<Self::Visitor<'a>, Self::Error> {
        let actual = provider.dim();
        if actual != self.dim {
            return Err(StrategyError {
                expected: self.dim,
                actual,
            });
        }
        let visitor = match &self.transient_ids {
            Some(ids) => Visitor::flaky(provider, Cow::Borrowed(ids)),
            None => Visitor::new(provider),
        };
        Ok(visitor)
    }

    fn build_query_computer(
        &self,
        from: &[f32],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        if from.len() != self.dim {
            return Err(StrategyError {
                expected: self.dim,
                actual: from.len(),
            });
        }
        Ok(f32::query_distance(from, Metric::L2))
    }
}
