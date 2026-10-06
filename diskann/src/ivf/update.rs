/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::utils::VectorId;

/// A change to the set of live lists.
#[derive(Debug, Clone, PartialEq)]
pub enum CentroidDelta<Id, L> {
    /// Install a the staged centroid with this Id.
    Install,
    /// Retire the Id and move the points in this list to other list Ids.
    Retire { moves: Box<[MoveTo<Id, L>]> },
}

#[derive(Debug, Clone, PartialEq, Copy, Eq)]
pub struct MoveTo<I, L> {
    id: I,
    to: L,
}

impl<I: Copy, L: Copy> MoveTo<I, L> {
    /// Move the point `id` into the list `to`.
    pub(super) fn new(id: I, to: L) -> Self {
        Self { id, to }
    }

    /// The point that moves.
    pub fn id(&self) -> I {
        self.id
    }

    /// The list the point joins.
    pub fn to(&self) -> L {
        self.to
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Delta<Id, L> {
    // Centroid delta keyed by `id`
    CentroidDelta {
        id: L,
        delta: CentroidDelta<Id, L>,
    },
    // Point moves due to re-assignments, grouped by source centroid Id.
    PointMoves {
        from: L,
        moves: Box<[MoveTo<Id, L>]>,
    },
    // Point insertions from staged points, grouped by destination centroid Id.
    PointAppends {
        to: L,
        ids: Box<[Id]>,
    },
}

/// The complete partition change made by bootstrap or one insert batch.
///
/// The index guarantees, and providers may rely on without re-checking, that:
///
/// - installed ids were reserved by this accessor, and their centroids are finite
///   with the accessor's dimension;
/// - retired ids are live;
/// - each list appears in at most one centroid delta, and each point in at most one
///   point delta;
/// - every point staged by this accessor is appended;
/// - a move's source holds the point and differs from its destination;
/// - every destination is live after the update: installed, or live and not retired;
///   and
/// - every member of a retired list is moved out, so a provider may drop a retired
///   list wholesale and only place the points moved out of it.
///
/// Deltas are in no particular order, so a provider that cannot apply them as one
/// atomic change should install centroids before placing points into them.
#[derive(Debug, PartialEq)]
pub struct Deltas<Id, L> {
    inner: Box<[Delta<Id, L>]>,
}

impl<Id: VectorId, L: VectorId> Deltas<Id, L> {
    /// Collect `deltas` into one partition change.
    pub(super) fn new(deltas: Vec<Delta<Id, L>>) -> Self {
        Self {
            inner: deltas.into(),
        }
    }

    /// The individual changes this update makes.
    pub fn deltas(&self) -> &[Delta<Id, L>] {
        &self.inner
    }
}

#[cfg(test)]
mod tests {}
