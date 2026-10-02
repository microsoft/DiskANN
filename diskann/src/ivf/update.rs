/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Declarative partition updates applied by [`MaintenanceAccessor::apply`].
//!
//! An [`InsertionUpdate`] lists the primitive changes made by bootstrap or one insert
//! batch: lists installed or retired, as [`CentroidDelta`]s, and points appended or
//! moved, as [`PointDelta`]s.
//!
//! Only the index constructs updates. It validates everything it reads from the
//! provider or its caller, and its planner guarantees the invariants documented on
//! [`InsertionUpdate`] by construction. Providers can only read updates.
//!
//! [`MaintenanceAccessor::apply`]: crate::ivf::dynamic::MaintenanceAccessor::apply

use crate::utils::VectorId;

/// A change to the set of live lists.
#[derive(Debug, Clone, PartialEq)]
pub enum CentroidDelta<L> {
    /// Make the fresh list `id` live with the full-precision `centroid`.
    Install { id: L, centroid: Box<[f32]> },
    /// Retire the live list `id`. Retired ids are never reused.
    Retire { id: L },
}

/// A change to one point's list membership.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointDelta<Id, L> {
    /// The point `id`, staged by this accessor, joins list `to`.
    Append { id: Id, to: L },
    /// The point `id` leaves list `from` and joins list `to`.
    Move { id: Id, from: L, to: L },
}

impl<Id: Copy, L: Copy> PointDelta<Id, L> {
    /// The point joining a list.
    pub fn id(&self) -> Id {
        match *self {
            Self::Append { id, .. } | Self::Move { id, .. } => id,
        }
    }

    /// The list the point joins.
    pub fn to(&self) -> L {
        match *self {
            Self::Append { to, .. } | Self::Move { to, .. } => to,
        }
    }
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
/// - a move's `from` holds the point and differs from `to`;
/// - every destination is live after the update: installed, or live and not retired;
/// - every member of a retired list is moved out, so a provider may drop a retired
///   list wholesale and only place the points moved out of it; and
/// - [`Self::points`] is sorted by destination, then id, so
///   `points().chunk_by(|a, b| a.to() == b.to())` yields each list's incoming points
///   in one run.
#[derive(Debug, PartialEq)]
pub struct InsertionUpdate<Id, L> {
    centroids: Vec<CentroidDelta<L>>,
    points: Vec<PointDelta<Id, L>>,
}

impl<Id: VectorId, L: VectorId> InsertionUpdate<Id, L> {
    /// Combine `centroids` and `points`, sorting `points` by destination, then id.
    pub(super) fn new(
        centroids: Vec<CentroidDelta<L>>,
        mut points: Vec<PointDelta<Id, L>>,
    ) -> Self {
        points.sort_unstable_by_key(|point| (point.to(), point.id()));
        Self { centroids, points }
    }

    /// The lists installed or retired.
    pub fn centroids(&self) -> &[CentroidDelta<L>] {
        &self.centroids
    }

    /// The points appended or moved, sorted by destination, then id.
    pub fn points(&self) -> &[PointDelta<Id, L>] {
        &self.points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_are_sorted_by_destination_then_id() {
        let update = InsertionUpdate::<u32, u32>::new(
            vec![CentroidDelta::Retire { id: 4 }],
            vec![
                PointDelta::Move {
                    id: 3,
                    from: 4,
                    to: 9,
                },
                PointDelta::Append { id: 8, to: 7 },
                PointDelta::Append { id: 1, to: 9 },
                PointDelta::Move {
                    id: 2,
                    from: 4,
                    to: 7,
                },
            ],
        );
        let order: Vec<_> = update
            .points()
            .iter()
            .map(|point| (point.to(), point.id()))
            .collect();
        assert_eq!(order, vec![(7, 2), (7, 8), (9, 1), (9, 3)]);
        assert_eq!(update.centroids(), &[CentroidDelta::Retire { id: 4 }]);
    }
}
