/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Co-location groups of centroids for locality-preserving list storage.
//!
//! Not yet integrated with [`super::dynamic::MaintenanceAccessor`] or the updates in
//! [`super::update`].

use crate::utils::VectorId;

/// An authoritative grouping of live centroids for locality-preserving storage.
///
/// Every live centroid belongs to exactly one co-location group. Implementations
/// may use an identity grouping, where each group contains one centroid. Group
/// ids remain stable until retired and are independent of centroid/list ids.
pub trait CoLocationSet: Send + Sync {
    /// Stable logical centroid/list id.
    type ListId: VectorId;

    /// Stable logical co-location group id.
    type CoLocationGroupId: VectorId;

    /// Number of live co-location groups.
    fn len(&self) -> usize;

    /// Whether there are no live co-location groups.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the centroid ids in a live co-location group.
    fn group(&self, id: Self::CoLocationGroupId) -> Option<&[Self::ListId]>;

    /// Return the live co-location group containing a centroid.
    fn group_for(&self, id: Self::ListId) -> Option<Self::CoLocationGroupId>;
}

/// A new co-location group to install.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoLocationGroup<CoLocationGroupId, ListId> {
    /// Fresh stable group id. Installed ids must never be reused after retirement.
    pub id: CoLocationGroupId,
    /// Live centroids that should be stored together.
    pub centroids: Vec<ListId>,
}

/// Changes to the authoritative co-location set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoLocationDelta<CoLocationGroupId, ListId> {
    /// Co-location groups made live by this update.
    pub insert: Vec<CoLocationGroup<CoLocationGroupId, ListId>>,
    /// Live co-location groups retired by this update.
    pub retire: Vec<CoLocationGroupId>,
}
