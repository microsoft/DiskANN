/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Complete destination memberships produced by regional clustering.

use diskann_utils::views::rowmajor;

/// One destination list in a [`Partitioning`].
#[derive(Debug)]
pub struct Partition<I, L> {
    /// `Some` retains a source list's identity and centroid; `None` creates a new list.
    pub id: Option<L>,
    pub members: Vec<I>,
}

/// A region's resulting centroids and complete memberships.
///
/// There is one centroid row per partition; row `i` corresponds to `partitions[i]`.
/// Retained centroid rows must match their source centroids. The member groups
/// together contain every source member exactly once, including pending appends.
///
/// Coordinates of points are not included: providers already own or have prepared
/// those vectors. Membership buffers can be transferred without copying their
/// contents; centroid rows are available for creating new lists.
#[derive(Debug)]
pub struct Partitioning<I, L> {
    pub centroids: rowmajor::Owned<f32>,
    pub partitions: Vec<Partition<I, L>>,
}
