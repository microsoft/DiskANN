/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`HasId`]: Traits such as [`NeighborAccessor`] and [`NeighborAccessorMut`] all need to
//! interact with the underlying [`DataProvider`] using an internal ID type.
//!
//! The [`HasId`] trait provides a common base-trait for these related concepts, which
//! ensures implementers only need to define and constrain it once.

use std::ops::Deref;

use crate::{ANNResult, graph::adjacencylist, utils::VectorId};

///////////
// HasId //
///////////

/// Indicate an association with an Id type.
pub trait HasId {
    type Id: VectorId;
}

impl<T> HasId for &T
where
    T: HasId,
{
    type Id = T::Id;
}

impl<T> HasId for &mut T
where
    T: HasId,
{
    type Id = T::Id;
}

macro_rules! has_id {
    ($T:ty = $Id:ty) => {
        impl $crate::provider::HasId for $T {
            type Id = $Id;
        }
    };
}

#[cfg(any(test, feature = "testing"))]
pub(crate) use has_id;

///////////////////////
// Neighbor Accessor //
///////////////////////

/// An accessor that provides random-access neighbor retrieval from a graph.
///
/// Generally, neighbor access and data access are logically decoupled, being served from
/// different stores. However, there are situations where data and neighbors are
/// interleaved in the underlying storage medium.
///
/// As such, data accessors used in conjunction with graph operations need to additionally
/// provide an implementation of this trait.
pub trait NeighborAccessor: HasId + Send + Sync {
    /// Get the neighbors for the node associated with `id`.
    ///
    /// Populate the neighbors into the `neighbors` out parameter.
    ///
    /// Implementations are expected to clear `neighbors` prior to populating.
    fn get_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: adjacencylist::Empty<'_, Self::Id>,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send;
}

/// A mutable extension of [`NeighborAccessor`] that enables the underlying graph to be
/// mutated.
pub trait NeighborAccessorMut: NeighborAccessor {
    /// Overwrite the neighbor list for the node associated with `id`.
    fn set_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send;

    /// Append all entries in `neighbors` to the neighbor list currently associated with `id`.
    ///
    /// The behavior when the resulting list exceeds some pre-configured capacity or
    /// contains duplicates is implementation defined.
    fn append_vector(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send;

    /// A potentially optimized bulk implementation of [`Self::set_neighbors`].
    ///
    /// For each `id`/`neighbors` pair in `iter`, set the adjacency list for the node associated
    /// with `id` to `neighbors`.
    ///
    /// Implementations are allowed to commit entries out of order.
    fn set_neighbors_bulk<I, T>(
        &mut self,
        iter: I,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        I: Iterator<Item = (Self::Id, T)> + Send,
        T: Deref<Target = [Self::Id]> + Send,
    {
        async move {
            for (vector_id, neighbors) in iter {
                self.set_neighbors(vector_id, neighbors.deref()).await?;
            }
            Ok(())
        }
    }

    //------------------//
    // Provided Methods //
    //------------------//

    /// Remove all neighbors from the adjacency list for `id`.
    fn clear(&mut self, id: Self::Id) -> impl std::future::Future<Output = ANNResult<()>> + Send {
        self.set_neighbors(id, &[])
    }
}
