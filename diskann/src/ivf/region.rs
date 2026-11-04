/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The lists filled for one stage of a build operation: a [`View`] to read them, and an
//! [`Update`] to record the changes of the stage.

use std::fmt::Debug;

use diskann_utils::{
    Reborrow,
    future::SendFuture,
    views::rowmajor::{self},
};

use crate::ANNResult;

/// The centroid and members of one filled list.
#[derive(Debug)]
pub struct List<'a, I> {
    /// Floating point representation for the centroid of the list.
    pub centroid: &'a [f32],
    /// Internal IDs of the members, including appends made earlier in the operation.
    pub members: &'a [I],
}

//////////
// View //
//////////

/// Read access to vectors and the [`List`] for a subset of list ids in the index.
///
/// An implementation of this trait will need to be able to reflect two kinds of updates
/// to the underlying index during the build -
///
/// 1. Routed appends, resulting from a call to [`route`](crate::ivf::traits::BuildAccessor::route).
/// 2. Every update `finish`ed by a previous call to [`fill`](crate::ivf::traits::BuildAccessor::fill).
///
/// The index may read one region from several threads at once.
pub trait View<I, L>: Send + Sync {
    /// Borrowed vector representation for a list.
    type VectorsRef<'a>;

    /// The handle returned by [`Self::get_vectors`]. Reborrowing it yields
    /// [`Self::VectorsRef`], to satisfy the borrow checker.
    type Vectors<'a>: for<'b> Reborrow<'b, Target = Self::VectorsRef<'b>>
    where
        Self: 'a;

    /// Return the centroid and members of list `i`, or `None` if `i` was not filled.
    fn get_list(&self, i: L) -> Option<List<'_, I>>;

    /// Return the vectors of list `i` or `None` if `i` was not filled.
    fn get_vectors(&self, i: L) -> Option<Self::Vectors<'_>>;
}

////////////
// Update //
////////////

/// Records the changes of one stage of a build operation.
///
/// A list is *live* from its creation until it is retired. Changes become
/// visible to later [`fill`](crate::ivf::traits::BuildAccessor::fill)
/// calls on the same accessor once the update [finishes](Self::finish).
///
/// Writes may target lists created earlier in the operation.
pub trait Update<I, L>: Send {
    /// Append distinct IDs that are not already members of the list.
    fn append_members(&mut self, i: L, ids: &[I]) -> ANNResult<()>;

    /// Remove members from list `i`.
    ///
    /// `ids` must be distinct members of list.
    fn remove_members(&mut self, i: L, ids: &[I]) -> ANNResult<()>;

    /// Retire list `i`, superseding earlier changes to its membership.
    fn retire(&mut self, i: L) -> ANNResult<()>;

    /// Record a new list with its centroid and complete membership, and return its ID.
    fn create(&mut self, centroid: &[f32], members: &[I]) -> ANNResult<L>;

    /// Make every change recorded through this update visible to the rest of the
    /// operation.
    fn finish(self) -> impl SendFuture<ANNResult<()>>;
}

//////////////
// AsMatRef //
//////////////

/// Conversion of borrowed vectors into a row-major matrix, one vector per row.
pub trait AsMatRef<'a, T> {
    /// Return the vectors as a matrix.
    fn as_view(self) -> rowmajor::Ref<'a, T>;
}

impl<'a, T> AsMatRef<'a, T> for rowmajor::Ref<'a, T> {
    fn as_view(self) -> rowmajor::Ref<'a, T> {
        self
    }
}
