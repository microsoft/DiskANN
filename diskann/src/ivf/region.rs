/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Regional input views and write operations.

use std::fmt::Debug;

use diskann_utils::{
    Reborrow,
    views::rowmajor::{self},
};

use crate::ANNResult;

/// One source list's centroid and membership list.
#[derive(Debug)]
pub struct List<'a, I, L> {
    pub id: L,
    pub centroid: &'a [f32],
    pub members: &'a [I],
}

/// Views for one build operation to perform splits/re-assignments .
pub trait View<I, L>: Send {
    /// The vector representation consumed by numerical algorithms.
    type VectorsRef<'a>;

    /// The returned handle supports shorter borrows for numerical algorithms.
    type Vectors<'a>: for<'b> Reborrow<'b, Target = Self::VectorsRef<'b>>
    where
        Self: 'a;

    /// Borrow a filled source's centroid and complete operation-local membership.
    fn get_list(&self, i: L) -> Option<List<'_, I, L>>;

    /// Borrow canonical rows in the same order as the source's member IDs.
    fn get_vectors(&self, i: L) -> Option<Self::Vectors<'_>>;
}

/// Regional writes for one build operation.
pub trait Update<I, L>: Send {
    /// Append distinct IDs that are not already members of the operation-local list.
    fn append_members(&mut self, i: L, ids: &[I]) -> ANNResult<()>;
    /// Remove membership without deleting the underlying vectors.
    ///
    /// IDs must be distinct current members of the operation-local list.
    fn remove_members(&mut self, i: L, ids: &[I]) -> ANNResult<()>;

    /// Retire a source list and its centroid, superseding earlier membership changes.
    fn retire(&mut self, i: L) -> ANNResult<()>;

    /// Record a new centroid and its complete membership.
    fn create(&mut self, centroid: &[f32], members: &[I]) -> ANNResult<()>;
}

/////////////
/// AsMat ///
/////////////

pub trait AsMatRef<'a, T> {
    fn as_view(self) -> rowmajor::Ref<'a, T>;
}

impl<'a, T> AsMatRef<'a, T> for rowmajor::Ref<'a, T> {
    fn as_view(self) -> rowmajor::Ref<'a, T> {
        self
    }
}
