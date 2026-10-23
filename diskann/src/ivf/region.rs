/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Regional input views and write operations.

use std::fmt::Debug;

use diskann_utils::{Reborrow, views::rowmajor};

use crate::ANNResult;

/// One source list's centroid and complete operation-local membership.
///
/// Only reference metadata is owned; point and centroid coordinates are borrowed.
/// A region may include both split targets and neighboring lists.
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
    ///
    /// There is one row per member, including pending accessor appends. Coordinates
    /// have the accessor's canonical dimension and are finite.
    fn get_vectors(&self, i: L) -> Option<Self::Vectors<'_>>;
}

/// Regional writes for one build operation.
pub trait Update<I, L>: Send {
    /// Record a complete membership replacement.
    fn set_members(&mut self, i: L, ids: &[I]) -> ANNResult<()>;

    /// Record retirement of a source list and its centroid.
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

pub trait AsMatMut<'a, T>: AsMatRef<'a, T> {
    fn as_mut_view(self) -> rowmajor::Mut<'a, T>;
}
