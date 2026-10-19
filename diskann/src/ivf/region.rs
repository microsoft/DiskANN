/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Regional input views, write operations, and a reusable canonical insertion batch.

use diskann_utils::views::rowmajor::{self, Matrix};
use hashbrown::HashMap;

use crate::{ANNError, ANNResult, utils::VectorId};

/// One source list's centroid and complete operation-local membership.
///
/// Only reference metadata is owned; point and centroid coordinates are borrowed.
/// A region may include both split targets and neighboring lists.
#[derive(Debug)]
pub struct View<'a, I, L> {
    pub id: L,
    pub centroid: &'a [f32],
    pub members: &'a [I],
}

/// Views for one build operation to perform splits/re-assignments .
pub trait RegionView<I, L>: Send {
    /// Borrow a filled source's centroid and complete operation-local membership.
    fn get_view(&self, i: L) -> Option<View<'_, I, L>>;

    /// Borrow canonical rows in the same order as the source's member IDs.
    ///
    /// There is one row per member, including pending accessor appends. Coordinates
    /// have the accessor's canonical dimension and are finite.
    fn get_vectors(&self, i: L) -> Option<rowmajor::Ref<'_, f32>>;
}

/// Regional writes for one build operation.
pub trait RegionUpdate<I, L>: Send {
    /// Record a complete membership replacement.
    fn set_members(&mut self, i: L, ids: &[I]) -> ANNResult<()>;

    /// Record retirement of a source list and its centroid.
    fn retire(&mut self, i: L) -> ANNResult<()>;

    /// Record a new centroid and its complete membership.
    fn create(&mut self, centroid: &[f32], members: &[I]) -> ANNResult<()>;
}
