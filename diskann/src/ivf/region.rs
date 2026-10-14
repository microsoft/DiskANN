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

/// A point's identity and borrowed canonical vector.
#[derive(Debug, Clone, Copy)]
pub struct PointRef<'a, I> {
    pub id: I,
    pub vector: &'a [f32],
}

/// Canonical rows shared between routing and an operation's build accessor.
///
/// IDs must be unique. The contiguous matrix is owned once; iteration and lookup
/// borrow its rows without copying coordinates.
#[derive(Debug)]
pub struct Batch<I: VectorId> {
    ids: Vec<I>,
    vectors: rowmajor::Owned<f32>,
    rows: HashMap<I, usize>,
}

impl<I: VectorId> Batch<I> {
    /// # Errors
    ///
    /// Fails if the matrix has zero columns or its row count differs from `ids`.
    pub fn new(ids: Vec<I>, vectors: rowmajor::Owned<f32>) -> ANNResult<Self> {
        if vectors.ncols() == 0 || vectors.nrows() != ids.len() {
            return Err(ANNError::message(
                "an IVF batch needs one nonempty canonical row per point ID",
            ));
        }
        let rows = ids
            .iter()
            .copied()
            .enumerate()
            .map(|(row, id)| (id, row))
            .collect();
        Ok(Self { ids, vectors, rows })
    }

    pub fn vectors(&self) -> rowmajor::Ref<'_, f32> {
        self.vectors.as_view()
    }

    pub fn points(&self) -> impl Iterator<Item = PointRef<'_, I>> {
        self.ids
            .iter()
            .copied()
            .zip(self.vectors.rows())
            .map(|(id, vector)| PointRef { id, vector })
    }

    pub fn get(&self, id: I) -> Option<&[f32]> {
        self.rows.get(&id).map(|&row| self.vectors.row(row))
    }
}
