/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Borrowed clustering views and a reusable canonical insertion batch.

use diskann_utils::views::rowmajor::{self, Matrix};
use hashbrown::HashMap;

use crate::{ANNError, ANNResult, utils::VectorId};

/// A point's identity and borrowed canonical vector.
#[derive(Debug, Clone, Copy)]
pub struct PointRef<'a, I> {
    pub id: I,
    pub vector: &'a [f32],
}

/// The complete operation-local membership of one list.
///
/// Only reference metadata is owned; point coordinates are borrowed.
#[derive(Debug)]
pub struct ListView<'a, I, L> {
    pub id: L,
    pub points: Vec<PointRef<'a, I>>,
}

/// Lists supplied by a build accessor, in the order requested by `fill`.
pub type WorkingSet<'a, I, L> = Vec<ListView<'a, I, L>>;

/// Canonical rows shared between routing and a build accessor's working-set seed.
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
