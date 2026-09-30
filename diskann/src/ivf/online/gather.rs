/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Validated reads from a maintenance accessor.

use std::ops::Range;

use diskann_utils::views::Matrix;

use super::index_error;
use crate::{
    ANNResult,
    error::ErrorExt,
    ivf::{dynamic::MaintenanceAccessor, grouped::Csr},
    utils::VectorId,
};

/// Read the canonical vector of every point in `ids`; row `i` belongs to `ids[i]`.
///
/// # Errors
///
/// Fails if the accessor fails, or leaves a row unwritten or non-finite.
pub(super) async fn read_rows<A, T>(
    accessor: &mut A,
    ids: &[A::Id],
    dim: usize,
) -> ANNResult<Matrix<f32>>
where
    A: MaintenanceAccessor<T>,
    T: Send,
{
    // Rows the accessor never writes stay NaN, so the finiteness check catches them.
    let mut rows = Matrix::new(f32::NAN, ids.len(), dim);
    accessor
        .read_vectors(ids, rows.as_mut_view())
        .await
        .escalate("maintenance must read canonical vectors")?;

    if let Some((id, _)) = ids
        .iter()
        .zip(rows.row_iter())
        .find(|(_, row)| row.iter().any(|x| !x.is_finite()))
    {
        return Err(index_error(format!(
            "no finite canonical vector for point {id}"
        )));
    }
    Ok(rows)
}

/// The existing members of a table of lists and their canonical vectors.
///
/// List `i`'s members occupy positions [`Self::range`]`(i)`, which index both the ids
/// and the vector rows.
pub(super) struct Members<Id> {
    ids: Csr<Id>,
    vectors: Matrix<f32>,
}

impl<Id: VectorId> Members<Id> {
    /// Read the members of every list in `lists`, then their canonical vectors.
    ///
    /// # Errors
    ///
    /// Fails if the accessor fails or [`read_rows`] fails.
    pub(super) async fn read<A, T>(accessor: &mut A, lists: &[A::ListId]) -> ANNResult<Self>
    where
        A: MaintenanceAccessor<T, Id = Id>,
        T: Send,
    {
        let mut ids = Csr::default();
        for &list in lists {
            let members = accessor
                .read_members(list)
                .await
                .escalate("maintenance must read list members")?;
            ids.push(members.iter().copied());
        }

        let dim = accessor.dim();
        let vectors = read_rows(accessor, ids.values(), dim).await?;
        Ok(Self { ids, vectors })
    }

    /// Total number of members across all lists.
    pub(super) fn total(&self) -> usize {
        self.ids.values().len()
    }

    /// Positions of `list`'s members.
    pub(super) fn range(&self, list: usize) -> Range<usize> {
        self.ids.range(list)
    }

    /// Member ids of `list`.
    pub(super) fn ids(&self, list: usize) -> &[Id] {
        self.ids.group(list)
    }

    /// The canonical vector at member position `position`.
    pub(super) fn vector(&self, position: usize) -> &[f32] {
        self.vectors.row(position)
    }

    /// Canonical vectors of `list`'s members, in member order.
    pub(super) fn vectors(&self, list: usize) -> impl ExactSizeIterator<Item = &[f32]> {
        self.range(list).map(|position| self.vectors.row(position))
    }
}
