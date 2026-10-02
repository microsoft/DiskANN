/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The online insert algorithm: stage and route a batch, then split every list the
//! batch overflows.
//!
//! Planning performs all of its accessor reads first and then computes the update
//! from owned, densely indexed data. Everything read from the accessor is validated
//! as it arrives, so the updates built from it need no further checks. Nothing here
//! mutates the index; the caller applies the planned update.

mod batch;
mod kernels;
mod split;

use std::fmt::{Debug, Display};

pub(super) use batch::StagedBatch;
pub(super) use split::SplitPlan;

use diskann_utils::views::Matrix;

use crate::{
    ANNError, ANNErrorKind, ANNResult,
    error::ErrorExt,
    ivf::dynamic::{CentroidIndex, MaintenanceAccessor, Provider},
    utils::VectorId,
};

#[track_caller]
pub(super) fn index_error<D>(message: D) -> ANNError
where
    D: Display + Debug + Send + Sync + 'static,
{
    ANNError::message(ANNErrorKind::IndexError, message)
}

/// Check that the accessor reserved `expected` distinct list ids, none of them live.
#[track_caller]
pub(super) fn check_reserved<C: CentroidIndex>(
    centroids: &C,
    ids: &[C::ListId],
    expected: usize,
) -> ANNResult<()> {
    if ids.len() != expected {
        return Err(index_error(format!(
            "accessor reserved {} list ids, expected {expected}",
            ids.len()
        )));
    }
    if let Some(id) = first_repeat(ids.iter().copied()) {
        return Err(index_error(format!("accessor reserved list id {id} twice")));
    }
    if let Some(id) = ids.iter().find(|&&id| centroids.centroid(id).is_some()) {
        return Err(index_error(format!("accessor reserved live list id {id}")));
    }
    Ok(())
}

/// The first value that occurs more than once.
fn first_repeat<V: Copy + Ord>(values: impl Iterator<Item = V>) -> Option<V> {
    let mut sorted: Vec<V> = values.collect();
    sorted.sort_unstable();
    sorted
        .windows(2)
        .find(|pair| pair[0] == pair[1])
        .map(|pair| pair[0])
}

/// Read the canonical vector of every visible point in `ids`; row `i` belongs to
/// `ids[i]`.
///
/// # Errors
///
/// Fails if the accessor fails, or leaves a row unwritten or non-finite.
async fn read_rows<P, A>(
    accessor: &mut A,
    ids: &[P::InternalId],
    dim: usize,
) -> ANNResult<Matrix<f32>>
where
    P: Provider,
    A: MaintenanceAccessor<P>,
{
    let mut rows = Matrix::new(f32::NAN, ids.len(), dim);
    accessor
        .read_vectors(ids, rows.as_mut_view())
        .await
        .escalate("maintenance must read canonical vectors")?;
    check_rows(ids, &rows)?;
    Ok(rows)
}

/// Check that every row the accessor was asked to write is finite.
///
/// Callers fill rows with NaN before handing them to the accessor, so rows it never
/// writes fail this check too.
fn check_rows<Id: VectorId>(ids: &[Id], rows: &Matrix<f32>) -> ANNResult<()> {
    match ids
        .iter()
        .zip(rows.row_iter())
        .find(|(_, row)| row.iter().any(|x| !x.is_finite()))
    {
        Some((id, _)) => Err(index_error(format!(
            "no finite canonical vector for point {id}"
        ))),
        None => Ok(()),
    }
}
