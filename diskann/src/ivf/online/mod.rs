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
mod gather;
mod kernels;
mod split;

use std::fmt::{Debug, Display};

pub(super) use batch::StagedBatch;
pub(super) use split::{Parent, SplitPlan};

use crate::{ANNError, ANNErrorKind, ANNResult, ivf::dynamic::CentroidIndex};

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
