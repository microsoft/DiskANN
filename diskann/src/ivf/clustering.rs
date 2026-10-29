/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-means fitting and region-wide assignment over borrowed point rows.

use std::time::{Duration, Instant};

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use rand::rngs::StdRng;

use super::{metrics::ComputeStats, workingset::ListView};

use crate::{ANNError, ANNResult};

mod reassignment;
mod two_means;
pub use reassignment::ReassignmentMode;
pub use two_means::TwoMeansMode;

pub(super) struct Children<I> {
    pub centroids: rowmajor::Owned<f32>,
    pub members: Vec<Vec<I>>,
    pub fit_time: Duration,
    pub assignment_time: Duration,
    pub compute: ComputeStats,
}

/// Fit two children per source, then assign every point across all children.
pub(super) fn split<I: Copy + Sync, L: Sync>(
    lists: &[ListView<'_, I, L>],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
    mode: TwoMeansMode,
    reassignment_mode: ReassignmentMode,
) -> ANNResult<Children<I>> {
    if lists.is_empty() || dim == 0 {
        return Err(ANNError::message(
            "IVF splitting needs a nonempty working set and positive dimension",
        ));
    }
    let mut centroids = rowmajor::Owned::try_from_element(2 * lists.len(), dim, 0.0)?;
    let fit_start = Instant::now();
    let mut compute = two_means::fit(lists, centroids.as_mut_slice(), dim, iterations, rng, mode)?;

    let fit_time = fit_start.elapsed();
    let assignment_start = Instant::now();
    let count = lists.iter().map(|list| list.points.len()).sum();
    let mut assignments = Vec::with_capacity(count);
    let mut sizes = vec![0; centroids.nrows()];
    compute.merge(reassignment::assign(
        lists,
        centroids.as_slice(),
        dim,
        reassignment_mode,
        &mut assignments,
        &mut sizes,
    )?);
    let assignment_time = assignment_start.elapsed();
    let mut members: Vec<Vec<I>> = sizes.into_iter().map(Vec::with_capacity).collect();
    for (point, to) in lists.iter().flat_map(|list| &list.points).zip(assignments) {
        members[to].push(point.id);
    }
    Ok(Children {
        centroids,
        members,
        fit_time,
        assignment_time,
        compute,
    })
}

#[cfg(feature = "ivf-parallel")]
pub(super) fn use_parallel(coordinate_visits: usize) -> bool {
    // Only use a caller's active Rayon pool. Do not implicitly create a global pool.
    coordinate_visits >= 262_144
        && rayon::current_thread_index().is_some()
        && rayon::current_num_threads() > 1
}
