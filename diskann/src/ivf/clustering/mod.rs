/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-means fitting and region-wide assignment over borrowed point rows.

#[cfg(feature = "ivf-experimental")]
mod diagnostics;
#[cfg(feature = "ivf-experimental")]
mod strategy;
mod two_means;

#[cfg(feature = "ivf-experimental")]
pub use diagnostics::fit_with_assignment;
#[cfg(feature = "ivf-experimental")]
pub use strategy::{Strategy, TwoMeans};

use std::time::{Duration, Instant};

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use rand::rngs::StdRng;

use super::workingset::{ListView, PointRef};
use crate::{ANNError, ANNResult};
use two_means::fit_two_means;

pub(super) struct Children<I> {
    pub centroids: rowmajor::Owned<f32>,
    pub members: Vec<Vec<I>>,
    pub timing: SplitTiming,
    pub point_count: usize,
}

#[derive(Debug, Default)]
pub(super) struct SplitTiming {
    pub two_means: Duration,
    pub reassignment: Duration,
}

/// Fit two children per source, then assign every point across all children.
pub(super) fn split<I: Copy, L>(
    lists: &[ListView<'_, I, L>],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
) -> ANNResult<Children<I>> {
    split_using(lists, dim, iterations, rng, fit_two_means)
}

#[cfg(feature = "ivf-experimental")]
pub(super) fn split_with_strategy<I: Copy, L>(
    lists: &[ListView<'_, I, L>],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
    strategy: Strategy,
) -> ANNResult<Children<I>> {
    if strategy == Strategy::PairedPoint {
        return split(lists, dim, iterations, rng);
    }
    split_using(
        lists,
        dim,
        iterations,
        rng,
        |points, centers, dim, iterations, rng, sums| {
            TwoMeans::new(points, dim, strategy)?.fit_into(centers, iterations, rng, sums)
        },
    )
}

fn split_using<I: Copy, L>(
    lists: &[ListView<'_, I, L>],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
    mut fit: impl FnMut(
        &[PointRef<'_, I>],
        &mut [f32],
        usize,
        usize,
        &mut StdRng,
        &mut [f64],
    ) -> ANNResult<()>,
) -> ANNResult<Children<I>> {
    if lists.is_empty() || dim == 0 {
        return Err(ANNError::message(
            "IVF splitting needs a nonempty working set and positive dimension",
        ));
    }
    let two_means_start = Instant::now();
    let mut centroids = rowmajor::Owned::try_from_element(2 * lists.len(), dim, 0.0)?;
    let mut sums = vec![0.0; 2 * dim];
    for (list, centers) in lists
        .iter()
        .zip(centroids.as_mut_slice().chunks_exact_mut(2 * dim))
    {
        fit(&list.points, centers, dim, iterations, rng, &mut sums)?;
    }
    let two_means = two_means_start.elapsed();

    let reassignment_start = Instant::now();
    let count = lists.iter().map(|list| list.points.len()).sum();
    let mut assignments = Vec::with_capacity(count);
    let mut sizes = vec![0; centroids.nrows()];
    for point in lists.iter().flat_map(|list| &list.points) {
        let to = nearest(point.vector, centroids.as_slice(), dim);
        assignments.push(to);
        sizes[to] += 1;
    }
    let mut members: Vec<Vec<I>> = sizes.into_iter().map(Vec::with_capacity).collect();
    for (point, to) in lists.iter().flat_map(|list| &list.points).zip(assignments) {
        members[to].push(point.id);
    }
    let reassignment = reassignment_start.elapsed();
    Ok(Children {
        centroids,
        members,
        timing: SplitTiming {
            two_means,
            reassignment,
        },
        point_count: count,
    })
}

/// Prefer the first centroid on ties. `centers` contains at least one full row.
fn nearest(point: &[f32], centers: &[f32], dim: usize) -> usize {
    let mut best = 0;
    let mut best_distance: f32 = SquaredL2::evaluate(point, &centers[..dim]);
    for (index, center) in centers.chunks_exact(dim).enumerate().skip(1) {
        let distance: f32 = SquaredL2::evaluate(point, center);
        if distance.total_cmp(&best_distance).is_lt() {
            best = index;
            best_distance = distance;
        }
    }
    best
}

#[cfg(test)]
mod tests;
