/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-means fitting and region-wide assignment over borrowed point rows.

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::distance::SquaredL2;
use rand::{Rng, rngs::StdRng};

use super::workingset::{ListView, PointRef};
use crate::{ANNError, ANNResult};

mod assignment;

pub(super) struct Children<I> {
    pub centroids: rowmajor::Owned<f32>,
    pub members: Vec<Vec<I>>,
}

/// Fit two children per source, then assign every point across all children.
pub(super) fn split<I: Copy, L>(
    lists: &[ListView<'_, I, L>],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
) -> ANNResult<Children<I>> {
    if lists.is_empty() || dim == 0 {
        return Err(ANNError::message(
            "IVF splitting needs a nonempty working set and positive dimension",
        ));
    }
    let mut centroids = rowmajor::Owned::try_from_element(2 * lists.len(), dim, 0.0)?;
    let mut sums = vec![0.0; 2 * dim];
    for (list, centers) in lists
        .iter()
        .zip(centroids.as_mut_slice().chunks_exact_mut(2 * dim))
    {
        fit_two_means(&list.points, centers, dim, iterations, rng, &mut sums)?;
    }

    let count = lists.iter().map(|list| list.points.len()).sum();
    let mut assignments = Vec::with_capacity(count);
    let mut sizes = vec![0; centroids.nrows()];
    // Reuse one distance row for every point; never materialize points x children.
    let mut distances = vec![0.0; centroids.nrows()];
    for point in lists.iter().flat_map(|list| &list.points) {
        let to = assignment::nearest(point.vector, centroids.as_slice(), dim, &mut distances);
        assignments.push(to);
        sizes[to] += 1;
    }
    let mut members: Vec<Vec<I>> = sizes.into_iter().map(Vec::with_capacity).collect();
    for (point, to) in lists.iter().flat_map(|list| &list.points).zip(assignments) {
        members[to].push(point.id);
    }
    Ok(Children { centroids, members })
}

fn fit_two_means<I>(
    points: &[PointRef<'_, I>],
    centers: &mut [f32],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
    sums: &mut [f64],
) -> ANNResult<()> {
    fit_two_means_with(points, centers, dim, iterations, rng, sums, nearest_two)
}

// A compile-time selector lets tests compare the original two-distance path
// without duplicating Lloyd's state transitions or introducing dynamic dispatch.
fn fit_two_means_with<I>(
    points: &[PointRef<'_, I>],
    centers: &mut [f32],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
    sums: &mut [f64],
    select: impl Fn(&[f32], &[f32], usize) -> usize,
) -> ANNResult<()> {
    if points.len() < 2 {
        return Err(ANNError::message("an IVF split needs at least two points"));
    }
    let first = rng.random_range(0..points.len());
    let second = (first + rng.random_range(1..points.len())) % points.len();
    centers[..dim].copy_from_slice(points[first].vector);
    centers[dim..].copy_from_slice(points[second].vector);

    for _ in 0..iterations.max(1) {
        sums.fill(0.0);
        let mut counts = [0; 2];
        for point in points {
            let to = select(point.vector, centers, dim);
            counts[to] += 1;
            for (sum, &value) in sums[to * dim..(to + 1) * dim].iter_mut().zip(point.vector) {
                // A finite mean must not overflow while accumulating its inputs.
                *sum += f64::from(value);
            }
        }
        for ((center, sum), count) in centers
            .chunks_exact_mut(dim)
            .zip(sums.chunks_exact(dim))
            .zip(counts)
        {
            if count != 0 {
                for (value, &sum) in center.iter_mut().zip(sum) {
                    *value = (sum / count as f64) as f32;
                }
            }
        }
    }
    Ok(())
}

#[inline]
fn nearest_two(point: &[f32], centers: &[f32], dim: usize) -> usize {
    let [first, second] = SquaredL2::evaluate_pair(point, &centers[..dim], &centers[dim..]);
    usize::from(second.total_cmp(&first).is_lt())
}

#[cfg(test)]
mod tests;
