/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-means fitting and region-wide assignment over borrowed point rows.

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use rand::{Rng, rngs::StdRng};

use super::{
    partition::{Partition, Partitioning},
    workingset::{ListView, PointRef},
};
use crate::{ANNError, ANNResult};

/// Replace each parent with two children, retaining the other source centroids.
///
/// `parents` is a distinct subset of `lists`. All regional points choose among
/// the children and retained centroids, never among retiring parent centroids.
pub(super) fn split<I: Copy, L: Copy + Eq>(
    lists: &[ListView<'_, I, L>],
    parents: &[L],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
) -> ANNResult<Partitioning<I, L>> {
    if lists.is_empty() || dim == 0 {
        return Err(ANNError::message(
            "IVF splitting needs a nonempty working set and positive dimension",
        ));
    }
    let destinations = lists.len() + parents.len();
    let mut centroids = rowmajor::Owned::try_from_element(destinations, dim, 0.0)?;
    let mut partitions = Vec::with_capacity(destinations);
    let mut sums = vec![0.0; 2 * dim];
    for list in lists {
        let row = partitions.len();
        if parents.contains(&list.id) {
            let centers = &mut centroids.as_mut_slice()[row * dim..(row + 2) * dim];
            fit_two_means(&list.points, centers, dim, iterations, rng, &mut sums)?;
            partitions.extend((0..2).map(|_| Partition {
                id: None,
                members: Vec::new(),
            }));
        } else {
            centroids.row_mut(row).copy_from_slice(list.centroid);
            partitions.push(Partition {
                id: Some(list.id),
                members: Vec::new(),
            });
        }
    }

    let count = lists.iter().map(|list| list.points.len()).sum();
    let mut assignments = Vec::with_capacity(count);
    let mut sizes = vec![0; centroids.nrows()];
    for point in lists.iter().flat_map(|list| &list.points) {
        let to = nearest(point.vector, centroids.as_slice(), dim);
        assignments.push(to);
        sizes[to] += 1;
    }
    for (partition, size) in partitions.iter_mut().zip(sizes) {
        partition.members = Vec::with_capacity(size);
    }
    for (point, to) in lists.iter().flat_map(|list| &list.points).zip(assignments) {
        partitions[to].members.push(point.id);
    }
    Ok(Partitioning {
        centroids,
        partitions,
    })
}

fn fit_two_means<I>(
    points: &[PointRef<'_, I>],
    centers: &mut [f32],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
    sums: &mut [f64],
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
            let to = nearest(point.vector, centers, dim);
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
