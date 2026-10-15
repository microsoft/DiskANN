/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-means fitting and region-wide assignment over borrowed point rows.

use std::collections::BTreeSet;

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use rand::{Rng, rngs::StdRng};

use super::{
    partition::{Partition, Partitioning},
    region::{RegionUpdate, RegionView},
};
use crate::{ANNError, ANNResult};

/// Fit two children per parent and reassign sources across children and retained neighbors.
///
/// Parents and neighbors are distinct, disjoint filled sources. Neighbor centroids
/// are retained unchanged. Other filled lists are not read or edited.
/// Computation completes before any edits are recorded. A recording error may
/// leave partial pending edits, which the caller must discard rather than publish.
pub(super) fn split<I: Copy, L: Copy, R: RegionView<I, L>, U: RegionUpdate<I, L>>(
    region: &R,
    update: &mut U,
    parents: &BTreeSet<L>,
    neighbors: &BTreeSet<L>,
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
) -> ANNResult<()> {
    let result = compute_split(region, parents, neighbors, dim, iterations, rng)?;
    for (centroid, partition) in result.centroids.rows().zip(result.partitions) {
        match partition.id {
            Some(id) => update.set_members(id, &partition.members)?,
            None => update.create(centroid, &partition.members)?,
        }
    }
    for &parent in parents {
        update.retire(parent)?;
    }
    Ok(())
}

fn compute_split<I: Copy, L: Copy, R: RegionView<I, L>>(
    region: &R,
    parents: &BTreeSet<L>,
    neighbors: &BTreeSet<L>,
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
) -> ANNResult<Partitioning<I, L>> {
    if parents.is_empty() || dim == 0 {
        return Err(ANNError::message(
            "IVF splitting needs parents and a positive dimension",
        ));
    }

    let child_count = 2 * parents.len();
    let mut centroids = rowmajor::Owned::try_from_element(child_count + neighbors.len(), dim, 0.0)?;
    let mut inputs = Vec::with_capacity(parents.len() + neighbors.len());
    for (index, &source) in parents.iter().chain(neighbors).enumerate() {
        let view = region
            .get_view(source)
            .ok_or_else(|| ANNError::message("IVF split source is missing from the region"))?;
        let vectors = region.get_vectors(source).ok_or_else(|| {
            ANNError::message("IVF split source vectors are missing from the region")
        })?;
        if index >= parents.len() {
            centroids
                .row_mut(child_count + index - parents.len())
                .copy_from_slice(view.centroid);
        }
        inputs.push((view.members, vectors));
    }

    let mut sums = vec![0.0; 2 * dim];
    for ((_, vectors), centers) in inputs[..parents.len()]
        .iter()
        .zip(centroids.as_mut_slice()[..child_count * dim].chunks_exact_mut(2 * dim))
    {
        fit_two_means(vectors.as_view(), centers, dim, iterations, rng, &mut sums)?;
    }

    let count = inputs.iter().map(|(members, _)| members.len()).sum();
    let mut assignments = Vec::with_capacity(count);
    let mut sizes = vec![0; centroids.nrows()];
    for (_, vectors) in &inputs {
        for vector in vectors.rows() {
            let to = nearest(vector, centroids.as_slice(), dim);
            assignments.push(to);
            sizes[to] += 1;
        }
    }
    let mut partitions: Vec<Partition<I, L>> = std::iter::repeat_n(None, child_count)
        .chain(neighbors.iter().copied().map(Some))
        .zip(sizes)
        .map(|(id, size)| Partition {
            id,
            members: Vec::with_capacity(size),
        })
        .collect();
    for (id, to) in inputs
        .iter()
        .flat_map(|(members, _)| members.iter().copied())
        .zip(assignments)
    {
        partitions[to].members.push(id);
    }
    Ok(Partitioning {
        centroids,
        partitions,
    })
}

fn fit_two_means(
    points: rowmajor::Ref<'_, f32>,
    centers: &mut [f32],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
    sums: &mut [f64],
) -> ANNResult<()> {
    if points.nrows() < 2 {
        return Err(ANNError::message("an IVF split needs at least two points"));
    }
    let first = rng.random_range(0..points.nrows());
    let second = (first + rng.random_range(1..points.nrows())) % points.nrows();
    centers[..dim].copy_from_slice(points.row(first));
    centers[dim..].copy_from_slice(points.row(second));

    for _ in 0..iterations.max(1) {
        sums.fill(0.0);
        let mut counts = [0; 2];
        for point in points.rows() {
            let to = nearest(point, centers, dim);
            counts[to] += 1;
            for (sum, &value) in sums[to * dim..(to + 1) * dim].iter_mut().zip(point) {
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
