/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared Lloyd state transitions and the default paired-L2 assignment.

use diskann_vector::distance::SquaredL2;
use rand::{Rng, rngs::StdRng};

use crate::{ANNError, ANNResult, ivf::workingset::PointRef};

pub(super) fn fit_two_means<I>(
    points: &[PointRef<'_, I>],
    centers: &mut [f32],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
    sums: &mut [f64],
) -> ANNResult<()> {
    fit_two_means_using(
        points,
        centers,
        dim,
        iterations,
        rng,
        sums,
        &mut PairedPoint,
    )
}

pub(super) trait TwoMeansIteration<I> {
    fn assign_and_sum(
        &mut self,
        points: &[PointRef<'_, I>],
        centers: &[f32],
        dim: usize,
        sums: &mut [f64],
        counts: &mut [usize; 2],
    ) -> ANNResult<()>;
}

pub(super) fn fit_two_means_using<I>(
    points: &[PointRef<'_, I>],
    centers: &mut [f32],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
    sums: &mut [f64],
    strategy: &mut impl TwoMeansIteration<I>,
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
        let mut counts = [0usize; 2];
        strategy.assign_and_sum(points, centers, dim, sums, &mut counts)?;
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

pub(super) fn sum_assignment(
    point: &[f32],
    to: usize,
    dim: usize,
    sums: &mut [f64],
    counts: &mut [usize; 2],
) {
    counts[to] += 1;
    for (sum, &value) in sums[to * dim..(to + 1) * dim].iter_mut().zip(point) {
        // A finite mean must not overflow while accumulating its inputs.
        *sum += f64::from(value);
    }
}

struct PairedPoint;

impl<I> TwoMeansIteration<I> for PairedPoint {
    fn assign_and_sum(
        &mut self,
        points: &[PointRef<'_, I>],
        centers: &[f32],
        dim: usize,
        sums: &mut [f64],
        counts: &mut [usize; 2],
    ) -> ANNResult<()> {
        for point in points {
            let to = nearest_two(point.vector, centers, dim);
            sum_assignment(point.vector, to, dim, sums, counts);
        }
        Ok(())
    }
}

#[inline]
pub(super) fn nearest_two(point: &[f32], centers: &[f32], dim: usize) -> usize {
    let [first, second] = SquaredL2::evaluate_pair(point, &centers[..dim], &centers[dim..]);
    usize::from(second.total_cmp(&first).is_lt())
}
