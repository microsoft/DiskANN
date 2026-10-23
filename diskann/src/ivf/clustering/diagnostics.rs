/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Assignment hooks for tracing the shared Lloyd loop outside timed fitting.

use rand::rngs::StdRng;

use super::{
    strategy::validate_shape,
    two_means::{TwoMeansIteration, fit_two_means_using, sum_assignment},
};
use crate::{ANNError, ANNResult, ivf::workingset::PointRef};

/// Fit with caller-supplied assignment while preserving IVF's seeding, ordered
/// f64 sums, fixed steps, and empty-center rule. Diagnostic scratch is allocated
/// once; the callback must write 0 or 1 for every point.
///
/// # Errors
/// Rejects invalid shapes/center IDs and propagates assignment errors.
pub fn fit_with_assignment<I>(
    points: &[PointRef<'_, I>],
    centers: &mut [f32],
    iterations: usize,
    rng: &mut StdRng,
    sums: &mut [f64],
    assign: impl FnMut(&[f32], &mut [u32]) -> ANNResult<()>,
) -> ANNResult<()> {
    let dim = points.first().map_or(0, |point| point.vector.len());
    validate_shape(points, dim)?;
    if centers.len() != 2 * dim || sums.len() != 2 * dim {
        return Err(ANNError::message(
            "two-means scratch must contain two full center rows",
        ));
    }
    let mut assignment = Assignment {
        assign,
        out: vec![u32::MAX; points.len()],
    };
    fit_two_means_using(points, centers, dim, iterations, rng, sums, &mut assignment)
}

struct Assignment<F> {
    assign: F,
    out: Vec<u32>,
}

impl<I, F> TwoMeansIteration<I> for Assignment<F>
where
    F: FnMut(&[f32], &mut [u32]) -> ANNResult<()>,
{
    fn assign_and_sum(
        &mut self,
        points: &[PointRef<'_, I>],
        centers: &[f32],
        dim: usize,
        sums: &mut [f64],
        counts: &mut [usize; 2],
    ) -> ANNResult<()> {
        self.out.fill(u32::MAX);
        (self.assign)(centers, &mut self.out)?;
        if self.out.iter().any(|&id| id > 1) {
            return Err(ANNError::message(
                "two-means assignment did not select a valid center for every point",
            ));
        }
        for (point, &to) in points.iter().zip(&self.out) {
            sum_assignment(point.vector, to as usize, dim, sums, counts);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
