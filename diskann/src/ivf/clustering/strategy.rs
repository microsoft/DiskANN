/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Opt-in fitting with the default distance arithmetic and shared Lloyd updates.

use diskann_vector::distance::TwoPointSquaredL2;
use rand::rngs::StdRng;

use super::two_means::{
    TwoMeansIteration, fit_two_means, fit_two_means_using, nearest_two, sum_assignment,
};
use crate::{ANNError, ANNResult, ivf::workingset::PointRef};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    #[default]
    PairedPoint,
    /// Borrow two rows at a time, keeping each point's original reduction tree.
    TiledExact,
}

/// A validated parent bucket. The candidate selects an ISA once per preparation;
/// it neither packs coordinates nor allocates per-point scratch.
pub struct TwoMeans<'a, I> {
    points: &'a [PointRef<'a, I>],
    dim: usize,
    tiled: Option<TwoPointSquaredL2>,
}

impl<'a, I> TwoMeans<'a, I> {
    /// The caller supplies finite coordinates. Both strategies preserve baseline
    /// distance bits, including overflow, on the compiled architecture.
    ///
    /// # Errors
    /// Rejects fewer than two rows, inconsistent/zero dimensions, or shape overflow.
    pub fn new(points: &'a [PointRef<'a, I>], dim: usize, strategy: Strategy) -> ANNResult<Self> {
        validate_shape(points, dim)?;
        Ok(Self {
            points,
            dim,
            tiled: (strategy == Strategy::TiledExact).then(TwoPointSquaredL2::new),
        })
    }

    /// Neither strategy owns a packed coordinate buffer.
    pub fn packed_bytes(&self) -> usize {
        0
    }

    /// Fit with original seeding, iteration count, ordered f64 sums and empty-center
    /// behavior. Centers and sums are caller-owned; each must contain two full rows.
    ///
    /// # Errors
    /// Returns an error for mismatched scratch dimensions.
    pub fn fit_into(
        &self,
        centers: &mut [f32],
        iterations: usize,
        rng: &mut StdRng,
        sums: &mut [f64],
    ) -> ANNResult<()> {
        if centers.len() != 2 * self.dim || sums.len() != 2 * self.dim {
            return Err(ANNError::message(
                "two-means scratch must contain two full center rows",
            ));
        }
        if let Some(tiled) = &self.tiled {
            fit_two_means_using(
                self.points,
                centers,
                self.dim,
                iterations,
                rng,
                sums,
                &mut TiledAssignment(tiled),
            )
        } else {
            fit_two_means(self.points, centers, self.dim, iterations, rng, sums)
        }
    }

    /// Neither strategy allocates an assignment array in the fitting path.
    pub fn scratch_bytes(&self) -> usize {
        0
    }

    /// Assign at supplied centers. The exact kernel needs no numerical fallback,
    /// so the diagnostic fallback count is always zero.
    ///
    /// # Errors
    /// Returns an error for mismatched centers or output shape.
    pub fn assign(&self, centers: &[f32], out: &mut [u32]) -> ANNResult<usize> {
        if centers.len() != 2 * self.dim || out.len() != self.points.len() {
            return Err(ANNError::message("two-means assignment shape mismatch"));
        }
        if let Some(tiled) = &self.tiled {
            for_each_tiled(tiled, self.points, centers, self.dim, |row, to| {
                out[row] = to as u32;
            });
        } else {
            for (point, to) in self.points.iter().zip(out) {
                *to = nearest_two(point.vector, centers, self.dim) as u32;
            }
        }
        Ok(0)
    }
}

fn for_each_tiled<I>(
    tiled: &TwoPointSquaredL2,
    points: &[PointRef<'_, I>],
    centers: &[f32],
    dim: usize,
    mut visit: impl FnMut(usize, usize),
) {
    let mut pairs = points.chunks_exact(2);
    for (index, pair) in pairs.by_ref().enumerate() {
        let distances = tiled.evaluate(
            [pair[0].vector, pair[1].vector],
            &centers[..dim],
            &centers[dim..],
        );
        for (lane, [first, second]) in distances.into_iter().enumerate() {
            visit(
                2 * index + lane,
                usize::from(second.total_cmp(&first).is_lt()),
            );
        }
    }
    if let [last] = pairs.remainder() {
        visit(points.len() - 1, nearest_two(last.vector, centers, dim));
    }
}

struct TiledAssignment<'a>(&'a TwoPointSquaredL2);

impl<I> TwoMeansIteration<I> for TiledAssignment<'_> {
    fn assign_and_sum(
        &mut self,
        points: &[PointRef<'_, I>],
        centers: &[f32],
        dim: usize,
        sums: &mut [f64],
        counts: &mut [usize; 2],
    ) -> ANNResult<()> {
        for_each_tiled(self.0, points, centers, dim, |row, to| {
            // The callback visits 0, 1, 2, ...; tiling never reorders the sums.
            sum_assignment(points[row].vector, to, dim, sums, counts);
        });
        Ok(())
    }
}

fn validate_shape<I>(points: &[PointRef<'_, I>], dim: usize) -> ANNResult<()> {
    if points.len() < 2 || dim == 0 || points.iter().any(|point| point.vector.len() != dim) {
        return Err(ANNError::message(
            "two-means requires at least two rows of one positive dimension",
        ));
    }
    if points.len().checked_mul(dim).is_none() || dim.checked_mul(2).is_none() {
        return Err(ANNError::message("two-means dimensions overflow"));
    }
    Ok(())
}

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
    struct Assignment<F> {
        assign: F,
        out: Vec<u32>,
    }
    impl<I, F: FnMut(&[f32], &mut [u32]) -> ANNResult<()>> TwoMeansIteration<I> for Assignment<F> {
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
    let mut assignment = Assignment {
        assign,
        out: vec![u32::MAX; points.len()],
    };
    fit_two_means_using(points, centers, dim, iterations, rng, sums, &mut assignment)
}

#[cfg(test)]
mod tests;
