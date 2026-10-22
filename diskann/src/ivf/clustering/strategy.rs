/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Opt-in fitting strategies. Numerical choices belong to the clustering algorithm;
//! timing, diagnostic assignments, and CLI names belong to its callers.

use rand::rngs::StdRng;

use super::{
    assignment::Assignment,
    two_means::{fit_two_means, fit_two_means_using},
};
use crate::{ANNError, ANNResult, ivf::workingset::PointRef};

/// Experimental assignment strategy. Separate and the paired variants preserve
/// squared-L2 distance bits; batched and dot-product variants may change partitions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Two independent squared-L2 evaluations per point.
    Separate,
    #[default]
    PairedPoint,
    /// Copy each parent once, then reuse contiguous rows for distances and f64 sums.
    PackedPaired,
    Batched4,
    Batched8,
    Sgemm,
    CenterDifference,
}

/// One parent bucket prepared for a fitting strategy. Packing, where needed, is
/// performed once and reused across all Lloyd steps. No measurements are collected.
pub struct TwoMeans<'a, I> {
    points: &'a [PointRef<'a, I>],
    dim: usize,
    strategy: Strategy,
    packed: Vec<f32>,
}

impl<'a, I> TwoMeans<'a, I> {
    /// Validate row shapes and copy coordinates only for strategies that need packing.
    ///
    /// The caller supplies finite coordinates. Batched and dot-product strategies
    /// can change assignments even for finite inputs; they are not bitwise substitutes.
    ///
    /// # Errors
    /// Returns an error for fewer than two rows, inconsistent or zero dimensions,
    /// or overflowing dimensions.
    pub fn new(points: &'a [PointRef<'a, I>], dim: usize, strategy: Strategy) -> ANNResult<Self> {
        validate_shape(points, dim)?;
        let packed = if matches!(strategy, Strategy::PackedPaired | Strategy::Sgemm) {
            let mut packed = Vec::with_capacity(points.len() * dim);
            for point in points {
                packed.extend_from_slice(point.vector);
            }
            packed
        } else {
            Vec::new()
        };
        Ok(Self {
            points,
            dim,
            strategy,
            packed,
        })
    }

    /// Coordinate storage owned by this preparation, excluding scratch and row views.
    pub fn packed_bytes(&self) -> usize {
        std::mem::size_of_val(self.packed.as_slice())
    }

    /// Fit using caller-owned center and f64 sum buffers. Point order, seed draws,
    /// fixed iteration count and empty-center behavior are shared with default IVF.
    /// Dot-product strategies use paired L2 for nonfinite scores; finite cancellation
    /// remains observable. Both buffers must contain exactly two center rows.
    ///
    /// # Errors
    /// Returns an error for mismatched scratch dimensions or a failed kernel call.
    pub fn fit_into(
        &self,
        centers: &mut [f32],
        iterations: usize,
        rng: &mut StdRng,
        sums: &mut [f64],
    ) -> ANNResult<()> {
        let dim = self.dim;
        if centers.len() != 2 * dim || sums.len() != 2 * dim {
            return Err(ANNError::message(
                "two-means scratch must contain two full center rows",
            ));
        }
        match self.strategy {
            Strategy::PairedPoint => {
                fit_two_means(self.points, centers, dim, iterations, rng, sums)
            }
            Strategy::PackedPaired | Strategy::Sgemm => {
                let points: Vec<_> = self
                    .packed
                    .chunks_exact(dim)
                    .enumerate()
                    .map(|(id, vector)| PointRef { id, vector })
                    .collect();
                if self.strategy == Strategy::PackedPaired {
                    fit_two_means(&points, centers, dim, iterations, rng, sums)
                } else {
                    let mut assignment =
                        Assignment::new(self.strategy, points.len(), dim, &self.packed);
                    fit_two_means_using(
                        &points,
                        centers,
                        dim,
                        iterations,
                        rng,
                        sums,
                        &mut assignment,
                    )
                }
            }
            _ => {
                let mut assignment = Assignment::new(self.strategy, self.points.len(), dim, &[]);
                fit_two_means_using(
                    self.points,
                    centers,
                    dim,
                    iterations,
                    rng,
                    sums,
                    &mut assignment,
                )
            }
        }
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

#[cfg(test)]
mod tests;
