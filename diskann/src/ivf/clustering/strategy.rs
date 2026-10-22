/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Opt-in fitting strategies. Numerical choices belong to the clustering algorithm;
//! timing, diagnostic assignments, and CLI names belong to its callers.

use rand::rngs::StdRng;

use diskann_quantization::{
    algorithms::kmeans::two_means as kernel, multi_vector::BlockTransposed,
};

use super::two_means::{
    TwoMeansIteration, fit_two_means, fit_two_means_using, nearest_two, sum_assignment,
};
use crate::{ANNError, ANNResult, ivf::workingset::PointRef};

/// The baseline and block-transposed, cross-point SIMD candidates. Transposed
/// candidates change the reduction order and may change partitions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    #[default]
    PairedPoint,
    TransposedL2,
    TransposedDot,
}

/// One parent bucket prepared for a fitting strategy. Packing, where needed, is
/// performed once and reused across all Lloyd steps. No measurements are collected.
pub struct TwoMeans<'a, I> {
    points: &'a [PointRef<'a, I>],
    dim: usize,
    strategy: Strategy,
    packed: Option<BlockTransposed<f32, 16>>,
}

impl<'a, I> TwoMeans<'a, I> {
    /// Validate row shapes and copy coordinates only for strategies that need packing.
    ///
    /// The caller supplies finite coordinates. Transposed strategies
    /// can change assignments even for finite inputs; they are not bitwise substitutes.
    ///
    /// # Errors
    /// Returns an error for fewer than two rows, inconsistent or zero dimensions,
    /// or overflowing dimensions.
    pub fn new(points: &'a [PointRef<'a, I>], dim: usize, strategy: Strategy) -> ANNResult<Self> {
        validate_shape(points, dim)?;
        let packed = if strategy != Strategy::PairedPoint {
            Some(
                kernel::pack_rows(points.iter().map(|point| point.vector), dim)
                    .map_err(ANNError::new)?,
            )
        } else {
            None
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
        self.packed
            .as_ref()
            .map_or(0, |packed| std::mem::size_of_val(packed.as_slice()))
    }

    /// Fit using caller-owned center and f64 sum buffers. Point order, seed draws,
    /// fixed iteration count and empty-center behavior are shared with default IVF.
    /// Transposed strategies use paired L2 for nonfinite scores; finite cancellation
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
            Strategy::TransposedL2 | Strategy::TransposedDot => {
                let mut assignment = TransposedAssignment {
                    prepared: self,
                    assignments: vec![0; self.points.len()],
                };
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

    /// Assignment output bytes allocated once by each fit and reused across steps.
    pub fn scratch_bytes(&self) -> usize {
        if self.packed.is_some() {
            self.points.len() * std::mem::size_of::<u32>()
        } else {
            0
        }
    }

    /// Assign at supplied centers without updating them; shared by fitting and diagnostics.
    /// A nonfinite score is recomputed with baseline paired L2. Returns fallback count.
    ///
    /// # Errors
    /// Returns an error for mismatched centers or output shape, or a kernel failure.
    pub fn assign(&self, centers: &[f32], out: &mut [u32]) -> ANNResult<usize> {
        if centers.len() != 2 * self.dim || out.len() != self.points.len() {
            return Err(ANNError::message("two-means assignment shape mismatch"));
        }
        if let Some(packed) = &self.packed {
            let metric = match self.strategy {
                Strategy::TransposedL2 => kernel::Metric::SquaredL2,
                Strategy::TransposedDot => kernel::Metric::DotProduct,
                Strategy::PairedPoint => unreachable!("paired strategy does not pack"),
            };
            kernel::assign(packed.as_view(), centers, metric, out).map_err(ANNError::new)?;
            let mut fallbacks = 0;
            for (point, to) in self.points.iter().zip(out) {
                if *to == kernel::INVALID {
                    *to = nearest_two(point.vector, centers, self.dim) as u32;
                    fallbacks += 1;
                }
            }
            Ok(fallbacks)
        } else {
            for (point, to) in self.points.iter().zip(out) {
                *to = nearest_two(point.vector, centers, self.dim) as u32;
            }
            Ok(0)
        }
    }
}

struct TransposedAssignment<'a, 'b, I> {
    prepared: &'b TwoMeans<'a, I>,
    assignments: Vec<u32>,
}

impl<I> TwoMeansIteration<I> for TransposedAssignment<'_, '_, I> {
    fn assign_and_sum(
        &mut self,
        points: &[PointRef<'_, I>],
        centers: &[f32],
        dim: usize,
        sums: &mut [f64],
        counts: &mut [usize; 2],
    ) -> ANNResult<()> {
        self.prepared.assign(centers, &mut self.assignments)?;
        for (point, &to) in points.iter().zip(&self.assignments) {
            sum_assignment(point.vector, to as usize, dim, sums, counts);
        }
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

/// Fit with a caller-supplied assignment operation while preserving IVF's seeding,
/// ordered f64 sums, fixed steps, and empty-center rule. Assignment scratch is
/// allocated once and reused; the callback must write 0 or 1 for every point.
///
/// # Errors
/// Rejects invalid shapes or center IDs and propagates assignment errors.
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
            self.out.fill(kernel::INVALID);
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
        out: vec![kernel::INVALID; points.len()],
    };
    fit_two_means_using(points, centers, dim, iterations, rng, sums, &mut assignment)
}

#[cfg(test)]
mod tests;
