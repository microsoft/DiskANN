/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Assignment kernels share the ordered f64 accumulation used by Lloyd updates.

use diskann_linalg::{Transpose, sgemm};
use diskann_vector::{
    MathematicalValue, PureDistanceFunction,
    distance::{InnerProduct, SquaredL2},
};
use diskann_wide::{ARCH, Architecture};

use super::{
    Strategy,
    batched::Batched,
    two_means::{TwoMeansIteration, sum_assignment},
};
use crate::{ANNError, ANNResult, ivf::workingset::PointRef};

#[inline]
pub(super) fn choose(first: f32, second: f32) -> u8 {
    // Explicit equal-zero handling also covers SGEMM scores of opposite zero signs.
    u8::from(first != second && second.total_cmp(&first).is_lt())
}

#[inline]
pub(super) fn paired(point: &[f32], centers: &[f32], dim: usize) -> u8 {
    let [a, b] = SquaredL2::evaluate_pair(point, &centers[..dim], &centers[dim..]);
    choose(a, b)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    let value: MathematicalValue<f32> = InnerProduct::evaluate(left, right);
    value.into_inner()
}

pub(super) struct Assignment<'a> {
    strategy: Strategy,
    packed: &'a [f32],
    assignments: Vec<u8>,
    scores: Vec<f32>,
    delta: Vec<f32>,
}

impl<'a> Assignment<'a> {
    pub(super) fn new(strategy: Strategy, rows: usize, dim: usize, packed: &'a [f32]) -> Self {
        Self {
            strategy,
            packed,
            assignments: if matches!(strategy, Strategy::Batched4 | Strategy::Batched8) {
                vec![0; rows]
            } else {
                Vec::new()
            },
            scores: if strategy == Strategy::Sgemm {
                vec![0.0; 2 * rows]
            } else {
                Vec::new()
            },
            delta: if strategy == Strategy::CenterDifference {
                vec![0.0; dim]
            } else {
                Vec::new()
            },
        }
    }
}

impl<I> TwoMeansIteration<I> for Assignment<'_> {
    fn assign_and_sum(
        &mut self,
        points: &[PointRef<'_, I>],
        centers: &[f32],
        dim: usize,
        sums: &mut [f64],
        counts: &mut [usize; 2],
    ) -> ANNResult<()> {
        match self.strategy {
            Strategy::Separate => {
                for point in points {
                    let a: f32 = SquaredL2::evaluate(point.vector, &centers[..dim]);
                    let b: f32 = SquaredL2::evaluate(point.vector, &centers[dim..]);
                    sum_assignment(point.vector, usize::from(choose(a, b)), dim, sums, counts);
                }
            }
            Strategy::Batched4 | Strategy::Batched8 => {
                if self.strategy == Strategy::Batched4 {
                    ARCH.run3(
                        Batched::<4> { dim },
                        points,
                        centers,
                        self.assignments.as_mut_slice(),
                    );
                } else {
                    ARCH.run3(
                        Batched::<8> { dim },
                        points,
                        centers,
                        self.assignments.as_mut_slice(),
                    );
                }
                for (point, &to) in points.iter().zip(&self.assignments) {
                    sum_assignment(point.vector, usize::from(to), dim, sums, counts);
                }
            }
            Strategy::Sgemm => {
                sgemm(
                    Transpose::None,
                    Transpose::Ordinary,
                    points.len(),
                    2,
                    dim,
                    -2.0,
                    self.packed,
                    centers,
                    None,
                    &mut self.scores,
                )
                .map_err(|error| ANNError::message(error.to_string()))?;
                let norms = [
                    dot(&centers[..dim], &centers[..dim]),
                    dot(&centers[dim..], &centers[dim..]),
                ];
                for (point, scores) in points.iter().zip(self.scores.chunks_exact(2)) {
                    let a = scores[0] + norms[0];
                    let b = scores[1] + norms[1];
                    let to = if a.is_finite() && b.is_finite() {
                        choose(a, b)
                    } else {
                        paired(point.vector, centers, dim)
                    };
                    sum_assignment(point.vector, usize::from(to), dim, sums, counts);
                }
            }
            Strategy::CenterDifference => {
                for ((out, &a), &b) in self
                    .delta
                    .iter_mut()
                    .zip(&centers[..dim])
                    .zip(&centers[dim..])
                {
                    *out = b - a;
                }
                let threshold =
                    dot(&centers[dim..], &centers[dim..]) - dot(&centers[..dim], &centers[..dim]);
                for point in points {
                    let difference = threshold - 2.0 * dot(point.vector, &self.delta);
                    let to = if difference.is_finite() {
                        u8::from(difference < 0.0)
                    } else {
                        paired(point.vector, centers, dim)
                    };
                    sum_assignment(point.vector, usize::from(to), dim, sums, counts);
                }
            }
            Strategy::PairedPoint | Strategy::PackedPaired => {
                return Err(ANNError::message(
                    "paired strategy must use the production fitter",
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
