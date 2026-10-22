/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Opt-in two-means experiments. No candidate changes the default IVF algorithm.
//!
//! PairedPoint and PackedPaired retain the production squared-L2 arithmetic.
//! Batched and dot-product candidates change reduction order and may change
//! assignments even for finite inputs. They require quality checks before use.

use std::time::{Duration, Instant};

use diskann_linalg::{Transpose, sgemm};
use diskann_vector::{
    MathematicalValue, PureDistanceFunction,
    distance::{InnerProduct, SquaredL2},
};
use diskann_wide::{ARCH, Architecture, SIMDMulAdd, SIMDSumTree, SIMDVector, arch::Target3};
use rand::{SeedableRng, rngs::StdRng};

use super::{
    clustering::{self, TwoMeansIteration},
    workingset::PointRef,
};
use crate::{ANNError, ANNResult};

/// Experimental assignment strategy; Separate and the paired variants preserve distance bits.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// The first PR layer: two independent existing squared-L2 evaluations.
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

impl Strategy {
    pub const ALL: [Self; 7] = [
        Self::Separate,
        Self::PairedPoint,
        Self::PackedPaired,
        Self::Batched4,
        Self::Batched8,
        Self::Sgemm,
        Self::CenterDifference,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Separate => "separate",
            Self::PairedPoint => "paired_point",
            Self::PackedPaired => "packed_paired",
            Self::Batched4 => "batched_4",
            Self::Batched8 => "batched_8",
            Self::Sgemm => "sgemm",
            Self::CenterDifference => "center_difference",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|strategy| strategy.name() == name)
    }
}

/// Complete fitting wall time; diagnostic assignments are computed after the timer.
#[derive(Debug)]
pub struct Run {
    pub total: Duration,
    pub packing: Duration,
    pub centers: Vec<f32>,
    /// Production paired-L2 assignment to the returned centers, as in final reassignment.
    pub assignments: Vec<u8>,
    pub margins: Vec<f64>,
    pub packed_bytes: usize,
}

/// Measure one complete fit, including scratch allocation, packing, seeding, and updates.
///
/// Errors on empty/one-point buckets or mismatched/overflowing dimensions. The caller
/// must supply finite coordinates; the benchmark validates its corpus before timing.
/// Finite values large enough to overflow squared distances are supported; dot-product
/// candidates fall back to paired L2 if their scores become nonfinite. Finite cancellation
/// is deliberately visible in comparison results, not treated as bitwise equivalence.
pub fn fit(
    points: &[PointRef<'_, usize>],
    dim: usize,
    iterations: usize,
    seed: u64,
    strategy: Strategy,
) -> ANNResult<Run> {
    validate_shape(points, dim)?;
    let start = Instant::now();
    let mut centers = vec![0.0; 2 * dim];
    let mut sums = vec![0.0; 2 * dim];
    let mut rng = StdRng::seed_from_u64(seed);
    let (packing, packed_bytes) = fit_into(
        points,
        &mut centers,
        dim,
        iterations,
        &mut rng,
        &mut sums,
        strategy,
    )?;
    let total = start.elapsed();
    let mut assignments = Vec::with_capacity(points.len());
    let mut margins = Vec::with_capacity(points.len());
    for point in points {
        let [a, b] = SquaredL2::evaluate_pair(point.vector, &centers[..dim], &centers[dim..]);
        assignments.push(choose(a, b));
        margins.push((f64::from(b) - f64::from(a)).abs());
    }
    Ok(Run {
        total,
        packing,
        centers,
        assignments,
        margins,
        packed_bytes,
    })
}

pub(super) fn fit_into<I>(
    points: &[PointRef<'_, I>],
    centers: &mut [f32],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
    sums: &mut [f64],
    strategy: Strategy,
) -> ANNResult<(Duration, usize)> {
    if strategy == Strategy::PairedPoint {
        clustering::fit_two_means(points, centers, dim, iterations, rng, sums)?;
        return Ok((Duration::ZERO, 0));
    }
    if matches!(strategy, Strategy::PackedPaired | Strategy::Sgemm) {
        let packing_start = Instant::now();
        let mut packed = Vec::with_capacity(points.len() * dim);
        for point in points {
            packed.extend_from_slice(point.vector);
        }
        let packed_points: Vec<_> = packed
            .chunks_exact(dim)
            .enumerate()
            .map(|(id, vector)| PointRef { id, vector })
            .collect();
        let packing = packing_start.elapsed();
        if strategy == Strategy::PackedPaired {
            clustering::fit_two_means(&packed_points, centers, dim, iterations, rng, sums)?;
        } else {
            let mut runner = Runner::new(strategy, points.len(), dim, &packed);
            clustering::fit_two_means_using(
                &packed_points,
                centers,
                dim,
                iterations,
                rng,
                sums,
                &mut runner,
            )?;
        }
        return Ok((packing, std::mem::size_of_val(packed.as_slice())));
    }
    let mut runner = Runner::new(strategy, points.len(), dim, &[]);
    clustering::fit_two_means_using(points, centers, dim, iterations, rng, sums, &mut runner)?;
    Ok((Duration::ZERO, 0))
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

#[inline]
fn choose(first: f32, second: f32) -> u8 {
    // Explicit equal-zero handling also covers SGEMM scores of opposite zero signs.
    u8::from(first != second && second.total_cmp(&first).is_lt())
}

#[inline]
fn paired(point: &[f32], centers: &[f32], dim: usize) -> u8 {
    let [a, b] = SquaredL2::evaluate_pair(point, &centers[..dim], &centers[dim..]);
    choose(a, b)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    let value: MathematicalValue<f32> = InnerProduct::evaluate(left, right);
    value.into_inner()
}

struct Runner<'a> {
    strategy: Strategy,
    packed: &'a [f32],
    assignments: Vec<u8>,
    scores: Vec<f32>,
    delta: Vec<f32>,
}

impl<'a> Runner<'a> {
    fn new(strategy: Strategy, rows: usize, dim: usize, packed: &'a [f32]) -> Self {
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

impl<I> TwoMeansIteration<I> for Runner<'_> {
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
                    clustering::sum_assignment(
                        point.vector,
                        usize::from(choose(a, b)),
                        dim,
                        sums,
                        counts,
                    );
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
                    clustering::sum_assignment(point.vector, usize::from(to), dim, sums, counts);
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
                    clustering::sum_assignment(point.vector, usize::from(to), dim, sums, counts);
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
                    clustering::sum_assignment(point.vector, usize::from(to), dim, sums, counts);
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

#[derive(Debug, Clone, Copy)]
struct Batched<const ROWS: usize> {
    dim: usize,
}

impl<A: Architecture, I, const ROWS: usize> Target3<A, (), &[PointRef<'_, I>], &[f32], &mut [u8]>
    for Batched<ROWS>
{
    #[inline]
    fn run(self, arch: A, points: &[PointRef<'_, I>], centers: &[f32], assignments: &mut [u8]) {
        diskann_wide::alias!(V<A> = f32x8);
        // Shapes are checked before entering the fitter or by the provider working set.
        assert_eq!(centers.len(), 2 * self.dim);
        assert_eq!(assignments.len(), points.len());
        assert!(points.iter().all(|point| point.vector.len() == self.dim));
        let mut groups = points.chunks_exact(ROWS);
        let mut outputs = assignments.chunks_exact_mut(ROWS);
        for (group, output) in groups.by_ref().zip(outputs.by_ref()) {
            let mut first = [V::<A>::default(arch); ROWS];
            let mut second = first;
            let full = self.dim / V::<A>::LANES * V::<A>::LANES;
            for d in (0..full).step_by(V::<A>::LANES) {
                // SAFETY: Shapes were checked. Each complete SIMD load ends within dim.
                unsafe {
                    let a = V::<A>::load_simd(arch, centers.as_ptr().add(d));
                    let b = V::<A>::load_simd(arch, centers.as_ptr().add(self.dim + d));
                    for row in 0..ROWS {
                        let x = V::<A>::load_simd(arch, group[row].vector.as_ptr().add(d));
                        let da = x - a;
                        let db = x - b;
                        first[row] = da.mul_add_simd(da, first[row]);
                        second[row] = db.mul_add_simd(db, second[row]);
                    }
                }
            }
            let tail = self.dim - full;
            if tail != 0 {
                // SAFETY: The masked loads access exactly the tail coordinates.
                unsafe {
                    let a = V::<A>::load_simd_first(arch, centers.as_ptr().add(full), tail);
                    let b =
                        V::<A>::load_simd_first(arch, centers.as_ptr().add(self.dim + full), tail);
                    for row in 0..ROWS {
                        let x = V::<A>::load_simd_first(
                            arch,
                            group[row].vector.as_ptr().add(full),
                            tail,
                        );
                        let da = x - a;
                        let db = x - b;
                        first[row] = da.mul_add_simd(da, first[row]);
                        second[row] = db.mul_add_simd(db, second[row]);
                    }
                }
            }
            for row in 0..ROWS {
                output[row] = choose(first[row].sum_tree(), second[row].sum_tree());
            }
        }
        // A partial final group uses the reference kernel, including its architecture tail.
        for (point, output) in groups.remainder().iter().zip(outputs.into_remainder()) {
            *output = paired(point.vector, centers, self.dim);
        }
    }
}

#[cfg(test)]
mod tests;
