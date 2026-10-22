/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Complete-fit timing. Trajectory diagnostics run separately from all timed rounds.

use std::time::{Duration, Instant};

use diskann::{
    ivf::{
        clustering::{fit_with_assignment, TwoMeans},
        workingset::PointRef,
    },
    ANNError, ANNResult,
};
use diskann_quantization::algorithms::kmeans::two_means::LloydReference;
use rand::{rngs::StdRng, SeedableRng};

pub(super) use crate::inputs::graph_ivf_two_means::FitStrategy as Strategy;

pub(super) const BASELINE: Strategy = Strategy::PairedPoint;

pub(super) enum Prepared<'a> {
    Index(TwoMeans<'a, usize>),
    LloydReference(LloydReference),
}

impl<'a> Prepared<'a> {
    pub fn new(
        points: &'a [PointRef<'a, usize>],
        dim: usize,
        strategy: Strategy,
    ) -> ANNResult<Self> {
        match strategy {
            Strategy::TransposedLloydReference => {
                LloydReference::new(points.iter().map(|p| p.vector), dim)
                    .map(Self::LloydReference)
                    .map_err(ANNError::new)
            }
            Strategy::PairedPoint => {
                TwoMeans::new(points, dim, diskann::ivf::clustering::Strategy::PairedPoint)
                    .map(Self::Index)
            }
            Strategy::TransposedL2 => TwoMeans::new(
                points,
                dim,
                diskann::ivf::clustering::Strategy::TransposedL2,
            )
            .map(Self::Index),
            Strategy::TransposedDot => TwoMeans::new(
                points,
                dim,
                diskann::ivf::clustering::Strategy::TransposedDot,
            )
            .map(Self::Index),
        }
    }

    pub fn assign(&self, centers: &[f32], out: &mut [u32]) -> ANNResult<usize> {
        match self {
            Self::Index(prepared) => prepared.assign(centers, out),
            Self::LloydReference(prepared) => {
                prepared.assign(centers, out).map_err(ANNError::new)?;
                Ok(0)
            }
        }
    }

    fn packed_bytes(&self) -> usize {
        match self {
            Self::Index(p) => p.packed_bytes(),
            Self::LloydReference(p) => p.packed_bytes(),
        }
    }

    fn scratch_bytes(&self, rows: usize) -> usize {
        match self {
            Self::Index(p) => p.scratch_bytes(),
            Self::LloydReference(p) => p.scratch_bytes() + rows * std::mem::size_of::<u32>(),
        }
    }
}

pub(super) struct Run {
    pub total: Duration,
    pub preparation: Duration,
    pub centers: Vec<f32>,
    pub packed_bytes: usize,
    pub scratch_bytes: usize,
}

pub(super) fn fit(
    points: &[PointRef<'_, usize>],
    dim: usize,
    iterations: usize,
    seed: u64,
    strategy: Strategy,
) -> anyhow::Result<Run> {
    let start = Instant::now();
    let prepared = Prepared::new(points, dim, strategy)?;
    let preparation = start.elapsed();
    let mut centers = vec![0.0; 2 * dim];
    let mut sums = vec![0.0; 2 * dim];
    let mut rng = StdRng::seed_from_u64(seed);
    match &prepared {
        Prepared::Index(p) => p.fit_into(&mut centers, iterations, &mut rng, &mut sums)?,
        Prepared::LloydReference(_) => fit_with_assignment(
            points,
            &mut centers,
            iterations,
            &mut rng,
            &mut sums,
            |centers, out| prepared.assign(centers, out).map(|_| ()),
        )?,
    }
    let packed_bytes = prepared.packed_bytes();
    let scratch_bytes =
        prepared.scratch_bytes(points.len()) + std::mem::size_of_val(sums.as_slice());
    drop(prepared);
    drop(sums);
    let total = start.elapsed();
    Ok(Run {
        total,
        preparation,
        centers,
        packed_bytes,
        scratch_bytes,
    })
}

#[cfg(test)]
mod tests;
