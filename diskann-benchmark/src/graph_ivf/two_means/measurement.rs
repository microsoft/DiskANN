/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Complete-fit timing. Trajectory diagnostics run separately from all timed rounds.

use std::time::{Duration, Instant};

use diskann::{
    ivf::{clustering::TwoMeans, workingset::PointRef},
    ANNResult,
};
use rand::{rngs::StdRng, SeedableRng};

pub(super) use crate::inputs::graph_ivf_two_means::FitStrategy as Strategy;

pub(super) const BASELINE: Strategy = Strategy::PairedPoint;

pub(super) struct Prepared<'a>(TwoMeans<'a, usize>);

impl<'a> Prepared<'a> {
    pub fn new(
        points: &'a [PointRef<'a, usize>],
        dim: usize,
        strategy: Strategy,
    ) -> ANNResult<Self> {
        let strategy = match strategy {
            Strategy::PairedPoint => diskann::ivf::clustering::Strategy::PairedPoint,
            Strategy::TiledExact => diskann::ivf::clustering::Strategy::TiledExact,
        };
        TwoMeans::new(points, dim, strategy).map(Self)
    }

    pub fn assign(&self, centers: &[f32], out: &mut [u32]) -> ANNResult<usize> {
        self.0.assign(centers, out)
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
    prepared
        .0
        .fit_into(&mut centers, iterations, &mut rng, &mut sums)?;
    let packed_bytes = prepared.0.packed_bytes();
    let scratch_bytes = prepared.0.scratch_bytes() + std::mem::size_of_val(sums.as_slice());
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
