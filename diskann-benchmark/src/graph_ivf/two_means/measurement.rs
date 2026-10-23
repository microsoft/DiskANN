/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Complete-fit timing. Trajectory diagnostics run separately from all timed rounds.

use std::time::{Duration, Instant};

use diskann::ivf::{clustering::TwoMeans, workingset::PointRef};
use rand::{rngs::StdRng, SeedableRng};

use crate::inputs::graph_ivf_two_means::TwoMeansStrategy;

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
    strategy: TwoMeansStrategy,
) -> anyhow::Result<Run> {
    let start = Instant::now();
    let prepared = TwoMeans::new(points, dim, strategy.into())?;
    let preparation = start.elapsed();
    let mut centers = vec![0.0; 2 * dim];
    let mut sums = vec![0.0; 2 * dim];
    let mut rng = StdRng::seed_from_u64(seed);
    prepared.fit_into(&mut centers, iterations, &mut rng, &mut sums)?;
    let packed_bytes = prepared.packed_bytes();
    let scratch_bytes = prepared.scratch_bytes() + std::mem::size_of_val(sums.as_slice());
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
