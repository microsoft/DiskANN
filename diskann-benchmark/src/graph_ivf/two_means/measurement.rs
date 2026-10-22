/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Complete-fit timing and untimed quality diagnostics stay in the benchmark crate.

use std::time::{Duration, Instant};

use diskann::ivf::{clustering::TwoMeans, workingset::PointRef};
use diskann_vector::distance::SquaredL2;
use rand::{rngs::StdRng, SeedableRng};

use crate::inputs::graph_ivf_two_means::TwoMeansStrategy;

pub(super) struct Run {
    pub total: Duration,
    /// Shape validation and coordinate packing; included in total.
    pub preparation: Duration,
    pub centers: Vec<f32>,
    pub assignments: Vec<u8>,
    pub margins: Vec<f64>,
    pub packed_bytes: usize,
}

pub(super) fn fit(
    points: &[PointRef<'_, usize>],
    dim: usize,
    iterations: usize,
    seed: u64,
    strategy: TwoMeansStrategy,
) -> anyhow::Result<Run> {
    let start = Instant::now();
    let preparation_start = Instant::now();
    let prepared = TwoMeans::new(points, dim, strategy.into())?;
    let preparation = preparation_start.elapsed();
    let mut centers = vec![0.0; 2 * dim];
    let mut sums = vec![0.0; 2 * dim];
    let mut rng = StdRng::seed_from_u64(seed);
    prepared.fit_into(&mut centers, iterations, &mut rng, &mut sums)?;
    let packed_bytes = prepared.packed_bytes();
    // Free parent packing inside the complete-fit timer, just as in index splitting.
    drop(prepared);
    let total = start.elapsed();
    let mut assignments = Vec::with_capacity(points.len());
    let mut margins = Vec::with_capacity(points.len());
    for point in points {
        let [a, b] = SquaredL2::evaluate_pair(point.vector, &centers[..dim], &centers[dim..]);
        assignments.push(u8::from(a != b && b.total_cmp(&a).is_lt()));
        margins.push((f64::from(b) - f64::from(a)).abs());
    }
    Ok(Run {
        total,
        preparation,
        centers,
        assignments,
        margins,
        packed_bytes,
    })
}

#[cfg(test)]
mod tests;
