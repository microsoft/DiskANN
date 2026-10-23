/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ivf::{
    clustering::{fit_with_assignment, TwoMeans},
    workingset::PointRef,
};
use diskann_vector::distance::SquaredL2;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use serde::Serialize;

use super::BASELINE;
use crate::inputs::graph_ivf_two_means::TwoMeansStrategy;

pub(super) struct Trace {
    steps: Vec<Step>,
    pub centers: Vec<f32>,
    sums: Vec<f64>,
    next_random: u64,
}

struct Step {
    centers: Vec<f32>,
    assignments: Vec<u32>,
    fallbacks: usize,
}

pub(super) fn trace(
    points: &[PointRef<'_, usize>],
    dim: usize,
    iterations: usize,
    seed: u64,
    strategy: TwoMeansStrategy,
) -> anyhow::Result<Trace> {
    let prepared = TwoMeans::new(points, dim, strategy.into())?;
    let mut centers = vec![0.0; 2 * dim];
    let mut sums = vec![0.0; 2 * dim];
    let mut steps = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);
    fit_with_assignment(
        points,
        &mut centers,
        iterations,
        &mut rng,
        &mut sums,
        |centers, out| {
            let fallbacks = prepared.assign(centers, out)?;
            steps.push(Step {
                centers: centers.to_vec(),
                assignments: out.to_vec(),
                fallbacks,
            });
            Ok(())
        },
    )?;
    Ok(Trace {
        steps,
        centers,
        sums,
        next_random: rng.next_u64(),
    })
}

#[derive(Serialize)]
pub(super) struct Comparison {
    pub steps: Vec<StepComparison>,
    pub first_divergent_iteration: Option<usize>,
    pub final_assignment_disagreements: usize,
    pub final_center_bit_disagreements: usize,
    pub max_center_absolute_error: f64,
    pub min_reference_margin_f64: f64,
    pub fallback_points: usize,
    pub final_sum_bit_disagreements: usize,
    pub rng_equal: bool,
}

impl Comparison {
    pub fn is_exact(&self) -> bool {
        self.first_divergent_iteration.is_none()
            && self.final_assignment_disagreements == 0
            && self.final_center_bit_disagreements == 0
            && self.final_sum_bit_disagreements == 0
            && self.rng_equal
    }
}

#[derive(Serialize)]
pub(super) struct StepComparison {
    iteration: usize,
    // Compare both kernels at the baseline trajectory's centers.
    same_center_disagreements: usize,
    same_center_fallback_points: usize,
    first_same_center_difference: Option<Difference>,
    // Compare each strategy following its own previous assignments.
    trajectory_disagreements: usize,
    center_bit_disagreements: usize,
    fallback_points: usize,
}

#[derive(Serialize)]
struct Difference {
    row: usize,
    point_id: usize,
    baseline: u32,
    candidate: u32,
    direct_l2_f64: [f64; 2],
    margin_f64: f64,
    paired_l2_f32: [f32; 2],
}

fn distances(point: &[f32], centers: &[f32], dim: usize) -> [f64; 2] {
    std::array::from_fn(|i| {
        point
            .iter()
            .zip(&centers[i * dim..(i + 1) * dim])
            .map(|(&x, &c)| (f64::from(x) - f64::from(c)).powi(2))
            .sum()
    })
}

pub(super) fn compare(
    points: &[PointRef<'_, usize>],
    dim: usize,
    strategy: TwoMeansStrategy,
    baseline: &Trace,
    candidate: &Trace,
) -> anyhow::Result<Comparison> {
    anyhow::ensure!(
        baseline.steps.len() == candidate.steps.len(),
        "Lloyd step count changed"
    );
    anyhow::ensure!(
        baseline.centers.len() == candidate.centers.len()
            && baseline.sums.len() == candidate.sums.len(),
        "diagnostic shape changed"
    );
    let prepared = TwoMeans::new(points, dim, strategy.into())?;
    let mut same_centers = vec![0; points.len()];
    let mut steps = Vec::new();
    let mut first = None;
    for (i, (a, b)) in baseline.steps.iter().zip(&candidate.steps).enumerate() {
        anyhow::ensure!(
            a.assignments.len() == b.assignments.len() && a.centers.len() == b.centers.len(),
            "step shape changed"
        );
        let fallbacks = prepared.assign(&a.centers, &mut same_centers)?;
        let same_differences = a
            .assignments
            .iter()
            .zip(&same_centers)
            .filter(|(a, b)| a != b)
            .count();
        let trajectory = a
            .assignments
            .iter()
            .zip(&b.assignments)
            .filter(|(a, b)| a != b)
            .count();
        let center_bits = a
            .centers
            .iter()
            .zip(&b.centers)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        if first.is_none() && (same_differences != 0 || trajectory != 0 || center_bits != 0) {
            first = Some(i);
        }
        let difference = a
            .assignments
            .iter()
            .zip(&same_centers)
            .position(|(a, b)| a != b)
            .map(|row| {
                let point = &points[row];
                let d = distances(point.vector, &a.centers, dim);
                Difference {
                    row,
                    point_id: point.id,
                    baseline: a.assignments[row],
                    candidate: same_centers[row],
                    direct_l2_f64: d,
                    margin_f64: (d[1] - d[0]).abs(),
                    paired_l2_f32: SquaredL2::evaluate_pair(
                        point.vector,
                        &a.centers[..dim],
                        &a.centers[dim..],
                    ),
                }
            });
        steps.push(StepComparison {
            iteration: i,
            same_center_disagreements: same_differences,
            same_center_fallback_points: fallbacks,
            first_same_center_difference: difference,
            trajectory_disagreements: trajectory,
            center_bit_disagreements: center_bits,
            fallback_points: b.fallbacks,
        });
    }
    let paired = TwoMeans::new(points, dim, BASELINE.into())?;
    let mut a = vec![0; points.len()];
    let mut b = a.clone();
    paired.assign(&baseline.centers, &mut a)?;
    paired.assign(&candidate.centers, &mut b)?;
    let min_margin = points
        .iter()
        .map(|point| {
            let d = distances(point.vector, &baseline.centers, dim);
            (d[1] - d[0]).abs()
        })
        .fold(f64::INFINITY, f64::min);
    Ok(Comparison {
        steps,
        first_divergent_iteration: first,
        final_assignment_disagreements: a.iter().zip(&b).filter(|(a, b)| a != b).count(),
        final_center_bit_disagreements: baseline
            .centers
            .iter()
            .zip(&candidate.centers)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count(),
        max_center_absolute_error: baseline
            .centers
            .iter()
            .zip(&candidate.centers)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .fold(0.0, f64::max),
        min_reference_margin_f64: min_margin,
        fallback_points: candidate.steps.iter().map(|step| step.fallbacks).sum(),
        final_sum_bit_disagreements: baseline
            .sums
            .iter()
            .zip(&candidate.sums)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count(),
        rng_equal: baseline.next_random == candidate.next_random,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_comparison_detects_intermediate_and_final_divergences() {
        let rows = [10001.0, 10000.0, 10000.25];
        let points: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(id, v)| PointRef {
                id: id + 10,
                vector: std::slice::from_ref(v),
            })
            .collect();
        let strategy = TwoMeansStrategy::TiledExact;
        let a = trace(&points, 1, 5, 7, BASELINE).unwrap();
        let mut b = trace(&points, 1, 5, 7, strategy).unwrap();
        assert!(compare(&points, 1, strategy, &a, &b).unwrap().is_exact());
        // A trajectory difference that later converges must still fail the gate.
        b.steps[2].assignments[0] ^= 1;
        let c = compare(&points, 1, strategy, &a, &b).unwrap();
        assert!(!c.is_exact());
        assert_eq!(c.first_divergent_iteration, Some(2));
        assert_eq!(c.final_center_bit_disagreements, 0);
        b.steps[2].assignments[0] ^= 1;
        b.next_random ^= 1;
        assert!(!compare(&points, 1, strategy, &a, &b).unwrap().is_exact());
        b.next_random ^= 1;
        b.sums[0] = b.sums[0].next_up();
        assert!(!compare(&points, 1, strategy, &a, &b).unwrap().is_exact());
    }
}
