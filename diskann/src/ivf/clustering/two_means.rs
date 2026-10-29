/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-means fitting over borrowed point rows.

use diskann_vector::{
    MathematicalValue, PureDistanceFunction,
    distance::{InnerProduct, SquaredL2},
};
use rand::{Rng, rngs::StdRng};
#[cfg(feature = "ivf-parallel")]
use rayon::prelude::*;

#[cfg(feature = "ivf-parallel")]
use super::use_parallel;
use crate::{
    ANNError, ANNResult,
    ivf::{
        metrics::ComputeStats,
        workingset::{ListView, PointRef},
    },
};

/// Two-means fitting policy. Projection can change partitions through rounding.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum TwoMeansMode {
    #[default]
    Baseline,
    Projection,
}

pub(super) fn fit<I: Sync, L: Sync>(
    lists: &[ListView<'_, I, L>],
    centers: &mut [f32],
    dim: usize,
    iterations: usize,
    rng: &mut StdRng,
    mode: TwoMeansMode,
) -> ANNResult<ComputeStats> {
    // Preserve RNG consumption order before parallel fitting, including errors.
    for (list, centers) in lists.iter().zip(centers.chunks_exact_mut(2 * dim)) {
        initialize_two_means(&list.points, centers, dim, rng)?;
    }

    Ok(fit_initialized(lists, centers, dim, iterations, mode))
}

fn initialize_two_means<I>(
    points: &[PointRef<'_, I>],
    centers: &mut [f32],
    dim: usize,
    rng: &mut StdRng,
) -> ANNResult<()> {
    if points.len() < 2 {
        return Err(ANNError::message("an IVF split needs at least two points"));
    }
    let first = rng.random_range(0..points.len());
    let second = (first + rng.random_range(1..points.len())) % points.len();
    centers[..dim].copy_from_slice(points[first].vector);
    centers[dim..].copy_from_slice(points[second].vector);
    Ok(())
}

struct FitScratch {
    sums: Vec<f64>,
    projection: Option<Projection>,
}

impl FitScratch {
    fn new(dim: usize, mode: TwoMeansMode) -> Self {
        Self {
            sums: vec![0.0; 2 * dim],
            projection: (mode == TwoMeansMode::Projection).then(|| Projection::new(dim)),
        }
    }
}

fn fit_initialized<I: Sync, L: Sync>(
    lists: &[ListView<'_, I, L>],
    centers: &mut [f32],
    dim: usize,
    iterations: usize,
    mode: TwoMeansMode,
) -> ComputeStats {
    let mut compute = ComputeStats::default();
    #[cfg(feature = "ivf-parallel")]
    {
        let count: usize = lists.iter().map(|list| list.points.len()).sum();
        if lists.len() > 1
            && use_parallel(
                count
                    .saturating_mul(dim)
                    .saturating_mul(2)
                    .saturating_mul(iterations.max(1)),
            )
        {
            let mut statistics = vec![ComputeStats::default(); lists.len()];
            lists
                .par_iter()
                .zip(centers.par_chunks_exact_mut(2 * dim))
                .zip(statistics.par_iter_mut())
                .for_each_init(
                    || FitScratch::new(dim, mode),
                    |scratch, ((list, centers), stats)| {
                        *stats = fit_two_means(&list.points, centers, dim, iterations, scratch);
                    },
                );
            for stats in statistics {
                compute.merge(stats);
            }
            return compute;
        }
    }
    let mut scratch = FitScratch::new(dim, mode);
    for (list, centers) in lists.iter().zip(centers.chunks_exact_mut(2 * dim)) {
        compute.merge(fit_two_means(
            &list.points,
            centers,
            dim,
            iterations,
            &mut scratch,
        ));
    }
    compute
}

fn fit_two_means<I>(
    points: &[PointRef<'_, I>],
    centers: &mut [f32],
    dim: usize,
    iterations: usize,
    scratch: &mut FitScratch,
) -> ComputeStats {
    let FitScratch { sums, projection } = scratch;
    let mut compute = ComputeStats::default();
    for _ in 0..iterations.max(1) {
        if let Some(projection) = projection {
            projection.update(centers);
        }
        sums.fill(0.0);
        let mut counts = [0; 2];
        let (left, right) = centers.split_at(dim);
        for point in points {
            let projected = projection.as_ref().and_then(|p| p.assign(point.vector));
            let to = if let Some(to) = projected {
                compute.projection_points += 1;
                to
            } else {
                if projection.is_some() {
                    compute.numeric_fallbacks += 1;
                }
                let [left_distance, right_distance] =
                    SquaredL2::evaluate_batch(point.vector, [left, right]);
                usize::from(right_distance.total_cmp(&left_distance).is_lt())
            };
            counts[to] += 1;
            for (sum, &value) in sums[to * dim..(to + 1) * dim].iter_mut().zip(point.vector) {
                // A finite mean must not overflow while accumulating its inputs.
                *sum += f64::from(value);
            }
        }
        let mut changed = false;
        for ((center, sum), count) in centers
            .chunks_exact_mut(dim)
            .zip(sums.chunks_exact(dim))
            .zip(counts)
        {
            if count != 0 {
                for (value, &sum) in center.iter_mut().zip(sum) {
                    let next = (sum / count as f64) as f32;
                    changed |= value.to_bits() != next.to_bits();
                    *value = next;
                }
            }
        }
        // Identical center bits repeat the same assignments and ordered f64 sums.
        if !changed {
            break;
        }
    }
    compute
}

/// Classify with x.(c1-c0) > (||c1||²-||c0||²)/2; equality selects c0.
/// Finite rounding differences from direct L2 are not detected or corrected.
struct Projection {
    normal: Vec<f32>,
    threshold: f64,
    valid: bool,
}

impl Projection {
    pub fn new(dim: usize) -> Self {
        Self {
            normal: vec![0.0; dim],
            threshold: 0.0,
            valid: false,
        }
    }

    pub fn update(&mut self, centers: &[f32]) {
        let dim = self.normal.len();
        debug_assert_eq!(centers.len(), 2 * dim);
        let (left, right) = centers.split_at(dim);
        self.threshold = 0.0;
        self.valid = true;
        for ((w, &a), &b) in self.normal.iter_mut().zip(left).zip(right) {
            *w = b - a;
            // Factor the difference of squares in f64 without rounded f32 norms.
            let (a, b) = (f64::from(a), f64::from(b));
            self.threshold += (b - a) * ((a + b) * 0.5);
            self.valid &= w.is_finite();
        }
        self.valid &= self.threshold.is_finite();
    }

    #[inline]
    pub fn assign(&self, point: &[f32]) -> Option<usize> {
        if !self.valid {
            return None;
        }
        // A plain f32 result from InnerProduct is negated; request the true dot.
        let dot: MathematicalValue<f32> = InnerProduct::evaluate(point, self.normal.as_slice());
        let dot = dot.into_inner();
        dot.is_finite()
            .then_some(usize::from(f64::from(dot) > self.threshold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_sign_ties_and_overflow() {
        let mut projection = Projection::new(1);
        projection.update(&[2.0, 8.0]);
        assert_eq!(projection.assign(&[4.0]), Some(0));
        assert_eq!(projection.assign(&[5.0]), Some(0));
        assert_eq!(projection.assign(&[6.0]), Some(1));
        projection.update(&[3.0, 3.0]);
        assert_eq!(projection.assign(&[100.0]), Some(0));

        projection.update(&[-f32::MAX, f32::MAX]);
        assert_eq!(projection.assign(&[0.0]), None);
        projection.update(&[0.0, f32::MAX]);
        assert_eq!(projection.assign(&[f32::MAX]), None);
        projection.update(&[2.0, 8.0]);
        assert_eq!(projection.assign(&[6.0]), Some(1));
    }

    #[test]
    fn baseline_matches_reference_lloyd_prefixes_and_rng() {
        use rand::SeedableRng;
        for dim in [1, 7, 8, 31, 32, 33, 127, 769] {
            let mut data_rng = StdRng::seed_from_u64(17);
            let rows: Vec<Vec<Vec<f32>>> = (0..4)
                .map(|parent| {
                    (0..37 + parent)
                        .map(|row| {
                            (0..dim)
                                .map(|_| (row % 5) as f32 * 10.0 + data_rng.random_range(-1.0..1.0))
                                .collect()
                        })
                        .collect()
                })
                .collect();
            let lists: Vec<_> = rows
                .iter()
                .enumerate()
                .map(|(id, rows)| ListView {
                    id,
                    points: rows
                        .iter()
                        .enumerate()
                        .map(|(id, vector)| PointRef { id, vector })
                        .collect(),
                })
                .collect();
            for iterations in (0..=8).chain([30]) {
                let mut expected = vec![0.0; 2 * dim * lists.len()];
                let mut expected_rng = StdRng::seed_from_u64(7);
                let mut sums = vec![0.0; 2 * dim];
                for (list, centers) in lists.iter().zip(expected.chunks_exact_mut(2 * dim)) {
                    reference_fit(
                        &list.points,
                        centers,
                        dim,
                        iterations,
                        &mut expected_rng,
                        &mut sums,
                    )
                    .unwrap();
                }
                let mut actual = vec![0.0; expected.len()];
                let mut actual_rng = StdRng::seed_from_u64(7);
                let stats = fit(
                    &lists,
                    &mut actual,
                    dim,
                    iterations,
                    &mut actual_rng,
                    TwoMeansMode::Baseline,
                )
                .unwrap();
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "D={dim}, steps={iterations}"
                );
                assert_eq!(actual_rng.random::<u64>(), expected_rng.random::<u64>());
                assert_eq!((stats.projection_points, stats.numeric_fallbacks), (0, 0));
            }
        }
    }
    fn reference_fit<I>(
        points: &[PointRef<'_, I>],
        centers: &mut [f32],
        dim: usize,
        iterations: usize,
        rng: &mut StdRng,
        sums: &mut [f64],
    ) -> ANNResult<()> {
        if points.len() < 2 {
            return Err(ANNError::message("an IVF split needs at least two points"));
        }
        let first = rng.random_range(0..points.len());
        let second = (first + rng.random_range(1..points.len())) % points.len();
        centers[..dim].copy_from_slice(points[first].vector);
        centers[dim..].copy_from_slice(points[second].vector);

        for _ in 0..iterations.max(1) {
            sums.fill(0.0);
            let mut counts = [0; 2];
            for point in points {
                let to = reference_nearest(point.vector, centers, dim);
                counts[to] += 1;
                for (sum, &value) in sums[to * dim..(to + 1) * dim].iter_mut().zip(point.vector) {
                    // A finite mean must not overflow while accumulating its inputs.
                    *sum += f64::from(value);
                }
            }
            for ((center, sum), count) in centers
                .chunks_exact_mut(dim)
                .zip(sums.chunks_exact(dim))
                .zip(counts)
            {
                if count != 0 {
                    for (value, &sum) in center.iter_mut().zip(sum) {
                        *value = (sum / count as f64) as f32;
                    }
                }
            }
        }
        Ok(())
    }

    /// Prefer the first centroid on ties. `centers` contains at least one full row.
    fn reference_nearest(point: &[f32], centers: &[f32], dim: usize) -> usize {
        let mut best = 0;
        let mut best_distance: f32 = SquaredL2::evaluate(point, &centers[..dim]);
        for (index, center) in centers.chunks_exact(dim).enumerate().skip(1) {
            let distance: f32 = SquaredL2::evaluate(point, center);
            if distance.total_cmp(&best_distance).is_lt() {
                best = index;
                best_distance = distance;
            }
        }
        best
    }
}
