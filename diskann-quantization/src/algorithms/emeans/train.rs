/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use rand::{Rng, distr::Open01, seq::SliceRandom};
use thiserror::Error;

use crate::algorithms::kmeans::{common::square_norm, lloyds::distances_in_place};
use crate::{cancel::Cancelation, multi_vector::BlockTransposed};

/// Scaling law for automatic batch-size selection from the dataset size, cluster count
/// and epoch budget. Ignored when [EmeansConfig::batch_size] is supplied explicitly.
/// The resolved batch size also determines the default [EmeansConfig::step_size].
#[derive(Debug, Clone, Copy, Default)]
pub enum BatchSizePolicy {
    /// Quality for dimensions <= 16, throughput otherwise, as in Faiss.
    #[default]
    Auto,
    /// Favor larger batches to amortize centroid-update overhead.
    Throughput,
    /// Use one eighth of the throughput law's coefficient for more frequent updates.
    /// Rounding and caps mean the final batch is not necessarily eight times smaller.
    Quality,
}

/// Training budget and optional overrides of the automatic recipe.
#[derive(Debug, Clone)]
pub struct EmeansConfig {
    /// Positive number of complete passes; default: 5.
    pub epochs: usize,
    /// Positive batch size, capped at the dataset size. None uses a scaling law.
    pub batch_size: Option<usize>,
    /// Initial EMA step in (0, 1]. None uses the step scaling law.
    pub step_size: Option<f64>,
    /// Policy for the batch scaling law.
    pub batch_size_policy: BatchSizePolicy,
    /// Enable mass-proportional splitting every five updates (only when d > 2).
    pub revive: bool,
}

impl Default for EmeansConfig {
    fn default() -> Self {
        Self {
            epochs: 5,
            batch_size: None,
            step_size: None,
            batch_size_policy: BatchSizePolicy::Auto,
            revive: true,
        }
    }
}

/// Assignments and resolved recipe for a completed run.
#[derive(Debug)]
pub struct EmeansResult {
    /// Labels in original input order, assigned while centers moved in the last epoch.
    /// They are not necessarily nearest-center labels for the final centers.
    pub assignments: Vec<u32>,
    /// Resolved batch size, including a possibly smaller final batch per epoch.
    pub batch_size: usize,
    /// Step before cosine annealing.
    pub initial_step_size: f64,
    /// Number of completed mini-batch updates.
    pub updates: usize,
    /// Total number of cluster splits.
    pub revived_clusters: usize,
}

/// Failures specific to e-means training.
#[derive(Debug, Error)]
pub enum EmeansError {
    #[error("invalid e-means configuration: {0}")]
    InvalidConfig(&'static str),
    #[error(
        "e-means requires nonempty matrices with equal, nonzero dimensions and 1 <= k <= min(n, u32::MAX)"
    )]
    InvalidShape,
    #[error("e-means workspace or update count overflows its supported size")]
    SizeOverflow,
    #[error("e-means requires finite coordinates and squared row norms <= f32::MAX / 8")]
    NumericalRange,
    #[error("e-means training was canceled")]
    Canceled,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedParameters {
    batch_size: usize,
    initial_step_size: f64,
    total_updates: usize,
}

impl EmeansConfig {
    fn resolve_parameters(
        &self,
        n: usize,
        k: usize,
        d: usize,
    ) -> Result<ResolvedParameters, EmeansError> {
        if self.epochs == 0 {
            return Err(EmeansError::InvalidConfig("epochs must be positive"));
        }
        if self.batch_size == Some(0) {
            return Err(EmeansError::InvalidConfig("batch_size must be positive"));
        }
        if self
            .step_size
            .is_some_and(|x| !x.is_finite() || x <= 0.0 || x > 1.0)
        {
            return Err(EmeansError::InvalidConfig("step_size must be in (0, 1]"));
        }
        let quality = match self.batch_size_policy {
            BatchSizePolicy::Auto => d <= 16,
            BatchSizePolicy::Quality => true,
            BatchSizePolicy::Throughput => false,
        };
        let batch_size = self.batch_size.map_or_else(
            || {
                let coefficient = if quality { 0.00439175 } else { 0.035134 };
                let raw = coefficient
                    * (k as f64).powf(0.134749)
                    * (n as f64).powf(0.876473)
                    * (self.epochs as f64).powf(0.813344);
                let cap = if quality {
                    n / 2
                } else {
                    round_power_of_two(n as f64 / 2.0)
                };
                round_power_of_two(raw).min(cap).clamp(1, n)
            },
            |b| b.min(n),
        );
        let initial_step_size = self.step_size.unwrap_or_else(|| {
            (0.8244
                * (k as f64).powf(-0.4617)
                * (batch_size as f64).powf(0.9003)
                * (n as f64).powf(-0.4498)
                * (self.epochs as f64).powf(-0.1057))
            .clamp(1e-4, 1.0)
        });
        let total_updates = n
            .div_ceil(batch_size)
            .checked_mul(self.epochs)
            .ok_or(EmeansError::SizeOverflow)?;
        Ok(ResolvedParameters {
            batch_size,
            initial_step_size,
            total_updates,
        })
    }
}

fn round_power_of_two(value: f64) -> usize {
    2.0_f64.powf(value.max(1.0).log2().round()) as usize
}

fn scheduled_step_size(initial_step_size: f64, update_index: usize, total_updates: usize) -> f64 {
    initial_step_size
        * (0.1
            + 0.45
                * (1.0 + (std::f64::consts::PI * update_index as f64 / total_updates as f64).cos()))
}

fn check_canceled(cancel: &impl Cancelation) -> Result<(), EmeansError> {
    if cancel.should_cancel() {
        Err(EmeansError::Canceled)
    } else {
        Ok(())
    }
}

fn checked_squared_norm(row: &[f32]) -> Result<f32, EmeansError> {
    let norm = square_norm(row);
    // This leaves headroom for the dot-product distance expansion and cluster splits.
    if !norm.is_finite() || norm > f32::MAX / 8.0 {
        Err(EmeansError::NumericalRange)
    } else {
        Ok(norm)
    }
}

fn prepare_training(
    data: rowmajor::Ref<'_, f32>,
    centers: rowmajor::Ref<'_, f32>,
    config: &EmeansConfig,
    cancel: &impl Cancelation,
) -> Result<(ResolvedParameters, Vec<f32>), EmeansError> {
    check_canceled(cancel)?;
    let (n, k, d) = (data.nrows(), centers.nrows(), data.ncols());
    if n == 0 || k == 0 || k > n || k > u32::MAX as usize || d == 0 || d != centers.ncols() {
        return Err(EmeansError::InvalidShape);
    }
    if centers.as_slice().len() > isize::MAX as usize / size_of::<f64>()
        || n > isize::MAX as usize / size_of::<usize>()
    {
        return Err(EmeansError::SizeOverflow);
    }
    let recipe = config.resolve_parameters(n, k, d)?;
    let mut norms = Vec::with_capacity(n);
    for (i, row) in data.rows().enumerate() {
        if i.is_multiple_of(4096) {
            check_canceled(cancel)?;
        }
        norms.push(checked_squared_norm(row)?);
    }
    Ok((recipe, norms))
}

/// Train from a uniform sample of distinct input rows, writing centers in place.
///
/// The centers buffer's row count specifies k; its initial contents are ignored.
/// Uses integer epoch budgets and exact f32 SIMD assignment, with f64 statistics.
///
/// # Errors
/// Returns an error for invalid shapes/configuration, unsupported numeric range or
/// cancellation. Cancellation can leave partially trained centers. Allocation failures
/// follow the allocator's usual behavior.
pub fn emeans(
    data: rowmajor::Ref<'_, f32>,
    mut centers: rowmajor::Mut<'_, f32>,
    config: &EmeansConfig,
    rng: &mut impl Rng,
    cancel: &impl Cancelation,
) -> Result<EmeansResult, EmeansError> {
    let (recipe, norms) = prepare_training(data, centers.as_view(), config, cancel)?;
    let chosen = rand::seq::index::sample(rng, data.nrows(), centers.nrows());
    for (row, index) in centers.rows_mut().zip(chosen.iter()) {
        row.copy_from_slice(data.row(index));
    }
    train_inner(data, centers, config, recipe, &norms, rng, cancel)
}

/// Train from caller-supplied centers, for warm starts or matched-initialization comparisons.
///
/// Final nonempty centers are means of the returned last-epoch online assignments.
/// Unassigned centers retain their last position. Evaluate nearest-center loss separately
/// if needed: finalization does not perform an extra assignment pass.
///
/// # Errors
/// Same as [emeans], additionally checking the supplied centers' numeric range.
pub fn emeans_from_centers(
    data: rowmajor::Ref<'_, f32>,
    centers: rowmajor::Mut<'_, f32>,
    config: &EmeansConfig,
    rng: &mut impl Rng,
    cancel: &impl Cancelation,
) -> Result<EmeansResult, EmeansError> {
    let (recipe, norms) = prepare_training(data, centers.as_view(), config, cancel)?;
    for center in centers.rows() {
        checked_squared_norm(center)?;
    }
    train_inner(data, centers, config, recipe, &norms, rng, cancel)
}

struct AssignmentBatch {
    transpose: BlockTransposed<f32, 16>,
    norms: Vec<f32>,
    labels: Vec<u32>,
}

impl AssignmentBatch {
    fn new(n: usize, d: usize) -> Result<Self, EmeansError> {
        Ok(Self {
            transpose: BlockTransposed::try_new(n, d).map_err(|_| EmeansError::SizeOverflow)?,
            norms: vec![0.0; n],
            labels: vec![0; n],
        })
    }

    fn assign(
        &mut self,
        data: rowmajor::Ref<'_, f32>,
        indices: &[usize],
        norms: &[f32],
        centers: rowmajor::Ref<'_, f32>,
        center_norms: &[f32],
    ) {
        // Gather directly into reusable SIMD blocks; no shuffled n*d copy or B*k matrix.
        let d = data.ncols();
        for (block, rows) in self
            .transpose
            .as_mut_slice()
            .chunks_mut(16 * d)
            .zip(indices.chunks(16))
        {
            for (lane, &index) in rows.iter().enumerate() {
                for (coordinate, &value) in data.row(index).iter().enumerate() {
                    block[16 * coordinate + lane] = value;
                }
            }
        }
        for (dst, &index) in self.norms.iter_mut().zip(indices) {
            *dst = norms[index];
        }
        // The kernel also computes an f32 residual; it is deliberately not reported because
        // it refers to moving centers and its reduction can overflow on very large datasets.
        distances_in_place(
            self.transpose.as_view(),
            &self.norms,
            centers,
            center_norms,
            &mut self.labels,
        );
    }
}

fn train_inner(
    data: rowmajor::Ref<'_, f32>,
    mut centers: rowmajor::Mut<'_, f32>,
    config: &EmeansConfig,
    recipe: ResolvedParameters,
    norms: &[f32],
    rng: &mut impl Rng,
    cancel: &impl Cancelation,
) -> Result<EmeansResult, EmeansError> {
    let (n, k, d) = (data.nrows(), centers.nrows(), data.ncols());
    let mut full_batch = AssignmentBatch::new(recipe.batch_size, d)?;
    let mut remainder_batch = AssignmentBatch::new(n % recipe.batch_size, d)?;
    let mut order: Vec<_> = (0..n).collect();
    let mut assignments = vec![0; n];
    let mut center_norms = vec![0.0; k];
    let mut masses = vec![1.0 / 65536.0; k];
    let mut counts = vec![0_usize; k];
    let mut sums = vec![0.0; k * d];
    let mut last_epoch_counts = vec![0_usize; k];
    let mut last_epoch_sums = vec![0.0; k * d];
    let mut step = 0;
    let mut revived_clusters = 0;
    for epoch in 0..config.epochs {
        check_canceled(cancel)?;
        order.shuffle(rng);
        let is_last_epoch = epoch == config.epochs - 1;
        for indices in order.chunks(recipe.batch_size) {
            check_canceled(cancel)?;
            for (norm, row) in center_norms.iter_mut().zip(centers.rows()) {
                *norm = checked_squared_norm(row)?;
            }
            let batch = if indices.len() == recipe.batch_size {
                &mut full_batch
            } else {
                &mut remainder_batch
            };
            batch.assign(data, indices, norms, centers.as_view(), &center_norms);
            counts.fill(0);
            sums.fill(0.0);
            for (&index, &label) in indices.iter().zip(&batch.labels) {
                let cluster = label as usize;
                counts[cluster] += 1;
                for (sum, &value) in sums[cluster * d..(cluster + 1) * d]
                    .iter_mut()
                    .zip(data.row(index))
                {
                    *sum += f64::from(value);
                }
                if is_last_epoch {
                    assignments[index] = label;
                }
            }
            if is_last_epoch {
                for (total, count) in last_epoch_counts.iter_mut().zip(&counts) {
                    *total += count;
                }
                for (total, sum) in last_epoch_sums.iter_mut().zip(&sums) {
                    *total += sum;
                }
            }
            let alpha = scheduled_step_size(recipe.initial_step_size, step, recipe.total_updates);
            for (j, row) in centers.rows_mut().enumerate() {
                let old_mass = (1.0 - alpha) * masses[j];
                let added_mass = alpha * counts[j] as f64;
                masses[j] = old_mass + added_mass;
                if counts[j] != 0 {
                    let weight = added_mass / masses[j];
                    for (coordinate, value) in row.iter_mut().enumerate() {
                        let mean = sums[j * d + coordinate] / counts[j] as f64;
                        *value = ((1.0 - weight) * f64::from(*value) + weight * mean) as f32;
                    }
                }
            }
            step += 1;
            if config.revive && d > 2 && step.is_multiple_of(5) {
                revived_clusters +=
                    revive_underused_clusters(centers.as_view_mut(), &mut masses, rng);
            }
        }
    }
    check_canceled(cancel)?;
    for (j, row) in centers.rows_mut().enumerate() {
        if last_epoch_counts[j] != 0 {
            for (coordinate, value) in row.iter_mut().enumerate() {
                *value = (last_epoch_sums[j * d + coordinate] / last_epoch_counts[j] as f64) as f32;
            }
        }
    }
    Ok(EmeansResult {
        assignments,
        batch_size: recipe.batch_size,
        initial_step_size: recipe.initial_step_size,
        updates: step,
        revived_clusters,
    })
}

fn revive_underused_clusters(
    mut centers: rowmajor::Mut<'_, f32>,
    masses: &mut [f64],
    rng: &mut impl Rng,
) -> usize {
    let threshold = 0.1 * masses.iter().sum::<f64>() / masses.len() as f64;
    let underused: Vec<_> = masses
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| (m < threshold).then_some(i))
        .collect();
    if underused.is_empty() {
        return 0;
    }
    // Gumbel top-k samples donors in proportion to mass, without replacement.
    let mut donors: Vec<_> = masses
        .iter()
        .enumerate()
        .filter(|&(_, &m)| m >= threshold && m > 0.0)
        .map(|(i, &m)| {
            let uniform: f64 = rng.sample(Open01);
            (m.ln() - (-uniform.ln()).ln(), i)
        })
        .collect();
    donors.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let splits = underused.len().min(donors.len());
    for (target, (_, source)) in underused.into_iter().zip(donors) {
        let half_mass = masses[source] * 0.5;
        masses[source] = half_mass;
        masses[target] = half_mass;
        for coordinate in 0..centers.ncols() {
            let value = centers.row(source)[coordinate];
            let delta = if coordinate.is_multiple_of(2) {
                1e-4
            } else {
                -1e-4
            };
            centers.row_mut(source)[coordinate] = value * (1.0 - delta);
            centers.row_mut(target)[coordinate] = value * (1.0 + delta);
        }
        if centers.row(source) == centers.row(target) {
            let value = centers.row(target)[0];
            let next = value.next_up();
            centers.row_mut(target)[0] = if next.is_finite() {
                next
            } else {
                value.next_down()
            };
        }
    }
    splits
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use rand::{SeedableRng, rngs::StdRng};

    use crate::cancel::DontCancel;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn assert_near(got: f64, expected: f64, tolerance: f64) {
        assert!(
            (got - expected).abs() <= tolerance,
            "expected {expected}, got {got}, tolerance {tolerance}"
        );
    }

    #[test]
    fn published_recipe_and_cosine_schedule() -> TestResult {
        let throughput = EmeansConfig {
            epochs: 25,
            batch_size_policy: BatchSizePolicy::Throughput,
            ..Default::default()
        };
        let quality = EmeansConfig {
            batch_size_policy: BatchSizePolicy::Quality,
            ..throughput.clone()
        };
        let recipe = throughput.resolve_parameters(1_000_000, 16_384, 128)?;
        assert_eq!(recipe.batch_size, 262_144);
        assert_eq!(recipe.total_updates, 100);
        assert_eq!(recipe.initial_step_size, 1.0);
        let recipe = quality.resolve_parameters(1_000_000, 16_384, 128)?;
        assert_eq!(recipe.batch_size, 32_768);
        assert_near(recipe.initial_step_size, 0.1545388281304627, 1e-14);
        let recipe = quality.resolve_parameters(100_000_000, 262_144, 128)?;
        assert_eq!(recipe.batch_size, 4_194_304);
        assert_near(recipe.initial_step_size, 0.42718555053293517, 1e-14);
        assert_near(scheduled_step_size(0.4, 0, 10), 0.4, 1e-15);
        assert_near(scheduled_step_size(0.4, 9, 10), 0.04880982706687237, 1e-15);
        for step in 1..10 {
            assert!(scheduled_step_size(0.4, step, 10) < scheduled_step_size(0.4, step - 1, 10));
        }
        Ok(())
    }

    #[test]
    fn automatic_policy_caps_rounding_and_explicit_overrides() -> TestResult {
        let config = EmeansConfig::default();
        let quality = EmeansConfig {
            batch_size_policy: BatchSizePolicy::Quality,
            ..config.clone()
        };
        let throughput = EmeansConfig {
            batch_size_policy: BatchSizePolicy::Throughput,
            ..config.clone()
        };
        assert_eq!(
            config.resolve_parameters(100_000, 256, 16)?.batch_size,
            quality.resolve_parameters(100_000, 256, 16)?.batch_size
        );
        assert_eq!(
            config.resolve_parameters(100_000, 256, 17)?.batch_size,
            throughput.resolve_parameters(100_000, 256, 17)?.batch_size
        );
        assert_eq!(round_power_of_two(5.0), 4);
        assert_eq!(round_power_of_two(6.0), 8);
        assert_eq!(round_power_of_two(0.5), 1);
        let capped = EmeansConfig {
            epochs: 100_000,
            ..quality
        };
        assert_eq!(capped.resolve_parameters(11, 3, 5)?.batch_size, 5);
        let capped = EmeansConfig {
            epochs: 100_000,
            ..throughput
        };
        assert_eq!(capped.resolve_parameters(11, 3, 5)?.batch_size, 4);
        let explicit = EmeansConfig {
            epochs: 3,
            batch_size: Some(usize::MAX),
            step_size: Some(0.75),
            ..config
        }
        .resolve_parameters(11, 3, 5)?;
        assert_eq!(explicit.batch_size, 11);
        assert_eq!(explicit.initial_step_size, 0.75);
        assert_eq!(explicit.total_updates, 3);
        Ok(())
    }

    // Reference uses direct squared differences and the sum/count EMA equations, independent
    // of the SIMD assignment kernel and production centroid interpolation formula.
    fn scalar_reference(
        data: rowmajor::Ref<'_, f32>,
        mut centers: rowmajor::Owned<f32>,
        epochs: usize,
        batch_size: usize,
        alpha0: f64,
        seed: u64,
    ) -> (rowmajor::Owned<f32>, Vec<u32>) {
        let (n, k, d) = (data.nrows(), centers.nrows(), data.ncols());
        let mut rng = StdRng::seed_from_u64(seed);
        let mut order: Vec<_> = (0..n).collect();
        let mut mass = vec![2.0_f64.powi(-16); k];
        let mut labels = vec![u32::MAX; n];
        let mut step = 0;
        for _ in 0..epochs {
            order.shuffle(&mut rng);
            for indices in order.chunks(batch_size) {
                let mut counts = vec![0; k];
                let mut sums = vec![0.0; k * d];
                for &i in indices {
                    let mut best = (f64::INFINITY, 0);
                    for (j, center) in centers.rows().enumerate() {
                        let distance: f64 = data
                            .row(i)
                            .iter()
                            .zip(center)
                            .map(|(&x, &c)| (f64::from(x) - f64::from(c)).powi(2))
                            .sum();
                        if distance < best.0 {
                            best = (distance, j);
                        }
                    }
                    labels[i] = best.1 as u32;
                    counts[best.1] += 1;
                    for (coordinate, &value) in data.row(i).iter().enumerate() {
                        sums[best.1 * d + coordinate] += f64::from(value);
                    }
                }
                let progress = step as f64 / (epochs * n.div_ceil(batch_size)) as f64;
                let alpha = alpha0 * (0.55 + 0.45 * (std::f64::consts::PI * progress).cos());
                for j in 0..k {
                    let old_mass = mass[j];
                    mass[j] = (1.0 - alpha) * old_mass + alpha * counts[j] as f64;
                    if counts[j] > 0 {
                        for (coordinate, value) in centers.row_mut(j).iter_mut().enumerate() {
                            let old_sum = old_mass * f64::from(*value);
                            let new_sum =
                                (1.0 - alpha) * old_sum + alpha * sums[j * d + coordinate];
                            *value = (new_sum / mass[j]) as f32;
                        }
                    }
                }
                step += 1;
            }
        }
        // Finalization is independently reconstructed by original row index.
        for j in 0..k {
            let members: Vec<_> = labels
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| (label as usize == j).then_some(i))
                .collect();
            if !members.is_empty() {
                for coordinate in 0..d {
                    let sum: f64 = members
                        .iter()
                        .map(|&i| f64::from(data.row(i)[coordinate]))
                        .sum();
                    centers.row_mut(j)[coordinate] = (sum / members.len() as f64) as f32;
                }
            }
        }
        (centers, labels)
    }

    #[test]
    fn matches_scalar_ema_with_tail_batches() -> TestResult {
        let mut data_rng = StdRng::seed_from_u64(0xf53ee);
        for dim in [1, 3, 7, 17, 32] {
            let data = rowmajor::Owned::from_fn(67, dim, |_| data_rng.random_range(-2.0_f32..2.0));
            for batch_size in [1, 7, 16, 24, 67] {
                let mut centers =
                    rowmajor::Owned::from_fn(5, dim, |rc| data.row(rc.row * 9)[rc.col]);
                let config = EmeansConfig {
                    epochs: 3,
                    batch_size: Some(batch_size),
                    step_size: Some(0.35),
                    revive: false,
                    ..Default::default()
                };
                let (expected, labels) =
                    scalar_reference(data.as_view(), centers.clone(), 3, batch_size, 0.35, 29);
                let result = emeans_from_centers(
                    data.as_view(),
                    centers.as_view_mut(),
                    &config,
                    &mut StdRng::seed_from_u64(29),
                    &DontCancel,
                )?;
                assert_eq!(result.assignments, labels, "d={dim}, batch={batch_size}");
                assert_eq!(result.updates, 3 * 67_usize.div_ceil(batch_size));
                for (&got, &expected) in centers.as_slice().iter().zip(expected.as_slice()) {
                    assert_near(f64::from(got), f64::from(expected), 2e-6);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn final_means_use_original_order_including_last_short_batch() -> TestResult {
        let data = rowmajor::Owned::from_fn(35, 3, |rc| {
            let group = (rc.row * 11) % 3;
            (20 * group + rc.row % 7 + rc.col) as f32
        });
        let mut centers = rowmajor::Owned::from_fn(3, 3, |rc| (20 * rc.row) as f32);
        let config = EmeansConfig {
            epochs: 4,
            batch_size: Some(16),
            ..Default::default()
        };
        let result = emeans_from_centers(
            data.as_view(),
            centers.as_view_mut(),
            &config,
            &mut StdRng::seed_from_u64(72),
            &DontCancel,
        )?;
        assert_eq!(result.assignments.len(), 35);
        assert_eq!(result.updates, 12);
        for j in 0..3 {
            let members: Vec<_> = result
                .assignments
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| (label as usize == j).then_some(i))
                .collect();
            assert!(!members.is_empty());
            for coordinate in 0..3 {
                let sum: f64 = members
                    .iter()
                    .map(|&i| f64::from(data.row(i)[coordinate]))
                    .sum();
                assert_eq!(
                    centers.row(j)[coordinate],
                    (sum / members.len() as f64) as f32
                );
            }
        }
        Ok(())
    }

    #[test]
    fn pq_sized_revival_preserves_final_means_with_a_tail_batch() -> TestResult {
        let (n, dim, k) = (521, 8, 256);
        let data = rowmajor::Owned::from_fn(n, dim, |rc| {
            let group = (rc.row * 11) % 16;
            (20 * group + rc.row % 7 + rc.col) as f32
        });
        let mut centers = rowmajor::Owned::from_fn(k, dim, |rc| {
            if rc.row < 16 {
                (20 * rc.row + rc.col) as f32
            } else {
                (10_000 + rc.row + rc.col) as f32
            }
        });
        // A single epoch makes every revival overlap the closing-mean accumulation.
        let config = EmeansConfig {
            epochs: 1,
            ..Default::default()
        };
        let result = emeans_from_centers(
            data.as_view(),
            centers.as_view_mut(),
            &config,
            &mut StdRng::seed_from_u64(72),
            &DontCancel,
        )?;
        assert_eq!(result.batch_size, 2);
        assert_eq!(n % result.batch_size, 1);
        assert_eq!(result.updates, n.div_ceil(result.batch_size));
        assert!(result.revived_clusters > 0);
        assert_eq!(result.assignments.len(), n);
        assert!(result.assignments.iter().all(|&label| (label as usize) < k));
        assert!(centers.as_slice().iter().all(|value| value.is_finite()));
        for j in 0..k {
            let members: Vec<_> = result
                .assignments
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| (label as usize == j).then_some(i))
                .collect();
            if members.is_empty() {
                continue;
            }
            for coordinate in 0..dim {
                let sum: f64 = members
                    .iter()
                    .map(|&i| f64::from(data.row(i)[coordinate]))
                    .sum();
                let expected = (sum / members.len() as f64) as f32;
                assert_near(
                    f64::from(centers.row(j)[coordinate]),
                    f64::from(expected),
                    0.0,
                );
            }
        }
        Ok(())
    }

    #[test]
    fn full_batch_one_epoch_equals_lloyd_update() -> TestResult {
        let data =
            rowmajor::Owned::from_fn(31, 5, |rc| (20 * (rc.row % 3) + rc.row % 5 + rc.col) as f32);
        let mut centers = rowmajor::Owned::from_fn(3, 5, |rc| (20 * rc.row) as f32);
        let mut baseline = centers.clone();
        let config = EmeansConfig {
            epochs: 1,
            batch_size: Some(31),
            step_size: Some(1.0),
            revive: false,
            ..Default::default()
        };
        let result = emeans_from_centers(
            data.as_view(),
            centers.as_view_mut(),
            &config,
            &mut StdRng::seed_from_u64(15),
            &DontCancel,
        )?;
        let (labels, _) =
            crate::algorithms::kmeans::lloyds::lloyds(data.as_view(), baseline.as_view_mut(), 1);
        assert_eq!(result.assignments, labels);
        assert_eq!(centers.as_slice(), baseline.as_slice());
        Ok(())
    }

    #[test]
    fn native_initialization_is_reproducible_and_ignores_output_contents() -> TestResult {
        let data = rowmajor::Owned::from_fn(79, 6, |rc| {
            ((rc.row * 17 + rc.col * 31) % 101) as f32 / 10.0
        });
        let mut first = rowmajor::Owned::from_element(7, 6, f32::NAN);
        let mut second = rowmajor::Owned::from_element(7, 6, 123.0);
        let config = EmeansConfig::default();
        let a = emeans(
            data.as_view(),
            first.as_view_mut(),
            &config,
            &mut StdRng::seed_from_u64(9),
            &DontCancel,
        )?;
        let b = emeans(
            data.as_view(),
            second.as_view_mut(),
            &config,
            &mut StdRng::seed_from_u64(9),
            &DontCancel,
        )?;
        assert_eq!(a.assignments, b.assignments);
        assert_eq!(first.as_slice(), second.as_slice());
        assert!(first.as_slice().iter().all(|x| x.is_finite()));
        Ok(())
    }

    #[test]
    fn single_point_all_centers_and_duplicate_rows() -> TestResult {
        for (n, k, value) in [(1, 1, 2.0), (19, 7, 0.0), (19, 19, -3.0)] {
            let data = rowmajor::Owned::from_element(n, 3, value);
            let mut centers = rowmajor::Owned::from_element(k, 3, 0.0);
            let result = emeans(
                data.as_view(),
                centers.as_view_mut(),
                &EmeansConfig::default(),
                &mut StdRng::seed_from_u64(8),
                &DontCancel,
            )?;
            assert_eq!(result.assignments.len(), n);
            assert!(result.assignments.iter().all(|&j| (j as usize) < k));
            assert!(centers.as_slice().iter().all(|x| x.is_finite()));
            for &label in &result.assignments {
                assert!(centers.row(label as usize).iter().all(|&x| x == value));
            }
        }
        let data = rowmajor::Owned::from_fn(17, 2, |rc| (rc.row * 10 + rc.col) as f32);
        let mut centers = rowmajor::Owned::from_element(17, 2, 0.0);
        let result = emeans(
            data.as_view(),
            centers.as_view_mut(),
            &EmeansConfig::default(),
            &mut StdRng::seed_from_u64(8),
            &DontCancel,
        )?;
        for (row, &label) in data.rows().zip(&result.assignments) {
            assert_eq!(row, centers.row(label as usize));
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_shapes_and_configs() {
        for (n, d, k, cd) in [
            (0, 3, 1, 3),
            (3, 0, 1, 0),
            (3, 3, 0, 3),
            (3, 3, 4, 3),
            (3, 3, 1, 2),
        ] {
            let data = rowmajor::Owned::from_element(n, d, 0.0);
            let mut centers = rowmajor::Owned::from_element(k, cd, 0.0);
            let result = emeans(
                data.as_view(),
                centers.as_view_mut(),
                &EmeansConfig::default(),
                &mut StdRng::seed_from_u64(0),
                &DontCancel,
            );
            assert!(matches!(result, Err(EmeansError::InvalidShape)));
        }
        let mut invalid = vec![
            EmeansConfig {
                epochs: 0,
                ..Default::default()
            },
            EmeansConfig {
                batch_size: Some(0),
                ..Default::default()
            },
        ];
        for alpha in [0.0, -0.5, 1.01, f64::NAN, f64::INFINITY] {
            invalid.push(EmeansConfig {
                step_size: Some(alpha),
                ..Default::default()
            });
        }
        for config in invalid {
            assert!(matches!(
                config.resolve_parameters(10, 2, 3),
                Err(EmeansError::InvalidConfig(_))
            ));
        }
        let overflow = EmeansConfig {
            epochs: usize::MAX,
            batch_size: Some(1),
            ..Default::default()
        };
        assert!(matches!(
            overflow.resolve_parameters(2, 1, 1),
            Err(EmeansError::SizeOverflow)
        ));
    }

    #[test]
    fn rejects_nonfinite_and_overflowing_norms_before_training() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let data = rowmajor::Owned::from_element(5, 3, value);
            let mut centers = rowmajor::Owned::from_element(2, 3, 1.0);
            let result = emeans(
                data.as_view(),
                centers.as_view_mut(),
                &EmeansConfig::default(),
                &mut StdRng::seed_from_u64(0),
                &DontCancel,
            );
            assert!(matches!(result, Err(EmeansError::NumericalRange)));
            assert!(centers.as_slice().iter().all(|&x| x == 1.0));
            let data = rowmajor::Owned::from_element(5, 3, 1.0);
            centers.row_mut(1)[0] = value;
            let result = emeans_from_centers(
                data.as_view(),
                centers.as_view_mut(),
                &EmeansConfig::default(),
                &mut StdRng::seed_from_u64(0),
                &DontCancel,
            );
            assert!(matches!(result, Err(EmeansError::NumericalRange)));
        }
    }

    struct CancelAfter {
        polls: Cell<usize>,
        allowed: usize,
    }

    impl Cancelation for CancelAfter {
        fn should_cancel(&self) -> bool {
            let polls = self.polls.get();
            self.polls.set(polls + 1);
            polls >= self.allowed
        }
    }

    #[test]
    fn cancellation_is_observed_before_and_during_training() {
        let data = rowmajor::Owned::from_fn(17, 3, |rc| (rc.row + rc.col) as f32);
        let config = EmeansConfig {
            batch_size: Some(4),
            ..Default::default()
        };
        for allowed in [0, 4] {
            let mut centers = rowmajor::Owned::from_element(2, 3, 100.0);
            let cancel = CancelAfter {
                polls: Cell::new(0),
                allowed,
            };
            let result = emeans_from_centers(
                data.as_view(),
                centers.as_view_mut(),
                &config,
                &mut StdRng::seed_from_u64(3),
                &cancel,
            );
            assert!(matches!(result, Err(EmeansError::Canceled)));
            if allowed == 0 {
                assert!(centers.as_slice().iter().all(|&x| x == 100.0));
            } else {
                assert!(centers.as_slice().iter().any(|&x| x != 100.0));
            }
        }
    }

    #[test]
    fn revival_uses_distinct_donors_and_conserves_active_mass() {
        let mut centers = rowmajor::Owned::from_fn(5, 3, |rc| (10 * rc.row + rc.col + 1) as f32);
        let original = centers.clone();
        let mut masses = vec![8.0, 4.0, 0.0, 0.0, 0.0];
        assert_eq!(
            revive_underused_clusters(
                centers.as_view_mut(),
                &mut masses,
                &mut StdRng::seed_from_u64(11)
            ),
            2
        );
        assert_eq!(&masses[..2], &[4.0, 2.0]);
        assert_eq!(masses.iter().sum::<f64>(), 12.0);
        assert_eq!(masses[4], 0.0);
        assert_eq!(centers.row(4), original.row(4));
        assert!((masses[2] == 4.0 && masses[3] == 2.0) || (masses[2] == 2.0 && masses[3] == 4.0));
        for (target, &mass) in masses.iter().enumerate().take(4).skip(2) {
            let source = if mass == 4.0 { 0 } else { 1 };
            for coordinate in 0..3 {
                assert_near(
                    f64::from(centers.row(source)[coordinate])
                        + f64::from(centers.row(target)[coordinate]),
                    2.0 * f64::from(original.row(source)[coordinate]),
                    4e-6,
                );
            }
        }
    }

    #[test]
    fn revival_separates_zero_centers_and_skips_active_clusters() {
        let mut centers = rowmajor::Owned::from_element(2, 3, 0.0);
        let mut masses = vec![2.0, 0.0];
        assert_eq!(
            revive_underused_clusters(
                centers.as_view_mut(),
                &mut masses,
                &mut StdRng::seed_from_u64(1)
            ),
            1
        );
        assert_eq!(masses, [1.0, 1.0]);
        assert_eq!(centers.row(0), &[0.0, 0.0, 0.0]);
        assert_eq!(centers.row(1), &[0.0_f32.next_up(), 0.0, 0.0]);
        let previous = centers.clone();
        assert_eq!(
            revive_underused_clusters(
                centers.as_view_mut(),
                &mut masses,
                &mut StdRng::seed_from_u64(1)
            ),
            0
        );
        assert_eq!(centers.as_slice(), previous.as_slice());
    }
}
