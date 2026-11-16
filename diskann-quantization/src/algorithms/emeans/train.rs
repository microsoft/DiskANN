/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::{
    Assignment, EmeansConfig, EmeansError,
    assignment::AssignmentBatch,
    config::{ResolvedParameters, scheduled_step_size},
    revival::{REVIVAL_CHECK_INTERVAL, REVIVAL_DISABLED_MAX_DIMENSION, revive_underused_clusters},
};
use crate::{algorithms::kmeans::common::square_norm, cancel::Cancelation};
use diskann_utils::{
    strided::Strided,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use rand::{Rng, seq::SliceRandom};

// Faiss's 2^-16 pseudo-count gives initial centers a small positive prior mass.
// Preserve this published choice (the paper uses 1e-10); see the source in config.rs.
const INITIAL_CLUSTER_MASS: f64 = 1.0 / 65_536.0;

// Local numerical guard, not a fitted parameter: squared L2 is at most 4L when
// both squared norms are <= L. This leaves headroom for rounding and donor splits.
const MAX_SQUARED_NORM: f32 = f32::MAX / 8.0;
// Local polling policy: keep input validation cooperatively cancellable without
// invoking a potentially dynamic cancellation callback for every row.
const CANCELLATION_POLL_ROWS: usize = 4096;

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

fn check_canceled(cancel: &(impl Cancelation + ?Sized)) -> Result<(), EmeansError> {
    if cancel.should_cancel() {
        Err(EmeansError::Canceled)
    } else {
        Ok(())
    }
}

fn checked_squared_norm(row: &[f32]) -> Result<f32, EmeansError> {
    let norm = square_norm(row);
    if !norm.is_finite() || norm > MAX_SQUARED_NORM {
        Err(EmeansError::NumericalRange)
    } else {
        Ok(norm)
    }
}

fn prepare_training(
    data: Strided<'_, f32>,
    centers: rowmajor::Ref<'_, f32>,
    config: &EmeansConfig,
    cancel: &(impl Cancelation + ?Sized),
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
        if i.is_multiple_of(CANCELLATION_POLL_ROWS) {
            check_canceled(cancel)?;
        }
        norms.push(checked_squared_norm(row)?);
    }
    Ok((recipe, norms))
}

/// Train from a uniform sample of distinct input rows, writing centers in place.
///
/// The centers buffer's row count specifies k; its initial contents are ignored.
/// Data may be contiguous or strided, including a borrowed PQ column partition.
/// Uses integer epoch budgets and the cached f32 SIMD assignment kernel with f64 statistics.
///
/// # Errors
/// Returns an error for invalid shapes/configuration, unsupported numeric range or
/// cancellation. Cancellation can leave partially trained centers. Allocation failures
/// follow the allocator's usual behavior.
pub fn emeans<'a>(
    data: impl Into<Strided<'a, f32>>,
    mut centers: rowmajor::Mut<'_, f32>,
    config: &EmeansConfig,
    rng: &mut (impl Rng + ?Sized),
    cancel: &(impl Cancelation + ?Sized),
) -> Result<EmeansResult, EmeansError> {
    let data = data.into();
    let (recipe, norms) = prepare_training(data, centers.as_view(), config, cancel)?;
    initialize_centers(data, centers.as_view_mut(), rng);
    train_default(data, centers, config, recipe, &norms, rng, cancel)
}

/// Refine caller-supplied centers, for warm starts or matched-initialization comparisons.
///
/// Final nonempty centers are means of the returned last-epoch online assignments.
/// Unassigned centers retain their last position. Evaluate nearest-center loss separately:
/// finalization does not perform an extra assignment pass.
///
/// # Errors
/// Same as [emeans], additionally checking the supplied centers' numeric range.
pub fn emeans_from_centers<'a>(
    data: impl Into<Strided<'a, f32>>,
    centers: rowmajor::Mut<'_, f32>,
    config: &EmeansConfig,
    rng: &mut (impl Rng + ?Sized),
    cancel: &(impl Cancelation + ?Sized),
) -> Result<EmeansResult, EmeansError> {
    let data = data.into();
    let (recipe, norms) = prepare_training(data, centers.as_view(), config, cancel)?;
    validate_centers(centers.as_view())?;
    train_default(data, centers, config, recipe, &norms, rng, cancel)
}

/// Train from sampled centers using a caller-supplied nearest-center backend.
///
/// Only the current batch is packed into contiguous rows for [Assignment]; strided
/// input does not require a full dataset copy. The default [emeans] path instead gathers
/// directly into cached SIMD blocks. Backend instances may be reused across runs.
///
/// # Errors
/// Same validation as [emeans]. Backend errors are retained as the source of
/// [EmeansError::Assignment]; invalid or unwritten labels return
/// [EmeansError::InvalidAssignment]. Errors can leave partially trained centers.
pub fn emeans_with_assignment<'a, A: Assignment + ?Sized>(
    data: impl Into<Strided<'a, f32>>,
    mut centers: rowmajor::Mut<'_, f32>,
    config: &EmeansConfig,
    assignment: &mut A,
    rng: &mut (impl Rng + ?Sized),
    cancel: &(impl Cancelation + ?Sized),
) -> Result<EmeansResult, EmeansError<A::Error>> {
    let data = data.into();
    let (recipe, norms) = prepare_training(data, centers.as_view(), config, cancel)
        .map_err(EmeansError::with_assignment_error)?;
    drop(norms);
    initialize_centers(data, centers.as_view_mut(), rng);
    train_with_assignment(data, centers, config, recipe, assignment, rng, cancel)
}

/// Refine caller-supplied centers using a caller-supplied nearest-center backend.
///
/// See [emeans_from_centers] for output semantics and [emeans_with_assignment] for
/// batch packing and backend requirements.
///
/// # Errors
/// Same as [emeans_with_assignment], additionally checking the supplied centers' range.
pub fn emeans_from_centers_with_assignment<'a, A: Assignment + ?Sized>(
    data: impl Into<Strided<'a, f32>>,
    centers: rowmajor::Mut<'_, f32>,
    config: &EmeansConfig,
    assignment: &mut A,
    rng: &mut (impl Rng + ?Sized),
    cancel: &(impl Cancelation + ?Sized),
) -> Result<EmeansResult, EmeansError<A::Error>> {
    let data = data.into();
    let (recipe, norms) = prepare_training(data, centers.as_view(), config, cancel)
        .map_err(EmeansError::with_assignment_error)?;
    drop(norms);
    validate_centers(centers.as_view()).map_err(EmeansError::with_assignment_error)?;
    train_with_assignment(data, centers, config, recipe, assignment, rng, cancel)
}

fn initialize_centers(
    data: Strided<'_, f32>,
    mut centers: rowmajor::Mut<'_, f32>,
    rng: &mut (impl Rng + ?Sized),
) {
    let chosen = rand::seq::index::sample(rng, data.nrows(), centers.nrows());
    for (row, index) in centers.rows_mut().zip(chosen.iter()) {
        row.copy_from_slice(data.row(index));
    }
}

fn validate_centers(centers: rowmajor::Ref<'_, f32>) -> Result<(), EmeansError> {
    for center in centers.rows() {
        checked_squared_norm(center)?;
    }
    Ok(())
}

fn train_default(
    data: Strided<'_, f32>,
    centers: rowmajor::Mut<'_, f32>,
    config: &EmeansConfig,
    recipe: ResolvedParameters,
    norms: &[f32],
    rng: &mut (impl Rng + ?Sized),
    cancel: &(impl Cancelation + ?Sized),
) -> Result<EmeansResult, EmeansError> {
    let mut full_batch = AssignmentBatch::new(recipe.batch_size, data.ncols())?;
    let mut remainder_batch = AssignmentBatch::new(data.nrows() % recipe.batch_size, data.ncols())?;
    train_inner(
        data,
        centers,
        config,
        recipe,
        rng,
        cancel,
        |indices, centers, center_norms, labels| {
            let batch = if indices.len() == recipe.batch_size {
                &mut full_batch
            } else {
                &mut remainder_batch
            };
            batch.assign(data, indices, norms, centers, center_norms, labels);
            Ok(())
        },
    )
}

fn train_with_assignment<A: Assignment + ?Sized>(
    data: Strided<'_, f32>,
    centers: rowmajor::Mut<'_, f32>,
    config: &EmeansConfig,
    recipe: ResolvedParameters,
    assignment: &mut A,
    rng: &mut (impl Rng + ?Sized),
    cancel: &(impl Cancelation + ?Sized),
) -> Result<EmeansResult, EmeansError<A::Error>> {
    let mut full_batch = rowmajor::Owned::try_from_element(recipe.batch_size, data.ncols(), 0.0)
        .map_err(|_| EmeansError::<A::Error>::SizeOverflow)?;
    let mut remainder_batch =
        rowmajor::Owned::try_from_element(data.nrows() % recipe.batch_size, data.ncols(), 0.0)
            .map_err(|_| EmeansError::<A::Error>::SizeOverflow)?;
    train_inner(
        data,
        centers,
        config,
        recipe,
        rng,
        cancel,
        |indices, centers, _, labels| {
            let batch = if indices.len() == recipe.batch_size {
                &mut full_batch
            } else {
                &mut remainder_batch
            };
            for (row, &index) in batch.rows_mut().zip(indices) {
                row.copy_from_slice(data.row(index));
            }
            assignment.assign(batch.as_view(), centers, labels)
        },
    )
}

fn train_inner<E, F>(
    data: Strided<'_, f32>,
    mut centers: rowmajor::Mut<'_, f32>,
    config: &EmeansConfig,
    recipe: ResolvedParameters,
    rng: &mut (impl Rng + ?Sized),
    cancel: &(impl Cancelation + ?Sized),
    mut assign: F,
) -> Result<EmeansResult, EmeansError<E>>
where
    F: FnMut(&[usize], rowmajor::Ref<'_, f32>, &[f32], &mut [u32]) -> Result<(), E>,
{
    let (n, k, d) = (data.nrows(), centers.nrows(), data.ncols());
    let mut labels = vec![u32::MAX; recipe.batch_size];
    let mut order: Vec<_> = (0..n).collect();
    let mut assignments = vec![0; n];
    let mut center_norms = vec![0.0; k];
    let mut masses = vec![INITIAL_CLUSTER_MASS; k];
    let mut counts = vec![0_usize; k];
    let mut sums = vec![0.0; k * d];
    let mut last_epoch_counts = vec![0_usize; k];
    let mut last_epoch_sums = vec![0.0; k * d];
    let mut step = 0;
    let mut revived_clusters = 0;
    for epoch in 0..config.epochs {
        check_canceled(cancel).map_err(EmeansError::with_assignment_error)?;
        order.shuffle(rng);
        let is_last_epoch = epoch == config.epochs - 1;
        for indices in order.chunks(recipe.batch_size) {
            check_canceled(cancel).map_err(EmeansError::with_assignment_error)?;
            for (norm, row) in center_norms.iter_mut().zip(centers.rows()) {
                *norm = checked_squared_norm(row).map_err(EmeansError::with_assignment_error)?;
            }
            let labels = &mut labels[..indices.len()];
            labels.fill(u32::MAX);
            assign(indices, centers.as_view(), &center_norms, labels)
                .map_err(EmeansError::Assignment)?;
            counts.fill(0);
            sums.fill(0.0);
            for (&index, &label) in indices.iter().zip(labels.iter()) {
                let cluster = label as usize;
                if cluster >= k {
                    return Err(EmeansError::InvalidAssignment {
                        row: index,
                        label,
                        clusters: k,
                    });
                }
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
            if config.revive
                && d > REVIVAL_DISABLED_MAX_DIMENSION
                && step.is_multiple_of(REVIVAL_CHECK_INTERVAL)
            {
                revived_clusters +=
                    revive_underused_clusters(centers.as_view_mut(), &mut masses, rng);
            }
        }
    }
    check_canceled(cancel).map_err(EmeansError::with_assignment_error)?;
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
