/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Deterministic overlapping partition construction for PiPNN.
//!
//! The stage maps real dataset rows to bounded leaf ID lists. Numerical work
//! reuses the partition kernel and dense GEMM; scratch belongs to the Rayon
//! iterator that uses it, so no thread-local cleanup protocol is required.

use std::collections::HashSet;

use crate::{utils::VectorRepr, ANNError, ANNResult};
use diskann_linalg::Transpose;
use diskann_utils::views::MatrixView;
use diskann_vector::{distance::Metric, norm::FastL2NormSquared, Norm};
use rand::{prelude::IndexedRandom, SeedableRng};
use rayon::prelude::*;

use crate::{
    partition_kernel::{nearest_leaders, PartitionTopK},
    PiPNNConfig,
};

// Private algorithm and batching constants live together. None are user policy.
const PARTITION_SEED: u64 = 1_000;
const LEADER_CAP: usize = 1_000;
const ASSIGNMENT_CACHE_TARGET_BYTES: usize = 512 * 1024;
const MIN_ASSIGNMENT_STRIPE_ROWS: usize = 32;
const MAX_ASSIGNMENT_STRIPE_ROWS: usize = 1_024;
const PARALLEL_SCATTER_MIN_POINTS: usize = 100_000;
const SCATTER_STRIPE_ROWS: usize = 64 * 1024;
const MAX_PARTITION_ITERATIONS: usize = 30;

/// A partition failure with enough context to diagnose non-progressing input.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum PartitionError {
    #[error("PiPNN cannot partition an empty dataset")]
    EmptyDataset,
    #[error("PiPNN cannot partition vectors with zero dimensions")]
    EmptyDimensions,
    #[error("dataset has {0} rows, which exceeds the u32 ID limit")]
    TooManyPoints(usize),
    #[error("{buffer} shape {rows} x {cols} overflows usize")]
    ShapeOverflow {
        buffer: &'static str,
        rows: usize,
        cols: usize,
    },
    #[error(
        "partition stopped after {limit} iterations with an oversized cluster of size \
         {size} at level {level}"
    )]
    IterationLimit {
        size: usize,
        level: usize,
        limit: usize,
    },
    #[error("partition produced an invalid leaf of size {size}; expected 1..={limit}")]
    InvalidLeaf { size: usize, limit: usize },
    #[error("invalid {buffer} length: expected {expected}, got {actual}")]
    InvalidBufferLength {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("partition worker did not publish its result")]
    MissingWorkerResult,
}

struct WorkItem {
    indices: Vec<u32>,
    level: usize,
    seed: u64,
}

#[derive(Default)]
struct StripeBuffers {
    points: Vec<f32>,
    dots: Vec<f32>,
    row_scales: Vec<f32>,
}

/// Partition every configured replica of `data` into leaves no larger than
/// `config.c_max`. The caller is responsible for installing this operation in
/// its build-owned Rayon pool.
pub(crate) fn partition<T>(
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    metric: Metric,
) -> ANNResult<Vec<Vec<u32>>>
where
    T: VectorRepr + Send + Sync,
{
    let points = data.nrows();
    if points == 0 {
        return Err(ANNError::opaque(PartitionError::EmptyDataset));
    }
    if data.ncols() == 0 {
        return Err(ANNError::opaque(PartitionError::EmptyDimensions));
    }
    if points > u32::MAX as usize {
        return Err(ANNError::opaque(PartitionError::TooManyPoints(points)));
    }

    let mut leaves = Vec::new();
    for replica in 0..config.replicas {
        let seed = mix_seed(PARTITION_SEED, replica as u64);
        let mut replica_leaves = partition_replica(data, config, metric, seed)?;
        leaves
            .try_reserve(replica_leaves.len())
            .map_err(ANNError::opaque)?;
        leaves.append(&mut replica_leaves);
    }
    validate_leaves(&leaves, config.c_max)?;
    Ok(leaves)
}

fn partition_replica<T>(
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    metric: Metric,
    seed: u64,
) -> ANNResult<Vec<Vec<u32>>>
where
    T: VectorRepr + Send + Sync,
{
    let initial_indices = point_ids(data.nrows())?;
    if data.nrows() <= config.c_max {
        let mut leaves = Vec::new();
        leaves.try_reserve_exact(1).map_err(ANNError::opaque)?;
        leaves.push(initial_indices);
        return Ok(leaves);
    }

    let mut leaves = Vec::new();
    let mut work = Vec::new();
    work.try_reserve_exact(1).map_err(ANNError::opaque)?;
    work.push(WorkItem {
        indices: initial_indices,
        level: 0,
        seed,
    });

    for _ in 0..MAX_PARTITION_ITERATIONS {
        if work.is_empty() {
            return global_merge_small(leaves, config.c_min, config.c_max);
        }

        let mut results = Vec::new();
        results
            .try_reserve_exact(work.len())
            .map_err(ANNError::opaque)?;
        results.resize_with(work.len(), || None);
        // build_graph installs this complete private call tree into the
        // caller-owned pool; the indexed fill cannot escape that pool.
        #[allow(clippy::disallowed_methods)]
        results
            .par_iter_mut()
            .zip(work.into_par_iter())
            .for_each(|(slot, item)| {
                *slot = Some(partition_one_level(data, config, metric, item));
            });

        let mut next_work = Vec::new();
        for result in results {
            let (mut pending, mut finished) =
                result.ok_or_else(|| ANNError::opaque(PartitionError::MissingWorkerResult))??;
            next_work
                .try_reserve(pending.len())
                .map_err(ANNError::opaque)?;
            leaves
                .try_reserve(finished.len())
                .map_err(ANNError::opaque)?;
            next_work.append(&mut pending);
            leaves.append(&mut finished);
        }
        work = next_work;
    }

    if work.is_empty() {
        return global_merge_small(leaves, config.c_min, config.c_max);
    }
    let mut largest = &work[0];
    for item in &work[1..] {
        if item.indices.len() > largest.indices.len() {
            largest = item;
        }
    }
    Err(ANNError::opaque(PartitionError::IterationLimit {
        size: largest.indices.len(),
        level: largest.level,
        limit: MAX_PARTITION_ITERATIONS,
    }))
}

fn partition_one_level<T>(
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    metric: Metric,
    item: WorkItem,
) -> ANNResult<(Vec<WorkItem>, Vec<Vec<u32>>)>
where
    T: VectorRepr + Send + Sync,
{
    let points = item.indices.len();
    let fanout = config.fanout.get(item.level).copied().unwrap_or(1);
    let leaders = sample_leaders(
        &item.indices,
        config.p_samp,
        mix_seed(item.seed, points as u64),
    )?;
    let clusters = assign_to_leaders(data, &item.indices, &leaders, fanout, metric)?;

    let mut pending = Vec::new();
    let mut finished = Vec::new();
    pending
        .try_reserve(clusters.len())
        .map_err(ANNError::opaque)?;
    finished
        .try_reserve(clusters.len())
        .map_err(ANNError::opaque)?;
    let child_seed = mix_seed(item.seed, points as u64);
    for cluster in clusters {
        if cluster.is_empty() {
            continue;
        }
        if cluster.len() <= config.c_max {
            finished.push(cluster);
        } else {
            pending.push(WorkItem {
                indices: cluster,
                level: item.level + 1,
                seed: child_seed,
            });
        }
    }
    Ok((pending, finished))
}

fn sample_leaders(points: &[u32], sampling_fraction: f64, seed: u64) -> ANNResult<Vec<u32>> {
    let count = sample_num_leaders(points.len(), sampling_fraction);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut leaders = Vec::new();
    leaders.try_reserve_exact(count).map_err(ANNError::opaque)?;
    leaders.extend(points.choose_multiple(&mut rng, count).copied());
    Ok(leaders)
}

fn sample_num_leaders(points: usize, sampling_fraction: f64) -> usize {
    ((points as f64 * sampling_fraction).ceil() as usize)
        .clamp(2, LEADER_CAP)
        .min(points)
}

// A single LCG mixer derives both replica and recursive seeds. Wrapping makes
// the mapping stable across debug/release builds and supported platforms.
fn mix_seed(seed: u64, salt: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(salt)
}

fn assign_to_leaders<T>(
    data: MatrixView<'_, T>,
    points: &[u32],
    leaders: &[u32],
    fanout: usize,
    metric: Metric,
) -> ANNResult<Vec<Vec<u32>>>
where
    T: VectorRepr + Send + Sync,
{
    let dimensions = data.ncols();
    let leader_values_len = checked_area("leader data", leaders.len(), dimensions)?;
    let mut leader_values = filled_vec(leader_values_len, 0.0f32)?;
    gather_rows(data, leaders, &mut leader_values)?;

    let mut leader_scales = if matches!(metric, Metric::L2 | Metric::Cosine) {
        filled_vec(leaders.len(), 0.0f32)?
    } else {
        Vec::new()
    };
    for (scale, row) in leader_scales
        .iter_mut()
        .zip(leader_values.chunks_exact(dimensions))
    {
        *scale = FastL2NormSquared.evaluate(row);
        if metric == Metric::Cosine {
            *scale = scale.sqrt();
        }
    }

    let fanout = fanout.min(leaders.len());
    let assignment_len = checked_area("partition assignments", points.len(), fanout)?;
    let mut assignments = filled_vec(assignment_len, 0u32)?;
    let stripe_rows = assignment_stripe_rows(leaders.len());
    let assignment_stripe = checked_area("assignment stripe", stripe_rows, fanout)?;

    // build_graph pins this terminal operation to the caller-owned pool.
    #[allow(clippy::disallowed_methods)]
    assignments
        .par_chunks_mut(assignment_stripe)
        .enumerate()
        .try_for_each_init(StripeBuffers::default, |buffers, (stripe, output)| {
            let first = stripe * stripe_rows;
            let rows = output.len() / fanout;
            let point_values_len = checked_area("point stripe", rows, dimensions)?;
            let dots_len = checked_area("dot-product stripe", rows, leaders.len())?;
            resize_fallible(&mut buffers.points, point_values_len, 0.0)?;
            resize_fallible(&mut buffers.dots, dots_len, 0.0)?;
            gather_rows(data, &points[first..first + rows], &mut buffers.points)?;
            diskann_linalg::sgemm(
                Transpose::None,
                Transpose::Ordinary,
                rows,
                leaders.len(),
                dimensions,
                1.0,
                &buffers.points,
                &leader_values,
                None,
                &mut buffers.dots,
            )
            .map_err(ANNError::opaque)?;

            let row_scales = if metric == Metric::Cosine {
                resize_fallible(&mut buffers.row_scales, rows, 0.0)?;
                for (scale, row) in buffers
                    .row_scales
                    .iter_mut()
                    .zip(buffers.points.chunks_exact(dimensions))
                {
                    *scale = FastL2NormSquared.evaluate(row);
                }
                buffers.row_scales.as_slice()
            } else {
                &[]
            };
            nearest_leaders(
                PartitionTopK {
                    dots: &buffers.dots,
                    rows,
                    leaders: leaders.len(),
                    row_scales,
                    leader_scales: &leader_scales,
                    metric,
                },
                fanout,
                output,
            )
            .map_err(ANNError::opaque)
        })?;

    scatter_assignments(points, &assignments, fanout, leaders.len())
}

fn gather_rows<T>(data: MatrixView<'_, T>, indices: &[u32], output: &mut [f32]) -> ANNResult<()>
where
    T: VectorRepr,
{
    let expected = checked_area("gather output", indices.len(), data.ncols())?;
    if output.len() != expected {
        return Err(ANNError::opaque(PartitionError::InvalidBufferLength {
            buffer: "gather output",
            expected,
            actual: output.len(),
        }));
    }
    for (&index, row) in indices.iter().zip(output.chunks_exact_mut(data.ncols())) {
        T::as_f32_into(data.row(index as usize), row).map_err(Into::<ANNError>::into)?;
    }
    Ok(())
}

fn scatter_assignments(
    points: &[u32],
    assignments: &[u32],
    fanout: usize,
    leaders: usize,
) -> ANNResult<Vec<Vec<u32>>> {
    if points.len() < PARALLEL_SCATTER_MIN_POINTS {
        return scatter_serial(points, assignments, fanout, leaders);
    }

    let assignment_stripe = checked_area("scatter assignment stripe", SCATTER_STRIPE_ROWS, fanout)?;
    let stripes = points.len().div_ceil(SCATTER_STRIPE_ROWS);
    let mut partials = Vec::new();
    partials
        .try_reserve_exact(stripes)
        .map_err(ANNError::opaque)?;
    partials.resize_with(stripes, || None);
    // See the pool invariant at the other partition terminal operations.
    #[allow(clippy::disallowed_methods)]
    partials
        .par_iter_mut()
        .zip(
            points
                .par_chunks(SCATTER_STRIPE_ROWS)
                .zip(assignments.par_chunks(assignment_stripe)),
        )
        .for_each(|(slot, (points, assignments))| {
            *slot = Some(scatter_serial(points, assignments, fanout, leaders));
        });

    let mut locals = Vec::new();
    locals
        .try_reserve_exact(stripes)
        .map_err(ANNError::opaque)?;
    for result in partials {
        locals.push(result.ok_or_else(|| ANNError::opaque(PartitionError::MissingWorkerResult))??);
    }

    let mut sizes = filled_vec(leaders, 0usize)?;
    for local in &locals {
        for (size, cluster) in sizes.iter_mut().zip(local) {
            *size = size.checked_add(cluster.len()).ok_or_else(|| {
                ANNError::opaque(PartitionError::ShapeOverflow {
                    buffer: "cluster size",
                    rows: *size,
                    cols: cluster.len(),
                })
            })?;
        }
    }

    let mut clusters = clusters_with_capacities(&sizes)?;
    for local in locals {
        for (cluster, part) in clusters.iter_mut().zip(local) {
            debug_assert!(cluster.capacity().saturating_sub(cluster.len()) >= part.len());
            cluster.extend(part);
        }
    }
    Ok(clusters)
}

fn scatter_serial(
    points: &[u32],
    assignments: &[u32],
    fanout: usize,
    leaders: usize,
) -> ANNResult<Vec<Vec<u32>>> {
    let mut sizes = filled_vec(leaders, 0usize)?;
    for &leader in assignments {
        let Some(size) = sizes.get_mut(leader as usize) else {
            return Err(ANNError::opaque(PartitionError::InvalidBufferLength {
                buffer: "leader assignment",
                expected: leaders,
                actual: leader as usize + 1,
            }));
        };
        *size = size.checked_add(1).ok_or_else(|| {
            ANNError::opaque(PartitionError::ShapeOverflow {
                buffer: "cluster size",
                rows: *size,
                cols: 1,
            })
        })?;
    }
    let mut clusters = clusters_with_capacities(&sizes)?;
    for (&point, row) in points.iter().zip(assignments.chunks_exact(fanout)) {
        for &leader in row {
            clusters[leader as usize].push(point);
        }
    }
    Ok(clusters)
}

fn clusters_with_capacities(sizes: &[usize]) -> ANNResult<Vec<Vec<u32>>> {
    let mut clusters = Vec::new();
    clusters
        .try_reserve_exact(sizes.len())
        .map_err(ANNError::opaque)?;
    for &size in sizes {
        let mut cluster = Vec::new();
        cluster.try_reserve_exact(size).map_err(ANNError::opaque)?;
        clusters.push(cluster);
    }
    Ok(clusters)
}

fn global_merge_small(
    leaves: Vec<Vec<u32>>,
    c_min: usize,
    c_max: usize,
) -> ANNResult<Vec<Vec<u32>>> {
    let mut merged = Vec::new();
    let mut small_leaves = Vec::new();
    merged.try_reserve(leaves.len()).map_err(ANNError::opaque)?;
    small_leaves
        .try_reserve(leaves.len())
        .map_err(ANNError::opaque)?;
    for leaf in leaves {
        if leaf.len() >= c_min {
            merged.push(leaf);
        } else {
            small_leaves.push(leaf);
        }
    }
    if small_leaves.is_empty() {
        return Ok(merged);
    }

    let mut small = HashSet::new();
    small.try_reserve(c_max).map_err(ANNError::opaque)?;

    for leaf in small_leaves {
        let combined = small.len().checked_add(leaf.len()).ok_or_else(|| {
            ANNError::opaque(PartitionError::ShapeOverflow {
                buffer: "small-leaf merge",
                rows: small.len(),
                cols: leaf.len(),
            })
        })?;
        if combined > c_max {
            merged.push(drain_sorted(&mut small)?);
        }
        small.try_reserve(leaf.len()).map_err(ANNError::opaque)?;
        small.extend(leaf);
        if small.len() >= c_min {
            merged.push(drain_sorted(&mut small)?);
        }
    }

    if !small.is_empty() {
        let mut remainder = drain_sorted(&mut small)?;
        if remainder.len() < c_min {
            if let Some(last) = merged.last_mut() {
                remainder.retain(|id| !last.contains(id));
                let combined = last.len().checked_add(remainder.len()).ok_or_else(|| {
                    ANNError::opaque(PartitionError::ShapeOverflow {
                        buffer: "small-leaf tail merge",
                        rows: last.len(),
                        cols: remainder.len(),
                    })
                })?;
                if combined <= c_max {
                    last.try_reserve(remainder.len())
                        .map_err(ANNError::opaque)?;
                    last.append(&mut remainder);
                    last.sort_unstable();
                }
            }
        }
        if !remainder.is_empty() {
            merged.push(remainder);
        }
    }

    validate_leaves(&merged, c_max)?;
    Ok(merged)
}

fn drain_sorted(set: &mut HashSet<u32>) -> ANNResult<Vec<u32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(set.len())
        .map_err(ANNError::opaque)?;
    values.extend(set.drain());
    values.sort_unstable();
    Ok(values)
}

fn validate_leaves(leaves: &[Vec<u32>], c_max: usize) -> ANNResult<()> {
    if let Some(leaf) = leaves
        .iter()
        .find(|leaf| leaf.is_empty() || leaf.len() > c_max)
    {
        return Err(ANNError::opaque(PartitionError::InvalidLeaf {
            size: leaf.len(),
            limit: c_max,
        }));
    }
    Ok(())
}

fn point_ids(points: usize) -> ANNResult<Vec<u32>> {
    let mut ids = Vec::new();
    ids.try_reserve_exact(points).map_err(ANNError::opaque)?;
    ids.extend(0..points as u32);
    Ok(ids)
}

fn filled_vec<T: Clone>(len: usize, value: T) -> ANNResult<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(ANNError::opaque)?;
    values.resize(len, value);
    Ok(values)
}

fn resize_fallible<T: Clone>(values: &mut Vec<T>, len: usize, value: T) -> ANNResult<()> {
    if len > values.len() {
        values
            .try_reserve(len - values.len())
            .map_err(ANNError::opaque)?;
    }
    values.resize(len, value);
    Ok(())
}

fn checked_area(buffer: &'static str, rows: usize, cols: usize) -> ANNResult<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| ANNError::opaque(PartitionError::ShapeOverflow { buffer, rows, cols }))
}

fn assignment_stripe_rows(leaders: usize) -> usize {
    (ASSIGNMENT_CACHE_TARGET_BYTES / (leaders.max(1) * size_of::<f32>()))
        .clamp(MIN_ASSIGNMENT_STRIPE_ROWS, MAX_ASSIGNMENT_STRIPE_ROWS)
}

#[cfg(test)]
mod tests;
