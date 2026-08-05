/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Deterministic overlapping partition construction for PiPNN.
//!
//! The stage maps real dataset points to bounded leaf ID lists. Numerical work
//! reuses the partition kernel and dense GEMM. A stage-owned pool leases scratch
//! to Rayon chunks and takes it back after each chunk; computation never holds
//! the pool lock, and no thread-local cleanup protocol is required.
//!
//! ```text
//! replica root IDs ──> work queue
//!                         │
//!                         v
//!                  sample leaders
//!                         │
//!        gather stripes ─> GEMM distances ─> nearest leaders
//!                                             │
//!                                             v
//!                                stable scatter by leader
//!                                  │                   │
//!                           size <= c_max        oversized cluster
//!                                  │                   │
//!                           completed leaf      next recursion level
//!                                  └──────────┬────────┘
//!                                             v
//!                              global small-leaf merge
//!                                             v
//!                               coverage/bound validation
//! ```
//!
//! | Recursion level | Assignment multiplicity |
//! | --- | --- |
//! | `level < fanout.len()` | `fanout[level]` nearest leaders |
//! | later levels | one nearest leader until bounded |
//! | replica boundary | independent deterministic seed |

use std::collections::HashSet;

use crate::{ANNError, ANNResult, utils::VectorRepr};
use diskann_linalg::Transpose;
use diskann_utils::{
    object_pool::{AsPooled, ObjectPool},
    views::{MatrixView, MutMatrixView},
};
use diskann_vector::{Norm, distance::Metric, norm::FastL2NormSquared};
use rand::{SeedableRng, prelude::IndexedRandom};
use rayon::prelude::*;

use super::{
    PiPNNConfig,
    partition_kernel::{PartitionInput, PartitionKernel, PartitionScales},
};

// Private algorithm and batching constants live together. None are user policy.
const PARTITION_SEED: u64 = 1_000;
const REPLICA_SEED_STEP: u64 = 7_919;
const LEADER_CAP: usize = 1_000;
const ASSIGNMENT_CACHE_TARGET_BYTES: usize = 524_288;
const MIN_ASSIGNMENT_STRIPE_POINTS: usize = 32;
const MAX_ASSIGNMENT_STRIPE_POINTS: usize = 1_024;
const PARALLEL_SCATTER_MIN_POINTS: usize = 100_000;
const MAX_PARTITION_ITERATIONS: usize = 30;

/// A partition failure with enough context to diagnose non-progressing input.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum PartitionError {
    #[error("PiPNN cannot partition an empty dataset")]
    EmptyDataset,
    #[error("PiPNN cannot partition vectors with zero dimensions")]
    EmptyDimensions,
    #[error("dataset has {0} points, which exceeds the u32 ID limit")]
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
    point_scales: Vec<f32>,
}

impl AsPooled<()> for StripeBuffers {
    fn create(_: ()) -> Self {
        Self::default()
    }

    fn modify(&mut self, _: ()) {
        // Scratch retains its high-water allocation across leases; active
        // prefixes are established by `assign_stripe` before every read.
    }
}

/// Stage-owned high-water scratch storage for partition assignment.
///
/// `ObjectPool` owns the short pop/push lock and returns leases through RAII,
/// including error and panic paths. Gather, GEMM, and top-k run while only the
/// leased `StripeBuffers` is held.
type StripeBufferPool = ObjectPool<StripeBuffers>;

/// Partition every configured replica into overlapping bounded leaves.
///
/// Each oversized work item samples `ceil(p_samp * points)` leaders (clamped
/// to the private leader bound), assigns every point to its nearest `fanout`
/// leaders for the current level, and recurses only on oversized clusters.
/// Levels beyond `fanout.len()` retain one leader assignment. Completed small
/// leaves are merged without exceeding `c_max`; every input point must remain
/// covered once per replica. The caller installs the operation in its pool.
pub(crate) fn partition<T>(
    data: MatrixView<'_, T>,
    config: PiPNNConfig,
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
    // Prepare metric and ISA dispatch before replicas spawn Rayon work. The
    // Copy handle is shared read-only; every stripe calls its direct function
    // pointer instead of redispatching in the recursive hot path.
    let kernel = PartitionKernel::new(metric);
    let stripe_buffers = StripeBufferPool::new((), 0, None);
    for replica in 0..config.replicas {
        let seed = replica_seed(replica);
        let mut replica_leaves =
            partition_replica(data, &config, metric, &kernel, seed, &stripe_buffers)?;
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
    kernel: &PartitionKernel,
    seed: u64,
    stripe_buffers: &StripeBufferPool,
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
                *slot = Some(partition_one_level(
                    data,
                    config,
                    metric,
                    kernel,
                    item,
                    stripe_buffers,
                ));
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
    let Some(largest) = work.iter().max_by_key(|item| item.indices.len()) else {
        return global_merge_small(leaves, config.c_min, config.c_max);
    };
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
    kernel: &PartitionKernel,
    item: WorkItem,
    stripe_buffers: &StripeBufferPool,
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
    let clusters = assign_to_leaders(
        data,
        &item.indices,
        &leaders,
        fanout,
        metric,
        kernel,
        stripe_buffers,
    )?;

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

fn replica_seed(replica: usize) -> u64 {
    PARTITION_SEED.wrapping_add((replica as u64).wrapping_mul(REPLICA_SEED_STEP))
}

// A single LCG mixer derives recursive seeds. Wrapping makes the mapping stable
// across debug/release builds and supported platforms.
fn mix_seed(seed: u64, salt: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(salt)
}

/// Assign each point to its nearest `fanout` sampled leaders.
///
/// Leader vectors are gathered once. Points are processed in cache-sized
/// stripes, while a worker chunk retains one leased scratch buffer across all of
/// its stripes. The flat assignment matrix preserves point order and is then
/// scattered into per-leader clusters; preserving order is required for fixed
/// seed determinism in later recursion levels.
fn assign_to_leaders<T>(
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    leader_ids: &[u32],
    fanout: usize,
    metric: Metric,
    kernel: &PartitionKernel,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<Vec<Vec<u32>>>
where
    T: VectorRepr + Send + Sync,
{
    let dimension_count = data.ncols();
    let leader_values_len = checked_area("leader data", leader_ids.len(), dimension_count)?;
    let mut leader_values = filled_vec(leader_values_len, 0.0f32)?;
    gather_vectors(data, leader_ids, &mut leader_values)?;

    let mut leader_scales = if matches!(metric, Metric::L2 | Metric::Cosine) {
        filled_vec(leader_ids.len(), 0.0f32)?
    } else {
        Vec::new()
    };
    for (scale, leader_vector) in leader_scales
        .iter_mut()
        .zip(leader_values.chunks_exact(dimension_count))
    {
        // Leader norms participate in the top-k ordering. Preserve the original
        // scalar reduction order: reassociating this short setup pass through a
        // SIMD norm changes low bits and can send near-tied points down different
        // recursive partition paths.
        *scale = leader_vector.iter().map(|value| value * value).sum();
        if metric == Metric::Cosine {
            *scale = scale.sqrt();
        }
    }

    let fanout = fanout.min(leader_ids.len());
    let assignment_len = checked_area("partition assignments", point_ids.len(), fanout)?;
    let mut assignments = filled_vec(assignment_len, 0u32)?;
    let stripe_points = assignment_stripe_point_count(leader_ids.len());
    let stripe_assignment_count = checked_area("assignment stripe", stripe_points, fanout)?;
    let stripe_count = point_ids.len().div_ceil(stripe_points);
    let worker_stripe_count = stripe_count.div_ceil(rayon::current_num_threads().max(1));
    let worker_point_count = checked_area("assignment worker", worker_stripe_count, stripe_points)?;
    let worker_assignment_count = checked_area("assignment worker", worker_point_count, fanout)?;

    // Each worker chunk owns one scratch value and reuses it for its stripes.
    // build_graph pins this terminal operation to the caller-owned pool.
    #[allow(clippy::disallowed_methods)]
    assignments
        .par_chunks_mut(worker_assignment_count)
        .enumerate()
        .try_for_each(|(worker, worker_assignments)| {
            let mut buffers = stripe_buffers.get_ref(());
            let worker_first = worker * worker_point_count;
            for (stripe, stripe_assignments) in worker_assignments
                .chunks_mut(stripe_assignment_count)
                .enumerate()
            {
                let first_point = worker_first + stripe * stripe_points;
                let stripe_point_count = stripe_assignments.len() / fanout;
                assign_stripe(
                    data,
                    &point_ids[first_point..first_point + stripe_point_count],
                    &leader_values,
                    &leader_scales,
                    metric,
                    kernel,
                    fanout,
                    &mut buffers,
                    stripe_assignments,
                )?;
            }
            Ok::<(), ANNError>(())
        })?;

    scatter_assignments(point_ids, &assignments, fanout, leader_ids.len())
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn assign_stripe<T>(
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    leader_values: &[f32],
    leader_scales: &[f32],
    metric: Metric,
    kernel: &PartitionKernel,
    fanout: usize,
    buffers: &mut StripeBuffers,
    assignments: &mut [u32],
) -> ANNResult<()>
where
    T: VectorRepr,
{
    let point_count = point_ids.len();
    let dimensions = data.ncols();
    let leader_count = leader_values.len() / dimensions;
    let point_values_len = checked_area("point stripe", point_count, dimensions)?;
    let dots_len = checked_area("dot-product stripe", point_count, leader_count)?;
    let output_len = checked_area("partition assignments", point_count, fanout)?;
    // Scratch keeps its high-water length and every consumer receives an
    // explicit active prefix. Resizing to the exact stripe shape would be
    // correct but re-zeroes the buffer whenever a pooled value moves between
    // work items with different leader counts: `stripe_points` is derived from
    // `leader_count`, so the point buffer swings between roughly 768 KiB and 6 MiB
    // and `Vec::resize` only truncates on the way down, then memsets the whole
    // delta on the way back up.
    grow_fallible(&mut buffers.points, point_values_len, 0.0)?;
    grow_fallible(&mut buffers.dots, dots_len, 0.0)?;
    let StripeBuffers {
        points: point_buffer,
        dots: dot_buffer,
        point_scales: point_scale_buffer,
    } = buffers;
    let point_values = &mut point_buffer[..point_values_len];
    let dots = &mut dot_buffer[..dots_len];
    gather_vectors(data, point_ids, point_values)?;
    diskann_linalg::sgemm(
        Transpose::None,
        Transpose::Ordinary,
        point_count,
        leader_count,
        dimensions,
        1.0,
        point_values,
        leader_values,
        None,
        dots,
    )
    .map_err(ANNError::opaque)?;

    let point_scales = if metric == Metric::Cosine {
        grow_fallible(point_scale_buffer, point_count, 0.0)?;
        let point_scales = &mut point_scale_buffer[..point_count];
        for (scale, point_values) in point_scales
            .iter_mut()
            .zip(point_values.chunks_exact(dimensions))
        {
            *scale = FastL2NormSquared.evaluate(point_values);
        }
        &*point_scales
    } else {
        &[]
    };
    let scales = match metric {
        Metric::L2 => PartitionScales::L2 {
            leader_squared_norms: leader_scales,
        },
        Metric::Cosine => PartitionScales::Cosine {
            point_squared_norms: point_scales,
            leader_norms: leader_scales,
        },
        Metric::CosineNormalized | Metric::InnerProduct => PartitionScales::None,
    };
    let dots = MatrixView::try_from(&*dots, point_count, leader_count).map_err(|_| {
        ANNError::opaque(PartitionError::InvalidBufferLength {
            buffer: "dot-product stripe",
            expected: dots_len,
            actual: dots.len(),
        })
    })?;
    let output = MutMatrixView::try_from(assignments, point_count, fanout).map_err(|error| {
        ANNError::opaque(PartitionError::InvalidBufferLength {
            buffer: "partition assignments",
            expected: output_len,
            actual: error.into_inner().len(),
        })
    })?;
    kernel
        .nearest_leaders(PartitionInput { dots, scales }, output)
        .map_err(ANNError::opaque)
}

fn gather_vectors<T>(data: MatrixView<'_, T>, indices: &[u32], output: &mut [f32]) -> ANNResult<()>
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
    for (&index, vector_output) in indices.iter().zip(output.chunks_exact_mut(data.ncols())) {
        T::as_f32_into(data.row(index as usize), vector_output).map_err(Into::<ANNError>::into)?;
    }
    Ok(())
}

/// Convert the flat point-major assignment matrix into leader-major clusters.
///
/// Small inputs use one serial exact-capacity pass. Large inputs form at most
/// one partial cluster set per Rayon worker, then merge each leader independently.
/// Concatenating partials in stripe order keeps the same member order as the
/// serial implementation while removing a large serial copy tail.
fn scatter_assignments(
    points: &[u32],
    assignments: &[u32],
    fanout: usize,
    leaders: usize,
) -> ANNResult<Vec<Vec<u32>>> {
    if points.len() < PARALLEL_SCATTER_MIN_POINTS {
        return scatter_serial(points, assignments, fanout, leaders);
    }

    let stripe_points = points.len().div_ceil(rayon::current_num_threads().max(1));
    let stripe_assignment_count = checked_area("scatter assignment stripe", stripe_points, fanout)?;
    let stripes = points.len().div_ceil(stripe_points);
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
                .par_chunks(stripe_points)
                .zip(assignments.par_chunks(stripe_assignment_count)),
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

    // See the pool invariant at the other partition terminal operations.
    #[allow(clippy::disallowed_methods)]
    sizes
        .into_par_iter()
        .enumerate()
        .map(|(leader, size)| {
            let mut cluster = Vec::new();
            cluster.try_reserve_exact(size).map_err(ANNError::opaque)?;
            for local in &locals {
                cluster.extend_from_slice(&local[leader]);
            }
            Ok(cluster)
        })
        .collect()
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
    for (&point, point_assignments) in points.iter().zip(assignments.chunks_exact(fanout)) {
        for &leader in point_assignments {
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
        if remainder.len() < c_min
            && let Some(last) = merged.last_mut()
        {
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

/// Grow `values` to at least `len` elements, never shrinking it.
///
/// Callers slice the active prefix themselves. Shrinking would force the next
/// larger stripe to re-zero the reclaimed tail, which is the dominant cost when
/// one pooled buffer serves work items with different stripe shapes.
fn grow_fallible<T: Clone>(values: &mut Vec<T>, len: usize, value: T) -> ANNResult<()> {
    if values.len() >= len {
        return Ok(());
    }
    values
        .try_reserve(len - values.len())
        .map_err(ANNError::opaque)?;
    values.resize(len, value);
    Ok(())
}

fn checked_area(buffer: &'static str, rows: usize, cols: usize) -> ANNResult<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| ANNError::opaque(PartitionError::ShapeOverflow { buffer, rows, cols }))
}

fn assignment_stripe_point_count(leader_count: usize) -> usize {
    let point_count = ASSIGNMENT_CACHE_TARGET_BYTES / (leader_count.max(1) * size_of::<f32>());
    let point_count = if point_count.is_power_of_two() {
        point_count
    } else {
        point_count.next_power_of_two() / 2
    };
    point_count.clamp(MIN_ASSIGNMENT_STRIPE_POINTS, MAX_ASSIGNMENT_STRIPE_POINTS)
}

#[cfg(test)]
mod tests;
