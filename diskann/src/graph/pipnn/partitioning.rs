/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Deterministic overlapping partition construction for PiPNN.
//!
//! This module converts dataset point IDs into bounded leaf ID lists. It uses
//! dense GEMM and the partition kernel for leader assignment. An `ObjectPool`
//! supplies one reusable buffer set to each Rayon worker chunk. The pool lock is
//! not held during gather, GEMM, or top-k selection.
//!
//! A configured level assigns each point to `fanout[level]` leaders. A deeper
//! level assigns each point to one leader. Each replica uses a different
//! deterministic seed.

use std::collections::HashSet;

use crate::{ANNError, ANNResult, utils::VectorRepr};
use diskann_linalg::Transpose;
use diskann_utils::{
    object_pool::{AsPooled, ObjectPool},
    views::{MatrixView, MutMatrixView},
};
use diskann_vector::{Norm, distance::Metric, norm::FastL2NormSquared};
use diskann_wide::{Architecture, SIMDMask, SIMDSelect, SIMDVector};
use rand::{SeedableRng, prelude::IndexedRandom};
use rayon::prelude::*;

use super::{
    PiPNNConfig,
    kernel_metric::KernelMetric,
    partition_kernel::{
        PartitionInput, PartitionKernelWorkspace, PartitionScales, nearest_leaders,
    },
};

// These constants control internal batching and deterministic seed generation.
const PARTITION_SEED: u64 = 1_000;
const REPLICA_SEED_STEP: u64 = 7_919;
const LEADER_CAP: usize = 1_000;
const ASSIGNMENT_CACHE_TARGET_BYTES: usize = 524_288;
const MIN_ASSIGNMENT_STRIPE_POINTS: usize = 32;
const MAX_ASSIGNMENT_STRIPE_POINTS: usize = 1_024;
const PARALLEL_SCATTER_MIN_POINTS: usize = 100_000;
const MAX_PARTITION_ITERATIONS: usize = 30;

/// Error from partition input checks, allocation, or recursion progress.
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
    kernel: PartitionKernelWorkspace,
}

impl AsPooled<()> for StripeBuffers {
    fn create(_: ()) -> Self {
        Self::default()
    }

    fn modify(&mut self, _: ()) {
        // Keep the largest allocation across leases. `assign_stripe` defines the
        // active prefix before each read.
    }
}

/// Reusable partition buffers owned by this partition call.
///
/// `ObjectPool` locks only when it gives or receives a lease. RAII returns each
/// lease after success, error, or panic. Numerical work holds only the lease.
type StripeBufferPool = ObjectPool<StripeBuffers>;

/// Partition all configured replicas into overlapping bounded leaves.
///
/// An oversized work item samples `ceil(p_samp * points)` leaders, up to
/// `LEADER_CAP`. It assigns each point to the configured number of nearest
/// leaders. It adds each cluster above `c_max` to the work queue.
///
/// A level without a configured fanout assigns each point to one leader. The
/// final merge does not make a leaf larger than `c_max`. Each replica covers
/// every input point. The caller supplies concrete architecture `A` and metric
/// `M`.
pub(super) fn partition<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: KernelMetric,
    T: VectorRepr + Send + Sync,
{
    let points = data.nrows();
    if points == 0 {
        return Err(ANNError::new(PartitionError::EmptyDataset));
    }
    if data.ncols() == 0 {
        return Err(ANNError::new(PartitionError::EmptyDimensions));
    }
    if points > u32::MAX as usize {
        return Err(ANNError::new(PartitionError::TooManyPoints(points)));
    }

    let mut leaves = Vec::new();
    let stripe_buffers = StripeBufferPool::new((), 0, None);
    for replica in 0..config.replicas {
        let seed = replica_seed(replica);
        let mut replica_leaves =
            partition_replica::<A, M, T>(arch, data, config, seed, &stripe_buffers)?;
        leaves
            .try_reserve(replica_leaves.len())
            .map_err(ANNError::new)?;
        leaves.append(&mut replica_leaves);
    }
    validate_leaves(&leaves, config.c_max)?;
    Ok(leaves)
}

fn partition_replica<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    seed: u64,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: KernelMetric,
    T: VectorRepr + Send + Sync,
{
    let initial_indices = point_ids(data.nrows())?;
    if data.nrows() <= config.c_max {
        let mut leaves = Vec::new();
        leaves.try_reserve_exact(1).map_err(ANNError::new)?;
        leaves.push(initial_indices);
        return Ok(leaves);
    }

    let mut leaves = Vec::new();
    let mut work = Vec::new();
    work.try_reserve_exact(1).map_err(ANNError::new)?;
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
            .map_err(ANNError::new)?;
        results.resize_with(work.len(), || None);
        // `build_graph` runs this Rayon operation in the pool from the build
        // context. Each worker writes only to its indexed result slot.
        #[allow(clippy::disallowed_methods)]
        results
            .par_iter_mut()
            .zip(work.into_par_iter())
            .for_each(|(slot, item)| {
                *slot = Some(partition_one_level::<A, M, T>(
                    arch,
                    data,
                    config,
                    item,
                    stripe_buffers,
                ));
            });

        let mut next_work = Vec::new();
        for result in results {
            let (mut pending, mut finished) =
                result.ok_or_else(|| ANNError::new(PartitionError::MissingWorkerResult))??;
            next_work
                .try_reserve(pending.len())
                .map_err(ANNError::new)?;
            leaves.try_reserve(finished.len()).map_err(ANNError::new)?;
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
    Err(ANNError::new(PartitionError::IterationLimit {
        size: largest.indices.len(),
        level: largest.level,
        limit: MAX_PARTITION_ITERATIONS,
    }))
}

fn partition_one_level<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    item: WorkItem,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<(Vec<WorkItem>, Vec<Vec<u32>>)>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: KernelMetric,
    T: VectorRepr + Send + Sync,
{
    let points = item.indices.len();
    let fanout = config.fanout.get(item.level).copied().unwrap_or(1);
    let leaders = sample_leaders(
        &item.indices,
        config.p_samp,
        mix_seed(item.seed, points as u64),
    )?;
    let clusters =
        assign_to_leaders::<A, M, T>(arch, data, &item.indices, &leaders, fanout, stripe_buffers)?;

    let mut pending = Vec::new();
    let mut finished = Vec::new();
    pending.try_reserve(clusters.len()).map_err(ANNError::new)?;
    finished
        .try_reserve(clusters.len())
        .map_err(ANNError::new)?;
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
    leaders.try_reserve_exact(count).map_err(ANNError::new)?;
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

// This LCG derives child seeds. Wrapping arithmetic gives the same mapping in
// debug and release builds on all supported platforms.
fn mix_seed(seed: u64, salt: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(salt)
}

/// Assign each point to its nearest sampled leaders.
///
/// The function gathers leader vectors once. It divides points into bounded
/// stripes. Each worker chunk reuses one leased buffer set for all its stripes.
///
/// The flat assignment matrix keeps point order. The scatter step keeps the same
/// order in each leader cluster. Recursive sampling depends on this order.
fn assign_to_leaders<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    leader_ids: &[u32],
    fanout: usize,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: KernelMetric,
    T: VectorRepr + Send + Sync,
{
    let dimension_count = data.ncols();
    let leader_values_len = checked_area("leader data", leader_ids.len(), dimension_count)?;
    let mut leader_values = filled_vec(leader_values_len, 0.0f32)?;
    gather_vectors(data, leader_ids, &mut leader_values)?;

    let mut leader_scales = if matches!(M::METRIC, Metric::L2 | Metric::Cosine) {
        filled_vec(leader_ids.len(), 0.0f32)?
    } else {
        Vec::new()
    };
    for (scale, leader_vector) in leader_scales
        .iter_mut()
        .zip(leader_values.chunks_exact(dimension_count))
    {
        // Leader norms affect top-k order. Use this scalar reduction order.
        // SIMD reassociation changes low bits and can change a near-tie branch.
        *scale = leader_vector.iter().map(|value| value * value).sum();
        if M::METRIC == Metric::Cosine {
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

    // Each worker chunk reuses one buffer lease for all its stripes.
    // `build_graph` runs this operation in the pool from the build context.
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
                assign_stripe::<A, M, T>(
                    arch,
                    data,
                    &point_ids[first_point..first_point + stripe_point_count],
                    &leader_values,
                    &leader_scales,
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
fn assign_stripe<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    leader_values: &[f32],
    leader_scales: &[f32],
    fanout: usize,
    buffers: &mut StripeBuffers,
    assignments: &mut [u32],
) -> ANNResult<()>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: KernelMetric,
    T: VectorRepr,
{
    let point_count = point_ids.len();
    let dimensions = data.ncols();
    let leader_count = leader_values.len() / dimensions;
    let point_values_len = checked_area("point stripe", point_count, dimensions)?;
    let dots_len = checked_area("dot-product stripe", point_count, leader_count)?;
    let output_len = checked_area("partition assignments", point_count, fanout)?;
    // Keep each buffer at its largest length. Every operation uses an explicit
    // active prefix.
    grow_fallible(&mut buffers.points, point_values_len, 0.0)?;
    grow_fallible(&mut buffers.dots, dots_len, 0.0)?;
    let StripeBuffers {
        points: point_buffer,
        dots: dot_buffer,
        point_scales: point_scale_buffer,
        kernel: kernel_workspace,
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
    .map_err(ANNError::new)?;

    let point_scales = if M::METRIC == Metric::Cosine {
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
    let scales = match M::METRIC {
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
        ANNError::new(PartitionError::InvalidBufferLength {
            buffer: "dot-product stripe",
            expected: dots_len,
            actual: dots.len(),
        })
    })?;
    let output = MutMatrixView::try_from(assignments, point_count, fanout).map_err(|error| {
        ANNError::new(PartitionError::InvalidBufferLength {
            buffer: "partition assignments",
            expected: output_len,
            actual: error.into_inner().len(),
        })
    })?;
    nearest_leaders::<A, M>(
        arch,
        PartitionInput { dots, scales },
        output,
        kernel_workspace,
    )
    .map_err(ANNError::new)
}

fn gather_vectors<T>(data: MatrixView<'_, T>, indices: &[u32], output: &mut [f32]) -> ANNResult<()>
where
    T: VectorRepr,
{
    let expected = checked_area("gather output", indices.len(), data.ncols())?;
    if output.len() != expected {
        return Err(ANNError::new(PartitionError::InvalidBufferLength {
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

/// Convert point-major assignments into leader-major clusters.
///
/// A small input uses one serial pass with exact capacities. A large input makes
/// at most one partial cluster set for each Rayon worker. It then merges each
/// leader independently. Stripe-order concatenation matches serial member order.
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
    partials.try_reserve_exact(stripes).map_err(ANNError::new)?;
    partials.resize_with(stripes, || None);
    // `build_graph` runs this Rayon operation in the pool from the build context.
    // Each worker writes only to its indexed partial result.
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
    locals.try_reserve_exact(stripes).map_err(ANNError::new)?;
    for result in partials {
        locals.push(result.ok_or_else(|| ANNError::new(PartitionError::MissingWorkerResult))??);
    }

    let mut sizes = filled_vec(leaders, 0usize)?;
    for local in &locals {
        for (size, cluster) in sizes.iter_mut().zip(local) {
            *size = size.checked_add(cluster.len()).ok_or_else(|| {
                ANNError::new(PartitionError::ShapeOverflow {
                    buffer: "cluster size",
                    rows: *size,
                    cols: cluster.len(),
                })
            })?;
        }
    }

    // `build_graph` runs this Rayon operation in the pool from the build context.
    // Each worker creates one independent leader cluster.
    #[allow(clippy::disallowed_methods)]
    sizes
        .into_par_iter()
        .enumerate()
        .map(|(leader, size)| {
            let mut cluster = Vec::new();
            cluster.try_reserve_exact(size).map_err(ANNError::new)?;
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
            return Err(ANNError::new(PartitionError::InvalidBufferLength {
                buffer: "leader assignment",
                expected: leaders,
                actual: leader as usize + 1,
            }));
        };
        *size = size.checked_add(1).ok_or_else(|| {
            ANNError::new(PartitionError::ShapeOverflow {
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
        .map_err(ANNError::new)?;
    for &size in sizes {
        let mut cluster = Vec::new();
        cluster.try_reserve_exact(size).map_err(ANNError::new)?;
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
    merged.try_reserve(leaves.len()).map_err(ANNError::new)?;
    small_leaves
        .try_reserve(leaves.len())
        .map_err(ANNError::new)?;
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
    small.try_reserve(c_max).map_err(ANNError::new)?;

    for leaf in small_leaves {
        let combined = small.len().checked_add(leaf.len()).ok_or_else(|| {
            ANNError::new(PartitionError::ShapeOverflow {
                buffer: "small-leaf merge",
                rows: small.len(),
                cols: leaf.len(),
            })
        })?;
        if combined > c_max {
            merged.push(drain_sorted(&mut small)?);
        }
        small.try_reserve(leaf.len()).map_err(ANNError::new)?;
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
                ANNError::new(PartitionError::ShapeOverflow {
                    buffer: "small-leaf tail merge",
                    rows: last.len(),
                    cols: remainder.len(),
                })
            })?;
            if combined <= c_max {
                last.try_reserve(remainder.len()).map_err(ANNError::new)?;
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
    values.try_reserve_exact(set.len()).map_err(ANNError::new)?;
    values.extend(set.drain());
    values.sort_unstable();
    Ok(values)
}

fn validate_leaves(leaves: &[Vec<u32>], c_max: usize) -> ANNResult<()> {
    if let Some(leaf) = leaves
        .iter()
        .find(|leaf| leaf.is_empty() || leaf.len() > c_max)
    {
        return Err(ANNError::new(PartitionError::InvalidLeaf {
            size: leaf.len(),
            limit: c_max,
        }));
    }
    Ok(())
}

fn point_ids(points: usize) -> ANNResult<Vec<u32>> {
    let mut ids = Vec::new();
    ids.try_reserve_exact(points).map_err(ANNError::new)?;
    ids.extend(0..points as u32);
    Ok(ids)
}

fn filled_vec<T: Clone>(len: usize, value: T) -> ANNResult<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(ANNError::new)?;
    values.resize(len, value);
    Ok(values)
}

/// Grow `values` to at least `len` elements and do not shrink it.
///
/// Callers use an explicit active prefix.
fn grow_fallible<T: Clone>(values: &mut Vec<T>, len: usize, value: T) -> ANNResult<()> {
    if values.len() >= len {
        return Ok(());
    }
    values
        .try_reserve(len - values.len())
        .map_err(ANNError::new)?;
    values.resize(len, value);
    Ok(())
}

fn checked_area(buffer: &'static str, rows: usize, cols: usize) -> ANNResult<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| ANNError::new(PartitionError::ShapeOverflow { buffer, rows, cols }))
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
mod tests {
    use diskann_utils::views::{Matrix, MatrixView};
    use diskann_vector::{Half, distance::Metric};
    use diskann_wide::{
        Architecture, SIMDMask, SIMDSelect, SIMDVector,
        arch::{self, Target1},
    };

    use super::*;

    struct PartitionCall<'a, T> {
        data: MatrixView<'a, T>,
        config: &'a PiPNNConfig,
    }

    struct DispatchPartition(Metric);

    impl<A, T> Target1<A, ANNResult<Vec<Vec<u32>>>, PartitionCall<'_, T>> for DispatchPartition
    where
        A: Architecture,
        A::f32x16: std::ops::Div<Output = A::f32x16>,
        <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
        u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
        T: VectorRepr + Send + Sync,
    {
        fn run(self, arch: A, call: PartitionCall<'_, T>) -> ANNResult<Vec<Vec<u32>>> {
            use super::super::kernel_metric::{Cosine, CosineNormalized, InnerProduct, L2};

            match self.0 {
                Metric::L2 => partition::<A, L2, T>(arch, call.data, call.config),
                Metric::Cosine => partition::<A, Cosine, T>(arch, call.data, call.config),
                Metric::CosineNormalized => {
                    partition::<A, CosineNormalized, T>(arch, call.data, call.config)
                }
                Metric::InnerProduct => {
                    partition::<A, InnerProduct, T>(arch, call.data, call.config)
                }
            }
        }
    }

    fn partition_with_runtime_metric<T>(
        data: MatrixView<'_, T>,
        config: &PiPNNConfig,
        metric: Metric,
    ) -> ANNResult<Vec<Vec<u32>>>
    where
        T: VectorRepr + Send + Sync,
    {
        arch::dispatch1_no_features(DispatchPartition(metric), PartitionCall { data, config })
    }

    fn config(c_min: usize, c_max: usize, fanout: Vec<usize>, replicas: usize) -> PiPNNConfig {
        PiPNNConfig {
            c_max,
            c_min,
            p_samp: 0.25,
            fanout,
            k: 1,
            replicas,
        }
    }

    fn clustered_data(points: usize, dimensions: usize) -> Matrix<f32> {
        Matrix::new(
            diskann_utils::views::Init({
                let mut position = 0usize;
                move || {
                    let point = position / dimensions;
                    let dimension = position % dimensions;
                    position += 1;
                    (point / 8) as f32 * 10.0 + dimension as f32 * 0.01 + point as f32 * 0.001
                }
            }),
            points,
            dimensions,
        )
    }

    fn directional_data(points: usize, dimensions: usize) -> Matrix<f32> {
        Matrix::new(
            diskann_utils::views::Init({
                let mut position = 0usize;
                move || {
                    let point = position / dimensions;
                    let dimension = position % dimensions;
                    position += 1;
                    let angle = std::f32::consts::TAU * point as f32 / points as f32;
                    match dimension {
                        0 => angle.cos(),
                        1 => angle.sin(),
                        _ => 0.0,
                    }
                }
            }),
            points,
            dimensions,
        )
    }

    fn sorted_memberships(leaves: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut memberships: Vec<Vec<u32>> = leaves
            .iter()
            .map(|leaf| {
                let mut ids = leaf.clone();
                ids.sort_unstable();
                ids
            })
            .collect();
        memberships.sort();
        memberships
    }

    fn assert_valid_partition_with_runtime_metric(
        leaves: &[Vec<u32>],
        points: usize,
        c_max: usize,
        replicas: usize,
    ) {
        assert!(
            leaves
                .iter()
                .all(|leaf| !leaf.is_empty() && leaf.len() <= c_max)
        );
        let mut counts = vec![0usize; points];
        for leaf in leaves {
            let mut ids = leaf.clone();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), leaf.len(), "duplicate ID inside a leaf");
            for &id in leaf {
                assert!((id as usize) < points);
                counts[id as usize] += 1;
            }
        }
        assert!(counts.iter().all(|&count| count >= replicas));
    }

    #[test]
    fn returns_one_leaf_at_and_below_c_max() {
        for points in [7, 8] {
            let data = clustered_data(points, 3);
            let leaves = partition_with_runtime_metric(
                data.as_view(),
                &config(2, 8, vec![2], 1),
                Metric::L2,
            )
            .unwrap();
            assert_eq!(leaves, vec![(0..points as u32).collect::<Vec<_>>()]);
        }
    }

    #[test]
    fn partition_is_fixed_seed_deterministic_and_bounded() {
        let data = clustered_data(96, 8);
        let config = config(4, 16, vec![3, 2], 2);

        let first = partition_with_runtime_metric(data.as_view(), &config, Metric::L2).unwrap();
        let second = partition_with_runtime_metric(data.as_view(), &config, Metric::L2).unwrap();

        assert_eq!(sorted_memberships(&first), sorted_memberships(&second));
        assert_valid_partition_with_runtime_metric(&first, 96, 16, 2);
        assert!(first.iter().map(Vec::len).sum::<usize>() > 96 * 2);
    }

    #[test]
    fn partition_remains_bounded_after_the_fanout_schedule_is_exhausted() {
        let data = clustered_data(80, 4);
        let leaves =
            partition_with_runtime_metric(data.as_view(), &config(2, 8, vec![2], 1), Metric::L2)
                .unwrap();

        assert_valid_partition_with_runtime_metric(&leaves, 80, 8, 1);
    }

    #[test]
    fn duplicate_points_return_iteration_limit_instead_of_oversized_leaf() {
        let data = Matrix::new(1.0f32, 24, 4);
        let error =
            partition_with_runtime_metric(data.as_view(), &config(2, 4, vec![1], 1), Metric::L2)
                .unwrap_err();
        let error = error.downcast::<PartitionError>().unwrap();

        assert!(matches!(
            error,
            PartitionError::IterationLimit {
                size: 24,
                limit: MAX_PARTITION_ITERATIONS,
                ..
            }
        ));
    }

    #[test]
    fn global_merge_canonicalizes_small_leaf_membership() {
        let leaves = vec![vec![9, 3, 1], vec![3, 2], vec![8]];

        let merged = global_merge_small(leaves, 4, 8).unwrap();

        assert_eq!(merged, vec![vec![1, 2, 3, 8, 9]]);
    }

    #[test]
    fn global_merge_never_overfills_before_reaching_c_min() {
        let leaves = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];

        let merged = global_merge_small(leaves, 11, 11).unwrap();

        assert_eq!(
            merged,
            vec![vec![0, 1, 2, 3, 4, 5, 6, 7], vec![8, 9, 10, 11]]
        );
    }

    #[test]
    fn global_merge_fills_exact_capacity_before_flushing() {
        let merged = global_merge_small(vec![vec![0, 1], vec![2, 3]], 4, 4).unwrap();

        assert_eq!(merged, vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn replicas_cover_every_point_once_or_more_per_replica() {
        let data = directional_data(72, 5);
        let leaves = partition_with_runtime_metric(
            data.as_view(),
            &config(3, 12, vec![3, 2], 3),
            Metric::CosineNormalized,
        )
        .unwrap();

        assert_valid_partition_with_runtime_metric(&leaves, 72, 12, 3);
    }

    fn assert_partition_conversion_matches_f32<T>(label: &str, convert: impl Fn(u8) -> T)
    where
        T: crate::utils::VectorRepr + Send + Sync,
    {
        let points = 64;
        // Partition gather converts source vectors before GEMM. Test conversion
        // tails around 4, 8, 16, and 32 elements.
        for dimensions in [1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let raw: Vec<u8> = (0..points * dimensions)
                .map(|index| {
                    let point = index / dimensions;
                    let dimension = index % dimensions;
                    ((point * 5 + dimension * 7 + point * dimension) % 23) as u8
                })
                .collect();
            let f32_data: Vec<f32> = raw.iter().map(|&value| value as f32).collect();
            let converted: Vec<T> = raw.iter().copied().map(&convert).collect();
            let config = config(2, 16, vec![2, 1], 1);
            let expected = partition_with_runtime_metric(
                MatrixView::try_from(&f32_data, points, dimensions).unwrap(),
                &config,
                Metric::L2,
            )
            .unwrap();
            let actual = partition_with_runtime_metric(
                MatrixView::try_from(&converted, points, dimensions).unwrap(),
                &config,
                Metric::L2,
            )
            .unwrap_or_else(|error| panic!("{label} dimensions={dimensions}: {error}"));

            assert_valid_partition_with_runtime_metric(&actual, points, 16, 1);
            assert_eq!(
                sorted_memberships(&actual),
                sorted_memberships(&expected),
                "{label} dimensions={dimensions}"
            );
        }
    }

    #[test]
    fn f16_partition_matches_f32_across_dimension_boundaries() {
        assert_partition_conversion_matches_f32("f16", |value| Half::from_f32(value as f32));
    }

    #[test]
    fn u8_partition_matches_f32_across_dimension_boundaries() {
        assert_partition_conversion_matches_f32("u8", |value| value);
    }

    #[test]
    fn i8_partition_matches_f32_across_dimension_boundaries() {
        // The same translation in every coordinate preserves L2 ordering.
        assert_partition_conversion_matches_f32("i8", |value| value as i8 - 11);
    }

    #[test]
    fn l2_leader_norms_preserve_scalar_reduction_order() {
        fn next(state: &mut u64) -> f32 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (((*state >> 40) as f32 / 8_388_608.0) - 1.0) * 1_000.0
        }

        // Scalar and SIMD-reassociated leader norms select different top-1
        // leaders for this case. Dot products still use the production GEMM. The
        // test changes only the leader-norm reduction.
        let dimensions = 129;
        let mut state = 0x3a85_f952_c718_6e49;
        let point: Vec<f32> = (0..dimensions).map(|_| next(&mut state)).collect();
        let leader_zero: Vec<f32> = (0..dimensions).map(|_| next(&mut state)).collect();
        let leader_one: Vec<f32> = (0..dimensions).map(|_| next(&mut state)).collect();
        let data: Vec<f32> = leader_zero
            .into_iter()
            .chain(leader_one)
            .chain(point)
            .collect();
        let data = MatrixView::try_from(data.as_slice(), 3, dimensions).unwrap();

        let clusters = assign_to_leaders::<_, super::super::kernel_metric::L2, _>(
            diskann_wide::ARCH,
            data,
            &[2],
            &[0, 1],
            1,
            &StripeBufferPool::new((), 0, None),
        )
        .unwrap();

        assert_eq!(clusters, [vec![], vec![2]]);
    }

    #[test]
    fn all_metrics_produce_valid_partitions() {
        let data = directional_data(64, 8);
        let config = config(2, 20, vec![2], 1);

        for metric in [
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct,
        ] {
            let leaves = partition_with_runtime_metric(data.as_view(), &config, metric).unwrap();
            assert_valid_partition_with_runtime_metric(&leaves, 64, 20, 1);
        }
    }

    #[test]
    fn leader_count_is_bounded() {
        assert_eq!(sample_num_leaders(1, 1.0), 1);
        assert_eq!(sample_num_leaders(10, 0.01), 2);
        assert_eq!(sample_num_leaders(50_000, 1.0), LEADER_CAP);
    }

    #[test]
    fn replica_seed_derivation_is_stable_and_distinct() {
        assert_eq!(replica_seed(0), 1_000);
        assert_eq!(replica_seed(1), 8_919);
    }

    #[test]
    fn assignment_stripes_use_power_of_two_point_counts() {
        assert_eq!(assignment_stripe_point_count(1_000), 128);
        assert_eq!(assignment_stripe_point_count(256), 512);
        assert_eq!(
            assignment_stripe_point_count(1),
            MAX_ASSIGNMENT_STRIPE_POINTS
        );
    }

    #[test]
    fn stripe_buffer_pool_reuses_returned_capacity() {
        let pool = StripeBufferPool::new((), 0, None);
        let points = {
            let mut buffers = pool.get_ref(());
            buffers.points.resize(16, 0.0);
            buffers.points.as_ptr()
        };

        let buffers = pool.get_ref(());
        assert_eq!(buffers.points.as_ptr(), points);
        assert_eq!(buffers.points.len(), 16);
    }

    #[test]
    fn leader_assignment_handles_multiple_stripes() {
        let points = 2_048;
        let data: Vec<f32> = (0..points).map(|point| point as f32).collect();
        let data = MatrixView::try_from(data.as_slice(), points, 1).unwrap();
        let point_ids: Vec<u32> = (0..points as u32).collect();

        let clusters = assign_to_leaders::<_, super::super::kernel_metric::L2, _>(
            diskann_wide::ARCH,
            data,
            &point_ids,
            &[0, 2_047],
            1,
            &StripeBufferPool::new((), 0, None),
        )
        .unwrap();

        assert_eq!(clusters[0], (0..1_024).collect::<Vec<_>>());
        assert_eq!(clusters[1], (1_024..2_048).collect::<Vec<_>>());
    }

    #[test]
    fn parallel_scatter_matches_serial_order() {
        let points: Vec<u32> = (0..PARALLEL_SCATTER_MIN_POINTS as u32).collect();
        let assignments: Vec<u32> = points
            .iter()
            .flat_map(|point| [point % 7, (point + 3) % 7])
            .collect();

        let expected = scatter_serial(&points, &assignments, 2, 7).unwrap();
        let actual = scatter_assignments(&points, &assignments, 2, 7).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn rejects_empty_dataset() {
        let data = Matrix::<f32>::new(0.0, 0, 4);
        let error =
            partition_with_runtime_metric(data.as_view(), &config(1, 4, vec![1], 1), Metric::L2)
                .unwrap_err();

        assert_eq!(
            error.downcast::<PartitionError>().unwrap(),
            PartitionError::EmptyDataset
        );
    }

    #[test]
    fn rejects_zero_dimensions() {
        let data = Matrix::<f32>::new(0.0, 4, 0);
        let error =
            partition_with_runtime_metric(data.as_view(), &config(1, 4, vec![1], 1), Metric::L2)
                .unwrap_err();

        assert_eq!(
            error.downcast::<PartitionError>().unwrap(),
            PartitionError::EmptyDimensions
        );
    }

    #[test]
    fn rejects_invalid_gather_output_length() {
        let data = Matrix::<f32>::new(0.0, 2, 2);
        let error = gather_vectors(data.as_view(), &[0, 1], &mut [0.0; 3]).unwrap_err();

        assert_eq!(
            error.downcast::<PartitionError>().unwrap(),
            PartitionError::InvalidBufferLength {
                buffer: "gather output",
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn rejects_assignment_to_an_unknown_leader() {
        let error = scatter_serial(&[7], &[2], 1, 2).unwrap_err();

        assert_eq!(
            error.downcast::<PartitionError>().unwrap(),
            PartitionError::InvalidBufferLength {
                buffer: "leader assignment",
                expected: 2,
                actual: 3,
            }
        );
    }

    #[test]
    fn rejects_empty_and_oversized_leaves() {
        for (leaves, size) in [(vec![vec![]], 0), (vec![vec![0, 1, 2]], 3)] {
            let error = validate_leaves(&leaves, 2).unwrap_err();
            assert_eq!(
                error.downcast::<PartitionError>().unwrap(),
                PartitionError::InvalidLeaf { size, limit: 2 }
            );
        }
    }
}
