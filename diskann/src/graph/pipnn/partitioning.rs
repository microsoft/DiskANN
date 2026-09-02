/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Deterministic overlapping partition construction for PiPNN.
//!
//! A leader is a sampled point that acts as the center of one child partition.
//! A point can join several leaders, so child partitions can overlap.
//!
//! This module owns recursive splitting, leader sampling, row gathering, and
//! assignment scatter. The partition kernel owns GEMM, norm preparation, and
//! local ranking. An `ObjectPool` supplies reusable scratch to Rayon workers.
//!
//! A configured level assigns each point to `fanout[level]` leaders. A deeper
//! level assigns each point to one leader. Each replica uses a different
//! deterministic seed.

use std::collections::HashSet;

use crate::{ANNError, ANNResult, utils::VectorRepr};
use diskann_quantization::spherical::Pairwise1BitScratch;
use diskann_utils::{
    object_pool::{AsPooled, ObjectPool},
    views::{MatrixView, MutMatrixView},
};
use rand::{SeedableRng, prelude::IndexedRandom};
use rayon::prelude::*;

use super::{
    PiPNNConfig,
    kernel_metric::PartitionMetric,
    partition_kernel::{
        PartitionKernelWorkspace, PreparedLeaders, UNASSIGNED_LEADER, assign_leaders,
        rank_final_scores,
    },
    simd::PiPNNSIMDSchema,
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

/// Error from partition shape checks or recursion progress.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum PartitionError {
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
}

struct WorkItem {
    indices: Vec<u32>,
    level: usize,
    seed: u64,
}

#[derive(Default)]
struct StripeBuffers {
    point_values: Vec<f32>,
    encoded_rows: Vec<u8>,
    scores: Vec<f32>,
    ranked_scores: Vec<(u32, f32)>,
    pairwise: Pairwise1BitScratch,
    kernel_workspace: PartitionKernelWorkspace,
}

impl AsPooled<()> for StripeBuffers {
    fn create(_: ()) -> Self {
        Self::default()
    }

    fn modify(&mut self, _: ()) {
        // Keep the largest allocation across leases. `assign_point_stripe` defines the
        // active prefix before each read.
    }
}

/// Reusable buffers for point-to-leader assignment.
///
/// `ObjectPool` locks only when it gives or receives a lease. Numerical work
/// holds the lease, not the pool lock.
type StripeBufferPool = ObjectPool<StripeBuffers>;

/// Build overlapping bounded leaves for all configured replicas.
///
/// Each split samples partition centers and assigns every cluster point to its
/// nearest centers. A cluster above `c_max` is split again. A level without a
/// configured fanout assigns each point to one center. Each replica covers every
/// input point.
pub(super) fn partition<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    scorer: Option<&super::rabitq1::Store>,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let mut leaves = Vec::new();
    let stripe_buffers = StripeBufferPool::new((), 0, None);
    for replica in 0..config.replicas {
        let seed = replica_seed(replica);
        let mut replica_leaves =
            partition_replica::<A, M, T>(arch, data, config, seed, &stripe_buffers, scorer)?;
        leaves.append(&mut replica_leaves);
    }
    Ok(leaves)
}

/// Partition one replica until each leaf has at most `c_max` points.
///
/// The function processes one work queue per recursion level. It merges leaves
/// smaller than `c_min` after the queue becomes empty.
fn partition_replica<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    seed: u64,
    stripe_buffers: &StripeBufferPool,
    scorer: Option<&super::rabitq1::Store>,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let initial_indices = point_ids(data.nrows());
    if data.nrows() <= config.c_max {
        return Ok(vec![initial_indices]);
    }

    let mut leaves = Vec::new();
    let mut work = vec![WorkItem {
        indices: initial_indices,
        level: 0,
        seed,
    }];

    for _ in 0..MAX_PARTITION_ITERATIONS {
        if work.is_empty() {
            return merge_undersized_leaves(leaves, config.c_min, config.c_max);
        }

        // Indexed parallel collection preserves work-item order.
        #[allow(clippy::disallowed_methods)]
        let results: ANNResult<Vec<_>> = work
            .into_par_iter()
            .map(|item| {
                partition_work_item::<A, M, T>(arch, data, config, item, stripe_buffers, scorer)
            })
            .collect();

        let mut next_work = Vec::new();
        for (mut pending, mut finished) in results? {
            next_work.append(&mut pending);
            leaves.append(&mut finished);
        }
        work = next_work;
    }

    let Some(largest) = work.iter().max_by_key(|item| item.indices.len()) else {
        return merge_undersized_leaves(leaves, config.c_min, config.c_max);
    };
    Err(ANNError::new(PartitionError::IterationLimit {
        size: largest.indices.len(),
        level: largest.level,
        limit: MAX_PARTITION_ITERATIONS,
    }))
}

/// Split one oversized cluster into child partitions.
///
/// The function samples center points, assigns the cluster points, and returns
/// bounded leaves separately from child clusters that need another split.
fn partition_work_item<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    item: WorkItem,
    stripe_buffers: &StripeBufferPool,
    scorer: Option<&super::rabitq1::Store>,
) -> ANNResult<(Vec<WorkItem>, Vec<Vec<u32>>)>
where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let points = item.indices.len();
    let fanout = config.fanout.get(item.level).copied().unwrap_or(1);
    let leaders = sample_leaders(
        &item.indices,
        config.p_samp,
        mix_seed(item.seed, points as u64),
    );
    let clusters = assign_to_leaders::<A, M, T>(
        arch,
        data,
        &item.indices,
        &leaders,
        fanout,
        stripe_buffers,
        scorer,
    )?;

    let mut pending = Vec::new();
    let mut finished = Vec::new();
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

/// Sample point IDs that act as centers for one partition split.
fn sample_leaders(points: &[u32], sampling_fraction: f64, seed: u64) -> Vec<u32> {
    let count = sampled_leader_count(points.len(), sampling_fraction);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    points.choose_multiple(&mut rng, count).copied().collect()
}

/// Return the number of centers to sample from one cluster.
///
/// The count is `ceil(points * sampling_fraction)`, limited by `LEADER_CAP` and
/// the number of available points. A cluster with at least two points uses at
/// least two centers.
fn sampled_leader_count(points: usize, sampling_fraction: f64) -> usize {
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

/// Assign each cluster point to its nearest sampled partition centers.
///
/// The function gathers center vectors once and evaluates points in bounded
/// stripes. The assignment matrix keeps point order. Scatter preserves this order
/// inside each child partition, which makes recursive sampling deterministic.
fn assign_to_leaders<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    leader_ids: &[u32],
    fanout: usize,
    stripe_buffers: &StripeBufferPool,
    scorer: Option<&super::rabitq1::Store>,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    if let Some(store) = scorer {
        return assign_to_leaders_rabitq1(
            arch,
            point_ids,
            leader_ids,
            fanout,
            stripe_buffers,
            store,
        );
    }
    let dimension_count = data.ncols();
    let leader_values_len = checked_area("leader data", leader_ids.len(), dimension_count)?;
    let mut leader_values = vec![0.0f32; leader_values_len];
    gather_vectors(data, leader_ids, &mut leader_values)?;

    let leader_matrix =
        MatrixView::try_from(leader_values.as_slice(), leader_ids.len(), dimension_count)
            .map_err(|error| ANNError::new(error.as_static()))?;
    let leaders = PreparedLeaders::<M>::new(leader_matrix);

    let fanout = fanout.min(leaders.len());
    let assignment_len = checked_area("partition assignments", point_ids.len(), fanout)?;
    let mut assignments = vec![0u32; assignment_len];
    let stripe_points = assignment_stripe_point_count(leaders.len());
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
                assign_point_stripe::<A, M, T>(
                    arch,
                    data,
                    &point_ids[first_point..first_point + stripe_point_count],
                    &leaders,
                    fanout,
                    &mut buffers,
                    stripe_assignments,
                )?;
            }
            Ok::<(), ANNError>(())
        })?;

    scatter_assignments(point_ids, &assignments, fanout, leader_ids.len())
}

fn assign_to_leaders_rabitq1<A>(
    arch: A,
    point_ids: &[u32],
    leader_ids: &[u32],
    requested_fanout: usize,
    stripe_buffers: &StripeBufferPool,
    store: &super::rabitq1::Store,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: PiPNNSIMDSchema,
{
    let mut leader_storage = Vec::new();
    store
        .gather(leader_ids, &mut leader_storage)
        .map_err(ANNError::new)?;
    let leaders = MatrixView::try_from(
        leader_storage.as_slice(),
        leader_ids.len(),
        store.row_bytes(),
    )
    .map_err(|error| ANNError::new(error.as_static()))?;
    let mut leader_positions: Vec<_> = leader_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(position, id)| (id, position))
        .collect();
    leader_positions.sort_unstable_by_key(|&(id, _)| id);
    let fanout = requested_fanout.min(leader_ids.len());
    let assignment_len = checked_area("partition assignments", point_ids.len(), fanout)?;
    let mut assignments = vec![0u32; assignment_len];
    let stripe_points = assignment_stripe_point_count(leader_ids.len());
    let stripe_assignment_count = checked_area("assignment stripe", stripe_points, fanout)?;
    let stripe_count = point_ids.len().div_ceil(stripe_points);
    let worker_stripe_count = stripe_count.div_ceil(rayon::current_num_threads().max(1));
    let worker_point_count = checked_area("assignment worker", worker_stripe_count, stripe_points)?;
    let worker_assignment_count = checked_area("assignment worker", worker_point_count, fanout)?;
    assignments
        .par_chunks_mut(worker_assignment_count)
        .enumerate()
        .try_for_each(|(worker, worker_assignments)| {
            let mut buffers = stripe_buffers.get_ref(());
            store.prepare_panel(arch, leaders, &mut buffers.pairwise);
            buffers.scores.resize(leader_ids.len(), 0.0);
            let StripeBuffers {
                scores,
                ranked_scores,
                pairwise,
                ..
            } = &mut *buffers;
            let worker_first = worker * worker_point_count;
            for (stripe, stripe_assignments) in worker_assignments
                .chunks_mut(stripe_assignment_count)
                .enumerate()
            {
                let first_point = worker_first + stripe * stripe_points;
                let stripe_point_count = stripe_assignments.len() / fanout;
                for (&point, output) in point_ids[first_point..first_point + stripe_point_count]
                    .iter()
                    .zip(stripe_assignments.chunks_exact_mut(fanout))
                {
                    let self_target = leader_positions
                        .binary_search_by_key(&point, |&(id, _)| id)
                        .ok()
                        .map(|index| leader_positions[index].1);
                    store
                        .score_prepared(
                            arch,
                            point,
                            self_target,
                            leaders,
                            &mut scores[..leader_ids.len()],
                            pairwise,
                        )
                        .map_err(ANNError::new)?;
                    rank_final_scores(arch, &scores[..leader_ids.len()], output, ranked_scores);
                }
            }
            Ok::<(), ANNError>(())
        })?;
    scatter_assignments(point_ids, &assignments, fanout, leader_ids.len())
}

fn rank_scores(scores: &[f32], output: &mut [u32]) {
    let mut tracker = [(u32::MAX, f32::INFINITY); 16];
    for (leader, &score) in scores.iter().enumerate() {
        let last = output.len() - 1;
        if score.partial_cmp(&tracker[last].1) != Some(std::cmp::Ordering::Less) {
            continue;
        }
        tracker[last] = (leader as u32, score);
        let mut slot = last;
        while slot > 0 && tracker[slot].1 < tracker[slot - 1].1 {
            tracker.swap(slot, slot - 1);
            slot -= 1;
        }
    }
    for (destination, &(leader, _)) in output.iter_mut().zip(&tracker) {
        *destination = leader;
    }
}

/// Assign one point stripe to sampled partition centers.
///
/// The function gathers point IDs into a packed `f32` matrix. The partition
/// kernel owns dot products, point norms, and ranking. This function writes the
/// returned leader-column IDs for partition scatter.
#[inline]
fn assign_point_stripe<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    leaders: &PreparedLeaders<'_, M>,
    fanout: usize,
    buffers: &mut StripeBuffers,
    assignments: &mut [u32],
) -> ANNResult<()>
where
    A: PiPNNSIMDSchema,
    M: PartitionMetric,
    T: VectorRepr,
{
    let point_count = point_ids.len();
    let dimensions = data.ncols();
    let point_values_len = checked_area("point stripe", point_count, dimensions)?;
    // Keep each buffer at its largest length. Every operation uses an explicit
    // active prefix.
    grow(&mut buffers.point_values, point_values_len, 0.0);
    let StripeBuffers {
        point_values,
        kernel_workspace,
        ..
    } = buffers;
    let mut points = MutMatrixView::try_from(
        &mut point_values[..point_values_len],
        point_count,
        dimensions,
    )
    .map_err(|error| ANNError::new(error.as_static()))?;
    gather_vectors(data, point_ids, points.as_mut_slice())?;
    let output = MutMatrixView::try_from(assignments, point_count, fanout)
        .map_err(|error| ANNError::new(error.as_static()))?;
    assign_leaders::<A, M>(arch, points.as_view(), leaders, output, kernel_workspace)
}

fn gather_vectors<T>(data: MatrixView<'_, T>, indices: &[u32], output: &mut [f32]) -> ANNResult<()>
where
    T: VectorRepr,
{
    for (&index, vector_output) in indices.iter().zip(output.chunks_exact_mut(data.ncols())) {
        T::as_f32_into(data.row(index as usize), vector_output).map_err(Into::<ANNError>::into)?;
    }
    Ok(())
}

/// Group assigned point IDs by child partition.
///
/// Both the serial and parallel paths preserve point order inside each child.
/// This order is required for deterministic recursive sampling.
fn scatter_assignments(
    points: &[u32],
    assignments: &[u32],
    fanout: usize,
    leaders: usize,
) -> ANNResult<Vec<Vec<u32>>> {
    if points.len() < PARALLEL_SCATTER_MIN_POINTS {
        return Ok(scatter_serial(points, assignments, fanout, leaders));
    }

    let stripe_points = points.len().div_ceil(rayon::current_num_threads().max(1));
    let stripe_assignment_count = checked_area("scatter assignment stripe", stripe_points, fanout)?;
    // Indexed parallel collection preserves stripe order.
    #[allow(clippy::disallowed_methods)]
    let locals: Vec<_> = points
        .par_chunks(stripe_points)
        .zip(assignments.par_chunks(stripe_assignment_count))
        .map(|(points, assignments)| scatter_serial(points, assignments, fanout, leaders))
        .collect();

    let mut sizes = vec![0usize; leaders];
    for local in &locals {
        for (size, cluster) in sizes.iter_mut().zip(local) {
            *size += cluster.len();
        }
    }

    // `build_graph` runs this Rayon operation in the pool from the build context.
    // Each worker creates one independent leader cluster.
    #[allow(clippy::disallowed_methods)]
    let clusters = sizes
        .into_par_iter()
        .enumerate()
        .map(|(leader, size)| {
            let mut cluster = Vec::with_capacity(size);
            for local in &locals {
                cluster.extend_from_slice(&local[leader]);
            }
            cluster
        })
        .collect();
    Ok(clusters)
}

fn scatter_serial(
    points: &[u32],
    assignments: &[u32],
    fanout: usize,
    leaders: usize,
) -> Vec<Vec<u32>> {
    let mut sizes = vec![0usize; leaders];
    for &leader in assignments {
        if leader != UNASSIGNED_LEADER {
            sizes[leader as usize] += 1;
        }
    }
    let mut clusters = clusters_with_capacities(&sizes);
    for (&point, point_assignments) in points.iter().zip(assignments.chunks_exact(fanout)) {
        for &leader in point_assignments {
            if leader != UNASSIGNED_LEADER {
                clusters[leader as usize].push(point);
            }
        }
    }
    clusters
}

fn clusters_with_capacities(sizes: &[usize]) -> Vec<Vec<u32>> {
    sizes.iter().map(|&size| Vec::with_capacity(size)).collect()
}

/// Merge leaves smaller than `c_min` without exceeding `c_max`.
///
/// A `HashSet` removes duplicate point IDs across merged leaves. The function
/// sorts each merged result before it returns.
fn merge_undersized_leaves(
    leaves: Vec<Vec<u32>>,
    c_min: usize,
    c_max: usize,
) -> ANNResult<Vec<Vec<u32>>> {
    let mut merged = Vec::with_capacity(leaves.len());
    let mut small_leaves = Vec::new();
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

    let mut small = HashSet::with_capacity(c_max);

    for leaf in small_leaves {
        let combined = small.len() + leaf.len();
        if combined > c_max {
            merged.push(drain_sorted(&mut small));
        }
        small.extend(leaf);
        if small.len() >= c_min {
            merged.push(drain_sorted(&mut small));
        }
    }

    if !small.is_empty() {
        let mut remainder = drain_sorted(&mut small);
        if remainder.len() < c_min
            && let Some(last) = merged.last_mut()
        {
            remainder.retain(|id| !last.contains(id));
            let combined = last.len() + remainder.len();
            if combined <= c_max {
                last.append(&mut remainder);
                last.sort_unstable();
            }
        }
        if !remainder.is_empty() {
            merged.push(remainder);
        }
    }

    Ok(merged)
}

fn drain_sorted(set: &mut HashSet<u32>) -> Vec<u32> {
    let mut values: Vec<_> = set.drain().collect();
    values.sort_unstable();
    values
}

fn point_ids(points: usize) -> Vec<u32> {
    (0..points as u32).collect()
}

fn grow<T: Clone>(values: &mut Vec<T>, len: usize, value: T) {
    if values.len() < len {
        values.resize(len, value);
    }
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
    use diskann_wide::arch::{self, Target1};
    use rstest::rstest;

    use super::*;

    struct PartitionCall<'a, T> {
        data: MatrixView<'a, T>,
        config: &'a PiPNNConfig,
    }

    struct DispatchPartition(Metric);

    impl<A, T> Target1<A, ANNResult<Vec<Vec<u32>>>, PartitionCall<'_, T>> for DispatchPartition
    where
        A: PiPNNSIMDSchema,
        T: VectorRepr + Send + Sync,
    {
        fn run(self, arch: A, call: PartitionCall<'_, T>) -> ANNResult<Vec<Vec<u32>>> {
            use super::super::kernel_metric::{Cosine, CosineNormalized, InnerProduct, L2};

            match self.0 {
                Metric::L2 => partition::<A, L2, T>(arch, call.data, call.config, None),
                Metric::Cosine => partition::<A, Cosine, T>(arch, call.data, call.config, None),
                Metric::CosineNormalized => {
                    partition::<A, CosineNormalized, T>(arch, call.data, call.config, None)
                }
                Metric::InnerProduct => {
                    partition::<A, InnerProduct, T>(arch, call.data, call.config, None)
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

    fn partition_config(
        c_min: usize,
        c_max: usize,
        fanout: Vec<usize>,
        replicas: usize,
    ) -> PiPNNConfig {
        PiPNNConfig {
            c_max,
            c_min,
            p_samp: 0.25,
            fanout,
            leaf_k: 1,
            replicas,
        }
    }

    fn separated_point_clusters(points: usize, dimensions: usize) -> Matrix<f32> {
        const POINTS_PER_CLUSTER: usize = 8;
        const CLUSTER_SEPARATION: f32 = 10.0;
        const POINT_OFFSET: f32 = 0.001;
        const DIMENSION_OFFSET: f32 = 0.01;

        Matrix::new(
            diskann_utils::views::Init({
                let mut position = 0usize;
                move || {
                    let point = position / dimensions;
                    let dimension = position % dimensions;
                    position += 1;
                    let cluster = point / POINTS_PER_CLUSTER;
                    cluster as f32 * CLUSTER_SEPARATION
                        + point as f32 * POINT_OFFSET
                        + dimension as f32 * DIMENSION_OFFSET
                }
            }),
            points,
            dimensions,
        )
    }

    fn unit_circle_points(points: usize, dimensions: usize) -> Matrix<f32> {
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

    fn assert_partition_invariants(
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

    #[rstest]
    #[case::below_c_max(7)]
    #[case::at_c_max(8)]
    fn partition_returns_one_leaf_when_point_count_does_not_exceed_c_max(
        #[case] point_count: usize,
    ) {
        // Given
        let data = separated_point_clusters(point_count, 3);
        let expected_leaf = vec![(0..point_count as u32).collect::<Vec<_>>()];

        // When
        let actual_leaves = partition_with_runtime_metric(
            data.as_view(),
            &partition_config(2, 8, vec![2], 1),
            Metric::L2,
        )
        .unwrap();

        // Then
        assert_eq!(actual_leaves, expected_leaf);
    }

    #[test]
    fn partition_membership_is_deterministic_for_a_fixed_seed() {
        // Given
        let data = separated_point_clusters(96, 8);
        let config = partition_config(4, 16, vec![3, 2], 2);

        // When
        let first_partition =
            partition_with_runtime_metric(data.as_view(), &config, Metric::L2).unwrap();
        let second_partition =
            partition_with_runtime_metric(data.as_view(), &config, Metric::L2).unwrap();

        // Then
        assert_eq!(
            sorted_memberships(&first_partition),
            sorted_memberships(&second_partition)
        );
    }

    #[test]
    fn partition_respects_capacity_and_replica_coverage() {
        // Given
        let data = separated_point_clusters(96, 8);
        let config = partition_config(4, 16, vec![3, 2], 2);

        // When
        let partition = partition_with_runtime_metric(data.as_view(), &config, Metric::L2).unwrap();

        // Then
        assert_partition_invariants(&partition, 96, 16, 2);
    }

    #[test]
    fn partition_remains_bounded_after_the_fanout_schedule_is_exhausted() {
        let data = separated_point_clusters(80, 4);
        let leaves = partition_with_runtime_metric(
            data.as_view(),
            &partition_config(2, 8, vec![2], 1),
            Metric::L2,
        )
        .unwrap();

        assert_partition_invariants(&leaves, 80, 8, 1);
    }

    #[test]
    fn duplicate_points_return_iteration_limit_instead_of_oversized_leaf() {
        let data = Matrix::new(1.0f32, 24, 4);
        let error = partition_with_runtime_metric(
            data.as_view(),
            &partition_config(2, 4, vec![1], 1),
            Metric::L2,
        )
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
        // Given
        let leaves = vec![vec![9, 3, 1], vec![3, 2], vec![8]];
        let expected_canonical_membership = vec![vec![1, 2, 3, 8, 9]];

        // When
        let actual_leaves = merge_undersized_leaves(leaves, 4, 8).unwrap();

        // Then
        assert_eq!(actual_leaves, expected_canonical_membership);
    }

    #[test]
    fn global_merge_never_overfills_before_reaching_c_min() {
        // Given
        let leaves = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]];
        let expected_capacity_bounded_leaves =
            vec![vec![0, 1, 2, 3, 4, 5, 6, 7], vec![8, 9, 10, 11]];

        // When
        let actual_leaves = merge_undersized_leaves(leaves, 11, 11).unwrap();

        // Then
        assert_eq!(actual_leaves, expected_capacity_bounded_leaves);
    }

    #[test]
    fn global_merge_fills_exact_capacity_before_flushing() {
        // Given
        let leaves = vec![vec![0, 1], vec![2, 3]];
        let expected_exact_capacity_leaf = vec![vec![0, 1, 2, 3]];

        // When
        let actual_leaves = merge_undersized_leaves(leaves, 4, 4).unwrap();

        // Then
        assert_eq!(actual_leaves, expected_exact_capacity_leaf);
    }

    #[test]
    fn replicas_cover_every_point_once_or_more_per_replica() {
        let data = unit_circle_points(72, 5);
        let leaves = partition_with_runtime_metric(
            data.as_view(),
            &partition_config(3, 12, vec![3, 2], 3),
            Metric::CosineNormalized,
        )
        .unwrap();

        assert_partition_invariants(&leaves, 72, 12, 3);
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
                    (point + dimension) as u8
                })
                .collect();
            let f32_data: Vec<f32> = raw.iter().map(|&value| value as f32).collect();
            let converted: Vec<T> = raw.iter().copied().map(&convert).collect();
            let config = partition_config(2, 16, vec![2, 1], 1);
            let expected_f32_partition = partition_with_runtime_metric(
                MatrixView::try_from(&f32_data, points, dimensions).unwrap(),
                &config,
                Metric::L2,
            )
            .unwrap();
            let actual_converted_partition = partition_with_runtime_metric(
                MatrixView::try_from(&converted, points, dimensions).unwrap(),
                &config,
                Metric::L2,
            )
            .unwrap_or_else(|error| panic!("{label} dimensions={dimensions}: {error}"));

            assert_partition_invariants(&actual_converted_partition, points, 16, 1);
            assert_eq!(
                sorted_memberships(&actual_converted_partition),
                sorted_memberships(&expected_f32_partition),
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
    fn leader_assignment_preserves_sequential_norm_reduction_with_l2() {
        fn next_reassociation_regression_value(state: &mut u64) -> f32 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (((*state >> 40) as f32 / 8_388_608.0) - 1.0) * 1_000.0
        }

        // Sequential and SIMD-reassociated leader norms select different top-1
        // leaders for this case. Dot products still use the production GEMM. The
        // test changes only the leader-norm reduction.
        const REDUCTION_BOUNDARY_DIMENSIONS: usize = 129;
        const REASSOCIATION_REGRESSION_SEED: u64 = 0x3a85_f952_c718_6e49;
        let mut state = REASSOCIATION_REGRESSION_SEED;
        let point: Vec<f32> = (0..REDUCTION_BOUNDARY_DIMENSIONS)
            .map(|_| next_reassociation_regression_value(&mut state))
            .collect();
        let leader_zero: Vec<f32> = (0..REDUCTION_BOUNDARY_DIMENSIONS)
            .map(|_| next_reassociation_regression_value(&mut state))
            .collect();
        let leader_one: Vec<f32> = (0..REDUCTION_BOUNDARY_DIMENSIONS)
            .map(|_| next_reassociation_regression_value(&mut state))
            .collect();
        let data: Vec<f32> = leader_zero
            .into_iter()
            .chain(leader_one)
            .chain(point)
            .collect();
        let data = MatrixView::try_from(data.as_slice(), 3, REDUCTION_BOUNDARY_DIMENSIONS).unwrap();
        let expected_clusters = [vec![], vec![2]];

        let actual_clusters = assign_to_leaders::<_, super::super::kernel_metric::L2, _>(
            diskann_wide::ARCH,
            data,
            &[2],
            &[0, 1],
            1,
            &StripeBufferPool::new((), 0, None),
            None,
        )
        .unwrap();

        assert_eq!(actual_clusters, expected_clusters);
    }

    #[rstest]
    fn partition_covers_all_points_without_exceeding_c_max(
        #[values(
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct
        )]
        metric: Metric,
    ) {
        // Given
        let data = unit_circle_points(64, 8);
        let config = partition_config(2, 20, vec![2], 1);

        // When
        let leaves = partition_with_runtime_metric(data.as_view(), &config, metric).unwrap();

        // Then
        assert_partition_invariants(&leaves, 64, 20, 1);
    }

    #[rstest]
    #[case::single_point_minimum(1, 1.0, 1)]
    #[case::sampled_count(10, 0.01, 2)]
    #[case::leader_cap(50_000, 1.0, LEADER_CAP)]
    fn sampled_leader_count_respects_data_size_sampling_and_cap(
        #[case] point_count: usize,
        #[case] sampling_probability: f64,
        #[case] expected_leader_count: usize,
    ) {
        assert_eq!(
            sampled_leader_count(point_count, sampling_probability),
            expected_leader_count
        );
    }

    #[rstest]
    #[case::first_replica(0, 1_000)]
    #[case::second_replica(1, 8_919)]
    fn replica_seed_is_stable(#[case] replica: usize, #[case] expected_seed: u64) {
        assert_eq!(replica_seed(replica), expected_seed);
    }

    #[test]
    fn leader_assignment_preserves_clusters_across_multiple_stripes() {
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
            None,
        )
        .unwrap();

        assert_eq!(clusters[0], (0..1_024).collect::<Vec<_>>());
        assert_eq!(clusters[1], (1_024..2_048).collect::<Vec<_>>());
    }

    #[test]
    fn scatter_omits_unassigned_slots() {
        // Given
        let points = [10, 11];
        let assignments = [0, UNASSIGNED_LEADER, UNASSIGNED_LEADER, 1];
        let expected_clusters = [vec![10], vec![11]];

        // When
        let actual_clusters = scatter_serial(&points, &assignments, 2, 2);

        // Then
        assert_eq!(actual_clusters, expected_clusters);
    }

    #[test]
    fn parallel_scatter_matches_serial_order() {
        // Given
        let leader_count = 7;
        let leaders_per_point = 2;
        let second_leader_offset = 3;
        let points: Vec<u32> = (0..PARALLEL_SCATTER_MIN_POINTS as u32).collect();
        let assignments: Vec<u32> = points
            .iter()
            .flat_map(|point| {
                [
                    point % leader_count,
                    (point + second_leader_offset) % leader_count,
                ]
            })
            .collect();

        // When
        let expected_serial_clusters = scatter_serial(
            &points,
            &assignments,
            leaders_per_point,
            leader_count as usize,
        );
        let actual_parallel_clusters = scatter_assignments(
            &points,
            &assignments,
            leaders_per_point,
            leader_count as usize,
        )
        .unwrap();

        // Then
        assert_eq!(actual_parallel_clusters, expected_serial_clusters);
    }
}
