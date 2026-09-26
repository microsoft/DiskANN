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
use diskann_utils::{
    object_pool::{AsPooled, ObjectPool},
    views::{MatrixView, MutMatrixView},
};
use rand::{SeedableRng, prelude::IndexedRandom};
use rayon::prelude::*;

use super::{
    PiPNNConfig,
    partition_kernel::{PartitionKernelWorkspace, assign_leaders},
    partition_metric::PartitionMetric,
    simd::Simd,
    topk::UNASSIGNED,
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

struct PendingPartition {
    point_ids: Vec<u32>,
    level: usize,
    seed: u64,
}

struct PartitionSplit {
    pending: Vec<PendingPartition>,
    leaves: Vec<Vec<u32>>,
}

#[derive(Default)]
struct StripeBuffers {
    point_values: Vec<f32>,
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
/// configured fanout assigns each point to one center. Non-rankable distances
/// do not produce assignments.
pub(super) fn partition<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let mut leaves = Vec::new();
    let stripe_buffers = StripeBufferPool::new((), 0, None);
    for replica in 0..config.replicas {
        let seed = replica_seed(replica);
        let mut replica_leaves =
            partition_replica::<A, M, T>(arch, data, config, seed, &stripe_buffers)?;
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
) -> ANNResult<Vec<Vec<u32>>>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let initial_point_ids = (0..data.nrows() as u32).collect();
    if data.nrows() <= config.c_max {
        return Ok(vec![initial_point_ids]);
    }

    let mut leaves = Vec::new();
    let mut pending = vec![PendingPartition {
        point_ids: initial_point_ids,
        level: 0,
        seed,
    }];

    for _ in 0..MAX_PARTITION_ITERATIONS {
        if pending.is_empty() {
            return Ok(merge_undersized_leaves(leaves, config.c_min, config.c_max));
        }

        // Indexed parallel collection preserves parent-partition order.
        let splits: ANNResult<Vec<_>> = pending
            .into_par_iter()
            .map(|partition| {
                split_partition::<A, M, T>(arch, data, config, partition, stripe_buffers)
            })
            .collect();

        let mut next_level = Vec::new();
        for mut split in splits? {
            next_level.append(&mut split.pending);
            leaves.append(&mut split.leaves);
        }
        pending = next_level;
    }

    let Some(largest) = pending
        .iter()
        .max_by_key(|partition| partition.point_ids.len())
    else {
        return Ok(merge_undersized_leaves(leaves, config.c_min, config.c_max));
    };
    Err(ANNError::new(PartitionError::IterationLimit {
        size: largest.point_ids.len(),
        level: largest.level,
        limit: MAX_PARTITION_ITERATIONS,
    }))
}

/// Split one oversized cluster into child partitions.
///
/// The function samples center points, assigns the cluster points, and returns
/// bounded leaves separately from child clusters that need another split.
fn split_partition<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    config: &PiPNNConfig,
    partition: PendingPartition,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<PartitionSplit>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let split_seed = mix_seed(partition.seed, partition.point_ids.len() as u64);
    let fanout = config.fanout.get(partition.level).copied().unwrap_or(1);
    let leaders = sample_leaders(&partition.point_ids, config.p_samp, split_seed);
    let clusters = assign_to_leaders::<A, M, T>(
        arch,
        data,
        &partition.point_ids,
        &leaders,
        fanout,
        stripe_buffers,
    )?;

    let mut pending = Vec::new();
    let mut leaves = Vec::new();
    for cluster in clusters {
        if cluster.is_empty() {
            continue;
        }
        if cluster.len() <= config.c_max {
            leaves.push(cluster);
        } else {
            pending.push(PendingPartition {
                point_ids: cluster,
                level: partition.level + 1,
                seed: split_seed,
            });
        }
    }
    Ok(PartitionSplit { pending, leaves })
}

/// Sample point IDs that act as centers for one partition split.
fn sample_leaders(points: &[u32], sampling_fraction: f64, seed: u64) -> Vec<u32> {
    let count = sampled_leader_count(points.len(), sampling_fraction);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    points.choose_multiple(&mut rng, count).copied().collect()
}

/// Return the number of centers to sample from one cluster.
///
/// Splitting requires at least two points. The validated sampling fraction is in
/// `(0, 1]`; round up its point count and keep between two and `LEADER_CAP` centers.
fn sampled_leader_count(points: usize, sampling_fraction: f64) -> usize {
    ((points as f64 * sampling_fraction).ceil() as usize).clamp(2, LEADER_CAP)
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
) -> ANNResult<Vec<Vec<u32>>>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let dimension_count = data.ncols();
    let leader_values_len = checked_area("leader data", leader_ids.len(), dimension_count)?;
    let mut leader_values = vec![0.0f32; leader_values_len];
    gather_vectors(data, leader_ids, &mut leader_values)?;

    let leader_matrix =
        MatrixView::try_from(leader_values.as_slice(), leader_ids.len(), dimension_count)
            .map_err(|error| ANNError::new(error.as_static()))?;
    let leaders = M::create_leaders(leader_matrix);
    let leader_count = M::leader_count(&leaders);

    let fanout = fanout.min(leader_count);
    let assignment_len = checked_area("partition assignments", point_ids.len(), fanout)?;
    let mut assignments = vec![0u32; assignment_len];
    let stripe_points = assignment_stripe_point_count(leader_count);
    let stripe_assignment_count = stripe_points * fanout;
    let stripe_count = point_ids.len().div_ceil(stripe_points);
    let worker_stripe_count = stripe_count.div_ceil(rayon::current_num_threads());
    let worker_point_count = checked_area("assignment worker", worker_stripe_count, stripe_points)?;
    let worker_assignment_count = checked_area("assignment worker", worker_point_count, fanout)?;

    // Each worker chunk reuses one buffer lease for all its stripes.
    // `build_graph` runs this operation in the pool from the build context.
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

    Ok(scatter_assignments(
        point_ids,
        &assignments,
        fanout,
        leader_count,
    ))
}

/// Assign one point stripe to sampled partition centers.
///
/// The function gathers point IDs into a packed `f32` matrix. The partition
/// kernel owns ranking-distance construction and ranking. This function writes the
/// returned leader-column IDs for partition scatter.
#[inline]
fn assign_point_stripe<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    leaders: &M::Leaders<'_>,
    fanout: usize,
    buffers: &mut StripeBuffers,
    assignments: &mut [u32],
) -> ANNResult<()>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr,
{
    let point_count = point_ids.len();
    let dimensions = data.ncols();
    let point_values_len = checked_area("point stripe", point_count, dimensions)?;
    // Keep each buffer at its largest length. Every operation uses an explicit
    // active prefix.
    if buffers.point_values.len() < point_values_len {
        buffers.point_values.resize(point_values_len, 0.0);
    }
    let StripeBuffers {
        point_values,
        kernel_workspace,
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
    super::conversion::gather_as_f32(data, indices, output).map_err(|(_, error)| error.into())
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
) -> Vec<Vec<u32>> {
    if points.len() < PARALLEL_SCATTER_MIN_POINTS {
        return scatter_serial(points, assignments, fanout, leaders);
    }

    let stripe_points = points.len().div_ceil(rayon::current_num_threads());
    let stripe_assignment_count = stripe_points * fanout;
    // Indexed parallel collection preserves stripe order.
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
    sizes
        .into_par_iter()
        .enumerate()
        .map(|(leader, size)| {
            let mut cluster = Vec::with_capacity(size);
            for local in &locals {
                cluster.extend_from_slice(&local[leader]);
            }
            cluster
        })
        .collect()
}

fn scatter_serial(
    points: &[u32],
    assignments: &[u32],
    fanout: usize,
    leaders: usize,
) -> Vec<Vec<u32>> {
    let mut sizes = vec![0usize; leaders];
    for &leader in assignments {
        if leader != UNASSIGNED {
            sizes[leader as usize] += 1;
        }
    }
    let mut clusters: Vec<Vec<u32>> = sizes.into_iter().map(Vec::with_capacity).collect();
    for (&point, point_assignments) in points.iter().zip(assignments.chunks_exact(fanout)) {
        for &leader in point_assignments {
            if leader != UNASSIGNED {
                clusters[leader as usize].push(point);
            }
        }
    }
    clusters
}

/// Merge leaves smaller than `c_min` without exceeding `c_max`.
///
/// A `HashSet` removes duplicate point IDs across merged leaves. The function
/// sorts each merged result before it returns.
fn merge_undersized_leaves(leaves: Vec<Vec<u32>>, c_min: usize, c_max: usize) -> Vec<Vec<u32>> {
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
        return merged;
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

    merged
}

fn drain_sorted(set: &mut HashSet<u32>) -> Vec<u32> {
    let mut values: Vec<_> = set.drain().collect();
    values.sort_unstable();
    values
}

fn checked_area(buffer: &'static str, rows: usize, cols: usize) -> ANNResult<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| ANNError::new(PartitionError::ShapeOverflow { buffer, rows, cols }))
}

fn assignment_stripe_point_count(leader_count: usize) -> usize {
    let point_count = ASSIGNMENT_CACHE_TARGET_BYTES / (leader_count * size_of::<f32>());
    let point_count = if point_count.is_power_of_two() {
        point_count
    } else {
        point_count.next_power_of_two() / 2
    };
    point_count.clamp(MIN_ASSIGNMENT_STRIPE_POINTS, MAX_ASSIGNMENT_STRIPE_POINTS)
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::graph::pipnn::{L2, test_support};
    use diskann_wide::ARCH;
    use rstest::rstest;

    // Neither leaf order nor point order within a leaf is part of the result.
    // Preserve repeated leaves and IDs so comparison still detects duplicates.
    fn sorted_leaf_memberships(leaves: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut leaves = test_support::sorted_members_per_row(leaves);
        leaves.sort_unstable();
        leaves
    }

    fn splitting_config() -> PiPNNConfig {
        PiPNNConfig {
            c_max: 2,
            c_min: 1,
            p_samp: 1.0,
            fanout: vec![2],
            leaf_k: 1,
            replicas: 1,
        }
    }

    #[rstest]
    #[case::minimum_sample(7, 0.01, 2)]
    #[case::rounded_fraction(7, 0.31, 3)]
    #[case::all_points(7, 1.0, 7)]
    #[case::below_cap(999, 1.0, 999)]
    #[case::at_cap(1000, 1.0, 1000)]
    #[case::above_cap(1001, 1.0, 1000)]
    fn sampled_leaders_are_a_repeatable_subset_without_replacement(
        #[case] point_count: u32,
        #[case] fraction: f64,
        #[case] expected_count: usize,
    ) {
        let ids: Vec<_> = (0..point_count).map(|id| 10 + 3 * id).collect();

        let mut leaders = sample_leaders(&ids, fraction, 1290);

        assert_eq!(leaders.len(), expected_count);
        assert_eq!(
            leaders.iter().copied().collect::<HashSet<_>>().len(),
            expected_count
        );
        assert!(leaders.iter().all(|id| ids.contains(id)));
        let mut repeated = sample_leaders(&ids, fraction, 1290);
        leaders.sort_unstable();
        repeated.sort_unstable();
        assert_eq!(leaders, repeated);
    }

    #[rstest]
    #[case::nearest_only(1, vec![vec![5, 1], vec![4]])]
    #[case::all_leaders(2, vec![vec![5, 4, 1], vec![5, 4, 1]])]
    #[case::fanout_above_leader_count(9, vec![vec![5, 4, 1], vec![5, 4, 1]])]
    fn points_join_their_nearest_leaders(#[case] fanout: usize, #[case] expected: Vec<Vec<u32>>) {
        // Leader columns are global IDs [2, 0], at x=10 and x=0.
        // The requested points [5, 4, 1] are at x=8, x=2 and x=100.
        let values = [0.0_f32, 100.0, 10.0, -100.0, 2.0, 8.0];
        let data = MatrixView::try_from(&values[..], 6, 1).unwrap();
        let buffers = StripeBufferPool::new((), 0, None);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let actual = pool
            .install(|| {
                assign_to_leaders::<_, L2, _>(ARCH, data, &[5, 4, 1], &[2, 0], fanout, &buffers)
            })
            .unwrap();

        assert_eq!(
            test_support::sorted_members_per_row(&actual),
            test_support::sorted_members_per_row(&expected)
        );
    }

    #[rstest]
    #[case::one_worker(1)]
    #[case::several_workers(3)]
    fn multiple_stripes_and_their_partial_tail_preserve_all_assignments(#[case] workers: usize) {
        // More than two maximum-sized stripes, with three points left over.
        // Each dense row selects two of the three centers at 10, 0 and 4.
        // Every row selects 4; its other center is 10 for x=10 and 0 otherwise.
        let point_count = 2 * MAX_ASSIGNMENT_STRIPE_POINTS + 3;
        let dimensions = 384;
        let values: Vec<_> = (0..point_count)
            .flat_map(|point| std::iter::repeat_n([0.0_f32, 4.0, 10.0][point % 3], dimensions))
            .collect();
        let data = MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
        let ids: Vec<_> = (0..point_count as u32).rev().collect();
        let expected = vec![
            ids.iter()
                .copied()
                .filter(|id| id % 3 == 2)
                .collect::<Vec<_>>(),
            ids.iter()
                .copied()
                .filter(|id| id % 3 != 2)
                .collect::<Vec<_>>(),
            ids.clone(),
        ];
        let buffers = StripeBufferPool::new((), 0, None);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();

        let actual = pool
            .install(|| assign_to_leaders::<_, L2, _>(ARCH, data, &ids, &[2, 0, 1], 2, &buffers))
            .unwrap();

        assert_eq!(
            test_support::sorted_members_per_row(&actual),
            test_support::sorted_members_per_row(&expected)
        );
    }

    #[test]
    fn reused_stripe_buffers_replace_previous_points_and_assignment_width() {
        let values = [100.0_f32, 0.0, 8.0, 10.0, 2.0];
        let data = MatrixView::try_from(&values[..], 5, 1).unwrap();
        let leader_values = [0.0_f32, 10.0, 100.0];
        let leaders = L2::create_leaders(MatrixView::try_from(&leader_values[..], 3, 1).unwrap());
        let mut buffers = StripeBuffers::default();

        for (ids, fanout, expected) in [
            (&[4, 2, 0, 3, 1][..], 2, &[0, 1, 1, 0, 2, 1, 1, 0, 0, 1][..]),
            (&[2, 1][..], 1, &[1, 0][..]),
            (&[1, 0, 4, 2][..], 2, &[0, 1, 2, 1, 0, 1, 1, 0][..]),
        ] {
            let mut assignments = vec![0; ids.len() * fanout];

            assign_point_stripe::<_, L2, _>(
                ARCH,
                data,
                ids,
                &leaders,
                fanout,
                &mut buffers,
                &mut assignments,
            )
            .unwrap();

            let actual: Vec<_> = assignments
                .chunks_exact(fanout)
                .map(<[u32]>::to_vec)
                .collect();
            let expected: Vec<_> = expected.chunks_exact(fanout).map(<[u32]>::to_vec).collect();
            assert_eq!(
                test_support::sorted_members_per_row(&actual),
                test_support::sorted_members_per_row(&expected),
                "points {ids:?}, fanout={fanout}"
            );
        }
    }

    #[rstest]
    #[case::small_input(7)]
    #[case::below_parallel_threshold(PARALLEL_SCATTER_MIN_POINTS - 1)]
    #[case::at_parallel_threshold(PARALLEL_SCATTER_MIN_POINTS)]
    #[case::uneven_parallel_chunks(PARALLEL_SCATTER_MIN_POINTS + 1)]
    fn scatter_groups_assigned_points_by_leader_and_omits_unassigned_slots(
        #[case] point_count: usize,
    ) {
        let ids: Vec<_> = (0..point_count).map(|i| 1_000_000 - i as u32).collect();
        let pattern = [
            [2, 0],
            [UNASSIGNED, 1],
            [0, UNASSIGNED],
            [UNASSIGNED, UNASSIGNED],
        ];
        let assignments: Vec<_> = (0..point_count).flat_map(|i| pattern[i % 4]).collect();
        let expected = vec![
            ids.iter().copied().step_by(2).collect::<Vec<_>>(),
            ids.iter().copied().skip(1).step_by(4).collect(),
            ids.iter().copied().step_by(4).collect(),
            vec![],
        ];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();

        let actual = pool.install(|| scatter_assignments(&ids, &assignments, 2, 4));

        assert_eq!(
            test_support::sorted_members_per_row(&actual),
            test_support::sorted_members_per_row(&expected)
        );
    }

    #[rstest]
    #[case::configured_overlap(0, vec![vec![0, 1], vec![0, 1], vec![2, 3], vec![2, 3]])]
    #[case::after_schedule(1, vec![vec![0], vec![1], vec![2], vec![3]])]
    fn a_split_uses_the_configured_fanout_then_falls_back_to_one_leader(
        #[case] level: usize,
        #[case] expected: Vec<Vec<u32>>,
    ) {
        let values = [0.0_f32, 1.0, 10.0, 11.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let config = splitting_config();
        let parent = PendingPartition {
            point_ids: vec![0, 1, 2, 3],
            level,
            seed: 1290,
        };
        let buffers = StripeBufferPool::new((), 0, None);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let actual = pool
            .install(|| split_partition::<_, L2, _>(ARCH, data, &config, parent, &buffers))
            .unwrap();

        // Every point is a leader, so membership is fixed regardless of output order.
        assert_eq!(
            sorted_leaf_memberships(&actual.leaves),
            sorted_leaf_memberships(&expected)
        );
        assert!(actual.pending.is_empty());
    }

    #[test]
    fn children_above_cmax_are_queued_at_the_next_level() {
        let values = [0.0_f32, 1.0, 10.0, 11.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let config = PiPNNConfig {
            c_max: 1,
            ..splitting_config()
        };
        let parent = PendingPartition {
            point_ids: vec![0, 1, 2, 3],
            level: 0,
            seed: 1290,
        };
        let buffers = StripeBufferPool::new((), 0, None);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let actual = pool
            .install(|| split_partition::<_, L2, _>(ARCH, data, &config, parent, &buffers))
            .unwrap();

        assert!(actual.leaves.is_empty());
        assert!(actual.pending.iter().all(|child| child.level == 1));
        let ids: Vec<_> = actual
            .pending
            .into_iter()
            .map(|child| child.point_ids)
            .collect();
        assert_eq!(
            sorted_leaf_memberships(&ids),
            [vec![0, 1], vec![0, 1], vec![2, 3], vec![2, 3]]
        );
    }

    #[rstest]
    #[case::below_limit(4)]
    #[case::at_limit(3)]
    fn data_within_cmax_forms_one_complete_leaf_per_replica(#[case] c_max: usize) {
        let values = [0.0_f32, 1.0, 10.0];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let config = PiPNNConfig {
            c_max,
            replicas: 2,
            ..splitting_config()
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let actual = pool
            .install(|| partition::<_, L2, _>(ARCH, data, &config))
            .unwrap();

        assert_eq!(
            sorted_leaf_memberships(&actual),
            [vec![0, 1, 2], vec![0, 1, 2]]
        );
    }

    #[test]
    fn recursive_partitioning_preserves_leaf_membership_across_worker_counts() {
        let point_count = 129;
        let values = test_support::dense_points(point_count, 17, 1290);
        let data = MatrixView::try_from(values.as_slice(), point_count, 17).unwrap();
        let config = PiPNNConfig {
            c_max: 16,
            c_min: 4,
            p_samp: 0.2,
            fanout: vec![2, 2],
            replicas: 2,
            ..splitting_config()
        };
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();

        let first = serial
            .install(|| partition::<_, L2, _>(ARCH, data, &config))
            .unwrap();
        let second = parallel
            .install(|| partition::<_, L2, _>(ARCH, data, &config))
            .unwrap();

        assert_eq!(
            sorted_leaf_memberships(&first),
            sorted_leaf_memberships(&second)
        );
        let mut ids: Vec<_> = first.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids, (0..point_count as u32).collect::<Vec<_>>());
        for leaf in first {
            assert!(
                !leaf.is_empty() && leaf.len() <= config.c_max,
                "leaf {leaf:?}"
            );
            assert_eq!(
                leaf.iter().copied().collect::<HashSet<_>>().len(),
                leaf.len(),
                "duplicate IDs in leaf {leaf:?}"
            );
        }
    }

    #[test]
    fn a_second_replica_adds_new_leaf_memberships() {
        let point_count = 33;
        let values = test_support::dense_points(point_count, 17, 1290);
        let data = MatrixView::try_from(values.as_slice(), point_count, 17).unwrap();
        let config = PiPNNConfig {
            c_max: 8,
            c_min: 1,
            p_samp: 0.25,
            fanout: vec![1],
            ..splitting_config()
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let first = pool
            .install(|| partition::<_, L2, _>(ARCH, data, &config))
            .unwrap();

        let combined = pool
            .install(|| {
                partition::<_, L2, _>(
                    ARCH,
                    data,
                    &PiPNNConfig {
                        replicas: 2,
                        ..config
                    },
                )
            })
            .unwrap();

        let first = sorted_leaf_memberships(&first);
        let combined = sorted_leaf_memberships(&combined);
        assert!(first.iter().all(|leaf| combined.contains(leaf)));
        // This fixed cloud and partial sample produce different memberships in
        // the second pass. Repeating the first seed would only duplicate leaves.
        assert!(combined.iter().any(|leaf| !first.contains(leaf)));
        // Fanout one gives every point exactly one membership per replica.
        let mut memberships = vec![0; point_count];
        for leaf in &combined {
            assert!(
                !leaf.is_empty() && leaf.len() <= config.c_max,
                "leaf {leaf:?}"
            );
            assert_eq!(leaf.iter().collect::<HashSet<_>>().len(), leaf.len());
            for &id in leaf {
                memberships[id as usize] += 1;
            }
        }
        assert_eq!(memberships, vec![2; point_count]);
    }

    #[rstest]
    #[case::one_unrankable([0.0, 3.0, f32::NAN], vec![vec![0], vec![1]])]
    #[case::all_unrankable([f32::NAN; 3], vec![])]
    fn unrankable_points_do_not_form_child_leaves(
        #[case] values: [f32; 3],
        #[case] expected: Vec<Vec<u32>>,
    ) {
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let config = PiPNNConfig {
            c_max: 1,
            fanout: vec![1],
            ..splitting_config()
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let actual = pool
            .install(|| partition::<_, L2, _>(ARCH, data, &config))
            .unwrap();

        assert_eq!(
            sorted_leaf_memberships(&actual),
            sorted_leaf_memberships(&expected)
        );
    }

    #[test]
    fn indistinguishable_points_report_the_oversized_cluster_when_splitting_stalls() {
        let values = [1.0_f32; 5];
        let data = MatrixView::try_from(&values[..], 5, 1).unwrap();
        let config = PiPNNConfig {
            fanout: vec![1],
            ..splitting_config()
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let error = pool
            .install(|| partition::<_, L2, _>(ARCH, data, &config))
            .unwrap_err();

        assert_eq!(
            error.downcast_ref::<PartitionError>().unwrap(),
            &PartitionError::IterationLimit {
                size: 5,
                level: MAX_PARTITION_ITERATIONS,
                limit: MAX_PARTITION_ITERATIONS,
            }
        );
    }

    #[rstest]
    #[case::empty(vec![], 2, 4, vec![])]
    #[case::large_leaves_unchanged(vec![vec![1, 4], vec![0, 3, 6, 8]], 2, 4,
        vec![vec![1, 4], vec![0, 3, 6, 8]])]
    #[case::overlapping_small_leaves(vec![vec![4], vec![1], vec![4], vec![2]], 3, 4,
        vec![vec![1, 2, 4]])]
    #[case::flush_before_exceeding_maximum(vec![vec![4, 5], vec![0, 1]], 3, 3,
        vec![vec![4, 5], vec![0, 1]])]
    #[case::remainder_fits_after_deduplication(vec![vec![1, 3, 5], vec![3], vec![6]], 3, 4,
        vec![vec![1, 3, 5, 6]])]
    #[case::remainder_does_not_fit(vec![vec![1, 3, 5], vec![6], vec![8]], 3, 4,
        vec![vec![1, 3, 5], vec![6, 8]])]
    #[case::only_an_undersized_remainder(vec![vec![7], vec![2]], 3, 4, vec![vec![2, 7]])]
    fn small_leaf_merging_deduplicates_ids_without_exceeding_cmax(
        #[case] leaves: Vec<Vec<u32>>,
        #[case] c_min: usize,
        #[case] c_max: usize,
        #[case] expected: Vec<Vec<u32>>,
    ) {
        let actual = merge_undersized_leaves(leaves, c_min, c_max);

        assert_eq!(
            sorted_leaf_memberships(&actual),
            sorted_leaf_memberships(&expected)
        );
    }

    #[test]
    fn an_overflowing_buffer_area_identifies_the_failed_shape() {
        let error = checked_area("point stripe", usize::MAX, 2).unwrap_err();

        assert_eq!(
            error.downcast_ref::<PartitionError>().unwrap(),
            &PartitionError::ShapeOverflow {
                buffer: "point stripe",
                rows: usize::MAX,
                cols: 2,
            }
        );
    }
}
