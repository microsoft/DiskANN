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
//!
//! Buffer sizes use plain multiplication. Point and leader IDs in one split are
//! distinct dataset rows, so a gathered `rows x dimensions` buffer is no larger
//! than the dataset. An assignment buffer holds `points x fanout` IDs. With at
//! most `u32::MAX` points and `LEADER_CAP` leaders, this count fits in a 64-bit
//! `usize`.

use std::collections::HashSet;

use crate::{ANNResult, utils::VectorRepr};
use diskann_utils::{
    object_pool::{AsPooled, ObjectPool},
    views::{MatrixView, MutMatrixView},
};
use rand::{SeedableRng, prelude::IndexedRandom};
use rayon::prelude::*;

use super::{
    PiPNNConfig,
    conversion::gather_as_f32,
    partition_kernel::{PartitionKernelWorkspace, assign_leaders},
    partition_metric::PartitionMetric,
    simd::Simd,
    topk::UNASSIGNED,
};

/// Seed of the first replica. Fixed seeds make each build reproducible.
const PARTITION_SEED: u64 = 1_000;
/// Seed step between replicas, so each replica samples different leaders.
const REPLICA_SEED_STEP: u64 = 7_919;
/// Maximum number of leaders in one split. The PiPNN paper uses the same hard
/// cap ("typically 1000", Section 4.1). A split costs
/// `points x leaders x dimensions` multiply-adds. Without the cap, the first
/// split of 10M points at `p_samp = 0.005` samples 50,000 leaders and does 50
/// times the work.
const LEADER_CAP: usize = 1_000;
/// Size target of the distance block of one stripe, `points x leaders x 4`
/// bytes. GEMM writes the block and the top-k scan reads it back, so the block
/// must stay in the L2 cache of the core. On an AMD EPYC 7763 (512 KiB L2 per
/// core), this value gave the shortest partition time for 10M points: 128 KiB
/// was about 7% slower and 2 MiB about 5% slower. Current server cores have
/// 512 KiB to 2 MiB of L2 cache.
const ASSIGNMENT_CACHE_TARGET_BYTES: usize = 512 * 1024;
/// Maximum number of points in one stripe. With few leaders, the cache target
/// alone allows very tall stripes. This bound keeps the gathered points of one
/// worker at `1024 x dimensions x 4` bytes or less.
const MAX_ASSIGNMENT_STRIPE_POINTS: usize = 1_024;
/// Clusters with at least this many points scatter in parallel. Such clusters
/// occur at the first levels, where few splits run at the same time. A serial
/// scatter of 10M points at fanout 10 takes 0.59 s, and a parallel scatter
/// takes 0.15 s on 16 threads (AMD EPYC 7763). The serial time does not
/// decrease with more threads, so its share of the build increases on larger
/// machines. Below this size, a serial scatter takes a few milliseconds.
const PARALLEL_SCATTER_MIN_POINTS: usize = 100_000;

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
///
/// The queue empties because the fanout schedule is finite. After the schedule,
/// a split puts each point into at most one child. [`split_partition`] makes sure
/// that such a child is smaller than its parent, so the cluster sizes decrease at
/// each later level.
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
    while !pending.is_empty() {
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
    Ok(merge_undersized_leaves(leaves, config.c_min, config.c_max))
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
    for (&leader, mut cluster) in leaders.iter().zip(clusters) {
        if fanout == 1 && cluster.len() == partition.point_ids.len() {
            // Every point chose this leader, so the split made no progress. Under
            // L2, each leader is nearest to itself. Thus this occurs only when all
            // sampled leaders are copies of one vector. Under cosine, vectors with
            // the same direction also tie. Under inner product, one large leader
            // can be first for every point.
            //
            // Copies of the leader have the same distance to every point, so any
            // grouping of them gives the same leaf neighbors. Put the copies into
            // leaves of equal size. The rest does not contain the leader, so it is
            // smaller than its parent. Each stalled level therefore removes at
            // least one point.
            let copy = bytemuck::cast_slice::<T, u8>(data.row(leader as usize));
            let (copies, rest): (Vec<_>, Vec<_>) = cluster
                .into_iter()
                .partition(|&id| bytemuck::cast_slice::<T, u8>(data.row(id as usize)) == copy);
            let leaf_len = copies.len().div_ceil(copies.len().div_ceil(config.c_max));
            leaves.extend(copies.chunks(leaf_len).map(<[u32]>::to_vec));
            cluster = rest;
        }
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

// This LCG step derives a child seed from the parent seed and the cluster size.
// The multiplier is Knuth's MMIX constant. Wrapping arithmetic gives the same
// mapping in debug and release builds on all supported platforms.
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
    let mut leader_values = vec![0.0f32; leader_ids.len() * dimension_count];
    gather_as_f32(data, leader_ids, &mut leader_values)?;

    let leader_matrix =
        MatrixView::try_from(leader_values.as_slice(), leader_ids.len(), dimension_count)?;
    let leaders = M::create_leaders(leader_matrix);
    let leader_count = M::leader_count(&leaders);

    let fanout = fanout.min(leader_count);
    let mut assignments = vec![0u32; point_ids.len() * fanout];
    let stripe_points = assignment_stripe_point_count(leader_count);
    // `build_graph` runs this operation in the pool from the build context.
    // A buffer lease takes one lock, which costs far less than the GEMM and
    // ranking of one stripe.
    assignments
        .par_chunks_mut(stripe_points * fanout)
        .zip(point_ids.par_chunks(stripe_points))
        .try_for_each(|(stripe_assignments, stripe_ids)| {
            assign_point_stripe::<A, M, T>(
                arch,
                data,
                stripe_ids,
                &leaders,
                fanout,
                &mut stripe_buffers.get_ref(()),
                stripe_assignments,
            )
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
    let point_values_len = point_count * dimensions;
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
    )?;
    gather_as_f32(data, point_ids, points.as_mut_slice())?;
    let output = MutMatrixView::try_from(assignments, point_count, fanout)?;
    assign_leaders::<A, M>(arch, points.as_view(), leaders, output, kernel_workspace)
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

/// Return the number of points in one assignment stripe.
///
/// `LEADER_CAP` keeps this count at 131 points or more.
fn assignment_stripe_point_count(leader_count: usize) -> usize {
    (ASSIGNMENT_CACHE_TARGET_BYTES / (leader_count * size_of::<f32>()))
        .min(MAX_ASSIGNMENT_STRIPE_POINTS)
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::graph::pipnn::{
        L2,
        test_support::{self, dense_points, sorted_members_per_row, thread_pool},
    };
    use diskann_vector::distance::Metric;
    use diskann_wide::ARCH;

    // Neither leaf order nor point order within a leaf is part of the result.
    // Preserve repeated leaves and IDs so comparison still detects duplicates.
    fn sorted_leaf_memberships(leaves: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut leaves = sorted_members_per_row(leaves);
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

    #[expect(clippy::unwrap_used, reason = "an invalid fixture fails the test")]
    fn split<M: PartitionMetric>(
        data: MatrixView<'_, f32>,
        config: &PiPNNConfig,
        parent: PendingPartition,
    ) -> PartitionSplit {
        let buffers = StripeBufferPool::new((), 0, None);
        thread_pool(2)
            .install(|| split_partition::<_, M, _>(ARCH, data, config, parent, &buffers))
            .unwrap()
    }

    #[test]
    fn sampled_leaders_are_a_repeatable_subset_without_replacement() {
        // (points, sampling fraction, expected leaders)
        for (point_count, fraction, expected_count) in [
            (7, 0.01, 2), // A split needs two leaders.
            (7, 0.31, 3), // 2.17 rounds up.
            (7, 1.0, 7),
            (999, 1.0, 999),
            (1000, 1.0, LEADER_CAP),
            (1001, 1.0, LEADER_CAP),
        ] {
            let case = format!("{point_count} points, fraction {fraction}");
            let ids: Vec<_> = (0..point_count).map(|id| 10 + 3 * id).collect();

            let mut leaders = sample_leaders(&ids, fraction, 1290);

            assert_eq!(leaders.len(), expected_count, "{case}");
            let distinct: HashSet<_> = leaders.iter().collect();
            assert_eq!(distinct.len(), expected_count, "{case}");
            assert!(leaders.iter().all(|id| ids.contains(id)), "{case}");
            let mut repeated = sample_leaders(&ids, fraction, 1290);
            leaders.sort_unstable();
            repeated.sort_unstable();
            assert_eq!(leaders, repeated, "{case}");
        }
    }

    #[test]
    fn points_join_their_nearest_leaders() {
        // Leader columns are global IDs [2, 0], at x=10 and x=0.
        // The requested points [5, 4, 1] are at x=8, x=2 and x=100.
        let values = [0.0_f32, 100.0, 10.0, -100.0, 2.0, 8.0];
        let data = MatrixView::try_from(&values[..], 6, 1).unwrap();
        let buffers = StripeBufferPool::new((), 0, None);
        let pool = thread_pool(2);

        for (fanout, expected) in [
            (1, [vec![5, 1], vec![4]]),
            (2, [vec![5, 4, 1], vec![5, 4, 1]]),
            // Fanout above the leader count assigns every point to every leader.
            (9, [vec![5, 4, 1], vec![5, 4, 1]]),
        ] {
            let actual = pool
                .install(|| {
                    assign_to_leaders::<_, L2, _>(ARCH, data, &[5, 4, 1], &[2, 0], fanout, &buffers)
                })
                .unwrap();

            assert_eq!(
                sorted_members_per_row(&actual),
                sorted_members_per_row(&expected),
                "fanout {fanout}"
            );
        }
    }

    #[test]
    fn multiple_stripes_and_their_partial_tail_preserve_all_assignments() {
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
        let expected = [
            ids.iter().copied().filter(|id| id % 3 == 2).collect(),
            ids.iter().copied().filter(|id| id % 3 != 2).collect(),
            ids.clone(),
        ];
        let buffers = StripeBufferPool::new((), 0, None);

        for workers in [1, 3] {
            let actual = thread_pool(workers)
                .install(|| {
                    assign_to_leaders::<_, L2, _>(ARCH, data, &ids, &[2, 0, 1], 2, &buffers)
                })
                .unwrap();

            assert_eq!(
                sorted_members_per_row(&actual),
                sorted_members_per_row(&expected),
                "{workers} workers"
            );
        }
    }

    #[test]
    fn assignments_at_production_shape_join_each_point_to_its_nearest_leaders() {
        // 300 leaders give stripes of 436 points, so 3,000 points use several
        // stripes, a partial tail, several workers and full SIMD leader groups.
        // Dense fixtures keep the f32 scores exact, so the check needs no
        // tolerance. It accepts any order of tied leaders.
        let (point_count, dimensions, fanout) = (3000, 32, 3);
        let values = dense_points(point_count, dimensions, 1290);
        let data = MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
        let row = |id: u32| &values[id as usize * dimensions..][..dimensions];
        let ids: Vec<u32> = (0..point_count as u32).rev().collect();
        let leaders: Vec<u32> = (0..point_count as u32).step_by(10).collect();
        let buffers = StripeBufferPool::new((), 0, None);

        let clusters = thread_pool(4)
            .install(|| assign_to_leaders::<_, L2, _>(ARCH, data, &ids, &leaders, fanout, &buffers))
            .unwrap();

        let mut chosen = vec![Vec::new(); point_count];
        for (column, cluster) in clusters.iter().enumerate() {
            // `ids` is descending, and scatter keeps the input order.
            assert!(cluster.is_sorted_by(|a, b| a > b), "leader {column}");
            for &id in cluster {
                chosen[id as usize].push(column);
            }
        }
        for (id, columns) in chosen.iter().enumerate() {
            let distance = |column: usize| {
                test_support::distance(Metric::L2, row(id as u32), row(leaders[column]))
            };
            let farthest_chosen = columns
                .iter()
                .map(|&column| distance(column))
                .fold(f64::MIN, f64::max);
            let nearest_other = (0..leaders.len())
                .filter(|column| !columns.contains(column))
                .map(distance)
                .fold(f64::INFINITY, f64::min);
            assert_eq!(columns.len(), fanout, "point {id}");
            assert!(
                farthest_chosen <= nearest_other,
                "point {id}: chosen {columns:?} at {farthest_chosen}, other at {nearest_other}"
            );
        }
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
                sorted_members_per_row(&actual),
                sorted_members_per_row(&expected),
                "points {ids:?}, fanout={fanout}"
            );
        }
    }

    #[test]
    fn scatter_groups_assigned_points_by_leader_and_omits_unassigned_slots() {
        let pattern = [
            [2, 0],
            [UNASSIGNED, 1],
            [0, UNASSIGNED],
            [UNASSIGNED, UNASSIGNED],
        ];
        let pool = thread_pool(3);

        // Counts around the parallel threshold cover both scatter paths and
        // parallel chunks of unequal length.
        for point_count in [
            7,
            PARALLEL_SCATTER_MIN_POINTS - 1,
            PARALLEL_SCATTER_MIN_POINTS,
            PARALLEL_SCATTER_MIN_POINTS + 1,
        ] {
            let ids: Vec<_> = (0..point_count).map(|i| 1_000_000 - i as u32).collect();
            let assignments: Vec<_> = (0..point_count).flat_map(|i| pattern[i % 4]).collect();
            let expected = [
                ids.iter().copied().step_by(2).collect(),
                ids.iter().copied().skip(1).step_by(4).collect(),
                ids.iter().copied().step_by(4).collect(),
                vec![],
            ];

            let actual = pool.install(|| scatter_assignments(&ids, &assignments, 2, 4));

            assert_eq!(
                sorted_members_per_row(&actual),
                sorted_members_per_row(&expected),
                "{point_count} points"
            );
        }
    }

    #[test]
    fn a_split_uses_the_configured_fanout_then_falls_back_to_one_leader() {
        // Every point is a leader, so membership is fixed regardless of output order.
        let values = [0.0_f32, 1.0, 10.0, 11.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();

        for (level, expected) in [
            (0, vec![vec![0, 1], vec![0, 1], vec![2, 3], vec![2, 3]]),
            (1, vec![vec![0], vec![1], vec![2], vec![3]]),
        ] {
            let parent = PendingPartition {
                point_ids: vec![0, 1, 2, 3],
                level,
                seed: 1290,
            };

            let actual = split::<L2>(data, &splitting_config(), parent);

            assert_eq!(
                sorted_leaf_memberships(&actual.leaves),
                sorted_leaf_memberships(&expected),
                "level {level}"
            );
            assert!(actual.pending.is_empty(), "level {level}");
        }
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

        let actual = split::<L2>(data, &config, parent);

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

    #[test]
    fn a_stalled_split_moves_the_leader_copies_into_leaves() {
        // IDs 0 to 5 are copies of one vector, and ID 6 differs. When both
        // sampled leaders are copies, all points tie and choose the same leader.
        let values = [1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0];
        let data = MatrixView::try_from(&values[..], 7, 1).unwrap();
        let ids: Vec<u32> = (0..7).collect();
        let config = PiPNNConfig {
            c_max: 4,
            p_samp: 0.01,
            fanout: vec![1],
            ..splitting_config()
        };
        // `split_partition` derives its sampling seed from the parent seed and size.
        let seed = (0..)
            .find(|&seed| {
                let leaders = sample_leaders(&ids, config.p_samp, mix_seed(seed, 7));
                leaders.iter().all(|&id| id < 6)
            })
            .unwrap();
        let parent = PendingPartition {
            point_ids: ids,
            level: 0,
            seed,
        };

        let actual = split::<L2>(data, &config, parent);

        // The copies fill two equal leaves. The distinct point is not grouped with them.
        assert_eq!(
            sorted_leaf_memberships(&actual.leaves),
            [vec![0, 1, 2], vec![3, 4, 5], vec![6]]
        );
        assert!(actual.pending.is_empty());
    }

    #[test]
    fn data_within_cmax_forms_one_complete_leaf_per_replica() {
        let values = [0.0_f32, 1.0, 10.0];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let pool = thread_pool(1);

        for c_max in [3, 4] {
            let config = PiPNNConfig {
                c_max,
                replicas: 2,
                ..splitting_config()
            };

            let actual = pool
                .install(|| partition::<_, L2, _>(ARCH, data, &config))
                .unwrap();

            assert_eq!(
                sorted_leaf_memberships(&actual),
                [vec![0, 1, 2], vec![0, 1, 2]],
                "c_max {c_max}"
            );
        }
    }

    #[test]
    fn recursive_partitioning_preserves_leaf_membership_across_worker_counts() {
        let point_count = 129;
        let values = dense_points(point_count, 17, 1290);
        let data = MatrixView::try_from(values.as_slice(), point_count, 17).unwrap();
        let config = PiPNNConfig {
            c_max: 16,
            c_min: 4,
            p_samp: 0.2,
            fanout: vec![2, 2],
            replicas: 2,
            ..splitting_config()
        };

        let first = thread_pool(1)
            .install(|| partition::<_, L2, _>(ARCH, data, &config))
            .unwrap();
        let second = thread_pool(3)
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
        let values = dense_points(point_count, 17, 1290);
        let data = MatrixView::try_from(values.as_slice(), point_count, 17).unwrap();
        let config = PiPNNConfig {
            c_max: 8,
            c_min: 1,
            p_samp: 0.25,
            fanout: vec![1],
            ..splitting_config()
        };
        let pool = thread_pool(2);
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

    #[test]
    fn unrankable_points_do_not_form_child_leaves() {
        let config = PiPNNConfig {
            c_max: 1,
            fanout: vec![1],
            ..splitting_config()
        };
        let pool = thread_pool(1);

        for (values, expected) in [
            ([0.0, 3.0, f32::NAN], vec![vec![0], vec![1]]),
            ([f32::NAN; 3], vec![]),
        ] {
            let data = MatrixView::try_from(&values[..], 3, 1).unwrap();

            let actual = pool
                .install(|| partition::<_, L2, _>(ARCH, data, &config))
                .unwrap();

            assert_eq!(
                sorted_leaf_memberships(&actual),
                sorted_leaf_memberships(&expected),
                "values {values:?}"
            );
        }
    }

    #[test]
    fn small_leaf_merging_deduplicates_ids_without_exceeding_cmax() {
        // (leaves, c_min, c_max, expected leaves)
        for (leaves, c_min, c_max, expected) in [
            (vec![], 2, 4, vec![]),
            (
                vec![vec![1, 4], vec![0, 3, 6, 8]],
                2,
                4,
                vec![vec![1, 4], vec![0, 3, 6, 8]],
            ),
            // Overlapping small leaves merge into one leaf without repeated IDs.
            (
                vec![vec![4], vec![1], vec![4], vec![2]],
                3,
                4,
                vec![vec![1, 2, 4]],
            ),
            // A merge that would exceed c_max flushes the first leaf.
            (
                vec![vec![4, 5], vec![0, 1]],
                3,
                3,
                vec![vec![4, 5], vec![0, 1]],
            ),
            // After duplicate removal, the undersized remainder fits the last leaf.
            (
                vec![vec![1, 3, 5], vec![3], vec![6]],
                3,
                4,
                vec![vec![1, 3, 5, 6]],
            ),
            (
                vec![vec![1, 3, 5], vec![6], vec![8]],
                3,
                4,
                vec![vec![1, 3, 5], vec![6, 8]],
            ),
            (vec![vec![7], vec![2]], 3, 4, vec![vec![2, 7]]),
        ] {
            let case = format!("{leaves:?}, c_min={c_min}, c_max={c_max}");

            let actual = merge_undersized_leaves(leaves, c_min, c_max);

            assert_eq!(
                sorted_leaf_memberships(&actual),
                sorted_leaf_memberships(&expected),
                "{case}"
            );
        }
    }
}
