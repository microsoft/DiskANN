/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Randomized ball carving: recursive, overlapping partitioning of the dataset
//! into leaves of at most `c_max` points.
//!
//! One split samples leaders from a cluster and assigns each point to its
//! `fanout` nearest leaders. The points of one leader form a child cluster, so
//! children overlap when `fanout > 1`. A child with at most `c_max` points is a
//! leaf; a larger child is split again at the next level. `fanout[level]` gives
//! the fanout of each level, and levels after the schedule use fanout 1. Each
//! replica repeats the process with a different seed.
//!
//! The partition kernel computes point-to-leader distances and selects the
//! nearest leaders. This module samples leaders, gathers vectors, and groups the
//! assignments into children.
//!
//! Buffer sizes use plain multiplication. The IDs of one split are distinct
//! dataset rows, so a gathered `rows x dimensions` buffer is not larger than the
//! dataset. An assignment buffer holds `points x fanout` IDs. With at most
//! `u32::MAX` points and `LEADER_CAP` leaders, this count fits in a 64-bit
//! `usize`.

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

/// A cluster with more than `c_max` points that waits for its split.
struct PendingPartition {
    point_ids: Vec<u32>,
    /// Position in the fanout schedule.
    level: usize,
    /// Seed of the parent split. The next split mixes it with the cluster size.
    seed: u64,
}

/// The result of one split: leaves, and clusters for the next level.
struct PartitionSplit {
    pending: Vec<PendingPartition>,
    leaves: Vec<Vec<u32>>,
}

/// Scratch for one assignment stripe: the gathered `f32` points and the kernel
/// workspace.
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
        // Keep the largest allocation across leases. `assign_point_stripe` sets
        // the active prefix before each read.
    }
}

/// Stripe scratch shared by all splits of a build.
///
/// `ObjectPool` locks only to give or take back a lease. Numerical work holds the
/// lease, not the pool lock.
type StripeBufferPool = ObjectPool<StripeBuffers>;

/// Build the leaves of all replicas.
///
/// A point whose distances to all leaders of a split are not rankable, for
/// example a NaN vector, joins no child of that split.
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
    for replica in 0..config.replicas.get() {
        let seed = replica_seed(replica);
        let mut replica_leaves =
            partition_replica::<A, M, T>(arch, data, config, seed, &stripe_buffers)?;
        leaves.append(&mut replica_leaves);
    }
    Ok(leaves)
}

/// Split one replica, one level at a time, until every cluster is a leaf.
///
/// The loop ends. The fanout schedule is finite, and each split sends its
/// clusters one level deeper. After the schedule, fanout is 1, so each point
/// joins one child. A child is then smaller than its parent, or it holds every
/// point and [`split_partition`] cuts it into leaves.
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
    let c_max = config.c_max.get();
    if data.nrows() <= c_max {
        return Ok(vec![initial_point_ids]);
    }

    let mut leaves = Vec::new();
    let mut pending = vec![PendingPartition {
        point_ids: initial_point_ids,
        level: 0,
        seed,
    }];
    while !pending.is_empty() {
        // Indexed parallel collection keeps the parent order.
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
    Ok(leaves)
}

/// Split one cluster into leaves and clusters for the next level.
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
    let fanout = config
        .fanout
        .get(partition.level)
        .map_or(1, |fanout| fanout.get());
    let c_max = config.c_max.get();
    let leader_count = sampled_leader_count(partition.point_ids.len(), config.p_samp);
    if fanout > 1 && fanout >= leader_count {
        // Every point would join every leader, and each child would repeat this
        // cluster. Split the cluster at the next level instead.
        return Ok(PartitionSplit {
            pending: vec![PendingPartition {
                point_ids: partition.point_ids,
                level: partition.level + 1,
                seed: split_seed,
            }],
            leaves: Vec::new(),
        });
    }

    let leaders = sample_leaders(&partition.point_ids, leader_count, split_seed);
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
        if cluster.len() <= c_max {
            leaves.push(cluster);
        } else if fanout == 1 && cluster.len() == partition.point_ids.len() {
            // Every point chose this leader, so the split made no progress. This
            // happens when all sampled leaders are copies of one vector: each
            // point is equally near to all of them. Another split can stall the
            // same way, so cut the cluster into equal leaves in point order. The
            // cut groups points arbitrarily, which can lower recall but keeps
            // the graph valid.
            let leaf_len = cluster.len().div_ceil(cluster.len().div_ceil(c_max));
            leaves.extend(cluster.chunks(leaf_len).map(<[u32]>::to_vec));
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

/// Sample `count` distinct leaders from the points of a cluster.
fn sample_leaders(points: &[u32], count: usize, seed: u64) -> Vec<u32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    points.choose_multiple(&mut rng, count).copied().collect()
}

/// Return the number of leaders for a cluster of `points` points.
///
/// A split needs two leaders. `LEADER_CAP` bounds the cost of one split.
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

/// Assign each point to its `fanout` nearest leaders and group the points by
/// leader.
///
/// Stripes of points run in parallel. Each child keeps the input order of its
/// points, so the next split samples the same leaders for any thread count.
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

    let mut assignments = vec![0u32; point_ids.len() * fanout];
    let stripe_points = assignment_stripe_point_count(leader_count);
    // This runs in the pool of the build context (see `build_graph`). A lease
    // for each stripe takes one lock, which costs far less than the GEMM and
    // ranking of the stripe.
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

/// Gather one stripe of points as `f32` and write the leader columns of the
/// `fanout` nearest leaders of each point to `assignments`.
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
    // The buffer keeps its largest length. This stripe uses a prefix of it.
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

/// Group the points by assigned leader. Unassigned slots join no child.
///
/// Both paths keep the input order of the points inside each child, because the
/// next split samples its leaders by position.
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
    // Scatter each stripe of points on its own. Indexed collection keeps the
    // stripe order.
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

    // Join the stripe parts of each child in stripe order.
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

/// Return the number of points in one assignment stripe.
///
/// `LEADER_CAP` keeps this count at 131 points or more.
fn assignment_stripe_point_count(leader_count: usize) -> usize {
    (ASSIGNMENT_CACHE_TARGET_BYTES / (leader_count * size_of::<f32>()))
        .min(MAX_ASSIGNMENT_STRIPE_POINTS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::{
        L2,
        test_support::{self, dense_points, nz, sorted_members_per_row, thread_pool},
    };
    use diskann_vector::distance::Metric;
    use diskann_wide::ARCH;
    use std::collections::HashSet;

    // Neither leaf order nor point order within a leaf is part of the result.
    // Preserve repeated leaves and IDs so comparison still detects duplicates.
    fn sorted_leaf_memberships(leaves: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut leaves = sorted_members_per_row(leaves);
        leaves.sort_unstable();
        leaves
    }

    fn splitting_config() -> PiPNNConfig {
        PiPNNConfig {
            c_max: nz(2),
            p_samp: 1.0,
            fanout: vec![nz(2)],
            leaf_k: nz(1),
            replicas: nz(1),
        }
    }

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

            let count = sampled_leader_count(ids.len(), fraction);
            let mut leaders = sample_leaders(&ids, count, 1290);

            assert_eq!(leaders.len(), expected_count, "{case}");
            let distinct: HashSet<_> = leaders.iter().collect();
            assert_eq!(distinct.len(), expected_count, "{case}");
            assert!(leaders.iter().all(|id| ids.contains(id)), "{case}");
            let mut repeated = sample_leaders(&ids, count, 1290);
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
    fn stripes_assign_each_point_to_its_nearest_leaders() {
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
    fn reused_stripe_buffers_do_not_carry_points_or_widths() {
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
    fn scatter_groups_points_by_leader_and_drops_unassigned_slots() {
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
            c_max: nz(1),
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
    fn a_fanout_one_split_without_progress_cuts_equal_leaves() {
        // Copies of one vector tie for every leader, so all points choose the
        // same leader and the child equals its parent.
        let values = [1.0_f32; 5];
        let data = MatrixView::try_from(&values[..], 5, 1).unwrap();
        let config = PiPNNConfig {
            fanout: vec![nz(1)],
            ..splitting_config()
        };
        let parent = PendingPartition {
            point_ids: (0..5).collect(),
            level: 0,
            seed: 1290,
        };

        let actual = split::<L2>(data, &config, parent);

        assert_eq!(
            sorted_leaf_memberships(&actual.leaves),
            [vec![0, 1], vec![2, 3], vec![4]]
        );
        assert!(actual.pending.is_empty());
    }

    #[test]
    fn a_child_that_repeats_its_parent_at_fanout_two_is_split_again() {
        // Points at x=0, x=10 and x=4 are all leaders. Each point joins itself and
        // x=4, so the child of x=4 holds all points. With fanout above one this
        // is overlap, not a stalled split.
        let values = [0.0_f32, 10.0, 4.0];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let parent = PendingPartition {
            point_ids: vec![0, 1, 2],
            level: 0,
            seed: 1290,
        };

        let actual = split::<L2>(data, &splitting_config(), parent);

        assert_eq!(
            sorted_leaf_memberships(&actual.leaves),
            [vec![0, 2], vec![1]]
        );
        let [child] = &actual.pending[..] else {
            panic!("expected one pending cluster");
        };
        assert_eq!(
            (child.point_ids.as_slice(), child.level),
            (&[0, 1, 2][..], 1)
        );
    }

    #[test]
    fn a_level_without_more_leaders_than_its_fanout_is_skipped() {
        // Four points at p_samp 0.01 give two leaders. Fanout 2 would put every
        // point into both children, so the cluster moves to the next level.
        let values = [0.0_f32, 1.0, 10.0, 11.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let config = PiPNNConfig {
            p_samp: 0.01,
            ..splitting_config()
        };
        let parent = PendingPartition {
            point_ids: vec![0, 1, 2, 3],
            level: 0,
            seed: 1290,
        };

        let actual = split::<L2>(data, &config, parent);

        assert!(actual.leaves.is_empty());
        let [child] = &actual.pending[..] else {
            panic!("expected one pending cluster");
        };
        assert_eq!(
            (child.point_ids.as_slice(), child.level),
            (&[0, 1, 2, 3][..], 1)
        );
    }

    #[test]
    fn data_within_cmax_forms_one_complete_leaf_per_replica() {
        let values = [0.0_f32, 1.0, 10.0];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let pool = thread_pool(1);

        for c_max in [3, 4] {
            let config = PiPNNConfig {
                c_max: nz(c_max),
                replicas: nz(2),
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
            c_max: nz(16),
            p_samp: 0.2,
            fanout: vec![nz(2), nz(2)],
            replicas: nz(2),
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
                !leaf.is_empty() && leaf.len() <= config.c_max.get(),
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
            c_max: nz(8),
            p_samp: 0.25,
            fanout: vec![nz(1)],
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
                        replicas: nz(2),
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
                !leaf.is_empty() && leaf.len() <= config.c_max.get(),
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
    fn unrankable_points_join_no_leaf() {
        let config = PiPNNConfig {
            c_max: nz(1),
            fanout: vec![nz(1)],
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
}
