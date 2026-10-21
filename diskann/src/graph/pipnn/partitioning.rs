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
//! The recursion stops at `max_depth` levels. A cluster that is still larger
//! than `c_max` at that level is cut into equal leaves in point order, and the
//! build logs a warning with the number of points it cut. A cluster can stay
//! large when its points are copies of one vector, or when, under inner
//! product, one leader with a large norm wins every point. These leaves group
//! points arbitrarily, which can lower recall.
//!
//! The partition kernel computes point-to-leader distances and selects the
//! nearest leaders. This module samples leaders, gathers vectors, and groups the
//! assignments into children.
//!
//! A split assigns its points in parallel stripes. An assignment stripe is a
//! batch of up to `ASSIGNMENT_STRIPE_POINTS` points that one task gathers as
//! `f32` and ranks against all leaders of the split.
//!
//! Buffer sizes use plain multiplication. The IDs of one split are distinct
//! dataset rows, so a gathered `rows x dimensions` buffer is not larger than the
//! dataset. An assignment buffer holds `points x fanout` IDs. The fanout of a
//! split is below its leader count, which is at most its point count, so with at
//! most `u32::MAX` points this count fits in a 64-bit `usize`.

use crate::{ANNResult, utils::VectorRepr};
use diskann_utils::{
    object_pool::{AsPooled, ObjectPool},
    views::{Matrix, MatrixView, MutMatrixView},
};
use rand::{Rng, SeedableRng, prelude::IndexedRandom, rngs::StdRng};
use rayon::prelude::*;

use super::{
    PiPNNConfig,
    conversion::gather_as_f32,
    partition_kernel::{PartitionKernelWorkspace, assign_leaders},
    partition_metric::PartitionMetric,
    simd::Simd,
    topk::UNASSIGNED,
};

/// Number of points in one assignment stripe. Short stripes repeat the fixed
/// cost of a kernel call, and tall stripes overflow the last-level cache once
/// every worker holds one. The height does not follow the detected cache size,
/// because a virtual machine reports a last-level cache that other tenants
/// share.
const ASSIGNMENT_STRIPE_POINTS: usize = 256;
/// Clusters with at least this many points group their points by leader in
/// parallel. Such clusters occur at the first levels, where few splits run at
/// the same time, so a serial pass would leave most workers idle. Smaller
/// clusters group serially, because the parallel pass costs more than it saves.
const PARALLEL_GROUPING_MIN_POINTS: usize = 100_000;

/// A cluster with more than `c_max` points that waits for its split.
struct Cluster {
    point_ids: Vec<u32>,
    /// Position in the fanout schedule.
    level: usize,
    /// Seed of the random stream of this cluster's split. The parent split
    /// draws it, so every split samples its leaders from its own stream.
    seed: u64,
}

/// The result of one split: leaves, and clusters for the next level.
struct ClusterSplit {
    next_level: Vec<Cluster>,
    leaves: Vec<Vec<u32>>,
    /// Number of points that the split cut into leaves at `max_depth`.
    cut_points: usize,
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
    dimensions: usize,
    config: &PiPNNConfig,
) -> ANNResult<Vec<Vec<u32>>>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let mut leaves = Vec::new();
    let mut cut_points = 0;
    let stripe_buffers = StripeBufferPool::new((), 0, None);
    // Replica `r` uses seed `base + r`. `StdRng::seed_from_u64` gives
    // independent streams for adjacent seeds.
    let base_seed = config.seed.unwrap_or_else(rand::random);
    for replica in 0..config.replicas.get() {
        let seed = base_seed.wrapping_add(replica as u64);
        let (mut replica_leaves, replica_cut_points) =
            partition_replica::<A, M, T>(arch, data, dimensions, config, seed, &stripe_buffers)?;
        leaves.append(&mut replica_leaves);
        cut_points += replica_cut_points;
    }
    if cut_points > 0 {
        tracing::warn!(
            cut_points,
            "PiPNN cut {cut_points} points into leaves in point order because their clusters \
             were still larger than c_max at max_depth; these leaves group points arbitrarily \
             and can lower recall"
        );
    }
    Ok(leaves)
}

/// Split one replica, one level at a time, until every cluster is a leaf.
///
/// The loop ends: each split sends its clusters one level deeper, and a
/// cluster at `max_depth` becomes leaves.
fn partition_replica<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    dimensions: usize,
    config: &PiPNNConfig,
    seed: u64,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<(Vec<Vec<u32>>, usize)>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let initial_point_ids = (0..data.nrows() as u32).collect();
    let c_max = config.c_max.get();
    if data.nrows() <= c_max {
        return Ok((vec![initial_point_ids], 0));
    }

    let mut leaves = Vec::new();
    let mut cut_points = 0;
    let mut pending = vec![Cluster {
        point_ids: initial_point_ids,
        level: 0,
        seed,
    }];
    while !pending.is_empty() {
        // Indexed parallel collection keeps the parent order.
        let splits: ANNResult<Vec<_>> = pending
            .into_par_iter()
            .map(|cluster| {
                split_cluster::<A, M, T>(arch, data, dimensions, config, cluster, stripe_buffers)
            })
            .collect();

        let mut next_level = Vec::new();
        for mut split in splits? {
            next_level.append(&mut split.next_level);
            leaves.append(&mut split.leaves);
            cut_points += split.cut_points;
        }
        pending = next_level;
    }
    Ok((leaves, cut_points))
}

/// Split one cluster into leaves and clusters for the next level.
fn split_cluster<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    dimensions: usize,
    config: &PiPNNConfig,
    cluster: Cluster,
    stripe_buffers: &StripeBufferPool,
) -> ANNResult<ClusterSplit>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr + Send + Sync,
{
    let c_max = config.c_max.get();
    // The recursion stops at `max_depth`. A cluster that reaches it is cut into
    // equal leaves in point order.
    if cluster.level >= config.max_depth.get() {
        let leaf_len = cluster
            .point_ids
            .len()
            .div_ceil(cluster.point_ids.len().div_ceil(c_max));
        let leaves = cluster
            .point_ids
            .chunks(leaf_len)
            .map(<[u32]>::to_vec)
            .collect();
        return Ok(ClusterSplit {
            next_level: Vec::new(),
            leaves,
            cut_points: cluster.point_ids.len(),
        });
    }

    let mut rng = StdRng::seed_from_u64(cluster.seed);
    let fanout = config
        .fanout
        .get(cluster.level)
        .map_or(1, |fanout| fanout.get());
    let leader_count = sampled_leader_count(
        cluster.point_ids.len(),
        config.p_samp,
        config.max_leaders.get(),
    );
    if fanout > 1 && fanout >= leader_count {
        // Every point would join every leader, and each child would repeat this
        // cluster. Split the cluster at the next level instead.
        return Ok(ClusterSplit {
            next_level: vec![Cluster {
                point_ids: cluster.point_ids,
                level: cluster.level + 1,
                seed: rng.random(),
            }],
            leaves: Vec::new(),
            cut_points: 0,
        });
    }

    let leaders = sample_leaders(&cluster.point_ids, leader_count, &mut rng);
    let children = assign_to_leaders::<A, M, T>(
        arch,
        data,
        dimensions,
        &cluster.point_ids,
        &leaders,
        fanout,
        stripe_buffers,
    )?;

    let mut next_level = Vec::new();
    let mut leaves = Vec::new();
    for child in children {
        if child.is_empty() {
            continue;
        }
        if child.len() <= c_max {
            leaves.push(child);
        } else {
            next_level.push(Cluster {
                point_ids: child,
                level: cluster.level + 1,
                seed: rng.random(),
            });
        }
    }
    Ok(ClusterSplit {
        next_level,
        leaves,
        cut_points: 0,
    })
}

/// Sample `count` distinct leaders from the points of a cluster.
fn sample_leaders(points: &[u32], count: usize, rng: &mut StdRng) -> Vec<u32> {
    points.choose_multiple(rng, count).copied().collect()
}

/// Return the number of leaders for a cluster of `points` points.
///
/// A split needs two leaders. `max_leaders` bounds the cost of one split.
fn sampled_leader_count(points: usize, sampling_fraction: f64, max_leaders: usize) -> usize {
    ((points as f64 * sampling_fraction).ceil() as usize).clamp(2, max_leaders)
}

/// Assign each point to its `fanout` nearest leaders and group the points by
/// leader.
///
/// Stripes of points run in parallel. Each child keeps the input order of its
/// points, so the next split samples the same leaders for any thread count.
fn assign_to_leaders<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    dimensions: usize,
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
    let mut leader_values = Matrix::new(0.0f32, leader_ids.len(), dimensions);
    gather_as_f32(data, leader_ids, leader_values.as_mut_view())?;

    let leaders = M::create_leaders(leader_values.as_view());
    let leader_count = M::leader_count(&leaders);

    let mut assignments = Matrix::new(0u32, point_ids.len(), fanout);
    // This runs in the pool of the build context (see `build_graph`). A lease
    // for each stripe takes one lock, which costs far less than the GEMM and
    // ranking of the stripe.
    assignments
        .par_window_iter_mut(ASSIGNMENT_STRIPE_POINTS)
        .zip(point_ids.par_chunks(ASSIGNMENT_STRIPE_POINTS))
        .try_for_each(|(stripe_assignments, stripe_ids)| {
            assign_point_stripe::<A, M, T>(
                arch,
                data,
                stripe_ids,
                dimensions,
                &leaders,
                &mut stripe_buffers.get_ref(()),
                stripe_assignments,
            )
        })?;

    Ok(group_by_leader(
        point_ids,
        assignments.as_view(),
        leader_count,
    ))
}

/// Gather one stripe of points as `f32` and write the leader columns of the
/// nearest leaders of each point to its row of `assignments`, one per column.
#[inline]
fn assign_point_stripe<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    dimensions: usize,
    leaders: &M::Leaders<'_>,
    buffers: &mut StripeBuffers,
    assignments: MutMatrixView<'_, u32>,
) -> ANNResult<()>
where
    A: Simd,
    M: PartitionMetric,
    T: VectorRepr,
{
    let point_count = point_ids.len();
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
    gather_as_f32(data, point_ids, points.as_mut_view())?;
    assign_leaders::<A, M>(
        arch,
        points.as_view(),
        leaders,
        assignments,
        kernel_workspace,
    )
}

/// Group the points by assigned leader. Unassigned slots join no child.
///
/// Both paths keep the input order of the points inside each child, because the
/// next split samples its leaders by position.
fn group_by_leader(
    points: &[u32],
    assignments: MatrixView<'_, u32>,
    leaders: usize,
) -> Vec<Vec<u32>> {
    if points.len() < PARALLEL_GROUPING_MIN_POINTS {
        return group_by_leader_serial(points, assignments, leaders);
    }

    let chunk_points = points.len().div_ceil(rayon::current_num_threads());
    // Group each chunk of points on its own. Indexed collection keeps the chunk
    // order.
    let locals: Vec<_> = points
        .par_chunks(chunk_points)
        .zip(assignments.par_window_iter(chunk_points))
        .map(|(points, assignments)| group_by_leader_serial(points, assignments, leaders))
        .collect();

    let mut sizes = vec![0usize; leaders];
    for local in &locals {
        for (size, cluster) in sizes.iter_mut().zip(local) {
            *size += cluster.len();
        }
    }

    // Join the chunk parts of each child in chunk order.
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

fn group_by_leader_serial(
    points: &[u32],
    assignments: MatrixView<'_, u32>,
    leaders: usize,
) -> Vec<Vec<u32>> {
    let mut sizes = vec![0usize; leaders];
    for &leader in assignments.as_slice() {
        if leader != UNASSIGNED {
            sizes[leader as usize] += 1;
        }
    }
    let mut clusters: Vec<Vec<u32>> = sizes.into_iter().map(Vec::with_capacity).collect();
    for (&point, point_assignments) in points.iter().zip(assignments.row_iter()) {
        for &leader in point_assignments {
            if leader != UNASSIGNED {
                clusters[leader as usize].push(point);
            }
        }
    }
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::{
        L2,
        test_support::{dense_points, nz, thread_pool},
    };
    use diskann_wide::ARCH;
    use std::collections::HashSet;

    /// Sort each leaf and then the list of leaves. Neither order is part of the
    /// result. Repeated leaves and points stay, so a comparison still catches
    /// duplicates.
    fn sorted_leaves(leaves: &[Vec<u32>]) -> Vec<Vec<u32>> {
        let mut leaves: Vec<_> = leaves
            .iter()
            .map(|leaf| {
                let mut points = leaf.clone();
                points.sort_unstable();
                points
            })
            .collect();
        leaves.sort_unstable();
        leaves
    }

    /// Every point is a leader (`p_samp = 1.0`), so leader sampling picks the
    /// same set for any seed.
    fn splitting_config() -> PiPNNConfig {
        PiPNNConfig {
            c_max: nz(2),
            p_samp: 1.0,
            max_leaders: nz(1000),
            fanout: vec![nz(2)],
            max_depth: nz(10),
            leaf_k: nz(1),
            replicas: nz(1),
            seed: Some(1290),
        }
    }

    /// Group `ids` by their `fanout` nearest `leaders` under L2, on `n_threads` threads.
    fn assign_l2(
        data: MatrixView<'_, f32>,
        ids: &[u32],
        leaders: &[u32],
        fanout: usize,
        n_threads: usize,
    ) -> Vec<Vec<u32>> {
        let dimensions = data.ncols();
        let initial_buffers = 0;
        let buffers = StripeBufferPool::new((), initial_buffers, None);
        thread_pool(n_threads)
            .install(|| {
                assign_to_leaders::<_, L2, _>(
                    ARCH, data, dimensions, ids, leaders, fanout, &buffers,
                )
            })
            .unwrap()
    }

    /// Split the cluster of `point_ids` at `level` in one-dimensional L2 points `x`.
    /// One thread uses a fixed sampling seed and an initially empty scratch pool.
    fn split_1d(
        x: &[f32],
        config: &PiPNNConfig,
        point_ids: Vec<u32>,
        level: usize,
    ) -> ClusterSplit {
        let dimensions = 1;
        let n_threads = 1;
        let initial_buffers = 0;
        let data = MatrixView::try_from(x, x.len(), dimensions).unwrap();
        let parent = Cluster {
            point_ids,
            level,
            seed: 1290,
        };
        let buffers = StripeBufferPool::new((), initial_buffers, None);
        thread_pool(n_threads)
            .install(|| split_cluster::<_, L2, _>(ARCH, data, dimensions, config, parent, &buffers))
            .unwrap()
    }

    /// Partition `data` under L2 on `n_threads` threads.
    fn partition_l2(
        data: MatrixView<'_, f32>,
        config: &PiPNNConfig,
        n_threads: usize,
    ) -> Vec<Vec<u32>> {
        let dimensions = data.ncols();
        thread_pool(n_threads)
            .install(|| partition::<_, L2, _>(ARCH, data, dimensions, config))
            .unwrap()
    }

    #[test]
    fn should_partition_successfully() {
        let point_count = 40;
        let dimensions = 2;
        let seed = 1290;
        let n_threads = 1;
        let mut rng = StdRng::seed_from_u64(seed);
        let values: Vec<f32> = (0..point_count * dimensions)
            .map(|_| rng.random())
            .collect();
        let data = MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
        // 40 x 0.25 = 10 leaders. Fanout 3 puts each point in 3 of 10 children
        // of about 12 points, above c_max = 8, so level 1 splits again with fanout 2.
        let config = PiPNNConfig {
            c_max: nz(8),
            p_samp: 0.25,
            fanout: vec![nz(3), nz(2)],
            max_depth: nz(10),
            replicas: nz(1),
            ..splitting_config()
        };

        let leaves = partition_l2(data, &config, n_threads);

        let mut ids: Vec<_> = leaves.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids, (0..point_count as u32).collect::<Vec<_>>());
        for leaf in leaves {
            assert!(!leaf.is_empty(), "leaf {leaf:?}");
            assert!(leaf.len() <= config.c_max.get(), "leaf {leaf:?}");
            assert_eq!(
                leaf.iter().copied().collect::<HashSet<_>>().len(),
                leaf.len(),
                "duplicate IDs in leaf {leaf:?}"
            );
        }
    }

    #[test]
    fn the_leader_count_rounds_up_and_stays_between_two_and_max_leaders() {
        let max_leaders = 5;

        for (points, sampling_fraction, expected) in [
            // 7 × 0.01 = 0.07 rounds up to 1, and a split needs two leaders.
            (7, 0.01, 2),
            // 7 × 0.31 = 2.17 rounds up to 3.
            (7, 0.31, 3),
            // 5 × 1.0 reaches the cap exactly.
            (5, 1.0, 5),
            // 6 × 1.0 is above the cap.
            (6, 1.0, 5),
        ] {
            assert_eq!(
                sampled_leader_count(points, sampling_fraction, max_leaders),
                expected,
                "{points} points, fraction {sampling_fraction}"
            );
        }
    }

    #[test]
    fn points_in_every_stripe_and_the_tail_join_their_nearest_leaders() {
        // Two full stripes of 256 points and a tail of 3.
        let point_count = 2 * ASSIGNMENT_STRIPE_POINTS + 3;
        let dimensions = 1;
        // Point i is at 0, 4 or 10 for i % 3 = 0, 1 or 2, so a stripe that
        // writes at a wrong offset gives a point the result of another kind.
        let x: Vec<_> = (0..point_count)
            .map(|i| [0.0_f32, 4.0, 10.0][i % 3])
            .collect();
        let data = MatrixView::try_from(x.as_slice(), point_count, dimensions).unwrap();
        // The IDs differ from the row positions, so using a row position as an
        // ID fails.
        let ids: Vec<u32> = (0..point_count as u32).rev().collect();
        // Leader points 2, 0 and 1 are at 10, 0 and 4. With fanout 2, each
        // point joins the leader at 4 and the nearer of the leaders at 0 and 10.
        let leaders = [2, 0, 1];
        let fanout = 2;
        let expected = [
            // The leader at 10 gets the points at 10.
            ids.iter().copied().filter(|id| id % 3 == 2).collect(),
            // The leader at 0 gets the points at 0 and at 4.
            ids.iter().copied().filter(|id| id % 3 != 2).collect(),
            // The leader at 4 gets every point.
            ids.clone(),
        ];

        let n_threads = 1;

        let actual = assign_l2(data, &ids, &leaders, fanout, n_threads);

        assert_eq!(actual, expected);
    }

    #[test]
    fn a_stripe_lists_each_points_nearest_leaders_nearest_first() {
        // x[i] is the one-dimensional vector of point i.
        let x = [100.0, 0.0, 8.0, 10.0, 2.0];
        let ids = [4, 2, 0, 3, 1];
        let leader_x = [0.0, 10.0, 100.0];
        let dimensions = 1;
        let fanout = 2;
        let initial_assignment = 0;
        let data = MatrixView::try_from(x.as_slice(), x.len(), dimensions).unwrap();
        let leaders = L2::create_leaders(
            MatrixView::try_from(leader_x.as_slice(), leader_x.len(), dimensions).unwrap(),
        );
        let mut buffers = StripeBuffers::default();
        let mut assignments = Matrix::new(initial_assignment, ids.len(), fanout);
        // Rows follow ids, and each row gives the two nearest indices in leader_x.
        let expected = vec![vec![0, 1], vec![1, 0], vec![2, 1], vec![1, 0], vec![0, 1]];

        assign_point_stripe::<_, L2, _>(
            ARCH,
            data,
            &ids,
            dimensions,
            &leaders,
            &mut buffers,
            assignments.as_mut_view(),
        )
        .unwrap();

        let actual: Vec<_> = assignments.row_iter().map(<[u32]>::to_vec).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn reused_stripe_buffers_do_not_carry_points() {
        // x[i] is the one-dimensional vector of point i.
        let x = [100.0, 0.0, 8.0, 10.0, 2.0];
        let first_ids = [4, 2, 0, 3, 1];
        let ids = [2, 1];
        let leader_x = [0.0, 10.0, 100.0];
        let dimensions = 1;
        let fanout = 2;
        let initial_assignment = 0;
        let data = MatrixView::try_from(x.as_slice(), x.len(), dimensions).unwrap();
        let leaders = L2::create_leaders(
            MatrixView::try_from(leader_x.as_slice(), leader_x.len(), dimensions).unwrap(),
        );
        let mut buffers = StripeBuffers::default();
        let mut first_assignments = Matrix::new(initial_assignment, first_ids.len(), fanout);
        let mut assignments = Matrix::new(initial_assignment, ids.len(), fanout);
        // The first stripe grows the buffers. The second stripe is smaller and
        // reads only a prefix of them. It must gather [8, 0], not retain [2, 8].
        let expected = vec![vec![1, 0], vec![0, 1]];

        assign_point_stripe::<_, L2, _>(
            ARCH,
            data,
            &first_ids,
            dimensions,
            &leaders,
            &mut buffers,
            first_assignments.as_mut_view(),
        )
        .unwrap();
        assign_point_stripe::<_, L2, _>(
            ARCH,
            data,
            &ids,
            dimensions,
            &leaders,
            &mut buffers,
            assignments.as_mut_view(),
        )
        .unwrap();

        let actual: Vec<_> = assignments.row_iter().map(<[u32]>::to_vec).collect();
        assert_eq!(actual, expected, "points {ids:?}");
    }

    #[test]
    fn group_by_leader_lists_each_leaders_points_in_input_order() {
        // IDs are not sorted, so sorting a group would change its expected order.
        let ids = [70, 20, 60, 10, 50, 30, 40];
        let assignment_rows = [[2, 0], [3, 1], [0, 3], [2, 3], [2, 0], [3, 1], [0, 3]];
        let fanout = 2;
        let leader_count = 4;
        let n_threads = 1;
        let assignments =
            MatrixView::try_from(assignment_rows.as_flattened(), ids.len(), fanout).unwrap();
        let expected = [
            vec![70, 60, 50, 40],
            vec![20, 30],
            vec![70, 10, 50],
            vec![20, 60, 10, 30, 40],
        ];

        let actual =
            thread_pool(n_threads).install(|| group_by_leader(&ids, assignments, leader_count));

        assert_eq!(actual, expected);
    }

    #[test]
    fn unassigned_slots_join_no_group() {
        // Point 10 has no assigned leader; the other empty slots must also be skipped.
        let ids = [70, 20, 60, 10, 50, 30, 40];
        let assignment_rows = [
            [2, 0],
            [UNASSIGNED, 1],
            [0, UNASSIGNED],
            [UNASSIGNED, UNASSIGNED],
            [2, 0],
            [UNASSIGNED, 1],
            [0, UNASSIGNED],
        ];
        let fanout = 2;
        let leader_count = 4;
        let n_threads = 1;
        let assignments =
            MatrixView::try_from(assignment_rows.as_flattened(), ids.len(), fanout).unwrap();
        let expected = [vec![70, 60, 50, 40], vec![20, 30], vec![70, 50], vec![]];

        let actual =
            thread_pool(n_threads).install(|| group_by_leader(&ids, assignments, leader_count));

        assert_eq!(actual, expected);
    }

    #[test]
    fn should_split_a_cluster_successfully() {
        // x[i] is the one-dimensional vector of point i.
        let x = [0.0, 1.0, 10.0, 11.0, 13.0];
        let level = 0;
        let config = splitting_config();
        // Point 4 joins leader 3, so that group exceeds c_max while the others fit.
        let expected = [vec![0, 1], vec![0, 1], vec![2, 3], vec![4]];

        let actual = split_1d(&x, &config, vec![0, 1, 2, 3, 4], level);

        assert_eq!(sorted_leaves(&actual.leaves), expected);
        let [child] = actual.next_level.as_slice() else {
            panic!("expected one pending cluster");
        };
        assert_eq!(child.point_ids, [2, 3, 4]);
        assert_eq!(child.level, 1);
        assert_eq!(actual.cut_points, 0);
    }

    #[test]
    fn a_level_past_the_fanout_list_uses_fanout_one() {
        // x[i] is the one-dimensional vector of point i.
        // Level 1 is past the fanout list, so each leader point joins only itself.
        let x = [0.0, 1.0, 10.0, 11.0];
        let level = 1;
        let config = splitting_config();
        let expected = [vec![0], vec![1], vec![2], vec![3]];

        let actual = split_1d(&x, &config, vec![0, 1, 2, 3], level);

        assert_eq!(sorted_leaves(&actual.leaves), expected);
        assert!(actual.next_level.is_empty());
        assert_eq!(actual.cut_points, 0);
    }

    #[test]
    fn a_level_without_more_leaders_than_its_fanout_is_skipped() {
        // x[i] is the one-dimensional vector of point i.
        // Four points at p_samp 0.01 give two leaders. Fanout 2 would put every
        // point into both children, so the cluster moves to the next level.
        let x = [0.0, 1.0, 10.0, 11.0];
        let level = 0;
        let config = PiPNNConfig {
            p_samp: 0.01,
            ..splitting_config()
        };

        let actual = split_1d(&x, &config, vec![0, 1, 2, 3], level);

        assert!(actual.leaves.is_empty());
        let [child] = actual.next_level.as_slice() else {
            panic!("expected one pending cluster");
        };
        assert_eq!(child.point_ids, [0, 1, 2, 3]);
        assert_eq!(child.level, 1);
    }

    #[test]
    fn an_empty_child_is_not_a_leaf() {
        // x[i] is the one-dimensional vector of point i.
        // No point chooses the NaN leader, so its empty group must be dropped.
        let x = [f32::NAN, 0.0, 2.0];
        let level = 0;
        let config = PiPNNConfig {
            fanout: vec![nz(1)],
            ..splitting_config()
        };
        let expected = [vec![1], vec![2]];

        let actual = split_1d(&x, &config, vec![0, 1, 2], level);

        assert_eq!(sorted_leaves(&actual.leaves), expected);
        assert!(actual.next_level.is_empty());
    }

    #[test]
    fn a_cluster_at_max_depth_is_cut_into_equal_leaves() {
        // x[i] is the one-dimensional vector of point i.
        // At max_depth, five points exceed c_max and must be cut in point order.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let config = PiPNNConfig {
            max_depth: nz(1),
            ..splitting_config()
        };
        let level = config.max_depth.get();
        let expected = [vec![0, 1], vec![2, 3], vec![4]];

        let actual = split_1d(&x, &config, vec![0, 1, 2, 3, 4], level);

        assert_eq!(actual.leaves, expected);
        assert!(actual.next_level.is_empty());
        assert_eq!(actual.cut_points, 5);
    }

    #[test]
    fn data_within_cmax_is_one_leaf() {
        // x[i] is the one-dimensional vector of point i.
        // The dataset fits c_max, so it returns as one leaf without splitting.
        let x = [0.0, 1.0, 10.0];
        let dimensions = 1;
        let n_threads = 1;
        let data = MatrixView::try_from(x.as_slice(), x.len(), dimensions).unwrap();

        for c_max in [x.len(), x.len() + 1] {
            let config = PiPNNConfig {
                c_max: nz(c_max),
                ..splitting_config()
            };

            let actual = partition_l2(data, &config, n_threads);

            assert_eq!(sorted_leaves(&actual), [vec![0, 1, 2]], "c_max {c_max}");
        }
    }

    #[test]
    fn replicas_use_different_seeds() {
        // With one seed for every replica, the second replica would repeat the
        // leaves of the first.
        let point_count = 33;
        let dimensions = 17;
        let seed = 1290;
        let n_threads = 1;
        let values = dense_points(point_count, dimensions, seed);
        let data = MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
        let config = PiPNNConfig {
            c_max: nz(8),
            p_samp: 0.25,
            fanout: vec![nz(1)],
            ..splitting_config()
        };
        let sorted_leaves_with = |replicas| {
            let config = PiPNNConfig {
                replicas: nz(replicas),
                ..config.clone()
            };
            let leaves = partition_l2(data, &config, n_threads);
            sorted_leaves(&leaves)
        };

        let one_replica = sorted_leaves_with(1);
        let two_replicas = sorted_leaves_with(2);

        assert!(two_replicas.iter().any(|leaf| !one_replica.contains(leaf)));
    }
}
