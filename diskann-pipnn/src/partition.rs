/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Randomized Ball Carving (RBC) partitioning.
//!
//! Recursively partitions the dataset into overlapping clusters:
//! - Sample a fraction of points as leaders
//! - Assign each point to its `fanout` nearest leaders (creating overlap)
//! - Merge undersized clusters
//! - Recurse on oversized clusters

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Maximum recursion depth to prevent stack overflow.
const MAX_DEPTH: usize = 30;

/// A leaf partition containing indices into the original dataset.
#[derive(Debug, Clone)]
pub struct Leaf {
    pub indices: Vec<usize>,
}

/// Configuration for RBC partitioning.
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    pub c_max: usize,
    pub c_min: usize,
    pub p_samp: f64,
    pub fanout: Vec<usize>,
}

/// Compute squared L2 distance between two f32 slices using manual loop
/// (auto-vectorized by the compiler).
#[inline]
fn l2_distance_inline(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        sum += d * d;
    }
    sum
}

/// Fused GEMM + assignment: compute distances to leaders in stripes and immediately
/// extract top-k assignments without materializing the full N x L distance matrix.
/// Peak memory: STRIPE * L * 4 bytes (~64MB) instead of N * L * 4 bytes (~4GB for 1M x 1000).
fn partition_assign(
    data: &[f32],
    ndims: usize,
    points: &[usize],
    leaders: &[usize],
    fanout: usize,
) -> Vec<Vec<usize>> {
    partition_assign_impl(data, ndims, points, leaders, fanout, true)
}

/// Core implementation with control over parallelism strategy.
/// `use_rayon_stripes`: true = many parallel stripes (for top-level with many points),
///                      false = fewer stripes with multi-threaded BLAS (not used currently).
fn partition_assign_impl(
    data: &[f32],
    ndims: usize,
    points: &[usize],
    leaders: &[usize],
    fanout: usize,
    use_rayon_stripes: bool,
) -> Vec<Vec<usize>> {
    let np = points.len();
    let nl = leaders.len();
    let num_assign = fanout.min(nl);

    // Extract leader data (shared, stays in cache).
    let mut l_data = vec![0.0f32; nl * ndims];
    for (i, &idx) in leaders.iter().enumerate() {
        l_data[i * ndims..(i + 1) * ndims]
            .copy_from_slice(&data[idx * ndims..(idx + 1) * ndims]);
    }
    let mut l_norms = vec![0.0f32; nl];
    for i in 0..nl {
        let row = &l_data[i * ndims..(i + 1) * ndims];
        let mut norm = 0.0f32;
        for &v in row { norm += v * v; }
        l_norms[i] = norm;
    }

    // Flat assignments: assignments[i * num_assign .. (i+1) * num_assign]
    let mut assignments = vec![0u32; np * num_assign];

    // Fused parallel stripes: GEMM + distance + top-k in one pass.
    const STRIPE: usize = 16_384;
    assignments
        .par_chunks_mut(STRIPE * num_assign)
        .enumerate()
        .for_each(|(stripe_idx, assign_chunk)| {
            let start = stripe_idx * STRIPE;
            let end = (start + STRIPE).min(np);
            let sn = end - start;
            let stripe_points = &points[start..end];

            let mut p_data = vec![0.0f32; sn * ndims];
            for (i, &idx) in stripe_points.iter().enumerate() {
                p_data[i * ndims..(i + 1) * ndims]
                    .copy_from_slice(&data[idx * ndims..(idx + 1) * ndims]);
            }

            let mut p_norms = vec![0.0f32; sn];
            for i in 0..sn {
                let row = &p_data[i * ndims..(i + 1) * ndims];
                let mut norm = 0.0f32;
                for &v in row { norm += v * v; }
                p_norms[i] = norm;
            }

            let mut dots = vec![0.0f32; sn * nl];
            crate::gemm::sgemm_abt(&p_data, sn, ndims, &l_data, nl, &mut dots);

            let mut buf: Vec<(u32, f32)> = Vec::with_capacity(nl);
            for i in 0..sn {
                let pi = p_norms[i];
                let dot_row = &dots[i * nl..(i + 1) * nl];

                buf.clear();
                for j in 0..nl {
                    let d = (pi + l_norms[j] - 2.0 * dot_row[j]).max(0.0);
                    buf.push((j as u32, d));
                }

                if num_assign < buf.len() {
                    buf.select_nth_unstable_by(num_assign, |a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                let out = &mut assign_chunk[i * num_assign..(i + 1) * num_assign];
                for k in 0..num_assign {
                    out[k] = buf[k].0;
                }
            }
        });

    // Aggregate into per-leader clusters.
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); nl];
    for i in 0..np {
        let row = &assignments[i * num_assign..(i + 1) * num_assign];
        for &li in row {
            clusters[li as usize].push(i);
        }
    }
    clusters
}

/// Force-split a set of indices into chunks of at most c_max, used as fallback.
fn force_split(indices: &[usize], c_max: usize) -> Vec<Leaf> {
    indices
        .chunks(c_max)
        .map(|chunk| Leaf {
            indices: chunk.to_vec(),
        })
        .collect()
}

/// Partition the dataset using Randomized Ball Carving.
///
/// `data` is row-major: npoints_global x ndims.
/// `indices` are the global indices of the points to partition.
pub fn partition(
    data: &[f32],
    ndims: usize,
    indices: &[usize],
    config: &PartitionConfig,
    level: usize,
    rng: &mut impl Rng,
) -> Vec<Leaf> {
    let n = indices.len();

    if n <= config.c_max {
        return vec![Leaf {
            indices: indices.to_vec(),
        }];
    }

    // For clusters at deep recursion levels or only marginally over c_max,
    // force-split is cheaper than doing another full GEMM + assignment.
    if level >= MAX_DEPTH || (level >= 2 && n <= config.c_max * 3) {
        return force_split(indices, config.c_max);
    }

    let fanout = if level < config.fanout.len() {
        config.fanout[level]
    } else {
        1
    };

    // Sample leaders.
    let num_leaders = ((n as f64 * config.p_samp).ceil() as usize)
        .max(2)
        .min(1000)
        .min(n);

    let mut sampled_indices: Vec<usize> = indices.to_vec();
    sampled_indices.shuffle(rng);
    let leaders: Vec<usize> = sampled_indices[..num_leaders].to_vec();

    // Fused GEMM + assignment (avoids materializing full distance matrix).
    let clusters_local = partition_assign(data, ndims, indices, &leaders, fanout);

    // Map local indices back to global.
    let mut clusters: Vec<Vec<usize>> = clusters_local
        .into_iter()
        .map(|local_cluster| {
            local_cluster.into_iter().map(|li| indices[li]).collect()
        })
        .collect();

    // Merge undersized clusters.
    let mut merged_clusters: Vec<Vec<usize>> = Vec::new();
    let mut small_clusters: Vec<Vec<usize>> = Vec::new();

    for cluster in clusters.drain(..) {
        if cluster.len() < config.c_min && !cluster.is_empty() {
            small_clusters.push(cluster);
        } else if !cluster.is_empty() {
            merged_clusters.push(cluster);
        }
    }

    if !small_clusters.is_empty() && !merged_clusters.is_empty() {
        // Merge small clusters into the nearest large cluster (by index, simple heuristic).
        for small in small_clusters {
            // Just append to the smallest existing cluster.
            let min_idx = merged_clusters
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| c.len())
                .map(|(i, _)| i)
                .unwrap_or(0);
            merged_clusters[min_idx].extend(small);
        }
    } else if merged_clusters.is_empty() {
        merged_clusters = small_clusters;
    }

    if merged_clusters.len() == 1 && merged_clusters[0].len() > config.c_max {
        return force_split(&merged_clusters[0], config.c_max);
    }

    let mut leaves = Vec::new();
    for cluster in merged_clusters {
        if cluster.len() <= config.c_max {
            leaves.push(Leaf { indices: cluster });
        } else {
            let sub_seed: u64 = rng.random();
            let mut sub_rng = rand::rngs::StdRng::seed_from_u64(sub_seed);
            let sub_leaves = partition(data, ndims, &cluster, config, level + 1, &mut sub_rng);
            leaves.extend(sub_leaves);
        }
    }

    leaves
}

/// Partition using parallelism at the top level.
/// Prints timing breakdown for the top-level operations.
pub fn parallel_partition(
    data: &[f32],
    ndims: usize,
    indices: &[usize],
    config: &PartitionConfig,
    seed: u64,
) -> Vec<Leaf> {
    let n = indices.len();

    if n <= config.c_max {
        return vec![Leaf {
            indices: indices.to_vec(),
        }];
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let fanout = if !config.fanout.is_empty() {
        config.fanout[0]
    } else {
        3
    };

    // Sample leaders.
    let num_leaders = ((n as f64 * config.p_samp).ceil() as usize)
        .max(2)
        .min(1000)
        .min(n);

    let mut sampled_indices: Vec<usize> = indices.to_vec();
    sampled_indices.shuffle(&mut rng);
    let leaders: Vec<usize> = sampled_indices[..num_leaders].to_vec();

    // Fused GEMM + assignment.
    let t0 = std::time::Instant::now();
    let clusters_local = partition_assign(data, ndims, indices, &leaders, fanout);
    let assign_time = t0.elapsed();

    let t1 = std::time::Instant::now();
    let mut clusters: Vec<Vec<usize>> = clusters_local
        .into_iter()
        .map(|local_cluster| {
            local_cluster.into_iter().map(|li| indices[li]).collect()
        })
        .collect();
    let map_time = t1.elapsed();

    eprintln!("    top-level: assign {:.3}s, map {:.3}s, {} leaders, fanout {}",
        assign_time.as_secs_f64(), map_time.as_secs_f64(), num_leaders, fanout);

    // Merge undersized clusters.
    let mut merged_clusters: Vec<Vec<usize>> = Vec::new();
    let mut small_clusters: Vec<Vec<usize>> = Vec::new();

    for cluster in clusters.drain(..) {
        if cluster.len() < config.c_min && !cluster.is_empty() {
            small_clusters.push(cluster);
        } else if !cluster.is_empty() {
            merged_clusters.push(cluster);
        }
    }

    if !small_clusters.is_empty() && !merged_clusters.is_empty() {
        for small in small_clusters {
            let min_idx = merged_clusters
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| c.len())
                .map(|(i, _)| i)
                .unwrap_or(0);
            merged_clusters[min_idx].extend(small);
        }
    } else if merged_clusters.is_empty() {
        merged_clusters = small_clusters;
    }

    let need_recurse = merged_clusters.iter().filter(|c| c.len() > config.c_max).count();
    let total_in_recurse: usize = merged_clusters.iter().filter(|c| c.len() > config.c_max).map(|c| c.len()).sum();
    eprintln!("    merge: {} clusters, {} need recursion ({} pts)",
        merged_clusters.len(), need_recurse, total_in_recurse);

    // Generate sub-seeds for parallel recursion.
    let sub_seeds: Vec<u64> = (0..merged_clusters.len())
        .map(|_| rng.random())
        .collect();

    // Recurse in parallel.
    let t2 = std::time::Instant::now();
    let results: Vec<Vec<Leaf>> = merged_clusters
        .par_iter()
        .zip(sub_seeds.par_iter())
        .map(|(cluster, sub_seed)| {
            if cluster.len() <= config.c_max {
                vec![Leaf {
                    indices: cluster.clone(),
                }]
            } else {
                let mut sub_rng = rand::rngs::StdRng::seed_from_u64(*sub_seed);
                partition(data, ndims, cluster, config, 1, &mut sub_rng)
            }
        })
        .collect();

    eprintln!("    recursion: {:.3}s", t2.elapsed().as_secs_f64());
    results.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_partition_small_dataset() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..10).collect();
        let config = PartitionConfig {
            c_max: 10,
            c_min: 3,
            p_samp: 0.5,
            fanout: vec![3],
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let leaves = partition(&data, 2, &indices, &config, 0, &mut rng);

        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].indices.len(), 10);
    }

    #[test]
    fn test_partition_needs_splitting() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..200)
            .map(|_| rand::Rng::random_range(&mut rng, -10.0..10.0))
            .collect();
        let indices: Vec<usize> = (0..100).collect();
        let config = PartitionConfig {
            c_max: 20,
            c_min: 5,
            p_samp: 0.1,
            fanout: vec![3, 2],
        };

        let mut rng2 = rand::rngs::StdRng::seed_from_u64(123);
        let leaves = partition(&data, 2, &indices, &config, 0, &mut rng2);

        assert!(leaves.len() > 1, "expected multiple leaves, got {}", leaves.len());

        for leaf in &leaves {
            assert!(
                leaf.indices.len() <= config.c_max,
                "leaf too large: {}",
                leaf.indices.len()
            );
        }

        let total: usize = leaves.iter().map(|l| l.indices.len()).sum();
        assert!(
            total >= indices.len(),
            "total assignments {} < original count {}",
            total,
            indices.len()
        );
    }

    #[test]
    fn test_parallel_partition() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..2000)
            .map(|_| rand::Rng::random_range(&mut rng, -10.0..10.0))
            .collect();
        let indices: Vec<usize> = (0..1000).collect();
        let config = PartitionConfig {
            c_max: 50,
            c_min: 10,
            p_samp: 0.05,
            fanout: vec![5, 3],
        };

        let leaves = parallel_partition(&data, 2, &indices, &config, 42);

        assert!(leaves.len() > 1);
        for leaf in &leaves {
            assert!(
                leaf.indices.len() <= config.c_max,
                "leaf too large: {}",
                leaf.indices.len()
            );
        }
    }
}
