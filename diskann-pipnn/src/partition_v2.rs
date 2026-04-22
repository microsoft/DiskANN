/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Randomized Ball Carving (RBC) partitioning — v2 (iterative, parallel at every level).
//!
//! Recursively partitions the dataset into overlapping clusters using an iterative
//! work-queue approach. All oversized clusters at each level are processed in parallel.

use diskann::utils::VectorRepr;
use rand::prelude::IndexedRandom;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::partition::{Leaf, PartitionConfig};


/// Maximum leaders per partition level (paper recommendation).
const LEADER_HARDCAP: usize = 1000;

/// Compute the number of leaders to sample.
#[inline]
fn sample_num_leaders(n: usize, p_samp: f64) -> usize {
    ((n as f64 * p_samp).ceil() as usize)
        .clamp(2, LEADER_HARDCAP)
        .min(n)
}

/// A cluster that needs further partitioning.
struct WorkItem {
    indices: Vec<u32>,
    level: usize,
    seed: u64,
}

/// Partition the dataset into overlapping leaves.
///
/// Uses an iterative work-queue: each round processes all oversized clusters in parallel,
/// producing new work items for the next round. Completed leaves are pushed to a shared
/// result vec. No recursion, parallel at every level.
// Allow: callers wrap this in `pool.install(|| ...)`, so parallel work uses the correct pool.
#[allow(clippy::disallowed_methods)]
pub fn partition<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    npoints: usize,
    config: &PartitionConfig,
    seed: u64,
) -> Vec<Leaf> {
    let initial_indices: Vec<u32> = (0..npoints as u32).collect();

    if npoints <= config.c_max {
        return vec![Leaf {
            indices: initial_indices,
        }];
    }

    let nl0 = sample_num_leaders(npoints, config.p_samp);
    tracing::info!(npoints, leaders = nl0, ndims, "Partition start");

    let mut leaves: Vec<Leaf> = Vec::new();
    let mut work = vec![WorkItem {
        indices: initial_indices,
        level: 0,
        seed,
    }];

    let mut iteration = 0;
    while !work.is_empty() {
        iteration += 1;
        assert!(iteration <= 50, "partition exceeded 50 iterations");
        // flat_map: parallel partition + collect all results in one pass.
        // Each partition_one_level returns (work_items, leaves) — we flatten
        // both into separate vecs without a serial extend loop.
        let results: Vec<(Vec<WorkItem>, Vec<Leaf>)> = work
            .into_par_iter()
            .map(|item| partition_one_level(data, ndims, config, item))
            .collect();

        let total_work: usize = results.iter().map(|(w, _)| w.len()).sum();
        let total_leaves: usize = results.iter().map(|(_, l)| l.len()).sum();
        let mut next_work = Vec::with_capacity(total_work);
        leaves.reserve(total_leaves);
        for (wi, lv) in results {
            next_work.extend(wi);
            leaves.extend(lv);
        }
        work = next_work;
    }

    leaves
}

/// Quantized partition variant using Hamming distance.
#[allow(clippy::disallowed_methods)]
pub fn partition_quantized(
    qdata: &crate::quantize::QuantizedData,
    npoints: usize,
    config: &PartitionConfig,
    seed: u64,
) -> Vec<Leaf> {
    let initial_indices: Vec<u32> = (0..npoints as u32).collect();

    if npoints <= config.c_max {
        return vec![Leaf {
            indices: initial_indices,
        }];
    }

    let mut leaves: Vec<Leaf> = Vec::new();
    let mut work = vec![WorkItem {
        indices: initial_indices,
        level: 0,
        seed,
    }];

    let mut iteration = 0;
    while !work.is_empty() {
        iteration += 1;
        assert!(iteration <= 50, "quantized partition exceeded 50 iterations");
        let results: Vec<(Vec<WorkItem>, Vec<Leaf>)> = work
            .into_par_iter()
            .map(|item| partition_one_level_quantized(qdata, config, item))
            .collect();

        let total_work: usize = results.iter().map(|(w, _)| w.len()).sum();
        let total_leaves: usize = results.iter().map(|(_, l)| l.len()).sum();
        let mut next_work = Vec::with_capacity(total_work);
        leaves.reserve(total_leaves);
        for (wi, lv) in results {
            next_work.extend(wi);
            leaves.extend(lv);
        }
        work = next_work;
    }

    tracing::info!(total_leaves = leaves.len(), levels = iteration, "Partition complete (quantized)");
    leaves
}

/// Process one cluster: assign to leaders, merge small, return oversized as new work items
/// and completed leaves.
fn partition_one_level<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    config: &PartitionConfig,
    item: WorkItem,
) -> (Vec<WorkItem>, Vec<Leaf>) {
    let n = item.indices.len();
    debug_assert!(n > config.c_max);

    let fanout = config
        .fanout
        .get(item.level)
        .copied()
        .unwrap_or(1)
        .min(n);
    let num_leaders = sample_num_leaders(n, config.p_samp);

    // Deterministic seed derived from parent: no syscall, reproducible.
    let seed = item.seed.wrapping_mul(6364136223846793005).wrapping_add(n as u64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let leaders: Vec<u32> = item
        .indices
        .choose_multiple(&mut rng, num_leaders)
        .copied()
        .collect();

    // Assign each point to its `fanout` nearest leaders → per-leader clusters.
    let clusters = assign_to_leaders(data, ndims, &item.indices, &leaders, fanout, config.metric);

    let merged = merge_small(clusters, config.c_min, config.c_max);

    let mut next_work = Vec::new();
    let mut finished_leaves = Vec::new();
    for cluster in merged {
        if cluster.len() <= config.c_max {
            finished_leaves.push(Leaf { indices: cluster });
        } else {
            next_work.push(WorkItem {
                indices: cluster,
                level: item.level + 1,
                seed,
            });
        }
    }
    (next_work, finished_leaves)
}

/// Quantized version of partition_one_level.
fn partition_one_level_quantized(
    qdata: &crate::quantize::QuantizedData,
    config: &PartitionConfig,
    item: WorkItem,
) -> (Vec<WorkItem>, Vec<Leaf>) {
    let n = item.indices.len();
    debug_assert!(n > config.c_max);

    let fanout = config
        .fanout
        .get(item.level)
        .copied()
        .unwrap_or(1)
        .min(n);
    let num_leaders = sample_num_leaders(n, config.p_samp);

    // Deterministic seed derived from parent: no syscall, reproducible.
    let seed = item.seed.wrapping_mul(6364136223846793005).wrapping_add(n as u64);

    let stride = (n / num_leaders).max(1);
    let leaders: Vec<u32> = (0..num_leaders)
        .map(|i| item.indices[i * stride])
        .collect();

    let clusters = assign_to_leaders_quantized(qdata, &item.indices, &leaders, fanout);

    let merged = merge_small(clusters, config.c_min, config.c_max);

    let mut next_work = Vec::new();
    let mut finished_leaves = Vec::new();
    for cluster in merged {
        if cluster.len() <= config.c_max {
            finished_leaves.push(Leaf { indices: cluster });
        } else {
            next_work.push(WorkItem {
                indices: cluster,
                level: item.level + 1,
                seed,
            });
        }
    }
    (next_work, finished_leaves)
}

// ─── Assignment ──────────────────────────────────────────────────────────────

/// Assign each point to its `fanout` nearest leaders using native SIMD distance.
/// Point-by-point: no large temporary matrix, works directly on native type T.
/// All indices are u32 global IDs. Returns per-leader clusters as Vec<Vec<u32>>.
fn assign_to_leaders<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    points: &[u32],
    leaders: &[u32],
    fanout: usize,
    metric: diskann_vector::distance::Metric,
) -> Vec<Vec<u32>> {
    let np = points.len();
    let nl = leaders.len();
    let num_assign = fanout.min(nl);

    use diskann_vector::distance::Metric;

    // Extract leader data into contiguous f32 array.
    let mut l_data = vec![0.0f32; nl * ndims];
    for (i, &idx) in leaders.iter().enumerate() {
        let src = &data[idx as usize * ndims..(idx as usize + 1) * ndims];
        T::as_f32_into(src, &mut l_data[i * ndims..(i + 1) * ndims]).expect("f32 conversion");
    }

    // Precompute leader norms for L2/Cosine.
    let l_norms: Vec<f32> = match metric {
        Metric::L2 => l_data
            .chunks_exact(ndims)
            .map(|row| row.iter().map(|v| v * v).sum())
            .collect(),
        Metric::Cosine => l_data
            .chunks_exact(ndims)
            .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect(),
        Metric::CosineNormalized | Metric::InnerProduct => Vec::new(),
    };

    // Flat assignments.
    let mut assignments = vec![0u32; np * num_assign];

    // 16MB dots output fits in cache for the top-k scan.
    let stripe: usize =
        ((16 * 1024 * 1024) / (nl.max(1) * std::mem::size_of::<f32>())).clamp(1, np);

    // Small clusters: run single-threaded (no rayon overhead).
    // Large clusters: parallel stripes.
    let process_stripe = |(stripe_idx, assign_chunk): (usize, &mut [u32])| {
            let start = stripe_idx * stripe;
            let end = (start + stripe).min(np);
            let sn = end - start;
            let stripe_points = &points[start..end];

            // Convert stripe to f32.
            let mut p_data = vec![0.0f32; sn * ndims];
            for (i, &idx) in stripe_points.iter().enumerate() {
                let src = &data[idx as usize * ndims..(idx as usize + 1) * ndims];
                T::as_f32_into(src, &mut p_data[i * ndims..(i + 1) * ndims])
                    .expect("f32 conversion");
            }

            // GEMM: dots[i * nl + j] = dot(point_i, leader_j).
            let mut dots = vec![0.0f32; sn * nl];
            crate::gemm::sgemm_abt(&p_data, sn, ndims, &l_data, nl, &mut dots);

            // For each point: compute distances from dots, select top-k.
            let mut buf: Vec<(u32, f32)> = Vec::with_capacity(nl);
            for i in 0..sn {
                let dot_row = &dots[i * nl..(i + 1) * nl];
                buf.clear();

                match metric {
                    Metric::CosineNormalized => {
                        for (j, &d) in dot_row.iter().enumerate() {
                            buf.push((j as u32, (1.0 - d).max(0.0)));
                        }
                    }
                    Metric::Cosine => {
                        let pi_sqrt: f32 = p_data[i * ndims..(i + 1) * ndims]
                            .iter()
                            .map(|v| v * v)
                            .sum::<f32>()
                            .sqrt();
                        for (j, &d) in dot_row.iter().enumerate() {
                            let denom = pi_sqrt * l_norms[j];
                            let cos = if denom > 0.0 { d / denom } else { 0.0 };
                            buf.push((j as u32, (1.0 - cos).max(0.0)));
                        }
                    }
                    Metric::L2 => {
                        let pi: f32 = p_data[i * ndims..(i + 1) * ndims]
                            .iter()
                            .map(|v| v * v)
                            .sum();
                        for (j, &d) in dot_row.iter().enumerate() {
                            buf.push((j as u32, (pi + l_norms[j] - 2.0 * d).max(0.0)));
                        }
                    }
                    Metric::InnerProduct => {
                        for (j, &d) in dot_row.iter().enumerate() {
                            buf.push((j as u32, -d));
                        }
                    }
                }

                if num_assign > 0 && num_assign < buf.len() {
                    buf.select_nth_unstable_by(num_assign - 1, |a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                let out = &mut assign_chunk[i * num_assign..(i + 1) * num_assign];
                for k in 0..num_assign {
                    out[k] = buf[k].0;
                }
            }
    };

    if np <= stripe {
        // Single stripe — run inline, no rayon.
        process_stripe((0, &mut assignments));
    } else {
        // Multiple stripes — parallel.
        #[allow(clippy::disallowed_methods)]
        assignments
            .par_chunks_mut(stripe * num_assign)
            .enumerate()
            .for_each(process_stripe);
    }

    // Aggregate into per-leader clusters using global point IDs.
    let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); nl];
    for (i, pt) in points.iter().enumerate() {
        let row = &assignments[i * num_assign..(i + 1) * num_assign];
        for &leader_local in row {
            clusters[leader_local as usize].push(*pt);
        }
    }
    clusters
}

/// Quantized assignment using Hamming distance.
fn assign_to_leaders_quantized(
    qdata: &crate::quantize::QuantizedData,
    points: &[u32],
    leaders: &[u32],
    fanout: usize,
) -> Vec<Vec<u32>> {
    let np = points.len();
    let nl = leaders.len();
    let num_assign = fanout.min(nl);
    let u64s = qdata.u64s_per_vec();

    // Pre-extract leader data.
    let mut leader_data: Vec<u64> = vec![0u64; nl * u64s];
    for (i, &idx) in leaders.iter().enumerate() {
        leader_data[i * u64s..(i + 1) * u64s].copy_from_slice(qdata.get_u64(idx as usize));
    }

    let mut assignments = vec![0u32; np * num_assign];

    // 16MB stripe for cache efficiency.
    let stripe: usize = ((16 * 1024 * 1024)
        / (nl.max(1) * std::mem::size_of::<u64>() * u64s.max(1)))
    .clamp(256, np);

    #[allow(clippy::disallowed_methods)]
    assignments
        .par_chunks_mut(stripe * num_assign)
        .enumerate()
        .for_each(|(stripe_idx, assign_chunk)| {
            let start = stripe_idx * stripe;
            let end = (start + stripe).min(np);
            let sn = end - start;

            let mut point_data: Vec<u64> = vec![0u64; sn * u64s];
            for i in 0..sn {
                let src = qdata.get_u64(points[start + i] as usize);
                point_data[i * u64s..(i + 1) * u64s].copy_from_slice(src);
            }

            let mut buf: Vec<(u32, f32)> = Vec::with_capacity(nl);
            let ld_ptr = leader_data.as_ptr();
            let pd_ptr = point_data.as_ptr();

            for i in 0..sn {
                // SAFETY: bounds checked by construction.
                let pt_base = unsafe { pd_ptr.add(i * u64s) };
                buf.clear();
                for j in 0..nl {
                    let ld_base = unsafe { ld_ptr.add(j * u64s) };
                    let mut h = 0u32;
                    for k in 0..u64s {
                        unsafe {
                            h += (*pt_base.add(k) ^ *ld_base.add(k)).count_ones();
                        }
                    }
                    buf.push((j as u32, h as f32));
                }
                if num_assign > 0 && num_assign < buf.len() {
                    buf.select_nth_unstable_by(num_assign - 1, |a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                let out = &mut assign_chunk[i * num_assign..(i + 1) * num_assign];
                for k in 0..num_assign {
                    out[k] = buf[k].0;
                }
            }
        });

    let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); nl];
    for (i, pt) in points.iter().enumerate() {
        let row = &assignments[i * num_assign..(i + 1) * num_assign];
        for &leader_local in row {
            clusters[leader_local as usize].push(*pt);
        }
    }
    clusters
}

// ─── Merge Small Clusters ────────────────────────────────────────────────────

/// Merge clusters smaller than c_min by combining them.
/// Deduplicates after each extend. Result clusters are always <= c_max.
fn merge_small(clusters: Vec<Vec<u32>>, c_min: usize, c_max: usize) -> Vec<Vec<u32>> {
    let mut result: Vec<Vec<u32>> = Vec::new();
    let mut buf = std::collections::HashSet::<u32>::new();

    for c in clusters {
        if c.is_empty() {
            continue;
        }
        if c.len() >= c_min {
            result.push(c);
        } else {
            if buf.len() + c.len() > c_max {
                result.push(buf.drain().collect());
            }
            buf.extend(c);
        }
    }
    if !buf.is_empty() {
        result.push(buf.into_iter().collect());
    }

    result
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_basic() {
        let npoints = 1000;
        let ndims = 8;
        let data: Vec<f32> = {
            use rand::{Rng, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            (0..npoints * ndims).map(|_| rng.random::<f32>()).collect()
        };
        let config = PartitionConfig {
            c_max: 64,
            c_min: 16,
            p_samp: 0.1,
            fanout: vec![4, 2],
            metric: diskann_vector::distance::Metric::L2,
        };
        let leaves = partition(&data, ndims, npoints, &config, 123);

        // All points should appear at least once (overlapping partitions).
        let mut seen = vec![false; npoints];
        for leaf in &leaves {
            assert!(leaf.indices.len() <= config.c_max, "leaf too large");
            for &idx in &leaf.indices {
                seen[idx as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "some points missing");
    }

    #[test]
    fn test_partition_small_dataset() {
        let npoints = 50;
        let ndims = 4;
        let data: Vec<f32> = vec![1.0; npoints * ndims];
        let config = PartitionConfig {
            c_max: 64,
            c_min: 8,
            p_samp: 0.1,
            fanout: vec![3],
            metric: diskann_vector::distance::Metric::L2,
        };
        let leaves = partition(&data, ndims, npoints, &config, 0);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].indices.len(), npoints);
    }
}
