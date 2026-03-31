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

use diskann::utils::VectorRepr;
use rand::prelude::IndexedRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Maximum recursion depth to prevent stack overflow.
const MAX_DEPTH: usize = 30;

/// Maximum leaders per partition level. The paper recommends ~1000 as a practical
/// upper bound; larger values increase partition GEMM cost without improving quality.
const LEADER_HARDCAP: usize = 1000;

/// Compute the number of leaders to sample for a partition level.
#[inline]
fn sample_num_leaders(n: usize, p_samp: f64) -> usize {
    ((n as f64 * p_samp).ceil() as usize)
        .max(2)
        .min(LEADER_HARDCAP)
        .min(n)
}

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
    /// Distance metric for partition assignment.
    pub metric: diskann_vector::distance::Metric,
}

/// Quantized version of partition_assign using Hamming distance on 1-bit data.
/// Pre-extracts leader u64 data for cache locality.
fn partition_assign_quantized(
    qdata: &crate::quantize::QuantizedData,
    points: &[usize],
    leaders: &[usize],
    fanout: usize,
) -> Vec<Vec<usize>> {
    let np = points.len();
    let nl = leaders.len();
    let num_assign = fanout.min(nl);
    let u64s = qdata.u64s_per_vec();

    // Pre-extract leader data into contiguous cache-friendly array.
    let mut leader_data: Vec<u64> = vec![0u64; nl * u64s];
    for (i, &idx) in leaders.iter().enumerate() {
        leader_data[i * u64s..(i + 1) * u64s].copy_from_slice(qdata.get_u64(idx));
    }

    let mut assignments = vec![0u32; np * num_assign];

    // Adaptive stripe: limit per-stripe memory to ~16 MB, matching the FP path.
    let stripe: usize = ((16 * 1024 * 1024)
        / (nl.max(1) * std::mem::size_of::<u64>() * u64s.max(1)))
    .clamp(256, 32_768);
    assignments
        .par_chunks_mut(stripe * num_assign)
        .enumerate()
        .for_each(|(stripe_idx, assign_chunk)| {
            let start = stripe_idx * stripe;
            let end = (start + stripe).min(np);
            let sn = end - start;

            // Pre-extract point data for this stripe.
            let mut point_data: Vec<u64> = vec![0u64; sn * u64s];
            for i in 0..sn {
                let src = qdata.get_u64(points[start + i]);
                point_data[i * u64s..(i + 1) * u64s].copy_from_slice(src);
            }

            let mut buf: Vec<(u32, f32)> = Vec::with_capacity(nl);
            let ld_ptr = leader_data.as_ptr();
            let pd_ptr = point_data.as_ptr();

            for i in 0..sn {
                let pt_base = unsafe { pd_ptr.add(i * u64s) };

                // Compute Hamming distance to all leaders + build buf in one pass.
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

    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); nl];
    for i in 0..np {
        let row = &assignments[i * num_assign..(i + 1) * num_assign];
        for &li in row {
            clusters[li as usize].push(i);
        }
    }
    clusters
}

/// Fused GEMM + assignment: compute distances to leaders in stripes and immediately
/// extract top-k assignments without materializing the full N x L distance matrix.
/// Peak memory: stripe * L * 4 bytes (~64MB) instead of N * L * 4 bytes.
/// Fused GEMM + assignment: compute distances to leaders in stripes and immediately
/// extract top-k assignments without materializing the full N x L distance matrix.
fn partition_assign<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    points: &[usize],
    leaders: &[usize],
    fanout: usize,
    metric: diskann_vector::distance::Metric,
) -> Vec<Vec<usize>> {
    let np = points.len();
    let nl = leaders.len();
    let num_assign = fanout.min(nl);

    use diskann_vector::distance::Metric;

    // Extract leader data (shared, stays in cache), converting T -> f32.
    let mut l_data = vec![0.0f32; nl * ndims];
    for (i, &idx) in leaders.iter().enumerate() {
        let src = &data[idx * ndims..(idx + 1) * ndims];
        let dst = &mut l_data[i * ndims..(i + 1) * ndims];
        T::as_f32_into(src, dst).expect("f32 conversion");
    }
    // Precompute leader norms.
    // L2 needs squared norms; Cosine needs sqrt norms; CosineNormalized/IP need none.
    let l_norms: Vec<f32> = match metric {
        Metric::L2 => {
            let mut norms = vec![0.0f32; nl];
            for i in 0..nl {
                let row = &l_data[i * ndims..(i + 1) * ndims];
                let mut norm = 0.0f32;
                for &v in row {
                    norm += v * v;
                }
                norms[i] = norm;
            }
            norms
        }
        Metric::Cosine => {
            let mut norms = vec![0.0f32; nl];
            for i in 0..nl {
                let row = &l_data[i * ndims..(i + 1) * ndims];
                let mut norm = 0.0f32;
                for &v in row {
                    norm += v * v;
                }
                norms[i] = norm.sqrt();
            }
            norms
        }
        Metric::CosineNormalized | Metric::InnerProduct => Vec::new(),
    };

    // Flat assignments: assignments[i * num_assign .. (i+1) * num_assign]
    let mut assignments = vec![0u32; np * num_assign];

    // Fused parallel stripes: GEMM + distance + top-k in one pass.
    // Adaptive stripe size: limit per-stripe GEMM output to ~16 MB.
    // Smaller stripes reduce concurrent memory from ~1.4 GB (8 threads × 90 MB)
    // to ~350 MB (8 threads × 22 MB), cutting partition peak RSS by ~1 GB.
    // Partition is <5% of total build time, so the throughput cost is negligible.
    let stripe: usize =
        ((16 * 1024 * 1024) / (nl.max(1) * std::mem::size_of::<f32>())).clamp(256, 16_384);
    assignments
        .par_chunks_mut(stripe * num_assign)
        .enumerate()
        .for_each(|(stripe_idx, assign_chunk)| {
            let start = stripe_idx * stripe;
            let end = (start + stripe).min(np);
            let sn = end - start;
            let stripe_points = &points[start..end];

            let mut p_data = vec![0.0f32; sn * ndims];
            for (i, &idx) in stripe_points.iter().enumerate() {
                let src = &data[idx * ndims..(idx + 1) * ndims];
                let dst = &mut p_data[i * ndims..(i + 1) * ndims];
                T::as_f32_into(src, dst).expect("f32 conversion");
            }

            let mut dots = vec![0.0f32; sn * nl];
            crate::gemm::sgemm_abt(&p_data, sn, ndims, &l_data, nl, &mut dots);

            let mut buf: Vec<(u32, f32)> = Vec::with_capacity(nl);
            for i in 0..sn {
                let dot_row = &dots[i * nl..(i + 1) * nl];

                buf.clear();
                match metric {
                    Metric::CosineNormalized => {
                        // Pre-normalized: dist = 1 - dot(a, b)
                        for j in 0..nl {
                            buf.push((j as u32, (1.0 - dot_row[j]).max(0.0)));
                        }
                    }
                    Metric::Cosine => {
                        // Unnormalized: dist = 1 - dot(a,b)/(||a||*||b||)
                        let mut pi = 0.0f32;
                        let row = &p_data[i * ndims..(i + 1) * ndims];
                        for &v in row {
                            pi += v * v;
                        }
                        let pi_sqrt = pi.sqrt();
                        for j in 0..nl {
                            let denom = pi_sqrt * l_norms[j];
                            let cos_sim = if denom > 0.0 { dot_row[j] / denom } else { 0.0 };
                            buf.push((j as u32, (1.0 - cos_sim).max(0.0)));
                        }
                    }
                    Metric::L2 => {
                        let mut pi = 0.0f32;
                        let row = &p_data[i * ndims..(i + 1) * ndims];
                        for &v in row {
                            pi += v * v;
                        }
                        for j in 0..nl {
                            let d = (pi + l_norms[j] - 2.0 * dot_row[j]).max(0.0);
                            buf.push((j as u32, d));
                        }
                    }
                    Metric::InnerProduct => {
                        for j in 0..nl {
                            buf.push((j as u32, -dot_row[j]));
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

/// Merge undersized quantized clusters into the nearest large cluster by Hamming distance.
/// Uses the first point of each small cluster as a representative and computes
/// Hamming distance to the first point of each large cluster (cheap proxy for centroid).
fn merge_small_quantized(
    qdata: &crate::quantize::QuantizedData,
    mut clusters: Vec<Vec<usize>>,
    c_min: usize,
) -> Vec<Vec<usize>> {
    let mut large: Vec<Vec<usize>> = Vec::new();
    let mut smalls: Vec<Vec<usize>> = Vec::new();
    for c in clusters.drain(..) {
        if c.len() < c_min && !c.is_empty() {
            smalls.push(c);
        } else if !c.is_empty() {
            large.push(c);
        }
    }
    if smalls.is_empty() || large.is_empty() {
        if large.is_empty() {
            return smalls;
        }
        return large;
    }
    for small in smalls {
        let rep = qdata.get_u64(small[0]);
        let nearest = large
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let d = crate::quantize::QuantizedData::hamming_u64(rep, qdata.get_u64(c[0]));
                (i, d)
            })
            .min_by_key(|&(_, d)| d)
            .map(|(i, _)| i)
            .unwrap_or(0);
        large[nearest].extend(small);
    }
    large
}

/// Merge undersized clusters into the nearest large cluster by centroid distance.
///
/// Paper (arXiv:2602.21247): "Merge undersized clusters into the nearest
/// (by centroid) appropriately-sized cluster."
fn merge_small_into_nearest<T: VectorRepr>(
    data: &[T],
    ndims: usize,
    mut clusters: Vec<Vec<usize>>,
    c_min: usize,
) -> Vec<Vec<usize>> {
    let mut large: Vec<Vec<usize>> = Vec::new();
    let mut smalls: Vec<Vec<usize>> = Vec::new();

    for c in clusters.drain(..) {
        if c.len() < c_min && !c.is_empty() {
            smalls.push(c);
        } else if !c.is_empty() {
            large.push(c);
        }
    }

    if smalls.is_empty() || large.is_empty() {
        if large.is_empty() {
            return smalls;
        }
        return large;
    }

    // Compute centroids for large clusters, converting T -> f32 per point.
    let centroids: Vec<Vec<f32>> = large
        .iter()
        .map(|c| {
            let mut centroid = vec![0.0f32; ndims];
            let inv = 1.0 / c.len() as f32;
            let mut point_buf = vec![0.0f32; ndims];
            for &idx in c {
                T::as_f32_into(&data[idx * ndims..(idx + 1) * ndims], &mut point_buf)
                    .expect("f32 conversion");
                for d in 0..ndims {
                    centroid[d] += point_buf[d];
                }
            }
            for d in 0..ndims {
                centroid[d] *= inv;
            }
            centroid
        })
        .collect();

    // For each small cluster, find nearest large cluster by L2 distance
    // from the small cluster's representative point to each large centroid.
    for small in smalls {
        let mut rep_buf = vec![0.0f32; ndims];
        T::as_f32_into(
            &data[small[0] * ndims..(small[0] + 1) * ndims],
            &mut rep_buf,
        )
        .expect("f32 conversion");
        let nearest = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut dist = 0.0f32;
                for d in 0..ndims {
                    let diff = unsafe { *rep_buf.get_unchecked(d) - *c.get_unchecked(d) };
                    dist += diff * diff;
                }
                (i, dist)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        large[nearest].extend(small);
    }

    large
}

/// Partition the dataset using Randomized Ball Carving.
///
/// `data` is row-major: npoints_global x ndims.
/// `indices` are the global indices of the points to partition.
pub fn partition<T: VectorRepr + Send + Sync>(
    data: &[T],
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
    if level >= MAX_DEPTH {
        return force_split(indices, config.c_max);
    }

    let fanout = if level < config.fanout.len() {
        config.fanout[level]
    } else {
        1
    };

    let num_leaders = sample_num_leaders(n, config.p_samp);
    let leaders: Vec<usize> = indices.choose_multiple(rng, num_leaders).copied().collect();

    // Fused GEMM + assignment (avoids materializing full distance matrix).
    let clusters_local = partition_assign(data, ndims, indices, &leaders, fanout, config.metric);

    // Map local indices back to global.
    let clusters: Vec<Vec<usize>> = clusters_local
        .into_iter()
        .map(|local_cluster| local_cluster.into_iter().map(|li| indices[li]).collect())
        .collect();

    // Merge undersized clusters into nearest large cluster by centroid proximity.
    let merged_clusters = merge_small_into_nearest(data, ndims, clusters, config.c_min);

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
pub fn parallel_partition<T: VectorRepr + Send + Sync>(
    data: &[T],
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

    let num_leaders = sample_num_leaders(n, config.p_samp);
    let leaders: Vec<usize> = indices
        .choose_multiple(&mut rng, num_leaders)
        .copied()
        .collect();

    let t0 = std::time::Instant::now();
    let clusters_local = partition_assign(data, ndims, indices, &leaders, fanout, config.metric);
    let assign_time = t0.elapsed();

    let t1 = std::time::Instant::now();
    let clusters: Vec<Vec<usize>> = clusters_local
        .into_iter()
        .map(|local_cluster| local_cluster.into_iter().map(|li| indices[li]).collect())
        .collect();
    let map_time = t1.elapsed();

    tracing::debug!(
        assign_secs = assign_time.as_secs_f64(),
        map_secs = map_time.as_secs_f64(),
        num_leaders = num_leaders,
        fanout = fanout,
        "top-level partition assign"
    );

    // Merge undersized clusters into nearest large cluster by centroid proximity.
    let merged_clusters = merge_small_into_nearest(data, ndims, clusters, config.c_min);

    let need_recurse = merged_clusters
        .iter()
        .filter(|c| c.len() > config.c_max)
        .count();
    let total_in_recurse: usize = merged_clusters
        .iter()
        .filter(|c| c.len() > config.c_max)
        .map(|c| c.len())
        .sum();
    tracing::debug!(
        num_clusters = merged_clusters.len(),
        need_recurse = need_recurse,
        total_in_recurse = total_in_recurse,
        "partition merge"
    );

    // Generate sub-seeds for parallel recursion.
    let sub_seeds: Vec<u64> = (0..merged_clusters.len()).map(|_| rng.random()).collect();

    // Recurse in parallel. Each cluster is either a leaf or needs further splitting.
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

    tracing::debug!(
        recursion_secs = t2.elapsed().as_secs_f64(),
        "partition recursion complete"
    );
    results.into_iter().flatten().collect()
}

/// Quantized version of parallel_partition using Hamming distance on 1-bit data.
pub fn parallel_partition_quantized(
    qdata: &crate::quantize::QuantizedData,
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

    let num_leaders = sample_num_leaders(n, config.p_samp);
    let leaders: Vec<usize> = indices
        .choose_multiple(&mut rng, num_leaders)
        .copied()
        .collect();

    let t0 = std::time::Instant::now();
    let clusters_local = partition_assign_quantized(qdata, indices, &leaders, fanout);
    let assign_time = t0.elapsed();

    let t1 = std::time::Instant::now();
    let clusters: Vec<Vec<usize>> = clusters_local
        .into_iter()
        .map(|local_cluster| local_cluster.into_iter().map(|li| indices[li]).collect())
        .collect();
    let map_time = t1.elapsed();

    tracing::debug!(
        assign_secs = assign_time.as_secs_f64(),
        map_secs = map_time.as_secs_f64(),
        num_leaders = num_leaders,
        fanout = fanout,
        "top-level partition assign (quantized)"
    );

    // Merge undersized clusters into nearest large cluster by Hamming distance.
    let merged_clusters = merge_small_quantized(qdata, clusters, config.c_min);

    let need_recurse = merged_clusters
        .iter()
        .filter(|c| c.len() > config.c_max)
        .count();
    tracing::debug!(
        num_clusters = merged_clusters.len(),
        need_recurse = need_recurse,
        "partition merge (quantized)"
    );

    let sub_seeds: Vec<u64> = (0..merged_clusters.len()).map(|_| rng.random()).collect();

    let t2 = std::time::Instant::now();
    let results: Vec<Vec<Leaf>> = merged_clusters
        .par_iter()
        .zip(sub_seeds.par_iter())
        .map(|(cluster, sub_seed)| {
            if cluster.len() <= config.c_max {
                vec![Leaf {
                    indices: cluster.clone(),
                }]
            } else if cluster.len() <= config.c_max * 3 {
                force_split(cluster, config.c_max)
            } else {
                // Recursive quantized partition.
                let mut sub_rng = rand::rngs::StdRng::seed_from_u64(*sub_seed);
                partition_quantized_recursive(qdata, cluster, config, 1, &mut sub_rng)
            }
        })
        .collect();

    tracing::debug!(
        recursion_secs = t2.elapsed().as_secs_f64(),
        "partition recursion complete (quantized)"
    );
    results.into_iter().flatten().collect()
}

fn partition_quantized_recursive(
    qdata: &crate::quantize::QuantizedData,
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
    if level >= MAX_DEPTH {
        return force_split(indices, config.c_max);
    }

    let fanout = if level < config.fanout.len() {
        config.fanout[level]
    } else {
        1
    };
    let num_leaders = sample_num_leaders(n, config.p_samp);

    let leaders: Vec<usize> = indices.choose_multiple(rng, num_leaders).copied().collect();

    let clusters_local = partition_assign_quantized(qdata, indices, &leaders, fanout);
    let clusters: Vec<Vec<usize>> = clusters_local
        .into_iter()
        .map(|lc| lc.into_iter().map(|li| indices[li]).collect())
        .collect();

    // Merge undersized clusters into nearest large cluster by Hamming distance.
    let merged = merge_small_quantized(qdata, clusters, config.c_min);

    if merged.len() == 1 && merged[0].len() > config.c_max {
        return force_split(&merged[0], config.c_max);
    }

    let mut leaves = Vec::new();
    for cluster in merged {
        if cluster.len() <= config.c_max {
            leaves.push(Leaf { indices: cluster });
        } else {
            let sub_seed: u64 = rng.random();
            let mut sub_rng = rand::rngs::StdRng::seed_from_u64(sub_seed);
            leaves.extend(partition_quantized_recursive(
                qdata,
                &cluster,
                config,
                level + 1,
                &mut sub_rng,
            ));
        }
    }
    leaves
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_vector::distance::Metric;
    use rand::{Rng, SeedableRng};

    fn gen_data(npoints: usize, ndims: usize, seed: u64) -> Vec<f32> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..npoints * ndims)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect()
    }

    #[test]
    fn test_partition_small_dataset() {
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..10).collect();
        let config = PartitionConfig {
            c_max: 10,
            c_min: 3,
            p_samp: 0.5,
            fanout: vec![3],
            metric: diskann_vector::distance::Metric::L2,
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
            metric: diskann_vector::distance::Metric::L2,
        };

        let mut rng2 = rand::rngs::StdRng::seed_from_u64(123);
        let leaves = partition(&data, 2, &indices, &config, 0, &mut rng2);

        assert!(
            leaves.len() > 1,
            "expected multiple leaves, got {}",
            leaves.len()
        );

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
            metric: diskann_vector::distance::Metric::L2,
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

    #[test]
    fn test_partition_overlap() {
        // With fanout > 1, each point is assigned to multiple leaders,
        // so the total assignments across all leaves should exceed the
        // original point count (overlap).
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let npoints = 500;
        let ndims = 4;
        let data: Vec<f32> = (0..npoints * ndims)
            .map(|_| rand::Rng::random_range(&mut rng, -5.0..5.0))
            .collect();
        let indices: Vec<usize> = (0..npoints).collect();
        let config = PartitionConfig {
            c_max: 100,
            c_min: 20,
            p_samp: 0.05,
            fanout: vec![3, 2], // fanout > 1 creates overlap
            metric: diskann_vector::distance::Metric::L2,
        };

        let leaves = parallel_partition(&data, ndims, &indices, &config, 42);

        let total_in_leaves: usize = leaves.iter().map(|l| l.indices.len()).sum();
        assert!(
            total_in_leaves >= npoints,
            "total in leaves ({}) should be >= npoints ({})",
            total_in_leaves,
            npoints
        );
    }

    #[test]
    fn test_partition_respects_c_max() {
        // All leaves must have at most c_max elements.
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let npoints = 300;
        let ndims = 4;
        let data: Vec<f32> = (0..npoints * ndims)
            .map(|_| rand::Rng::random_range(&mut rng, -5.0..5.0))
            .collect();
        let indices: Vec<usize> = (0..npoints).collect();
        let config = PartitionConfig {
            c_max: 40,
            c_min: 10,
            p_samp: 0.1,
            fanout: vec![5, 2],
            metric: diskann_vector::distance::Metric::L2,
        };

        let leaves = parallel_partition(&data, ndims, &indices, &config, 99);
        for (i, leaf) in leaves.iter().enumerate() {
            assert!(
                leaf.indices.len() <= config.c_max,
                "leaf {} has size {} > c_max {}",
                i,
                leaf.indices.len(),
                config.c_max
            );
        }
    }

    #[test]
    fn test_partition_single_point() {
        let data = vec![1.0f32, 2.0];
        let indices = vec![0usize];
        let config = PartitionConfig {
            c_max: 10,
            c_min: 1,
            p_samp: 0.5,
            fanout: vec![3],
            metric: diskann_vector::distance::Metric::L2,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let leaves = partition(&data, 2, &indices, &config, 0, &mut rng);
        assert_eq!(leaves.len(), 1, "single point should produce 1 leaf");
        assert_eq!(
            leaves[0].indices.len(),
            1,
            "leaf should contain exactly 1 point"
        );
        assert_eq!(leaves[0].indices[0], 0, "leaf should contain index 0");
    }

    #[test]
    fn test_partition_two_points() {
        let data = vec![0.0f32, 0.0, 10.0, 10.0];
        let indices = vec![0, 1];
        let config = PartitionConfig {
            c_max: 5,
            c_min: 1,
            p_samp: 0.5,
            fanout: vec![3],
            metric: diskann_vector::distance::Metric::L2,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let leaves = partition(&data, 2, &indices, &config, 0, &mut rng);
        assert_eq!(
            leaves.len(),
            1,
            "two points with c_max=5 should produce 1 leaf"
        );
        assert_eq!(
            leaves[0].indices.len(),
            2,
            "leaf should contain both points"
        );
    }

    #[test]
    fn test_partition_all_identical() {
        // All identical vectors should still partition without crashing.
        let npoints = 100;
        let ndims = 4;
        let data = vec![42.0f32; npoints * ndims];
        let indices: Vec<usize> = (0..npoints).collect();
        let config = PartitionConfig {
            c_max: 20,
            c_min: 5,
            p_samp: 0.1,
            fanout: vec![3],
            metric: diskann_vector::distance::Metric::L2,
        };
        let leaves = parallel_partition(&data, ndims, &indices, &config, 42);
        assert!(!leaves.is_empty(), "should produce at least one leaf");
        let total: usize = leaves.iter().map(|l| l.indices.len()).sum();
        assert!(
            total >= npoints,
            "total in leaves ({}) should be >= npoints ({})",
            total,
            npoints
        );
        for (i, leaf) in leaves.iter().enumerate() {
            assert!(
                leaf.indices.len() <= config.c_max,
                "leaf {} has {} elements > c_max={}",
                i,
                leaf.indices.len(),
                config.c_max
            );
        }
    }

    #[test]
    fn test_partition_high_fanout() {
        // fanout > npoints should still work (clamped to num_leaders).
        let npoints = 20;
        let ndims = 4;
        let mut rng_data = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..npoints * ndims)
            .map(|_| rand::Rng::random_range(&mut rng_data, -10.0..10.0))
            .collect();
        let indices: Vec<usize> = (0..npoints).collect();
        let config = PartitionConfig {
            c_max: 5,
            c_min: 2,
            p_samp: 0.5,
            fanout: vec![100], // much larger than npoints
            metric: diskann_vector::distance::Metric::L2,
        };
        let leaves = parallel_partition(&data, ndims, &indices, &config, 42);
        assert!(
            !leaves.is_empty(),
            "high fanout should still produce leaves"
        );
        for (i, leaf) in leaves.iter().enumerate() {
            assert!(
                leaf.indices.len() <= config.c_max,
                "leaf {} has {} elements > c_max={}",
                i,
                leaf.indices.len(),
                config.c_max
            );
        }
    }

    #[test]
    fn test_partition_multi_level_fanout() {
        // Multi-level fanout vec![4,2] should work and produce valid leaves.
        let npoints = 200;
        let ndims = 4;
        let mut rng_data = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..npoints * ndims)
            .map(|_| rand::Rng::random_range(&mut rng_data, -10.0..10.0))
            .collect();
        let indices: Vec<usize> = (0..npoints).collect();
        let config = PartitionConfig {
            c_max: 30,
            c_min: 8,
            p_samp: 0.1,
            fanout: vec![4, 2],
            metric: diskann_vector::distance::Metric::L2,
        };
        let leaves = parallel_partition(&data, ndims, &indices, &config, 42);
        assert!(
            leaves.len() > 1,
            "multi-level fanout should produce multiple leaves"
        );
        for (i, leaf) in leaves.iter().enumerate() {
            assert!(
                leaf.indices.len() <= config.c_max,
                "leaf {} has {} elements > c_max={}",
                i,
                leaf.indices.len(),
                config.c_max
            );
        }
    }

    #[test]
    fn test_partition_c_min_equals_c_max() {
        // c_min == c_max is a valid (if unusual) configuration.
        let npoints = 100;
        let ndims = 4;
        let mut rng_data = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..npoints * ndims)
            .map(|_| rand::Rng::random_range(&mut rng_data, -10.0..10.0))
            .collect();
        let indices: Vec<usize> = (0..npoints).collect();
        let config = PartitionConfig {
            c_max: 30,
            c_min: 30,
            p_samp: 0.1,
            fanout: vec![3],
            metric: diskann_vector::distance::Metric::L2,
        };
        let leaves = parallel_partition(&data, ndims, &indices, &config, 42);
        assert!(!leaves.is_empty(), "c_min == c_max should produce leaves");
        for (i, leaf) in leaves.iter().enumerate() {
            assert!(
                leaf.indices.len() <= config.c_max,
                "leaf {} has {} elements > c_max={}",
                i,
                leaf.indices.len(),
                config.c_max
            );
        }
    }

    #[test]
    fn test_partition_large_p_samp() {
        // p_samp=1.0 means sample all points as leaders.
        let npoints = 50;
        let ndims = 4;
        let mut rng_data = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..npoints * ndims)
            .map(|_| rand::Rng::random_range(&mut rng_data, -10.0..10.0))
            .collect();
        let indices: Vec<usize> = (0..npoints).collect();
        let config = PartitionConfig {
            c_max: 10,
            c_min: 3,
            p_samp: 1.0,
            fanout: vec![3],
            metric: diskann_vector::distance::Metric::L2,
        };
        let leaves = parallel_partition(&data, ndims, &indices, &config, 42);
        assert!(!leaves.is_empty(), "p_samp=1.0 should produce leaves");
        for (i, leaf) in leaves.iter().enumerate() {
            assert!(
                leaf.indices.len() <= config.c_max,
                "leaf {} has {} elements > c_max={}",
                i,
                leaf.indices.len(),
                config.c_max
            );
        }
    }

    #[test]
    fn test_partition_quantized() {
        // Quantized partition should produce valid leaves with same constraints.
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let npoints = 300;
        let ndims = 64; // must be multiple of 64 for u64 alignment
        let data: Vec<f32> = (0..npoints * ndims)
            .map(|_| rand::Rng::random_range(&mut rng, -5.0..5.0))
            .collect();
        let indices: Vec<usize> = (0..npoints).collect();
        let config = PartitionConfig {
            c_max: 80,
            c_min: 20,
            p_samp: 0.05,
            fanout: vec![3, 2],
            metric: diskann_vector::distance::Metric::L2,
        };

        let (shift, inverse_scale) = {
            use diskann_quantization::scalar::train::ScalarQuantizationParameters;
            use diskann_utils::views::MatrixView;
            let dm = MatrixView::try_from(data.as_slice(), npoints, ndims).unwrap();
            let q = ScalarQuantizationParameters::default().train(dm);
            let s = q.scale();
            (q.shift().to_vec(), if s == 0.0 { 1.0 } else { 1.0 / s })
        };
        let qdata = crate::quantize::quantize_1bit(&data, npoints, ndims, &shift, inverse_scale);
        let leaves = parallel_partition_quantized(&qdata, &indices, &config, 42);

        assert!(!leaves.is_empty(), "no leaves produced");
        for (i, leaf) in leaves.iter().enumerate() {
            assert!(
                leaf.indices.len() <= config.c_max,
                "quantized leaf {} has size {} > c_max {}",
                i,
                leaf.indices.len(),
                config.c_max
            );
            // All indices should be valid.
            for &idx in &leaf.indices {
                assert!(idx < npoints, "index {} out of range", idx);
            }
        }
    }

    #[test]
    fn test_partition_cosine_normalized() {
        // Pre-normalized vectors on the unit circle.
        let npoints = 200;
        let ndims = 8;
        let mut data = gen_data(npoints, ndims, 42);
        for i in 0..npoints {
            let row = &mut data[i * ndims..(i + 1) * ndims];
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }
        let indices: Vec<usize> = (0..npoints).collect();
        let config = PartitionConfig {
            c_max: 64,
            c_min: 16,
            p_samp: 0.1,
            fanout: vec![4],
            metric: Metric::CosineNormalized,
        };
        let leaves = parallel_partition(&data, ndims, &indices, &config, 42);
        assert!(!leaves.is_empty());
        let total: usize = leaves.iter().map(|l| l.indices.len()).sum();
        // With fanout=4, total assignments > npoints due to overlap.
        assert!(total >= npoints);
        for leaf in &leaves {
            assert!(leaf.indices.len() <= config.c_max);
        }
    }

    #[test]
    fn test_partition_cosine_unnormalized() {
        // Vectors with varying norms — cosine should normalize internally.
        let npoints = 100;
        let ndims = 4;
        let data = gen_data(npoints, ndims, 99);
        let indices: Vec<usize> = (0..npoints).collect();
        let config = PartitionConfig {
            c_max: 32,
            c_min: 8,
            p_samp: 0.1,
            fanout: vec![3],
            metric: Metric::Cosine,
        };
        let leaves = parallel_partition(&data, ndims, &indices, &config, 42);
        assert!(!leaves.is_empty());
        for leaf in &leaves {
            assert!(leaf.indices.len() <= config.c_max);
        }
    }

    #[test]
    fn test_partition_zero_norm_vectors() {
        // Mix of zero-norm and normal vectors — should not panic.
        let mut data = gen_data(50, 4, 42);
        // Set first 5 vectors to all zeros.
        for v in data[..20].iter_mut() {
            *v = 0.0;
        }
        let indices: Vec<usize> = (0..50).collect();
        let config = PartitionConfig {
            c_max: 16,
            c_min: 4,
            p_samp: 0.2,
            fanout: vec![2],
            metric: Metric::Cosine,
        };
        let leaves = parallel_partition(&data, 4, &indices, &config, 42);
        assert!(!leaves.is_empty());
        // Zero-norm vectors should appear in at least one leaf.
        let all_indices: std::collections::HashSet<usize> = leaves
            .iter()
            .flat_map(|l| l.indices.iter().copied())
            .collect();
        for i in 0..5 {
            assert!(
                all_indices.contains(&i),
                "zero-norm point {} missing from all leaves",
                i
            );
        }
    }
}
