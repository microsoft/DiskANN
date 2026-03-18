/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Main PiPNN builder: orchestrates partitioning, leaf building, and edge merging.
//!
//! Algorithm (from arXiv:2602.21247):
//! 1. G <- empty graph
//! 2. B <- Partition(X) via RBC
//! 3. For each leaf b_i in B (in parallel):
//!      edges <- Pick(b_i)  // GEMM + bi-directed k-NN
//!      G.Prune_And_Add_Edges(edges)  // stream to HashPrune
//! 4. Optional: final RobustPrune on each node
//! 5. return G

use std::time::Instant;

use rayon::prelude::*;

use crate::hash_prune::HashPrune;
use crate::leaf_build;
use crate::partition::{self, PartitionConfig};
use crate::PiPNNConfig;

/// L2 squared distance.
#[inline]
fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    use diskann_vector::PureDistanceFunction;
    use diskann_vector::distance::SquaredL2;
    SquaredL2::evaluate(a, b)
}

/// Cosine distance for normalized vectors: 1 - dot(a, b).
#[inline]
fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        unsafe { dot += *a.get_unchecked(i) * *b.get_unchecked(i); }
    }
    (1.0 - dot).max(0.0)
}

/// The result of building a PiPNN index.
pub struct PiPNNGraph {
    /// Adjacency lists: graph[i] contains the neighbor indices for point i.
    pub adjacency: Vec<Vec<u32>>,
    /// Number of points.
    pub npoints: usize,
    /// Number of dimensions.
    pub ndims: usize,
    /// Cached medoid (entry point for search).
    pub medoid: usize,
    /// Whether to use cosine distance (1 - dot) instead of L2.
    pub use_cosine: bool,
}

impl PiPNNGraph {
    /// Get neighbors of a point.
    pub fn neighbors(&self, idx: usize) -> &[u32] {
        &self.adjacency[idx]
    }

    /// Get the average out-degree.
    pub fn avg_degree(&self) -> f64 {
        let total: usize = self.adjacency.iter().map(|adj| adj.len()).sum();
        total as f64 / self.npoints as f64
    }

    /// Get the max out-degree.
    pub fn max_degree(&self) -> usize {
        self.adjacency.iter().map(|adj| adj.len()).max().unwrap_or(0)
    }

    /// Count the number of points with zero out-degree.
    pub fn num_isolated(&self) -> usize {
        self.adjacency.iter().filter(|adj| adj.is_empty()).count()
    }

    /// Perform greedy graph search starting from the cached medoid.
    ///
    /// Returns the indices and distances of the `k` approximate nearest neighbors.
    pub fn search(
        &self,
        data: &[f32],
        query: &[f32],
        k: usize,
        search_list_size: usize,
    ) -> Vec<(usize, f32)> {
        let ndims = self.ndims;
        let npoints = self.npoints;

        if npoints == 0 {
            return Vec::new();
        }

        let dist_fn = if self.use_cosine {
            cosine_dist
        } else {
            l2_dist
        };

        let start = self.medoid;

        // Greedy beam search.
        let l = search_list_size.max(k);
        let mut visited = vec![false; npoints];
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(l + 1);

        let start_dist = dist_fn(
            &data[start * ndims..(start + 1) * ndims],
            query,
        );
        candidates.push((start, start_dist));
        visited[start] = true;

        let mut pointer = 0;

        while pointer < candidates.len() {
            let (current, _) = candidates[pointer];
            pointer += 1;

            for &neighbor in &self.adjacency[current] {
                let neighbor = neighbor as usize;
                if neighbor >= npoints || visited[neighbor] {
                    continue;
                }
                visited[neighbor] = true;

                let dist = dist_fn(
                    &data[neighbor * ndims..(neighbor + 1) * ndims],
                    query,
                );

                if candidates.len() < l || dist < candidates.last().map(|c| c.1).unwrap_or(f32::MAX) {
                    let pos = candidates
                        .binary_search_by(|c| {
                            c.1.partial_cmp(&dist).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or_else(|e| e);
                    candidates.insert(pos, (neighbor, dist));
                    if candidates.len() > l {
                        candidates.truncate(l);
                    }

                    if pos < pointer {
                        pointer = pos;
                    }
                }
            }
        }

        candidates.truncate(k);
        candidates
    }
}

/// Find the medoid: the point closest to the centroid.
fn find_medoid(data: &[f32], npoints: usize, ndims: usize, use_cosine: bool) -> usize {
    let dist_fn = if use_cosine { cosine_dist } else { l2_dist };

    // Compute centroid.
    let mut centroid = vec![0.0f32; ndims];
    for i in 0..npoints {
        let point = &data[i * ndims..(i + 1) * ndims];
        for d in 0..ndims {
            centroid[d] += point[d];
        }
    }
    let inv_n = 1.0 / npoints as f32;
    for d in 0..ndims {
        centroid[d] *= inv_n;
    }

    // For cosine, normalize the centroid.
    if use_cosine {
        let norm: f32 = centroid.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for d in 0..ndims { centroid[d] /= norm; }
        }
    }

    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for i in 0..npoints {
        let point = &data[i * ndims..(i + 1) * ndims];
        let dist = dist_fn(point, &centroid);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    best_idx
}

/// Build a PiPNN index.
///
/// `data` is row-major: npoints x ndims.
pub fn build(data: &[f32], npoints: usize, ndims: usize, config: &PiPNNConfig) -> PiPNNGraph {
    assert_eq!(data.len(), npoints * ndims, "data length mismatch");

    eprintln!(
        "PiPNN build: {} points x {} dims, k={}, R={}, c_max={}, replicas={}",
        npoints, ndims, config.k, config.max_degree, config.c_max, config.replicas
    );

    let use_cosine = config.metric == diskann_vector::distance::Metric::CosineNormalized
        || config.metric == diskann_vector::distance::Metric::Cosine;

    // Optionally quantize data to 1-bit for faster build.
    let qdata = if config.quantize_bits == Some(1) {
        eprintln!("  Quantizing to 1-bit...");
        let t = Instant::now();
        let q = crate::quantize::quantize_1bit(data, npoints, ndims);
        eprintln!("  Quantization: {:.3}s ({} bytes/vec)", t.elapsed().as_secs_f64(), q.bytes_per_vec);
        Some(q)
    } else {
        None
    };

    // Compute medoid once upfront.
    let medoid = find_medoid(data, npoints, ndims, use_cosine);

    // Initialize HashPrune for edge merging.
    let t0 = Instant::now();
    let hash_prune = HashPrune::new(
        data,
        npoints,
        ndims,
        config.num_hash_planes,
        config.l_max,
        config.max_degree,
        42,
    );
    eprintln!("  HashPrune init: {:.3}s", t0.elapsed().as_secs_f64());

    // Run multiple replicas of partitioning + leaf building.
    for replica in 0..config.replicas {
        let seed = 1000 + replica as u64 * 7919;

        let t1 = Instant::now();
        let partition_config = PartitionConfig {
            c_max: config.c_max,
            c_min: config.c_min,
            p_samp: config.p_samp,
            fanout: config.fanout.clone(),
        };

        let indices: Vec<usize> = (0..npoints).collect();
        let leaves = if let Some(ref q) = qdata {
            partition::parallel_partition_quantized(q, &indices, &partition_config, seed)
        } else {
            partition::parallel_partition(data, ndims, &indices, &partition_config, seed)
        };
        let partition_time = t1.elapsed();

        let total_pts: usize = leaves.iter().map(|l| l.indices.len()).sum();
        let leaf_sizes: Vec<usize> = leaves.iter().map(|l| l.indices.len()).collect();
        let small_leaves = leaf_sizes.iter().filter(|&&s| s < 64).count();
        let med_leaves = leaf_sizes.iter().filter(|&&s| s >= 64 && s < 512).count();
        let big_leaves = leaf_sizes.iter().filter(|&&s| s >= 512).count();
        eprintln!(
            "  Replica {}: partition {:.3}s, {} leaves (avg {:.1}, max {}, total_pts {})",
            replica,
            partition_time.as_secs_f64(),
            leaves.len(),
            total_pts as f64 / leaves.len().max(1) as f64,
            leaf_sizes.iter().max().unwrap_or(&0),
            total_pts,
        );
        eprintln!(
            "    leaf size distribution: <64: {}, 64-512: {}, 512+: {}, overlap: {:.1}x",
            small_leaves, med_leaves, big_leaves,
            total_pts as f64 / npoints as f64,
        );

        // Build leaves in parallel, streaming edges to HashPrune per-leaf.
        let t2 = Instant::now();
        let use_cosine = config.metric == diskann_vector::distance::Metric::CosineNormalized
            || config.metric == diskann_vector::distance::Metric::Cosine;

        use std::sync::atomic::{AtomicUsize, Ordering};
        let total_edges = AtomicUsize::new(0);

        leaves.par_iter().for_each(|leaf| {
            let edges = if let Some(ref q) = qdata {
                leaf_build::build_leaf_quantized(q, &leaf.indices, config.k)
            } else {
                leaf_build::build_leaf(data, ndims, &leaf.indices, config.k, use_cosine)
            };
            total_edges.fetch_add(edges.len(), Ordering::Relaxed);
            hash_prune.add_edges_batched(&edges);
        });

        eprintln!(
            "  Replica {}: leaf+merge wall {:.3}s, {} edges",
            replica,
            t2.elapsed().as_secs_f64(),
            total_edges.load(Ordering::Relaxed),
        );
    }

    // Extract final graph from HashPrune.
    let t3 = Instant::now();
    let adjacency = hash_prune.extract_graph();
    eprintln!("  Extract graph: {:.3}s", t3.elapsed().as_secs_f64());

    // Optional final prune pass.
    let adjacency = if config.final_prune {
        eprintln!("  Applying final prune...");
        final_prune(data, ndims, &adjacency, config.max_degree, use_cosine)
    } else {
        adjacency
    };

    let graph = PiPNNGraph {
        adjacency,
        npoints,
        ndims,
        medoid,
        use_cosine,
    };

    eprintln!(
        "PiPNN build complete: avg_degree={:.1}, max_degree={}, isolated={}",
        graph.avg_degree(),
        graph.max_degree(),
        graph.num_isolated()
    );

    graph
}

/// RobustPrune-like final pass: diversity-aware pruning via alpha-pruning.
fn final_prune(
    data: &[f32],
    ndims: usize,
    adjacency: &[Vec<u32>],
    max_degree: usize,
    use_cosine: bool,
) -> Vec<Vec<u32>> {
    let dist_fn = if use_cosine { cosine_dist } else { l2_dist };
    let alpha = 1.2f32;

    adjacency
        .par_iter()
        .enumerate()
        .map(|(i, neighbors)| {
            if neighbors.len() <= max_degree {
                return neighbors.clone();
            }

            let point_i = &data[i * ndims..(i + 1) * ndims];

            // Compute distances from i to all its current neighbors.
            let mut candidates: Vec<(u32, f32)> = neighbors
                .iter()
                .map(|&j| {
                    let point_j = &data[j as usize * ndims..(j as usize + 1) * ndims];
                    let dist = dist_fn(point_i, point_j);
                    (j, dist)
                })
                .collect();

            candidates.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Greedy diversity-aware selection.
            let mut selected: Vec<u32> = Vec::with_capacity(max_degree);

            for &(cand_id, cand_dist) in &candidates {
                if selected.len() >= max_degree {
                    break;
                }

                let is_pruned = selected.iter().any(|&sel_id| {
                    let point_sel =
                        &data[sel_id as usize * ndims..(sel_id as usize + 1) * ndims];
                    let point_cand =
                        &data[cand_id as usize * ndims..(cand_id as usize + 1) * ndims];
                    let dist_sel_cand = dist_fn(point_sel, point_cand);
                    dist_sel_cand * alpha < cand_dist
                });

                if !is_pruned {
                    selected.push(cand_id);
                }
            }

            // Fill remaining from sorted list.
            if selected.len() < max_degree {
                for &(cand_id, _) in &candidates {
                    if selected.len() >= max_degree {
                        break;
                    }
                    if !selected.contains(&cand_id) {
                        selected.push(cand_id);
                    }
                }
            }

            selected
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_data(npoints: usize, ndims: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..npoints * ndims)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect()
    }

    #[test]
    fn test_build_small() {
        let npoints = 100;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };

        let graph = build(&data, npoints, ndims, &config);

        assert_eq!(graph.npoints, npoints);
        assert!(graph.avg_degree() > 0.0);
        assert!(graph.num_isolated() < npoints);
    }

    #[test]
    fn test_search_basic() {
        let npoints = 200;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 4,
            max_degree: 32,
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };

        let graph = build(&data, npoints, ndims, &config);

        let query = &data[0..ndims];
        let results = graph.search(&data, query, 10, 50);

        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn test_recall() {
        use crate::leaf_build::brute_force_knn;

        let npoints = 500;
        let ndims = 16;
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 128,
            c_min: 32,
            k: 4,
            max_degree: 32,
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };

        let graph = build(&data, npoints, ndims, &config);

        let k = 10;
        let search_l = 100;
        let num_queries = 20;

        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query: Vec<f32> = (0..ndims).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();

            let approx = graph.search(&data, &query, k, search_l);
            let exact = brute_force_knn(&data, ndims, npoints, &query, k);

            let exact_set: std::collections::HashSet<usize> =
                exact.iter().map(|&(id, _)| id).collect();
            let recall = approx
                .iter()
                .filter(|&&(id, _)| exact_set.contains(&id))
                .count() as f64
                / k as f64;

            total_recall += recall;
        }

        let avg_recall = total_recall / num_queries as f64;
        eprintln!("Average recall@{}: {:.4}", k, avg_recall);

        assert!(
            avg_recall > 0.2,
            "recall too low: {:.4}",
            avg_recall
        );
    }
}
