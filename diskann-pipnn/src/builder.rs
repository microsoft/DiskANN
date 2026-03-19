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
use crate::{PiPNNConfig, PiPNNError, PiPNNResult};

/// Ask glibc to return freed pages to the OS.
/// Without this, RSS stays inflated after large temporary allocations
/// (e.g. partition GEMM buffers) even though the memory is freed.
#[cfg(target_os = "linux")]
fn trim_heap() {
    unsafe {
        extern "C" { fn malloc_trim(pad: usize) -> i32; }
        malloc_trim(0);
    }
}

#[cfg(not(target_os = "linux"))]
fn trim_heap() {}


use diskann_vector::distance::{Distance, DistanceProvider, Metric};

/// Create a DiskANN distance functor for the given metric.
///
/// Uses the exact same SIMD-accelerated distance implementations as DiskANN:
/// - `L2` → `SquaredL2` (squared euclidean)
/// - `Cosine` → `Cosine` (normalizes + 1 - dot)
/// - `CosineNormalized` → `CosineNormalized` (1 - dot, assumes pre-normalized)
/// - `InnerProduct` → `InnerProduct` (-dot)
fn make_dist_fn(metric: Metric) -> Distance<f32, f32> {
    <f32 as DistanceProvider<f32>>::distance_comparer(metric, None)
}

/// The result of building a PiPNN index.
#[derive(Debug)]
pub struct PiPNNGraph {
    /// Adjacency lists: graph[i] contains the neighbor indices for point i.
    pub adjacency: Vec<Vec<u32>>,
    /// Number of points.
    pub npoints: usize,
    /// Number of dimensions.
    pub ndims: usize,
    /// Cached medoid (entry point for search).
    pub medoid: usize,
    /// Distance metric used to build this graph.
    pub metric: Metric,
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

    /// Save the graph in DiskANN's canonical graph format.
    ///
    /// Format:
    ///   Header (24 bytes):
    ///     - u64 LE: total file size (header + data)
    ///     - u32 LE: max degree (observed)
    ///     - u32 LE: start point ID (medoid)
    ///     - u64 LE: number of additional/frozen points
    ///   Per node:
    ///     - u32 LE: number of neighbors
    ///     - N x u32 LE: neighbor IDs
    pub fn save_graph(&self, path: &std::path::Path) -> PiPNNResult<()> {
        use std::io::{Write, Seek, SeekFrom, BufWriter};
        use std::fs::File;

        let mut f = BufWriter::new(File::create(path)?);

        let mut index_size: u64 = 24;
        let mut observed_max_degree: u32 = 0;
        let start_point = self.medoid as u32;

        // Write placeholder header
        f.write_all(&index_size.to_le_bytes())?;
        f.write_all(&observed_max_degree.to_le_bytes())?;
        f.write_all(&start_point.to_le_bytes())?;
        // Must be 1 to indicate the medoid is a frozen/start point.
        // The disk layout writer uses this to record the frozen point location.
        let num_additional: u64 = 1;
        f.write_all(&num_additional.to_le_bytes())?;

        // Write per-node adjacency lists
        for adj in &self.adjacency {
            let num_neighbors = adj.len() as u32;
            f.write_all(&num_neighbors.to_le_bytes())?;
            for &neighbor in adj {
                f.write_all(&neighbor.to_le_bytes())?;
            }
            observed_max_degree = observed_max_degree.max(num_neighbors);
            index_size += (4 + adj.len() * 4) as u64;
        }

        // Seek back and write correct header
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&index_size.to_le_bytes())?;
        f.write_all(&observed_max_degree.to_le_bytes())?;
        f.flush()?;

        tracing::info!(
            path = %path.display(),
            npoints = self.npoints,
            max_degree = observed_max_degree,
            start_point = start_point,
            "Saved PiPNN graph in DiskANN format"
        );

        Ok(())
    }

}

/// Search is only available for testing.
/// Production search goes through DiskANN's disk-based search pipeline.
#[cfg(test)]
impl PiPNNGraph {
    /// Perform greedy graph search starting from the cached medoid.
    ///
    /// This method is for testing and benchmarking only. Production search
    /// should use DiskANN's disk-based search pipeline which operates on the
    /// saved graph format.
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

        let dist_fn = make_dist_fn(self.metric);

        let start = self.medoid;

        // Greedy beam search.
        let l = search_list_size.max(k);
        let mut visited = vec![false; npoints];
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(l + 1);

        let start_dist = dist_fn.call(
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

                let dist = dist_fn.call(
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
///
/// Uses squared L2 distance to find the nearest point to the centroid,
/// matching DiskANN's `find_medoid_with_sampling` behavior. The centroid
/// is a geometric center, so L2 is the natural metric regardless of the
/// build distance metric.
fn find_medoid(data: &[f32], npoints: usize, ndims: usize) -> usize {
    let dist_fn = make_dist_fn(Metric::L2);

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

    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for i in 0..npoints {
        let point = &data[i * ndims..(i + 1) * ndims];
        let dist = dist_fn.call(point, &centroid);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    best_idx
}

/// Build a PiPNN index from typed vector data.
///
/// Converts input data to f32 before building (GEMM requires f32).
/// `data` is a flat slice of `T` in row-major order: npoints x ndims.
pub fn build_typed<T: diskann::utils::VectorRepr>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
) -> PiPNNResult<PiPNNGraph> {
    let expected_len = npoints * ndims;
    if data.len() != expected_len {
        return Err(PiPNNError::DataLengthMismatch {
            expected: expected_len,
            actual: data.len(),
            npoints,
            ndims,
        });
    }

    // Convert to f32 using VectorRepr::as_f32_into
    let mut f32_data = vec![0.0f32; expected_len];
    for i in 0..npoints {
        let src = &data[i * ndims..(i + 1) * ndims];
        let dst = &mut f32_data[i * ndims..(i + 1) * ndims];
        T::as_f32_into(src, dst).map_err(|e| PiPNNError::Config(format!("{}", e)))?;
    }

    build(&f32_data, npoints, ndims, config)
}

/// Build a PiPNN index.
///
/// `data` is row-major: npoints x ndims.
pub fn build(data: &[f32], npoints: usize, ndims: usize, config: &PiPNNConfig) -> PiPNNResult<PiPNNGraph> {
    config.validate()?;

    if npoints == 0 || ndims == 0 {
        return Err(PiPNNError::Config(
            "npoints and ndims must be > 0".into(),
        ));
    }

    if data.len() != npoints * ndims {
        return Err(PiPNNError::DataLengthMismatch {
            expected: npoints * ndims,
            actual: data.len(),
            npoints,
            ndims,
        });
    }

    tracing::info!(
        npoints = npoints,
        ndims = ndims,
        k = config.k,
        max_degree = config.max_degree,
        c_max = config.c_max,
        replicas = config.replicas,
        "PiPNN build started"
    );

    // The build() path always builds at full precision.
    // For quantized builds, use build_with_sq() which accepts pre-trained SQ params.
    build_internal(data, npoints, ndims, config, None)
}

/// Pre-trained scalar quantizer parameters for 1-bit quantization.
///
/// These can be extracted from DiskANN's trained `ScalarQuantizer` to ensure
/// identical quantization between Vamana and PiPNN builds.
pub struct SQParams {
    /// Per-dimension shift (length = ndims).
    pub shift: Vec<f32>,
    /// Global inverse scale (1.0 / scale).
    pub inverse_scale: f32,
}

/// Build a PiPNN index using a pre-trained scalar quantizer for 1-bit mode.
///
/// When DiskANN's build pipeline has already trained a `ScalarQuantizer`,
/// this function reuses those parameters instead of training from scratch.
/// This ensures identical quantization between Vamana and PiPNN builds.
///
/// `data` is row-major f32: npoints x ndims.
pub fn build_with_sq(
    data: &[f32],
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
    sq_params: &SQParams,
) -> PiPNNResult<PiPNNGraph> {
    config.validate()?;

    if data.len() != npoints * ndims {
        return Err(PiPNNError::DataLengthMismatch {
            expected: npoints * ndims,
            actual: data.len(),
            npoints,
            ndims,
        });
    }
    if npoints == 0 || ndims == 0 {
        return Err(PiPNNError::Config("npoints and ndims must be > 0".into()));
    }
    if sq_params.shift.len() != ndims {
        return Err(PiPNNError::DimensionMismatch {
            expected: ndims,
            actual: sq_params.shift.len(),
        });
    }

    tracing::info!(
        npoints = npoints,
        ndims = ndims,
        k = config.k,
        max_degree = config.max_degree,
        c_max = config.c_max,
        replicas = config.replicas,
        "PiPNN build started (with pre-trained SQ)"
    );

    // Quantize using pre-trained parameters.
    tracing::info!("Quantizing to 1-bit with pre-trained SQ params");
    let t = Instant::now();
    let qdata = crate::quantize::quantize_1bit(
        data,
        npoints,
        ndims,
        &sq_params.shift,
        sq_params.inverse_scale,
    );
    tracing::info!(
        elapsed_secs = t.elapsed().as_secs_f64(),
        bytes_per_vec = qdata.bytes_per_vec,
        "Quantization complete (pre-trained SQ)"
    );

    // Build using the internal build loop with pre-quantized data.
    build_internal(data, npoints, ndims, config, Some(qdata))
}

/// Internal build logic shared between `build()` and `build_with_sq()`.
fn build_internal(
    data: &[f32],
    npoints: usize,
    ndims: usize,
    config: &PiPNNConfig,
    qdata: Option<crate::quantize::QuantizedData>,
) -> PiPNNResult<PiPNNGraph> {
    // Compute medoid once upfront.
    let medoid = find_medoid(data, npoints, ndims);

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
    tracing::info!(elapsed_secs = t0.elapsed().as_secs_f64(), "HashPrune init complete");

    // Run multiple replicas of partitioning + leaf building.
    for replica in 0..config.replicas {
        let seed = 1000 + replica as u64 * 7919;

        let t1 = Instant::now();
        let partition_config = PartitionConfig {
            c_max: config.c_max,
            c_min: config.c_min,
            p_samp: config.p_samp,
            fanout: config.fanout.clone(),
            metric: config.metric,
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
        tracing::info!(
            replica = replica,
            partition_secs = partition_time.as_secs_f64(),
            num_leaves = leaves.len(),
            avg_leaf_size = total_pts as f64 / leaves.len().max(1) as f64,
            max_leaf_size = leaf_sizes.iter().max().unwrap_or(&0),
            total_pts = total_pts,
            "Partition complete"
        );
        // Return freed partition GEMM buffers to the OS so they don't inflate
        // peak RSS during the subsequent leaf build + reservoir filling phase.
        trim_heap();
        tracing::debug!(
            small_leaves = small_leaves,
            med_leaves = med_leaves,
            big_leaves = big_leaves,
            overlap = total_pts as f64 / npoints as f64,
            "Leaf size distribution"
        );

        // Build leaves in parallel, streaming edges to HashPrune per-leaf.
        let t2 = Instant::now();

        use std::sync::atomic::{AtomicUsize, Ordering};
        let total_edges = AtomicUsize::new(0);

        leaves.par_iter().for_each(|leaf| {
            let edges = if let Some(ref q) = qdata {
                leaf_build::build_leaf_quantized(q, &leaf.indices, config.k)
            } else {
                leaf_build::build_leaf(data, ndims, &leaf.indices, config.k, config.metric)
            };
            total_edges.fetch_add(edges.len(), Ordering::Relaxed);
            hash_prune.add_edges_batched(&edges);
        });

        tracing::info!(
            replica = replica,
            elapsed_secs = t2.elapsed().as_secs_f64(),
            total_edges = total_edges.load(Ordering::Relaxed),
            "Leaf build and merge complete"
        );
    }

    // Release thread-local leaf buffers so their arena pages can be reclaimed.
    // Without this, ~3 MB per rayon thread pins entire 64 MB glibc heap segments,
    // preventing ~500 MB of freed reservoir memory from being returned to the OS.
    (0..rayon::current_num_threads()).into_par_iter().for_each(|_| {
        leaf_build::release_thread_buffers();
    });

    // Extract final graph from HashPrune.
    let t3 = Instant::now();
    let adjacency = hash_prune.extract_graph();
    tracing::info!(elapsed_secs = t3.elapsed().as_secs_f64(), "Graph extraction complete");
    trim_heap();

    // Optional final prune pass.
    let adjacency = if config.final_prune {
        tracing::info!("Applying final prune");
        final_prune(data, ndims, &adjacency, config.max_degree, config.metric, config.alpha)
    } else {
        adjacency
    };

    let graph = PiPNNGraph {
        adjacency,
        npoints,
        ndims,
        medoid,
        metric: config.metric,
    };

    // Return all freed memory (reservoirs, sketches, partition buffers, leaf buffers)
    // to the OS before handing off to the disk layout phase.
    trim_heap();

    tracing::info!(
        avg_degree = graph.avg_degree(),
        max_degree = graph.max_degree(),
        isolated = graph.num_isolated(),
        "PiPNN build complete"
    );

    Ok(graph)
}

/// RobustPrune-like final pass: diversity-aware pruning via alpha-pruning.
/// Uses the same occlusion factor (alpha) as DiskANN's RobustPrune.
fn final_prune(
    data: &[f32],
    ndims: usize,
    adjacency: &[Vec<u32>],
    max_degree: usize,
    metric: Metric,
    alpha: f32,
) -> Vec<Vec<u32>> {
    let dist_fn = make_dist_fn(metric);

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
                    let dist = dist_fn.call(point_i, point_j);
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
                    let dist_sel_cand = dist_fn.call(point_sel, point_cand);
                    dist_sel_cand * alpha < cand_dist
                });

                if !is_pruned {
                    selected.push(cand_id);
                }
            }

            // Fill remaining from sorted list.
            if selected.len() < max_degree {
                let selected_set: std::collections::HashSet<u32> = selected.iter().copied().collect();
                for &(cand_id, _) in &candidates {
                    if selected.len() >= max_degree {
                        break;
                    }
                    if !selected_set.contains(&cand_id) {
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

        let graph = build(&data, npoints, ndims, &config).unwrap();

        assert_eq!(graph.npoints, npoints);
        assert!(graph.avg_degree() > 0.0);
        assert!(graph.num_isolated() < npoints);
    }

    #[test]
    fn test_build_data_length_mismatch() {
        let data = vec![0.0f32; 10];
        let config = PiPNNConfig::default();

        let result = build(&data, 5, 3, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, PiPNNError::DataLengthMismatch { .. }));
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

        let graph = build(&data, npoints, ndims, &config).unwrap();

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

        let graph = build(&data, npoints, ndims, &config).unwrap();

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

    #[test]
    fn test_config_validate() {
        let config = PiPNNConfig::default();
        assert!(config.validate().is_ok());

        let bad = PiPNNConfig { c_max: 0, ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig { c_min: 0, ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig { c_min: 2048, c_max: 1024, ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig { p_samp: 0.0, ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig { p_samp: 1.5, ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig { fanout: vec![], ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig { fanout: vec![0], ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig { num_hash_planes: 0, ..Default::default() };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig { num_hash_planes: 17, ..Default::default() };
        assert!(bad.validate().is_err());

    }

    #[test]
    fn test_config_validate_failures() {
        // max_degree = 0
        let bad = PiPNNConfig { max_degree: 0, ..Default::default() };
        assert!(bad.validate().is_err());

        // k = 0
        let bad = PiPNNConfig { k: 0, ..Default::default() };
        assert!(bad.validate().is_err());

        // replicas = 0
        let bad = PiPNNConfig { replicas: 0, ..Default::default() };
        assert!(bad.validate().is_err());

        // l_max = 0
        let bad = PiPNNConfig { l_max: 0, ..Default::default() };
        assert!(bad.validate().is_err());

        // p_samp exactly 1.0 is valid
        let ok = PiPNNConfig { p_samp: 1.0, ..Default::default() };
        assert!(ok.validate().is_ok());

        // num_hash_planes = 1 (boundary) is valid
        let ok = PiPNNConfig { num_hash_planes: 1, ..Default::default() };
        assert!(ok.validate().is_ok());

        // num_hash_planes = 16 (boundary) is valid
        let ok = PiPNNConfig { num_hash_planes: 16, ..Default::default() };
        assert!(ok.validate().is_ok());
    }

    #[test]
    fn test_build_cosine() {
        let npoints = 100;
        let ndims = 8;
        // Generate random data and normalize each vector for cosine.
        let mut data = generate_random_data(npoints, ndims, 42);
        for i in 0..npoints {
            let row = &mut data[i * ndims..(i + 1) * ndims];
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in row.iter_mut() { *v /= norm; }
            }
        }

        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            metric: diskann_vector::distance::Metric::Cosine,
            ..Default::default()
        };

        let graph = build(&data, npoints, ndims, &config).unwrap();
        assert!(matches!(graph.metric, Metric::Cosine));
        assert_eq!(graph.npoints, npoints);
        assert!(graph.avg_degree() > 0.0);
    }

    /// Train SQ parameters from data. Test-only helper.
    fn train_sq_params(data: &[f32], npoints: usize, ndims: usize) -> SQParams {
        use diskann_quantization::scalar::train::ScalarQuantizationParameters;
        use diskann_utils::views::MatrixView;

        let data_matrix = MatrixView::try_from(data, npoints, ndims)
            .expect("data length must equal npoints * ndims");
        let quantizer = ScalarQuantizationParameters::default().train(data_matrix);
        let shift = quantizer.shift().to_vec();
        let scale = quantizer.scale();
        let inverse_scale = if scale == 0.0 { 1.0 } else { 1.0 / scale };
        SQParams { shift, inverse_scale }
    }

    #[test]
    fn test_build_with_sq() {
        let npoints = 100;
        let ndims = 64; // must be multiple of 64 for u64 alignment in quantize
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

        let sq_params = train_sq_params(&data, npoints, ndims);

        let graph = super::build_with_sq(&data, npoints, ndims, &config, &sq_params).unwrap();
        assert_eq!(graph.npoints, npoints);
        assert!(graph.avg_degree() > 0.0);
    }

    #[test]
    fn test_build_typed_f32() {
        let npoints = 60;
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

        let graph_direct = build(&data, npoints, ndims, &config).unwrap();
        let graph_typed = build_typed::<f32>(&data, npoints, ndims, &config).unwrap();

        // Both should produce the same npoints and medoid.
        assert_eq!(graph_direct.npoints, graph_typed.npoints);
        assert_eq!(graph_direct.medoid, graph_typed.medoid);
    }

    #[test]
    fn test_save_graph_format() {
        let npoints = 50;
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

        let graph = build(&data, npoints, ndims, &config).unwrap();

        let dir = std::env::temp_dir().join("pipnn_test_save_graph");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_graph.bin");
        graph.save_graph(&path).unwrap();

        // Read back and verify the header.
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() >= 24, "file too small: {} bytes", bytes.len());

        // First 8 bytes: u64 LE file size.
        let file_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(file_size as usize, bytes.len(), "header file_size mismatch");

        // Bytes 8..12: u32 LE max degree.
        let max_deg = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(max_deg as usize, graph.max_degree());

        // Bytes 12..16: u32 LE start point (medoid).
        let start_pt = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(start_pt as usize, graph.medoid);

        // Clean up.
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_medoid_is_valid() {
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

        let graph = build(&data, npoints, ndims, &config).unwrap();
        assert!(
            graph.medoid < npoints,
            "medoid {} is out of range [0, {})",
            graph.medoid,
            npoints
        );
    }

    #[test]
    fn test_graph_connectivity() {
        // With sufficient replicas and params, no nodes should be isolated.
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

        let graph = build(&data, npoints, ndims, &config).unwrap();

        // With these settings no node should be completely isolated.
        assert_eq!(
            graph.num_isolated(), 0,
            "found {} isolated nodes with replicas=2",
            graph.num_isolated()
        );
    }

    #[test]
    fn test_build_zero_npoints() {
        let data: Vec<f32> = vec![];
        let config = PiPNNConfig::default();
        let result = build(&data, 0, 8, &config);
        assert!(result.is_err(), "npoints=0 should error");
    }

    #[test]
    fn test_build_zero_ndims() {
        let data: Vec<f32> = vec![];
        let config = PiPNNConfig::default();
        let result = build(&data, 10, 0, &config);
        assert!(result.is_err(), "ndims=0 should error");
    }

    #[test]
    fn test_build_single_point() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 1,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, 1, 4, &config).unwrap();
        assert_eq!(graph.npoints, 1, "should have 1 point");
        assert_eq!(graph.adjacency[0].len(), 0, "single point should have 0 edges");
    }

    #[test]
    fn test_build_two_points() {
        let data = vec![0.0f32, 0.0, 1.0, 0.0];
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 1,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, 2, 2, &config).unwrap();
        assert_eq!(graph.npoints, 2, "should have 2 points");
        // With 2 points, they should connect to each other.
        let total_edges: usize = graph.adjacency.iter().map(|a| a.len()).sum();
        assert!(total_edges > 0, "two points should have at least one edge between them");
    }

    #[test]
    fn test_build_duplicate_points() {
        // All identical points; build should still succeed.
        let npoints = 20;
        let ndims = 4;
        let data = vec![1.0f32; npoints * ndims];
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 4,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();
        assert_eq!(graph.npoints, npoints, "should build successfully with duplicate points");
    }

    #[test]
    fn test_build_very_small_k() {
        let npoints = 50;
        let ndims = 4;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 1,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();
        assert_eq!(graph.npoints, npoints, "k=1 should produce valid graph");
        assert!(graph.avg_degree() > 0.0, "k=1 should still produce some edges");
    }

    #[test]
    fn test_build_k_larger_than_leaf() {
        // k > c_max should still work (clamped inside extract_knn).
        let npoints = 50;
        let ndims = 4;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 100, // larger than c_max
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();
        assert_eq!(graph.npoints, npoints, "k > c_max should still produce valid graph");
    }

    #[test]
    fn test_search_empty_graph() {
        let graph = PiPNNGraph {
            adjacency: vec![],
            npoints: 0,
            ndims: 4,
            medoid: 0,
            metric: Metric::L2,
        };
        let query = vec![1.0f32, 2.0, 3.0, 4.0];
        let results = graph.search(&[], &query, 5, 10);
        assert!(results.is_empty(), "search on empty graph should return empty results");
    }

    #[test]
    fn test_search_k_larger_than_npoints() {
        let npoints = 10;
        let ndims = 4;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 4,
            k: 3,
            max_degree: 16,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();
        let query = &data[0..ndims];
        // Request more neighbors than points exist.
        let results = graph.search(&data, query, 100, 200);
        assert!(
            results.len() <= npoints,
            "should not return more results than npoints, got {}",
            results.len()
        );
    }

    #[test]
    fn test_search_with_self_query() {
        let npoints = 100;
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
        let graph = build(&data, npoints, ndims, &config).unwrap();
        // Query with the medoid point itself.
        let medoid = graph.medoid;
        let query = &data[medoid * ndims..(medoid + 1) * ndims];
        let results = graph.search(&data, query, 5, 50);
        assert!(!results.is_empty(), "search should return at least one result");
        assert_eq!(
            results[0].0, medoid,
            "searching with a data point should find itself first"
        );
        assert!(
            results[0].1 < 1e-6,
            "self-distance should be near zero, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_search_different_l_values() {
        use crate::leaf_build::brute_force_knn;

        let npoints = 300;
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
        let graph = build(&data, npoints, ndims, &config).unwrap();

        let k = 10;
        let query = &data[0..ndims];
        let exact = brute_force_knn(&data, ndims, npoints, query, k);
        let exact_set: std::collections::HashSet<usize> =
            exact.iter().map(|&(id, _)| id).collect();

        // Compare recall for small L vs large L.
        let results_small_l = graph.search(&data, query, k, k);
        let recall_small: f64 = results_small_l
            .iter()
            .filter(|&&(id, _)| exact_set.contains(&id))
            .count() as f64
            / k as f64;

        let results_large_l = graph.search(&data, query, k, 200);
        let recall_large: f64 = results_large_l
            .iter()
            .filter(|&&(id, _)| exact_set.contains(&id))
            .count() as f64
            / k as f64;

        assert!(
            recall_large >= recall_small,
            "larger L ({:.4}) should give recall >= smaller L ({:.4})",
            recall_large,
            recall_small
        );
    }

    #[test]
    fn test_build_with_sq_wrong_shift_dims() {
        let npoints = 50;
        let ndims = 64;
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
        // Shift length != ndims.
        let sq_params = SQParams {
            shift: vec![0.0f32; ndims + 5], // wrong length
            inverse_scale: 1.0,
        };
        let result = build_with_sq(&data, npoints, ndims, &config, &sq_params);
        assert!(
            result.is_err(),
            "shift length != ndims should produce an error"
        );
        assert!(
            matches!(result.unwrap_err(), PiPNNError::DimensionMismatch { .. }),
            "should be a DimensionMismatch error"
        );
    }

    #[test]
    fn test_build_with_sq_produces_connected_graph() {
        let npoints = 100;
        let ndims = 64;
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
        let sq_params = train_sq_params(&data, npoints, ndims);
        let graph = build_with_sq(&data, npoints, ndims, &config, &sq_params).unwrap();
        assert_eq!(
            graph.num_isolated(), 0,
            "build_with_sq should produce a connected graph with sufficient replicas, found {} isolated nodes",
            graph.num_isolated()
        );
    }

    #[test]
    fn test_build_typed_data_length_mismatch() {
        let data = vec![1.0f32; 30]; // 30 elements
        let config = PiPNNConfig::default();
        // npoints=5, ndims=8 expects 40 elements but data has 30.
        let result = build_typed::<f32>(&data, 5, 8, &config);
        assert!(
            result.is_err(),
            "data length mismatch should produce an error"
        );
    }

    #[test]
    fn test_save_graph_single_node() {
        let graph = PiPNNGraph {
            adjacency: vec![vec![]],
            npoints: 1,
            ndims: 4,
            medoid: 0,
            metric: Metric::L2,
        };
        let dir = std::env::temp_dir().join("pipnn_test_save_single");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("single.bin");
        graph.save_graph(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() >= 24, "file too small for single node graph");
        let file_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(file_size as usize, bytes.len(), "header file_size mismatch for single node");

        // Max degree should be 0 for single node with no edges.
        let max_deg = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(max_deg, 0, "single node with no edges should have max_degree=0");

        // Read back neighbor count for the single node.
        let num_neighbors = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        assert_eq!(num_neighbors, 0, "single node should have 0 neighbors");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_save_graph_large() {
        let npoints = 1000;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);
        let config = PiPNNConfig {
            c_max: 128,
            c_min: 32,
            k: 4,
            max_degree: 32,
            replicas: 1,
            l_max: 64,
            ..Default::default()
        };
        let graph = build(&data, npoints, ndims, &config).unwrap();

        let dir = std::env::temp_dir().join("pipnn_test_save_large");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("large.bin");
        graph.save_graph(&path).unwrap();

        // Read back and verify we can parse all adjacency lists.
        let bytes = std::fs::read(&path).unwrap();
        let file_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(file_size as usize, bytes.len(), "header file_size mismatch for large graph");

        let mut offset = 24usize;
        let mut total_parsed_nodes = 0usize;
        while offset < bytes.len() {
            let num_neighbors = u32::from_le_bytes(
                bytes[offset..offset + 4].try_into().unwrap()
            ) as usize;
            offset += 4;
            for _ in 0..num_neighbors {
                let neighbor = u32::from_le_bytes(
                    bytes[offset..offset + 4].try_into().unwrap()
                ) as usize;
                assert!(
                    neighbor < npoints,
                    "neighbor index {} out of range for node {}",
                    neighbor, total_parsed_nodes
                );
                offset += 4;
            }
            total_parsed_nodes += 1;
        }
        assert_eq!(
            total_parsed_nodes, npoints,
            "expected to parse {} nodes but got {}",
            npoints, total_parsed_nodes
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_config_c_min_greater_than_c_max() {
        let config = PiPNNConfig {
            c_min: 2048,
            c_max: 1024,
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "c_min > c_max should fail validation"
        );
    }

    #[test]
    fn test_config_empty_fanout() {
        let config = PiPNNConfig {
            fanout: vec![],
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "empty fanout should fail validation"
        );
    }

    #[test]
    fn test_config_zero_fanout_element() {
        let config = PiPNNConfig {
            fanout: vec![5, 0, 2],
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "fanout containing 0 should fail validation"
        );
    }

    #[test]
    fn test_config_p_samp_zero() {
        let config = PiPNNConfig {
            p_samp: 0.0,
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "p_samp=0.0 should fail validation"
        );
    }

    #[test]
    fn test_config_p_samp_negative() {
        let config = PiPNNConfig {
            p_samp: -0.5,
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "p_samp < 0 should fail validation"
        );
    }

    #[test]
    fn test_config_hash_planes_zero() {
        let config = PiPNNConfig {
            num_hash_planes: 0,
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "num_hash_planes=0 should fail validation"
        );
    }

    #[test]
    fn test_config_hash_planes_17() {
        let config = PiPNNConfig {
            num_hash_planes: 17,
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "num_hash_planes=17 (> 16) should fail validation"
        );
    }

    #[test]
    fn test_final_prune_reduces_degree() {
        let npoints = 200;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        // Build without final prune, then build with, and compare max degree.
        let config_no_prune = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 6,
            max_degree: 16,
            replicas: 2,
            l_max: 64,
            final_prune: false,
            ..Default::default()
        };
        let config_with_prune = PiPNNConfig {
            final_prune: true,
            ..config_no_prune.clone()
        };

        let graph_no = build(&data, npoints, ndims, &config_no_prune).unwrap();
        let graph_yes = build(&data, npoints, ndims, &config_with_prune).unwrap();

        // Final prune should not increase max degree beyond max_degree.
        assert!(
            graph_yes.max_degree() <= config_with_prune.max_degree,
            "final_prune max_degree {} > config max_degree {}",
            graph_yes.max_degree(),
            config_with_prune.max_degree
        );

        // Both should be valid graphs.
        assert!(graph_no.avg_degree() > 0.0);
        assert!(graph_yes.avg_degree() > 0.0);
    }
}
