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
//!    edges <- Pick(b_i)  // GEMM + bi-directed k-NN
//!    G.Prune_And_Add_Edges(edges)  // stream to HashPrune
//! 4. Optional: final diversity prune on each node
//! 5. return G

use std::time::Instant;

use diskann::utils::VectorRepr;
use diskann_providers::utils::MAX_MEDOID_SAMPLE_SIZE;
use rayon::prelude::*;

use crate::hash_prune::HashPrune;
use crate::leaf_build;
use crate::partition::PartitionConfig;
use crate::cpu_dispatch::{tier, SimdTier};
use crate::rayon_util::ParIterInstalled;
use crate::{PiPNNBuildContext, PiPNNError, PiPNNResult};

use diskann_vector::distance::{Distance, DistanceProvider, Metric};

/// Create a DiskANN distance functor for the given metric.
///
/// Uses the exact same SIMD-accelerated distance implementations as DiskANN:
/// - `L2` → `SquaredL2` (squared euclidean)
/// - `Cosine` → `Cosine` (normalizes + 1 - dot)
/// - `CosineNormalized` → `CosineNormalized` (1 - dot, assumes pre-normalized)
/// - `InnerProduct` → `InnerProduct` (-dot)
fn make_dist_fn(metric: Metric, ndims: usize) -> Distance<f32, f32> {
    <f32 as DistanceProvider<f32>>::distance_comparer(metric, Some(ndims))
}

/// Log which SIMD tier the hand-written kernels in partition / leaf_build /
/// hash_prune will dispatch to at runtime.
///
/// The tier is selected by [`crate::cpu_dispatch::tier`] based on the host
/// CPU's `is_x86_feature_detected!` results, not the compile-time
/// `target_feature` flags — so a single binary built against the workspace
/// `target-cpu=x86-64-v3` floor still picks up the AVX-512 paths on hosts
/// that support them.
fn log_simd_tier() {
    let tier = match crate::cpu_dispatch::tier() {
        crate::cpu_dispatch::SimdTier::Avx512 => "AVX-512",
        crate::cpu_dispatch::SimdTier::Avx2 => "AVX2",
        crate::cpu_dispatch::SimdTier::Scalar => {
            if cfg!(target_arch = "x86_64") {
                "scalar"
            } else {
                "scalar (non-x86)"
            }
        }
    };
    tracing::info!(simd_tier = tier, "PiPNN SIMD tier");
}

/// Timing breakdown for the PiPNN build phases.
#[derive(Debug, Clone, Default)]
pub struct PiPNNBuildStats {
    pub total_secs: f64,
    pub sketch_secs: f64,
    pub partition_secs: f64,
    pub leaf_build_secs: f64,
    pub extract_secs: f64,
    pub final_prune_secs: f64,
    pub num_leaves: usize,
    pub total_edges: usize,
}

impl std::fmt::Display for PiPNNBuildStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PiPNN Build Timing")?;
        writeln!(f, "  LSH sketches:   {:.3}s", self.sketch_secs)?;
        writeln!(
            f,
            "  Partition:      {:.3}s  ({} leaves)",
            self.partition_secs, self.num_leaves
        )?;
        writeln!(
            f,
            "  Leaf build:     {:.3}s  ({} edges)",
            self.leaf_build_secs, self.total_edges
        )?;
        writeln!(f, "  Graph extract:  {:.3}s", self.extract_secs)?;
        writeln!(f, "  Final prune:    {:.3}s", self.final_prune_secs)?;
        writeln!(f, "  Total:          {:.3}s", self.total_secs)
    }
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
    /// Build timing breakdown.
    pub build_stats: PiPNNBuildStats,
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
        self.adjacency
            .iter()
            .map(|adj| adj.len())
            .max()
            .unwrap_or(0)
    }

    /// Count the number of points with zero out-degree.
    pub fn num_isolated(&self) -> usize {
        self.adjacency.iter().filter(|adj| adj.is_empty()).count()
    }

    /// Save the graph in DiskANN's canonical graph format.
    ///
    /// Delegates to [`diskann_providers::storage::bin::save_graph_to_writer`]
    /// — pipnn doesn't re-implement the file format, it just satisfies the
    /// [`GetAdjacencyList`](diskann_providers::storage::bin::GetAdjacencyList)
    /// trait that the workspace serializer expects.
    pub fn save_graph(&self, path: &std::path::Path) -> PiPNNResult<()> {
        let file = std::fs::File::create(path)?;
        diskann_providers::storage::bin::save_graph_to_writer(self, self.medoid as u32, file)
            .map_err(|e| PiPNNError::Config(format!("save_graph failed: {}", e)))?;

        tracing::info!(
            path = %path.display(),
            npoints = self.npoints,
            start_point = self.medoid as u32,
            "Saved PiPNN graph in DiskANN format"
        );
        Ok(())
    }
}

/// Hook into the workspace graph serializer. Pipnn appends one frozen
/// start-point node after the `npoints` real nodes; the frozen node duplicates
/// the medoid's adjacency list (DiskANN's loader expects exactly
/// `additional_points()` such nodes after the real ones).
impl diskann_providers::storage::bin::GetAdjacencyList for PiPNNGraph {
    type Element = u32;
    type Item<'a> = &'a [u32];

    fn get_adjacency_list(&self, i: usize) -> diskann::ANNResult<Self::Item<'_>> {
        // Index `npoints` is the synthetic frozen start point.
        let idx = if i == self.npoints { self.medoid } else { i };
        Ok(self.adjacency[idx].as_slice())
    }

    fn total(&self) -> usize {
        self.npoints + 1 // +1 for the frozen start point
    }

    fn additional_points(&self) -> u64 {
        1
    }

    fn max_degree(&self) -> Option<u32> {
        None // use observed max
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

        let dist_fn = make_dist_fn(self.metric, ndims);

        let start = self.medoid;

        // Greedy best-first search with a sorted candidate list of size L.
        let l = search_list_size.max(k);
        let mut visited = vec![false; npoints];
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(l + 1);

        let start_dist = dist_fn.call(&data[start * ndims..(start + 1) * ndims], query);
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

                let dist = dist_fn.call(&data[neighbor * ndims..(neighbor + 1) * ndims], query);

                if candidates.len() < l || dist < candidates.last().map(|c| c.1).unwrap_or(f32::MAX)
                {
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

/// Find the medoid (point closest to the centroid) using squared L2.
///
/// Centroid is averaged over a stride-sample capped at
/// [`MAX_MEDOID_SAMPLE_SIZE`]; argmin scans the full dataset, like
/// [`diskann_providers::utils::find_medoid_with_sampling`].
fn find_medoid<T: VectorRepr>(data: &[T], npoints: usize, ndims: usize) -> PiPNNResult<usize> {
    let dist_fn = make_dist_fn(Metric::L2, ndims);
    let convert_err =
        |e: T::Error| PiPNNError::Config(format!("find_medoid: vector conversion failed: {}", e));

    let stride = npoints.div_ceil(MAX_MEDOID_SAMPLE_SIZE).max(1);
    let mut centroid = vec![0.0f32; ndims];
    let mut point_buf = vec![0.0f32; ndims];
    let mut sample_count = 0usize;
    for i in (0..npoints).step_by(stride) {
        T::as_f32_into(&data[i * ndims..(i + 1) * ndims], &mut point_buf).map_err(convert_err)?;
        for d in 0..ndims {
            centroid[d] += point_buf[d];
        }
        sample_count += 1;
    }
    let inv_n = 1.0 / sample_count as f32;
    for c in &mut centroid {
        *c *= inv_n;
    }

    // Parallel argmin: each thread converts its slice + finds local best, then reduce.
    use rayon::prelude::*;
    let (best_idx, _best_dist) = (0..npoints)
        .into_par_iter()
        .fold(
            || (usize::MAX, f32::MAX),
            |(bi, bd), i| {
                let mut buf = vec![0.0f32; ndims];
                if T::as_f32_into(&data[i * ndims..(i + 1) * ndims], &mut buf).is_err() {
                    return (bi, bd);
                }
                let d = dist_fn.call(&buf, &centroid);
                if d < bd { (i, d) } else { (bi, bd) }
            },
        )
        .reduce(|| (usize::MAX, f32::MAX), |a, b| if a.1 <= b.1 { a } else { b });

    Ok(best_idx)
}

/// Build a PiPNN index from typed vector data.
///
/// Keeps data in its native type T and converts to f32 on-the-fly at each access point,
/// avoiding a full f32 copy of the dataset.
/// `data` is a flat slice of `T` in row-major order: npoints x ndims.
pub fn build_typed<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    ctx: &PiPNNBuildContext,
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

    if npoints == 0 || ndims == 0 {
        return Err(PiPNNError::Config("npoints and ndims must be > 0".into()));
    }

    let config = ctx.config();
    tracing::info!(
        npoints = npoints,
        ndims = ndims,
        k = config.k,
        max_degree = ctx.max_degree().get(),
        c_max = config.c_max,
        replicas = config.replicas,
        "PiPNN build started (typed)"
    );

    build_internal(data, npoints, ndims, ctx)
}

/// Internal build logic shared between entry points.
fn build_internal<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    ctx: &PiPNNBuildContext,
) -> PiPNNResult<PiPNNGraph> {
    // Respect num_threads: install a scoped rayon pool so all par_iter() calls
    // within this build use the configured thread count instead of all cores.
    if ctx.num_threads() > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(ctx.num_threads())
            .build()
            .map_err(|e| PiPNNError::Config(format!("Failed to create thread pool: {}", e)))?;
        return pool.install(|| build_internal_impl(data, npoints, ndims, ctx));
    }
    build_internal_impl(data, npoints, ndims, ctx)
}

fn build_internal_impl<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    ctx: &PiPNNBuildContext,
) -> PiPNNResult<PiPNNGraph> {
    let config = ctx.config();
    let metric = ctx.metric();
    let max_degree = ctx.max_degree().get();
    let t_total = Instant::now();

    log_simd_tier();

    // Compute medoid once upfront.
    let medoid = find_medoid(data, npoints, ndims)?;

    // Initialize HashPrune for edge merging.
    let t0 = Instant::now();
    let hash_prune = HashPrune::new(
        data,
        npoints,
        ndims,
        config.num_hash_planes,
        config.l_max,
        max_degree,
        42,
    );
    let sketch_secs = t0.elapsed().as_secs_f64();
    tracing::info!(elapsed_secs = sketch_secs, "HashPrune init complete");

    // Run multiple replicas of partitioning + leaf building.
    let mut partition_secs = 0.0f64;
    let mut leaf_build_secs = 0.0f64;
    let mut total_leaves = 0usize;
    let mut total_edges_count = 0usize;

    for replica in 0..config.replicas {
        let seed = 1000 + replica as u64 * 7919;

        let t1 = Instant::now();
        let partition_config = PartitionConfig::new(
            config.c_max,
            config.c_min,
            config.p_samp,
            config.fanout.clone(),
            metric,
            config.leader_cap,
        )?;

        let leaves = crate::partition::partition(data, ndims, npoints, &partition_config, seed);
        partition_secs += t1.elapsed().as_secs_f64();

        let total_pts: usize = leaves.iter().map(|l| l.indices.len()).sum();
        let leaf_sizes: Vec<usize> = leaves.iter().map(|l| l.indices.len()).collect();
        total_leaves += leaves.len();
        let small_leaves = leaf_sizes.iter().filter(|&&s| s < 64).count();
        let med_leaves = leaf_sizes
            .iter()
            .filter(|&&s| (64..512).contains(&s))
            .count();
        let big_leaves = leaf_sizes.iter().filter(|&&s| s >= 512).count();
        tracing::info!(
            replica = replica,
            partition_secs = t1.elapsed().as_secs_f64(),
            num_leaves = leaves.len(),
            avg_leaf_size = total_pts as f64 / leaves.len().max(1) as f64,
            max_leaf_size = leaf_sizes.iter().max().unwrap_or(&0),
            total_pts = total_pts,
            "Partition complete"
        );
        // Hint to return freed partition GEMM buffers to the OS.
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

        // Leaves processed in parallel via par_chunks. Each chunk shares one
        // thread-local buffer set, amortizing TLS + RefCell + Vec allocation
        // overhead across multiple leaves.
        const LEAF_BATCH: usize = 256;
        let num_planes = hash_prune.num_planes();
        leaves.par_chunks(LEAF_BATCH).for_each_installed(|chunk| {
            leaf_build::LEAF_BUFFERS.with(|cell| {
                let mut bufs = cell.borrow_mut();
                for leaf in chunk {
                    let _edge_count = leaf_build::build_leaf_into(
                        data,
                        ndims,
                        &leaf.indices,
                        config.k,
                        metric,
                        &mut bufs,
                    );
                    let n = leaf.indices.len();
                    let group_edges = bufs.group_starts[n] as usize;
                    total_edges.fetch_add(group_edges, Ordering::Relaxed);

                    // Gather a small (n × num_planes) per-leaf sketches cache.
                    // L1-resident: 130 × 12 × 4 = ~6 KB.
                    let need = n * num_planes;
                    if bufs.local_sketches.len() < need {
                        bufs.local_sketches.resize(need, 0.0);
                    }
                    hash_prune.gather_sketches_into(&leaf.indices, &mut bufs.local_sketches[..need]);

                    hash_prune.add_edges_grouped_local_sketches(
                        &bufs.group_starts,
                        &bufs.group_data[..group_edges],
                        &leaf.indices,
                        &bufs.local_sketches[..need],
                    );
                }
            });
        });

        let replica_edges = total_edges.load(Ordering::Relaxed);
        total_edges_count += replica_edges;
        leaf_build_secs += t2.elapsed().as_secs_f64();

        tracing::info!(
            replica = replica,
            elapsed_secs = t2.elapsed().as_secs_f64(),
            total_edges = replica_edges,
            "Leaf build and merge complete"
        );
    }

    // Release thread-local leaf buffers so their arena pages can be reclaimed.
    (0..rayon::current_num_threads())
        .into_par_iter()
        .for_each_installed(|_| {
            leaf_build::release_thread_buffers();
        });

    // Extract graph and optionally apply diversity-aware final prune.
    let t3 = Instant::now();
    let (adjacency, extract_secs, final_prune_secs) = if config.final_prune {
        // Extract full reservoir (l_max candidates with distances) for diversity prune.
        let candidates = hash_prune.extract_graph_for_prune();
        let extract_secs = t3.elapsed().as_secs_f64();
        tracing::info!(
            elapsed_secs = extract_secs,
            "Graph extraction complete (full reservoir)"
        );

        let t4 = Instant::now();
        tracing::info!(
            "Applying final prune (selecting {} from {} candidates)",
            max_degree,
            config.l_max
        );
        // Log candidate stats before pruning.
        let total_candidates: usize = candidates.iter().map(|c| c.len()).sum();
        let nodes_over_degree = candidates.iter().filter(|c| c.len() > max_degree).count();
        let max_cand = candidates.iter().map(|c| c.len()).max().unwrap_or(0);
        let avg_cand = if candidates.is_empty() {
            0.0
        } else {
            total_candidates as f64 / candidates.len() as f64
        };
        println!(
            "  Final prune input: {} nodes, avg candidates={:.1}, max={}, nodes_over_max_degree={}",
            candidates.len(), avg_cand, max_cand, nodes_over_degree
        );
        tracing::info!(
            nodes = candidates.len(),
            avg_candidates = avg_cand,
            max_candidates = max_cand,
            nodes_over_max_degree = nodes_over_degree,
            max_degree = max_degree,
            "Final prune input"
        );

        let adj = final_prune_from_candidates(
            data, ndims, &candidates, max_degree, metric, config.alpha,
            config.saturate_after_prune,
        );

        // Log output stats after pruning.
        let total_edges: usize = adj.iter().map(|a| a.len()).sum();
        let avg_degree = if adj.is_empty() {
            0.0
        } else {
            total_edges as f64 / adj.len() as f64
        };
        let pruned_count = candidates
            .iter()
            .zip(adj.iter())
            .filter(|(c, a)| a.len() < c.len())
            .count();
        tracing::info!(
            avg_degree,
            pruned_count,
            pruned_pct = 100.0 * pruned_count as f64 / candidates.len().max(1) as f64,
            "Final prune output"
        );

        let final_prune_secs = t4.elapsed().as_secs_f64();
        (adj, extract_secs, final_prune_secs)
    } else {
        // No prune: truncate to max_degree by distance (original path).
        let adj = hash_prune.extract_graph();
        let extract_secs = t3.elapsed().as_secs_f64();
        tracing::info!(elapsed_secs = extract_secs, "Graph extraction complete");
        (adj, extract_secs, 0.0)
    };

    let total_secs = t_total.elapsed().as_secs_f64();

    let build_stats = PiPNNBuildStats {
        total_secs,
        sketch_secs,
        partition_secs,
        leaf_build_secs,
        extract_secs,
        final_prune_secs,
        num_leaves: total_leaves,
        total_edges: total_edges_count,
    };

    let graph = PiPNNGraph {
        adjacency,
        npoints,
        ndims,
        medoid,
        metric,
        build_stats,
    };

    // Return all freed memory (reservoirs, sketches, partition buffers, leaf buffers)
    // to the OS before handing off to the disk layout phase.

    tracing::info!(
        avg_degree = graph.avg_degree(),
        max_degree = graph.max_degree(),
        isolated = graph.num_isolated(),
        "PiPNN build complete"
    );

    Ok(graph)
}

/// occludes it (dist_to_selected < alpha * dist_to_query), else mark occluded.
/// Optionally saturate the result with occluded candidates until `max_degree`
/// is filled (controlled by `saturate`).
///
/// Candidates are pre-sorted ascending by HashPrune output.
// ───── Inline distance kernels for final_prune's inner pair-loop ─────
//
// Microbench (BigANN 10M f[8,3] candidate shape, d=128, nc=67):
//   DistanceProvider dispatch path: 6.10 ns/dist, 164 M dist/s
//   Inline AVX-512 d=128:           3.69 ns/dist, 271 M dist/s  (1.66x faster)
//   Inline AVX-2+FMA d=128:         4.91 ns/dist, 204 M dist/s  (1.24x faster)
//
// The DistanceProvider path's overhead is the indirect-fn-pointer call plus
// the closure boundary, NOT the FMA work (perf annotate proved V4 contains
// the actual vfmadd231ps loop). Inlining a d=128 const-generic kernel right
// in the inner loop avoids the call boundary entirely.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn final_prune_sql2_d128_avx512(a: *const f32, b: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    // d=128 = 4 × 32 = 8 × 16 lanes; unrolled in 4 chunks × 2 zmm each.
    for c in 0..4 {
        let va = _mm512_loadu_ps(a.add(c * 32));
        let vb = _mm512_loadu_ps(b.add(c * 32));
        let dif = _mm512_sub_ps(va, vb);
        acc0 = _mm512_fmadd_ps(dif, dif, acc0);
        let va = _mm512_loadu_ps(a.add(c * 32 + 16));
        let vb = _mm512_loadu_ps(b.add(c * 32 + 16));
        let dif = _mm512_sub_ps(va, vb);
        acc1 = _mm512_fmadd_ps(dif, dif, acc1);
    }
    _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn final_prune_sql2_d128_avx2(a: *const f32, b: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    // d=128 = 8 × 16 = 16 × 8 lanes; unrolled in 8 chunks × 2 ymm each.
    for c in 0..8 {
        let va = _mm256_loadu_ps(a.add(c * 16));
        let vb = _mm256_loadu_ps(b.add(c * 16));
        let dif = _mm256_sub_ps(va, vb);
        acc0 = _mm256_fmadd_ps(dif, dif, acc0);
        let va = _mm256_loadu_ps(a.add(c * 16 + 8));
        let vb = _mm256_loadu_ps(b.add(c * 16 + 8));
        let dif = _mm256_sub_ps(va, vb);
        acc1 = _mm256_fmadd_ps(dif, dif, acc1);
    }
    let sum = _mm256_add_ps(acc0, acc1);
    let lo = _mm256_castps256_ps128(sum);
    let hi = _mm256_extractf128_ps::<1>(sum);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s)
}

/// Dispatch the d=128 squared-L2 to the best available inline kernel, falling
/// back to the DistanceProvider path for other dimensions / metrics.
enum FinalPruneKernel {
    DispatchL2D128Avx512,
    DispatchL2D128Avx2,
    Generic(diskann_vector::distance::Distance<f32, f32>),
}

impl FinalPruneKernel {
    #[inline(always)]
    fn call(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            #[cfg(target_arch = "x86_64")]
            FinalPruneKernel::DispatchL2D128Avx512 => unsafe {
                final_prune_sql2_d128_avx512(a.as_ptr(), b.as_ptr())
            },
            #[cfg(target_arch = "x86_64")]
            FinalPruneKernel::DispatchL2D128Avx2 => unsafe {
                final_prune_sql2_d128_avx2(a.as_ptr(), b.as_ptr())
            },
            FinalPruneKernel::Generic(d) => d.call(a, b),
            #[cfg(not(target_arch = "x86_64"))]
            #[allow(unreachable_patterns)]
            _ => 0.0,
        }
    }
}

fn final_prune_from_candidates<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    candidates_per_node: &[Vec<(u32, f32)>],
    max_degree: usize,
    metric: Metric,
    alpha: f32,
    saturate: bool,
) -> Vec<Vec<u32>> {
    // Per-node thread-local scratch buffers eliminate ~30M Vec allocations
    // (cand_f32, state, selected). The allocator (mimalloc + glibc arenas) is
    // contention-prone at high thread count; thread-local reuse removes that.
    thread_local! {
        static FP_CAND_F32: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        static FP_STATE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        static FP_SELECTED: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    // Tier selection once per phase. d=128 L2 hits the inline AVX-512 path
    // (the BigANN production shape); everything else falls back to the
    // DistanceProvider dispatch.
    let use_inline_d128_l2 = ndims == 128 && matches!(metric, Metric::L2);
    let inline_tier = if use_inline_d128_l2 { tier() } else { SimdTier::Scalar };
    let generic_dist = <f32 as DistanceProvider<f32>>::distance_comparer(metric, Some(ndims));

    candidates_per_node
        .par_iter()
        .map(|candidates| {
            if candidates.is_empty() {
                return Vec::new();
            }
            let nc = candidates.len();

            // Per-thread tier-specialised inline kernel; generic fallback owns
            // the comparer for cross-thread safety (closure is Send/Sync).
            let kernel = match (use_inline_d128_l2, inline_tier) {
                #[cfg(target_arch = "x86_64")]
                (true, SimdTier::Avx512) => FinalPruneKernel::DispatchL2D128Avx512,
                #[cfg(target_arch = "x86_64")]
                (true, SimdTier::Avx2) => FinalPruneKernel::DispatchL2D128Avx2,
                _ => FinalPruneKernel::Generic(generic_dist.clone()),
            };

            FP_CAND_F32.with(|cf_cell| {
            FP_STATE.with(|st_cell| {
            FP_SELECTED.with(|sel_cell| {
                let mut cf = cf_cell.borrow_mut();
                if cf.len() < nc * ndims { cf.resize(nc * ndims, 0.0); }
                let cand_f32 = &mut cf[..nc * ndims];
                for (ci, &(id, _)) in candidates.iter().enumerate() {
                    let src = &data[id as usize * ndims..(id as usize + 1) * ndims];
                    T::as_f32_into(src, &mut cand_f32[ci * ndims..(ci + 1) * ndims])
                        .expect("f32 conversion");
                }

                // Lazy RobustPrune (paper "Optimizing RobustPrune" section).
                // Standard form admits the closest unvisited neighbor and
                // then scans ALL remaining candidates to mark occluded ones —
                // O(R·N) per node where R=max_degree, N=nc. With nc averaging
                // ~67 on this workload and R=64, that's ~1300 distance
                // computes per node × 10M nodes = the dominant final_prune
                // cost.
                //
                // Lazy form: walk candidates in ascending-distance-to-x
                // order; for each candidate c, check against the already-
                // admitted neighbors and EARLY-EXIT on the first occluder.
                // Output is identical (same admission order, same admission
                // rule) — only the work distribution differs. Most rejects
                // are occluded by one of the first few admitted neighbors,
                // so the early-exit collapses the inner-loop cost on
                // average.
                //
                // `selected_local` holds local candidate indices (not global
                // neighbor IDs) so the next candidate can be compared against
                // each admitted neighbor's f32 row by indexing `cand_f32`.
                let mut sel_local = st_cell.borrow_mut();
                sel_local.clear();
                sel_local.reserve(max_degree);

                let mut sel = sel_cell.borrow_mut();
                sel.clear();
                sel.reserve(max_degree);

                for i in 0..nc {
                    if sel.len() >= max_degree { break; }
                    let dist_x_z = candidates[i].1;
                    let z_f32 = &cand_f32[i * ndims..(i + 1) * ndims];

                    let mut occluded = false;
                    // SAFETY: sel_local stores u8 candidate indices in [0,nc);
                    // cand_f32 has nc*ndims rows so the slice index is in
                    // bounds.
                    for &local in sel_local.iter() {
                        let li = local as usize;
                        let y_f32 = &cand_f32[li * ndims..(li + 1) * ndims];
                        let dist_y_z = kernel.call(y_f32, z_f32);
                        if alpha * dist_y_z < dist_x_z {
                            occluded = true;
                            break;
                        }
                    }

                    if !occluded {
                        sel_local.push(i as u8);
                        sel.push(candidates[i].0);
                    }
                }

                if saturate && sel.len() < max_degree {
                    // Pad with closest-to-x candidates not already admitted.
                    // sel_local is monotonic in candidate distance (admitted
                    // in sorted order), so a two-pointer walk catches the
                    // gaps.
                    let mut next_admitted = 0usize;
                    for i in 0..nc {
                        if sel.len() >= max_degree { break; }
                        let is_admitted = next_admitted < sel_local.len()
                            && sel_local[next_admitted] as usize == i;
                        if is_admitted {
                            next_admitted += 1;
                        } else {
                            sel.push(candidates[i].0);
                        }
                    }
                }

                sel.clone()
            })})})
        })
        .collect_installed()
}


/// Standalone benchmark entry point: runs ONLY the leaf-build + HashPrune-insert
/// loop given pre-computed partition leaves and an initialized [`HashPrune`].
///
/// Mirrors the inner block of [`build_internal_impl`] (the `par_chunks(LEAF_BATCH)`
/// loop, lines around 460-492). Returns `(wall_secs, total_edges)`.
///
/// Used by isolation benches/perf profiles to attribute cycles strictly to
/// leaf+HP without partition / LSH-init / extract / final-prune contamination.
pub fn bench_leaf_hp_phase<T: VectorRepr + Send + Sync + 'static>(
    data: &[T],
    ndims: usize,
    leaves: &[crate::partition::Leaf],
    hash_prune: &HashPrune,
    k: usize,
    metric: Metric,
) -> (f64, usize) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    const LEAF_BATCH: usize = 256;
    let total_edges = AtomicUsize::new(0);
    let num_planes = hash_prune.num_planes();
    let t = Instant::now();
    leaves.par_chunks(LEAF_BATCH).for_each_installed(|chunk| {
        leaf_build::LEAF_BUFFERS.with(|cell| {
            let mut bufs = cell.borrow_mut();
            for leaf in chunk {
                let _ = leaf_build::build_leaf_into(
                    data, ndims, &leaf.indices, k, metric, &mut bufs,
                );
                let n = leaf.indices.len();
                let group_edges = bufs.group_starts[n] as usize;
                total_edges.fetch_add(group_edges, Ordering::Relaxed);
                let need = n * num_planes;
                if bufs.local_sketches.len() < need {
                    bufs.local_sketches.resize(need, 0.0);
                }
                hash_prune.gather_sketches_into(&leaf.indices, &mut bufs.local_sketches[..need]);
                hash_prune.add_edges_grouped_local_sketches(
                    &bufs.group_starts,
                    &bufs.group_data[..group_edges],
                    &leaf.indices,
                    &bufs.local_sketches[..need],
                );
            }
        });
    });
    let wall = t.elapsed().as_secs_f64();
    (wall, total_edges.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::PiPNNConfig;

    use super::*;

    fn generate_random_data(npoints: usize, ndims: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..npoints * ndims)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect()
    }

    fn nonzero(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("test value must be > 0")
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
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::L2, 0).unwrap();

        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

        assert_eq!(graph.npoints, npoints);
        assert!(graph.avg_degree() > 0.0);
        assert!(graph.num_isolated() < npoints);
    }

    #[test]
    fn test_build_data_length_mismatch() {
        let data = vec![0.0f32; 10];
        let ctx =
            PiPNNBuildContext::new(PiPNNConfig::default(), nonzero(64), Metric::L2, 0).unwrap();

        let result = build_typed::<f32>(&data, 5, 3, &ctx);
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
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(32), Metric::L2, 0).unwrap();

        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

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
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(32), Metric::L2, 0).unwrap();

        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

        let k = 10;
        let search_l = 100;
        let num_queries = 20;

        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query: Vec<f32> = (0..ndims)
                .map(|_| rng.random_range(-1.0f32..1.0f32))
                .collect();

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
        // Test-only println: avoids capturing tracing subscriber config.
        println!("Average recall@{}: {:.4}", k, avg_recall);

        assert!(avg_recall > 0.2, "recall too low: {:.4}", avg_recall);
    }

    #[test]
    fn test_config_validate() {
        let config = PiPNNConfig::default();
        assert!(config.validate().is_ok());

        let bad = PiPNNConfig {
            c_max: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            c_min: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            c_min: 2048,
            c_max: 1024,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            p_samp: 0.0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            p_samp: 1.5,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            fanout: vec![],
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            fanout: vec![0],
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            num_hash_planes: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad = PiPNNConfig {
            num_hash_planes: 17,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_config_validate_failures() {
        // k = 0
        let bad = PiPNNConfig {
            k: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        // replicas = 0
        let bad = PiPNNConfig {
            replicas: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        // l_max = 0
        let bad = PiPNNConfig {
            l_max: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        // p_samp exactly 1.0 is valid
        let ok = PiPNNConfig {
            p_samp: 1.0,
            ..Default::default()
        };
        assert!(ok.validate().is_ok());

        // num_hash_planes = 1 (boundary) is valid
        let ok = PiPNNConfig {
            num_hash_planes: 1,
            ..Default::default()
        };
        assert!(ok.validate().is_ok());

        // num_hash_planes = 16 (boundary) is valid
        let ok = PiPNNConfig {
            num_hash_planes: 16,
            ..Default::default()
        };
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
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }

        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };

        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::Cosine, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert!(matches!(graph.metric, Metric::Cosine));
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
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::L2, 0).unwrap();

        let graph_direct = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        let graph_typed = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

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
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::L2, 0).unwrap();

        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

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
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::L2, 0).unwrap();

        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
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
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(32), Metric::L2, 0).unwrap();

        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

        // With these settings no node should be completely isolated.
        assert_eq!(
            graph.num_isolated(),
            0,
            "found {} isolated nodes with replicas=2",
            graph.num_isolated()
        );
    }

    #[test]
    fn test_build_zero_npoints() {
        let data: Vec<f32> = vec![];
        let ctx =
            PiPNNBuildContext::new(PiPNNConfig::default(), nonzero(64), Metric::L2, 0).unwrap();
        let result = build_typed::<f32>(&data, 0, 8, &ctx);
        assert!(result.is_err(), "npoints=0 should error");
    }

    #[test]
    fn test_build_zero_ndims() {
        let data: Vec<f32> = vec![];
        let ctx =
            PiPNNBuildContext::new(PiPNNConfig::default(), nonzero(64), Metric::L2, 0).unwrap();
        let result = build_typed::<f32>(&data, 10, 0, &ctx);
        assert!(result.is_err(), "ndims=0 should error");
    }

    #[test]
    fn test_build_single_point() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 1,
            k: 3,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, 1, 4, &ctx).unwrap();
        assert_eq!(graph.npoints, 1, "should have 1 point");
        assert_eq!(
            graph.adjacency[0].len(),
            0,
            "single point should have 0 edges"
        );
    }

    #[test]
    fn test_build_two_points() {
        let data = vec![0.0f32, 0.0, 1.0, 0.0];
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 1,
            k: 3,
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, 2, 2, &ctx).unwrap();
        assert_eq!(graph.npoints, 2, "should have 2 points");
        // With 2 points, they should connect to each other.
        let total_edges: usize = graph.adjacency.iter().map(|a| a.len()).sum();
        assert!(
            total_edges > 0,
            "two points should have at least one edge between them"
        );
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
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert_eq!(
            graph.npoints, npoints,
            "should build successfully with duplicate points"
        );
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
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert_eq!(graph.npoints, npoints, "k=1 should produce valid graph");
        assert!(
            graph.avg_degree() > 0.0,
            "k=1 should still produce some edges"
        );
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
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert_eq!(
            graph.npoints, npoints,
            "k > c_max should still produce valid graph"
        );
    }

    #[test]
    fn test_search_empty_graph() {
        let graph = PiPNNGraph {
            adjacency: vec![],
            npoints: 0,
            ndims: 4,
            medoid: 0,
            metric: Metric::L2,
            build_stats: Default::default(),
        };
        let query = vec![1.0f32, 2.0, 3.0, 4.0];
        let results = graph.search(&[], &query, 5, 10);
        assert!(
            results.is_empty(),
            "search on empty graph should return empty results"
        );
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
            replicas: 1,
            l_max: 32,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
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
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(32), Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        // Query with the medoid point itself.
        let medoid = graph.medoid;
        let query = &data[medoid * ndims..(medoid + 1) * ndims];
        let results = graph.search(&data, query, 5, 50);
        assert!(
            !results.is_empty(),
            "search should return at least one result"
        );
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
            replicas: 2,
            l_max: 64,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(32), Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

        let k = 10;
        let query = &data[0..ndims];
        let exact = brute_force_knn(&data, ndims, npoints, query, k);
        let exact_set: std::collections::HashSet<usize> = exact.iter().map(|&(id, _)| id).collect();

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
    fn test_build_typed_data_length_mismatch() {
        let data = vec![1.0f32; 30]; // 30 elements
        let ctx =
            PiPNNBuildContext::new(PiPNNConfig::default(), nonzero(64), Metric::L2, 0).unwrap();
        // npoints=5, ndims=8 expects 40 elements but data has 30.
        let result = build_typed::<f32>(&data, 5, 8, &ctx);
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
            build_stats: Default::default(),
        };
        let dir = std::env::temp_dir().join("pipnn_test_save_single");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("single.bin");
        graph.save_graph(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() >= 24, "file too small for single node graph");
        let file_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(
            file_size as usize,
            bytes.len(),
            "header file_size mismatch for single node"
        );

        // Max degree should be 0 for single node with no edges.
        let max_deg = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        assert_eq!(
            max_deg, 0,
            "single node with no edges should have max_degree=0"
        );

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
            replicas: 1,
            l_max: 64,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(32), Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

        let dir = std::env::temp_dir().join("pipnn_test_save_large");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("large.bin");
        graph.save_graph(&path).unwrap();

        // Read back and verify we can parse all adjacency lists.
        let bytes = std::fs::read(&path).unwrap();
        let file_size = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(
            file_size as usize,
            bytes.len(),
            "header file_size mismatch for large graph"
        );

        let mut offset = 24usize;
        let mut total_parsed_nodes = 0usize;
        while offset < bytes.len() {
            let num_neighbors =
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            for _ in 0..num_neighbors {
                let neighbor =
                    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                assert!(
                    neighbor < npoints,
                    "neighbor index {} out of range for node {}",
                    neighbor,
                    total_parsed_nodes
                );
                offset += 4;
            }
            total_parsed_nodes += 1;
        }
        assert_eq!(
            total_parsed_nodes,
            npoints + 1, // +1 for frozen start point
            "expected to parse {} nodes ({}+1 frozen) but got {}",
            npoints + 1,
            npoints,
            total_parsed_nodes
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
        let max_degree = 16;
        let config_no_prune = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 6,
            replicas: 2,
            l_max: 64,
            final_prune: false,
            ..Default::default()
        };
        let config_with_prune = PiPNNConfig {
            final_prune: true,
            ..config_no_prune.clone()
        };

        let ctx_no =
            PiPNNBuildContext::new(config_no_prune, nonzero(max_degree), Metric::L2, 0).unwrap();
        let ctx_yes =
            PiPNNBuildContext::new(config_with_prune, nonzero(max_degree), Metric::L2, 0).unwrap();
        let graph_no = build_typed::<f32>(&data, npoints, ndims, &ctx_no).unwrap();
        let graph_yes = build_typed::<f32>(&data, npoints, ndims, &ctx_yes).unwrap();

        // Final prune should not increase max degree beyond max_degree.
        assert!(
            graph_yes.max_degree() <= max_degree,
            "final_prune max_degree {} > expected max_degree {}",
            graph_yes.max_degree(),
            max_degree
        );

        // Both should be valid graphs.
        assert!(graph_no.avg_degree() > 0.0);
        assert!(graph_yes.avg_degree() > 0.0);
    }

    #[test]
    fn test_final_prune_from_candidates_diversity() {
        // 4 points: 0=(0,0), 1=(1,0), 2=(0,1), 3=(0.1,0) -- point 3 is occluded by 1.
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.1, 0.0];
        let candidates = vec![
            // Node 0's candidates sorted by distance: 3 (close, same direction as 1), 1 (far along x), 2 (far along y)
            vec![(3, 0.01f32), (1, 1.0f32), (2, 1.0f32)],
            vec![],
            vec![],
            vec![],
        ];

        let result = final_prune_from_candidates(&data, 2, &candidates, 2, Metric::L2, 1.2, true);
        let node0 = &result[0];
        // With alpha=1.2, point 3 should be selected first (closest).
        // Point 1 might be pruned because dist(3,1) * 1.2 < dist(0,1).
        // Point 2 should survive (different direction).
        assert!(!node0.is_empty());
        assert!(node0.len() <= 2, "should respect max_degree=2");
        // Node 0 should keep at least one neighbor.
        assert!(
            node0.contains(&3),
            "closest candidate should always be selected"
        );
    }

    #[test]
    fn test_final_prune_from_candidates_empty() {
        let data: Vec<f32> = vec![0.0; 8];
        let candidates: Vec<Vec<(u32, f32)>> = vec![vec![], vec![], vec![], vec![]];
        let result = final_prune_from_candidates(&data, 2, &candidates, 10, Metric::L2, 1.2, true);
        assert!(result.iter().all(|adj| adj.is_empty()));
    }

    #[test]
    fn test_final_prune_from_candidates_single_candidate() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0];
        let candidates = vec![vec![(1, 1.0f32)], vec![(0, 1.0f32)]];
        let result = final_prune_from_candidates(&data, 2, &candidates, 10, Metric::L2, 1.2, true);
        assert_eq!(result[0], vec![1]);
        assert_eq!(result[1], vec![0]);
    }

    #[test]
    fn test_final_prune_alpha_effect() {
        // Higher alpha = less aggressive pruning = more edges retained.
        let npoints = 200;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config_aggressive = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 6,
            replicas: 2,
            l_max: 64,
            final_prune: true,
            alpha: 1.0,
            ..Default::default()
        };
        let config_relaxed = PiPNNConfig {
            alpha: 2.0,
            ..config_aggressive.clone()
        };

        let ctx_aggressive =
            PiPNNBuildContext::new(config_aggressive, nonzero(16), Metric::L2, 0).unwrap();
        let ctx_relaxed =
            PiPNNBuildContext::new(config_relaxed, nonzero(16), Metric::L2, 0).unwrap();
        let graph_aggressive = build_typed::<f32>(&data, npoints, ndims, &ctx_aggressive).unwrap();
        let graph_relaxed = build_typed::<f32>(&data, npoints, ndims, &ctx_relaxed).unwrap();

        // Relaxed alpha should yield denser graph (more edges survive pruning).
        assert!(
            graph_relaxed.avg_degree() >= graph_aggressive.avg_degree(),
            "alpha=2.0 ({:.1}) should produce >= degree than alpha=1.0 ({:.1})",
            graph_relaxed.avg_degree(),
            graph_aggressive.avg_degree()
        );
    }

    #[test]
    fn test_build_final_prune_vs_no_prune_recall() {
        // Both should produce searchable graphs with reasonable recall.
        let npoints = 500;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config_no_prune = PiPNNConfig {
            c_max: 128,
            c_min: 32,
            k: 3,
            replicas: 2,
            l_max: 64,
            final_prune: false,
            ..Default::default()
        };
        let config_prune = PiPNNConfig {
            l_max: 64,
            final_prune: true,
            ..config_no_prune.clone()
        };

        let ctx_no = PiPNNBuildContext::new(config_no_prune, nonzero(32), Metric::L2, 0).unwrap();
        let ctx_yes = PiPNNBuildContext::new(config_prune, nonzero(32), Metric::L2, 0).unwrap();
        let graph_no = build_typed::<f32>(&data, npoints, ndims, &ctx_no).unwrap();
        let graph_yes = build_typed::<f32>(&data, npoints, ndims, &ctx_yes).unwrap();

        // Both should have non-trivial degree.
        assert!(graph_no.avg_degree() > 1.0);
        assert!(graph_yes.avg_degree() > 1.0);

        // Final prune should produce sparser graph.
        assert!(
            graph_yes.avg_degree() <= graph_no.avg_degree(),
            "pruned ({:.1}) should be <= unpruned ({:.1})",
            graph_yes.avg_degree(),
            graph_no.avg_degree()
        );

        // Both should be searchable.
        let query = &data[0..ndims];
        let r1 = crate::leaf_build::brute_force_knn(&data, ndims, npoints, query, 10);
        let s_no = graph_no.search(&data, query, 10, 50);
        let s_yes = graph_yes.search(&data, query, 10, 50);
        assert!(!s_no.is_empty(), "no_prune search should return results");
        assert!(!s_yes.is_empty(), "prune search should return results");

        // Both should find the nearest neighbor (query is point 0).
        assert_eq!(s_no[0].0, r1[0].0, "no_prune should find NN");
        assert_eq!(s_yes[0].0, r1[0].0, "prune should find NN");
    }

    #[test]
    fn test_build_cosine_normalized() {
        let npoints = 100;
        let ndims = 8;
        let mut data = generate_random_data(npoints, ndims, 42);
        // Normalize all vectors.
        for i in 0..npoints {
            let row = &mut data[i * ndims..(i + 1) * ndims];
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }

        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            replicas: 2,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::CosineNormalized, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert!(graph.avg_degree() > 0.0);
        assert_eq!(graph.metric, Metric::CosineNormalized);

        // Search should work with cosine metric.
        let query = &data[0..ndims];
        let results = graph.search(&data, query, 5, 20);
        assert!(!results.is_empty());
        // First result should be the query point itself.
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_build_inner_product_end_to_end() {
        let npoints = 200;
        let ndims = 8;
        let data = generate_random_data(npoints, ndims, 42);

        let config = PiPNNConfig {
            c_max: 64,
            c_min: 16,
            k: 3,
            replicas: 2,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(16), Metric::InnerProduct, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert!(graph.avg_degree() > 0.0);
        assert_eq!(graph.metric, Metric::InnerProduct);

        let query = &data[0..ndims];
        let results = graph.search(&data, query, 5, 20);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_config_validate_alpha_infinity() {
        let config = PiPNNConfig {
            alpha: f32::INFINITY,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_p_samp_nan() {
        let config = PiPNNConfig {
            p_samp: f64::NAN,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_pipnn_graph_stats() {
        let npoints = 100;
        let ndims = 4;
        let data = generate_random_data(npoints, ndims, 42);
        let max_degree = 16;
        let config = PiPNNConfig {
            c_max: 32,
            c_min: 8,
            k: 3,
            ..Default::default()
        };
        let ctx = PiPNNBuildContext::new(config, nonzero(max_degree), Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

        assert_eq!(graph.npoints, npoints);
        assert_eq!(graph.ndims, ndims);
        assert!(graph.medoid < npoints);
        assert!(graph.max_degree() <= max_degree);
        assert!(graph.avg_degree() > 0.0);
        assert!(graph.avg_degree() <= max_degree as f64);
        // num_isolated should be 0 for a well-connected graph.
        assert_eq!(
            graph.num_isolated(),
            0,
            "graph should have no isolated nodes"
        );
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = PiPNNConfig::default();
        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: PiPNNConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(config, deserialized);
    }
}
