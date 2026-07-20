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

use diskann::{
    graph::{config::PruneKind, prune},
    neighbor::Neighbor,
    utils::VectorRepr,
};
use rayon::prelude::*;

use crate::cpu_dispatch::{tier, SimdTier};
use crate::hash_prune::HashPrune;
use crate::leaf_build;
use crate::partition::PartitionConfig;
use crate::rayon_util::ParIterInstalled;
use crate::{PiPNNBuildContext, PiPNNError, PiPNNResult};

use diskann_vector::distance::{DistanceProvider, Metric};

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

/// The graph and statistics produced by a PiPNN build.
#[derive(Debug)]
pub struct PiPNNBuildOutput {
    /// Adjacency lists: graph[i] contains the neighbor indices for point i.
    pub adjacency: Vec<Vec<u32>>,
    /// Build timing breakdown.
    pub build_stats: PiPNNBuildStats,
}

impl PiPNNBuildOutput {
    /// Get the average out-degree.
    pub fn avg_degree(&self) -> f64 {
        let total: usize = self.adjacency.iter().map(|adj| adj.len()).sum();
        total as f64 / self.adjacency.len() as f64
    }

    /// Get the max out-degree.
    fn max_degree(&self) -> usize {
        self.adjacency
            .iter()
            .map(|adj| adj.len())
            .max()
            .unwrap_or(0)
    }

    /// Count the number of points with zero out-degree.
    fn num_isolated(&self) -> usize {
        self.adjacency.iter().filter(|adj| adj.is_empty()).count()
    }
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
) -> PiPNNResult<PiPNNBuildOutput> {
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
) -> PiPNNResult<PiPNNBuildOutput> {
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
) -> PiPNNResult<PiPNNBuildOutput> {
    let config = ctx.config();
    let metric = ctx.metric();
    let max_degree = ctx.max_degree().get();
    let t_total = Instant::now();

    log_simd_tier();

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

    let mut partition_secs = 0.0f64;
    let mut leaf_build_secs = 0.0f64;

    // Replicas are a PARTITION concept (cf. gp-ann's dense ball carving,
    // `BuildApproximateNearestNeighborGraph`): each replica re-partitions the
    // full dataset with a different seed and its leaves are ACCUMULATED into one
    // shared pool. The leaf build (GEMM k-NN) and HashPrune merge then run ONCE
    // over the combined pool — not per replica — so a point's neighbors from
    // overlapping leaves across all replicas merge together before top-k. The
    // partition config is identical across replicas (only the seed varies), so
    // build it once here.
    let partition_config = PartitionConfig::new(
        config.c_max,
        config.c_min,
        config.p_samp,
        config.fanout.clone(),
        metric,
        crate::partition::LEADER_CAP,
    )?;

    let mut leaves: Vec<crate::partition::Leaf> = Vec::new();
    for replica in 0..config.replicas {
        let seed = 1000 + replica as u64 * 7919;

        let t1 = Instant::now();
        let replica_leaves =
            crate::partition::partition(data, ndims, npoints, &partition_config, seed);
        partition_secs += t1.elapsed().as_secs_f64();

        let total_pts: usize = replica_leaves.iter().map(|l| l.indices.len()).sum();
        tracing::info!(
            replica = replica,
            partition_secs = t1.elapsed().as_secs_f64(),
            num_leaves = replica_leaves.len(),
            avg_leaf_size = total_pts as f64 / replica_leaves.len().max(1) as f64,
            total_pts = total_pts,
            overlap = total_pts as f64 / npoints as f64,
            "Partition complete"
        );
        leaves.extend(replica_leaves);
    }
    let total_leaves = leaves.len();

    // Free partition's per-thread stripe buffers (~20 MB/thread × 48 threads
    // ≈ 1 GB) so peak RSS during leaf+HP is max-of-phases instead of
    // sum-of-phases. Mirrors `release_thread_buffers` for leaf_build.
    (0..rayon::current_num_threads())
        .into_par_iter()
        .for_each_installed(|_| {
            crate::partition::release_thread_buffers();
        });

    // Build all leaves (from every replica) in parallel, streaming edges to the
    // single HashPrune. This is the one leaf-build + merge pass.
    let t2 = Instant::now();

    use std::sync::atomic::{AtomicUsize, Ordering};
    let total_edges = AtomicUsize::new(0);

    // Leaves processed in parallel via par_chunks. Each chunk shares one
    // thread-local buffer set, amortizing TLS + RefCell + Vec allocation
    // overhead across multiple leaves. Chunk size scales with leaf count
    // and rayon pool size so every thread gets ~4 work-stealing units.
    let nthreads = rayon::current_num_threads().max(1);
    let leaf_batch = (leaves.len() / (nthreads * 4)).clamp(1, 256);
    let num_planes = hash_prune.num_planes();
    leaves.par_chunks(leaf_batch).for_each_installed(|chunk| {
        leaf_build::LEAF_BUFFERS.with(|cell| {
            let mut bufs = cell.borrow_mut();
            for leaf in chunk {
                let group_edges = leaf_build::build_leaf_into(
                    data,
                    ndims,
                    &leaf.indices,
                    config.k,
                    metric,
                    &mut bufs,
                );
                let n = leaf.indices.len();
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
    // `leaves` is dropped right after the build to free the accumulated index
    // Vecs (~n·overlap u32) before extraction. (It is fully live *during* the
    // build — every leaf is an input — so incremental freeing can't lower the
    // leaf-build peak; the reduction target there is the reservoir slab, below.)
    drop(leaves);

    let total_edges_count = total_edges.load(Ordering::Relaxed);
    leaf_build_secs += t2.elapsed().as_secs_f64();

    tracing::info!(
        elapsed_secs = t2.elapsed().as_secs_f64(),
        total_leaves = total_leaves,
        total_edges = total_edges_count,
        "Leaf build and merge complete"
    );

    // Release thread-local leaf buffers so their arena pages can be reclaimed.
    (0..rayon::current_num_threads())
        .into_par_iter()
        .for_each_installed(|_| {
            leaf_build::release_thread_buffers();
        });

    // Extract graph and optionally apply diversity-aware final prune.
    let t3 = Instant::now();
    let (adjacency, extract_secs, final_prune_secs) = if config.final_prune {
        // Extract each node's candidate ids (up to l_max) for the diversity
        // prune. final_prune recomputes distances from the base data, so ids
        // alone suffice — dropping the reservoir's distance/hash slabs before
        // this copy (see extract_graph_ids).
        let candidates = hash_prune.extract_graph_ids();
        let extract_secs = t3.elapsed().as_secs_f64();
        tracing::info!(
            elapsed_secs = extract_secs,
            "Graph extraction complete (full reservoir)"
        );

        let t4 = Instant::now();
        tracing::info!(
            nodes = candidates.len(),
            "Applying final prune (selecting {} from up to {} candidates)",
            max_degree,
            config.l_max
        );

        let adj =
            final_prune_from_candidates(data, ndims, candidates, max_degree, metric, ctx.alpha())?;

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

    let graph = PiPNNBuildOutput {
        adjacency,
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

// ───── Inline distance kernels for final_prune's inner pair-loop ─────
//
// The `DistanceProvider` path indirects through a fn pointer + closure on
// every pair; the inline kernels below skip that boundary so the FMA chain
// stays in the same basic block as the alpha-occlusion check.

/// AVX-512 squared-L2 for any `d`. Two-accumulator FMA chain over 32-wide
/// pairs (2 ZMM per iter) to hide FMA latency; tail handled with a 16-wide
/// pass and a masked load. Generic over `d` — no bench-specific dim baked in.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn final_prune_sql2_avx512(a: *const f32, b: *const f32, d: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let pairs = d / 32;
    for c in 0..pairs {
        let va = _mm512_loadu_ps(a.add(c * 32));
        let vb = _mm512_loadu_ps(b.add(c * 32));
        let dif = _mm512_sub_ps(va, vb);
        acc0 = _mm512_fmadd_ps(dif, dif, acc0);
        let va = _mm512_loadu_ps(a.add(c * 32 + 16));
        let vb = _mm512_loadu_ps(b.add(c * 32 + 16));
        let dif = _mm512_sub_ps(va, vb);
        acc1 = _mm512_fmadd_ps(dif, dif, acc1);
    }
    let mut off = pairs * 32;
    if d - off >= 16 {
        let va = _mm512_loadu_ps(a.add(off));
        let vb = _mm512_loadu_ps(b.add(off));
        let dif = _mm512_sub_ps(va, vb);
        acc0 = _mm512_fmadd_ps(dif, dif, acc0);
        off += 16;
    }
    let tail = d - off;
    if tail > 0 {
        let mask: u16 = ((1u32 << tail) - 1) as u16;
        let va = _mm512_maskz_loadu_ps(mask, a.add(off));
        let vb = _mm512_maskz_loadu_ps(mask, b.add(off));
        let dif = _mm512_sub_ps(va, vb);
        acc1 = _mm512_fmadd_ps(dif, dif, acc1);
    }
    _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1))
}

/// AVX-2 squared-L2 for any `d`. Two-accumulator FMA chain over 16-wide
/// pairs (2 YMM per iter); tail handled with an 8-wide pass and a scalar
/// loop. Generic over `d`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn final_prune_sql2_avx2(a: *const f32, b: *const f32, d: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let pairs = d / 16;
    for c in 0..pairs {
        let va = _mm256_loadu_ps(a.add(c * 16));
        let vb = _mm256_loadu_ps(b.add(c * 16));
        let dif = _mm256_sub_ps(va, vb);
        acc0 = _mm256_fmadd_ps(dif, dif, acc0);
        let va = _mm256_loadu_ps(a.add(c * 16 + 8));
        let vb = _mm256_loadu_ps(b.add(c * 16 + 8));
        let dif = _mm256_sub_ps(va, vb);
        acc1 = _mm256_fmadd_ps(dif, dif, acc1);
    }
    let mut off = pairs * 16;
    if d - off >= 8 {
        let va = _mm256_loadu_ps(a.add(off));
        let vb = _mm256_loadu_ps(b.add(off));
        let dif = _mm256_sub_ps(va, vb);
        acc0 = _mm256_fmadd_ps(dif, dif, acc0);
        off += 8;
    }
    let sum = _mm256_add_ps(acc0, acc1);
    let lo = _mm256_castps256_ps128(sum);
    let hi = _mm256_extractf128_ps::<1>(sum);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_hadd_ps(s, s);
    let s = _mm_hadd_ps(s, s);
    let mut acc = _mm_cvtss_f32(s);
    while off < d {
        let v = *a.add(off) - *b.add(off);
        acc += v * v;
        off += 1;
    }
    acc
}

/// Dispatch squared-L2 to the inline tier-specific kernel (now dim-generic),
/// falling back to the DistanceProvider path for non-L2 metrics. `ndims` is
/// the runtime dim threaded through `call` to the inline kernel.
enum FinalPruneKernel {
    InlineL2Avx512 { ndims: usize },
    InlineL2Avx2 { ndims: usize },
    Generic(diskann_vector::distance::Distance<f32, f32>),
}

impl FinalPruneKernel {
    #[inline(always)]
    fn call(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: tier selection verifies AVX-512 support, and both slices
            // contain at least `ndims` elements.
            FinalPruneKernel::InlineL2Avx512 { ndims } => unsafe {
                final_prune_sql2_avx512(a.as_ptr(), b.as_ptr(), *ndims)
            },
            #[cfg(target_arch = "x86_64")]
            // SAFETY: tier selection verifies AVX2/FMA support, and both slices
            // contain at least `ndims` elements.
            FinalPruneKernel::InlineL2Avx2 { ndims } => unsafe {
                final_prune_sql2_avx2(a.as_ptr(), b.as_ptr(), *ndims)
            },
            FinalPruneKernel::Generic(d) => d.call(a, b),
            #[cfg(not(target_arch = "x86_64"))]
            #[allow(unreachable_patterns)]
            _ => 0.0,
        }
    }
}

pub(crate) fn final_prune_from_candidates<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    candidates_per_node: Vec<Vec<u32>>,
    max_degree: usize,
    metric: Metric,
    alpha: f32,
) -> PiPNNResult<Vec<Vec<u32>>> {
    // Per-node thread-local scratch buffers eliminate ~30M Vec allocations.
    // The allocator (mimalloc + glibc arenas) is
    // contention-prone at high thread count; thread-local reuse removes that.
    thread_local! {
        static FP_CAND_F32: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        static FP_NODE_F32: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        static FP_ORDER: std::cell::RefCell<Vec<(f32, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
        static FP_SCRATCH: std::cell::RefCell<prune::Scratch<u32>> = std::cell::RefCell::new(prune::Scratch::new());
        static FP_CACHE: std::cell::RefCell<Vec<(f32, Option<usize>)>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    // Tier selection once per phase. d=128 L2 hits the inline AVX-512 path
    // (the BigANN production shape); everything else falls back to the
    // DistanceProvider dispatch.
    let use_inline_l2 = matches!(metric, Metric::L2);
    let inline_tier = if use_inline_l2 {
        tier()
    } else {
        SimdTier::Scalar
    };
    let generic_dist = <f32 as DistanceProvider<f32>>::distance_comparer(metric, Some(ndims));

    candidates_per_node
        .into_par_iter()
        .enumerate()
        .map(|(node_idx, candidates)| -> PiPNNResult<Vec<u32>> {
            if candidates.is_empty() {
                return Ok(Vec::new());
            }
            let nc = candidates.len();

            // RobustPrune is only needed to reduce overfull candidate pools.
            // If a node already fits the published degree bound, preserve the
            // candidate list directly; otherwise under-degree nodes lose useful
            // edges and still pay the full prune cost. Move (not clone) the input
            // Vec straight into the output — the common case, and consuming by
            // value here means each input list is freed as its output is produced
            // instead of the whole input set co-existing with the whole output.
            if nc <= max_degree {
                return Ok(candidates);
            }

            // Per-thread tier-specialised inline kernel; generic fallback owns
            // the comparer for cross-thread safety (closure is Send/Sync).
            let kernel = match (use_inline_l2, inline_tier) {
                #[cfg(target_arch = "x86_64")]
                (true, SimdTier::Avx512) => FinalPruneKernel::InlineL2Avx512 { ndims },
                #[cfg(target_arch = "x86_64")]
                (true, SimdTier::Avx2) => FinalPruneKernel::InlineL2Avx2 { ndims },
                _ => FinalPruneKernel::Generic(generic_dist),
            };

            FP_CAND_F32.with(|cf_cell| {
                FP_NODE_F32.with(|nf_cell| {
                    FP_ORDER.with(|ord_cell| {
                        FP_SCRATCH.with(|scratch_cell| {
                            FP_CACHE.with(|cache_cell| {
                                let mut cf = cf_cell.borrow_mut();
                                if cf.len() < nc * ndims {
                                    cf.resize(nc * ndims, 0.0);
                                }
                                let cand_f32 = &mut cf[..nc * ndims];
                                for (ci, &id) in candidates.iter().enumerate() {
                                    let src = &data[id as usize * ndims..(id as usize + 1) * ndims];
                                    T::as_f32_into(
                                        src,
                                        &mut cand_f32[ci * ndims..(ci + 1) * ndims],
                                    )
                                    .map_err(|error| PiPNNError::Conversion(error.to_string()))?;
                                }

                                // Node x's own vector as fresh f32. final_prune
                                // recomputes EVERY distance from the f32 data — the
                                // reservoir's bf16 candidate distances are lossy and,
                                // for InnerProduct, stored as sign-incorrect u16 sort
                                // keys, so they are never trusted here.
                                let mut nf = nf_cell.borrow_mut();
                                if nf.len() < ndims {
                                    nf.resize(ndims, 0.0);
                                }
                                let x_f32 = &mut nf[..ndims];
                                {
                                    let src = &data[node_idx * ndims..(node_idx + 1) * ndims];
                                    T::as_f32_into(src, x_f32).map_err(|error| {
                                        PiPNNError::Conversion(error.to_string())
                                    })?;
                                }

                                // Fresh f32 distance-to-x for every candidate, sorted
                                // closest-first. Replaces relying on the reservoir's
                                // (lossy / IP-mis-ordered) ordering.
                                let mut order = ord_cell.borrow_mut();
                                order.clear();
                                order.reserve(nc);
                                for i in 0..nc {
                                    let z = &cand_f32[i * ndims..(i + 1) * ndims];
                                    order.push((kernel.call(x_f32, z), i as u32));
                                }
                                // Break exact-distance ties by candidate point
                                // id so the admitted set is independent of the
                                // input candidate order (extract emits ids
                                // unsorted). Tie-breaking by the array index
                                // would NOT be order-invariant — the index→id
                                // map differs between layouts; the id is stable.
                                order.sort_unstable_by(|a, b| {
                                    a.0.total_cmp(&b.0).then_with(|| {
                                        candidates[a.1 as usize].cmp(&candidates[b.1 as usize])
                                    })
                                });

                                let mut scratch = scratch_cell.borrow_mut();
                                scratch.candidates_mut().clear();
                                scratch.candidates_mut().extend(
                                    order
                                        .iter()
                                        .map(|&(distance, local)| Neighbor::new(local, distance)),
                                );

                                let policy = prune::Policy::new(
                                    max_degree,
                                    alpha,
                                    PruneKind::from_metric(metric),
                                    false,
                                );
                                let mut cache = cache_cell.borrow_mut();
                                {
                                    let mut context = scratch.as_sorted_context(nc);
                                    prune::robust_prune(
                                        &mut context,
                                        policy,
                                        &mut cache,
                                        |local| Some(local as usize),
                                        |&left, &right| {
                                            let left = &cand_f32[left * ndims..(left + 1) * ndims];
                                            let right =
                                                &cand_f32[right * ndims..(right + 1) * ndims];
                                            Ok::<_, std::convert::Infallible>(
                                                kernel.call(left, right),
                                            )
                                        },
                                        |_| false,
                                    )
                                    .map_err(|error| PiPNNError::Prune(error.to_string()))?;
                                }

                                Ok(scratch
                                    .neighbors()
                                    .iter()
                                    .map(|&local| candidates[local as usize])
                                    .collect())
                            })
                        })
                    })
                })
            })
        })
        .collect_installed::<PiPNNResult<Vec<_>>>()
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
        let ctx = PiPNNBuildContext::new(config, nonzero(16), 1.2, Metric::L2, 0).unwrap();

        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

        assert_eq!(graph.adjacency.len(), npoints);
        assert!(graph.avg_degree() > 0.0);
        assert!(graph.num_isolated() < npoints);
    }

    #[test]
    fn test_build_data_length_mismatch() {
        let data = vec![0.0f32; 10];
        let ctx = PiPNNBuildContext::new(PiPNNConfig::default(), nonzero(64), 1.2, Metric::L2, 0)
            .unwrap();

        let result = build_typed::<f32>(&data, 5, 3, &ctx);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, PiPNNError::DataLengthMismatch { .. }));
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

        let ctx = PiPNNBuildContext::new(config, nonzero(16), 1.2, Metric::Cosine, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert_eq!(graph.adjacency.len(), npoints);
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
        let ctx = PiPNNBuildContext::new(config, nonzero(16), 1.2, Metric::L2, 0).unwrap();

        let graph_direct = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        let graph_typed = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

        assert_eq!(graph_direct.adjacency, graph_typed.adjacency);
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
        let ctx = PiPNNBuildContext::new(config, nonzero(32), 1.2, Metric::L2, 0).unwrap();

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
        let ctx = PiPNNBuildContext::new(PiPNNConfig::default(), nonzero(64), 1.2, Metric::L2, 0)
            .unwrap();
        let result = build_typed::<f32>(&data, 0, 8, &ctx);
        assert!(result.is_err(), "npoints=0 should error");
    }

    #[test]
    fn test_build_zero_ndims() {
        let data: Vec<f32> = vec![];
        let ctx = PiPNNBuildContext::new(PiPNNConfig::default(), nonzero(64), 1.2, Metric::L2, 0)
            .unwrap();
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
        let ctx = PiPNNBuildContext::new(config, nonzero(16), 1.2, Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, 1, 4, &ctx).unwrap();
        assert_eq!(graph.adjacency.len(), 1, "should have 1 point");
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
        let ctx = PiPNNBuildContext::new(config, nonzero(16), 1.2, Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, 2, 2, &ctx).unwrap();
        assert_eq!(graph.adjacency.len(), 2, "should have 2 points");
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
        let ctx = PiPNNBuildContext::new(config, nonzero(16), 1.2, Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert_eq!(
            graph.adjacency.len(),
            npoints,
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
        let ctx = PiPNNBuildContext::new(config, nonzero(16), 1.2, Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert_eq!(
            graph.adjacency.len(),
            npoints,
            "k=1 should produce valid graph"
        );
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
        let ctx = PiPNNBuildContext::new(config, nonzero(16), 1.2, Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert_eq!(
            graph.adjacency.len(),
            npoints,
            "k > c_max should still produce valid graph"
        );
    }

    #[test]
    fn test_build_typed_data_length_mismatch() {
        let data = vec![1.0f32; 30]; // 30 elements
        let ctx = PiPNNBuildContext::new(PiPNNConfig::default(), nonzero(64), 1.2, Metric::L2, 0)
            .unwrap();
        // npoints=5, ndims=8 expects 40 elements but data has 30.
        let result = build_typed::<f32>(&data, 5, 8, &ctx);
        assert!(
            result.is_err(),
            "data length mismatch should produce an error"
        );
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
            PiPNNBuildContext::new(config_no_prune, nonzero(max_degree), 1.2, Metric::L2, 0)
                .unwrap();
        let ctx_yes =
            PiPNNBuildContext::new(config_with_prune, nonzero(max_degree), 1.2, Metric::L2, 0)
                .unwrap();
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
        // Candidate ids only; final_prune recomputes distances from `data`
        // and re-sorts, so input order is irrelevant.
        let candidates = vec![vec![3u32, 1, 2], vec![], vec![], vec![]];

        let result = final_prune_from_candidates(&data, 2, candidates, 2, Metric::L2, 1.2).unwrap();
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
    fn test_final_prune_ties_use_point_id() {
        let data = vec![
            0.0f32, 0.0, // node 0
            1.0, 0.0, // candidate 1
            -1.0, 0.0, // candidate 2, same distance to node 0
            0.0, 1.0, // candidate 3, same distance to node 0
        ];
        let prune = |candidates| {
            final_prune_from_candidates(
                &data,
                2,
                vec![candidates, vec![], vec![], vec![]],
                1,
                Metric::L2,
                1.2,
            )
            .unwrap()
        };

        assert_eq!(prune(vec![3, 2, 1])[0], vec![1]);
        assert_eq!(prune(vec![2, 1, 3])[0], vec![1]);
    }

    #[test]
    fn test_final_prune_from_candidates_empty() {
        let data: Vec<f32> = vec![0.0; 8];
        let candidates: Vec<Vec<u32>> = vec![vec![], vec![], vec![], vec![]];
        let result =
            final_prune_from_candidates(&data, 2, candidates, 10, Metric::L2, 1.2).unwrap();
        assert!(result.iter().all(|adj| adj.is_empty()));
    }

    #[test]
    fn test_final_prune_from_candidates_single_candidate() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0];
        let candidates = vec![vec![1u32], vec![0u32]];
        let result =
            final_prune_from_candidates(&data, 2, candidates, 10, Metric::L2, 1.2).unwrap();
        assert_eq!(result[0], vec![1]);
        assert_eq!(result[1], vec![0]);
    }

    #[test]
    fn test_final_prune_keeps_under_degree_candidates() {
        let data: Vec<f32> = vec![
            0.0, 0.0, // node 0
            0.1, 0.0, // candidate 1, would occlude candidate 2
            1.0, 0.0, // candidate 2
            0.0, 1.0, // candidate 3
        ];
        let candidates = vec![vec![1u32, 2, 3], vec![], vec![], vec![]];

        let result = final_prune_from_candidates(&data, 2, candidates, 4, Metric::L2, 1.2).unwrap();

        assert_eq!(result[0], vec![1, 2, 3]);
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
            ..Default::default()
        };
        let config_relaxed = config_aggressive.clone();

        let ctx_aggressive =
            PiPNNBuildContext::new(config_aggressive, nonzero(16), 1.0, Metric::L2, 0).unwrap();
        let ctx_relaxed =
            PiPNNBuildContext::new(config_relaxed, nonzero(16), 2.0, Metric::L2, 0).unwrap();
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
        // Both modes should produce non-trivial graphs.
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

        let ctx_no =
            PiPNNBuildContext::new(config_no_prune, nonzero(32), 1.2, Metric::L2, 0).unwrap();
        let ctx_yes =
            PiPNNBuildContext::new(config_prune, nonzero(32), 1.2, Metric::L2, 0).unwrap();
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
        let ctx =
            PiPNNBuildContext::new(config, nonzero(16), 1.2, Metric::CosineNormalized, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert!(graph.avg_degree() > 0.0);
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
        let ctx =
            PiPNNBuildContext::new(config, nonzero(16), 1.2, Metric::InnerProduct, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();
        assert!(graph.avg_degree() > 0.0);
    }

    #[test]
    fn test_build_context_rejects_invalid_alpha() {
        assert!(PiPNNBuildContext::new(
            PiPNNConfig::default(),
            nonzero(16),
            f32::INFINITY,
            Metric::L2,
            0,
        )
        .is_err());
        assert!(
            PiPNNBuildContext::new(PiPNNConfig::default(), nonzero(16), 0.9, Metric::L2, 0,)
                .is_err()
        );
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
        let ctx = PiPNNBuildContext::new(config, nonzero(max_degree), 1.2, Metric::L2, 0).unwrap();
        let graph = build_typed::<f32>(&data, npoints, ndims, &ctx).unwrap();

        assert_eq!(graph.adjacency.len(), npoints);
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
