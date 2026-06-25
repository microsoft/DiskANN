/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Randomized Ball Carving (RBC) partitioning — iterative, parallel at every level.
//!
//! Recursively partitions the dataset into overlapping clusters using an iterative
//! work-queue approach. All oversized clusters at each level are processed in parallel.

use crate::rayon_util::ParIterInstalled;
use crate::{PiPNNError, PiPNNResult};
use diskann::utils::VectorRepr;
use rand::prelude::IndexedRandom;
use rand::SeedableRng;
use rayon::prelude::*;

/// Maximum supported `fanout` value: hard upper bound on the size of the
/// stack-allocated top-k tracker [`assign_to_leaders`] uses on its hot path.
/// Enforced by [`crate::PiPNNConfig::validate`].
pub const MAX_FANOUT: usize = 16;

/// A leaf partition containing indices into the original dataset.
///
/// Uses `u32` instead of `usize` to halve memory on 64-bit platforms.
/// Sufficient for datasets up to 4 billion points.
#[derive(Debug, Clone)]
pub struct Leaf {
    pub indices: Vec<u32>,
}

/// Configuration for RBC partitioning.
///
/// Fields are private; construct via [`PartitionConfig::new`], which enforces
/// the partition-layer invariants on `fanout` and `leader_cap`.
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    pub(crate) c_max: usize,
    pub(crate) c_min: usize,
    pub(crate) p_samp: f64,
    pub(crate) fanout: Vec<usize>,
    pub(crate) metric: diskann_vector::distance::Metric,
    pub(crate) leader_cap: usize,
}

/// Max iterations of the partition loop before remaining oversized clusters
/// are accepted as leaves. Guards against pathological hub geometries.
const MAX_PARTITION_ITER: usize = 30;

/// Per-partition-level leader hardcap (paper recommendation). Static — there is
/// no runtime override.
pub(crate) const LEADER_CAP: usize = 1000;

impl PartitionConfig {
    /// Construct a validated [`PartitionConfig`]. Returns an error if any
    /// partition-layer invariant is violated (see [`Self::validate_params`]).
    pub fn new(
        c_max: usize,
        c_min: usize,
        p_samp: f64,
        fanout: Vec<usize>,
        metric: diskann_vector::distance::Metric,
        leader_cap: usize,
    ) -> PiPNNResult<Self> {
        Self::validate_params(c_max, c_min, p_samp, &fanout, leader_cap)?;
        Ok(Self {
            c_max,
            c_min,
            p_samp,
            fanout,
            metric,
            leader_cap,
        })
    }

    /// Validate raw partition parameters without constructing.
    ///
    /// Owns the full set of partition-layer rules so that upstream config
    /// validators (e.g. [`crate::PiPNNConfig::validate`]) can fail-fast on bad
    /// inputs by calling this directly, while [`Self::new`] enforces the same
    /// rules at construction time.
    pub(crate) fn validate_params(
        c_max: usize,
        c_min: usize,
        p_samp: f64,
        fanout: &[usize],
        leader_cap: usize,
    ) -> PiPNNResult<()> {
        if c_max == 0 {
            return Err(PiPNNError::Config("c_max must be > 0".into()));
        }
        if c_min == 0 {
            return Err(PiPNNError::Config("c_min must be > 0".into()));
        }
        if c_min > c_max {
            return Err(PiPNNError::Config(format!(
                "c_min ({}) must be <= c_max ({})",
                c_min, c_max
            )));
        }
        if !p_samp.is_finite() {
            return Err(PiPNNError::Config("p_samp must be finite".into()));
        }
        if p_samp <= 0.0 || p_samp > 1.0 {
            return Err(PiPNNError::Config(format!(
                "p_samp ({}) must be in (0.0, 1.0]",
                p_samp
            )));
        }
        if fanout.is_empty() {
            return Err(PiPNNError::Config("fanout must not be empty".into()));
        }
        if fanout.contains(&0) {
            return Err(PiPNNError::Config("all fanout values must be > 0".into()));
        }
        if let Some(&over) = fanout.iter().find(|&&f| f > MAX_FANOUT) {
            return Err(PiPNNError::Config(format!(
                "fanout value {} exceeds MAX_FANOUT ({})",
                over, MAX_FANOUT
            )));
        }
        if leader_cap < 2 {
            return Err(PiPNNError::Config(format!(
                "leader_cap ({}) must be >= 2",
                leader_cap
            )));
        }
        Ok(())
    }
}

/// Compute the number of leaders to sample, capped by leader_cap.
#[inline]
fn sample_num_leaders(n: usize, p_samp: f64, leader_cap: usize) -> usize {
    ((n as f64 * p_samp).ceil() as usize)
        .clamp(2, leader_cap)
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

    let nl0 = sample_num_leaders(npoints, config.p_samp, config.leader_cap);
    tracing::info!(npoints, leaders = nl0, ndims, "Partition start");

    let mut leaves: Vec<Leaf> = Vec::new();
    let mut work = vec![WorkItem {
        indices: initial_indices,
        level: 0,
        seed,
    }];

    // Iteration cap: if the leader-based partition can't crack remaining
    // oversized clusters within `max_iter` rounds (e.g. CLIP-like hub geometry
    // where sampled leaders are themselves bunched in the dense region and
    // argmin assignment degenerates), accept them as oversized leaves and
    // hand off to the leaf builder.
    let max_iter: usize = MAX_PARTITION_ITER;
    let mut iteration = 0;
    while !work.is_empty() {
        iteration += 1;
        let work_points: usize = work.iter().map(|w| w.indices.len()).sum();
        tracing::info!(
            iter = iteration,
            work_items = work.len(),
            work_points,
            leaves = leaves.len(),
            "partition iter"
        );

        if iteration > max_iter {
            tracing::warn!(
                iter = iteration,
                work_items = work.len(),
                work_points,
                "partition iter cap hit, accepting remaining work as oversized leaves"
            );
            for item in work.drain(..) {
                leaves.push(Leaf {
                    indices: item.indices,
                });
            }
            break;
        }

        let results: Vec<(Vec<WorkItem>, Vec<Leaf>)> = work
            .into_par_iter()
            .map(|item| partition_one_level(data, ndims, config, item))
            .collect_installed();

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

    // Global merge of sub-c_min leaves across all work items / levels.
    // Eliminates the bug where per-call `merge_small` would leave one
    // < c_min remainder per partition_one_level call (10s of thousands of
    // tiny leaves at deep BFS levels in the prior implementation).
    global_merge_small(leaves, config.c_min, config.c_max)
}

/// Process one cluster: assign to leaders, emit oversized clusters as new work
/// items and the rest (including under-c_min) as leaves. Cross-work-item small
/// leaves are then combined by a single global `global_merge_small` pass at
/// the end of `partition()` — replaces the per-call `merge_small` which left
/// one < c_min remainder per work item (58K+ tiny leaves at deep levels).
fn partition_one_level<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    config: &PartitionConfig,
    item: WorkItem,
) -> (Vec<WorkItem>, Vec<Leaf>) {
    let n = item.indices.len();
    debug_assert!(n > config.c_max);

    // When recursion depth exceeds fanout.len(), collapse to a fanout of 1.
    let fanout = config.fanout.get(item.level).copied().unwrap_or(1).min(n);
    let num_leaders = sample_num_leaders(n, config.p_samp, config.leader_cap);

    // Deterministic seed derived from parent: no syscall, reproducible.
    let seed = item
        .seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(n as u64);
    let t_sample = std::time::Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let leaders: Vec<u32> = item
        .indices
        .choose_multiple(&mut rng, num_leaders)
        .copied()
        .collect();
    let sample_us = t_sample.elapsed().as_micros() as u64;

    // Assign each point to its `fanout` nearest leaders → per-leader clusters.
    let t_assign = std::time::Instant::now();
    let clusters = assign_to_leaders(
        data,
        ndims,
        &item.indices,
        &leaders,
        fanout,
        config.metric,
    );
    let assign_us = t_assign.elapsed().as_micros() as u64;

    let n_oversized: usize = clusters.iter().filter(|c| c.len() > config.c_max).count();
    let n_finished: usize = clusters.iter().filter(|c| !c.is_empty() && c.len() <= config.c_max).count();
    tracing::info!(
        level = item.level,
        n = n,
        num_leaders = num_leaders,
        fanout = fanout,
        sample_us = sample_us,
        assign_us = assign_us,
        n_oversized = n_oversized,
        n_finished = n_finished,
        "partition_one_level"
    );

    let mut next_work = Vec::new();
    let mut finished_leaves = Vec::new();
    for cluster in clusters {
        if cluster.is_empty() {
            continue;
        }
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

// ─── Thread-local stripe buffers ─────────────────────────────────────────────

use std::cell::RefCell;

/// Reusable per-thread buffers for partition stripe processing.
/// Avoids per-stripe alloc + memset on hot path.
struct StripeBuffers {
    p_data: Vec<f32>,
    dots: Vec<f32>,
}

impl StripeBuffers {
    fn new() -> Self {
        Self {
            p_data: Vec::new(),
            dots: Vec::new(),
        }
    }
}

thread_local! {
    static STRIPE_BUFS: RefCell<StripeBuffers> = RefCell::new(StripeBuffers::new());
}

/// Free this thread's partition stripe buffers (typically called after the
/// partition phase finishes, so they don't sum into the leaf-build /
/// HP-extract peak RSS). Mirror of [`crate::leaf_build::release_thread_buffers`].
pub(crate) fn release_thread_buffers() {
    STRIPE_BUFS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        bufs.p_data = Vec::new();
        bufs.dots = Vec::new();
    });
}

/// SIMD batch fp16 → fp32 gather. When `T` is `half::f16` (size 2, align 2),
/// runtime-cast `&[T]` to `&[u16]` and use AVX-512 `vcvtph2ps` to convert
/// 16 lanes per instruction. Faster than the generic `T::as_f32_into` path
/// for fp16 input.
///
/// SAFETY: only enters the SIMD path when `size_of::<T>() == 2` and
/// `align_of::<T>() == 2` — matches `half::f16`'s `repr(transparent) u16`
/// layout. Falls back to the generic trait for other `T`.
///
/// Dispatch: runtime-checked via [`cpu_dispatch::tier`]. The SIMD bodies are
/// `#[target_feature(enable = "...")]` `unsafe fn` so they generate the same
/// codegen as the previous compile-time `#[cfg(target_feature)]` paths
/// without requiring `target-cpu=native`. Hot-path call sites in
/// `assign_to_leaders` hoist the tier check to a fn pointer once per
/// closure invocation.
#[inline]
pub(crate) fn gather_f16_to_f32_simd<T: VectorRepr>(
    src: &[T], gi: usize, ndims: usize, dst: &mut [f32],
) {
    debug_assert!(dst.len() >= ndims);
    #[cfg(target_arch = "x86_64")]
    {
        if std::mem::size_of::<T>() == 2 && std::mem::align_of::<T>() == 2 {
            use crate::cpu_dispatch::{tier, SimdTier};
            let src_ptr = src.as_ptr() as *const u16;
            match tier() {
                SimdTier::Avx512 => {
                    // SAFETY: tier()==Avx512 implies AVX-512F at runtime, so
                    // the `#[target_feature]` precondition holds.
                    unsafe { gather_f16_avx512(src_ptr, gi, ndims, dst.as_mut_ptr()) };
                    return;
                }
                SimdTier::Avx2 => {
                    // SAFETY: tier()==Avx2 implies AVX2 + F16C at runtime.
                    unsafe { gather_f16_avx2_f16c(src_ptr, gi, ndims, dst.as_mut_ptr()) };
                    return;
                }
                SimdTier::Scalar => {}
            }
        }
    }
    let src_row = &src[gi * ndims..(gi + 1) * ndims];
    T::as_f32_into(src_row, &mut dst[..ndims]).unwrap_or_else(|e| panic!("VectorRepr::as_f32_into failed during partition: {}", e));
}

/// AVX-512 + F16C `vcvtph2ps` 16-lane fp16→fp32 row gather.
///
/// SAFETY: caller must guarantee AVX-512F is available at runtime (matched
/// by [`crate::cpu_dispatch::tier`] == `Avx512`). `src_ptr` must point to
/// at least `(gi + 1) * ndims` `u16` elements, and `dst_ptr` must be writable
/// for at least `ndims` `f32` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gather_f16_avx512(src_ptr: *const u16, gi: usize, ndims: usize, dst_ptr: *mut f32) {
    use std::arch::x86_64::*;
    // SAFETY: f16 is repr(transparent) u16; the raw slice aliases the
    // same bytes. ndims-sized rows of f16 at `gi * ndims` stay in
    // bounds by `src.len() >= (gi + 1) * ndims` which the caller
    // enforces via `data[gi * ndims..(gi+1) * ndims]` slicing.
    let row_ptr = src_ptr.add(gi * ndims);
    let chunks = ndims / 16;
    let tail = ndims - chunks * 16;
    for c in 0..chunks {
        let h = _mm256_loadu_si256(row_ptr.add(c * 16) as *const __m256i);
        let f = _mm512_cvtph_ps(h);
        _mm512_storeu_ps(dst_ptr.add(c * 16), f);
    }
    if tail > 0 {
        let kmask: u16 = (1u16 << tail) - 1;
        let h = _mm256_maskz_loadu_epi16(kmask, row_ptr.add(chunks * 16) as *const i16);
        let f = _mm512_cvtph_ps(h);
        _mm512_mask_storeu_ps(dst_ptr.add(chunks * 16), kmask, f);
    }
}

/// AVX2 + F16C 8-lane fp16→fp32 row gather. F16C is independent of AVX-512
/// and ships with every Ivy Bridge+ / AMD Bulldozer+ CPU.
///
/// SAFETY: caller must guarantee AVX2 + F16C are available at runtime
/// (matched by [`crate::cpu_dispatch::tier`] == `Avx2`). `src_ptr` must
/// point to at least `(gi + 1) * ndims` `u16` elements, and `dst_ptr` must
/// be writable for at least `ndims` `f32` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn gather_f16_avx2_f16c(src_ptr: *const u16, gi: usize, ndims: usize, dst_ptr: *mut f32) {
    use std::arch::x86_64::*;
    // SAFETY: see AVX-512 helper; same alias + bounds invariants.
    let row_ptr = src_ptr.add(gi * ndims);
    let chunks = ndims / 8;
    let tail = ndims - chunks * 8;
    for c in 0..chunks {
        let h = _mm_loadu_si128(row_ptr.add(c * 8) as *const __m128i);
        let f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst_ptr.add(c * 8), f);
    }
    if tail > 0 {
        // AVX2 has no masked 16-bit load; copy tail through a tiny
        // stack buffer (zero-padded) so the SIMD convert is safe.
        let mut hbuf = [0u16; 8];
        std::ptr::copy_nonoverlapping(row_ptr.add(chunks * 8), hbuf.as_mut_ptr(), tail);
        let h = _mm_loadu_si128(hbuf.as_ptr() as *const __m128i);
        let f = _mm256_cvtph_ps(h);
        let mut fbuf = [0f32; 8];
        _mm256_storeu_ps(fbuf.as_mut_ptr(), f);
        std::ptr::copy_nonoverlapping(fbuf.as_ptr(), dst_ptr.add(chunks * 8), tail);
    }
}

fn compute_p_norm_sq_batch(p_data: &[f32], np: usize, ndims: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; np];
    compute_p_norm_sq_into_impl(p_data, np, ndims, &mut out);
    out
}

/// Inner kernel shared by the `_into` and owning forms. Three arch paths:
/// AVX-512 (16-wide FMA), AVX2 (8-wide FMA), scalar fallback.
///
/// Dispatch is runtime-checked once per call via [`cpu_dispatch::tier`]; the
/// per-tier bodies are `#[target_feature(enable = "...")] unsafe fn` so
/// their SIMD codegen matches the prior compile-time `#[cfg]` paths without
/// requiring `target-cpu=native`. The match cost is amortized across `np`
/// FMA chains by the caller (called twice per partition pass).
#[inline(always)]
fn compute_p_norm_sq_into_impl(p_data: &[f32], np: usize, ndims: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        use crate::cpu_dispatch::{tier, SimdTier};
        match tier() {
            SimdTier::Avx512 => {
                // SAFETY: tier()==Avx512 implies AVX-512F at runtime.
                unsafe { compute_p_norm_sq_avx512(p_data, np, ndims, out) };
                return;
            }
            SimdTier::Avx2 => {
                // SAFETY: tier()==Avx2 implies AVX2 + FMA at runtime.
                unsafe { compute_p_norm_sq_avx2_fma(p_data, np, ndims, out) };
                return;
            }
            SimdTier::Scalar => {}
        }
    }
    // Pure-Rust fallback. LLVM auto-vectorizes on whatever the base
    // target-cpu supports (v3 baseline already has AVX2).
    for i in 0..np {
        let row = &p_data[i * ndims..(i + 1) * ndims];
        out[i] = row.iter().map(|v| v * v).sum();
    }
}

/// AVX-512 16-wide FMA ||p||² per row.
///
/// SAFETY: caller must guarantee AVX-512F is available at runtime, and
/// `p_data.len() >= np * ndims` and `out.len() >= np`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn compute_p_norm_sq_avx512(p_data: &[f32], np: usize, ndims: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let chunks = ndims / 16;
    let tail = ndims - chunks * 16;
    // SAFETY: AVX-512 enabled by target_feature; pointer arithmetic stays
    // in p_data's allocation since i < np and `p_data.len() >= np * ndims`
    // (caller invariant).
    for i in 0..np {
        let p = p_data.as_ptr().add(i * ndims);
        let mut acc = _mm512_setzero_ps();
        for c in 0..chunks {
            let v = _mm512_loadu_ps(p.add(c * 16));
            acc = _mm512_fmadd_ps(v, v, acc);
        }
        if tail > 0 {
            let kmask: u16 = (1u16 << tail) - 1;
            let v = _mm512_maskz_loadu_ps(kmask, p.add(chunks * 16));
            acc = _mm512_fmadd_ps(v, v, acc);
        }
        *out.get_unchecked_mut(i) = _mm512_reduce_add_ps(acc);
    }
}

/// AVX2 + FMA 8-wide ||p||² per row, dual-accumulator to feed both FMA units.
///
/// SAFETY: caller must guarantee AVX2 + FMA are available at runtime, and
/// `p_data.len() >= np * ndims` and `out.len() >= np`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_p_norm_sq_avx2_fma(p_data: &[f32], np: usize, ndims: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let chunks = ndims / 8;
    let tail = ndims - chunks * 8;
    // SAFETY: AVX2+FMA enabled by target_feature; same bounds as AVX-512 helper.
    for i in 0..np {
        let p = p_data.as_ptr().add(i * ndims);
        // Two independent accumulators to feed both FMA units.
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let pair_chunks = chunks / 2;
        for c in 0..pair_chunks {
            let v0 = _mm256_loadu_ps(p.add(c * 16));
            let v1 = _mm256_loadu_ps(p.add(c * 16 + 8));
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        }
        let leftover_chunk = pair_chunks * 2;
        if leftover_chunk < chunks {
            let v = _mm256_loadu_ps(p.add(leftover_chunk * 8));
            acc0 = _mm256_fmadd_ps(v, v, acc0);
        }
        let acc = _mm256_add_ps(acc0, acc1);
        // Horizontal reduce: 8 → 4 → 2 → 1.
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let s4 = _mm_add_ps(lo, hi);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0b01));
        let mut tail_sum = _mm_cvtss_f32(s1);
        if tail > 0 {
            let base = chunks * 8;
            for j in 0..tail {
                let v = *p.add(base + j);
                tail_sum += v * v;
            }
        }
        *out.get_unchecked_mut(i) = tail_sum;
    }
}

// ─── Assignment ──────────────────────────────────────────────────────────────

/// Assign each point to its `fanout` nearest leaders using native SIMD distance.
/// Point-by-point: no large temporary matrix, works directly on native type T.
/// All indices are u32 global IDs. Returns per-leader clusters as Vec<Vec<u32>>.
fn assign_to_leaders<T: VectorRepr + Send + Sync + 'static>(
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
        T::as_f32_into(src, &mut l_data[i * ndims..(i + 1) * ndims]).unwrap_or_else(|e| panic!("VectorRepr::as_f32_into failed during partition: {}", e));
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

    // Single-layer chunking with runtime MB sized to the detected L2 cache.
    // Matches v2's `assign_to_leaders_v2` structure: par_chunks_mut at MB
    // granularity, no inner mini-batch loop. The empirical ablation showed
    // the old stripe+inner-MB structure is equivalent at the same closure
    // body size — keeping the simpler single-layer form to avoid the dead
    // codegen bloat that previously bound v1 to its specific chunk grain.
    let l2 = crate::partition_inner::l2_size_bytes();
    let mb = crate::partition_inner::compute_mb(nl, l2);

    // Skip-MB path: whole problem comfortably fits L2 → one sequential GEMM.
    if crate::partition_inner::should_skip_mb(np, nl, l2) {
        STRIPE_BUFS.with(|cell| {
            let mut bufs = cell.borrow_mut();
            if bufs.p_data.len() < np * ndims {
                bufs.p_data.resize(np * ndims, 0.0);
            }
            if bufs.dots.len() < np * nl {
                bufs.dots.resize(np * nl, 0.0);
            }
            let StripeBuffers { ref mut p_data, ref mut dots } = *bufs;
            let p_slice = &mut p_data[..np * ndims];
            let dots_slice = &mut dots[..np * nl];
            for (i, &idx) in points.iter().enumerate() {
                gather_f16_to_f32_simd(data, idx as usize, ndims,
                    &mut p_slice[i * ndims..(i + 1) * ndims]);
            }
            diskann_linalg::sgemm_abt(p_slice, np, ndims, &l_data, nl, dots_slice);
            // Batch-precompute ||p||² for all rows in one tight SIMD loop —
            // hoisted out of `process_row` so the inner per-leader loop
            // doesn't recompute it every call.
            let p_norm_sq = compute_p_norm_sq_batch(p_slice, np, ndims);
            for i in 0..np {
                let dot_row = &dots_slice[i * nl..(i + 1) * nl];
                let out = &mut assignments[i * num_assign..(i + 1) * num_assign];
                crate::partition_inner::process_row(
                    dot_row, p_norm_sq[i], &l_norms, metric, num_assign, out,
                );
            }
        });
    } else {
        // Chunked path: par_chunks_mut at MB granularity, one GEMM per chunk,
        // top-k per row via process_row. STRIPE_BUFS provides per-thread
        // p_data / dots reuse across many chunks.
        let chunk_size = mb * num_assign;
        assignments
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each_installed(|(idx, assign_chunk)| {
                STRIPE_BUFS.with(|cell| {
                    let mut bufs = cell.borrow_mut();
                    let row_start = idx * mb;
                    let chunk_rows = (row_start + mb).min(np) - row_start;
                    if bufs.p_data.len() < chunk_rows * ndims {
                        bufs.p_data.resize(mb * ndims, 0.0);
                    }
                    if bufs.dots.len() < chunk_rows * nl {
                        bufs.dots.resize(mb * nl, 0.0);
                    }
                    let StripeBuffers { ref mut p_data, ref mut dots } = *bufs;
                    let p_slice = &mut p_data[..chunk_rows * ndims];
                    let dots_slice = &mut dots[..chunk_rows * nl];
                    for (i, &gi) in points[row_start..row_start + chunk_rows].iter().enumerate() {
                        gather_f16_to_f32_simd(data, gi as usize, ndims,
                            &mut p_slice[i * ndims..(i + 1) * ndims]);
                    }
                    diskann_linalg::sgemm_abt(p_slice, chunk_rows, ndims, &l_data, nl, dots_slice);
                    let p_norm_sq = compute_p_norm_sq_batch(p_slice, chunk_rows, ndims);
                    for i in 0..chunk_rows {
                        let dot_row = &dots_slice[i * nl..(i + 1) * nl];
                        let out = &mut assign_chunk[i * num_assign..(i + 1) * num_assign];
                        crate::partition_inner::process_row(
                            dot_row, p_norm_sq[i], &l_norms, metric, num_assign, out,
                        );
                    }
                });
            });
    }


    // Aggregate into per-leader clusters using global point IDs. For large
    // np, the serial scatter (np × num_assign pushes into per-leader Vecs)
    // is a real serial tail — parallelize via per-thread partial clusters
    // + per-leader merge.
    if np >= 100_000 {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk_size = np.div_ceil(n_threads);

        let partials: Vec<Vec<Vec<u32>>> = (0..n_threads)
            .into_par_iter()
            .map(|t| {
                let start = t * chunk_size;
                let end = ((t + 1) * chunk_size).min(np);
                let mut local: Vec<Vec<u32>> = (0..nl)
                    .map(|_| Vec::with_capacity(((end - start) * num_assign).div_ceil(nl)))
                    .collect();
                for i in start..end {
                    let pt = points[i];
                    let row = &assignments[i * num_assign..(i + 1) * num_assign];
                    for &leader_local in row {
                        local[leader_local as usize].push(pt);
                    }
                }
                local
            })
            .collect_installed();

        // Sum per-leader sizes once so each final Vec is allocated with
        // exact capacity (no realloc churn).
        let mut sizes = vec![0usize; nl];
        for partial in &partials {
            for (li, v) in partial.iter().enumerate() {
                sizes[li] += v.len();
            }
        }

        let clusters: Vec<Vec<u32>> = sizes
            .par_iter()
            .enumerate()
            .map(|(li, &sz)| {
                let mut out = Vec::with_capacity(sz);
                for partial in &partials {
                    out.extend_from_slice(&partial[li]);
                }
                out
            })
            .collect_installed();
        clusters
    } else {
        let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); nl];
        for (i, pt) in points.iter().enumerate() {
            let row = &assignments[i * num_assign..(i + 1) * num_assign];
            for &leader_local in row {
                clusters[leader_local as usize].push(*pt);
            }
        }
        clusters
    }
}

// ─── Global Merge of Small Leaves ────────────────────────────────────────────

/// Combine sub-c_min leaves into c_min/c_max-sized leaves via a HashSet buffer
/// that deduplicates overlapping point IDs. Big leaves (>= c_min) pass through
/// untouched. The final remainder, if still < c_min, is appended to the most
/// recent big leaf when it fits, otherwise emitted as a small tail leaf.
fn global_merge_small(leaves: Vec<Leaf>, c_min: usize, c_max: usize) -> Vec<Leaf> {
    let (mut good, small): (Vec<Leaf>, Vec<Leaf>) =
        leaves.into_iter().partition(|l| l.indices.len() >= c_min);

    if small.is_empty() {
        return good;
    }

    let mut buf = std::collections::HashSet::<u32>::with_capacity(c_max);
    for leaf in small {
        if buf.len() + leaf.indices.len() > c_max && buf.len() >= c_min {
            good.push(Leaf {
                indices: buf.drain().collect(),
            });
        }
        buf.extend(leaf.indices);
        if buf.len() >= c_min {
            good.push(Leaf {
                indices: buf.drain().collect(),
            });
        }
    }
    if !buf.is_empty() {
        let remainder: Vec<u32> = buf.into_iter().collect();
        if remainder.len() < c_min {
            // Try to attach to the last big leaf if it fits.
            if let Some(last) = good.last_mut() {
                if last.indices.len() + remainder.len() <= c_max {
                    last.indices.extend(remainder);
                    return good;
                }
            }
        }
        good.push(Leaf { indices: remainder });
    }

    good
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
            leader_cap: 1000,
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
            leader_cap: 1000,
        };
        let leaves = partition(&data, ndims, npoints, &config, 0);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].indices.len(), npoints);
    }

    // Baseline partition rules (c_max > 0, c_min <= c_max, p_samp range,
    // fanout non-empty / non-zero) are covered via PiPNNConfig::validate
    // delegate path in `builder::tests::test_config_validate`. Only the
    // partition-specific rules introduced alongside `validate_params` are
    // tested here.

    #[test]
    fn validate_params_rejects_fanout_above_max() {
        let err = PartitionConfig::validate_params(1024, 256, 0.05, &[MAX_FANOUT + 1], 1000)
            .expect_err("fanout > MAX_FANOUT must reject");
        assert!(format!("{err}").contains("MAX_FANOUT"));
        // Boundary: exactly MAX_FANOUT is accepted.
        PartitionConfig::validate_params(1024, 256, 0.05, &[MAX_FANOUT], 1000)
            .expect("fanout == MAX_FANOUT must accept");
    }

    #[test]
    fn validate_params_rejects_leader_cap_below_two() {
        for bad in [0usize, 1] {
            let err = PartitionConfig::validate_params(1024, 256, 0.05, &[10, 3], bad)
                .expect_err(&format!("leader_cap={bad} must reject"));
            assert!(format!("{err}").contains("leader_cap"));
        }
        // Boundary: exactly 2 is accepted (matches sample_num_leaders clamp floor).
        PartitionConfig::validate_params(1024, 256, 0.05, &[10, 3], 2)
            .expect("leader_cap == 2 must accept");
    }

    #[test]
    fn new_propagates_validate_params_error() {
        // Contract test for the public constructor: external callers (e.g.
        // benches) hit `new` rather than going through PiPNNConfig::validate.
        let err = PartitionConfig::new(
            1024,
            256,
            0.05,
            vec![MAX_FANOUT + 1],
            diskann_vector::distance::Metric::L2,
            1000,
        )
        .expect_err("PartitionConfig::new must propagate validate_params errors");
        assert!(format!("{err}").contains("MAX_FANOUT"));
    }
}
