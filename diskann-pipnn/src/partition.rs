/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Randomized Ball Carving (RBC) partitioning — iterative, parallel at every level.
//!
//! Recursively partitions the dataset into overlapping clusters using an iterative
//! work-queue approach. All oversized clusters at each level are processed in parallel.

use crate::{rayon_util::ParIterInstalled, PiPNNConfig};
use diskann::{utils::VectorRepr, ANNError, ANNResult};
use diskann_vector::{distance::Metric, Half};
use rand::prelude::IndexedRandom;
use rand::SeedableRng;
use rayon::prelude::*;

/// Maximum supported `fanout` value: hard upper bound on the size of the
/// stack-allocated top-k tracker [`assign_to_leaders`] uses on its hot path.
/// Enforced by [`crate::PiPNNConfig::validate`].
pub(crate) const MAX_FANOUT: usize = 16;

/// Insert a candidate into the sorted prefix of the fixed-size top-K buffer.
///
/// The partition kernels initialize the buffer with sentinels and pass
/// `threshold_idx = K - 1`, so the last live slot is always the current worst
/// candidate. Keeping this helper next to the partition implementation avoids
/// exposing a PiPNN-specific hot-loop primitive from `diskann-vector`.
#[inline(always)]
pub(crate) fn topk_insert<const K: usize>(
    top: &mut [(u32, f32); K],
    threshold_idx: usize,
    idx: u32,
    dist: f32,
) {
    if dist >= top[threshold_idx].1 {
        return;
    }
    top[threshold_idx] = (idx, dist);
    let mut position = threshold_idx;
    while position > 0 && top[position].1 < top[position - 1].1 {
        top.swap(position, position - 1);
        position -= 1;
    }
}

/// A leaf partition containing indices into the original dataset.
///
/// Uses `u32` instead of `usize` to halve memory on 64-bit platforms.
/// Sufficient for datasets up to 4 billion points.
#[derive(Debug, Clone)]
pub(crate) struct Leaf {
    pub indices: Vec<u32>,
}

/// Max iterations of the partition loop before remaining oversized clusters
/// are accepted as leaves. Guards against pathological hub geometries.
const MAX_PARTITION_ITER: usize = 30;

/// Per-partition-level leader hardcap (paper recommendation). Static — there is
/// no runtime override.
const LEADER_CAP: usize = 1000;

/// Compute the number of leaders to sample.
#[inline]
fn sample_num_leaders(n: usize, p_samp: f64) -> usize {
    ((n as f64 * p_samp).ceil() as usize)
        .clamp(2, LEADER_CAP)
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
pub(crate) fn partition<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    npoints: usize,
    config: &PiPNNConfig,
    metric: Metric,
    seed: u64,
) -> ANNResult<Vec<Leaf>> {
    let initial_indices: Vec<u32> = (0..npoints as u32).collect();

    if npoints <= config.c_max {
        return Ok(vec![Leaf {
            indices: initial_indices,
        }]);
    }

    let nl0 = sample_num_leaders(npoints, config.p_samp);
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

        let results: ANNResult<Vec<(Vec<WorkItem>, Vec<Leaf>)>> = work
            .into_par_iter()
            .map(|item| partition_one_level(data, ndims, config, metric, item))
            .collect_installed();
        let results = results?;

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

    // Merge sub-c_min leaves once across all work items and levels.
    Ok(global_merge_small(leaves, config.c_min, config.c_max))
}

/// Process one cluster: assign to leaders, emit oversized clusters as new work
/// items and the rest (including under-c_min) as leaves. Cross-work-item small
/// leaves are combined by one `global_merge_small` pass at the end.
fn partition_one_level<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    config: &PiPNNConfig,
    metric: Metric,
    item: WorkItem,
) -> ANNResult<(Vec<WorkItem>, Vec<Leaf>)> {
    let n = item.indices.len();
    debug_assert!(n > config.c_max);

    // When recursion depth exceeds fanout.len(), collapse to a fanout of 1.
    let fanout = config.fanout.get(item.level).copied().unwrap_or(1).min(n);
    let num_leaders = sample_num_leaders(n, config.p_samp);

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
    let clusters = assign_to_leaders(data, ndims, &item.indices, &leaders, fanout, metric)?;
    let assign_us = t_assign.elapsed().as_micros() as u64;

    let n_oversized: usize = clusters.iter().filter(|c| c.len() > config.c_max).count();
    let n_finished: usize = clusters
        .iter()
        .filter(|c| !c.is_empty() && c.len() <= config.c_max)
        .count();
    tracing::debug!(
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
    Ok((next_work, finished_leaves))
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
    STRIPE_BUFS.with(|cell| *cell.borrow_mut() = StripeBuffers::new());
}

/// Gather selected rows into a contiguous full-precision matrix. The exact
/// half type gets a chunk-level SIMD path; every other representation uses
/// its checked conversion implementation.
pub(crate) fn gather_rows<T: VectorRepr + 'static>(
    data: &[T],
    indices: &[u32],
    ndims: usize,
    dst: &mut [f32],
) -> ANNResult<()> {
    let expected_len = indices
        .len()
        .checked_mul(ndims)
        .ok_or_else(|| crate::config_error("gather output shape overflows usize"))?;
    if dst.len() != expected_len {
        return Err(crate::config_error(format!(
            "gather output length mismatch: expected {expected_len}, got {}",
            dst.len()
        )));
    }
    for &index in indices {
        let end = (index as usize + 1)
            .checked_mul(ndims)
            .ok_or_else(|| crate::config_error("row offset overflow during gather"))?;
        if end > data.len() {
            return Err(crate::config_error(format!(
                "row index {index} is outside a {}-element matrix with {ndims} columns",
                data.len()
            )));
        }
    }

    #[cfg(target_arch = "x86_64")]
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<Half>() {
        let src = data.as_ptr().cast::<u16>();
        let width = crate::cpu_dispatch::half_width();
        match width {
            // SAFETY: feature selection checked AVX-512F; row bounds and the
            // destination length were validated above.
            crate::cpu_dispatch::VectorWidth::Wide => unsafe {
                gather_half_rows_wide(src, indices, ndims, dst.as_mut_ptr())
            },
            // SAFETY: feature selection checked AVX2 + F16C and the same
            // bounds validation applies.
            crate::cpu_dispatch::VectorWidth::Narrow => unsafe {
                gather_half_rows_narrow(src, indices, ndims, dst.as_mut_ptr())
            },
            crate::cpu_dispatch::VectorWidth::Scalar => {}
        }
        if !matches!(width, crate::cpu_dispatch::VectorWidth::Scalar) {
            return Ok(());
        }
    }

    for (&index, out) in indices.iter().zip(dst.chunks_exact_mut(ndims)) {
        let start = index as usize * ndims;
        T::as_f32_into(&data[start..start + ndims], out).map_err(Into::<ANNError>::into)?;
    }
    Ok(())
}

/// AVX-512 + F16C `vcvtph2ps` 16-lane fp16→fp32 row gather.
///
/// SAFETY: caller has checked AVX-512F and every input row bound; `dst_ptr`
/// spans `indices.len() * ndims` writable `f32` values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gather_half_rows_wide(
    src_ptr: *const u16,
    indices: &[u32],
    ndims: usize,
    dst_ptr: *mut f32,
) {
    use std::arch::x86_64::*;
    // SAFETY: f16 is repr(transparent) u16; the raw slice aliases the
    // same bytes. ndims-sized rows of f16 at `gi * ndims` stay in
    // bounds by `src.len() >= (gi + 1) * ndims` which the caller
    // enforces via `data[gi * ndims..(gi+1) * ndims]` slicing.
    for (row, &index) in indices.iter().enumerate() {
        let row_ptr = src_ptr.add(index as usize * ndims);
        let out = dst_ptr.add(row * ndims);
        let chunks = ndims / 16;
        let tail = ndims - chunks * 16;
        for c in 0..chunks {
            let h = _mm256_loadu_si256(row_ptr.add(c * 16) as *const __m256i);
            _mm512_storeu_ps(out.add(c * 16), _mm512_cvtph_ps(h));
        }
        if tail > 0 {
            let mut hbuf = [0u16; 16];
            std::ptr::copy_nonoverlapping(row_ptr.add(chunks * 16), hbuf.as_mut_ptr(), tail);
            let h = _mm256_loadu_si256(hbuf.as_ptr() as *const __m256i);
            let mut fbuf = [0f32; 16];
            _mm512_storeu_ps(fbuf.as_mut_ptr(), _mm512_cvtph_ps(h));
            std::ptr::copy_nonoverlapping(fbuf.as_ptr(), out.add(chunks * 16), tail);
        }
    }
}

/// AVX2 + F16C 8-lane fp16→fp32 row gather. F16C is independent of AVX-512
/// and ships with every Ivy Bridge+ / AMD Bulldozer+ CPU.
///
/// SAFETY: caller has checked AVX2 + F16C and every input row bound;
/// `dst_ptr` spans `indices.len() * ndims` writable `f32` values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn gather_half_rows_narrow(
    src_ptr: *const u16,
    indices: &[u32],
    ndims: usize,
    dst_ptr: *mut f32,
) {
    use std::arch::x86_64::*;
    // SAFETY: see AVX-512 helper; same alias + bounds invariants.
    for (row, &index) in indices.iter().enumerate() {
        let row_ptr = src_ptr.add(index as usize * ndims);
        let out = dst_ptr.add(row * ndims);
        let chunks = ndims / 8;
        let tail = ndims - chunks * 8;
        for c in 0..chunks {
            let h = _mm_loadu_si128(row_ptr.add(c * 8) as *const __m128i);
            _mm256_storeu_ps(out.add(c * 8), _mm256_cvtph_ps(h));
        }
        if tail > 0 {
            let mut hbuf = [0u16; 8];
            std::ptr::copy_nonoverlapping(row_ptr.add(chunks * 8), hbuf.as_mut_ptr(), tail);
            let h = _mm_loadu_si128(hbuf.as_ptr() as *const __m128i);
            let mut fbuf = [0f32; 8];
            _mm256_storeu_ps(fbuf.as_mut_ptr(), _mm256_cvtph_ps(h));
            std::ptr::copy_nonoverlapping(fbuf.as_ptr(), out.add(chunks * 8), tail);
        }
    }
}

fn compute_p_norm_sq_batch(p_data: &[f32], np: usize, ndims: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; np];
    compute_p_norm_sq_into_impl(p_data, np, ndims, &mut out);
    out
}

/// Detect the CPU's private L2 cache size, falling back to 512 KiB.
fn l2_size_bytes() -> usize {
    use std::sync::OnceLock;
    static L2_SIZE: OnceLock<usize> = OnceLock::new();
    *L2_SIZE.get_or_init(|| {
        #[cfg(target_os = "linux")]
        if let Ok(value) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size")
        {
            let value = value.trim();
            let parsed = value
                .strip_suffix('K')
                .and_then(|n| n.parse::<usize>().ok().map(|n| n * 1024))
                .or_else(|| {
                    value
                        .strip_suffix('M')
                        .and_then(|n| n.parse::<usize>().ok().map(|n| n * 1024 * 1024))
                })
                .or_else(|| value.parse().ok());
            if let Some(bytes) = parsed {
                return bytes;
            }
        }
        512 * 1024
    })
}

/// Choose a power-of-two row batch whose dot-product tile fits in private L2.
fn assignment_batch_rows(leaders: usize, l2_bytes: usize) -> usize {
    let max_rows = (l2_bytes / (leaders.max(1) * std::mem::size_of::<f32>())).max(32);
    let rows = if max_rows.is_power_of_two() {
        max_rows
    } else {
        max_rows.next_power_of_two() / 2
    };
    rows.clamp(32, 1024)
}

fn assignments_fit_l2(points: usize, leaders: usize, l2_bytes: usize) -> bool {
    (points as u64)
        .saturating_mul(leaders as u64)
        .saturating_mul(std::mem::size_of::<f32>() as u64)
        < l2_bytes as u64 / 2
}

/// Inner kernel shared by the `_into` and owning forms. Feature selection
/// happens once for the whole batch.
#[inline(always)]
fn compute_p_norm_sq_into_impl(p_data: &[f32], np: usize, ndims: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        use crate::cpu_dispatch::{fma_width, VectorWidth};
        match fma_width() {
            VectorWidth::Wide => {
                // SAFETY: fma_width()==Wide implies AVX-512F at runtime.
                unsafe { compute_p_norm_sq_wide(p_data, np, ndims, out) };
                return;
            }
            VectorWidth::Narrow => {
                // SAFETY: fma_width()==Narrow implies AVX2 + FMA at runtime.
                unsafe { compute_p_norm_sq_narrow(p_data, np, ndims, out) };
                return;
            }
            VectorWidth::Scalar => {}
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
unsafe fn compute_p_norm_sq_wide(p_data: &[f32], np: usize, ndims: usize, out: &mut [f32]) {
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
unsafe fn compute_p_norm_sq_narrow(p_data: &[f32], np: usize, ndims: usize, out: &mut [f32]) {
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
) -> ANNResult<Vec<Vec<u32>>> {
    let np = points.len();
    let nl = leaders.len();
    let num_assign = fanout.min(nl);

    use diskann_vector::distance::Metric;

    // Extract leader data into contiguous f32 array.
    let mut l_data = vec![0.0f32; nl * ndims];
    gather_rows(data, leaders, ndims, &mut l_data)?;

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

    // Use one chunking layer sized from the detected private L2 cache.
    let l2 = l2_size_bytes();
    let mb = assignment_batch_rows(nl, l2);

    // If the whole assignment tile fits L2, use one sequential GEMM.
    if assignments_fit_l2(np, nl, l2) {
        STRIPE_BUFS.with(|cell| -> ANNResult<()> {
            let mut bufs = cell.borrow_mut();
            if bufs.p_data.len() < np * ndims {
                bufs.p_data.resize(np * ndims, 0.0);
            }
            if bufs.dots.len() < np * nl {
                bufs.dots.resize(np * nl, 0.0);
            }
            let StripeBuffers {
                ref mut p_data,
                ref mut dots,
            } = *bufs;
            let p_slice = &mut p_data[..np * ndims];
            let dots_slice = &mut dots[..np * nl];
            gather_rows(data, points, ndims, p_slice)?;
            diskann_linalg::sgemm(
                diskann_linalg::Transpose::None,
                diskann_linalg::Transpose::Ordinary,
                np,
                nl,
                ndims,
                1.0,
                p_slice,
                &l_data,
                None,
                dots_slice,
            )
            .map_err(ANNError::opaque)?;
            // ||p||² is a per-point constant, so it shifts every leader's L2
            // distance by the same amount and can't change the top-k ranking;
            // CosineNormalized/InnerProduct ignore it too. Only Cosine (which
            // divides by ‖p‖) actually needs it — skip the batch reduce otherwise.
            let p_norm_sq = if matches!(metric, Metric::Cosine) {
                compute_p_norm_sq_batch(p_slice, np, ndims)
            } else {
                Vec::new()
            };
            crate::partition_inner::process_rows(
                dots_slice,
                &p_norm_sq,
                &l_norms,
                metric,
                num_assign,
                &mut assignments,
            );
            Ok(())
        })?;
    } else {
        // Otherwise run one GEMM and top-k pass per chunk. STRIPE_BUFS reuses
        // p_data and dots across chunks on each worker.
        let chunk_size = mb * num_assign;
        assignments
            .par_chunks_mut(chunk_size)
            .enumerate()
            .map(|(idx, assign_chunk)| {
                STRIPE_BUFS.with(|cell| -> ANNResult<()> {
                    let mut bufs = cell.borrow_mut();
                    let row_start = idx * mb;
                    let chunk_rows = (row_start + mb).min(np) - row_start;
                    if bufs.p_data.len() < chunk_rows * ndims {
                        bufs.p_data.resize(mb * ndims, 0.0);
                    }
                    if bufs.dots.len() < chunk_rows * nl {
                        bufs.dots.resize(mb * nl, 0.0);
                    }
                    let StripeBuffers {
                        ref mut p_data,
                        ref mut dots,
                    } = *bufs;
                    let p_slice = &mut p_data[..chunk_rows * ndims];
                    let dots_slice = &mut dots[..chunk_rows * nl];
                    gather_rows(
                        data,
                        &points[row_start..row_start + chunk_rows],
                        ndims,
                        p_slice,
                    )?;
                    diskann_linalg::sgemm(
                        diskann_linalg::Transpose::None,
                        diskann_linalg::Transpose::Ordinary,
                        chunk_rows,
                        nl,
                        ndims,
                        1.0,
                        p_slice,
                        &l_data,
                        None,
                        dots_slice,
                    )
                    .map_err(ANNError::opaque)?;
                    // Only Cosine needs ‖p‖; L2's constant
                    // ‖p‖² offset can't reorder the top-k.
                    let p_norm_sq = if matches!(metric, Metric::Cosine) {
                        compute_p_norm_sq_batch(p_slice, chunk_rows, ndims)
                    } else {
                        Vec::new()
                    };
                    crate::partition_inner::process_rows(
                        dots_slice,
                        &p_norm_sq,
                        &l_norms,
                        metric,
                        num_assign,
                        assign_chunk,
                    );
                    Ok(())
                })
            })
            .collect_installed::<ANNResult<()>>()?;
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

        // `assignments` is dead once `partials` holds the scattered IDs. Free
        // it before allocating the final clusters to reduce phase overlap.
        drop(assignments);

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
        Ok(clusters)
    } else {
        let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); nl];
        for (i, pt) in points.iter().enumerate() {
            let row = &assignments[i * num_assign..(i + 1) * num_assign];
            for &leader_local in row {
                clusters[leader_local as usize].push(*pt);
            }
        }
        Ok(clusters)
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
mod tests;
