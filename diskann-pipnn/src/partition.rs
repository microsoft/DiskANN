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
    c_max: usize,
    c_min: usize,
    p_samp: f64,
    fanout: Vec<usize>,
    metric: diskann_vector::distance::Metric,
    leader_cap: usize,
}

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

use diskann_vector::topk::topk_insert;

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

    let mut iteration = 0;
    while !work.is_empty() {
        iteration += 1;
        // 50 iterations is a safety net against a partitioning loop divergence.
        // Leaves halve in size each level, so 50 covers > 2^50 points in
        // healthy runs. If we hit this the partition config is degenerate.
        assert!(
            iteration <= 50,
            "partition diverged at iteration {} (npoints={}, c_max={}, c_min={}, fanout={:?})",
            iteration,
            npoints,
            config.c_max,
            config.c_min,
            config.fanout
        );

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

    let fanout = config.fanout.get(item.level).copied().unwrap_or(1).min(n);
    let num_leaders = sample_num_leaders(n, config.p_samp, config.leader_cap);

    // Deterministic seed derived from parent: no syscall, reproducible.
    let seed = item
        .seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(n as u64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let leaders: Vec<u32> = item
        .indices
        .choose_multiple(&mut rng, num_leaders)
        .copied()
        .collect();

    // Assign each point to its `fanout` nearest leaders → per-leader clusters.
    let clusters = assign_to_leaders(data, ndims, &item.indices, &leaders, fanout, config.metric);

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

    // Outer stripe size: granularity for rayon parallelism only.
    // We mini-batch INSIDE the stripe so the dots tile stays L2-resident.
    let stripe: usize =
        ((16 * 1024 * 1024) / (nl.max(1) * std::mem::size_of::<f32>())).clamp(1, np);

    // Mini-batch size for fused GEMM + top-k. Sized so the dots tile
    // (MB × nl × 4 bytes) stays in private L2 (~2 MB per core).
    const MINI_BATCH: usize = 128;

    // Small clusters: run single-threaded (no rayon overhead).
    // Large clusters: parallel stripes.
    // Uses thread-local buffers to avoid per-stripe alloc + memset.
    let process_stripe = |(stripe_idx, assign_chunk): (usize, &mut [u32])| {
        STRIPE_BUFS.with(|cell| {
            let mut bufs = cell.borrow_mut();
            let start = stripe_idx * stripe;
            let end = (start + stripe).min(np);
            let sn = end - start;
            let stripe_points = &points[start..end];

            // Buffers sized for one MINI_BATCH only (not full stripe) so the
            // dots tile stays in private L2.
            let mb_pd_len = MINI_BATCH * ndims;
            if bufs.p_data.len() < mb_pd_len {
                bufs.p_data.resize(mb_pd_len, 0.0);
            }
            let mb_dots_len = MINI_BATCH * nl;
            if bufs.dots.len() < mb_dots_len {
                bufs.dots.resize(mb_dots_len, 0.0);
            }
            // Destructure to allow simultaneous mutable borrows of different fields.
            let StripeBuffers {
                ref mut p_data,
                ref mut dots,
            } = *bufs;

            // Fused distance + top-k: compute distance AND track top-k in single pass.
            // Keeps hot top-k array in registers. Capped at MAX_FANOUT by
            // PiPNNConfig::validate, so num_assign <= MAX_FANOUT.
            let mut top: [(u32, f32); MAX_FANOUT] = [(u32::MAX, f32::MAX); MAX_FANOUT];

            // Outer loop over mini-batches inside the stripe. Each mini-batch:
            //  1. Gather mb rows of p_data (f16→f32 via as_f32_into)
            //  2. One GEMM call producing dots[..mb*nl]
            //  3. Top-k pass over mb rows — dots stays in L2 throughout
            let mut mb_start = 0usize;
            while mb_start < sn {
                let mb = (sn - mb_start).min(MINI_BATCH);
                let mb_points = &stripe_points[mb_start..mb_start + mb];

                // Gather + GEMM: dots[i * nl + j] = dot(point_i, leader_j) for i in 0..mb.
                {
                    let p32 = &mut p_data[..mb * ndims];
                    for (i, &idx) in mb_points.iter().enumerate() {
                        let src = &data[idx as usize * ndims..(idx as usize + 1) * ndims];
                        T::as_f32_into(src, &mut p32[i * ndims..(i + 1) * ndims])
                            .expect("f32 conversion");
                    }
                    let dots_mb = &mut dots[..mb * nl];
                    diskann_linalg::sgemm_abt(p32, mb, ndims, &l_data, nl, dots_mb);
                }

                for i in 0..mb {
                    let dot_row = &dots[i * nl..(i + 1) * nl];

                    for t in top[..num_assign].iter_mut() {
                        *t = (u32::MAX, f32::MAX);
                    }
                    let threshold_idx = num_assign - 1;

                    match metric {
                        Metric::CosineNormalized => {
                            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                            {
                                use std::arch::x86_64::*;
                                let chunks = nl / 16;
                                unsafe {
                                    let one = _mm512_set1_ps(1.0);
                                    for chunk in 0..chunks {
                                        let base = chunk * 16;
                                        let thresh = _mm512_set1_ps(top[threshold_idx].1);
                                        let dots = _mm512_loadu_ps(dot_row.as_ptr().add(base));
                                        let d = _mm512_sub_ps(one, dots);
                                        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(d, thresh);
                                        if mask != 0 {
                                            let mut d_arr = [0.0f32; 16];
                                            _mm512_storeu_ps(d_arr.as_mut_ptr(), d);
                                            let mut m = mask;
                                            while m != 0 {
                                                let lane = m.trailing_zeros() as usize;
                                                m &= m - 1;
                                                let j = base + lane;
                                                let dist = d_arr[lane];
                                                if dist < top[threshold_idx].1 {
                                                    top[threshold_idx] = (j as u32, dist);
                                                    let mut t = threshold_idx;
                                                    while t > 0 && top[t].1 < top[t - 1].1 {
                                                        top.swap(t, t - 1);
                                                        t -= 1;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    for j in (chunks * 16)..nl {
                                        let d = 1.0 - *dot_row.get_unchecked(j);
                                        if d < top[threshold_idx].1 {
                                            top[threshold_idx] = (j as u32, d);
                                            let mut t = threshold_idx;
                                            while t > 0 && top[t].1 < top[t - 1].1 {
                                                top.swap(t, t - 1);
                                                t -= 1;
                                            }
                                        }
                                    }
                                }
                            }
                            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
                            {
                                use std::arch::x86_64::*;
                                let chunks = nl / 8;
                                // SAFETY: AVX2 cfg-gated. `dot_row.as_ptr().add(base)`
                                // stays in bounds because `base + 8 ≤ nl = dot_row.len()`.
                                // Tail loop uses `j < nl` and `get_unchecked`.
                                unsafe {
                                    let one = _mm256_set1_ps(1.0);
                                    for chunk in 0..chunks {
                                        let base = chunk * 8;
                                        let thresh = _mm256_set1_ps(top[threshold_idx].1);
                                        let dots = _mm256_loadu_ps(dot_row.as_ptr().add(base));
                                        let d = _mm256_sub_ps(one, dots);
                                        let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(
                                            d, thresh,
                                        ));
                                        if mask != 0 {
                                            let mut d_arr = [0.0f32; 8];
                                            _mm256_storeu_ps(d_arr.as_mut_ptr(), d);
                                            let mut m = mask as u32;
                                            while m != 0 {
                                                let lane = m.trailing_zeros() as usize;
                                                m &= m - 1;
                                                let j = base + lane;
                                                let dist = d_arr[lane];
                                                if dist < top[threshold_idx].1 {
                                                    top[threshold_idx] = (j as u32, dist);
                                                    let mut t = threshold_idx;
                                                    while t > 0 && top[t].1 < top[t - 1].1 {
                                                        top.swap(t, t - 1);
                                                        t -= 1;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    for j in (chunks * 8)..nl {
                                        let d = 1.0 - *dot_row.get_unchecked(j);
                                        if d < top[threshold_idx].1 {
                                            top[threshold_idx] = (j as u32, d);
                                            let mut t = threshold_idx;
                                            while t > 0 && top[t].1 < top[t - 1].1 {
                                                top.swap(t, t - 1);
                                                t -= 1;
                                            }
                                        }
                                    }
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                for j in 0..nl {
                                    let d = 1.0 - unsafe { *dot_row.get_unchecked(j) };
                                    if d < top[threshold_idx].1 {
                                        top[threshold_idx] = (j as u32, d);
                                        let mut t = threshold_idx;
                                        while t > 0 && top[t].1 < top[t - 1].1 {
                                            top.swap(t, t - 1);
                                            t -= 1;
                                        }
                                    }
                                }
                            }
                        }
                        Metric::Cosine => {
                            let pi_sqrt: f32 = p_data[i * ndims..(i + 1) * ndims]
                                .iter()
                                .map(|v| v * v)
                                .sum::<f32>()
                                .sqrt();
                            // d = 1 - dot / (pi_sqrt * l_norms[j]). pi_sqrt is computed
                            // once per row and l_norms is precomputed once; the inner
                            // loop is one mul / one div / one sub per lane.
                            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                            {
                                use std::arch::x86_64::*;
                                let chunks = nl / 16;
                                // SAFETY: AVX-512 cfg-gated. dot_row.len() == nl,
                                // l_norms.len() == nl, base + 16 ≤ nl.
                                unsafe {
                                    let one = _mm512_set1_ps(1.0);
                                    let pi_v = _mm512_set1_ps(pi_sqrt);
                                    let zero = _mm512_setzero_ps();
                                    for chunk in 0..chunks {
                                        let base = chunk * 16;
                                        let thresh = _mm512_set1_ps(top[threshold_idx].1);
                                        let dots = _mm512_loadu_ps(dot_row.as_ptr().add(base));
                                        let ln = _mm512_loadu_ps(l_norms.as_ptr().add(base));
                                        let denom = _mm512_mul_ps(pi_v, ln);
                                        let denom_mask =
                                            _mm512_cmp_ps_mask::<_CMP_GT_OQ>(denom, zero);
                                        let cos = _mm512_mask_div_ps(zero, denom_mask, dots, denom);
                                        let d = _mm512_sub_ps(one, cos);
                                        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(d, thresh);
                                        if mask != 0 {
                                            let mut d_arr = [0.0f32; 16];
                                            _mm512_storeu_ps(d_arr.as_mut_ptr(), d);
                                            let mut m = mask;
                                            while m != 0 {
                                                let lane = m.trailing_zeros() as usize;
                                                m &= m - 1;
                                                let j = (base + lane) as u32;
                                                topk_insert(
                                                    &mut top,
                                                    threshold_idx,
                                                    j,
                                                    d_arr[lane],
                                                );
                                            }
                                        }
                                    }
                                    for j in (chunks * 16)..nl {
                                        let dot = *dot_row.get_unchecked(j);
                                        let ln = *l_norms.get_unchecked(j);
                                        let denom = pi_sqrt * ln;
                                        let cos = if denom > 0.0 { dot / denom } else { 0.0 };
                                        topk_insert(&mut top, threshold_idx, j as u32, 1.0 - cos);
                                    }
                                }
                            }
                            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
                            {
                                use std::arch::x86_64::*;
                                let chunks = nl / 8;
                                // SAFETY: AVX2 cfg-gated; same bounds as AVX-512 with stride 8.
                                unsafe {
                                    let one = _mm256_set1_ps(1.0);
                                    let pi_v = _mm256_set1_ps(pi_sqrt);
                                    let zero = _mm256_setzero_ps();
                                    for chunk in 0..chunks {
                                        let base = chunk * 8;
                                        let thresh = _mm256_set1_ps(top[threshold_idx].1);
                                        let dots = _mm256_loadu_ps(dot_row.as_ptr().add(base));
                                        let ln = _mm256_loadu_ps(l_norms.as_ptr().add(base));
                                        let denom = _mm256_mul_ps(pi_v, ln);
                                        let div = _mm256_div_ps(dots, denom);
                                        // No masked div on AVX2 — zero out lanes where denom == 0.
                                        let zero_mask = _mm256_cmp_ps::<_CMP_GT_OQ>(denom, zero);
                                        let cos = _mm256_and_ps(div, zero_mask);
                                        let d = _mm256_sub_ps(one, cos);
                                        let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(
                                            d, thresh,
                                        ));
                                        if mask != 0 {
                                            let mut d_arr = [0.0f32; 8];
                                            _mm256_storeu_ps(d_arr.as_mut_ptr(), d);
                                            let mut m = mask as u32;
                                            while m != 0 {
                                                let lane = m.trailing_zeros() as usize;
                                                m &= m - 1;
                                                let j = (base + lane) as u32;
                                                topk_insert(
                                                    &mut top,
                                                    threshold_idx,
                                                    j,
                                                    d_arr[lane],
                                                );
                                            }
                                        }
                                    }
                                    for j in (chunks * 8)..nl {
                                        let dot = *dot_row.get_unchecked(j);
                                        let ln = *l_norms.get_unchecked(j);
                                        let denom = pi_sqrt * ln;
                                        let cos = if denom > 0.0 { dot / denom } else { 0.0 };
                                        topk_insert(&mut top, threshold_idx, j as u32, 1.0 - cos);
                                    }
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            for (j, &ln) in l_norms.iter().enumerate().take(nl) {
                                // SAFETY: `dot_row.len() == nl` and `j < nl`.
                                let dot = unsafe { *dot_row.get_unchecked(j) };
                                let denom = pi_sqrt * ln;
                                let cos = if denom > 0.0 { dot / denom } else { 0.0 };
                                topk_insert(&mut top, threshold_idx, j as u32, 1.0 - cos);
                            }
                        }
                        Metric::L2 => {
                            let pi: f32 = p_data[i * ndims..(i + 1) * ndims]
                                .iter()
                                .map(|v| v * v)
                                .sum();
                            // Process 16 leaders at a time using AVX-512: compute
                            // distances in SIMD, only drop to scalar for the rare
                            // lanes that beat the current threshold.
                            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                            {
                                use std::arch::x86_64::*;
                                let chunks = nl / 16;
                                // SAFETY: AVX-512 cfg-gated. `dot_row.as_ptr().add(base)` and
                                // `l_norms.as_ptr().add(base)` stay in bounds: `base + 16 ≤ nl`,
                                // and both slices span ≥ nl floats.
                                unsafe {
                                    let pi_v = _mm512_set1_ps(pi);
                                    let two = _mm512_set1_ps(2.0);
                                    for chunk in 0..chunks {
                                        let base = chunk * 16;
                                        let thresh = _mm512_set1_ps(top[threshold_idx].1);
                                        let norms = _mm512_loadu_ps(l_norms.as_ptr().add(base));
                                        let dots = _mm512_loadu_ps(dot_row.as_ptr().add(base));
                                        // d = pi + norms - 2*dots
                                        let d =
                                            _mm512_add_ps(pi_v, _mm512_fnmadd_ps(two, dots, norms));
                                        // mask = lanes where d < threshold
                                        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(d, thresh);
                                        if mask != 0 {
                                            // Extract passing lanes to scalar for top-k insertion.
                                            // Typically 0-2 lanes pass out of 16.
                                            let mut d_arr = [0.0f32; 16];
                                            _mm512_storeu_ps(d_arr.as_mut_ptr(), d);
                                            let mut m = mask;
                                            while m != 0 {
                                                let lane = m.trailing_zeros() as usize;
                                                m &= m - 1; // clear lowest bit
                                                let j = base + lane;
                                                let dist = d_arr[lane];
                                                if dist < top[threshold_idx].1 {
                                                    top[threshold_idx] = (j as u32, dist);
                                                    let mut t = threshold_idx;
                                                    while t > 0 && top[t].1 < top[t - 1].1 {
                                                        top.swap(t, t - 1);
                                                        t -= 1;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // Handle remainder with scalar loop.
                                    for j in (chunks * 16)..nl {
                                        let dot = *dot_row.get_unchecked(j);
                                        let d = pi + *l_norms.get_unchecked(j) - 2.0 * dot;
                                        if d < top[threshold_idx].1 {
                                            top[threshold_idx] = (j as u32, d);
                                            let mut t = threshold_idx;
                                            while t > 0 && top[t].1 < top[t - 1].1 {
                                                top.swap(t, t - 1);
                                                t -= 1;
                                            }
                                        }
                                    }
                                }
                            }
                            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
                            {
                                use std::arch::x86_64::*;
                                let chunks = nl / 8;
                                // SAFETY: AVX2 cfg-gated. `dot_row.as_ptr().add(base)` and
                                // `l_norms.as_ptr().add(base)` stay in bounds: `base + 8 ≤
                                // nl`, and both slices span ≥ nl floats.
                                unsafe {
                                    let pi_v = _mm256_set1_ps(pi);
                                    let two = _mm256_set1_ps(2.0);
                                    for chunk in 0..chunks {
                                        let base = chunk * 8;
                                        let thresh = _mm256_set1_ps(top[threshold_idx].1);
                                        let norms = _mm256_loadu_ps(l_norms.as_ptr().add(base));
                                        let dots = _mm256_loadu_ps(dot_row.as_ptr().add(base));
                                        // d = pi + (norms - 2*dots) via fnmadd: norms - 2*dots
                                        let d = _mm256_add_ps(pi_v, _mm256_fnmadd_ps(two, dots, norms));
                                        let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(
                                            d, thresh,
                                        ));
                                        if mask != 0 {
                                            let mut d_arr = [0.0f32; 8];
                                            _mm256_storeu_ps(d_arr.as_mut_ptr(), d);
                                            let mut m = mask as u32;
                                            while m != 0 {
                                                let lane = m.trailing_zeros() as usize;
                                                m &= m - 1;
                                                let j = base + lane;
                                                let dist = d_arr[lane];
                                                if dist < top[threshold_idx].1 {
                                                    top[threshold_idx] = (j as u32, dist);
                                                    let mut t = threshold_idx;
                                                    while t > 0 && top[t].1 < top[t - 1].1 {
                                                        top.swap(t, t - 1);
                                                        t -= 1;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    for j in (chunks * 8)..nl {
                                        let dot = *dot_row.get_unchecked(j);
                                        let d = pi + *l_norms.get_unchecked(j) - 2.0 * dot;
                                        if d < top[threshold_idx].1 {
                                            top[threshold_idx] = (j as u32, d);
                                            let mut t = threshold_idx;
                                            while t > 0 && top[t].1 < top[t - 1].1 {
                                                top.swap(t, t - 1);
                                                t -= 1;
                                            }
                                        }
                                    }
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                for j in 0..nl {
                                    let dot = unsafe { *dot_row.get_unchecked(j) };
                                    let d = pi + unsafe { *l_norms.get_unchecked(j) } - 2.0 * dot;
                                    if d < top[threshold_idx].1 {
                                        top[threshold_idx] = (j as u32, d);
                                        let mut t = threshold_idx;
                                        while t > 0 && top[t].1 < top[t - 1].1 {
                                            top.swap(t, t - 1);
                                            t -= 1;
                                        }
                                    }
                                }
                            }
                        }
                        Metric::InnerProduct => {
                            // d = -dot. Sign-flip via XOR with -0.0 in SIMD.
                            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                            {
                                use std::arch::x86_64::*;
                                let chunks = nl / 16;
                                // SAFETY: AVX-512 cfg-gated. `dot_row.as_ptr().add(base)`
                                // stays in bounds: base + 16 ≤ nl = dot_row.len(). Tail
                                // loop uses `j < nl` and `get_unchecked`.
                                unsafe {
                                    let sign = _mm512_set1_ps(-0.0f32);
                                    for chunk in 0..chunks {
                                        let base = chunk * 16;
                                        let thresh = _mm512_set1_ps(top[threshold_idx].1);
                                        let dots = _mm512_loadu_ps(dot_row.as_ptr().add(base));
                                        let d = _mm512_xor_ps(dots, sign);
                                        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(d, thresh);
                                        if mask != 0 {
                                            let mut d_arr = [0.0f32; 16];
                                            _mm512_storeu_ps(d_arr.as_mut_ptr(), d);
                                            let mut m = mask;
                                            while m != 0 {
                                                let lane = m.trailing_zeros() as usize;
                                                m &= m - 1;
                                                let j = (base + lane) as u32;
                                                topk_insert(
                                                    &mut top,
                                                    threshold_idx,
                                                    j,
                                                    d_arr[lane],
                                                );
                                            }
                                        }
                                    }
                                    for j in (chunks * 16)..nl {
                                        let d = -*dot_row.get_unchecked(j);
                                        topk_insert(&mut top, threshold_idx, j as u32, d);
                                    }
                                }
                            }
                            #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
                            {
                                use std::arch::x86_64::*;
                                let chunks = nl / 8;
                                // SAFETY: AVX2 cfg-gated; same bounds reasoning as the
                                // AVX-512 path with stride 8.
                                unsafe {
                                    let sign = _mm256_set1_ps(-0.0f32);
                                    for chunk in 0..chunks {
                                        let base = chunk * 8;
                                        let thresh = _mm256_set1_ps(top[threshold_idx].1);
                                        let dots = _mm256_loadu_ps(dot_row.as_ptr().add(base));
                                        let d = _mm256_xor_ps(dots, sign);
                                        let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(
                                            d, thresh,
                                        ));
                                        if mask != 0 {
                                            let mut d_arr = [0.0f32; 8];
                                            _mm256_storeu_ps(d_arr.as_mut_ptr(), d);
                                            let mut m = mask as u32;
                                            while m != 0 {
                                                let lane = m.trailing_zeros() as usize;
                                                m &= m - 1;
                                                let j = (base + lane) as u32;
                                                topk_insert(
                                                    &mut top,
                                                    threshold_idx,
                                                    j,
                                                    d_arr[lane],
                                                );
                                            }
                                        }
                                    }
                                    for j in (chunks * 8)..nl {
                                        let d = -*dot_row.get_unchecked(j);
                                        topk_insert(&mut top, threshold_idx, j as u32, d);
                                    }
                                }
                            }
                            #[cfg(not(target_arch = "x86_64"))]
                            for j in 0..nl {
                                // SAFETY: `dot_row.len() == nl` and `j < nl`.
                                let d = -(unsafe { *dot_row.get_unchecked(j) });
                                topk_insert(&mut top, threshold_idx, j as u32, d);
                            }
                        }
                    }

                    let global_i = mb_start + i;
                    let out = &mut assign_chunk[global_i * num_assign..(global_i + 1) * num_assign];
                    for k in 0..num_assign {
                        out[k] = top[k].0;
                    }
                }

                mb_start += mb;
            }
        });
    };

    if np <= stripe {
        // Single stripe — run inline, no rayon.
        process_stripe((0, &mut assignments));
    } else {
        // Multiple stripes — parallel.
        assignments
            .par_chunks_mut(stripe * num_assign)
            .enumerate()
            .for_each_installed(process_stripe);
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
