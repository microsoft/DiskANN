/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-row distance + top-K kernel, shared by v1 (`partition`) and v2
//! (`partition_v2`). One row = one point's dot products vs all leaders →
//! the row's `num_assign` nearest leaders, written into `out`.
//!
//! Extracted verbatim from the v1 inner closure: same SIMD codepaths
//! (AVX-512, AVX2, scalar) per metric.

use diskann_vector::distance::Metric;
use diskann_vector::topk::topk_insert;

/// Hard upper bound on `num_assign`. Mirrors `partition::MAX_FANOUT`.
pub(crate) const MAX_FANOUT_INNER: usize = 16;

/// Compute distance + top-K for one point row.
///
/// - `dot_row[j]` = dot(point, leader_j). Length = `nl`.
/// - `p_row` is the point's f32 data — only read for L2 / Cosine (||p||²).
/// - `l_norms[j]` = ||leader_j||² (L2) or ||leader_j|| (Cosine). Empty for
///   CosineNormalized / InnerProduct.
/// - On return `out[..num_assign]` holds the `num_assign` nearest leader
///   indices in ascending-distance order.
#[inline]
pub(crate) fn process_row(
    dot_row: &[f32],
    p_norm_sq: f32,
    l_norms: &[f32],
    metric: Metric,
    num_assign: usize,
    out: &mut [u32],
) {
    // p_norm_sq = ||p||² (precomputed by caller, batched SIMD reduce). Used as
    // `pi` for L2 distance and as `sqrt(pi)` for Cosine. CosineNormalized and
    // InnerProduct ignore it.
    let _ = p_norm_sq;
    let nl = dot_row.len();
    debug_assert!(num_assign <= MAX_FANOUT_INNER);
    debug_assert!(out.len() >= num_assign);

    let mut top: [(u32, f32); MAX_FANOUT_INNER] = [(u32::MAX, f32::MAX); MAX_FANOUT_INNER];
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
                unsafe {
                    let one = _mm256_set1_ps(1.0);
                    for chunk in 0..chunks {
                        let base = chunk * 8;
                        let thresh = _mm256_set1_ps(top[threshold_idx].1);
                        let dots = _mm256_loadu_ps(dot_row.as_ptr().add(base));
                        let d = _mm256_sub_ps(one, dots);
                        let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(d, thresh));
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
            let pi_sqrt: f32 = p_norm_sq.sqrt();
            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                use std::arch::x86_64::*;
                let chunks = nl / 16;
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
                        let denom_mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(denom, zero);
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
                                topk_insert(&mut top, threshold_idx, j, d_arr[lane]);
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
                        let zero_mask = _mm256_cmp_ps::<_CMP_GT_OQ>(denom, zero);
                        let cos = _mm256_and_ps(div, zero_mask);
                        let d = _mm256_sub_ps(one, cos);
                        let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(d, thresh));
                        if mask != 0 {
                            let mut d_arr = [0.0f32; 8];
                            _mm256_storeu_ps(d_arr.as_mut_ptr(), d);
                            let mut m = mask as u32;
                            while m != 0 {
                                let lane = m.trailing_zeros() as usize;
                                m &= m - 1;
                                let j = (base + lane) as u32;
                                topk_insert(&mut top, threshold_idx, j, d_arr[lane]);
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
                let dot = unsafe { *dot_row.get_unchecked(j) };
                let denom = pi_sqrt * ln;
                let cos = if denom > 0.0 { dot / denom } else { 0.0 };
                topk_insert(&mut top, threshold_idx, j as u32, 1.0 - cos);
            }
        }
        Metric::L2 => {
            let pi: f32 = p_norm_sq;
            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                use std::arch::x86_64::*;
                let chunks = nl / 16;
                unsafe {
                    let pi_v = _mm512_set1_ps(pi);
                    let two = _mm512_set1_ps(2.0);
                    for chunk in 0..chunks {
                        let base = chunk * 16;
                        let thresh = _mm512_set1_ps(top[threshold_idx].1);
                        let norms = _mm512_loadu_ps(l_norms.as_ptr().add(base));
                        let dots = _mm512_loadu_ps(dot_row.as_ptr().add(base));
                        let d = _mm512_add_ps(pi_v, _mm512_fnmadd_ps(two, dots, norms));
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
                unsafe {
                    let pi_v = _mm256_set1_ps(pi);
                    let two = _mm256_set1_ps(2.0);
                    for chunk in 0..chunks {
                        let base = chunk * 8;
                        let thresh = _mm256_set1_ps(top[threshold_idx].1);
                        let norms = _mm256_loadu_ps(l_norms.as_ptr().add(base));
                        let dots = _mm256_loadu_ps(dot_row.as_ptr().add(base));
                        let d = _mm256_add_ps(pi_v, _mm256_fnmadd_ps(two, dots, norms));
                        let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(d, thresh));
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
            #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
            {
                use std::arch::x86_64::*;
                let chunks = nl / 16;
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
                                topk_insert(&mut top, threshold_idx, j, d_arr[lane]);
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
                unsafe {
                    let sign = _mm256_set1_ps(-0.0f32);
                    for chunk in 0..chunks {
                        let base = chunk * 8;
                        let thresh = _mm256_set1_ps(top[threshold_idx].1);
                        let dots = _mm256_loadu_ps(dot_row.as_ptr().add(base));
                        let d = _mm256_xor_ps(dots, sign);
                        let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(d, thresh));
                        if mask != 0 {
                            let mut d_arr = [0.0f32; 8];
                            _mm256_storeu_ps(d_arr.as_mut_ptr(), d);
                            let mut m = mask as u32;
                            while m != 0 {
                                let lane = m.trailing_zeros() as usize;
                                m &= m - 1;
                                let j = (base + lane) as u32;
                                topk_insert(&mut top, threshold_idx, j, d_arr[lane]);
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
                let d = -(unsafe { *dot_row.get_unchecked(j) });
                topk_insert(&mut top, threshold_idx, j as u32, d);
            }
        }
    }

    for k in 0..num_assign {
        out[k] = top[k].0;
    }
}

/// Detect the CPU's private L2 cache size in bytes. Cached on first call.
/// Override via `PIPNN_L2_SIZE=<bytes>`. Falls back to 512 KB (Zen 3 default).
pub(crate) fn l2_size_bytes() -> usize {
    use std::sync::OnceLock;
    static L2_SIZE: OnceLock<usize> = OnceLock::new();
    *L2_SIZE.get_or_init(|| {
        if let Ok(s) = std::env::var("PIPNN_L2_SIZE") {
            if let Ok(v) = s.parse::<usize>() {
                tracing::info!(l2_bytes = v, source = "env", "PiPNN: L2 cache size");
                return v;
            }
        }
        #[cfg(target_os = "linux")]
        if let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size") {
            let trimmed = s.trim();
            let parsed = if let Some(n) = trimmed.strip_suffix('K') {
                n.parse::<usize>().ok().map(|v| v * 1024)
            } else if let Some(n) = trimmed.strip_suffix('M') {
                n.parse::<usize>().ok().map(|v| v * 1024 * 1024)
            } else {
                trimmed.parse::<usize>().ok()
            };
            if let Some(v) = parsed {
                tracing::info!(l2_bytes = v, source = "/sys", "PiPNN: L2 cache size");
                return v;
            }
        }
        let fallback = 512 * 1024;
        tracing::info!(l2_bytes = fallback, source = "default", "PiPNN: L2 cache size");
        fallback
    })
}

/// Mini-batch row count sized so that the `dots` tile (`MB × nl × 4`) fits in
/// private L2. Power-of-2 rounded down, clamped to `[32, 1024]`.
///
/// Only `dots` needs L2 residency: it's written by the GEMM and immediately
/// read by the per-row top-K scan. The leader matrix and point gather are
/// streamed (read once per chunk), so they don't compete for the same residency
/// budget. Matches the `mb_sweep` bench optima: nl=1000→128, nl=500→256, etc.
pub(crate) fn compute_mb(nl: usize, _k: usize, l2_bytes: usize) -> usize {
    let mb_max = (l2_bytes / (nl.max(1) * 4)).max(32);
    let pow2 = if mb_max.is_power_of_two() {
        mb_max
    } else {
        mb_max.next_power_of_two() >> 1
    };
    pow2.clamp(32, 1024)
}

/// Returns `true` if the whole `m × nl` dots tile fits comfortably in L2 —
/// i.e. we should do one sequential GEMM and skip the mini-batch loop.
pub(crate) fn should_skip_mb(m: usize, nl: usize, l2_bytes: usize) -> bool {
    let dots_bytes = (m as u64).saturating_mul(nl as u64).saturating_mul(4);
    dots_bytes < (l2_bytes as u64 / 2)
}
