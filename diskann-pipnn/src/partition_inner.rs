/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-row distance + top-K kernel used by `partition::assign_to_leaders`.
//! One row = one point's dot products vs all leaders → the row's
//! `num_assign` nearest leader indices, written into `out`.
//!
//! Runtime SIMD dispatch via `cpu_dispatch::tier()`: each metric arm matches
//! on tier and calls a `#[target_feature]` `unsafe fn`. The tier match runs
//! once per `process_row` call (per point).

use diskann_vector::distance::Metric;
use diskann_vector::topk::topk_insert;

use crate::cpu_dispatch::{self, SimdTier};

/// Hard upper bound on `num_assign`. Mirrors `partition::MAX_FANOUT`.
pub(crate) const MAX_FANOUT_INNER: usize = 16;

// Type alias for the working top-K buffer used by every kernel variant.
// Length = `MAX_FANOUT_INNER`, but only the first `num_assign` slots are
// ever live.
type TopBuf = [(u32, f32); MAX_FANOUT_INNER];

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
    debug_assert!(num_assign <= MAX_FANOUT_INNER);
    debug_assert!(out.len() >= num_assign);

    let mut top: TopBuf = [(u32::MAX, f32::MAX); MAX_FANOUT_INNER];
    let threshold_idx = num_assign - 1;

    let tier = cpu_dispatch::tier();

    match metric {
        Metric::CosineNormalized => match tier {
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx512 => unsafe {
                process_row_cosnorm_avx512(dot_row, &mut top, threshold_idx);
            },
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx2 => unsafe {
                process_row_cosnorm_avx2(dot_row, &mut top, threshold_idx);
            },
            _ => process_row_cosnorm_scalar(dot_row, &mut top, threshold_idx),
        },
        Metric::Cosine => {
            let pi_sqrt: f32 = p_norm_sq.sqrt();
            match tier {
                #[cfg(target_arch = "x86_64")]
                SimdTier::Avx512 => unsafe {
                    process_row_cosine_avx512(dot_row, l_norms, pi_sqrt, &mut top, threshold_idx);
                },
                #[cfg(target_arch = "x86_64")]
                SimdTier::Avx2 => unsafe {
                    process_row_cosine_avx2(dot_row, l_norms, pi_sqrt, &mut top, threshold_idx);
                },
                _ => process_row_cosine_scalar(dot_row, l_norms, pi_sqrt, &mut top, threshold_idx),
            }
        }
        Metric::L2 => {
            let pi: f32 = p_norm_sq;
            match tier {
                #[cfg(target_arch = "x86_64")]
                SimdTier::Avx512 => unsafe {
                    process_row_l2_avx512(dot_row, l_norms, pi, &mut top, threshold_idx);
                },
                #[cfg(target_arch = "x86_64")]
                SimdTier::Avx2 => unsafe {
                    process_row_l2_avx2(dot_row, l_norms, pi, &mut top, threshold_idx);
                },
                _ => process_row_l2_scalar(dot_row, l_norms, pi, &mut top, threshold_idx),
            }
        }
        Metric::InnerProduct => match tier {
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx512 => unsafe {
                process_row_ip_avx512(dot_row, &mut top, threshold_idx);
            },
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx2 => unsafe {
                process_row_ip_avx2(dot_row, &mut top, threshold_idx);
            },
            _ => process_row_ip_scalar(dot_row, &mut top, threshold_idx),
        },
    }

    for k in 0..num_assign {
        out[k] = top[k].0;
    }
}

// ---------------------------------------------------------------------------
// CosineNormalized — distance = 1 - dot(p, l)  (vectors pre-normalized)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn process_row_cosnorm_avx512(dot_row: &[f32], top: &mut TopBuf, threshold_idx: usize) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 16;
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn process_row_cosnorm_avx2(dot_row: &[f32], top: &mut TopBuf, threshold_idx: usize) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 8;
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

fn process_row_cosnorm_scalar(dot_row: &[f32], top: &mut TopBuf, threshold_idx: usize) {
    let nl = dot_row.len();
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

// ---------------------------------------------------------------------------
// Cosine — distance = 1 - dot/(||p|| · ||l||)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn process_row_cosine_avx512(
    dot_row: &[f32],
    l_norms: &[f32],
    pi_sqrt: f32,
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 16;
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
                topk_insert(top, threshold_idx, j, d_arr[lane]);
            }
        }
    }
    for j in (chunks * 16)..nl {
        let dot = *dot_row.get_unchecked(j);
        let ln = *l_norms.get_unchecked(j);
        let denom = pi_sqrt * ln;
        let cos = if denom > 0.0 { dot / denom } else { 0.0 };
        topk_insert(top, threshold_idx, j as u32, 1.0 - cos);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn process_row_cosine_avx2(
    dot_row: &[f32],
    l_norms: &[f32],
    pi_sqrt: f32,
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 8;
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
                topk_insert(top, threshold_idx, j, d_arr[lane]);
            }
        }
    }
    for j in (chunks * 8)..nl {
        let dot = *dot_row.get_unchecked(j);
        let ln = *l_norms.get_unchecked(j);
        let denom = pi_sqrt * ln;
        let cos = if denom > 0.0 { dot / denom } else { 0.0 };
        topk_insert(top, threshold_idx, j as u32, 1.0 - cos);
    }
}

fn process_row_cosine_scalar(
    dot_row: &[f32],
    l_norms: &[f32],
    pi_sqrt: f32,
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    let nl = dot_row.len();
    for (j, &ln) in l_norms.iter().enumerate().take(nl) {
        let dot = unsafe { *dot_row.get_unchecked(j) };
        let denom = pi_sqrt * ln;
        let cos = if denom > 0.0 { dot / denom } else { 0.0 };
        topk_insert(top, threshold_idx, j as u32, 1.0 - cos);
    }
}

// ---------------------------------------------------------------------------
// L2 — distance = ||p||² + ||l||² - 2·dot(p, l)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn process_row_l2_avx512(
    dot_row: &[f32],
    l_norms: &[f32],
    pi: f32,
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 16;
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn process_row_l2_avx2(
    dot_row: &[f32],
    l_norms: &[f32],
    pi: f32,
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 8;
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

fn process_row_l2_scalar(
    dot_row: &[f32],
    l_norms: &[f32],
    pi: f32,
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    let nl = dot_row.len();
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

// ---------------------------------------------------------------------------
// InnerProduct — distance = -dot(p, l)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn process_row_ip_avx512(dot_row: &[f32], top: &mut TopBuf, threshold_idx: usize) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 16;
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
                topk_insert(top, threshold_idx, j, d_arr[lane]);
            }
        }
    }
    for j in (chunks * 16)..nl {
        let d = -*dot_row.get_unchecked(j);
        topk_insert(top, threshold_idx, j as u32, d);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn process_row_ip_avx2(dot_row: &[f32], top: &mut TopBuf, threshold_idx: usize) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 8;
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
                topk_insert(top, threshold_idx, j, d_arr[lane]);
            }
        }
    }
    for j in (chunks * 8)..nl {
        let d = -*dot_row.get_unchecked(j);
        topk_insert(top, threshold_idx, j as u32, d);
    }
}

fn process_row_ip_scalar(dot_row: &[f32], top: &mut TopBuf, threshold_idx: usize) {
    let nl = dot_row.len();
    for j in 0..nl {
        let d = -(unsafe { *dot_row.get_unchecked(j) });
        topk_insert(top, threshold_idx, j as u32, d);
    }
}

/// Detect the CPU's private L2 cache size in bytes. Cached on first call.
/// Falls back to 512 KB (Zen 3 default). Callers may override via
/// `PartitionConfig.l2_size_override` — this auto-detected value is used when
/// no override is set.
pub(crate) fn l2_size_bytes() -> usize {
    use std::sync::OnceLock;
    static L2_SIZE: OnceLock<usize> = OnceLock::new();
    *L2_SIZE.get_or_init(|| {
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
