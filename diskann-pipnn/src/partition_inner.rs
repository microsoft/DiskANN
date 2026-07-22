/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Batched distance + top-K kernels used by `partition::assign_to_leaders`.
//! Runtime SIMD selection happens once for the whole matrix chunk; the
//! architecture-specific functions remain private to this module.

use diskann_vector::distance::Metric;
use diskann_vector::topk::topk_insert;

use crate::cpu_dispatch::{self, VectorWidth};

/// Hard upper bound on `num_assign`. Mirrors `partition::MAX_FANOUT`.
const MAX_FANOUT_INNER: usize = crate::partition::MAX_FANOUT;

// Type alias for the working top-K buffer used by every kernel variant.
// Length = `MAX_FANOUT_INNER`, but only the first `num_assign` slots are
// ever live.
type TopBuf = [(u32, f32); MAX_FANOUT_INNER];

/// Compute distance + top-K for every row in a dot-product matrix.
///
/// - Each row in `dots` contains dot(point, leader_j) for every leader.
/// - `row_norms_sq` contains point norms for unnormalized cosine only.
/// - `l_norms[j]` = ||leader_j||² (L2) or ||leader_j|| (Cosine). Empty for
///   CosineNormalized / InnerProduct.
/// - On return `out[..num_assign]` holds the `num_assign` nearest leader
///   indices in ascending-distance order.
pub(crate) fn process_rows(
    dots: &[f32],
    row_norms_sq: &[f32],
    l_norms: &[f32],
    metric: Metric,
    num_assign: usize,
    out: &mut [u32],
) {
    assert!((1..=MAX_FANOUT_INNER).contains(&num_assign));
    assert_eq!(out.len() % num_assign, 0);
    let rows = out.len() / num_assign;
    if rows == 0 {
        assert!(dots.is_empty());
        return;
    }
    assert_eq!(dots.len() % rows, 0);
    let leaders = dots.len() / rows;
    assert!(num_assign <= leaders);
    match metric {
        Metric::Cosine | Metric::L2 => assert_eq!(l_norms.len(), leaders),
        Metric::CosineNormalized | Metric::InnerProduct => assert!(l_norms.is_empty()),
    }
    if matches!(metric, Metric::Cosine) {
        assert_eq!(row_norms_sq.len(), rows);
    } else {
        assert!(row_norms_sq.is_empty());
    }

    let width = if matches!(metric, Metric::L2) {
        cpu_dispatch::fma_width()
    } else {
        cpu_dispatch::f32_width()
    };
    let kernel = select_row_kernel(metric, width);
    // SAFETY: `select_row_kernel` only returns a target-feature kernel after
    // checking the required CPU features. All slice shapes were checked above.
    unsafe {
        for (row_idx, (dot_row, out_row)) in dots
            .chunks_exact(leaders)
            .zip(out.chunks_exact_mut(num_assign))
            .enumerate()
        {
            let mut top: TopBuf = [(u32::MAX, f32::MAX); MAX_FANOUT_INNER];
            kernel(
                dot_row,
                row_norms_sq.get(row_idx).copied().unwrap_or(0.0),
                l_norms,
                &mut top,
                num_assign - 1,
            );
            for (dst, &(leader, _)) in out_row.iter_mut().zip(&top) {
                *dst = leader;
            }
        }
    }
}

type RowKernel = unsafe fn(&[f32], f32, &[f32], &mut TopBuf, usize);

fn select_row_kernel(metric: Metric, width: VectorWidth) -> RowKernel {
    match (metric, width) {
        #[cfg(target_arch = "x86_64")]
        (Metric::CosineNormalized, VectorWidth::Wide) => process_row_cosnorm_wide,
        #[cfg(target_arch = "x86_64")]
        (Metric::CosineNormalized, VectorWidth::Narrow) => process_row_cosnorm_narrow,
        (Metric::CosineNormalized, _) => process_row_cosnorm_scalar,
        #[cfg(target_arch = "x86_64")]
        (Metric::Cosine, VectorWidth::Wide) => process_row_cosine_wide,
        #[cfg(target_arch = "x86_64")]
        (Metric::Cosine, VectorWidth::Narrow) => process_row_cosine_narrow,
        (Metric::Cosine, _) => process_row_cosine_scalar,
        #[cfg(target_arch = "x86_64")]
        (Metric::L2, VectorWidth::Wide) => process_row_l2_wide,
        #[cfg(target_arch = "x86_64")]
        (Metric::L2, VectorWidth::Narrow) => process_row_l2_narrow,
        (Metric::L2, _) => process_row_l2_scalar,
        #[cfg(target_arch = "x86_64")]
        (Metric::InnerProduct, VectorWidth::Wide) => process_row_ip_wide,
        #[cfg(target_arch = "x86_64")]
        (Metric::InnerProduct, VectorWidth::Narrow) => process_row_ip_narrow,
        (Metric::InnerProduct, _) => process_row_ip_scalar,
    }
}

// ---------------------------------------------------------------------------
// CosineNormalized — distance = 1 - dot(p, l)  (vectors pre-normalized)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn process_row_cosnorm_wide(
    dot_row: &[f32],
    _row_norm_sq: f32,
    _leader_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
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
                topk_insert(top, threshold_idx, (base + lane) as u32, d_arr[lane]);
            }
        }
    }
    for j in (chunks * 16)..nl {
        let d = 1.0 - *dot_row.get_unchecked(j);
        topk_insert(top, threshold_idx, j as u32, d);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn process_row_cosnorm_narrow(
    dot_row: &[f32],
    _row_norm_sq: f32,
    _leader_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
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
                topk_insert(top, threshold_idx, (base + lane) as u32, d_arr[lane]);
            }
        }
    }
    for j in (chunks * 8)..nl {
        let d = 1.0 - *dot_row.get_unchecked(j);
        topk_insert(top, threshold_idx, j as u32, d);
    }
}

unsafe fn process_row_cosnorm_scalar(
    dot_row: &[f32],
    _row_norm_sq: f32,
    _leader_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    for (j, &dot) in dot_row.iter().enumerate() {
        let d = 1.0 - dot;
        topk_insert(top, threshold_idx, j as u32, d);
    }
}

// ---------------------------------------------------------------------------
// Cosine — distance = 1 - dot/(||p|| · ||l||)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn process_row_cosine_wide(
    dot_row: &[f32],
    row_norm_sq: f32,
    l_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 16;
    let one = _mm512_set1_ps(1.0);
    let pi_sqrt = row_norm_sq.sqrt();
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
unsafe fn process_row_cosine_narrow(
    dot_row: &[f32],
    row_norm_sq: f32,
    l_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 8;
    let one = _mm256_set1_ps(1.0);
    let pi_sqrt = row_norm_sq.sqrt();
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

unsafe fn process_row_cosine_scalar(
    dot_row: &[f32],
    row_norm_sq: f32,
    l_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    let pi_sqrt = row_norm_sq.sqrt();
    for (j, (&dot, &ln)) in dot_row.iter().zip(l_norms).enumerate() {
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
unsafe fn process_row_l2_wide(
    dot_row: &[f32],
    _row_norm_sq: f32,
    l_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 16;
    let pi = 0.0;
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
                topk_insert(top, threshold_idx, (base + lane) as u32, d_arr[lane]);
            }
        }
    }
    for j in (chunks * 16)..nl {
        let dot = *dot_row.get_unchecked(j);
        let d = pi + *l_norms.get_unchecked(j) - 2.0 * dot;
        topk_insert(top, threshold_idx, j as u32, d);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn process_row_l2_narrow(
    dot_row: &[f32],
    _row_norm_sq: f32,
    l_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    use std::arch::x86_64::*;
    let nl = dot_row.len();
    let chunks = nl / 8;
    let pi = 0.0;
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
                topk_insert(top, threshold_idx, (base + lane) as u32, d_arr[lane]);
            }
        }
    }
    for j in (chunks * 8)..nl {
        let dot = *dot_row.get_unchecked(j);
        let d = pi + *l_norms.get_unchecked(j) - 2.0 * dot;
        topk_insert(top, threshold_idx, j as u32, d);
    }
}

unsafe fn process_row_l2_scalar(
    dot_row: &[f32],
    _row_norm_sq: f32,
    l_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    let pi = 0.0;
    for (j, (&dot, &norm)) in dot_row.iter().zip(l_norms).enumerate() {
        let d = pi + norm - 2.0 * dot;
        topk_insert(top, threshold_idx, j as u32, d);
    }
}

// ---------------------------------------------------------------------------
// InnerProduct — distance = -dot(p, l)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn process_row_ip_wide(
    dot_row: &[f32],
    _row_norm_sq: f32,
    _leader_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
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
unsafe fn process_row_ip_narrow(
    dot_row: &[f32],
    _row_norm_sq: f32,
    _leader_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
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

unsafe fn process_row_ip_scalar(
    dot_row: &[f32],
    _row_norm_sq: f32,
    _leader_norms: &[f32],
    top: &mut TopBuf,
    threshold_idx: usize,
) {
    for (j, &dot) in dot_row.iter().enumerate() {
        let d = -dot;
        topk_insert(top, threshold_idx, j as u32, d);
    }
}
