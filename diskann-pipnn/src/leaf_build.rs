/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf building: GEMM-based all-pairs distance computation and bi-directed k-NN extraction.
//!
//! For each leaf partition (bounded by the configured `c_max`):
//! 1. Compute all-pairs distance matrix via GEMM
//!    For L2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*(a.b)
//!    The dot product matrix A * A^T is computed as a GEMM operation.
//! 2. Extract k nearest neighbors per point with a fused triangular scan
//! 3. Create bi-directed edges (both forward and reverse k-NN)

use std::cell::RefCell;

use crate::cpu_dispatch::{fma_width, VectorWidth};
use diskann::utils::VectorRepr;

/// Thread-local reusable buffers for leaf building.
/// Avoids repeated allocation/deallocation of large matrices.
pub(crate) struct LeafBuffers {
    local_data: Vec<f32>,
    norms_sq: Vec<f32>,
    dot_matrix: Vec<f32>,
    seen: Vec<bool>,
    /// Reusable buffer for knn results: (local_dst_idx, distance) per row×k.
    knn_result: Vec<(u32, f32)>,
    /// Reusable Cosine sqrt-of-norms scratch (only filled for `Metric::Cosine`).
    cosine_denoms: Vec<f32>,
    /// CSR-style per-source edge groups: data[starts[src]..starts[src+1]] is
    /// the list of (local_dst, dist) pairs for local source `src`. Avoids per-leaf
    /// Vec<Vec<...>> allocation while preserving src-grouped insertion order for
    /// HP. Sized by ensure_capacity to fit `n + 1` starts and `2 * n * k` entries.
    group_starts: Vec<u32>,
    group_data: Vec<(u32, f32)>,
    /// Per-source write cursor used during the CSR pass-2 fill. Initialised
    /// from `group_starts[..n]` at the start of each leaf's CSR build.
    cursor: Vec<u32>,
    /// Per-row top-k threshold (current k-th smallest distance) for the fused
    /// dual-end scan. Sized `n`. Initialised to `f32::MAX`.
    worst: Vec<f32>,
}

impl Default for LeafBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl LeafBuffers {
    fn new() -> Self {
        Self {
            local_data: Vec::new(),
            norms_sq: Vec::new(),
            dot_matrix: Vec::new(),
            seen: Vec::new(),
            knn_result: Vec::new(),
            cosine_denoms: Vec::new(),
            group_starts: Vec::new(),
            group_data: Vec::new(),
            cursor: Vec::new(),
            worst: Vec::new(),
        }
    }

    /// Ensure all buffers are large enough for a leaf of size n × ndims with leaf_k=k.
    fn ensure_capacity(&mut self, n: usize, ndims: usize, k: usize) {
        let nd = n * ndims;
        let nn = n * n;
        if self.local_data.len() < nd {
            self.local_data.resize(nd, 0.0);
        }
        if self.norms_sq.len() < n {
            self.norms_sq.resize(n, 0.0);
        }
        if self.dot_matrix.len() < nn {
            self.dot_matrix.resize(nn, 0.0);
        }
        if self.seen.len() < nn {
            self.seen.resize(nn, false);
        }
        // Pre-size knn_result so per-leaf writes hit no realloc
        // (Vec realloc contends on the glibc malloc arena at high thread count).
        let actual_k = k.min(n.saturating_sub(1));
        let max_knn = n * actual_k.max(1);
        let max_edges = 2 * max_knn;
        if self.knn_result.capacity() < max_knn {
            self.knn_result
                .reserve(max_knn - self.knn_result.capacity());
        }
        if self.group_starts.capacity() < n + 1 {
            self.group_starts
                .reserve(n + 1 - self.group_starts.capacity());
        }
        if self.group_data.capacity() < max_edges {
            self.group_data
                .reserve(max_edges - self.group_data.capacity());
        }
        // worst[] threshold array: n entries (+15 for the 16-wide chunk tail
        // load so we can read past `i` without OOB). Initialised by caller.
        if self.worst.len() < n + 16 {
            self.worst.resize(n + 16, f32::MAX);
        }
    }

    pub(crate) fn edges(&self, edge_count: usize) -> (&[u32], &[(u32, f32)]) {
        debug_assert_eq!(self.group_starts.last().copied(), Some(edge_count as u32));
        (&self.group_starts, &self.group_data[..edge_count])
    }

    fn clear(&mut self) {
        *self = Self::new();
    }
}

thread_local! {
    /// Thread-local reusable buffers for leaf building. Public so builder can
    /// batch multiple leaves per TLS access (amortizes the `with()` overhead).
    pub(crate) static LEAF_BUFFERS: RefCell<LeafBuffers> = RefCell::new(LeafBuffers::new());
}

/// Release thread-local leaf build buffers on the calling thread.
///
/// After leaf building is complete, these buffers pin pages in glibc's
/// per-thread arenas, preventing `malloc_trim` from returning freed
/// reservoir memory to the OS. Calling this from each rayon thread
/// helps glibc reclaim arena pages (best-effort: depends on rayon work-stealing touching all workers).
pub(crate) fn release_thread_buffers() {
    LEAF_BUFFERS.with(|cell| cell.borrow_mut().clear());
}

// ───── Metric kernels for the fused dual-end scan ─────
//
// One trait, four implementations. The trait's three methods each return the
// distance (smaller-is-better) for the metric, given the raw dot product and
// optional per-row scalar (`ni`, `nj`). The norms-meaning depends on metric:
//   L2:                norms[i] = ‖x_i‖²
//   CosineNormalized:  ignored (data is unit-norm; dist = max(1 − dot, 0))
//   Cosine:            norms[i] = ‖x_i‖ (precomputed sqrt of L2 norm-sq)
//   InnerProduct:      ignored (dist = −dot, smaller-is-better)
//
// The fused scan calls the width-specific distance implementation selected once per leaf.
// inside the hot loop; monomorphization inlines the formula at compile time.

trait MetricKernel {
    #[cfg(target_arch = "x86_64")]
    unsafe fn dist_wide(
        dot: std::arch::x86_64::__m512,
        ni: std::arch::x86_64::__m512,
        nj: std::arch::x86_64::__m512,
    ) -> std::arch::x86_64::__m512;
    #[cfg(target_arch = "x86_64")]
    unsafe fn dist_narrow(
        dot: std::arch::x86_64::__m256,
        ni: std::arch::x86_64::__m256,
        nj: std::arch::x86_64::__m256,
    ) -> std::arch::x86_64::__m256;
    fn dist_scalar(dot: f32, ni: f32, nj: f32) -> f32;
}

struct KL2;
impl MetricKernel for KL2 {
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dist_wide(
        dot: std::arch::x86_64::__m512,
        ni: std::arch::x86_64::__m512,
        nj: std::arch::x86_64::__m512,
    ) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        let two = _mm512_set1_ps(2.0);
        let sum_norms = _mm512_add_ps(ni, nj);
        let two_dot = _mm512_mul_ps(two, dot);
        _mm512_max_ps(_mm512_sub_ps(sum_norms, two_dot), _mm512_setzero_ps())
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dist_narrow(
        dot: std::arch::x86_64::__m256,
        ni: std::arch::x86_64::__m256,
        nj: std::arch::x86_64::__m256,
    ) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;
        let two = _mm256_set1_ps(2.0);
        let sum_norms = _mm256_add_ps(ni, nj);
        let two_dot = _mm256_mul_ps(two, dot);
        _mm256_max_ps(_mm256_sub_ps(sum_norms, two_dot), _mm256_setzero_ps())
    }
    #[inline(always)]
    fn dist_scalar(dot: f32, ni: f32, nj: f32) -> f32 {
        (ni + nj - 2.0 * dot).max(0.0)
    }
}

struct KCosNorm;
impl MetricKernel for KCosNorm {
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dist_wide(
        dot: std::arch::x86_64::__m512,
        _ni: std::arch::x86_64::__m512,
        _nj: std::arch::x86_64::__m512,
    ) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        _mm512_max_ps(_mm512_sub_ps(_mm512_set1_ps(1.0), dot), _mm512_setzero_ps())
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dist_narrow(
        dot: std::arch::x86_64::__m256,
        _ni: std::arch::x86_64::__m256,
        _nj: std::arch::x86_64::__m256,
    ) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;
        _mm256_max_ps(_mm256_sub_ps(_mm256_set1_ps(1.0), dot), _mm256_setzero_ps())
    }
    #[inline(always)]
    fn dist_scalar(dot: f32, _ni: f32, _nj: f32) -> f32 {
        (1.0 - dot).max(0.0)
    }
}

struct KCosine;
impl MetricKernel for KCosine {
    // norms = sqrt(‖x‖²). For ni·nj == 0 we fall to cos = 0 → dist = 1.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dist_wide(
        dot: std::arch::x86_64::__m512,
        ni: std::arch::x86_64::__m512,
        nj: std::arch::x86_64::__m512,
    ) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        let denom = _mm512_mul_ps(ni, nj);
        let zero_mask = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(denom, _mm512_setzero_ps());
        let safe_denom = _mm512_mask_blend_ps(zero_mask, denom, _mm512_set1_ps(1.0));
        let cos = _mm512_div_ps(dot, safe_denom);
        // Force cos to 0 in zero-denom lanes → dist = 1.
        let cos = _mm512_mask_blend_ps(zero_mask, cos, _mm512_setzero_ps());
        _mm512_max_ps(_mm512_sub_ps(_mm512_set1_ps(1.0), cos), _mm512_setzero_ps())
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dist_narrow(
        dot: std::arch::x86_64::__m256,
        ni: std::arch::x86_64::__m256,
        nj: std::arch::x86_64::__m256,
    ) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;
        let denom = _mm256_mul_ps(ni, nj);
        let zero_mask = _mm256_cmp_ps::<_CMP_EQ_OQ>(denom, _mm256_setzero_ps());
        let safe_denom = _mm256_blendv_ps(denom, _mm256_set1_ps(1.0), zero_mask);
        let cos = _mm256_div_ps(dot, safe_denom);
        let cos = _mm256_blendv_ps(cos, _mm256_setzero_ps(), zero_mask);
        _mm256_max_ps(_mm256_sub_ps(_mm256_set1_ps(1.0), cos), _mm256_setzero_ps())
    }
    #[inline(always)]
    fn dist_scalar(dot: f32, ni: f32, nj: f32) -> f32 {
        let denom = ni * nj;
        let cos = if denom > 0.0 { dot / denom } else { 0.0 };
        (1.0 - cos).max(0.0)
    }
}

struct KIp;
impl MetricKernel for KIp {
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dist_wide(
        dot: std::arch::x86_64::__m512,
        _ni: std::arch::x86_64::__m512,
        _nj: std::arch::x86_64::__m512,
    ) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        _mm512_sub_ps(_mm512_setzero_ps(), dot)
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dist_narrow(
        dot: std::arch::x86_64::__m256,
        _ni: std::arch::x86_64::__m256,
        _nj: std::arch::x86_64::__m256,
    ) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;
        _mm256_sub_ps(_mm256_setzero_ps(), dot)
    }
    #[inline(always)]
    fn dist_scalar(dot: f32, _ni: f32, _nj: f32) -> f32 {
        -dot
    }
}

/// Generic small-k insert into `knn_result[base..base+k]` (sorted ascending by
/// `.1`). Caller has already verified `d < knn_result[base+k-1].1`. Bubble-up
/// with branchless cmov, `k − 1` iterations. Returns the new worst (largest).
#[inline(always)]
unsafe fn insert_topk_linear(
    knn_result: &mut [(u32, f32)],
    base: usize,
    k: usize,
    idx: u32,
    d: f32,
) -> f32 {
    debug_assert!(k >= 1);
    let new_e = (idx, d);
    match k {
        1 => {
            *knn_result.get_unchecked_mut(base) = new_e;
            d
        }
        2 => {
            let a = *knn_result.get_unchecked(base);
            if d < a.1 {
                *knn_result.get_unchecked_mut(base) = new_e;
                *knn_result.get_unchecked_mut(base + 1) = a;
                a.1
            } else {
                *knn_result.get_unchecked_mut(base + 1) = new_e;
                d
            }
        }
        3 => {
            let a = *knn_result.get_unchecked(base);
            let b = *knn_result.get_unchecked(base + 1);
            if d < a.1 {
                *knn_result.get_unchecked_mut(base) = new_e;
                *knn_result.get_unchecked_mut(base + 1) = a;
                *knn_result.get_unchecked_mut(base + 2) = b;
                b.1
            } else if d < b.1 {
                *knn_result.get_unchecked_mut(base + 1) = new_e;
                *knn_result.get_unchecked_mut(base + 2) = b;
                b.1
            } else {
                *knn_result.get_unchecked_mut(base + 2) = new_e;
                d
            }
        }
        // k >= 4 (not a production leaf_k; correctness fallback): branchless
        // insertion-sort bubble-up, identical result to the specialized arms.
        _ => {
            *knn_result.get_unchecked_mut(base + k - 1) = new_e;
            let mut pos = base + k - 1;
            while pos > base {
                let cur = *knn_result.get_unchecked(pos);
                let prev = *knn_result.get_unchecked(pos - 1);
                let swap = cur.1 < prev.1;
                *knn_result.get_unchecked_mut(pos) = if swap { prev } else { cur };
                *knn_result.get_unchecked_mut(pos - 1) = if swap { cur } else { prev };
                pos -= 1;
            }
            knn_result.get_unchecked(base + k - 1).1
        }
    }
}

/// Init `worst[..n+16]` and `knn_result[..n*k]` to sentinels. Shared init for
/// all three tier-specific fused scans below.
#[inline(always)]
unsafe fn fused_init(knn_result: &mut [(u32, f32)], worst: &mut [f32], n: usize, k: usize) {
    for i in 0..n {
        *worst.get_unchecked_mut(i) = f32::MAX;
        let off = i * k;
        for t in 0..k {
            *knn_result.get_unchecked_mut(off + t) = (u32::MAX, f32::MAX);
        }
    }
    // OOB pad so a 16-wide load at j..j+16 with j up to n-1 is in-bounds.
    let pad_end = n + 16;
    if worst.len() >= pad_end {
        for t in n..pad_end {
            *worst.get_unchecked_mut(t) = f32::MAX;
        }
    }
}

/// 16-wide AVX-512 fused dual-end top-k scan. See `KL2` etc. for the metric
/// kernel. Symmetric metrics only (all four PiPNN metrics qualify).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fused_dual_topk_wide<K: MetricKernel>(
    dot: &[f32],
    norms: &[f32],
    knn_result: &mut [(u32, f32)],
    worst: &mut [f32],
    n: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    fused_init(knn_result, worst, n, k);

    for i in 1..n {
        let ni = *norms.get_unchecked(i);
        let ni_v = _mm512_set1_ps(ni);
        let mut local_worst_i = *worst.get_unchecked(i);
        let row_base = i * n;
        let knn_i_base = i * k;

        let mut j = 0usize;
        while j + 16 <= i {
            let dot_v = _mm512_loadu_ps(dot.as_ptr().add(row_base + j));
            let nj_v = _mm512_loadu_ps(norms.as_ptr().add(j));
            let d_v = K::dist_wide(dot_v, ni_v, nj_v);

            let thresh_i_v = _mm512_set1_ps(local_worst_i);
            let mask_row = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(d_v, thresh_i_v);
            let worst_col_v = _mm512_loadu_ps(worst.as_ptr().add(j));
            let mask_col = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(d_v, worst_col_v);

            if (mask_row | mask_col) != 0 {
                let mut d_arr = [0.0f32; 16];
                _mm512_storeu_ps(d_arr.as_mut_ptr(), d_v);

                let mut m = mask_row;
                while m != 0 {
                    let lane = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let d = *d_arr.get_unchecked(lane);
                    // Re-check against the LIVE local_worst_i: earlier lanes in this
                    // chunk may have already tightened the threshold. Without this,
                    // a strictly-worse lane displaces the genuine k-th best because
                    // insert_topk_linear writes unconditionally to slot k-1.
                    if d >= local_worst_i {
                        continue;
                    }
                    let j_abs = (j + lane) as u32;
                    local_worst_i = insert_topk_linear(knn_result, knn_i_base, k, j_abs, d);
                }
                let mut m = mask_col;
                while m != 0 {
                    let lane = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let d = *d_arr.get_unchecked(lane);
                    let j_target = j + lane;
                    let new_worst = insert_topk_linear(knn_result, j_target * k, k, i as u32, d);
                    *worst.get_unchecked_mut(j_target) = new_worst;
                }
            }
            j += 16;
        }
        while j < i {
            let dot_ij = *dot.get_unchecked(row_base + j);
            let nj = *norms.get_unchecked(j);
            let d = K::dist_scalar(dot_ij, ni, nj);
            if d < local_worst_i {
                local_worst_i = insert_topk_linear(knn_result, knn_i_base, k, j as u32, d);
            }
            if d < *worst.get_unchecked(j) {
                let new_worst = insert_topk_linear(knn_result, j * k, k, i as u32, d);
                *worst.get_unchecked_mut(j) = new_worst;
            }
            j += 1;
        }
        *worst.get_unchecked_mut(i) = local_worst_i;
    }
}

/// 8-wide AVX-2 fused dual-end top-k scan. Mirror of the AVX-512 variant with
/// `_mm256` ops and `movemask_ps`-based mask extraction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_dual_topk_narrow<K: MetricKernel>(
    dot: &[f32],
    norms: &[f32],
    knn_result: &mut [(u32, f32)],
    worst: &mut [f32],
    n: usize,
    k: usize,
) {
    use std::arch::x86_64::*;
    fused_init(knn_result, worst, n, k);

    for i in 1..n {
        let ni = *norms.get_unchecked(i);
        let ni_v = _mm256_set1_ps(ni);
        let mut local_worst_i = *worst.get_unchecked(i);
        let row_base = i * n;
        let knn_i_base = i * k;

        let mut j = 0usize;
        while j + 8 <= i {
            let dot_v = _mm256_loadu_ps(dot.as_ptr().add(row_base + j));
            let nj_v = _mm256_loadu_ps(norms.as_ptr().add(j));
            let d_v = K::dist_narrow(dot_v, ni_v, nj_v);

            let thresh_i_v = _mm256_set1_ps(local_worst_i);
            let cmp_row = _mm256_cmp_ps::<_CMP_LT_OQ>(d_v, thresh_i_v);
            let worst_col_v = _mm256_loadu_ps(worst.as_ptr().add(j));
            let cmp_col = _mm256_cmp_ps::<_CMP_LT_OQ>(d_v, worst_col_v);
            let mask_row = _mm256_movemask_ps(cmp_row) as u32;
            let mask_col = _mm256_movemask_ps(cmp_col) as u32;

            if (mask_row | mask_col) != 0 {
                let mut d_arr = [0.0f32; 8];
                _mm256_storeu_ps(d_arr.as_mut_ptr(), d_v);

                let mut m = mask_row;
                while m != 0 {
                    let lane = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let d = *d_arr.get_unchecked(lane);
                    // Re-check against the LIVE local_worst_i: earlier lanes in this
                    // chunk may have already tightened the threshold. Without this,
                    // a strictly-worse lane displaces the genuine k-th best because
                    // insert_topk_linear writes unconditionally to slot k-1.
                    if d >= local_worst_i {
                        continue;
                    }
                    let j_abs = (j + lane) as u32;
                    local_worst_i = insert_topk_linear(knn_result, knn_i_base, k, j_abs, d);
                }
                let mut m = mask_col;
                while m != 0 {
                    let lane = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let d = *d_arr.get_unchecked(lane);
                    let j_target = j + lane;
                    let new_worst = insert_topk_linear(knn_result, j_target * k, k, i as u32, d);
                    *worst.get_unchecked_mut(j_target) = new_worst;
                }
            }
            j += 8;
        }
        while j < i {
            let dot_ij = *dot.get_unchecked(row_base + j);
            let nj = *norms.get_unchecked(j);
            let d = K::dist_scalar(dot_ij, ni, nj);
            if d < local_worst_i {
                local_worst_i = insert_topk_linear(knn_result, knn_i_base, k, j as u32, d);
            }
            if d < *worst.get_unchecked(j) {
                let new_worst = insert_topk_linear(knn_result, j * k, k, i as u32, d);
                *worst.get_unchecked_mut(j) = new_worst;
            }
            j += 1;
        }
        *worst.get_unchecked_mut(i) = local_worst_i;
    }
}

/// Scalar fallback fused dual-end scan. Same logic, one lane at a time.
unsafe fn fused_dual_topk_scalar<K: MetricKernel>(
    dot: &[f32],
    norms: &[f32],
    knn_result: &mut [(u32, f32)],
    worst: &mut [f32],
    n: usize,
    k: usize,
) {
    fused_init(knn_result, worst, n, k);
    for i in 1..n {
        let ni = *norms.get_unchecked(i);
        let mut local_worst_i = *worst.get_unchecked(i);
        let row_base = i * n;
        let knn_i_base = i * k;
        for j in 0..i {
            let dot_ij = *dot.get_unchecked(row_base + j);
            let nj = *norms.get_unchecked(j);
            let d = K::dist_scalar(dot_ij, ni, nj);
            if d < local_worst_i {
                local_worst_i = insert_topk_linear(knn_result, knn_i_base, k, j as u32, d);
            }
            if d < *worst.get_unchecked(j) {
                let new_worst = insert_topk_linear(knn_result, j * k, k, i as u32, d);
                *worst.get_unchecked_mut(j) = new_worst;
            }
        }
        *worst.get_unchecked_mut(i) = local_worst_i;
    }
}

/// Tier × metric dispatch for the fused dual-end scan. Caller has already
/// filled `norms` per the metric's requirement (see `MetricKernel` doc).
#[inline]
fn fused_dual_topk(
    metric: diskann_vector::distance::Metric,
    dot: &[f32],
    norms: &[f32],
    knn_result: &mut [(u32, f32)],
    worst: &mut [f32],
    n: usize,
    k: usize,
) {
    use diskann_vector::distance::Metric;
    // SAFETY: each vector-width branch is only entered after checking the
    // matching CPU features.
    unsafe {
        match (fma_width(), metric) {
            #[cfg(target_arch = "x86_64")]
            (VectorWidth::Wide, Metric::L2) => {
                fused_dual_topk_wide::<KL2>(dot, norms, knn_result, worst, n, k)
            }
            #[cfg(target_arch = "x86_64")]
            (VectorWidth::Wide, Metric::CosineNormalized) => {
                fused_dual_topk_wide::<KCosNorm>(dot, norms, knn_result, worst, n, k)
            }
            #[cfg(target_arch = "x86_64")]
            (VectorWidth::Wide, Metric::Cosine) => {
                fused_dual_topk_wide::<KCosine>(dot, norms, knn_result, worst, n, k)
            }
            #[cfg(target_arch = "x86_64")]
            (VectorWidth::Wide, Metric::InnerProduct) => {
                fused_dual_topk_wide::<KIp>(dot, norms, knn_result, worst, n, k)
            }
            #[cfg(target_arch = "x86_64")]
            (VectorWidth::Narrow, Metric::L2) => {
                fused_dual_topk_narrow::<KL2>(dot, norms, knn_result, worst, n, k)
            }
            #[cfg(target_arch = "x86_64")]
            (VectorWidth::Narrow, Metric::CosineNormalized) => {
                fused_dual_topk_narrow::<KCosNorm>(dot, norms, knn_result, worst, n, k)
            }
            #[cfg(target_arch = "x86_64")]
            (VectorWidth::Narrow, Metric::Cosine) => {
                fused_dual_topk_narrow::<KCosine>(dot, norms, knn_result, worst, n, k)
            }
            #[cfg(target_arch = "x86_64")]
            (VectorWidth::Narrow, Metric::InnerProduct) => {
                fused_dual_topk_narrow::<KIp>(dot, norms, knn_result, worst, n, k)
            }
            (_, Metric::L2) => fused_dual_topk_scalar::<KL2>(dot, norms, knn_result, worst, n, k),
            (_, Metric::CosineNormalized) => {
                fused_dual_topk_scalar::<KCosNorm>(dot, norms, knn_result, worst, n, k)
            }
            (_, Metric::Cosine) => {
                fused_dual_topk_scalar::<KCosine>(dot, norms, knn_result, worst, n, k)
            }
            (_, Metric::InnerProduct) => {
                fused_dual_topk_scalar::<KIp>(dot, norms, knn_result, worst, n, k)
            }
        }
    }
}

/// Build a leaf into the caller-provided CSR buffers and return the edge count.
///
/// Pipeline: one triangular GEMM (`sgemm_aat_lower`) over the full leaf, read
/// per-row norms off the SYRK diagonal, then either the fused dual-end SIMD
/// scan selected for the active metric and CPU width.
pub(crate) fn build_leaf_into<T: VectorRepr + 'static>(
    data: &[T],
    ndims: usize,
    indices: &[u32],
    k: usize,
    metric: diskann_vector::distance::Metric,
    bufs: &mut LeafBuffers,
) -> diskann::ANNResult<usize> {
    use diskann_vector::distance::Metric;

    let n = indices.len();
    bufs.ensure_capacity(n, ndims, k);

    let actual_k = if k == 0 || n <= 1 { 0 } else { k.min(n - 1) };

    // ───── Gather (norms computed below from the SYRK diagonal) ─────
    {
        let local_data = &mut bufs.local_data[..n * ndims];
        crate::partition::gather_rows(data, indices, ndims, local_data)?;
    }

    bufs.knn_result.clear();
    if actual_k > 0 {
        bufs.knn_result.resize(n * actual_k, (u32::MAX, f32::MAX));
    }

    if actual_k > 0 {
        // ───── Step 1: triangular GEMM (lower triangle only) ─────
        // C = A · Aᵀ, n × n output; only the lower triangle (j ≤ i) is written.
        // The diagonal `dot[i*n + i]` equals `‖A[i,:]‖²` — exactly the per-row
        // norm we used to compute in a separate pass. We read it directly
        // below, skipping `compute_p_norm_sq_batch_into` entirely (saves one
        // streaming read of the n × ndims data matrix per leaf).
        {
            let a_full = &bufs.local_data[..n * ndims];
            let dot = &mut bufs.dot_matrix[..n * n];
            diskann_linalg::sgemm_aat_lower(a_full, n, ndims, dot)
                .map_err(diskann::ANNError::opaque)?;
        }

        // ───── Step 2: extract per-row scalar from the SYRK diagonal ─────
        // L2 needs ‖x‖² directly (read off `dot[i*n+i]`).
        // Cosine needs ‖x‖ (sqrt of the SYRK diagonal).
        // CosineNormalized and InnerProduct ignore the norms array, so we still
        // populate `norms_sq` with the diagonal as a harmless filler.
        let norms_target: &[f32] = {
            let dot = &bufs.dot_matrix[..n * n];
            let norms_sq = &mut bufs.norms_sq[..n];
            for (i, norm) in norms_sq.iter_mut().enumerate() {
                // SAFETY: i < n bounds the index; dot is at least n*n.
                *norm = unsafe { *dot.get_unchecked(i * n + i) };
            }
            if matches!(metric, Metric::Cosine) {
                if bufs.cosine_denoms.len() < n {
                    bufs.cosine_denoms.resize(n, 0.0);
                }
                for i in 0..n {
                    bufs.cosine_denoms[i] = bufs.norms_sq[i].sqrt();
                }
                &bufs.cosine_denoms[..n]
            } else {
                &bufs.norms_sq[..n]
            }
        };

        // ───── Step 3: fused dual-end top-k scan ─────
        // Walks the strictly-lower triangle once; one distance updates BOTH
        // tracker[i] AND tracker[j] (all four PiPNN metrics are symmetric).
        // Dispatches per (SIMD tier, metric); no upper-triangle materialisation.
        fused_dual_topk(
            metric,
            &bufs.dot_matrix[..n * n],
            norms_target,
            &mut bufs.knn_result[..n * actual_k],
            &mut bufs.worst[..n + 16],
            n,
            actual_k,
        );
    }

    // ───── CSR build (bidirected edges, dedup against `seen`) ─────
    let seen = &mut bufs.seen[..n * n];
    seen.fill(false);
    bufs.group_starts.clear();
    bufs.group_starts.resize(n + 1, 0);
    for i in 0..n {
        let row_knn = &bufs.knn_result[i * actual_k..(i + 1) * actual_k];
        for &(dst_local, _) in row_knn {
            if dst_local == u32::MAX {
                continue;
            }
            let dst = dst_local as usize;
            if !seen[i * n + dst] {
                seen[i * n + dst] = true;
                bufs.group_starts[i + 1] += 1;
            }
            if !seen[dst * n + i] {
                seen[dst * n + i] = true;
                bufs.group_starts[dst + 1] += 1;
            }
        }
    }
    for i in 1..=n {
        bufs.group_starts[i] += bufs.group_starts[i - 1];
    }
    let total_edges_n = bufs.group_starts[n] as usize;
    if bufs.group_data.len() < total_edges_n {
        bufs.group_data.resize(total_edges_n, (0, 0.0));
    }
    bufs.cursor.clear();
    bufs.cursor.extend_from_slice(&bufs.group_starts[..n]);
    seen.fill(false);
    for i in 0..n {
        let row_knn = &bufs.knn_result[i * actual_k..(i + 1) * actual_k];
        for &(dst_local, dist) in row_knn {
            if dst_local == u32::MAX {
                continue;
            }
            let dst = dst_local as usize;
            if !seen[i * n + dst] {
                seen[i * n + dst] = true;
                let pos = bufs.cursor[i] as usize;
                bufs.group_data[pos] = (dst_local, dist);
                bufs.cursor[i] = (pos + 1) as u32;
            }
            if !seen[dst * n + i] {
                seen[dst * n + i] = true;
                let pos = bufs.cursor[dst] as usize;
                bufs.group_data[pos] = (i as u32, dist);
                bufs.cursor[dst] = (pos + 1) as u32;
            }
        }
    }

    Ok(total_edges_n)
}

#[cfg(test)]
mod tests;
