/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf building: GEMM-based all-pairs distance computation and bi-directed k-NN extraction.
//!
//! For each leaf partition (bounded by C_max, typically 1024-2048):
//! 1. Compute all-pairs distance matrix via GEMM
//!    For L2: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*(a.b)
//!    The dot product matrix A * A^T is computed as a GEMM operation.
//! 2. Extract k nearest neighbors per point using partial sort
//! 3. Create bi-directed edges (both forward and reverse k-NN)

use std::cell::RefCell;

use diskann::utils::VectorRepr;
#[cfg(test)]
use diskann_vector::distance::SquaredL2;
#[cfg(test)]
use diskann_vector::PureDistanceFunction;

use crate::cpu_dispatch::{tier, SimdTier};

/// Thread-local reusable buffers for leaf building.
/// Avoids repeated allocation/deallocation of large matrices.
pub(crate) struct LeafBuffers {
    pub local_data: Vec<f32>,
    pub norms_sq: Vec<f32>,
    pub dot_matrix: Vec<f32>,
    pub seen: Vec<bool>,
    /// Reusable buffer for knn results: (local_dst_idx, distance) per row×k.
    pub knn_result: Vec<(u32, f32)>,
    /// Reusable buffer for bidirected edges output.
    pub edges: Vec<Edge>,
    /// Reusable Cosine sqrt-of-norms scratch (only filled for `Metric::Cosine`).
    pub cosine_denoms: Vec<f32>,
    /// CSR-style per-source edge groups: data[starts[src]..starts[src+1]] is
    /// the list of (local_dst, dist) pairs for local source `src`. Avoids per-leaf
    /// Vec<Vec<...>> allocation while preserving src-grouped insertion order for
    /// HP. Sized by ensure_capacity to fit `n + 1` starts and `2 * n * k` entries.
    pub group_starts: Vec<u32>,
    pub group_data: Vec<(u32, f32)>,
    /// L1-resident per-leaf cache of LSH sketches for the leaf's local points.
    /// Sized `n × num_planes`. Populated alongside `local_data` in gather so HP
    /// insert reads from this buffer (cache-hot) instead of the global sketches
    /// array (multi-hundred-MB, cache-cold).
    pub local_sketches: Vec<f32>,
    /// Per-row top-k threshold (current k-th smallest distance) for the fused
    /// dual-end scan path (v3). Sized `n`. Initialised to `f32::MAX`.
    pub worst: Vec<f32>,
}

impl Default for LeafBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl LeafBuffers {
    pub fn new() -> Self {
        Self {
            local_data: Vec::new(),
            norms_sq: Vec::new(),
            dot_matrix: Vec::new(),
            seen: Vec::new(),
            knn_result: Vec::new(),
            edges: Vec::new(),
            cosine_denoms: Vec::new(),
            group_starts: Vec::new(),
            group_data: Vec::new(),
            local_sketches: Vec::new(),
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
        // Pre-size knn_result and edges so per-leaf push/extend hits no realloc
        // (Vec realloc contends on the glibc malloc arena at high thread count).
        let actual_k = k.min(n.saturating_sub(1));
        let max_knn = n * actual_k.max(1);
        let max_edges = 2 * max_knn;
        if self.knn_result.capacity() < max_knn {
            self.knn_result
                .reserve(max_knn - self.knn_result.capacity());
        }
        if self.edges.capacity() < max_edges {
            self.edges.reserve(max_edges - self.edges.capacity());
        }
        if self.group_starts.capacity() < n + 1 {
            self.group_starts.reserve(n + 1 - self.group_starts.capacity());
        }
        if self.group_data.capacity() < max_edges {
            self.group_data.reserve(max_edges - self.group_data.capacity());
        }
        // worst[] threshold array: n entries (+15 for the 16-wide chunk tail
        // load so we can read past `i` without OOB). Initialised by caller.
        if self.worst.len() < n + 16 {
            self.worst.resize(n + 16, f32::MAX);
        }
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
    LEAF_BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        bufs.local_data = Vec::new();
        bufs.norms_sq = Vec::new();
        bufs.dot_matrix = Vec::new();
        bufs.seen = Vec::new();
        bufs.knn_result = Vec::new();
        bufs.edges = Vec::new();
        bufs.cosine_denoms = Vec::new();
        bufs.group_starts = Vec::new();
        bufs.group_data = Vec::new();
        bufs.local_sketches = Vec::new();
        bufs.worst = Vec::new();
    });
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
// The fused scan calls `K::dist_avx512` / `K::dist_avx2` / `K::dist_scalar`
// inside the hot loop; monomorphization inlines the formula at compile time.

trait MetricKernel {
    #[cfg(target_arch = "x86_64")]
    unsafe fn dist_avx512(
        dot: std::arch::x86_64::__m512,
        ni: std::arch::x86_64::__m512,
        nj: std::arch::x86_64::__m512,
    ) -> std::arch::x86_64::__m512;
    #[cfg(target_arch = "x86_64")]
    unsafe fn dist_avx2(
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
    unsafe fn dist_avx512(
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
    unsafe fn dist_avx2(
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
    unsafe fn dist_avx512(
        dot: std::arch::x86_64::__m512,
        _ni: std::arch::x86_64::__m512,
        _nj: std::arch::x86_64::__m512,
    ) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        _mm512_max_ps(_mm512_sub_ps(_mm512_set1_ps(1.0), dot), _mm512_setzero_ps())
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dist_avx2(
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
    unsafe fn dist_avx512(
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
    unsafe fn dist_avx2(
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
    unsafe fn dist_avx512(
        dot: std::arch::x86_64::__m512,
        _ni: std::arch::x86_64::__m512,
        _nj: std::arch::x86_64::__m512,
    ) -> std::arch::x86_64::__m512 {
        use std::arch::x86_64::*;
        _mm512_sub_ps(_mm512_setzero_ps(), dot)
    }
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn dist_avx2(
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
    *knn_result.get_unchecked_mut(base + k - 1) = (idx, d);
    let mut pos = base + k - 1;
    while pos > base {
        let cur = *knn_result.get_unchecked(pos);
        let prev = *knn_result.get_unchecked(pos - 1);
        let swap = cur.1 < prev.1;
        *knn_result.get_unchecked_mut(pos) = if swap { prev } else { cur };
        *knn_result.get_unchecked_mut(pos - 1) = if swap { cur } else { prev };
        pos -= 1;
    }
    (*knn_result.get_unchecked(base + k - 1)).1
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
unsafe fn fused_dual_topk_avx512<K: MetricKernel>(
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
            let d_v = K::dist_avx512(dot_v, ni_v, nj_v);

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
                    let j_abs = (j + lane) as u32;
                    local_worst_i = insert_topk_linear(knn_result, knn_i_base, k, j_abs, d);
                }
                let mut m = mask_col;
                while m != 0 {
                    let lane = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let d = *d_arr.get_unchecked(lane);
                    let j_target = j + lane;
                    let new_worst =
                        insert_topk_linear(knn_result, j_target * k, k, i as u32, d);
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
unsafe fn fused_dual_topk_avx2<K: MetricKernel>(
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
            let d_v = K::dist_avx2(dot_v, ni_v, nj_v);

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
                    let j_abs = (j + lane) as u32;
                    local_worst_i = insert_topk_linear(knn_result, knn_i_base, k, j_abs, d);
                }
                let mut m = mask_col;
                while m != 0 {
                    let lane = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let d = *d_arr.get_unchecked(lane);
                    let j_target = j + lane;
                    let new_worst =
                        insert_topk_linear(knn_result, j_target * k, k, i as u32, d);
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
    // SAFETY: each AVX-tier branch is only entered when the matching feature
    // was confirmed at startup by `cpu_dispatch::tier()`.
    unsafe {
        match (tier(), metric) {
            #[cfg(target_arch = "x86_64")]
            (SimdTier::Avx512, Metric::L2) => fused_dual_topk_avx512::<KL2>(dot, norms, knn_result, worst, n, k),
            #[cfg(target_arch = "x86_64")]
            (SimdTier::Avx512, Metric::CosineNormalized) => fused_dual_topk_avx512::<KCosNorm>(dot, norms, knn_result, worst, n, k),
            #[cfg(target_arch = "x86_64")]
            (SimdTier::Avx512, Metric::Cosine) => fused_dual_topk_avx512::<KCosine>(dot, norms, knn_result, worst, n, k),
            #[cfg(target_arch = "x86_64")]
            (SimdTier::Avx512, Metric::InnerProduct) => fused_dual_topk_avx512::<KIp>(dot, norms, knn_result, worst, n, k),
            #[cfg(target_arch = "x86_64")]
            (SimdTier::Avx2, Metric::L2) => fused_dual_topk_avx2::<KL2>(dot, norms, knn_result, worst, n, k),
            #[cfg(target_arch = "x86_64")]
            (SimdTier::Avx2, Metric::CosineNormalized) => fused_dual_topk_avx2::<KCosNorm>(dot, norms, knn_result, worst, n, k),
            #[cfg(target_arch = "x86_64")]
            (SimdTier::Avx2, Metric::Cosine) => fused_dual_topk_avx2::<KCosine>(dot, norms, knn_result, worst, n, k),
            #[cfg(target_arch = "x86_64")]
            (SimdTier::Avx2, Metric::InnerProduct) => fused_dual_topk_avx2::<KIp>(dot, norms, knn_result, worst, n, k),
            (_, Metric::L2) => fused_dual_topk_scalar::<KL2>(dot, norms, knn_result, worst, n, k),
            (_, Metric::CosineNormalized) => fused_dual_topk_scalar::<KCosNorm>(dot, norms, knn_result, worst, n, k),
            (_, Metric::Cosine) => fused_dual_topk_scalar::<KCosine>(dot, norms, knn_result, worst, n, k),
            (_, Metric::InnerProduct) => fused_dual_topk_scalar::<KIp>(dot, norms, knn_result, worst, n, k),
        }
    }
}

/// An edge produced by leaf building: `(source, destination, distance)`.
///
/// Returned by [`build_leaf`] for benchmarking. Production code uses
/// [`build_leaf_into`] which writes edges into a caller-provided buffer.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    /// Global source point index.
    pub src: u32,
    /// Global destination point index.
    pub dst: u32,
    /// Distance from `src` to `dst` under the build metric.
    pub distance: f32,
}

/// Build a leaf partition: compute all-pairs distances and extract bi-directed k-NN edges.
///
/// Returns edges as (global_src, global_dst, distance).
pub fn build_leaf<T: VectorRepr + 'static>(
    data: &[T],
    ndims: usize,
    indices: &[u32],
    k: usize,
    metric: diskann_vector::distance::Metric,
) -> Vec<Edge> {
    let n = indices.len();
    if n <= 1 {
        return Vec::new();
    }

    LEAF_BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        build_leaf_with_buffers(data, ndims, indices, k, metric, &mut bufs)
    })
}

/// Fused per-row top-k for k<=3. Dispatches to the AVX-512, AVX-2, or scalar
/// implementation based on the runtime CPU tier. Writes `actual_k` entries
/// into `out[..actual_k]`.

/// Materialize `bufs.edges` from the CSR (`group_starts`, `group_data`) layout.
/// Called from `build_leaf_with_buffers` for back-compat — the public `Vec<Edge>`
/// return is used by tests and `build_leaf`. Production callers go through
/// `build_leaf_into` and consume the CSR directly via
/// `HashPrune::add_edges_grouped_local_sketches`, but `bufs.edges` is also
/// re-populated here so the API stays consistent for downstream readers.
fn materialize_edges_from_csr(bufs: &mut LeafBuffers, indices: &[u32]) {
    let n = indices.len();
    let total = bufs.group_starts[n] as usize;
    bufs.edges.clear();
    bufs.edges.reserve(total);
    for local_src in 0..n {
        let start = bufs.group_starts[local_src] as usize;
        let end = bufs.group_starts[local_src + 1] as usize;
        let src = indices[local_src];
        for &(dst_local, dist) in &bufs.group_data[start..end] {
            bufs.edges.push(Edge {
                src,
                dst: indices[dst_local as usize],
                distance: dist,
            });
        }
    }
}

/// Build a leaf into `bufs.edges` without allocating. Returns edge count.
/// The caller reads edges from `bufs.edges[..returned_count]`.
pub(crate) fn build_leaf_into<T: VectorRepr + 'static>(
    data: &[T],
    ndims: usize,
    indices: &[u32],
    k: usize,
    metric: diskann_vector::distance::Metric,
    bufs: &mut LeafBuffers,
) -> usize {
    let _edges = build_leaf_with_buffers(data, ndims, indices, k, metric, bufs);
    bufs.group_starts[indices.len()] as usize
}


/// Build a leaf using caller-provided buffers, bypassing thread-local access.
///
/// Pipeline: one triangular GEMM (`sgemm_aat_lower`) over the full leaf, read
/// per-row norms off the SYRK diagonal, then either the fused dual-end SIMD
/// scan (L2 + k=3 + AVX-512) or symmetrize + per-row convert + per-row top-k.
pub(crate) fn build_leaf_with_buffers<T: VectorRepr + 'static>(
    data: &[T],
    ndims: usize,
    indices: &[u32],
    k: usize,
    metric: diskann_vector::distance::Metric,
    bufs: &mut LeafBuffers,
) -> Vec<Edge> {
    use diskann_vector::distance::Metric;

    let n = indices.len();
    bufs.ensure_capacity(n, ndims, k);

    let actual_k = if k == 0 || n <= 1 { 0 } else { k.min(n - 1) };

    // ───── Gather (norms computed below from the SYRK diagonal) ─────
    {
        let local_data = &mut bufs.local_data[..n * ndims];
        for (i, &idx) in indices.iter().enumerate() {
            crate::partition::gather_f16_to_f32_simd(
                data,
                idx as usize,
                ndims,
                &mut local_data[i * ndims..(i + 1) * ndims],
            );
        }
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
            diskann_linalg::sgemm_aat_lower(a_full, n, ndims, dot);
        }

        // ───── Step 2: extract per-row scalar from the SYRK diagonal ─────
        // L2 needs ‖x‖² directly (read off `dot[i*n+i]`).
        // Cosine needs ‖x‖ (sqrt of the SYRK diagonal).
        // CosineNormalized and InnerProduct ignore the norms array, so we still
        // populate `norms_sq` with the diagonal as a harmless filler.
        let norms_target: &[f32] = {
            let dot = &bufs.dot_matrix[..n * n];
            let norms_sq = &mut bufs.norms_sq[..n];
            for i in 0..n {
                // SAFETY: i < n bounds the index; dot is at least n*n.
                norms_sq[i] = unsafe { *dot.get_unchecked(i * n + i) };
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
    let mut cursor: Vec<u32> = bufs.group_starts[..n].to_vec();
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
                let pos = cursor[i] as usize;
                bufs.group_data[pos] = (dst_local, dist);
                cursor[i] = (pos + 1) as u32;
            }
            if !seen[dst * n + i] {
                seen[dst * n + i] = true;
                let pos = cursor[dst] as usize;
                bufs.group_data[pos] = (i as u32, dist);
                cursor[dst] = (pos + 1) as u32;
            }
        }
    }

    if cfg!(test) {
        materialize_edges_from_csr(bufs, indices);
    }
    std::mem::take(&mut bufs.edges)
}

/// Brute-force exact k-NN under squared-L2 distance. Test-only helper.
#[cfg(test)]
pub(crate) fn brute_force_knn(
    data: &[f32],
    ndims: usize,
    npoints: usize,
    query: &[f32],
    k: usize,
) -> Vec<(usize, f32)> {
    let mut dists: Vec<(usize, f32)> = (0..npoints)
        .map(|i| {
            let point = &data[i * ndims..(i + 1) * ndims];
            let dist = SquaredL2::evaluate(point, query);
            (i, dist)
        })
        .collect();

    let actual_k = k.min(npoints);
    if actual_k > 0 && actual_k < dists.len() {
        dists.select_nth_unstable_by(actual_k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        dists.truncate(actual_k);
    }
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_vector::distance::{DistanceProvider, Metric};

    #[test]
    fn test_gemm_aat() {
        // 2x3 matrix:
        // [1 2 3]
        // [4 5 6]
        // A * A^T should be:
        // [14 32]
        // [32 77]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut result = vec![0.0; 4];
        diskann_linalg::sgemm_aat(&a, 2, 3, &mut result);

        assert!((result[0] - 14.0).abs() < 1e-6);
        assert!((result[1] - 32.0).abs() < 1e-6);
        assert!((result[2] - 32.0).abs() < 1e-6);
        assert!((result[3] - 77.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_l2() {
        let dist_fn = <f32 as DistanceProvider<f32>>::distance_comparer(Metric::L2, Some(2));
        let p0 = [0.0f32, 0.0];
        let p1 = [1.0f32, 0.0];
        let p2 = [0.0f32, 1.0];
        // dist(0,1) = 1
        assert!((dist_fn.call(&p0, &p1) - 1.0).abs() < 1e-6);
        // dist(0,2) = 1
        assert!((dist_fn.call(&p0, &p2) - 1.0).abs() < 1e-6);
        // dist(1,2) = 2
        assert!((dist_fn.call(&p1, &p2) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_leaf() {
        let data = vec![
            0.0, 0.0, // point 0
            1.0, 0.0, // point 1
            0.0, 1.0, // point 2
            1.0, 1.0, // point 3
        ];
        let indices = vec![0, 1, 2, 3];

        let edges = build_leaf(&data, 2, &indices, 2, Metric::L2);

        assert!(!edges.is_empty());

        for edge in &edges {
            assert!(edge.src < 4);
            assert!(edge.dst < 4);
            assert!(edge.src != edge.dst);
            assert!(edge.distance >= 0.0);
        }
    }

    #[test]
    fn test_brute_force_knn() {
        let data = vec![
            0.0, 0.0, // point 0
            1.0, 0.0, // point 1
            0.0, 1.0, // point 2
            1.0, 1.0, // point 3
        ];
        let query = vec![0.1, 0.1];
        let results = brute_force_knn(&data, 2, 4, &query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_build_leaf_cosine() {
        // Verify that cosine distance path works correctly with normalized vectors.
        let mut data = vec![
            1.0, 0.0, // point 0: along x
            0.0, 1.0, // point 1: along y
            0.707, 0.707, // point 2: 45 degrees
            -1.0, 0.0, // point 3: negative x
        ];
        // Normalize all vectors.
        for i in 0..4 {
            let row = &mut data[i * 2..(i + 1) * 2];
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }

        let indices = vec![0, 1, 2, 3];
        let edges = build_leaf(&data, 2, &indices, 2, Metric::CosineNormalized);

        assert!(!edges.is_empty(), "cosine leaf should produce edges");

        for edge in &edges {
            assert!(edge.src < 4);
            assert!(edge.dst < 4);
            assert_ne!(edge.src, edge.dst);
            // Cosine distance for normalized vectors is in [0, 2].
            assert!(edge.distance >= 0.0, "negative cosine distance");
        }
    }

    #[test]
    fn test_build_leaf_single_point() {
        // A leaf with 1 point should produce no edges.
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let indices = vec![0];
        let edges = build_leaf(&data, 4, &indices, 3, Metric::L2);
        assert!(
            edges.is_empty(),
            "single point leaf should produce 0 edges, got {}",
            edges.len()
        );
    }

    #[test]
    fn test_build_leaf_two_points() {
        // A leaf with 2 points should produce bidirectional edges.
        let data = vec![0.0f32, 0.0, 1.0, 0.0];
        let indices = vec![0, 1];
        let edges = build_leaf(&data, 2, &indices, 3, Metric::L2);
        assert!(!edges.is_empty(), "two point leaf should produce edges");

        // Should have both directions: 0->1 and 1->0.
        let has_0_to_1 = edges.iter().any(|e| e.src == 0 && e.dst == 1);
        let has_1_to_0 = edges.iter().any(|e| e.src == 1 && e.dst == 0);
        assert!(has_0_to_1, "should have edge 0 -> 1");
        assert!(has_1_to_0, "should have edge 1 -> 0");
    }

    #[test]
    fn test_build_leaf_k_equals_n() {
        // k >= n, every point should connect to every other.
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let indices = vec![0, 1, 2, 3];
        let n = indices.len();
        // k = n means each point gets n-1 nearest neighbors = all others.
        let edges = build_leaf(&data, 2, &indices, n, Metric::L2);

        // Collect directed edges.
        let edge_set: std::collections::HashSet<(u32, u32)> =
            edges.iter().map(|e| (e.src, e.dst)).collect();

        // Every pair (i, j) with i != j should be present.
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(
                        edge_set.contains(&(i as u32, j as u32)),
                        "k >= n: edge ({} -> {}) should exist",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_build_leaf_with_buffers_reuse() {
        // Call build_leaf_with_buffers twice and verify buffers are reused.
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let indices = vec![0, 1, 2, 3];
        let mut bufs = LeafBuffers::new();

        let edges1 = build_leaf_with_buffers(&data, 2, &indices, 2, Metric::L2, &mut bufs);
        assert!(!edges1.is_empty(), "first call should produce edges");

        // Verify buffers are allocated.
        assert!(
            !bufs.local_data.is_empty(),
            "buffers should be allocated after first call"
        );

        // Second call with same data should still work.
        let edges2 = build_leaf_with_buffers(&data, 2, &indices, 2, Metric::L2, &mut bufs);
        assert_eq!(
            edges1.len(),
            edges2.len(),
            "same input should produce same number of edges with reused buffers"
        );
    }

    #[test]
    fn test_brute_force_knn_single_point() {
        let data = vec![5.0f32, 10.0];
        let query = vec![5.0, 10.0];
        let results = brute_force_knn(&data, 2, 1, &query, 5);
        assert_eq!(
            results.len(),
            1,
            "brute force on 1 point should return 1 result"
        );
        assert_eq!(results[0].0, 0, "should return the only point (index 0)");
        assert!(
            results[0].1 < 1e-6,
            "distance to identical query should be near zero"
        );
    }

    #[test]
    fn test_brute_force_knn_identity() {
        // query = data point, first result should be self with distance 0.
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let query = vec![1.0, 0.0]; // same as point 1
        let results = brute_force_knn(&data, 2, 4, &query, 3);
        assert_eq!(
            results[0].0, 1,
            "query identical to point 1 should find it first"
        );
        assert!(
            results[0].1 < 1e-6,
            "self-distance should be 0, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_edge_symmetry() {
        // Verify that build_leaf produces bi-directed edges:
        // if (a -> b) exists, then (b -> a) should also exist.
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        let indices = vec![0, 1, 2, 3, 4];
        let edges = build_leaf(&data, 2, &indices, 2, Metric::L2);

        // Collect directed edges as a set.
        let edge_set: std::collections::HashSet<(u32, u32)> =
            edges.iter().map(|e| (e.src, e.dst)).collect();

        // For every edge (a, b), (b, a) should also exist.
        for edge in &edges {
            assert!(
                edge_set.contains(&(edge.dst, edge.src)),
                "edge ({} -> {}) exists but reverse ({} -> {}) does not",
                edge.src,
                edge.dst,
                edge.dst,
                edge.src
            );
        }
    }

    #[test]
    fn test_build_leaf_cosine_unnormalized() {
        // Cosine (unnormalized) path: distance = 1 - dot(a,b)/(|a|*|b|).
        // Vectors with different norms but same direction should have distance ~0.
        let data = vec![
            1.0, 0.0, // point 0: unit x
            3.0, 0.0, // point 1: 3x in same direction
            0.0, 1.0, // point 2: unit y (orthogonal)
            1.0, 1.0, // point 3: 45 degrees
        ];
        let indices = vec![0, 1, 2, 3];
        let edges = build_leaf(&data, 2, &indices, 2, Metric::Cosine);

        assert!(!edges.is_empty());
        // Points 0 and 1 are co-linear — cosine distance should be ~0.
        let e01 = edges.iter().find(|e| e.src == 0 && e.dst == 1);
        assert!(e01.is_some(), "co-linear points should be neighbors");
        assert!(
            e01.unwrap().distance < 0.01,
            "cosine dist between co-linear should be ~0, got {}",
            e01.unwrap().distance
        );
    }

    #[test]
    fn test_build_leaf_inner_product() {
        // InnerProduct: distance = -dot(a,b). Lower (more negative) = closer.
        let data = vec![
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // dot with self = 2, dot with (1,0) = 1
        ];
        let indices = vec![0, 1, 2];
        let edges = build_leaf(&data, 2, &indices, 1, Metric::InnerProduct);
        assert!(!edges.is_empty());
    }

    #[test]
    fn test_build_leaf_large_k_clamped() {
        // k=1000 on 5 points should produce all-pairs edges (clamped to n-1=4).
        let data = vec![0.0f32; 5 * 4];
        let indices = vec![0, 1, 2, 3, 4];
        let edges = build_leaf(&data, 4, &indices, 1000, Metric::L2);
        let edge_set: std::collections::HashSet<(u32, u32)> =
            edges.iter().map(|e| (e.src, e.dst)).collect();
        // All pairs should exist.
        for i in 0..5 {
            for j in 0..5 {
                if i != j {
                    assert!(
                        edge_set.contains(&(i, j)),
                        "all-pairs edge ({}, {}) missing",
                        i,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_build_leaf_distances_nonnegative() {
        // All distance metrics should produce non-negative distances.
        let data = vec![-1.5, 2.3, 0.1, 0.7, -0.4, 1.9, 1.0, 1.0, 1.0];
        let indices = vec![0, 1, 2];
        for metric in [Metric::L2, Metric::Cosine, Metric::CosineNormalized] {
            let edges = build_leaf(&data, 3, &indices, 2, metric);
            for e in &edges {
                assert!(
                    e.distance >= 0.0,
                    "{:?}: negative distance {} for ({},{})",
                    metric,
                    e.distance,
                    e.src,
                    e.dst
                );
            }
        }
    }

    #[test]
    fn test_build_leaf_buffer_reuse_different_sizes() {
        // First call with large leaf, second with small — buffers should handle both.
        let data_large = vec![1.0f32; 20 * 4];
        let indices_large: Vec<u32> = (0..20).collect();
        let edges1 = build_leaf(&data_large, 4, &indices_large, 2, Metric::L2);
        assert!(!edges1.is_empty());

        // Second call with smaller leaf on same thread — should reuse thread-local buffers.
        let data_small = vec![1.0f32; 4 * 4];
        let indices_small: Vec<u32> = (0..4).collect();
        let edges2 = build_leaf(&data_small, 4, &indices_small, 2, Metric::L2);
        assert!(
            !edges2.is_empty(),
            "small leaf after large should work with reused buffers"
        );
    }
}
