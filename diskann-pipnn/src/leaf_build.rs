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

/// Fused dual-end top-k scan over the STRICTLY LOWER triangle of `dot`.
///
/// For each (i, j) with j < i, computes d = (norms[i] + norms[j] - 2·dot[i*n+j]).max(0).
/// That single distance is the (i, j) edge weight AND the (j, i) edge weight by
/// symmetry — so we update tracker[i] AND tracker[j] in one pass without ever
/// materialising the upper triangle.
///
/// The scan is 16-wide AVX-512: for each chunk of 16 lower-triangle entries in
/// row i, compute 16 distances, compare against the broadcast worst[i] AND a
/// contiguous load of worst[j..j+16]. Both masks are typically near-zero after
/// the k-th smallest threshold tightens, so the SIMD fast-path is just two
/// vector compares per chunk — no scalar work.
///
/// Diagonal (j == i, self) is naturally skipped since the inner loop runs j < i.
///
/// Replaces the (symmetrize + convert + topk) trio with a single 1.5× memory-cost
/// pass: reads only N²/2 distinct dot entries (half the matrix), and writes only
/// the k-NN trackers (O(N·k)) + worst[] (O(N)).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fused_dual_topk_l2_avx512(
    dot: &[f32],          // n*n; only strictly-lower triangle is valid (from sgemm_aat_lower)
    norms: &[f32],        // n: dot[i*n + i] diagonal extracted earlier
    knn_result: &mut [(u32, f32)], // n * actual_k
    worst: &mut [f32],    // n (+16 tail pad to allow OOB loads, see ensure_capacity)
    n: usize,
    actual_k: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(actual_k == 3, "fused path supports k=3 only; fall back otherwise");

    // Init worst[] and knn_result for this leaf.
    for i in 0..n {
        worst[i] = f32::MAX;
        let off = i * actual_k;
        for k in 0..actual_k {
            *knn_result.get_unchecked_mut(off + k) = (u32::MAX, f32::MAX);
        }
    }
    // Tail pad so a 16-wide load at index n-15..n is in-bounds.
    for k in n..(n + 16) {
        worst[k] = f32::MAX;
    }

    let two_v = _mm512_set1_ps(2.0);

    for i in 1..n {
        let ni = *norms.get_unchecked(i);
        let ni_v = _mm512_set1_ps(ni);
        let mut local_worst_i = worst[i];
        let row_base = i * n;
        let knn_i_base = i * actual_k;

        let mut j = 0usize;
        while j + 16 <= i {
            // Load 16 dot[i, j..j+16] (lower triangle row prefix).
            let dot_v = _mm512_loadu_ps(dot.as_ptr().add(row_base + j));
            let norms_j = _mm512_loadu_ps(norms.as_ptr().add(j));
            // d = max(0, ni + nj - 2 * dot)
            let sum_norms = _mm512_add_ps(ni_v, norms_j);
            let two_dot = _mm512_mul_ps(two_v, dot_v);
            let d_v = _mm512_max_ps(_mm512_sub_ps(sum_norms, two_dot), _mm512_setzero_ps());

            // Row-side: d < worst[i] (broadcast).
            let thresh_i_v = _mm512_set1_ps(local_worst_i);
            let mask_row = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(d_v, thresh_i_v);

            // Column-side: d < worst[j..j+16] (contig load).
            let worst_col_v = _mm512_loadu_ps(worst.as_ptr().add(j));
            let mask_col = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(d_v, worst_col_v);

            // Materialise 16 distances only if at least one side has a hit.
            if (mask_row | mask_col) != 0 {
                let mut d_arr = [0.0f32; 16];
                _mm512_storeu_ps(d_arr.as_mut_ptr(), d_v);

                // Row-side: update tracker[i] for each set lane.
                let mut m = mask_row;
                while m != 0 {
                    let lane = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let d = *d_arr.get_unchecked(lane);
                    let j_abs = (j + lane) as u32;
                    local_worst_i = insert3_branchless(knn_result, knn_i_base, j_abs, d);
                }

                // Column-side: update tracker[j+lane] for each set lane.
                let mut m = mask_col;
                while m != 0 {
                    let lane = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let d = *d_arr.get_unchecked(lane);
                    let j_target = j + lane;
                    let new_worst = insert3_branchless(knn_result, j_target * actual_k, i as u32, d);
                    *worst.get_unchecked_mut(j_target) = new_worst;
                }
            }
            j += 16;
        }
        // Scalar tail (j..i, where j + 16 > i).
        while j < i {
            let dot_ij = *dot.get_unchecked(row_base + j);
            let nj = *norms.get_unchecked(j);
            let d = (ni + nj - 2.0 * dot_ij).max(0.0);
            if d < local_worst_i {
                local_worst_i = insert3_branchless(knn_result, knn_i_base, j as u32, d);
            }
            if d < *worst.get_unchecked(j) {
                let new_worst = insert3_branchless(knn_result, j * actual_k, i as u32, d);
                *worst.get_unchecked_mut(j) = new_worst;
            }
            j += 1;
        }
        worst[i] = local_worst_i;
    }
}

/// Branchless heap-3 insert into `knn_result[base..base+3]`. Returns the new
/// worst (knn_result[base+2].1). Same recipe as topk_row_small's per-lane
/// insert: 3 cmov-style conditional swaps, zero branches in the body.
#[inline(always)]
unsafe fn insert3_branchless(
    knn_result: &mut [(u32, f32)],
    base: usize,
    idx: u32,
    d: f32,
) -> f32 {
    let e0 = *knn_result.get_unchecked(base);
    let e1 = *knn_result.get_unchecked(base + 1);
    let mut e2 = *knn_result.get_unchecked(base + 2);

    // d already known to be < e2.1 by caller; replace e2.
    if d < e2.1 {
        e2 = (idx, d);
    }
    // Bubble up.
    let sw = e2.1 < e1.1;
    let new_e1 = if sw { e2 } else { e1 };
    let new_e2 = if sw { e1 } else { e2 };
    let e1 = new_e1;
    let e2 = new_e2;

    let sw = e1.1 < e0.1;
    let new_e0 = if sw { e1 } else { e0 };
    let new_e1 = if sw { e0 } else { e1 };
    let e0 = new_e0;
    let e1 = new_e1;

    *knn_result.get_unchecked_mut(base) = e0;
    *knn_result.get_unchecked_mut(base + 1) = e1;
    *knn_result.get_unchecked_mut(base + 2) = e2;
    e2.1
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
#[inline]
fn topk_row_small(
    dist_row: &mut [f32],
    n: usize,
    self_idx: usize,
    actual_k: usize,
    out: &mut [(u32, f32)],
) {
    // SAFETY: each tier-specialized variant is dispatched only when the
    // matching runtime feature was detected by `cpu_dispatch::tier()`.
    unsafe {
        match tier() {
            SimdTier::Avx512 => topk_row_small_dispatch_avx512(dist_row, n, self_idx, actual_k, out),
            SimdTier::Avx2 => topk_row_small_dispatch_avx2(dist_row, n, self_idx, actual_k, out),
            SimdTier::Scalar => topk_row_small_scalar(dist_row, n, self_idx, actual_k, out),
        }
    }
}

/// AVX-512 variant of `topk_row_small`. 16-wide scan, branchless heap insert.
///
/// The heap-3 bubble-up is rewritten as 3 cmov-style conditional swaps with no
/// data-dependent branches: every accepted candidate goes through exactly the
/// same code path. This eliminates the source of HP's 79% branch-mispredict
/// concentration on topk_row_small per PMU profile.
///
/// `dist_row` is taken `&mut` so we can stamp `self_idx` to `f32::MAX` once at
/// entry, removing the per-iteration self-check branch from the hot loop.
///
/// SAFETY: caller MUST ensure the CPU supports AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn topk_row_small_dispatch_avx512(
    dist_row: &mut [f32],
    n: usize,
    self_idx: usize,
    actual_k: usize,
    out: &mut [(u32, f32)],
) {
    debug_assert!(actual_k <= 3);
    debug_assert!(self_idx < n);
    // Stamp self ineligible. One store; removes the per-iter self_idx check.
    *dist_row.get_unchecked_mut(self_idx) = f32::MAX;

    // Three running top entries kept in separate scalar slots (helps the
    // compiler keep them in registers; tuple swaps were preventing cmov
    // emission). Invariant after each insert: d0 ≤ d1 ≤ d2.
    let mut d0 = f32::MAX;
    let mut i0 = u32::MAX;
    let mut d1 = f32::MAX;
    let mut i1 = u32::MAX;
    let mut d2 = f32::MAX;
    let mut i2 = u32::MAX;

    use std::arch::x86_64::*;
    let chunks = n / 16;
    for chunk in 0..chunks {
        let base = chunk * 16;
        let thresh = _mm512_set1_ps(d2);
        let dists = _mm512_loadu_ps(dist_row.as_ptr().add(base));
        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(dists, thresh);
        if mask != 0 {
            let mut d_arr = [0.0f32; 16];
            _mm512_storeu_ps(d_arr.as_mut_ptr(), dists);
            let mut m = mask;
            while m != 0 {
                let lane = m.trailing_zeros() as usize;
                m &= m - 1;
                let j = (base + lane) as u32;
                let d = d_arr[lane];
                // Branchless 3-slot insert.
                let take = d < d2;
                d2 = if take { d } else { d2 };
                i2 = if take { j } else { i2 };
                let sw = d2 < d1;
                let nd1 = if sw { d2 } else { d1 };
                let nd2 = if sw { d1 } else { d2 };
                let ni1 = if sw { i2 } else { i1 };
                let ni2 = if sw { i1 } else { i2 };
                d1 = nd1;
                d2 = nd2;
                i1 = ni1;
                i2 = ni2;
                let sw = d1 < d0;
                let nd0 = if sw { d1 } else { d0 };
                let nd1 = if sw { d0 } else { d1 };
                let ni0 = if sw { i1 } else { i0 };
                let ni1 = if sw { i0 } else { i1 };
                d0 = nd0;
                d1 = nd1;
                i0 = ni0;
                i1 = ni1;
            }
        }
    }
    // Tail: scalar branchless insert for the n % 16 leftover.
    for j in (chunks * 16)..n {
        let d = *dist_row.get_unchecked(j);
        let j_u32 = j as u32;
        let take = d < d2;
        d2 = if take { d } else { d2 };
        i2 = if take { j_u32 } else { i2 };
        let sw = d2 < d1;
        let nd1 = if sw { d2 } else { d1 };
        let nd2 = if sw { d1 } else { d2 };
        let ni1 = if sw { i2 } else { i1 };
        let ni2 = if sw { i1 } else { i2 };
        d1 = nd1;
        d2 = nd2;
        i1 = ni1;
        i2 = ni2;
        let sw = d1 < d0;
        let nd0 = if sw { d1 } else { d0 };
        let nd1 = if sw { d0 } else { d1 };
        let ni0 = if sw { i1 } else { i0 };
        let ni1 = if sw { i0 } else { i1 };
        d0 = nd0;
        d1 = nd1;
        i0 = ni0;
        i1 = ni1;
    }

    let final_top: [(u32, f32); 3] = [(i0, d0), (i1, d1), (i2, d2)];
    out[..actual_k].copy_from_slice(&final_top[..actual_k]);
}

/// AVX-2 variant of `topk_row_small`. 8-wide scan, branchless heap insert.
///
/// Same branchless treatment as the AVX-512 variant: stamp self once,
/// run 3 cmov-style conditional swaps per accepted candidate.
///
/// SAFETY: caller MUST ensure the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn topk_row_small_dispatch_avx2(
    dist_row: &mut [f32],
    n: usize,
    self_idx: usize,
    actual_k: usize,
    out: &mut [(u32, f32)],
) {
    debug_assert!(actual_k <= 3);
    debug_assert!(self_idx < n);
    *dist_row.get_unchecked_mut(self_idx) = f32::MAX;

    let mut d0 = f32::MAX;
    let mut i0 = u32::MAX;
    let mut d1 = f32::MAX;
    let mut i1 = u32::MAX;
    let mut d2 = f32::MAX;
    let mut i2 = u32::MAX;

    use std::arch::x86_64::*;
    let chunks = n / 8;
    for chunk in 0..chunks {
        let base = chunk * 8;
        let thresh = _mm256_set1_ps(d2);
        let dists = _mm256_loadu_ps(dist_row.as_ptr().add(base));
        let mask = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(dists, thresh)) as u32;
        if mask != 0 {
            let mut d_arr = [0.0f32; 8];
            _mm256_storeu_ps(d_arr.as_mut_ptr(), dists);
            let mut m = mask;
            while m != 0 {
                let lane = m.trailing_zeros() as usize;
                m &= m - 1;
                let j = (base + lane) as u32;
                let d = d_arr[lane];
                let take = d < d2;
                d2 = if take { d } else { d2 };
                i2 = if take { j } else { i2 };
                let sw = d2 < d1;
                let nd1 = if sw { d2 } else { d1 };
                let nd2 = if sw { d1 } else { d2 };
                let ni1 = if sw { i2 } else { i1 };
                let ni2 = if sw { i1 } else { i2 };
                d1 = nd1;
                d2 = nd2;
                i1 = ni1;
                i2 = ni2;
                let sw = d1 < d0;
                let nd0 = if sw { d1 } else { d0 };
                let nd1 = if sw { d0 } else { d1 };
                let ni0 = if sw { i1 } else { i0 };
                let ni1 = if sw { i0 } else { i1 };
                d0 = nd0;
                d1 = nd1;
                i0 = ni0;
                i1 = ni1;
            }
        }
    }
    for j in (chunks * 8)..n {
        let d = *dist_row.get_unchecked(j);
        let j_u32 = j as u32;
        let take = d < d2;
        d2 = if take { d } else { d2 };
        i2 = if take { j_u32 } else { i2 };
        let sw = d2 < d1;
        let nd1 = if sw { d2 } else { d1 };
        let nd2 = if sw { d1 } else { d2 };
        let ni1 = if sw { i2 } else { i1 };
        let ni2 = if sw { i1 } else { i2 };
        d1 = nd1;
        d2 = nd2;
        i1 = ni1;
        i2 = ni2;
        let sw = d1 < d0;
        let nd0 = if sw { d1 } else { d0 };
        let nd1 = if sw { d0 } else { d1 };
        let ni0 = if sw { i1 } else { i0 };
        let ni1 = if sw { i0 } else { i1 };
        d0 = nd0;
        d1 = nd1;
        i0 = ni0;
        i1 = ni1;
    }

    let final_top: [(u32, f32); 3] = [(i0, d0), (i1, d1), (i2, d2)];
    out[..actual_k].copy_from_slice(&final_top[..actual_k]);
}

/// Scalar variant of `topk_row_small`. Used on non-x86 or when neither AVX-2
/// nor AVX-512 is present at runtime. Branchless 3-slot insert.
unsafe fn topk_row_small_scalar(
    dist_row: &mut [f32],
    n: usize,
    self_idx: usize,
    actual_k: usize,
    out: &mut [(u32, f32)],
) {
    debug_assert!(actual_k <= 3);
    debug_assert!(self_idx < n);
    *dist_row.get_unchecked_mut(self_idx) = f32::MAX;

    let mut d0 = f32::MAX;
    let mut i0 = u32::MAX;
    let mut d1 = f32::MAX;
    let mut i1 = u32::MAX;
    let mut d2 = f32::MAX;
    let mut i2 = u32::MAX;

    for (j, &d) in dist_row.iter().enumerate().take(n) {
        let j_u32 = j as u32;
        let take = d < d2;
        d2 = if take { d } else { d2 };
        i2 = if take { j_u32 } else { i2 };
        let sw = d2 < d1;
        let nd1 = if sw { d2 } else { d1 };
        let nd2 = if sw { d1 } else { d2 };
        let ni1 = if sw { i2 } else { i1 };
        let ni2 = if sw { i1 } else { i2 };
        d1 = nd1;
        d2 = nd2;
        i1 = ni1;
        i2 = ni2;
        let sw = d1 < d0;
        let nd0 = if sw { d1 } else { d0 };
        let nd1 = if sw { d0 } else { d1 };
        let ni0 = if sw { i1 } else { i0 };
        let ni1 = if sw { i0 } else { i1 };
        d0 = nd0;
        d1 = nd1;
        i0 = ni0;
        i1 = ni1;
    }

    let final_top: [(u32, f32); 3] = [(i0, d0), (i1, d1), (i2, d2)];
    out[..actual_k].copy_from_slice(&final_top[..actual_k]);
}

// On non-x86 hosts the AVX dispatch arms are unreachable; provide stub
// variants so the dispatch `match` is exhaustive without a separate cfg
// branch. The runtime tier on non-x86 is always `Scalar`, so these stubs
// are never called.
#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
unsafe fn topk_row_small_dispatch_avx512(
    dist_row: &mut [f32],
    n: usize,
    self_idx: usize,
    actual_k: usize,
    out: &mut [(u32, f32)],
) {
    topk_row_small_scalar(dist_row, n, self_idx, actual_k, out)
}
#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
unsafe fn topk_row_small_dispatch_avx2(
    dist_row: &mut [f32],
    n: usize,
    self_idx: usize,
    actual_k: usize,
    out: &mut [(u32, f32)],
) {
    topk_row_small_scalar(dist_row, n, self_idx, actual_k, out)
}

// ───── Per-row dist conversion kernels (CosineNormalized / L2 / InnerProduct) ─────
//
// Each metric has three tier-specialized variants (AVX-512, AVX-2, scalar) and
// a public `convert_<metric>` dispatcher. The dispatcher takes the row in
// place and (for L2) the `ni = norms_sq[global_i]` scalar plus a pointer to
// the per-row `norms_sq` for the column term. The bodies are byte-for-byte
// copies of the original `#[cfg(target_feature)]` blocks; only the gating
// has changed from compile-time to runtime.

/// CosineNormalized: row[j] = max(1 - row[j], 0).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn convert_cosnorm_avx512(row: &mut [f32], n: usize) {
    use std::arch::x86_64::*;
    let one = _mm512_set1_ps(1.0);
    let zero = _mm512_setzero_ps();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let d = _mm512_loadu_ps(row.as_ptr().add(base));
        let v = _mm512_max_ps(_mm512_sub_ps(one, d), zero);
        _mm512_storeu_ps(row.as_mut_ptr().add(base), v);
    }
    for j in (chunks * 16)..n {
        row[j] = (1.0 - row[j]).max(0.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convert_cosnorm_avx2(row: &mut [f32], n: usize) {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let d = _mm256_loadu_ps(row.as_ptr().add(base));
        let v = _mm256_max_ps(_mm256_sub_ps(one, d), zero);
        _mm256_storeu_ps(row.as_mut_ptr().add(base), v);
    }
    for j in (chunks * 8)..n {
        row[j] = (1.0 - row[j]).max(0.0);
    }
}

fn convert_cosnorm_scalar(row: &mut [f32], _n: usize) {
    for val in row.iter_mut() {
        *val = (1.0 - *val).max(0.0);
    }
}

#[inline]
fn convert_cosnorm(row: &mut [f32], n: usize) {
    // SAFETY: dispatched on cached `tier()`; SIMD variants are only invoked
    // when the matching feature was detected at startup.
    unsafe {
        match tier() {
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx512 => convert_cosnorm_avx512(row, n),
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx2 => convert_cosnorm_avx2(row, n),
            _ => convert_cosnorm_scalar(row, n),
        }
    }
}

/// L2: row[j] = max(ni + norms_sq[j] - 2*row[j], 0).
///
/// SAFETY: caller MUST ensure `norms_sq_ptr` points to at least `n` valid
/// f32 elements and `row.len() >= n`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn convert_l2_avx512(row: &mut [f32], n: usize, ni: f32, norms_sq_ptr: *const f32) {
    use std::arch::x86_64::*;
    let ni_v = _mm512_set1_ps(ni);
    let two = _mm512_set1_ps(2.0);
    let zero = _mm512_setzero_ps();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let dot = _mm512_loadu_ps(row.as_ptr().add(base));
        let norm = _mm512_loadu_ps(norms_sq_ptr.add(base));
        // d = ni + norm - 2*dot  (fnmadd: -2*dot + norm + ni)
        let d = _mm512_add_ps(ni_v, _mm512_fnmadd_ps(two, dot, norm));
        let v = _mm512_max_ps(d, zero);
        _mm512_storeu_ps(row.as_mut_ptr().add(base), v);
    }
    for j in (chunks * 16)..n {
        let nj = *norms_sq_ptr.add(j);
        row[j] = (ni + nj - 2.0 * row[j]).max(0.0);
    }
}

/// SAFETY: see `convert_l2_avx512`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn convert_l2_avx2(row: &mut [f32], n: usize, ni: f32, norms_sq_ptr: *const f32) {
    use std::arch::x86_64::*;
    let ni_v = _mm256_set1_ps(ni);
    let two = _mm256_set1_ps(2.0);
    let zero = _mm256_setzero_ps();
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let dot = _mm256_loadu_ps(row.as_ptr().add(base));
        let norm = _mm256_loadu_ps(norms_sq_ptr.add(base));
        let d = _mm256_add_ps(ni_v, _mm256_fnmadd_ps(two, dot, norm));
        let v = _mm256_max_ps(d, zero);
        _mm256_storeu_ps(row.as_mut_ptr().add(base), v);
    }
    for j in (chunks * 8)..n {
        let nj = *norms_sq_ptr.add(j);
        row[j] = (ni + nj - 2.0 * row[j]).max(0.0);
    }
}

/// SAFETY: caller MUST ensure `norms_sq_ptr` points to at least `n` valid
/// f32 elements.
unsafe fn convert_l2_scalar(row: &mut [f32], n: usize, ni: f32, norms_sq_ptr: *const f32) {
    for (j, val) in row.iter_mut().enumerate().take(n) {
        let nj = *norms_sq_ptr.add(j);
        *val = (ni + nj - 2.0 * *val).max(0.0);
    }
}

/// SAFETY: caller MUST ensure `norms_sq_ptr` is valid for at least `n` reads.
#[inline]
unsafe fn convert_l2(row: &mut [f32], n: usize, ni: f32, norms_sq_ptr: *const f32) {
    match tier() {
        #[cfg(target_arch = "x86_64")]
        SimdTier::Avx512 => convert_l2_avx512(row, n, ni, norms_sq_ptr),
        #[cfg(target_arch = "x86_64")]
        SimdTier::Avx2 => convert_l2_avx2(row, n, ni, norms_sq_ptr),
        _ => convert_l2_scalar(row, n, ni, norms_sq_ptr),
    }
}

/// InnerProduct: row[j] = -row[j].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn convert_ip_avx512(row: &mut [f32], n: usize) {
    use std::arch::x86_64::*;
    let sign = _mm512_set1_ps(-0.0f32);
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let d = _mm512_loadu_ps(row.as_ptr().add(base));
        let v = _mm512_xor_ps(d, sign);
        _mm512_storeu_ps(row.as_mut_ptr().add(base), v);
    }
    for j in (chunks * 16)..n {
        row[j] = -row[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn convert_ip_avx2(row: &mut [f32], n: usize) {
    use std::arch::x86_64::*;
    let sign = _mm256_set1_ps(-0.0f32);
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let d = _mm256_loadu_ps(row.as_ptr().add(base));
        let v = _mm256_xor_ps(d, sign);
        _mm256_storeu_ps(row.as_mut_ptr().add(base), v);
    }
    for j in (chunks * 8)..n {
        row[j] = -row[j];
    }
}

fn convert_ip_scalar(row: &mut [f32], _n: usize) {
    for val in row.iter_mut() {
        *val = -*val;
    }
}

#[inline]
fn convert_ip(row: &mut [f32], n: usize) {
    // SAFETY: dispatched on cached `tier()`.
    unsafe {
        match tier() {
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx512 => convert_ip_avx512(row, n),
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx2 => convert_ip_avx2(row, n),
            _ => convert_ip_scalar(row, n),
        }
    }
}

/// Build a leaf using caller-provided buffers, bypassing thread-local access.
///
/// Tiles the leaf GEMM in MR-row chunks (≤ 2 MB dot buffer → L2-resident),
/// converts to distance and extracts top-k inline. Never materializes the
/// full m×m dist matrix.
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

    // Fused row-tile size: tile output (mb × n × 4) targets L2 (≤2 MB/core).
    // At MR=256, n=1024 → 1 MB; at n=2048 → 2 MB. Falls back to MR=n for small n.
    const MR: usize = 256;
    let mr = MR.min(n);

    let actual_k = if k == 0 || n <= 1 { 0 } else { k.min(n - 1) };

    // ───── Gather + compute norms (no full GEMM yet) ─────
    let needs_norms = matches!(metric, Metric::L2 | Metric::Cosine);
    {
        let local_data = &mut bufs.local_data[..n * ndims];
        for (i, &idx) in indices.iter().enumerate() {
            crate::partition::gather_f16_to_f32_simd(
                data, idx as usize, ndims,
                &mut local_data[i * ndims..(i + 1) * ndims],
            );
        }
        if needs_norms {
            let norms_sq = &mut bufs.norms_sq[..n];
            crate::partition::compute_p_norm_sq_batch_into(local_data, n, ndims, norms_sq);
        }
    }

    // For Cosine, precompute sqrt(norms_sq) once into the reusable buf (no per-leaf alloc).
    if matches!(metric, Metric::Cosine) {
        if bufs.cosine_denoms.len() < n {
            bufs.cosine_denoms.resize(n, 0.0);
        }
        for i in 0..n {
            bufs.cosine_denoms[i] = bufs.norms_sq[i].sqrt();
        }
    }

    // ───── Reset knn_result to hold n × actual_k slots ─────
    bufs.knn_result.clear();
    if actual_k > 0 {
        bufs.knn_result.resize(n * actual_k, (u32::MAX, f32::MAX));
    }

    // ───── Fused tile loop: GEMM row-tile → dist convert → top-k ─────
    // dot_tile reuses bufs.dot_matrix prefix (capacity n*n ≥ mr*n).
    // Avoid borrow conflict by capturing raw pointers to disjoint bufs fields.
    if actual_k > 0 {
        let norms_sq_ptr = bufs.norms_sq.as_ptr();
        let cosine_denoms_ptr = bufs.cosine_denoms.as_ptr();
        let mut tile_start = 0usize;
        while tile_start < n {
            let mb = (n - tile_start).min(mr);
            let tile_len = mb * n;

            // 1. GEMM: dot_tile = A[tile_rows] · A^T (mb × n)
            // Raw-pointer slices needed because a_full and dot_tile borrow disjoint
            // fields of bufs but the borrow checker can't see that through indexing.
            {
                let a_full = &bufs.local_data[..n * ndims];
                let a_tile_ptr = a_full[tile_start * ndims..].as_ptr();
                let a_full_ptr = a_full.as_ptr();
                let a_full_len = a_full.len();
                let dot_tile = &mut bufs.dot_matrix[..tile_len];
                // SAFETY: a_tile_ptr/a_full_ptr come from the live `&bufs.local_data`
                // borrow above; lengths `mb*ndims` and `a_full_len` match the slice
                // they were taken from. Disjoint from `dot_tile` (different field).
                let a_tile_slice = unsafe { std::slice::from_raw_parts(a_tile_ptr, mb * ndims) };
                // SAFETY: see preceding block — same source slice and lifetime.
                let a_full_slice = unsafe { std::slice::from_raw_parts(a_full_ptr, a_full_len) };
                diskann_linalg::sgemm_abt(a_tile_slice, mb, ndims, a_full_slice, n, dot_tile);
            }

            // 2 + 3. Convert dot row to dist inline + extract top-k per row.
            // Output: bufs.knn_result[(tile_start+local_i) * actual_k..]
            let dot_tile = &mut bufs.dot_matrix[..tile_len];
            for local_i in 0..mb {
                let global_i = tile_start + local_i;
                let row = &mut dot_tile[local_i * n..(local_i + 1) * n];

                // norms_sq_ptr and cosine_denoms_ptr point to bufs fields that
                // are disjoint from bufs.dot_matrix (which `row` borrows). All reads are
                // in-bounds (i < n, j < n). SIMD tier is now selected at runtime by
                // the convert_* dispatcher rather than at compile time.
                match metric {
                    Metric::CosineNormalized => {
                        convert_cosnorm(row, n);
                    }
                    Metric::L2 => {
                        // SAFETY: norms_sq_ptr points to bufs.norms_sq, sized `n`, and
                        // `global_i = tile_start + local_i < n`. The pointer remains
                        // valid throughout the loop because bufs is borrowed for the
                        // call.
                        let ni = unsafe { *norms_sq_ptr.add(global_i) };
                        // SAFETY: norms_sq_ptr spans `n` floats; convert_l2 reads up to
                        // index `n-1`.
                        unsafe { convert_l2(row, n, ni, norms_sq_ptr) };
                    }
                    Metric::Cosine => {
                        // SAFETY: cosine_denoms_ptr spans `n` floats, `global_i < n`.
                        let ni_sqrt = unsafe { *cosine_denoms_ptr.add(global_i) };
                        // Cosine path is rare in PiPNN. Keep scalar — the sqrt+div
                        // would dominate any SIMD savings anyway.
                        for (j, val) in row.iter_mut().enumerate() {
                            // SAFETY: same as above — `j < n` and ptr spans `n`.
                            let nj = unsafe { *cosine_denoms_ptr.add(j) };
                            let denom = ni_sqrt * nj;
                            let cos = if denom > 0.0 { *val / denom } else { 0.0 };
                            *val = (1.0 - cos).max(0.0);
                        }
                    }
                    Metric::InnerProduct => {
                        convert_ip(row, n);
                    }
                }

                if actual_k <= 3 {
                    let out = &mut bufs.knn_result[global_i * actual_k..(global_i + 1) * actual_k];
                    topk_row_small(row, n, global_i, actual_k, out);
                } else {
                    // k > 3 (rare in PiPNN — leaf_k is typically 2-3): per-row
                    // sort. Mark self as ineligible, then sort indices by distance.
                    row[global_i] = f32::MAX;
                    let mut idxs: Vec<u32> = (0..n as u32).collect();
                    idxs.sort_unstable_by(|&a, &b| {
                        row[a as usize]
                            .partial_cmp(&row[b as usize])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let out = &mut bufs.knn_result[global_i * actual_k..(global_i + 1) * actual_k];
                    for t in 0..actual_k {
                        let j = idxs[t] as usize;
                        out[t] = (j as u32, row[j]);
                    }
                }
            }

            tile_start += mb;
        }
    }

    // ───── Bidirected edges, CSR-grouped by local source ─────
    // All four PiPNN metrics yield symmetric distances; reverse-edge insertion
    // adds candidates that aren't otherwise covered by leaf overlap alone (drops
    // avg_degree from 62.2 → 49.8 if omitted).
    //
    // CSR layout: group_starts[i]..group_starts[i+1] indexes group_data for
    // edges with local source `i`. HP::add_edges_grouped locks each src's
    // reservoir once and processes all its inserts in cache — eliminates the
    // ~5x lock+cache cost of interleaved-source insertion.
    let seen = &mut bufs.seen[..n * n];
    seen.fill(false);

    // Pass 1: count edges per local src (with dedup).
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

    // Pass 2: write per-src using a per-row cursor (Vec reused via group_data_cursor).
    // Local var: clone of group_starts as cursor.
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
    // `PIPNN_LEAF_V2=1` routes through the triangular-GEMM path
    // (single sgemm_aat_lower call + symmetrize) instead of the row-tile
    // fused path. Same output, ~50% fewer FLOPs in the GEMM phase.
    if std::env::var("PIPNN_LEAF_V2").as_deref() == Ok("1") {
        let _edges = build_leaf_v2_with_buffers(data, ndims, indices, k, metric, bufs);
    } else {
        let _edges = build_leaf_with_buffers(data, ndims, indices, k, metric, bufs);
    }
    bufs.group_starts[indices.len()] as usize
}

/// Symmetrize the lower triangle of a `n × n` row-major matrix into the upper
/// triangle. After this, `dot[i*n + j] == dot[j*n + i]` for all i, j.
/// Diagonal is left untouched (already correct from the GEMM).
///
/// AVX-2 8×8 block transpose: for each strictly-lower 8×8 block, transpose
/// in-register (8 loads + 24 shuffles + 8 stores) and store to mirrored upper
/// position. The 8×8 shuffle recipe is the standard one from Intel's optim
/// guide. Diagonal blocks fall back to scalar (small count: only n/8).
#[inline]
fn symmetrize_lower_to_upper(dot: &mut [f32], n: usize) {
    debug_assert!(dot.len() >= n * n);
    match tier() {
        #[cfg(target_arch = "x86_64")]
        SimdTier::Avx2 | SimdTier::Avx512 => unsafe { symmetrize_avx2_block8(dot, n) },
        _ => symmetrize_scalar(dot, n),
    }
}

fn symmetrize_scalar(dot: &mut [f32], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            unsafe {
                let src = *dot.get_unchecked(j * n + i);
                *dot.get_unchecked_mut(i * n + j) = src;
            }
        }
    }
}

/// 8×8 in-register transpose using AVX-2 + permute2f128. Standard recipe from
/// Intel optim guide. Reads block at lower (br, bc) — row range `br..br+8`,
/// col range `bc..bc+8` — and writes the transposed block to upper position
/// (bc, br).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn symmetrize_avx2_block8(dot: &mut [f32], n: usize) {
    use std::arch::x86_64::*;
    const B: usize = 8;
    let nblocks = n / B;
    let p = dot.as_mut_ptr();

    // Strictly-lower 8×8 blocks: (br, bc) with bc < br.
    for br in 0..nblocks {
        for bc in 0..br {
            let row_off = br * B;
            let col_off = bc * B;
            // Load 8 rows of 8 cols from lower block.
            let r0 = _mm256_loadu_ps(p.add((row_off + 0) * n + col_off));
            let r1 = _mm256_loadu_ps(p.add((row_off + 1) * n + col_off));
            let r2 = _mm256_loadu_ps(p.add((row_off + 2) * n + col_off));
            let r3 = _mm256_loadu_ps(p.add((row_off + 3) * n + col_off));
            let r4 = _mm256_loadu_ps(p.add((row_off + 4) * n + col_off));
            let r5 = _mm256_loadu_ps(p.add((row_off + 5) * n + col_off));
            let r6 = _mm256_loadu_ps(p.add((row_off + 6) * n + col_off));
            let r7 = _mm256_loadu_ps(p.add((row_off + 7) * n + col_off));
            // Stage 1: pairwise unpack within 128-bit lanes.
            let t0 = _mm256_unpacklo_ps(r0, r1);
            let t1 = _mm256_unpackhi_ps(r0, r1);
            let t2 = _mm256_unpacklo_ps(r2, r3);
            let t3 = _mm256_unpackhi_ps(r2, r3);
            let t4 = _mm256_unpacklo_ps(r4, r5);
            let t5 = _mm256_unpackhi_ps(r4, r5);
            let t6 = _mm256_unpacklo_ps(r6, r7);
            let t7 = _mm256_unpackhi_ps(r6, r7);
            // Stage 2: 64-bit-pair shuffle within 128-bit lanes.
            let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
            let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
            let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
            let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
            let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
            let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
            let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
            let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
            // Stage 3: 128-bit lane permute.
            let q0 = _mm256_permute2f128_ps::<0x20>(s0, s4);
            let q1 = _mm256_permute2f128_ps::<0x20>(s1, s5);
            let q2 = _mm256_permute2f128_ps::<0x20>(s2, s6);
            let q3 = _mm256_permute2f128_ps::<0x20>(s3, s7);
            let q4 = _mm256_permute2f128_ps::<0x31>(s0, s4);
            let q5 = _mm256_permute2f128_ps::<0x31>(s1, s5);
            let q6 = _mm256_permute2f128_ps::<0x31>(s2, s6);
            let q7 = _mm256_permute2f128_ps::<0x31>(s3, s7);
            // Store transposed to upper position (col_off, row_off).
            _mm256_storeu_ps(p.add((col_off + 0) * n + row_off), q0);
            _mm256_storeu_ps(p.add((col_off + 1) * n + row_off), q1);
            _mm256_storeu_ps(p.add((col_off + 2) * n + row_off), q2);
            _mm256_storeu_ps(p.add((col_off + 3) * n + row_off), q3);
            _mm256_storeu_ps(p.add((col_off + 4) * n + row_off), q4);
            _mm256_storeu_ps(p.add((col_off + 5) * n + row_off), q5);
            _mm256_storeu_ps(p.add((col_off + 6) * n + row_off), q6);
            _mm256_storeu_ps(p.add((col_off + 7) * n + row_off), q7);
        }
    }
    // Diagonal blocks (br == bc): scalar within block (only nblocks of them,
    // ~28 ops each at B=8 vs 256 in a transpose; not worth SIMD).
    for b in 0..nblocks {
        let base = b * B;
        for i in 0..B {
            for j in (i + 1)..B {
                let row = base + i;
                let col = base + j;
                *dot.get_unchecked_mut(row * n + col) =
                    *dot.get_unchecked(col * n + row);
            }
        }
    }
    // Tail (rows/cols >= nblocks * B): scalar fallback.
    let tail = nblocks * B;
    for i in 0..n {
        let j_start = if i >= tail { i + 1 } else { tail };
        for j in j_start..n {
            *dot.get_unchecked_mut(i * n + j) = *dot.get_unchecked(j * n + i);
        }
    }
}

/// 16×16 block transpose using AVX-512. Mirrors strictly-lower blocks to the
/// strictly-upper diagonal-mirror position. Per block: 16 loads + transpose
/// (~20 shuffles) + 16 stores = ~52 vector ops for 256 elements = ~5x
/// throughput vs scalar loop.

/// v2 build: single triangular GEMM over the full leaf (no tile loop),
/// symmetrize the lower triangle into the upper, then per-row convert + topk.
///
/// Compared to [`build_leaf_with_buffers`]:
/// - One `sgemm_aat_lower` call instead of `ceil(n/MR)` `sgemm_abt` tiles.
/// - ~50% fewer FMAs in the GEMM phase (faer's `triangular::matmul` skips
///   upper-triangle FMAs at the microkernel-dispatch layer).
/// - +N²/2 memory writes from the symmetrize pass (L2-resident at n ≤ 1024).
/// - Lose the tiled fused convert+topk; instead convert+topk run over the
///   full `n × n` matrix per-row. At n ≤ 1024, dot_matrix fits L2 and the
///   per-row stride is L1-friendly anyway.
pub(crate) fn build_leaf_v2_with_buffers<T: VectorRepr + 'static>(
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

        // ───── Step 2: extract norms from the SYRK diagonal ─────
        let needs_norms = matches!(metric, Metric::L2 | Metric::Cosine);
        if needs_norms {
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
            }
        }

        // ───── Step 3 (v3 fused path for L2 + k=3) ─────
        // PIPNN_FUSED=1 enables the dual-end fused scan: walks strictly-lower
        // triangle once, updates both row and column top-k trackers in one
        // pass. No upper-triangle materialisation, no separate convert pass.
        let fused_eligible = matches!(metric, Metric::L2) && actual_k == 3
            && std::env::var("PIPNN_FUSED").as_deref() == Ok("1")
            && matches!(tier(), SimdTier::Avx512);
        if fused_eligible {
            unsafe {
                fused_dual_topk_l2_avx512(
                    &bufs.dot_matrix[..n * n],
                    &bufs.norms_sq[..n],
                    &mut bufs.knn_result[..n * actual_k],
                    &mut bufs.worst[..n + 16],
                    n,
                    actual_k,
                );
            }
            // CSR build runs after; skip the symmetrize + convert + topk block.
            // Fall through to the post-topk CSR build (shared with v2).
        } else {
        // ───── Step 3: symmetrize lower → upper ─────
        // PIPNN_NO_SYMM=1 skips this for cost-attribution. Output is invalid
        // (upper triangle holds garbage) but lets us measure symmetrize wall.
        if std::env::var("PIPNN_NO_SYMM").as_deref() != Ok("1") {
            symmetrize_lower_to_upper(&mut bufs.dot_matrix[..n * n], n);
        }

        // ───── Step 4: per-row distance convert + top-k ─────
        let norms_sq_ptr = bufs.norms_sq.as_ptr();
        let cosine_denoms_ptr = bufs.cosine_denoms.as_ptr();
        for global_i in 0..n {
            let row = &mut bufs.dot_matrix[global_i * n..(global_i + 1) * n];

            match metric {
                Metric::CosineNormalized => convert_cosnorm(row, n),
                Metric::L2 => {
                    // SAFETY: norms_sq_ptr spans n floats; global_i < n.
                    let ni = unsafe { *norms_sq_ptr.add(global_i) };
                    unsafe { convert_l2(row, n, ni, norms_sq_ptr) };
                }
                Metric::Cosine => {
                    let ni_sqrt = unsafe { *cosine_denoms_ptr.add(global_i) };
                    for (j, val) in row.iter_mut().enumerate() {
                        let nj = unsafe { *cosine_denoms_ptr.add(j) };
                        let denom = ni_sqrt * nj;
                        let cos = if denom > 0.0 { *val / denom } else { 0.0 };
                        *val = (1.0 - cos).max(0.0);
                    }
                }
                Metric::InnerProduct => convert_ip(row, n),
            }

            if actual_k <= 3 {
                let out = &mut bufs.knn_result[global_i * actual_k..(global_i + 1) * actual_k];
                topk_row_small(row, n, global_i, actual_k, out);
            } else {
                row[global_i] = f32::MAX;
                let mut idxs: Vec<u32> = (0..n as u32).collect();
                idxs.sort_unstable_by(|&a, &b| {
                    row[a as usize]
                        .partial_cmp(&row[b as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let out = &mut bufs.knn_result[global_i * actual_k..(global_i + 1) * actual_k];
                for t in 0..actual_k {
                    let j = idxs[t] as usize;
                    out[t] = (j as u32, row[j]);
                }
            }
        }
        } // close `else` of `if fused_eligible`
    }

    // knn_result was allocated as a Vec but only up to `n * actual_k` entries
    // are populated. Pre-size it via clear+resize to match v1/v2 semantics.
    // (The fused path writes via get_unchecked_mut, so the Vec must already
    // be sized correctly. ensure_capacity reserves capacity but not length.)
    // Note: build_leaf_v2_with_buffers below does .clear() + .resize() above,
    // so this is just a sanity comment.

    // ───── CSR build: identical to v1 (bidirected edges, dedup) ─────
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
