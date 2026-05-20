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
    });
}

/// An edge produced by leaf building: `(source, destination, distance)`.
///
/// Returned by [`build_leaf`] for benchmarking. Production code uses
/// [`build_leaf_into`] which writes edges into a caller-provided buffer.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    /// Global source point index.
    pub src: usize,
    /// Global destination point index.
    pub dst: usize,
    /// Distance from `src` to `dst` under the build metric.
    pub distance: f32,
}


/// Build a leaf partition: compute all-pairs distances and extract bi-directed k-NN edges.
///
/// Returns edges as (global_src, global_dst, distance).
pub fn build_leaf<T: VectorRepr + 'static>(
    data: &[T],
    ndims: usize,
    indices: &[usize],
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

/// Fused per-row top-k for k<=3, scanning AVX-512 16-wide over a dist row.
/// Writes `actual_k` entries into `out[..actual_k]`.
#[inline]
fn topk_row_small(
    dist_row: &[f32],
    n: usize,
    self_idx: usize,
    actual_k: usize,
    out: &mut [(u32, f32)],
) {
    debug_assert!(actual_k <= 3);
    let mut top: [(u32, f32); 3] = [(u32::MAX, f32::MAX); 3];
    let threshold_idx = actual_k - 1;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        use std::arch::x86_64::*;
        let chunks = n / 16;
        unsafe {
            for chunk in 0..chunks {
                let base = chunk * 16;
                let thresh = _mm512_set1_ps(top[threshold_idx].1);
                let dists = _mm512_loadu_ps(dist_row.as_ptr().add(base));
                let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(dists, thresh);
                if mask != 0 {
                    let mut d_arr = [0.0f32; 16];
                    _mm512_storeu_ps(d_arr.as_mut_ptr(), dists);
                    let mut m = mask;
                    while m != 0 {
                        let lane = m.trailing_zeros() as usize;
                        m &= m - 1;
                        let j = base + lane;
                        if j == self_idx {
                            continue;
                        }
                        let d = d_arr[lane];
                        if d < top[threshold_idx].1 {
                            top[threshold_idx] = (j as u32, d);
                            for t in (1..actual_k).rev() {
                                if top[t].1 < top[t - 1].1 {
                                    top.swap(t, t - 1);
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            for j in (chunks * 16)..n {
                if j == self_idx {
                    continue;
                }
                let d = *dist_row.get_unchecked(j);
                if d < top[threshold_idx].1 {
                    top[threshold_idx] = (j as u32, d);
                    for t in (1..actual_k).rev() {
                        if top[t].1 < top[t - 1].1 {
                            top.swap(t, t - 1);
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        for (j, &d) in dist_row.iter().enumerate().take(n) {
            if j == self_idx {
                continue;
            }
            if d < top[threshold_idx].1 {
                top[threshold_idx] = (j as u32, d);
                for t in (1..actual_k).rev() {
                    if top[t].1 < top[t - 1].1 {
                        top.swap(t, t - 1);
                    } else {
                        break;
                    }
                }
            }
        }
    }

    out[..actual_k].copy_from_slice(&top[..actual_k]);
}

/// Build a leaf using caller-provided buffers, bypassing thread-local access.
///
/// Tiles the leaf GEMM in MR-row chunks (≤ 2 MB dot buffer → L2-resident),
/// converts to distance and extracts top-k inline. Never materializes the
/// full m×m dist matrix.
pub(crate) fn build_leaf_with_buffers<T: VectorRepr + 'static>(
    data: &[T],
    ndims: usize,
    indices: &[usize],
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
            let src = &data[idx * ndims..(idx + 1) * ndims];
            let dst = &mut local_data[i * ndims..(i + 1) * ndims];
            T::as_f32_into(src, dst).expect("f32 conversion");
        }
        if needs_norms {
            let norms_sq = &mut bufs.norms_sq[..n];
            for i in 0..n {
                let row = &local_data[i * ndims..(i + 1) * ndims];
                let mut s = 0.0f32;
                for &v in row.iter() {
                    s += v * v;
                }
                norms_sq[i] = s;
            }
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
                crate::gemm::sgemm_abt(a_tile_slice, mb, ndims, a_full_slice, n, dot_tile);
            }

            // 2 + 3. Convert dot row to dist inline + extract top-k per row.
            // Output: bufs.knn_result[(tile_start+local_i) * actual_k..]
            let dot_tile = &mut bufs.dot_matrix[..tile_len];
            for local_i in 0..mb {
                let global_i = tile_start + local_i;
                let row = &mut dot_tile[local_i * n..(local_i + 1) * n];

                // SAFETY: norms_sq_ptr and cosine_denoms_ptr point to bufs fields that
                // are disjoint from bufs.dot_matrix (which `row` borrows). All reads are
                // in-bounds (i < n, j < n).
                #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                let simd_dist = true;
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
                let simd_dist = false;

                match metric {
                    Metric::CosineNormalized => {
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                        if simd_dist {
                            use std::arch::x86_64::*;
                            unsafe {
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
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
                        for val in row.iter_mut() {
                            *val = (1.0 - *val).max(0.0);
                        }
                        let _ = simd_dist;
                    }
                    Metric::L2 => {
                        // SAFETY: norms_sq_ptr points to bufs.norms_sq, sized `n`, and
                        // `global_i = tile_start + local_i < n`.
                        let ni = unsafe { *norms_sq_ptr.add(global_i) };
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                        if simd_dist {
                            use std::arch::x86_64::*;
                            // SAFETY: AVX-512 cfg-gated. Pointer arithmetic stays in
                            // bounds: chunks = n/16, base = c*16, so base+16 ≤ n; row
                            // and norms_sq_ptr both span ≥ n floats.
                            unsafe {
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
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
                        for (j, val) in row.iter_mut().enumerate() {
                            // SAFETY: norms_sq_ptr spans `n` floats and `j < n`.
                            let nj = unsafe { *norms_sq_ptr.add(j) };
                            *val = (ni + nj - 2.0 * *val).max(0.0);
                        }
                        let _ = simd_dist;
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
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                        if simd_dist {
                            use std::arch::x86_64::*;
                            unsafe {
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
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
                        for val in row.iter_mut() {
                            *val = -*val;
                        }
                        let _ = simd_dist;
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

    // ───── Bidirected edges (symmetric metric: rev_dist = dist) ─────
    // All four PiPNN metrics yield symmetric distances:
    //   - L2: ||a-b||^2 = ||b-a||^2
    //   - Cosine / CosineNormalized: cos(a,b) = cos(b,a)
    //   - InnerProduct: dot(a,b) = dot(b,a)
    // So we can skip the dist_matrix[dst*n+i] lookup and reuse the forward `dist`.
    let seen = &mut bufs.seen[..n * n];
    seen.fill(false);
    bufs.edges.clear();
    bufs.edges.reserve(n * actual_k.max(1) * 2);
    for i in 0..n {
        let row_knn = &bufs.knn_result[i * actual_k..(i + 1) * actual_k];
        for &(dst_local, dist) in row_knn {
            if dst_local == u32::MAX {
                continue;
            }
            let dst = dst_local as usize;
            if !seen[i * n + dst] {
                seen[i * n + dst] = true;
                bufs.edges.push(Edge {
                    src: indices[i],
                    dst: indices[dst],
                    distance: dist,
                });
            }
            if !seen[dst * n + i] {
                seen[dst * n + i] = true;
                bufs.edges.push(Edge {
                    src: indices[dst],
                    dst: indices[i],
                    distance: dist,
                });
            }
        }
    }

    std::mem::take(&mut bufs.edges)
}

/// Build a leaf into `bufs.edges` without allocating. Returns edge count.
/// The caller reads edges from `bufs.edges[..returned_count]`.
pub(crate) fn build_leaf_into<T: VectorRepr + 'static>(
    data: &[T],
    ndims: usize,
    indices: &[usize],
    k: usize,
    metric: diskann_vector::distance::Metric,
    bufs: &mut LeafBuffers,
) -> usize {
    let edges = build_leaf_with_buffers(data, ndims, indices, k, metric, bufs);
    let count = edges.len();
    // Put the Vec back into bufs for capacity reuse on next call.
    bufs.edges = edges;
    count
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
        crate::gemm::sgemm_aat(&a, 2, 3, &mut result);

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
        let edge_set: std::collections::HashSet<(usize, usize)> =
            edges.iter().map(|e| (e.src, e.dst)).collect();

        // Every pair (i, j) with i != j should be present.
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(
                        edge_set.contains(&(i, j)),
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
        let edge_set: std::collections::HashSet<(usize, usize)> =
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
        let edge_set: std::collections::HashSet<(usize, usize)> =
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
        let indices_large: Vec<usize> = (0..20).collect();
        let edges1 = build_leaf(&data_large, 4, &indices_large, 2, Metric::L2);
        assert!(!edges1.is_empty());

        // Second call with smaller leaf on same thread — should reuse thread-local buffers.
        let data_small = vec![1.0f32; 4 * 4];
        let indices_small: Vec<usize> = (0..4).collect();
        let edges2 = build_leaf(&data_small, 4, &indices_small, 2, Metric::L2);
        assert!(
            !edges2.is_empty(),
            "small leaf after large should work with reused buffers"
        );
    }
}
