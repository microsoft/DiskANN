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
use diskann_vector::distance::SquaredL2;
use diskann_vector::PureDistanceFunction;

/// Thread-local reusable buffers for leaf building.
/// Avoids repeated allocation/deallocation of large matrices.
pub struct LeafBuffers {
    pub local_data: Vec<f32>,
    pub norms_sq: Vec<f32>,
    pub dot_matrix: Vec<f32>,
    pub dist_matrix: Vec<f32>,
    pub seen: Vec<bool>,
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
            dist_matrix: Vec::new(),
            seen: Vec::new(),
        }
    }

    /// Ensure all buffers are large enough for a leaf of size n x ndims.
    fn ensure_capacity(&mut self, n: usize, ndims: usize) {
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
        if self.dist_matrix.len() < nn {
            self.dist_matrix.resize(nn, 0.0);
        }
        if self.seen.len() < nn {
            self.seen.resize(nn, false);
        }
    }
}

/// Thread-local reusable buffers for quantized leaf building.
struct QuantLeafBuffers {
    local_u64: Vec<u64>,
    dist_matrix: Vec<f32>,
    seen: Vec<bool>,
}

impl QuantLeafBuffers {
    fn new() -> Self {
        Self {
            local_u64: Vec::new(),
            dist_matrix: Vec::new(),
            seen: Vec::new(),
        }
    }
}

thread_local! {
    static LEAF_BUFFERS: RefCell<LeafBuffers> = RefCell::new(LeafBuffers::new());
    static QUANT_BUFFERS: RefCell<QuantLeafBuffers> = RefCell::new(QuantLeafBuffers::new());
}

/// Release thread-local leaf build buffers on the calling thread.
///
/// After leaf building is complete, these buffers pin pages in glibc's
/// per-thread arenas, preventing `malloc_trim` from returning freed
/// reservoir memory to the OS. Calling this from each rayon thread
/// allows the arena heaps to be reclaimed.
pub fn release_thread_buffers() {
    LEAF_BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        bufs.local_data = Vec::new();
        bufs.norms_sq = Vec::new();
        bufs.dot_matrix = Vec::new();
        bufs.dist_matrix = Vec::new();
        bufs.seen = Vec::new();
    });
    QUANT_BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        bufs.local_u64 = Vec::new();
        bufs.dist_matrix = Vec::new();
        bufs.seen = Vec::new();
    });
}

/// An edge produced by leaf building: (source, destination, distance).
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
    pub distance: f32,
}

/// Extract k nearest neighbors for each point from the distance matrix.
///
/// Uses index-sort: partitions a u32 index array by indirect distance comparison.
/// Sorting 4-byte indices instead of 8-byte (index, distance) pairs reduces memory
/// movement during quickselect, yielding ~1.5x speedup over the pair-based approach.
fn extract_knn(dist_matrix: &[f32], n: usize, k: usize) -> Vec<(usize, usize, f32)> {
    if n <= 1 || k == 0 {
        return Vec::new();
    }
    let actual_k = k.min(n - 1);
    let mut edges = Vec::with_capacity(n * actual_k);

    // Reuse index buffer across all rows (4 bytes per element vs 8 for pairs).
    let mut indices: Vec<u32> = (0..n as u32).collect();

    for i in 0..n {
        let row = &dist_matrix[i * n..(i + 1) * n];

        // Reset indices for this row.
        for j in 0..n {
            // SAFETY: `j` is in 0..n and `indices` has length n, so the access is in bounds.
            unsafe {
                *indices.get_unchecked_mut(j) = j as u32;
            }
        }

        if actual_k < n {
            indices.select_nth_unstable_by(actual_k - 1, |&a, &b| {
                // SAFETY: `a` originates from the `indices` array which contains
                // values in 0..n, and `row` has length n, so the access is in bounds.
                let da = unsafe { *row.get_unchecked(a as usize) };
                // SAFETY: `b` originates from the `indices` array which contains
                // values in 0..n, and `row` has length n, so the access is in bounds.
                let db = unsafe { *row.get_unchecked(b as usize) };
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        for idx in 0..actual_k {
            // SAFETY: `idx` is in 0..actual_k where actual_k <= n, and `indices`
            // has length n, so the access is in bounds.
            let j = unsafe { *indices.get_unchecked(idx) } as usize;
            edges.push((i, j, row[j]));
        }
    }

    edges
}

/// Build a leaf partition: compute all-pairs distances and extract bi-directed k-NN edges.
///
/// Returns edges as (global_src, global_dst, distance).
pub fn build_leaf<T: VectorRepr>(
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

fn build_leaf_with_buffers<T: VectorRepr>(
    data: &[T],
    ndims: usize,
    indices: &[usize],
    k: usize,
    metric: diskann_vector::distance::Metric,
    bufs: &mut LeafBuffers,
) -> Vec<Edge> {
    let n = indices.len();
    bufs.ensure_capacity(n, ndims);

    // Extract local data into reused buffer, converting T -> f32 on the fly.
    let local_data = &mut bufs.local_data[..n * ndims];
    for (i, &idx) in indices.iter().enumerate() {
        let src = &data[idx * ndims..(idx + 1) * ndims];
        let dst = &mut local_data[i * ndims..(i + 1) * ndims];
        T::as_f32_into(src, dst).expect("f32 conversion");
    }

    // Compute norms into reused buffer.
    let norms_sq = &mut bufs.norms_sq[..n];
    for i in 0..n {
        let row = &local_data[i * ndims..(i + 1) * ndims];
        let mut norm = 0.0f32;
        for &v in row.iter() {
            norm += v * v;
        }
        norms_sq[i] = norm;
    }

    // GEMM: dots = local_data * local_data^T
    let dot_matrix = &mut bufs.dot_matrix[..n * n];
    crate::gemm::sgemm_aat(local_data, n, ndims, dot_matrix);

    let norms_sq = &bufs.norms_sq[..n];

    // Convert to distance matrix using the target metric.
    use diskann_vector::distance::Metric;
    let dist_matrix = match metric {
        Metric::CosineNormalized => {
            // Pre-normalized: dist = 1 - dot(a, b)
            for i in 0..n {
                let row = &mut dot_matrix[i * n..(i + 1) * n];
                for val in row.iter_mut() {
                    *val = (1.0 - *val).max(0.0);
                }
                row[i] = f32::MAX;
            }
            &mut bufs.dot_matrix[..n * n]
        }
        Metric::Cosine => {
            // Unnormalized: dist = 1 - dot(a,b)/(||a||*||b||)
            let dist = &mut bufs.dist_matrix[..n * n];
            for i in 0..n {
                let ni_sqrt = norms_sq[i].sqrt();
                for j in 0..n {
                    let denom = ni_sqrt * norms_sq[j].sqrt();
                    let cos_sim = if denom > 0.0 {
                        dot_matrix[i * n + j] / denom
                    } else {
                        0.0
                    };
                    dist[i * n + j] = (1.0 - cos_sim).max(0.0);
                }
                dist[i * n + i] = f32::MAX;
            }
            dist
        }
        Metric::L2 => {
            let dist = &mut bufs.dist_matrix[..n * n];
            for i in 0..n {
                let ni = norms_sq[i];
                for j in 0..n {
                    dist[i * n + j] = (ni + norms_sq[j] - 2.0 * dot_matrix[i * n + j]).max(0.0);
                }
                dist[i * n + i] = f32::MAX;
            }
            dist
        }
        Metric::InnerProduct => {
            for i in 0..n {
                let row = &mut dot_matrix[i * n..(i + 1) * n];
                for val in row.iter_mut() {
                    *val = -*val;
                }
                row[i] = f32::MAX;
            }
            &mut bufs.dot_matrix[..n * n]
        }
    };

    let local_edges = extract_knn(dist_matrix, n, k);
    let seen = &mut bufs.seen[..n * n];
    seen.fill(false);
    make_bidirected_edges(&local_edges, dist_matrix, n, indices, seen)
}

/// Build a leaf using 1-bit quantized vectors with Hamming distance.
pub fn build_leaf_quantized(
    qdata: &crate::quantize::QuantizedData,
    indices: &[usize],
    k: usize,
) -> Vec<Edge> {
    let n = indices.len();
    if n <= 1 {
        return Vec::new();
    }

    QUANT_BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        let u64s = qdata.u64s_per_vec();
        let nn = n * n;

        // Ensure buffers are large enough, reusing across leaves.
        if bufs.local_u64.len() < n * u64s {
            bufs.local_u64.resize(n * u64s, 0);
        }
        if bufs.dist_matrix.len() < nn {
            bufs.dist_matrix.resize(nn, 0.0);
        }
        if bufs.seen.len() < nn {
            bufs.seen.resize(nn, false);
        }

        // Destructure for simultaneous mutable borrows.
        let QuantLeafBuffers {
            local_u64,
            dist_matrix,
            seen,
        } = &mut *bufs;

        // Gather contiguous u64 data.
        let local = &mut local_u64[..n * u64s];
        for (i, &idx) in indices.iter().enumerate() {
            local[i * u64s..(i + 1) * u64s].copy_from_slice(qdata.get_u64(idx));
        }

        // Compute all-pairs Hamming distance in-place.
        let dist = &mut dist_matrix[..nn];
        let local_ptr = local.as_ptr();
        let dist_ptr = dist.as_mut_ptr();
        for i in 0..n {
            // SAFETY: `i` is in 0..n, so `i * n + i` is within the n*n-element dist buffer.
            unsafe {
                *dist_ptr.add(i * n + i) = f32::MAX;
            }
            // SAFETY: `i * u64s` is within the `n * u64s`-element local buffer.
            let a_base = unsafe { local_ptr.add(i * u64s) };
            for j in (i + 1)..n {
                // SAFETY: `j * u64s` is within the `n * u64s`-element local buffer.
                let b_base = unsafe { local_ptr.add(j * u64s) };
                let mut h = 0u32;
                for k_idx in 0..u64s {
                    // SAFETY: `k_idx` is in 0..u64s, so `a_base.add(k_idx)` and
                    // `b_base.add(k_idx)` are within their respective vector slices.
                    unsafe {
                        h += (*a_base.add(k_idx) ^ *b_base.add(k_idx)).count_ones();
                    }
                }
                let d = h as f32;
                // SAFETY: `i < j < n`, so both `i * n + j` and `j * n + i` are
                // within the n*n-element dist buffer.
                unsafe {
                    *dist_ptr.add(i * n + j) = d;
                    *dist_ptr.add(j * n + i) = d;
                }
            }
        }

        let local_edges = extract_knn(dist, n, k);
        let seen = &mut seen[..nn];
        seen.fill(false);
        make_bidirected_edges(&local_edges, dist, n, indices, seen)
    })
}

/// Convert k-NN edges to bi-directed global edges, deduplicating via a seen buffer.
/// For symmetric metrics, dist(a,b) == dist(b,a) but we use the matrix lookup
/// for the reverse edge to stay correct for any future asymmetric metric.
fn make_bidirected_edges(
    local_edges: &[(usize, usize, f32)],
    dist_matrix: &[f32],
    n: usize,
    indices: &[usize],
    seen: &mut [bool],
) -> Vec<Edge> {
    let mut global_edges = Vec::with_capacity(local_edges.len() * 2);
    for &(src, dst, dist) in local_edges {
        if !seen[src * n + dst] {
            seen[src * n + dst] = true;
            global_edges.push(Edge {
                src: indices[src],
                dst: indices[dst],
                distance: dist,
            });
        }
        if !seen[dst * n + src] {
            seen[dst * n + src] = true;
            global_edges.push(Edge {
                src: indices[dst],
                dst: indices[src],
                distance: dist_matrix[dst * n + src],
            });
        }
    }
    global_edges
}

/// Brute-force search the dataset using L2 distance.
///
/// Returns the `k` nearest neighbor indices and distances for the query.
pub fn brute_force_knn(
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
    fn test_extract_knn() {
        let dist = vec![f32::MAX, 1.0, 4.0, 1.0, f32::MAX, 1.0, 4.0, 1.0, f32::MAX];
        let edges = extract_knn(&dist, 3, 1);

        assert_eq!(edges.len(), 3);

        let p0_edges: Vec<_> = edges.iter().filter(|e| e.0 == 0).collect();
        assert_eq!(p0_edges.len(), 1);
        assert_eq!(p0_edges[0].1, 1);

        let p2_edges: Vec<_> = edges.iter().filter(|e| e.0 == 2).collect();
        assert_eq!(p2_edges.len(), 1);
        assert_eq!(p2_edges[0].1, 1);
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
    fn test_build_leaf_quantized() {
        // Build a leaf using quantized data and verify basic correctness.
        let ndims = 64;
        let npoints = 10;
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..npoints * ndims)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();

        let (shift, inverse_scale) = {
            use diskann_quantization::scalar::train::ScalarQuantizationParameters;
            use diskann_utils::views::MatrixView;
            let dm = MatrixView::try_from(data.as_slice(), npoints, ndims).unwrap();
            let q = ScalarQuantizationParameters::default().train(dm);
            let s = q.scale();
            (q.shift().to_vec(), if s == 0.0 { 1.0 } else { 1.0 / s })
        };
        let qdata = crate::quantize::quantize_1bit(&data, npoints, ndims, &shift, inverse_scale);
        let indices: Vec<usize> = (0..npoints).collect();
        let edges = build_leaf_quantized(&qdata, &indices, 3);

        assert!(!edges.is_empty(), "quantized leaf should produce edges");

        for edge in &edges {
            assert!(edge.src < npoints, "src {} out of range", edge.src);
            assert!(edge.dst < npoints, "dst {} out of range", edge.dst);
            assert_ne!(edge.src, edge.dst);
            assert!(edge.distance >= 0.0);
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
    fn test_extract_knn_k_larger_than_n() {
        // k > n-1 should be clamped.
        let dist = vec![f32::MAX, 1.0, 1.0, f32::MAX];
        let edges = extract_knn(&dist, 2, 100); // k=100 but only 2 points
        assert_eq!(
            edges.len(),
            2,
            "k > n-1 should be clamped, each point gets 1 neighbor, total 2 edges"
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
    fn test_extract_knn_k_zero() {
        let dist = vec![f32::MAX, 1.0, 1.0, f32::MAX];
        let edges = extract_knn(&dist, 2, 0);
        assert!(edges.is_empty(), "k=0 should return no edges");
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
