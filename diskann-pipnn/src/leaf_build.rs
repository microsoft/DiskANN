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

use diskann_vector::PureDistanceFunction;
use diskann_vector::distance::SquaredL2;

/// Thread-local reusable buffers for leaf building.
/// Avoids repeated allocation/deallocation of large matrices.
pub struct LeafBuffers {
    pub local_data: Vec<f32>,
    pub norms_sq: Vec<f32>,
    pub dot_matrix: Vec<f32>,
    pub dist_matrix: Vec<f32>,
    pub seen: Vec<bool>,
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
        if self.local_data.len() < nd { self.local_data.resize(nd, 0.0); }
        if self.norms_sq.len() < n { self.norms_sq.resize(n, 0.0); }
        if self.dot_matrix.len() < nn { self.dot_matrix.resize(nn, 0.0); }
        if self.dist_matrix.len() < nn { self.dist_matrix.resize(nn, 0.0); }
        if self.seen.len() < nn { self.seen.resize(nn, false); }
    }
}

thread_local! {
    static LEAF_BUFFERS: RefCell<LeafBuffers> = RefCell::new(LeafBuffers::new());
    static QUANT_SEEN: RefCell<Vec<bool>> = RefCell::new(Vec::new());
}

/// An edge produced by leaf building: (source, destination, distance).
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
    pub distance: f32,
}

/// Compute the all-pairs distance matrix for a set of points within a leaf.
///
/// `data` is the global data array (row-major, npoints_global x ndims).
/// `indices` are the global indices of points in this leaf.
/// `use_cosine`: if true, distance = 1 - dot(a,b) (for normalized vectors).
///
/// Returns a flat distance matrix of size n x n (row-major).
#[allow(dead_code)] // Alternative implementation kept for benchmarking/debugging.
fn compute_distance_matrix(data: &[f32], ndims: usize, indices: &[usize], use_cosine: bool) -> Vec<f32> {
    let n = indices.len();

    // Extract the local data for this leaf into contiguous memory.
    let mut local_data = vec![0.0f32; n * ndims];
    for (i, &idx) in indices.iter().enumerate() {
        local_data[i * ndims..(i + 1) * ndims]
            .copy_from_slice(&data[idx * ndims..(idx + 1) * ndims]);
    }

    // Compute squared norms.
    let mut norms_sq = vec![0.0f32; n];
    for i in 0..n {
        let row = &local_data[i * ndims..(i + 1) * ndims];
        let mut norm = 0.0f32;
        for &v in row {
            norm += v * v;
        }
        norms_sq[i] = norm;
    }

    // Compute dot product matrix: dot[i][j] = local_data[i] . local_data[j]
    // This is the GEMM: A * A^T where A is n x ndims.
    let mut dot_matrix = vec![0.0f32; n * n];
    gemm_aat(&local_data, n, ndims, &mut dot_matrix);

    // Compute distance matrix from dot products.
    let mut dist_matrix = vec![0.0f32; n * n];
    if use_cosine {
        // For normalized vectors: distance = 1 - dot(a,b)
        for i in 0..n {
            let dist_row = &mut dist_matrix[i * n..(i + 1) * n];
            let dot_row = &dot_matrix[i * n..(i + 1) * n];
            for j in 0..n {
                dist_row[j] = (1.0 - dot_row[j]).max(0.0);
            }
            dist_row[i] = f32::MAX;
        }
    } else {
        // L2: dist[i][j] = norms_sq[i] + norms_sq[j] - 2 * dot[i][j]
        for i in 0..n {
            let ni = norms_sq[i];
            let dist_row = &mut dist_matrix[i * n..(i + 1) * n];
            let dot_row = &dot_matrix[i * n..(i + 1) * n];
            for j in 0..n {
                let d = ni + norms_sq[j] - 2.0 * dot_row[j];
                dist_row[j] = d.max(0.0);
            }
            dist_row[i] = f32::MAX;
        }
    }

    dist_matrix
}

/// Direct pairwise distance computation for small leaves (avoids GEMM overhead).
#[allow(dead_code)] // Alternative implementation kept for benchmarking/debugging.
fn compute_distance_matrix_direct(data: &[f32], ndims: usize, indices: &[usize], use_cosine: bool) -> Vec<f32> {
    let n = indices.len();
    let mut dist_matrix = vec![f32::MAX; n * n];

    for i in 0..n {
        let a = &data[indices[i] * ndims..(indices[i] + 1) * ndims];
        for j in (i + 1)..n {
            let b = &data[indices[j] * ndims..(indices[j] + 1) * ndims];
            let d = if use_cosine {
                let mut dot = 0.0f32;
                for k in 0..ndims {
                    unsafe { dot += *a.get_unchecked(k) * *b.get_unchecked(k); }
                }
                (1.0 - dot).max(0.0)
            } else {
                let mut sum = 0.0f32;
                for k in 0..ndims {
                    let diff = unsafe { *a.get_unchecked(k) - *b.get_unchecked(k) };
                    sum += diff * diff;
                }
                sum
            };
            dist_matrix[i * n + j] = d;
            dist_matrix[j * n + i] = d;
        }
    }
    dist_matrix
}

/// Compute A * A^T using matrixmultiply for near-BLAS performance.
///
/// A is n x d (row-major), result is n x n (row-major).
#[allow(dead_code)] // Alternative implementation kept for benchmarking/debugging.
fn gemm_aat(a: &[f32], n: usize, d: usize, result: &mut [f32]) {
    debug_assert_eq!(a.len(), n * d);
    debug_assert_eq!(result.len(), n * n);
    result.fill(0.0);

    // Compute A * A^T. A^T has row stride 1, col stride d.
    unsafe {
        matrixmultiply::sgemm(
            n,     // m
            d,     // k
            n,     // n
            1.0,   // alpha
            a.as_ptr(),
            d as isize,  // row stride of A
            1,           // col stride of A
            a.as_ptr(),
            1,           // row stride of A^T
            d as isize,  // col stride of A^T
            0.0,   // beta
            result.as_mut_ptr(),
            n as isize,  // row stride of C
            1,           // col stride of C
        );
    }
}

/// Extract k nearest neighbors for each point from the distance matrix.
///
/// Uses partial sort (select_nth_unstable) for O(n) per point instead of full sort.
fn extract_knn(dist_matrix: &[f32], n: usize, k: usize) -> Vec<(usize, usize, f32)> {
    let actual_k = k.min(n - 1);
    let mut edges = Vec::with_capacity(n * actual_k);

    // Reuse buffer across all points to avoid n allocations.
    let mut dists: Vec<(u32, f32)> = Vec::with_capacity(n);

    for i in 0..n {
        let row = &dist_matrix[i * n..(i + 1) * n];

        dists.clear();
        for j in 0..n {
            dists.push((j as u32, unsafe { *row.get_unchecked(j) }));
        }

        if actual_k < dists.len() {
            dists.select_nth_unstable_by(actual_k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        for idx in 0..actual_k {
            let (j, dist) = unsafe { *dists.get_unchecked(idx) };
            edges.push((i, j as usize, dist));
        }
    }

    edges
}

/// Build a leaf partition: compute all-pairs distances and extract bi-directed k-NN edges.
///
/// Returns edges as (global_src, global_dst, distance).
pub fn build_leaf(
    data: &[f32],
    ndims: usize,
    indices: &[usize],
    k: usize,
    use_cosine: bool,
) -> Vec<Edge> {
    let n = indices.len();
    if n <= 1 {
        return Vec::new();
    }

    LEAF_BUFFERS.with(|cell| {
        let mut bufs = cell.borrow_mut();
        build_leaf_with_buffers(data, ndims, indices, k, use_cosine, &mut bufs)
    })
}

fn build_leaf_with_buffers(
    data: &[f32],
    ndims: usize,
    indices: &[usize],
    k: usize,
    use_cosine: bool,
    bufs: &mut LeafBuffers,
) -> Vec<Edge> {
    let n = indices.len();
    bufs.ensure_capacity(n, ndims);

    // Extract local data into reused buffer.
    let local_data = &mut bufs.local_data[..n * ndims];
    for (i, &idx) in indices.iter().enumerate() {
        local_data[i * ndims..(i + 1) * ndims]
            .copy_from_slice(&data[idx * ndims..(idx + 1) * ndims]);
    }

    // Compute norms into reused buffer.
    let norms_sq = &mut bufs.norms_sq[..n];
    for i in 0..n {
        let row = &local_data[i * ndims..(i + 1) * ndims];
        let mut norm = 0.0f32;
        for &v in row.iter() { norm += v * v; }
        norms_sq[i] = norm;
    }

    // GEMM: dots = local_data * local_data^T (using OpenBLAS)
    // sgemm with beta=0.0 zeroes the output — no explicit fill needed.
    let dot_matrix = &mut bufs.dot_matrix[..n * n];
    crate::gemm::sgemm_aat(local_data, n, ndims, dot_matrix);

    // Convert to distance matrix.
    // For cosine: convert in-place (each element only depends on itself).
    // For L2: need separate buffer since dist[i][j] depends on norms + dot[i][j].
    let dist_matrix = if use_cosine {
        // In-place: dist = 1 - dot
        for i in 0..n {
            let row = &mut dot_matrix[i * n..(i + 1) * n];
            for j in 0..n { row[j] = (1.0 - row[j]).max(0.0); }
            row[i] = f32::MAX;
        }
        &mut bufs.dot_matrix[..n * n] // dot_matrix IS now the dist_matrix
    } else {
        // L2: dist[i][j] = norms_sq[i] + norms_sq[j] - 2*dot[i][j]
        let dist = &mut bufs.dist_matrix[..n * n];
        for i in 0..n {
            let ni = norms_sq[i];
            for j in 0..n {
                dist[i * n + j] = (ni + norms_sq[j] - 2.0 * dot_matrix[i * n + j]).max(0.0);
            }
            dist[i * n + i] = f32::MAX;
        }
        dist
    };

    // Extract k-NN edges.
    let local_edges = extract_knn(dist_matrix, n, k);

    // Create bi-directed edges using reused seen buffer.
    let seen = &mut bufs.seen[..n * n];
    seen.fill(false);

    let mut global_edges = Vec::with_capacity(local_edges.len() * 2);

    for &(src, dst, dist) in &local_edges {
        if !seen[src * n + dst] {
            seen[src * n + dst] = true;
            global_edges.push(Edge { src: indices[src], dst: indices[dst], distance: dist });
        }
        if !seen[dst * n + src] {
            seen[dst * n + src] = true;
            global_edges.push(Edge { src: indices[dst], dst: indices[src], distance: dist_matrix[dst * n + src] });
        }
    }

    global_edges
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

    let dist_matrix = qdata.compute_distance_matrix(indices);
    let local_edges = extract_knn(&dist_matrix, n, k);

    QUANT_SEEN.with(|cell| {
        let mut seen = cell.borrow_mut();
        seen.resize(n * n, false);
        seen.fill(false);

        let mut global_edges = Vec::with_capacity(local_edges.len() * 2);

        for &(src, dst, dist) in &local_edges {
            if !seen[src * n + dst] {
                seen[src * n + dst] = true;
                global_edges.push(Edge { src: indices[src], dst: indices[dst], distance: dist });
            }
            if !seen[dst * n + src] {
                seen[dst * n + src] = true;
                global_edges.push(Edge { src: indices[dst], dst: indices[src], distance: dist_matrix[dst * n + src] });
            }
        }

        global_edges
    })
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
    if actual_k < dists.len() {
        dists.select_nth_unstable_by(actual_k, |a, b| {
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
        gemm_aat(&a, 2, 3, &mut result);

        assert!((result[0] - 14.0).abs() < 1e-6);
        assert!((result[1] - 32.0).abs() < 1e-6);
        assert!((result[2] - 32.0).abs() < 1e-6);
        assert!((result[3] - 77.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_matrix() {
        let data = vec![
            0.0, 0.0, // point 0
            1.0, 0.0, // point 1
            0.0, 1.0, // point 2
        ];
        let indices = vec![0, 1, 2];
        let dist = compute_distance_matrix(&data, 2, &indices, false);

        // Self-distances should be MAX (for k-NN).
        assert_eq!(dist[0], f32::MAX);
        // dist(0,1) = 1
        assert!((dist[1] - 1.0).abs() < 1e-6);
        // dist(0,2) = 1
        assert!((dist[2] - 1.0).abs() < 1e-6);
        // dist(1,2) = 2
        assert!((dist[1 * 3 + 2] - 2.0).abs() < 1e-6);
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

        let edges = build_leaf(&data, 2, &indices, 2, false);

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
        let dist = vec![
            f32::MAX, 1.0, 4.0,
            1.0, f32::MAX, 1.0,
            4.0, 1.0, f32::MAX,
        ];
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
            1.0, 0.0,   // point 0: along x
            0.0, 1.0,   // point 1: along y
            0.707, 0.707, // point 2: 45 degrees
            -1.0, 0.0,  // point 3: negative x
        ];
        // Normalize all vectors.
        for i in 0..4 {
            let row = &mut data[i * 2..(i + 1) * 2];
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in row.iter_mut() { *v /= norm; }
            }
        }

        let indices = vec![0, 1, 2, 3];
        let edges = build_leaf(&data, 2, &indices, 2, true);

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
        let edges = build_leaf(&data, 4, &indices, 3, false);
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
        let edges = build_leaf(&data, 2, &indices, 3, false);
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
        let data = vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ];
        let indices = vec![0, 1, 2, 3];
        let n = indices.len();
        // k = n means each point gets n-1 nearest neighbors = all others.
        let edges = build_leaf(&data, 2, &indices, n, false);

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
                        i, j
                    );
                }
            }
        }
    }

    #[test]
    fn test_build_leaf_with_buffers_reuse() {
        // Call build_leaf_with_buffers twice and verify buffers are reused.
        let data = vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ];
        let indices = vec![0, 1, 2, 3];
        let mut bufs = LeafBuffers::new();

        let edges1 = build_leaf_with_buffers(&data, 2, &indices, 2, false, &mut bufs);
        assert!(!edges1.is_empty(), "first call should produce edges");

        // Verify buffers are allocated.
        assert!(!bufs.local_data.is_empty(), "buffers should be allocated after first call");

        // Second call with same data should still work.
        let edges2 = build_leaf_with_buffers(&data, 2, &indices, 2, false, &mut bufs);
        assert_eq!(
            edges1.len(), edges2.len(),
            "same input should produce same number of edges with reused buffers"
        );
    }

    #[test]
    fn test_extract_knn_k_larger_than_n() {
        // k > n-1 should be clamped.
        let dist = vec![
            f32::MAX, 1.0,
            1.0, f32::MAX,
        ];
        let edges = extract_knn(&dist, 2, 100); // k=100 but only 2 points
        assert_eq!(
            edges.len(), 2,
            "k > n-1 should be clamped, each point gets 1 neighbor, total 2 edges"
        );
    }

    #[test]
    fn test_brute_force_knn_single_point() {
        let data = vec![5.0f32, 10.0];
        let query = vec![5.0, 10.0];
        let results = brute_force_knn(&data, 2, 1, &query, 5);
        assert_eq!(results.len(), 1, "brute force on 1 point should return 1 result");
        assert_eq!(results[0].0, 0, "should return the only point (index 0)");
        assert!(
            results[0].1 < 1e-6,
            "distance to identical query should be near zero"
        );
    }

    #[test]
    fn test_brute_force_knn_identity() {
        // query = data point, first result should be self with distance 0.
        let data = vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ];
        let query = vec![1.0, 0.0]; // same as point 1
        let results = brute_force_knn(&data, 2, 4, &query, 3);
        assert_eq!(results[0].0, 1, "query identical to point 1 should find it first");
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
        let data = vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            0.5, 0.5,
        ];
        let indices = vec![0, 1, 2, 3, 4];
        let edges = build_leaf(&data, 2, &indices, 2, false);

        // Collect directed edges as a set.
        let edge_set: std::collections::HashSet<(usize, usize)> =
            edges.iter().map(|e| (e.src, e.dst)).collect();

        // For every edge (a, b), (b, a) should also exist.
        for edge in &edges {
            assert!(
                edge_set.contains(&(edge.dst, edge.src)),
                "edge ({} -> {}) exists but reverse ({} -> {}) does not",
                edge.src, edge.dst, edge.dst, edge.src
            );
        }
    }
}
