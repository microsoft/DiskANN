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

    for i in 0..n {
        let row = &dist_matrix[i * n..(i + 1) * n];

        // Collect (index, distance) pairs.
        let mut dists: Vec<(usize, f32)> = (0..n)
            .map(|j| (j, row[j]))
            .collect();

        // Partial sort to get the k nearest.
        if actual_k < dists.len() {
            dists.select_nth_unstable_by(actual_k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists.truncate(actual_k);
        }

        for (j, dist) in dists {
            edges.push((i, j, dist));
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
    let dot_matrix = &mut bufs.dot_matrix[..n * n];
    crate::gemm::sgemm_aat(local_data, n, ndims, dot_matrix);

    // Compute distance matrix into reused buffer.
    let dist_matrix = &mut bufs.dist_matrix[..n * n];
    if use_cosine {
        for i in 0..n {
            let dr = &mut dist_matrix[i * n..(i + 1) * n];
            let dotr = &dot_matrix[i * n..(i + 1) * n];
            for j in 0..n { dr[j] = (1.0 - dotr[j]).max(0.0); }
            dr[i] = f32::MAX;
        }
    } else {
        for i in 0..n {
            let ni = norms_sq[i];
            let dr = &mut dist_matrix[i * n..(i + 1) * n];
            let dotr = &dot_matrix[i * n..(i + 1) * n];
            for j in 0..n { dr[j] = (ni + norms_sq[j] - 2.0 * dotr[j]).max(0.0); }
            dr[i] = f32::MAX;
        }
    }

    // Extract k-NN edges.
    let local_edges = extract_knn(dist_matrix, n, k);

    // Create bi-directed edges using reused seen buffer.
    let seen = &mut bufs.seen[..n * n];
    for v in seen.iter_mut() { *v = false; }

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
}
