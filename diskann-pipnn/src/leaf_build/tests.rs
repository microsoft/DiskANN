/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;
use diskann_vector::distance::{DistanceProvider, Metric};

#[derive(Debug, Clone, Copy)]
struct Edge {
    src: u32,
    dst: u32,
    distance: f32,
}

fn build_leaf(data: &[f32], ndims: usize, indices: &[u32], k: usize, metric: Metric) -> Vec<Edge> {
    let mut bufs = LeafBuffers::new();
    build_leaf_into(data, ndims, indices, k, metric, &mut bufs).unwrap();
    let mut edges = Vec::with_capacity(bufs.group_starts[indices.len()] as usize);
    for local_src in 0..indices.len() {
        let start = bufs.group_starts[local_src] as usize;
        let end = bufs.group_starts[local_src + 1] as usize;
        for &(local_dst, distance) in &bufs.group_data[start..end] {
            edges.push(Edge {
                src: indices[local_src],
                dst: indices[local_dst as usize],
                distance,
            });
        }
    }
    edges
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
fn test_build_leaf_buffer_reuse() {
    // Call the production leaf builder twice and verify buffers are reused.
    let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let indices = vec![0, 1, 2, 3];
    let mut bufs = LeafBuffers::new();

    let edges1 = build_leaf_into(&data, 2, &indices, 2, Metric::L2, &mut bufs).unwrap();
    assert!(edges1 > 0, "first call should produce edges");

    // Verify buffers are allocated.
    assert!(
        !bufs.local_data.is_empty(),
        "buffers should be allocated after first call"
    );

    // Second call with same data should still work.
    let edges2 = build_leaf_into(&data, 2, &indices, 2, Metric::L2, &mut bufs).unwrap();
    assert_eq!(
        edges1, edges2,
        "same input should produce same number of edges with reused buffers"
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
    let mut bufs = LeafBuffers::new();
    let edges1 = build_leaf_into(&data_large, 4, &indices_large, 2, Metric::L2, &mut bufs).unwrap();
    assert!(edges1 > 0);

    // Second call with smaller leaf on same thread — should reuse thread-local buffers.
    let data_small = vec![1.0f32; 4 * 4];
    let indices_small: Vec<u32> = (0..4).collect();
    let edges2 = build_leaf_into(&data_small, 4, &indices_small, 2, Metric::L2, &mut bufs).unwrap();
    assert!(
        edges2 > 0,
        "small leaf after large should work with reused buffers"
    );
}
