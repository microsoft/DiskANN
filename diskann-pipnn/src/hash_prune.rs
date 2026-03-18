/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! HashPrune: LSH-based online pruning for merging edges from overlapping partitions.
//!
//! Uses random hyperplanes to hash candidate neighbors relative to each point.
//! Maintains a reservoir of l_max entries per point, keyed by hash bucket.
//! This is history-independent (order of insertion does not matter).

use std::sync::Mutex;

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Precomputed LSH sketches for a set of vectors.
///
/// For each vector v, Sketch(v) = [v . H_i for i=0..m] where H_i are random hyperplanes.
/// Sketches are computed as a GEMM: Sketches = Data * Hyperplanes^T.
pub struct LshSketches {
    /// Number of hyperplanes (m).
    num_planes: usize,
    /// Precomputed sketches: npoints x m, stored row-major.
    /// sketch[i * m + j] = dot(point_i, hyperplane_j)
    sketches: Vec<f32>,
    /// Number of points.
    npoints: usize,
}

impl LshSketches {
    /// Create new LSH sketches for the given data using GEMM.
    ///
    /// `data` is row-major: npoints x ndims.
    pub fn new(data: &[f32], npoints: usize, ndims: usize, num_planes: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Generate random hyperplanes from standard normal distribution.
        // Stored as num_planes x ndims (row-major).
        let hyperplanes: Vec<f32> = (0..num_planes * ndims)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

        // Compute sketches in parallel using direct dot products.
        // For tall-thin output (npoints x 12), this is faster than GEMM.
        let mut sketches = vec![0.0f32; npoints * num_planes];

        sketches
            .par_chunks_mut(num_planes)
            .enumerate()
            .for_each(|(i, sketch_row)| {
                let point = &data[i * ndims..(i + 1) * ndims];
                for j in 0..num_planes {
                    let plane = &hyperplanes[j * ndims..(j + 1) * ndims];
                    let mut dot = 0.0f32;
                    for d in 0..ndims {
                        unsafe {
                            dot += *point.get_unchecked(d) * *plane.get_unchecked(d);
                        }
                    }
                    sketch_row[j] = dot;
                }
            });

        Self {
            num_planes,
            sketches,
            npoints,
        }
    }

    /// Compute the hash of candidate c relative to point p.
    ///
    /// h_p(c) = concat of sign bits of (Sketch(c) - Sketch(p))
    /// Returns a u16 hash (supports up to 16 hyperplanes, matching paper's 8-byte entry).
    #[inline(always)]
    pub fn relative_hash(&self, p: usize, c: usize) -> u16 {
        debug_assert!(p < self.npoints);
        debug_assert!(c < self.npoints);
        debug_assert!(self.num_planes <= 16);

        let m = self.num_planes;
        let p_sketch = &self.sketches[p * m..(p + 1) * m];
        let c_sketch = &self.sketches[c * m..(c + 1) * m];

        let mut hash: u16 = 0;
        for j in 0..m {
            let diff = c_sketch[j] - p_sketch[j];
            if diff >= 0.0 {
                hash |= 1u16 << j;
            }
        }
        hash
    }
}

/// Compute A * B^T where A is n x d and B is m x d.
/// Result is n x m (row-major).
/// Uses matrixmultiply for near-BLAS performance.
fn gemm_abt(a: &[f32], n: usize, d: usize, b: &[f32], m: usize, result: &mut [f32]) {
    debug_assert_eq!(a.len(), n * d);
    debug_assert_eq!(b.len(), m * d);
    debug_assert_eq!(result.len(), n * m);
    result.fill(0.0);

    unsafe {
        matrixmultiply::sgemm(
            n,
            d,
            m,
            1.0,
            a.as_ptr(),
            d as isize,
            1,
            b.as_ptr(),
            1,
            d as isize,
            0.0,
            result.as_mut_ptr(),
            m as isize,
            1,
        );
    }
}

/// A single entry in the HashPrune reservoir.
/// Packed to 8 bytes matching the paper's design.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct ReservoirEntry {
    /// The candidate neighbor index.
    neighbor: u32,
    /// Hash bucket (16-bit).
    hash: u16,
    /// Distance stored as bf16-like (we use f32 but the struct is for the concept).
    /// We store the raw f32 distance separately for accuracy.
    distance: f32,
}

/// HashPrune reservoir for a single point.
///
/// Uses a flat sorted Vec for O(log l) hash lookups instead of HashMap.
/// Caches the farthest entry for O(1) eviction checks.
pub struct HashPruneReservoir {
    /// Entries sorted by hash for binary search.
    entries: Vec<ReservoirEntry>,
    /// Maximum reservoir size.
    l_max: usize,
    /// Cached farthest distance and its index in entries.
    farthest_dist: f32,
    farthest_idx: usize,
}

impl HashPruneReservoir {
    pub fn new(l_max: usize) -> Self {
        Self {
            entries: Vec::with_capacity(l_max),
            l_max,
            farthest_dist: f32::NEG_INFINITY,
            farthest_idx: 0,
        }
    }

    /// Create a reservoir without pre-allocating capacity.
    /// Saves memory and init time when most reservoirs stay small.
    pub fn new_lazy(l_max: usize) -> Self {
        Self {
            entries: Vec::new(),
            l_max,
            farthest_dist: f32::NEG_INFINITY,
            farthest_idx: 0,
        }
    }

    /// Find entry with matching hash using binary search.
    #[inline]
    fn find_hash(&self, hash: u16) -> Option<usize> {
        self.entries
            .binary_search_by_key(&hash, |e| e.hash)
            .ok()
    }

    /// Update the cached farthest entry.
    #[inline]
    fn update_farthest(&mut self) {
        if self.entries.is_empty() {
            self.farthest_dist = f32::NEG_INFINITY;
            self.farthest_idx = 0;
            return;
        }
        let mut max_dist = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.distance > max_dist {
                max_dist = entry.distance;
                max_idx = idx;
            }
        }
        self.farthest_dist = max_dist;
        self.farthest_idx = max_idx;
    }

    /// Try to insert a candidate neighbor with the given hash and distance.
    #[inline]
    pub fn insert(&mut self, hash: u16, neighbor: u32, distance: f32) -> bool {
        // If the hash bucket already exists, keep the closer point.
        if let Some(idx) = self.find_hash(hash) {
            if distance < self.entries[idx].distance {
                let was_farthest = idx == self.farthest_idx;
                self.entries[idx].neighbor = neighbor;
                self.entries[idx].distance = distance;
                if was_farthest {
                    self.update_farthest();
                }
                return true;
            }
            return false;
        }

        // If reservoir is not full, insert in sorted position.
        if self.entries.len() < self.l_max {
            let pos = self.entries
                .binary_search_by_key(&hash, |e| e.hash)
                .unwrap_or_else(|e| e);
            self.entries.insert(pos, ReservoirEntry { neighbor, distance, hash });
            if distance > self.farthest_dist {
                self.farthest_dist = distance;
                // Position may have shifted
                self.update_farthest();
            } else if self.entries.len() == 1 {
                self.farthest_dist = distance;
                self.farthest_idx = 0;
            }
            return true;
        }

        // Reservoir is full: evict farthest if new is closer.
        if distance < self.farthest_dist {
            self.entries.remove(self.farthest_idx);
            let pos = self.entries
                .binary_search_by_key(&hash, |e| e.hash)
                .unwrap_or_else(|e| e);
            self.entries.insert(pos, ReservoirEntry { neighbor, distance, hash });
            self.update_farthest();
            return true;
        }

        false
    }

    /// Get all neighbors in the reservoir, sorted by distance.
    pub fn get_neighbors_sorted(&self) -> Vec<(u32, f32)> {
        let mut neighbors: Vec<(u32, f32)> = self
            .entries
            .iter()
            .map(|e| (e.neighbor, e.distance))
            .collect();
        neighbors.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbors
    }

    /// Get the number of entries in the reservoir.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the reservoir is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The global HashPrune state managing reservoirs for all points.
/// Uses per-point Mutex for thread-safe parallel edge insertion.
pub struct HashPrune {
    /// One reservoir per point, each behind a Mutex for parallel access.
    reservoirs: Vec<Mutex<HashPruneReservoir>>,
    /// LSH sketches.
    sketches: LshSketches,
    /// Maximum degree for the final graph.
    max_degree: usize,
}

impl HashPrune {
    /// Create a new HashPrune instance.
    ///
    /// `data` is row-major: npoints x ndims.
    pub fn new(
        data: &[f32],
        npoints: usize,
        ndims: usize,
        num_planes: usize,
        l_max: usize,
        max_degree: usize,
        seed: u64,
    ) -> Self {
        let t0 = std::time::Instant::now();
        let sketches = LshSketches::new(data, npoints, ndims, num_planes, seed);
        eprintln!("    sketch: {:.3}s", t0.elapsed().as_secs_f64());
        let t1 = std::time::Instant::now();
        // Use lazy allocation: don't pre-allocate reservoir capacity.
        // Reservoirs grow on demand as edges are inserted.
        let reservoirs = (0..npoints)
            .map(|_| Mutex::new(HashPruneReservoir::new_lazy(l_max)))
            .collect();
        eprintln!("    reservoirs: {:.3}s", t1.elapsed().as_secs_f64());

        Self {
            reservoirs,
            sketches,
            max_degree,
        }
    }

    /// Add an edge from point `p` to candidate `c` with the given distance.
    /// Thread-safe: acquires lock on p's reservoir only.
    #[inline]
    pub fn add_edge(&self, p: usize, c: usize, distance: f32) {
        let hash = self.sketches.relative_hash(p, c);
        self.reservoirs[p].lock().unwrap().insert(hash, c as u32, distance);
    }

    /// Add a batch of edges in parallel. Each edge is (point_idx, neighbor_idx, distance).
    pub fn add_edges_parallel(&self, edges: &[(usize, usize, f32)]) {
        edges.par_iter().for_each(|&(p, c, dist)| {
            self.add_edge(p, c, dist);
        });
    }

    /// Add edges from a leaf build result, batching by source point.
    /// Sorts edges by source to acquire each lock once per unique source.
    pub fn add_edges_batched(&self, edges: &[crate::leaf_build::Edge]) {
        if edges.is_empty() {
            return;
        }

        let mut sorted: Vec<&crate::leaf_build::Edge> = edges.iter().collect();
        sorted.sort_unstable_by_key(|e| e.src);

        let mut i = 0;
        while i < sorted.len() {
            let src = sorted[i].src;
            let mut reservoir = self.reservoirs[src].lock().unwrap();
            while i < sorted.len() && sorted[i].src == src {
                let edge = sorted[i];
                let hash = self.sketches.relative_hash(src, edge.dst);
                reservoir.insert(hash, edge.dst as u32, edge.distance);
                i += 1;
            }
        }
    }

    /// Extract the final graph as adjacency lists.
    ///
    /// Returns a vector of neighbor lists (one per point), each truncated to max_degree.
    pub fn extract_graph(&self) -> Vec<Vec<u32>> {
        self.reservoirs
            .par_iter()
            .map(|reservoir| {
                let res = reservoir.lock().unwrap();
                let mut neighbors = res.get_neighbors_sorted();
                neighbors.truncate(self.max_degree);
                neighbors.into_iter().map(|(id, _)| id).collect()
            })
            .collect()
    }

    /// Get the number of points.
    pub fn num_points(&self) -> usize {
        self.reservoirs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_basic() {
        let mut reservoir = HashPruneReservoir::new(3);
        assert!(reservoir.is_empty());

        // Insert three entries with different hashes.
        assert!(reservoir.insert(0, 1, 1.0));
        assert!(reservoir.insert(1, 2, 2.0));
        assert!(reservoir.insert(2, 3, 3.0));
        assert_eq!(reservoir.len(), 3);

        // Reservoir is full. New closer entry should evict the farthest.
        assert!(reservoir.insert(3, 4, 0.5));
        assert_eq!(reservoir.len(), 3);

        let neighbors = reservoir.get_neighbors_sorted();
        // Should not contain the farthest entry (neighbor 3, distance 3.0).
        assert!(!neighbors.iter().any(|(id, _)| *id == 3));
        // Should contain the new closer entry.
        assert!(neighbors.iter().any(|(id, _)| *id == 4));
    }

    #[test]
    fn test_reservoir_same_hash_keeps_closer() {
        let mut reservoir = HashPruneReservoir::new(10);

        assert!(reservoir.insert(0, 1, 2.0));
        assert_eq!(reservoir.len(), 1);

        // Same hash, closer distance: should update.
        assert!(reservoir.insert(0, 2, 1.0));
        assert_eq!(reservoir.len(), 1);

        let neighbors = reservoir.get_neighbors_sorted();
        assert_eq!(neighbors[0].0, 2);
        assert_eq!(neighbors[0].1, 1.0);

        // Same hash, farther distance: should not update.
        assert!(!reservoir.insert(0, 3, 5.0));
        assert_eq!(reservoir.len(), 1);
    }

    #[test]
    fn test_lsh_sketches() {
        // Simple test with 4 points in 2D.
        let data = vec![
            1.0, 0.0, // point 0
            0.0, 1.0, // point 1
            -1.0, 0.0, // point 2
            0.0, -1.0, // point 3
        ];
        let sketches = LshSketches::new(&data, 4, 2, 4, 42);

        // Relative hash of a point with itself: all diffs are 0, 0.0 >= 0.0 is true.
        let h00 = sketches.relative_hash(0, 0);
        assert_eq!(h00, (1u16 << 4) - 1);

        // Different points should generally have different hashes.
        let h01 = sketches.relative_hash(0, 1);
        let h02 = sketches.relative_hash(0, 2);
        let _ = (h01, h02);
    }

    #[test]
    fn test_hash_prune_end_to_end() {
        // 4 points in 2D.
        let data = vec![
            0.0, 0.0, // point 0
            1.0, 0.0, // point 1
            0.0, 1.0, // point 2
            1.0, 1.0, // point 3
        ];

        let hp = HashPrune::new(&data, 4, 2, 4, 10, 3, 42);

        // Add some edges.
        hp.add_edge(0, 1, 1.0);
        hp.add_edge(0, 2, 1.0);
        hp.add_edge(0, 3, 1.414);
        hp.add_edge(1, 0, 1.0);
        hp.add_edge(1, 3, 1.0);
        hp.add_edge(2, 0, 1.0);
        hp.add_edge(2, 3, 1.0);
        hp.add_edge(3, 1, 1.0);
        hp.add_edge(3, 2, 1.0);

        let graph = hp.extract_graph();
        assert_eq!(graph.len(), 4);

        for (i, neighbors) in graph.iter().enumerate() {
            assert!(
                !neighbors.is_empty(),
                "point {} has no neighbors",
                i
            );
        }
    }
}
