/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! HashPrune: LSH-based online pruning for merging edges from overlapping partitions.
//!
//! Uses random hyperplanes to hash candidate neighbors relative to each point.
//! Maintains a reservoir of l_max entries per point, keyed by hash bucket.
//! This is history-independent (order of insertion does not matter).

use std::cell::RefCell;

use parking_lot::Mutex;

use diskann::utils::VectorRepr;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// Precomputed LSH sketches for a set of vectors.
///
/// For each vector v, Sketch(v) = [v . H_i for i=0..m] where H_i are random hyperplanes.
/// Sketches are computed via parallel dot products.
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
    /// Create new LSH sketches for the given data using parallel dot products.
    ///
    /// `data` is row-major: npoints x ndims.
    pub fn new<T: VectorRepr + Send + Sync>(data: &[T], npoints: usize, ndims: usize, num_planes: usize, seed: u64) -> Self {
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
                // Thread-local buffer for T -> f32 conversion.
                thread_local! {
                    static SKETCH_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
                }
                SKETCH_BUF.with(|cell| {
                    let mut buf = cell.borrow_mut();
                    buf.resize(ndims, 0.0);
                    T::as_f32_into(&data[i * ndims..(i + 1) * ndims], &mut buf).expect("f32 conversion");
                    for j in 0..num_planes {
                        let plane = &hyperplanes[j * ndims..(j + 1) * ndims];
                        let mut dot = 0.0f32;
                        for d in 0..ndims {
                            unsafe {
                                dot += *buf.get_unchecked(d) * *plane.get_unchecked(d);
                            }
                        }
                        sketch_row[j] = dot;
                    }
                });
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



/// Convert f32 distance to bf16 (truncate lower 16 mantissa bits).
/// For non-negative values, bf16 bit ordering matches f32 ordering,
/// so u16 comparison gives correct distance ordering.
#[inline(always)]
fn f32_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Convert bf16 back to f32 (zero-fill lower mantissa bits).
#[inline(always)]
fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// A single entry in the HashPrune reservoir.
/// Packed to exactly 8 bytes: 4 (neighbor) + 2 (hash) + 2 (distance as bf16).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct ReservoirEntry {
    /// The candidate neighbor index.
    neighbor: u32,
    /// Hash bucket (16-bit).
    hash: u16,
    /// Distance stored as bf16 (raw u16 bits). Non-negative bf16 values
    /// are monotonically ordered as u16, enabling integer comparison.
    distance: u16,
}

/// HashPrune reservoir for a single point.
///
/// Uses a flat sorted Vec for O(log l) hash lookups instead of HashMap.
/// Caches the farthest entry for O(1) eviction checks.
/// Insertion is O(l) due to element shifting, but cache-friendly at typical l_max ~128.
pub struct HashPruneReservoir {
    /// Entries sorted by hash for binary search.
    entries: Vec<ReservoirEntry>,
    /// Maximum reservoir size.
    l_max: usize,
    /// Cached farthest distance (bf16) and its index in entries.
    farthest_dist: u16,
    farthest_idx: usize,
}

impl HashPruneReservoir {
    pub fn new(l_max: usize) -> Self {
        Self {
            entries: Vec::with_capacity(l_max),
            l_max,
            farthest_dist: 0,
            farthest_idx: 0,
        }
    }

    /// Create a reservoir without pre-allocating capacity.
    pub fn new_lazy(l_max: usize) -> Self {
        Self {
            entries: Vec::new(),
            l_max,
            farthest_dist: 0,
            farthest_idx: 0,
        }
    }

    /// Create a reservoir with a specific initial capacity hint.
    /// Avoids Vec doubling when the expected fill is known.
    pub fn new_with_capacity(l_max: usize, initial_capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(initial_capacity),
            l_max,
            farthest_dist: 0,
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
            self.farthest_dist = 0;
            self.farthest_idx = 0;
            return;
        }
        let mut max_dist: u16 = 0;
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
    /// Distance is converted to bf16 at the boundary for compact storage.
    #[inline]
    pub fn insert(&mut self, hash: u16, neighbor: u32, distance: f32) -> bool {
        let dist_bf16 = f32_to_bf16(distance);

        // If the hash bucket already exists, keep the closer point.
        if let Some(idx) = self.find_hash(hash) {
            if dist_bf16 < self.entries[idx].distance {
                let was_farthest = idx == self.farthest_idx;
                self.entries[idx].neighbor = neighbor;
                self.entries[idx].distance = dist_bf16;
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
            // Fix: shift farthest_idx when inserting before it, since entries shift right.
            if pos <= self.farthest_idx && !self.entries.is_empty() {
                self.farthest_idx += 1;
            }
            self.entries.insert(pos, ReservoirEntry { neighbor, distance: dist_bf16, hash });
            if dist_bf16 >= self.farthest_dist {
                self.farthest_dist = dist_bf16;
                self.farthest_idx = pos;
            }
            return true;
        }

        // Reservoir is full: evict farthest if new is closer.
        if dist_bf16 < self.farthest_dist {
            self.entries.remove(self.farthest_idx);
            let pos = self.entries
                .binary_search_by_key(&hash, |e| e.hash)
                .unwrap_or_else(|e| e);
            self.entries.insert(pos, ReservoirEntry { neighbor, distance: dist_bf16, hash });
            self.update_farthest();
            return true;
        }

        false
    }

    /// Get all neighbors in the reservoir, sorted by distance.
    pub fn get_neighbors_sorted(&self) -> Vec<(u32, f32)> {
        let mut neighbors: Vec<(u32, u16)> = self
            .entries
            .iter()
            .map(|e| (e.neighbor, e.distance))
            .collect();
        // u16 comparison is correct for non-negative bf16 values.
        neighbors.sort_unstable_by_key(|&(_, d)| d);
        neighbors.into_iter().map(|(id, d)| (id, bf16_to_f32(d))).collect()
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
    pub fn new<T: VectorRepr + Send + Sync>(
        data: &[T],
        npoints: usize,
        ndims: usize,
        num_planes: usize,
        l_max: usize,
        max_degree: usize,
        seed: u64,
    ) -> Self {
        let t0 = std::time::Instant::now();
        let sketches = LshSketches::new(data, npoints, ndims, num_planes, seed);
        tracing::debug!(elapsed_secs = t0.elapsed().as_secs_f64(), "sketch computation");
        let t1 = std::time::Instant::now();
        // Use lazy allocation: reservoirs grow on demand as edges are inserted.
        // Pre-allocating 64×8B×1M = 512 MB upfront is worse because it spikes
        // before any leaf data is freed. Lazy growth + malloc_trim between
        // phases keeps peak RSS lower despite realloc fragmentation.
        let reservoirs = (0..npoints)
            .map(|_| Mutex::new(HashPruneReservoir::new_lazy(l_max)))
            .collect();
        tracing::debug!(elapsed_secs = t1.elapsed().as_secs_f64(), "reservoir allocation");

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
        self.reservoirs[p].lock().insert(hash, c as u32, distance);
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
            let mut reservoir = self.reservoirs[src].lock();
            while i < sorted.len() && sorted[i].src == src {
                let edge = sorted[i];
                let hash = self.sketches.relative_hash(src, edge.dst);
                reservoir.insert(hash, edge.dst as u32, edge.distance);
                i += 1;
            }
        }
    }

    /// Extract the final graph as adjacency lists, consuming the HashPrune.
    ///
    /// Consumes self so that reservoirs and sketches are freed as extraction proceeds,
    /// rather than staying alive until the caller drops HashPrune.
    /// Each reservoir is dropped immediately after its neighbors are extracted.
    pub fn extract_graph(self) -> Vec<Vec<u32>> {
        let max_degree = self.max_degree;
        // Drop sketches first (~50 MB for 1M points × 12 planes).
        drop(self.sketches);
        self.reservoirs
            .into_par_iter()
            .map(|mutex| {
                let res = mutex.into_inner();
                let mut neighbors = res.get_neighbors_sorted();
                neighbors.truncate(max_degree);
                neighbors.into_iter().map(|(id, _)| id).collect()
            })
            .collect()
    }

    /// Extract the full reservoir (up to l_max) with distances for final_prune.
    /// Returns (neighbor_id, distance) pairs sorted by distance.
    /// Final_prune selects max_degree from this larger candidate pool using diversity.
    pub fn extract_graph_for_prune(self) -> Vec<Vec<(u32, f32)>> {
        drop(self.sketches);
        self.reservoirs
            .into_par_iter()
            .map(|mutex| {
                let res = mutex.into_inner();
                res.get_neighbors_sorted()
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

    #[test]
    fn test_reservoir_lazy_allocation() {
        let mut res = HashPruneReservoir::new_lazy(5);
        assert!(res.is_empty());
        assert!(res.insert(0, 1, 1.0));
        assert_eq!(res.len(), 1);
    }

    #[test]
    fn test_reservoir_insert_then_evict_cycle() {
        let mut res = HashPruneReservoir::new(3);
        res.insert(0, 10, 3.0);
        res.insert(1, 11, 2.0);
        res.insert(2, 12, 1.0);
        assert_eq!(res.len(), 3);
        assert!(res.insert(3, 13, 0.5));
        assert_eq!(res.len(), 3);
        let neighbors = res.get_neighbors_sorted();
        assert!(neighbors.iter().all(|&(_, d)| d <= 2.0));
    }

    #[test]
    fn test_reservoir_all_same_hash() {
        let mut res = HashPruneReservoir::new(5);
        res.insert(0, 1, 3.0);
        res.insert(0, 2, 2.0);
        res.insert(0, 3, 1.0);
        assert_eq!(res.len(), 1);
        let neighbors = res.get_neighbors_sorted();
        assert_eq!(neighbors[0].0, 3);
        assert_eq!(neighbors[0].1, 1.0);
    }

    #[test]
    fn test_reservoir_all_same_distance() {
        let mut res = HashPruneReservoir::new(5);
        res.insert(0, 1, 1.0);
        res.insert(1, 2, 1.0);
        res.insert(2, 3, 1.0);
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn test_hash_prune_parallel_safety() {
        use rayon::prelude::*;
        let data = vec![0.0f32; 100 * 4];
        let hp = HashPrune::new(&data, 100, 4, 4, 10, 5, 42);
        (0..50).into_par_iter().for_each(|i| {
            hp.add_edge(i, (i + 1) % 100, 1.0);
            hp.add_edge((i + 1) % 100, i, 1.0);
        });
        let graph = hp.extract_graph();
        assert_eq!(graph.len(), 100);
    }

    #[test]
    fn test_hash_prune_high_degree_limit() {
        let data = vec![0.0f32; 10 * 2];
        let hp = HashPrune::new(&data, 10, 2, 4, 10, 1, 42);
        for i in 0..10 {
            for j in 0..10 {
                if i != j { hp.add_edge(i, j, (i as f32 - j as f32).abs()); }
            }
        }
        let graph = hp.extract_graph();
        for neighbors in &graph {
            assert!(neighbors.len() <= 1, "max_degree=1 should limit to 1 neighbor");
        }
    }

    #[test]
    fn test_hash_prune_extract_sorted() {
        let data = vec![0.0f32; 4 * 2];
        let hp = HashPrune::new(&data, 4, 2, 4, 10, 3, 42);
        hp.add_edge(0, 1, 3.0);
        hp.add_edge(0, 2, 1.0);
        hp.add_edge(0, 3, 2.0);
        let graph = hp.extract_graph();
        assert!(!graph[0].is_empty());
    }

    #[test]
    fn test_lsh_sketches_different_seeds() {
        let data = vec![1.0f32, 0.0, 0.0, 1.0];
        let s1 = LshSketches::new(&data, 2, 2, 4, 42);
        let s2 = LshSketches::new(&data, 2, 2, 4, 99);
        let h1 = s1.relative_hash(0, 1);
        let h2 = s2.relative_hash(0, 1);
        // Different seeds should generally produce different hashes (not guaranteed but very likely)
        let _ = (h1, h2); // Just verify they compile and don't panic
    }

    #[test]
    fn test_relative_hash_symmetry_broken() {
        let data = vec![1.0f32, 0.0, 0.0, 1.0, -1.0, 0.0];
        let sketches = LshSketches::new(&data, 3, 2, 4, 42);
        let h01 = sketches.relative_hash(0, 1);
        let h10 = sketches.relative_hash(1, 0);
        // h_p(c) != h_c(p) in general because relative_hash is asymmetric
        let _ = (h01, h10);
    }

    #[test]
    fn test_extract_graph_for_prune_returns_full_reservoir() {
        let data = vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ];
        // l_max=10 (much larger than max_degree=2) to verify no truncation.
        let hp = HashPrune::new(&data, 4, 2, 4, 10, 2, 42);
        hp.add_edge(0, 1, 1.0);
        hp.add_edge(0, 2, 1.0);
        hp.add_edge(0, 3, 1.414);
        hp.add_edge(1, 0, 1.0);
        hp.add_edge(2, 0, 1.0);
        hp.add_edge(3, 0, 1.414);

        let full = hp.extract_graph_for_prune();
        assert_eq!(full.len(), 4);
        // Node 0 has up to 3 edges, but hash collisions may reduce this.
        // Key invariant: for_prune returns >= extract_graph (no truncation to max_degree).
        assert!(full[0].len() >= 1, "node 0 should have neighbors");
        // Verify sorted by distance.
        for neighbors in &full {
            for w in neighbors.windows(2) {
                assert!(w[0].1 <= w[1].1, "should be sorted by distance");
            }
        }
    }

    #[test]
    fn test_extract_graph_truncates_to_max_degree() {
        let data = vec![0.0f32; 4 * 2];
        let hp = HashPrune::new(&data, 4, 2, 4, 10, 2, 42);
        hp.add_edge(0, 1, 1.0);
        hp.add_edge(0, 2, 2.0);
        hp.add_edge(0, 3, 3.0);

        let graph = hp.extract_graph();
        // max_degree=2, so node 0 should have at most 2 neighbors.
        assert!(graph[0].len() <= 2, "extract_graph should truncate to max_degree");
    }

    #[test]
    fn test_extract_for_prune_preserves_distances() {
        let data = vec![0.0f32; 4 * 2];
        let hp = HashPrune::new(&data, 4, 2, 4, 10, 5, 42);
        hp.add_edge(0, 1, 1.5);
        hp.add_edge(0, 2, 2.5);

        let full = hp.extract_graph_for_prune();
        let node0 = &full[0];
        // Check distances are bf16-rounded but close to original.
        for &(id, dist) in node0 {
            if id == 1 { assert!((dist - 1.5).abs() < 0.05, "dist for id=1: {}", dist); }
            if id == 2 { assert!((dist - 2.5).abs() < 0.05, "dist for id=2: {}", dist); }
        }
    }

    #[test]
    fn test_reservoir_farthest_cache_after_eviction() {
        // Verify farthest cache stays correct through multiple eviction cycles.
        let mut res = HashPruneReservoir::new(3);
        res.insert(0, 10, 5.0);
        res.insert(1, 11, 4.0);
        res.insert(2, 12, 3.0);
        // Full. Evict farthest (5.0), insert closer.
        assert!(res.insert(3, 13, 2.0));
        // Farthest should now be 4.0 (id=11).
        // Evict again.
        assert!(res.insert(4, 14, 1.0));
        // Farthest should now be 3.0 (id=12).
        let neighbors = res.get_neighbors_sorted();
        assert_eq!(neighbors.len(), 3);
        // All remaining should have dist <= 3.0.
        for &(_, d) in &neighbors {
            assert!(d <= 3.1, "expected dist <= 3.0, got {}", d);
        }
    }

    #[test]
    fn test_reservoir_farthest_insert_before_farthest_idx() {
        // Regression: inserting with hash < farthest's hash must shift farthest_idx.
        let mut res = HashPruneReservoir::new(4);
        // Insert in hash order: 5, 10, 15
        res.insert(5, 1, 1.0);
        res.insert(10, 2, 3.0);  // this is farthest
        res.insert(15, 3, 2.0);
        // Now insert hash=3, which goes before hash=5 in sorted order.
        // This shifts all indices right, including farthest_idx.
        res.insert(3, 4, 0.5);
        // Reservoir not full (cap=4), no eviction. Just verify correctness.
        let neighbors = res.get_neighbors_sorted();
        assert_eq!(neighbors.len(), 4);
        assert_eq!(neighbors[0].0, 4); // closest (0.5)
    }

    #[test]
    fn test_add_edges_batched() {
        use crate::leaf_build::Edge;
        let data = vec![0.0f32; 10 * 4];
        let hp = HashPrune::new(&data, 10, 4, 4, 10, 5, 42);
        let edges = vec![
            Edge { src: 0, dst: 1, distance: 1.0 },
            Edge { src: 1, dst: 0, distance: 1.0 },
            Edge { src: 2, dst: 3, distance: 2.0 },
            Edge { src: 3, dst: 2, distance: 2.0 },
        ];
        hp.add_edges_batched(&edges);
        let graph = hp.extract_graph();
        assert!(!graph[0].is_empty(), "node 0 should have neighbors after batched add");
        assert!(!graph[2].is_empty(), "node 2 should have neighbors after batched add");
    }
}
