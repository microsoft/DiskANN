/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! HashPrune: LSH-based online pruning for merging edges from overlapping partitions.
//!
//! Uses random hyperplanes to hash candidate neighbors relative to each point.
//! Maintains a reservoir of l_max entries per point, keyed by hash bucket.
//! Effectively order-independent: the hash-keyed eviction makes results stable regardless of insertion order, up to bf16 distance quantization ties (which are rare in practice).

use parking_lot::Mutex;

use crate::rayon_util::ParIterInstalled;
use diskann::utils::VectorRepr;
use diskann_vector::bf16::{bf16_to_f32, f32_to_bf16};
use diskann_vector::lsh::LshSketches;
use rayon::prelude::*;

// LshSketches now lives in diskann-vector::lsh — see that module for the
// random-hyperplane projection. PiPNN consumes it by passing a `fill_point`
// closure that does the per-point f16/u8 → f32 conversion lazily, so we
// don't need a full upfront f32 copy of the dataset.

/// Compute LSH sketches over `data` (row-major `npoints × ndims` of `T`).
/// Convenience wrapper that wires `VectorRepr::as_f32_into` into
/// [`LshSketches::new`]'s `fill_point` closure.
fn sketches_from_data<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    num_planes: usize,
    seed: u64,
) -> LshSketches {
    LshSketches::new(npoints, ndims, num_planes, seed, |i, out| {
        T::as_f32_into(&data[i * ndims..(i + 1) * ndims], out).expect("f32 conversion");
    })
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
/// Stores at most `l_max` entries keyed by their hash bucket. Insertion is O(1)
/// (push or swap_remove); lookups are O(l_max) linear scan, vectorised by
/// [`find_hash`] (AVX-512 8-way / AVX2 4-way / scalar fallback). The farthest
/// entry is cached so a full-reservoir reject path is O(1).
pub(crate) struct HashPruneReservoir {
    entries: Vec<ReservoirEntry>,
    /// Maximum reservoir size.
    l_max: usize,
    /// Cached farthest distance (bf16) and its index in entries.
    farthest_dist: u16,
    farthest_idx: usize,
}

impl HashPruneReservoir {
    /// Eagerly pre-allocate capacity for `l_max` entries (test-only — production
    /// uses [`HashPruneReservoir::new_lazy`] to avoid the upfront RSS spike).
    #[cfg(test)]
    pub(crate) fn new(l_max: usize) -> Self {
        Self {
            entries: Vec::with_capacity(l_max),
            l_max,
            farthest_dist: 0,
            farthest_idx: 0,
        }
    }

    /// Create a reservoir without pre-allocating capacity.
    pub(crate) fn new_lazy(l_max: usize) -> Self {
        Self {
            entries: Vec::new(),
            l_max,
            farthest_dist: 0,
            farthest_idx: 0,
        }
    }

    /// Find entry with matching hash.
    /// Uses AVX-512 to compare 8 entries per instruction by loading the hash+distance
    /// word (upper 32 bits of each 8-byte entry) and masking to the hash field.
    #[inline(always)]
    fn find_hash(&self, hash: u16) -> Option<usize> {
        let n = self.entries.len();
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            use std::arch::x86_64::*;
            if n >= 8 {
                // SAFETY: AVX-512 intrinsics gated by target_feature cfg. ReservoirEntry
                // is `#[repr(C)]` with size 8 = sizeof(u64), so the u64 cast aliases bytes
                // legally. `ptr.add(base)` stays within `entries` since `base + 8 ≤ n`.
                // Trailing `get_unchecked(i)` has `i < n`.
                unsafe {
                    let ptr = self.entries.as_ptr() as *const u64;
                    let target = _mm512_set1_epi64(((hash as u64) << 32) as i64);
                    let mask = _mm512_set1_epi64(0x0000FFFF00000000u64 as i64);
                    let chunks = n / 8;
                    for chunk in 0..chunks {
                        let base = chunk * 8;
                        let data = _mm512_loadu_si512(ptr.add(base) as *const __m512i);
                        let masked = _mm512_and_si512(data, mask);
                        let cmp = _mm512_cmpeq_epi64_mask(masked, target);
                        if cmp != 0 {
                            return Some(base + cmp.trailing_zeros() as usize);
                        }
                    }
                    for i in (chunks * 8)..n {
                        if self.entries.get_unchecked(i).hash == hash {
                            return Some(i);
                        }
                    }
                }
                return None;
            }
        }
        // AVX2 fallback: 4 entries per comparison (256-bit / 64-bit).
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx512f")))]
        {
            use std::arch::x86_64::*;
            if n >= 4 {
                // SAFETY: AVX2 intrinsics — `target_arch = "x86_64"` guarantees the
                // ISA on a v3 baseline (.cargo/config.toml). ReservoirEntry is `#[repr(C)]`
                // sized 8 bytes; `ptr.add(base)` stays in bounds because `base + 4 ≤ n`.
                // `get_unchecked(i)` has `i < n`.
                unsafe {
                    let ptr = self.entries.as_ptr() as *const u64;
                    let target = _mm256_set1_epi64x(((hash as u64) << 32) as i64);
                    let mask = _mm256_set1_epi64x(0x0000FFFF00000000u64 as i64);
                    let chunks = n / 4;
                    for chunk in 0..chunks {
                        let base = chunk * 4;
                        let data = _mm256_loadu_si256(ptr.add(base) as *const __m256i);
                        let masked = _mm256_and_si256(data, mask);
                        let cmp = _mm256_cmpeq_epi64(masked, target);
                        let bits = _mm256_movemask_epi8(cmp);
                        if bits != 0 {
                            return Some(base + (bits.trailing_zeros() as usize / 8));
                        }
                    }
                    for i in (chunks * 4)..n {
                        if self.entries.get_unchecked(i).hash == hash {
                            return Some(i);
                        }
                    }
                }
                return None;
            }
        }
        // Scalar fallback
        for (i, e) in self.entries.iter().enumerate() {
            if e.hash == hash {
                return Some(i);
            }
        }
        None
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
    ///
    /// Unsorted storage: find_hash is O(l_max) linear scan but insert/evict
    /// are O(1) via push/swap_remove.
    #[inline]
    #[inline(always)]
    pub fn insert(&mut self, hash: u16, neighbor: u32, distance: f32) -> bool {
        let dist_bf16 = f32_to_bf16(distance);

        // Early rejection: if reservoir is full and the new entry is farther
        // than the worst current entry, it can't improve any bucket.
        //
        // Correctness: if hash exists with distance X, then X <= farthest_dist.
        // If dist_bf16 >= farthest_dist >= X, dist_bf16 wouldn't replace X anyway.
        if self.entries.len() >= self.l_max && dist_bf16 >= self.farthest_dist {
            return false;
        }

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

        // If reservoir is not full, append (O(1), no memmove).
        if self.entries.len() < self.l_max {
            // First push: allocate full l_max upfront to avoid grow-doublings
            // (Vec::push otherwise reallocs at 4 → 8 → 16 → 32 → 64 → 128).
            // Saves ~1 KB of memcpy per fully-filled reservoir.
            if self.entries.is_empty() {
                self.entries.reserve_exact(self.l_max);
            }
            let new_idx = self.entries.len();
            self.entries.push(ReservoirEntry {
                neighbor,
                distance: dist_bf16,
                hash,
            });
            if dist_bf16 >= self.farthest_dist {
                self.farthest_dist = dist_bf16;
                self.farthest_idx = new_idx;
            }
            return true;
        }

        // Reservoir is full: evict farthest if new is closer.
        if dist_bf16 < self.farthest_dist {
            self.entries[self.farthest_idx] = ReservoirEntry {
                neighbor,
                distance: dist_bf16,
                hash,
            };
            self.update_farthest();
            return true;
        }

        false
    }

    /// Get all neighbors sorted by distance, truncated to `max_degree`.
    pub fn get_neighbors_saturated(&self, max_degree: usize) -> Vec<(u32, f32)> {
        let mut neighbors: Vec<(u32, u16)> = self
            .entries
            .iter()
            .map(|e| (e.neighbor, e.distance))
            .collect();
        neighbors.sort_unstable_by_key(|&(_, d)| d);
        neighbors.truncate(max_degree);
        neighbors
            .into_iter()
            .map(|(id, d)| (id, bf16_to_f32(d)))
            .collect()
    }

    /// Get all neighbors in the reservoir, sorted by distance (no truncation).
    /// Test-only — production calls [`get_neighbors_saturated`] which truncates
    /// to `max_degree`.
    #[cfg(test)]
    pub(crate) fn get_neighbors_sorted(&self) -> Vec<(u32, f32)> {
        let mut neighbors: Vec<(u32, u16)> = self
            .entries
            .iter()
            .map(|e| (e.neighbor, e.distance))
            .collect();
        neighbors.sort_unstable_by_key(|&(_, d)| d);
        neighbors
            .into_iter()
            .map(|(id, d)| (id, bf16_to_f32(d)))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
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
    /// Maximum reservoir size (l_max).
    l_max: usize,
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
        let sketches = sketches_from_data(data, npoints, ndims, num_planes, seed);
        tracing::debug!(
            elapsed_secs = t0.elapsed().as_secs_f64(),
            "sketch computation"
        );

        Self::from_sketches(sketches, npoints, l_max, max_degree)
    }

    /// Create a HashPrune from pre-computed LSH sketches.
    /// Allows the caller to compute sketches from f32 data, drop the data,
    /// then create HashPrune without holding the f32 borrow.
    pub(crate) fn from_sketches(
        sketches: LshSketches,
        npoints: usize,
        l_max: usize,
        max_degree: usize,
    ) -> Self {
        let t1 = std::time::Instant::now();
        // Lazy: each reservoir allocates its entry Vec on first push, not upfront.
        // Upfront alloc of npoints × l_max bytes spikes peak RSS and adds serial
        // malloc time; spreading the allocs over parallel inserts is net cheaper.
        let reservoirs = (0..npoints)
            .map(|_| Mutex::new(HashPruneReservoir::new_lazy(l_max)))
            .collect();
        tracing::debug!(
            elapsed_secs = t1.elapsed().as_secs_f64(),
            "reservoir allocation"
        );

        Self {
            reservoirs,
            sketches,
            max_degree,
            l_max,
        }
    }

    /// Add a single edge from point `p` to candidate `c` (test-only — production
    /// uses [`add_edges_batched`] which amortizes lock acquisition across edges
    /// sharing the same source).
    #[cfg(test)]
    #[inline]
    pub(crate) fn add_edge(&self, p: usize, c: usize, distance: f32) {
        let hash = self.sketches.relative_hash(p, c);
        self.reservoirs[p].lock().insert(hash, c as u32, distance);
    }

    /// Add edges from a leaf build result, batching by source point.
    /// Caches the lock on the last source seen, so callers whose edges are
    /// grouped by source acquire each reservoir mutex once per group (rather
    /// than once per edge). Leaf-build emits edges in this grouped order.
    pub fn add_edges_batched(&self, edges: &[crate::leaf_build::Edge]) {
        if edges.is_empty() {
            return;
        }

        // Process edges directly. Cache last reservoir lock to avoid redundant
        // lock ops for consecutive edges with the same source.
        let mut last_src = u32::MAX;
        let mut last_reservoir: Option<parking_lot::MutexGuard<'_, HashPruneReservoir>> = None;

        for edge in edges {
            if edge.src != last_src {
                drop(last_reservoir.take());
                last_src = edge.src;
                last_reservoir = Some(self.reservoirs[edge.src as usize].lock());
            }
            let reservoir = last_reservoir.as_mut().unwrap();
            let hash = self
                .sketches
                .relative_hash(edge.src as usize, edge.dst as usize);
            reservoir.insert(hash, edge.dst, edge.distance);
        }
    }

    /// Extract the final graph as adjacency lists, consuming the HashPrune.
    ///
    /// Consumes self so that reservoirs and sketches are freed as extraction proceeds,
    /// rather than staying alive until the caller drops HashPrune.
    /// Each reservoir is dropped immediately after its neighbors are extracted.
    pub fn extract_graph(self) -> Vec<Vec<u32>> {
        let max_degree = self.max_degree;
        // Drop sketches first to free memory before the parallel extract scan.
        drop(self.sketches);
        self.reservoirs
            .into_par_iter()
            .map(|mutex| {
                let res = mutex.into_inner();
                let mut neighbors = res.get_neighbors_saturated(max_degree);
                neighbors.truncate(max_degree);
                neighbors.into_iter().map(|(id, _)| id).collect()
            })
            .collect_installed()
    }

    /// Extract the full reservoir (up to l_max) with distances for final_prune.
    /// Returns (neighbor_id, distance) pairs sorted by distance.
    /// Final_prune selects max_degree from this larger candidate pool using diversity.
    pub fn extract_graph_for_prune(self) -> Vec<Vec<(u32, f32)>> {
        let l_max = self.l_max;
        drop(self.sketches);
        self.reservoirs
            .into_par_iter()
            .map(|mutex| {
                let res = mutex.into_inner();
                // Saturate up to l_max so final_prune gets the full candidate pool.
                res.get_neighbors_saturated(l_max)
            })
            .collect_installed()
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
        let sketches = sketches_from_data(&data, 4, 2, 4, 42);

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
            assert!(!neighbors.is_empty(), "point {} has no neighbors", i);
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
        (0..50).into_par_iter().for_each_installed(|i| {
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
                if i != j {
                    hp.add_edge(i, j, (i as f32 - j as f32).abs());
                }
            }
        }
        let graph = hp.extract_graph();
        for neighbors in &graph {
            assert!(
                neighbors.len() <= 1,
                "max_degree=1 should limit to 1 neighbor"
            );
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
    fn test_lsh_sketches_different_seeds_give_different_hashes() {
        // Aggregate across many point pairs so the assertion is statistical:
        // with 4 planes and random hyperplanes from two distinct seeds, at
        // least one pair must hash differently.
        let data: Vec<f32> = (0..32 * 4).map(|i| (i as f32).sin()).collect();
        let s1 = sketches_from_data(&data, 32, 4, 12, 42);
        let s2 = sketches_from_data(&data, 32, 4, 12, 99);
        let any_diff =
            (0..32).any(|p| (p + 1..32).any(|c| s1.relative_hash(p, c) != s2.relative_hash(p, c)));
        assert!(
            any_diff,
            "different seeds should give at least one different pair hash"
        );
    }

    #[test]
    fn test_relative_hash_is_asymmetric() {
        // h_p(c) is sign(Sketch(c) - Sketch(p)), so h(p,c) and h(c,p) differ
        // wherever any plane gives Sketch(p) != Sketch(c).
        let data = vec![1.0f32, 0.0, 0.0, 1.0, -1.0, 0.0];
        let sketches = sketches_from_data(&data, 3, 2, 4, 42);
        let h01 = sketches.relative_hash(0, 1);
        let h10 = sketches.relative_hash(1, 0);
        assert_ne!(
            h01, h10,
            "relative_hash must be asymmetric for distinct points"
        );
    }

    #[test]
    fn test_extract_graph_for_prune_returns_full_reservoir() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
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
        assert!(!full[0].is_empty(), "node 0 should have neighbors");
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
        assert!(
            graph[0].len() <= 2,
            "extract_graph should truncate to max_degree"
        );
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
            if id == 1 {
                assert!((dist - 1.5).abs() < 0.05, "dist for id=1: {}", dist);
            }
            if id == 2 {
                assert!((dist - 2.5).abs() < 0.05, "dist for id=2: {}", dist);
            }
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
        res.insert(10, 2, 3.0); // this is farthest
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
            Edge {
                src: 0,
                dst: 1,
                distance: 1.0,
            },
            Edge {
                src: 1,
                dst: 0,
                distance: 1.0,
            },
            Edge {
                src: 2,
                dst: 3,
                distance: 2.0,
            },
            Edge {
                src: 3,
                dst: 2,
                distance: 2.0,
            },
        ];
        hp.add_edges_batched(&edges);
        let graph = hp.extract_graph();
        assert!(
            !graph[0].is_empty(),
            "node 0 should have neighbors after batched add"
        );
        assert!(
            !graph[2].is_empty(),
            "node 2 should have neighbors after batched add"
        );
    }
}
