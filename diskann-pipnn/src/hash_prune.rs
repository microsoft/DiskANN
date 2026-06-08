/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! HashPrune: LSH-based online pruning for merging edges from overlapping partitions.
//!
//! Storage is AoSoA hot/cold split:
//! - `hot: Vec<HotSlot>` — one 16-byte slot per point (mutex + len + farthest).
//!   Hot path early-reject and lock-acquire only touch this slab (~160 MB at 10M points).
//! - `cold: Vec<ColdSlot>` — one 512-byte slot per point. Inside each ColdSlot,
//!   hashes / distances / neighbors are split into three contiguous arrays, so
//!   `find_hash` only walks 2 cache lines (128 B of u16) instead of 8 cache lines
//!   of mixed AoS.
//!
//! Both slabs are one Vec each (contiguous in VA), so `madvise(HUGEPAGE)` is
//! effective when the kernel actually backs THP.
//!
//! `L_MAX_MAX` is hard-coded at 64 — the production value. Configs with
//! `l_max > 64` are rejected at construction time.

use parking_lot::lock_api::RawMutex as RawMutexTrait;

use crate::rayon_util::ParIterInstalled;
use diskann::utils::VectorRepr;
use diskann_vector::bf16::{bf16_to_f32, f32_to_bf16};
use diskann_vector::lsh::LshSketches;
use rayon::prelude::*;

/// Fixed cap on reservoir size. Production benchmark (`l_max = 64`) is the
/// tuned value; `find_hash_simd` is also 64-lane-optimal at 2 SIMD ops per
/// scan. Tests that use `PiPNNConfig::default()` (`l_max = 128`) must override
/// to `l_max = 64`.
pub(crate) const L_MAX_MAX: usize = 128;

/// Compute LSH sketches over `data` (row-major `npoints × ndims` of `T`).
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

// ─── HotSlot: 16-byte per-point mutex + cached fields ─────────────────────────

#[repr(C, align(16))]
pub(crate) struct HotSlot {
    lock: parking_lot::RawMutex,
    len: u8,
    farthest_idx: u8,
    _pad0: u8,
    farthest_dist: u16,
    _pad1: [u8; 10],
}

impl HotSlot {
    const fn new_empty() -> Self {
        Self {
            lock: <parking_lot::RawMutex as RawMutexTrait>::INIT,
            len: 0,
            farthest_idx: 0,
            _pad0: 0,
            farthest_dist: 0,
            _pad1: [0; 10],
        }
    }
}

const _: () = assert!(std::mem::size_of::<HotSlot>() == 16);

// ─── ColdSlot: AoSoA entries ──────────────────────────────────────────────────

#[repr(C, align(64))]
pub(crate) struct ColdSlot {
    pub(crate) hashes: [u16; L_MAX_MAX],
    pub(crate) distances: [u16; L_MAX_MAX],
    pub(crate) neighbors: [u32; L_MAX_MAX],
}

const _: () = assert!(std::mem::size_of::<ColdSlot>() == 1024);

// ─── find_hash SIMD: 32-way u16 compare ───────────────────────────────────────

#[inline(always)]
fn find_hash_simd(hashes: &[u16; L_MAX_MAX], len: u8, target: u16) -> Option<usize> {
    let len = len as usize;
    if len == 0 {
        return None;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    {
        use std::arch::x86_64::*;
        // SAFETY: AVX-512 BW cfg-gated. Iterate L_MAX_MAX/32 chunks of 32 u16
        // each. Compute per-chunk eq mask, accumulate into a u128, mask by
        // `len`, then trailing_zeros. Works for L_MAX_MAX = 32, 64, 128, 256.
        unsafe {
            let t = _mm512_set1_epi16(target as i16);
            const CHUNKS: usize = L_MAX_MAX / 32;
            let mut combined: u128 = 0;
            for chunk in 0..CHUNKS {
                let v = _mm512_loadu_si512(hashes.as_ptr().add(chunk * 32) as *const __m512i);
                let m = _mm512_cmpeq_epi16_mask(v, t) as u128;
                combined |= m << (chunk * 32);
            }
            let len_mask: u128 = if len >= 128 { u128::MAX } else { (1u128 << len) - 1 };
            let valid = combined & len_mask;
            if valid != 0 {
                Some(valid.trailing_zeros() as usize)
            } else {
                None
            }
        }
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512bw")
    ))]
    {
        use std::arch::x86_64::*;
        // SAFETY: AVX2 cfg-gated; CHUNKS = L_MAX_MAX/16 chunks × 16 u16.
        unsafe {
            let t = _mm256_set1_epi16(target as i16);
            const CHUNKS: usize = L_MAX_MAX / 16;
            for chunk in 0..CHUNKS {
                let v = _mm256_loadu_si256(hashes.as_ptr().add(chunk * 16) as *const __m256i);
                let m = _mm256_cmpeq_epi16(v, t);
                let bits = _mm256_movemask_epi8(m) as u32;
                if bits != 0 {
                    let lane = chunk * 16 + (bits.trailing_zeros() as usize) / 2;
                    if lane < len {
                        return Some(lane);
                    }
                }
            }
            None
        }
    }
    #[cfg(not(any(target_feature = "avx512bw", target_feature = "avx2")))]
    {
        for i in 0..len {
            if hashes[i] == target {
                return Some(i);
            }
        }
        None
    }
}

// ─── Per-reservoir mutation helpers (caller holds lock) ───────────────────────

#[inline]
fn update_farthest(hot: &mut HotSlot, cold: &ColdSlot) {
    if hot.len == 0 {
        hot.farthest_dist = 0;
        hot.farthest_idx = 0;
        return;
    }
    let mut max_dist: u16 = 0;
    let mut max_idx: u8 = 0;
    for i in 0..hot.len as usize {
        let d = cold.distances[i];
        if d > max_dist {
            max_dist = d;
            max_idx = i as u8;
        }
    }
    hot.farthest_dist = max_dist;
    hot.farthest_idx = max_idx;
}

#[inline(always)]
fn insert_locked(
    hot: &mut HotSlot,
    cold: &mut ColdSlot,
    hash: u16,
    neighbor: u32,
    distance: f32,
    l_max: u8,
) -> bool {
    let dist_bf16 = f32_to_bf16(distance);

    if hot.len >= l_max && dist_bf16 >= hot.farthest_dist {
        return false;
    }

    if let Some(idx) = find_hash_simd(&cold.hashes, hot.len, hash) {
        if dist_bf16 < cold.distances[idx] {
            let was_farthest = idx == hot.farthest_idx as usize;
            cold.neighbors[idx] = neighbor;
            cold.distances[idx] = dist_bf16;
            if was_farthest {
                update_farthest(hot, cold);
            }
            return true;
        }
        return false;
    }

    if hot.len < l_max {
        let new_idx = hot.len as usize;
        cold.hashes[new_idx] = hash;
        cold.distances[new_idx] = dist_bf16;
        cold.neighbors[new_idx] = neighbor;
        hot.len += 1;
        if dist_bf16 >= hot.farthest_dist {
            hot.farthest_dist = dist_bf16;
            hot.farthest_idx = new_idx as u8;
        }
        return true;
    }

    if dist_bf16 < hot.farthest_dist {
        let idx = hot.farthest_idx as usize;
        cold.hashes[idx] = hash;
        cold.distances[idx] = dist_bf16;
        cold.neighbors[idx] = neighbor;
        update_farthest(hot, cold);
        return true;
    }
    false
}

fn get_neighbors_saturated(hot: &HotSlot, cold: &ColdSlot, max_degree: usize) -> Vec<(u32, f32)> {
    let n = hot.len as usize;
    let mut tmp: Vec<(u32, u16)> = (0..n)
        .map(|i| (cold.neighbors[i], cold.distances[i]))
        .collect();
    tmp.sort_unstable_by_key(|&(_, d)| d);
    tmp.truncate(max_degree);
    tmp.into_iter().map(|(id, d)| (id, bf16_to_f32(d))).collect()
}

// ─── Test-only thin wrapper preserving the old HashPruneReservoir API ─────────

#[cfg(test)]
pub(crate) struct HashPruneReservoir {
    hot: HotSlot,
    cold: ColdSlot,
    l_max: u8,
}

#[cfg(test)]
impl HashPruneReservoir {
    pub(crate) fn new(l_max: usize) -> Self {
        assert!(l_max <= L_MAX_MAX);
        Self {
            hot: HotSlot::new_empty(),
            cold: ColdSlot {
                hashes: [0; L_MAX_MAX],
                distances: [0; L_MAX_MAX],
                neighbors: [0; L_MAX_MAX],
            },
            l_max: l_max as u8,
        }
    }

    pub(crate) fn new_lazy(l_max: usize) -> Self {
        Self::new(l_max)
    }

    pub fn insert(&mut self, hash: u16, neighbor: u32, distance: f32) -> bool {
        insert_locked(
            &mut self.hot,
            &mut self.cold,
            hash,
            neighbor,
            distance,
            self.l_max,
        )
    }

    pub fn get_neighbors_saturated(&self, max_degree: usize) -> Vec<(u32, f32)> {
        get_neighbors_saturated(&self.hot, &self.cold, max_degree)
    }

    pub(crate) fn get_neighbors_sorted(&self) -> Vec<(u32, f32)> {
        get_neighbors_saturated(&self.hot, &self.cold, L_MAX_MAX)
    }

    pub(crate) fn len(&self) -> usize {
        self.hot.len as usize
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.hot.len == 0
    }
}

// ─── HashPrune ────────────────────────────────────────────────────────────────

pub struct HashPrune {
    hot: Vec<HotSlot>,
    cold: Vec<ColdSlot>,
    sketches: LshSketches,
    max_degree: usize,
    l_max: usize,
}

// SAFETY: HotSlot has interior mutability via RawMutex. ColdSlot is plain data
// guarded by HotSlot[i].lock per index. Disjoint-index parallel access is safe.
unsafe impl Send for HashPrune {}
unsafe impl Sync for HashPrune {}

impl HashPrune {
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

    pub(crate) fn from_sketches(
        sketches: LshSketches,
        npoints: usize,
        l_max: usize,
        max_degree: usize,
    ) -> Self {
        assert!(
            l_max <= L_MAX_MAX,
            "AoSoA HashPrune requires l_max ≤ {}, got {}",
            L_MAX_MAX,
            l_max
        );

        let t1 = std::time::Instant::now();

        // Hot slab: one HotSlot per point, contiguous. 160 MB at 10M.
        let mut hot: Vec<HotSlot> = Vec::with_capacity(npoints);
        for _ in 0..npoints {
            hot.push(HotSlot::new_empty());
        }

        // Cold slab: 512 B × npoints. 5 GB at 10M. Plain bit-pattern type,
        // zeros are valid → use zeroed-page allocator path (memset of fresh
        // mmap'd pages is "free" via lazy faulting).
        let cold: Vec<ColdSlot> = {
            let mut v: Vec<ColdSlot> = Vec::with_capacity(npoints);
            // SAFETY: ColdSlot is repr(C) of u16/u32 arrays; all-zeros is a
            // valid bit pattern. write_bytes covers exactly capacity*sizeof.
            unsafe {
                std::ptr::write_bytes(v.as_mut_ptr(), 0, npoints);
                v.set_len(npoints);
            }
            v
        };

        // Both slabs contiguous → madvise hugepages so DTLB pressure scales
        // with 2 MB pages instead of 4 KB.
        #[cfg(target_os = "linux")]
        {
            let hot_bytes = hot.len() * std::mem::size_of::<HotSlot>();
            let cold_bytes = cold.len() * std::mem::size_of::<ColdSlot>();
            // SAFETY: as_ptr() points at each Vec's contiguous backing alloc;
            // bytes matches its capacity. madvise non-fatal on failure.
            unsafe {
                if hot_bytes > 2 * 1024 * 1024 {
                    libc::madvise(
                        hot.as_ptr() as *mut libc::c_void,
                        hot_bytes,
                        libc::MADV_HUGEPAGE,
                    );
                }
                if cold_bytes > 2 * 1024 * 1024 {
                    libc::madvise(
                        cold.as_ptr() as *mut libc::c_void,
                        cold_bytes,
                        libc::MADV_HUGEPAGE,
                    );
                }
            }
        }

        tracing::debug!(
            elapsed_secs = t1.elapsed().as_secs_f64(),
            "reservoir allocation"
        );

        Self {
            hot,
            cold,
            sketches,
            max_degree,
            l_max,
        }
    }

    /// SAFETY: idx must be in bounds for both hot and cold.
    #[inline]
    unsafe fn lock_slot(&self, idx: usize) -> (*mut HotSlot, *mut ColdSlot) {
        let hot_ptr = (self.hot.as_ptr() as *mut HotSlot).add(idx);
        let cold_ptr = (self.cold.as_ptr() as *mut ColdSlot).add(idx);
        (*hot_ptr).lock.lock();
        (hot_ptr, cold_ptr)
    }

    /// SAFETY: caller must hold the lock at idx.
    #[inline]
    unsafe fn unlock_slot(&self, idx: usize) {
        let hot_ptr = (self.hot.as_ptr() as *mut HotSlot).add(idx);
        (*hot_ptr).lock.unlock();
    }

    #[inline(always)]
    fn with_locked<R>(&self, idx: usize, f: impl FnOnce(&mut HotSlot, &mut ColdSlot) -> R) -> R {
        struct UnlockOnDrop<'a> {
            hp: &'a HashPrune,
            idx: usize,
        }
        impl<'a> Drop for UnlockOnDrop<'a> {
            fn drop(&mut self) {
                unsafe {
                    self.hp.unlock_slot(self.idx);
                }
            }
        }
        debug_assert!(idx < self.hot.len());
        // SAFETY: bounds-checked above; UnlockOnDrop unlocks on panic.
        let (hot_ptr, cold_ptr) = unsafe { self.lock_slot(idx) };
        let _guard = UnlockOnDrop { hp: self, idx };
        unsafe { f(&mut *hot_ptr, &mut *cold_ptr) }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn add_edge(&self, p: usize, c: usize, distance: f32) {
        let hash = self.sketches.relative_hash(p, c);
        let l_max = self.l_max as u8;
        self.with_locked(p, |hot, cold| {
            insert_locked(hot, cold, hash, c as u32, distance, l_max);
        });
    }

    pub fn add_edges_batched(&self, edges: &[crate::leaf_build::Edge]) {
        if edges.is_empty() {
            return;
        }
        let l_max = self.l_max as u8;

        let mut i = 0;
        while i < edges.len() {
            let src = edges[i].src;
            let mut j = i + 1;
            while j < edges.len() && edges[j].src == src {
                j += 1;
            }
            self.with_locked(src as usize, |hot, cold| {
                for edge in &edges[i..j] {
                    let hash = self
                        .sketches
                        .relative_hash(edge.src as usize, edge.dst as usize);
                    insert_locked(hot, cold, hash, edge.dst, edge.distance, l_max);
                }
            });
            i = j;
        }
    }

    pub fn add_edges_grouped(
        &self,
        group_starts: &[u32],
        group_data: &[(u32, f32)],
        local_indices: &[u32],
    ) {
        if group_data.is_empty() {
            return;
        }
        let n = local_indices.len();
        let l_max = self.l_max as u8;
        debug_assert!(group_starts.len() >= n + 1);

        for local_src in 0..n {
            let start = group_starts[local_src] as usize;
            let end = group_starts[local_src + 1] as usize;
            if start == end {
                continue;
            }
            let global_src = local_indices[local_src] as usize;
            self.with_locked(global_src, |hot, cold| {
                for &(dst_local, dist) in &group_data[start..end] {
                    let global_dst = local_indices[dst_local as usize];
                    let hash = self
                        .sketches
                        .relative_hash(global_src, global_dst as usize);
                    insert_locked(hot, cold, hash, global_dst, dist, l_max);
                }
            });
        }
    }

    pub fn add_edges_grouped_local_sketches(
        &self,
        group_starts: &[u32],
        group_data: &[(u32, f32)],
        local_indices: &[u32],
        local_sketches: &[f32],
    ) {
        if group_data.is_empty() {
            return;
        }
        let n = local_indices.len();
        let m = self.sketches.num_planes();
        let l_max = self.l_max as u8;
        debug_assert!(group_starts.len() >= n + 1);
        debug_assert!(local_sketches.len() >= n * m);

        for local_src in 0..n {
            let start = group_starts[local_src] as usize;
            let end = group_starts[local_src + 1] as usize;
            if start == end {
                continue;
            }
            let global_src = local_indices[local_src] as usize;

            // Prefetch NEXT non-empty source's hot+cold slots (cf. Phase 3
            // prefetch on prior AoS layout — same idea, narrower targets).
            #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
            {
                use std::arch::x86_64::*;
                if local_src + 1 < n {
                    let mut nxt = local_src + 1;
                    while nxt < n && group_starts[nxt] == group_starts[nxt + 1] {
                        nxt += 1;
                    }
                    if nxt < n {
                        let nxt_global = local_indices[nxt] as usize;
                        // SAFETY: nxt_global < npoints; both slabs in bounds.
                        unsafe {
                            let hot_p = self.hot.as_ptr().add(nxt_global) as *const i8;
                            _mm_prefetch::<{ _MM_HINT_T0 }>(hot_p);
                            let cold_p = self.cold.as_ptr().add(nxt_global) as *const i8;
                            _mm_prefetch::<{ _MM_HINT_T0 }>(cold_p);
                            _mm_prefetch::<{ _MM_HINT_T0 }>(cold_p.add(64));
                        }
                    }
                }
            }

            let src_sketch = &local_sketches[local_src * m..(local_src + 1) * m];
            self.with_locked(global_src, |hot, cold| {
                for &(dst_local, dist) in &group_data[start..end] {
                    let global_dst = local_indices[dst_local as usize];
                    let dst_sketch =
                        &local_sketches[dst_local as usize * m..(dst_local as usize + 1) * m];
                    let hash: u16;
                    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
                    {
                        use std::arch::x86_64::*;
                        let kmask: u16 = if m >= 16 { 0xFFFF } else { (1u16 << m) - 1 };
                        // SAFETY: AVX-512 cfg-gated; masked load bounded by kmask.
                        unsafe {
                            let dst_v = _mm512_maskz_loadu_ps(kmask, dst_sketch.as_ptr());
                            let src_v = _mm512_maskz_loadu_ps(kmask, src_sketch.as_ptr());
                            let diff = _mm512_sub_ps(dst_v, src_v);
                            let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(diff, _mm512_setzero_ps());
                            hash = mask & kmask;
                        }
                    }
                    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
                    {
                        let mut h: u16 = 0;
                        for j in 0..m {
                            let diff = dst_sketch[j] - src_sketch[j];
                            let bit = ((!diff.is_sign_negative()) as u16) << j;
                            h |= bit;
                        }
                        hash = h;
                    }
                    insert_locked(hot, cold, hash, global_dst, dist, l_max);
                }
            });
        }
    }

    pub fn gather_sketches_into(&self, indices: &[u32], out: &mut [f32]) {
        let m = self.sketches.num_planes();
        let src = self.sketches.sketches();
        debug_assert_eq!(out.len(), indices.len() * m);
        for (i, &idx) in indices.iter().enumerate() {
            let g = idx as usize;
            out[i * m..(i + 1) * m].copy_from_slice(&src[g * m..(g + 1) * m]);
        }
    }

    pub fn num_planes(&self) -> usize {
        self.sketches.num_planes()
    }

    pub fn extract_graph(self) -> Vec<Vec<u32>> {
        let max_degree = self.max_degree;
        drop(self.sketches);
        let HashPrune { hot, cold, .. } = self;
        (0..hot.len())
            .into_par_iter()
            .map(|i| {
                let mut nbrs = get_neighbors_saturated(&hot[i], &cold[i], max_degree);
                nbrs.truncate(max_degree);
                nbrs.into_iter().map(|(id, _)| id).collect()
            })
            .collect_installed()
    }

    pub fn extract_graph_for_prune(self) -> Vec<Vec<(u32, f32)>> {
        let l_max = self.l_max;
        drop(self.sketches);
        let HashPrune { hot, cold, .. } = self;
        (0..hot.len())
            .into_par_iter()
            .map(|i| get_neighbors_saturated(&hot[i], &cold[i], l_max))
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

        assert!(reservoir.insert(0, 1, 1.0));
        assert!(reservoir.insert(1, 2, 2.0));
        assert!(reservoir.insert(2, 3, 3.0));
        assert_eq!(reservoir.len(), 3);

        assert!(reservoir.insert(3, 4, 0.5));
        assert_eq!(reservoir.len(), 3);

        let neighbors = reservoir.get_neighbors_sorted();
        assert!(!neighbors.iter().any(|(id, _)| *id == 3));
        assert!(neighbors.iter().any(|(id, _)| *id == 4));
    }

    #[test]
    fn test_reservoir_same_hash_keeps_closer() {
        let mut reservoir = HashPruneReservoir::new(10);

        assert!(reservoir.insert(0, 1, 2.0));
        assert_eq!(reservoir.len(), 1);

        assert!(reservoir.insert(0, 2, 1.0));
        assert_eq!(reservoir.len(), 1);

        let neighbors = reservoir.get_neighbors_sorted();
        assert_eq!(neighbors[0].0, 2);
        assert_eq!(neighbors[0].1, 1.0);

        assert!(!reservoir.insert(0, 3, 5.0));
        assert_eq!(reservoir.len(), 1);
    }

    #[test]
    fn test_lsh_sketches() {
        let data = vec![
            1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0,
        ];
        let sketches = sketches_from_data(&data, 4, 2, 4, 42);

        let h00 = sketches.relative_hash(0, 0);
        assert_eq!(h00, (1u16 << 4) - 1);

        let h01 = sketches.relative_hash(0, 1);
        let h02 = sketches.relative_hash(0, 2);
        let _ = (h01, h02);
    }

    #[test]
    fn test_hash_prune_end_to_end() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let hp = HashPrune::new(&data, 4, 2, 4, 10, 3, 42);

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
        let hp = HashPrune::new(&data, 4, 2, 4, 10, 2, 42);
        hp.add_edge(0, 1, 1.0);
        hp.add_edge(0, 2, 1.0);
        hp.add_edge(0, 3, 1.414);
        hp.add_edge(1, 0, 1.0);
        hp.add_edge(2, 0, 1.0);
        hp.add_edge(3, 0, 1.414);

        let full = hp.extract_graph_for_prune();
        assert_eq!(full.len(), 4);
        assert!(!full[0].is_empty(), "node 0 should have neighbors");
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
        let mut res = HashPruneReservoir::new(3);
        res.insert(0, 10, 5.0);
        res.insert(1, 11, 4.0);
        res.insert(2, 12, 3.0);
        assert!(res.insert(3, 13, 2.0));
        assert!(res.insert(4, 14, 1.0));
        let neighbors = res.get_neighbors_sorted();
        assert_eq!(neighbors.len(), 3);
        for &(_, d) in &neighbors {
            assert!(d <= 3.1, "expected dist <= 3.0, got {}", d);
        }
    }

    #[test]
    fn test_reservoir_farthest_insert_before_farthest_idx() {
        let mut res = HashPruneReservoir::new(4);
        res.insert(5, 1, 1.0);
        res.insert(10, 2, 3.0);
        res.insert(15, 3, 2.0);
        res.insert(3, 4, 0.5);
        let neighbors = res.get_neighbors_sorted();
        assert_eq!(neighbors.len(), 4);
        assert_eq!(neighbors[0].0, 4);
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
