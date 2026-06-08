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

use crate::cpu_dispatch::{tier, SimdTier};
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

/// Scalar fallback for `find_hash`. Linear scan, no SIMD.
#[inline(always)]
fn find_hash_scalar(entries: &[ReservoirEntry], hash: u16) -> Option<usize> {
    for (i, e) in entries.iter().enumerate() {
        if e.hash == hash {
            return Some(i);
        }
    }
    None
}

/// AoSoA-packed AVX-512 `find_hash`: scans `&[u16]` 32 hashes per
/// `_mm512_cmpeq_epi16_mask`. Microbench shows 1.6-2× speedup over the
/// 8-byte-stride scan of [`find_hash_avx512`] (3.95ns vs 6.16ns at l_max=64;
/// 4.55ns vs 9.32ns at l_max=128).
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F + AVX-512BW.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn find_hash_packed_avx512(hashes: &[u16], target_hash: u16) -> Option<usize> {
    use std::arch::x86_64::*;
    let n = hashes.len();
    let target = _mm512_set1_epi16(target_hash as i16);
    let chunks = n / 32;
    // Fast path for n <= 64 (the designed l_max range): the per-chunk 32-bit
    // cmpeq masks fit one u64 with no shift overflow -> branch-free accumulate,
    // byte-identical to the original hot path (no regression at l_max <= 64).
    if n <= 64 {
        let mut found: u64 = 0;
        for chunk in 0..chunks {
            let base = chunk * 32;
            let data = _mm512_loadu_si512(hashes.as_ptr().add(base) as *const __m512i);
            let cmp = _mm512_cmpeq_epi16_mask(data, target);
            found |= (cmp as u64) << (chunk * 32);
        }
        let tail = n - chunks * 32;
        if tail > 0 {
            let kmask: u32 = (1u32 << tail) - 1;
            let base = chunks * 32;
            let data = _mm512_maskz_loadu_epi16(kmask, hashes.as_ptr().add(base) as *const i16);
            let cmp = _mm512_cmpeq_epi16_mask(data, target) & kmask;
            found |= (cmp as u64) << base;
        }
        return if found != 0 {
            Some(found.trailing_zeros() as usize)
        } else {
            None
        };
    }
    // n > 64: a single u64 accumulator would overflow `<< (chunk*32)` (the old
    // bug mis-indexed any match past entry 64). Use a branch-free running-min
    // first-match instead. Correct for any n.
    let mut result: u32 = u32::MAX;
    for chunk in 0..chunks {
        let base = (chunk * 32) as u32;
        let data = _mm512_loadu_si512(hashes.as_ptr().add(chunk * 32) as *const __m512i);
        let cmp = _mm512_cmpeq_epi16_mask(data, target);
        let cand = if cmp != 0 { base + cmp.trailing_zeros() } else { u32::MAX };
        result = result.min(cand);
    }
    let tail = n - chunks * 32;
    if tail > 0 {
        let kmask: u32 = (1u32 << tail) - 1;
        let base = (chunks * 32) as u32;
        let data = _mm512_maskz_loadu_epi16(kmask, hashes.as_ptr().add(chunks * 32) as *const i16);
        let cmp = _mm512_cmpeq_epi16_mask(data, target) & kmask;
        let cand = if cmp != 0 { base + cmp.trailing_zeros() } else { u32::MAX };
        result = result.min(cand);
    }
    if result != u32::MAX {
        Some(result as usize)
    } else {
        None
    }
}

/// AoSoA-packed AVX-2 `find_hash`: scans `&[u16]` 16 hashes per `_mm256_cmpeq_epi16`.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_hash_packed_avx2(hashes: &[u16], target_hash: u16) -> Option<usize> {
    use std::arch::x86_64::*;
    let n = hashes.len();
    let target = _mm256_set1_epi16(target_hash as i16);
    let chunks = n / 16;
    // Branch-free first-match: cmov-record the lowest matching absolute index.
    // The previous `found |= bits << (chunk*16)` / `1u64 << i` overflowed the u64
    // accumulator for n > 64 (l_max=128), mis-indexing any match past entry 64.
    let mut result: u32 = u32::MAX;
    for chunk in 0..chunks {
        let base = (chunk * 16) as u32;
        let data = _mm256_loadu_si256(hashes.as_ptr().add(chunk * 16) as *const __m256i);
        let cmp = _mm256_cmpeq_epi16(data, target);
        let bytes = _mm256_movemask_epi8(cmp) as u32;
        let mut bits: u32 = 0;
        let mut i = 0usize;
        while i < 16 {
            bits |= ((bytes >> (i * 2)) & 1) << i;
            i += 1;
        }
        let cand = if bits != 0 { base + bits.trailing_zeros() } else { u32::MAX };
        result = result.min(cand);
    }
    // Scalar tail
    for i in (chunks * 16)..n {
        if *hashes.get_unchecked(i) == target_hash {
            result = result.min(i as u32);
        }
    }
    if result != u32::MAX {
        Some(result as usize)
    } else {
        None
    }
}

/// AoSoA-packed scalar `find_hash`.
#[inline(always)]
fn find_hash_packed_scalar(hashes: &[u16], target_hash: u16) -> Option<usize> {
    hashes.iter().position(|&h| h == target_hash)
}

/// AVX-512 `find_hash`: 8 entries per cmpeq mask, branch-free OR over all chunks.
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F. The `#[target_feature]`
/// attribute promises the compiler we are running with that feature set.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(dead_code)]
unsafe fn find_hash_avx512(entries: &[ReservoirEntry], hash: u16) -> Option<usize> {
    use std::arch::x86_64::*;
    let n = entries.len();
    if n >= 8 {
        // SAFETY: AVX-512F enabled by target_feature. ReservoirEntry is
        // `#[repr(C)]` with size 8 = sizeof(u64), so the u64 cast aliases bytes
        // legally. `ptr.add(base)` stays within `entries` since `base + 8 ≤ n`.
        //
        // Branch-free: scan ALL chunks unconditionally, OR the per-chunk
        // 8-bit cmpeq masks into a single 64-bit "found" mask (8 bits per
        // chunk × up to 8 chunks for l_max=64). Then trailing_zeros() of
        // the 64-bit value gives the absolute lane index.
        //
        // Eliminates the data-dependent `if cmp != 0 { return }` per chunk —
        // that branch was the dominant source of HP's 19% bad-spec slots
        // (per c4-48 PMU profile). Cost: scan all chunks even when an early
        // chunk matches; benefit: zero misprediction.
        let ptr = entries.as_ptr() as *const u64;
        let target = _mm512_set1_epi64(((hash as u64) << 32) as i64);
        let mask = _mm512_set1_epi64(0x0000FFFF00000000u64 as i64);
        let chunks = n / 8;
        let mut found: u64 = 0;
        for chunk in 0..chunks {
            let base = chunk * 8;
            let data = _mm512_loadu_si512(ptr.add(base) as *const __m512i);
            let masked = _mm512_and_si512(data, mask);
            let cmp = _mm512_cmpeq_epi64_mask(masked, target);
            found |= (cmp as u64) << (chunk * 8);
        }
        // Tail (n % 8 entries): use a final masked load to fold into `found`.
        let tail = n - chunks * 8;
        if tail > 0 {
            let kmask: u8 = (1u8 << tail) - 1;
            let base = chunks * 8;
            let data = _mm512_maskz_loadu_epi64(kmask, ptr.add(base) as *const i64);
            let masked = _mm512_and_si512(data, mask);
            let cmp = _mm512_cmpeq_epi64_mask(masked, target) & kmask;
            found |= (cmp as u64) << base;
        }
        if found != 0 {
            return Some(found.trailing_zeros() as usize);
        }
        return None;
    }
    // Small-n path: scalar.
    find_hash_scalar(entries, hash)
}

/// AVX2 `find_hash`: 4 entries per cmpeq mask, branch-free OR over all chunks.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_hash_avx2(entries: &[ReservoirEntry], hash: u16) -> Option<usize> {
    use std::arch::x86_64::*;
    let n = entries.len();
    if n >= 4 {
        // SAFETY: AVX2 enabled by target_feature; ReservoirEntry is `#[repr(C)]`
        // size 8, so the u64 cast aliases legally. `ptr.add(base)` stays in
        // bounds by `base + 4 ≤ n`. Tail `get_unchecked(i)` has `i < n`.
        let ptr = entries.as_ptr() as *const u64;
        let target = _mm256_set1_epi64x(((hash as u64) << 32) as i64);
        let mask = _mm256_set1_epi64x(0x0000FFFF00000000u64 as i64);
        let chunks = n / 4;
        let mut found: u64 = 0;
        for chunk in 0..chunks {
            let base = chunk * 4;
            let data = _mm256_loadu_si256(ptr.add(base) as *const __m256i);
            let masked = _mm256_and_si256(data, mask);
            let cmp = _mm256_cmpeq_epi64(masked, target);
            // movemask_pd: 4 bits, one per 64-bit lane.
            let bits = _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) as u64;
            found |= bits << (chunk * 4);
        }
        for i in (chunks * 4)..n {
            if entries.get_unchecked(i).hash == hash {
                found |= 1u64 << i;
            }
        }
        if found != 0 {
            return Some(found.trailing_zeros() as usize);
        }
        return None;
    }
    find_hash_scalar(entries, hash)
}

/// Scalar relative_hash on local f32 sketches.
///
/// `relative_hash`: bit j = (dst[j] - src[j] >= 0). Branch-free: sign bit is 0
/// when diff >= 0 (incl +0). `m` is the number of planes (up to 16).
#[inline(always)]
fn relative_hash_local_scalar(dst: &[f32], src: &[f32], m: usize) -> u16 {
    let mut h: u16 = 0;
    for j in 0..m {
        let diff = dst[j] - src[j];
        let bit = ((!diff.is_sign_negative()) as u16) << j;
        h |= bit;
    }
    h
}

/// AVX-512 relative_hash: one masked sub + cmp_ge_mask gives the bit pattern
/// directly.
///
/// # Safety
/// Caller must ensure the CPU supports AVX-512F. `dst` and `src` must each
/// have at least `m.min(16)` valid f32 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn relative_hash_local_avx512(dst: &[f32], src: &[f32], m: usize) -> u16 {
    use std::arch::x86_64::*;
    // SAFETY: AVX-512F enabled. We bound the load to `m` lanes via kmask;
    // planes is validated 1..=16 by PiPNNConfig.
    let kmask: u16 = if m >= 16 { 0xFFFF } else { (1u16 << m) - 1 };
    let dst_v = _mm512_maskz_loadu_ps(kmask, dst.as_ptr());
    let src_v = _mm512_maskz_loadu_ps(kmask, src.as_ptr());
    let diff = _mm512_sub_ps(dst_v, src_v);
    let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(diff, _mm512_setzero_ps());
    mask & kmask
}

/// AVX2 relative_hash: 8-lane SIMD via `_mm256_maskload_ps` +
/// `_mm256_movemask_ps`. Splits up-to-16 planes into low-8 and optional
/// high-(m-8) halves.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn relative_hash_local_avx2(dst: &[f32], src: &[f32], m: usize) -> u16 {
    use std::arch::x86_64::*;
    // SAFETY: AVX2 enabled; maskload returns 0 for masked-off lanes so reading
    // past the end is safe.
    let make_mask = |count: usize| -> __m256i {
        // -1 for active lanes, 0 for inactive.
        let m_arr: [i32; 8] = std::array::from_fn(|k| if k < count { -1 } else { 0 });
        _mm256_loadu_si256(m_arr.as_ptr() as *const __m256i)
    };
    let lo_count = m.min(8);
    let mask_lo = make_mask(lo_count);
    let dst_lo = _mm256_maskload_ps(dst.as_ptr(), mask_lo);
    let src_lo = _mm256_maskload_ps(src.as_ptr(), mask_lo);
    let diff_lo = _mm256_sub_ps(dst_lo, src_lo);
    let cmp_lo = _mm256_cmp_ps::<_CMP_GE_OQ>(diff_lo, _mm256_setzero_ps());
    let bits_lo = _mm256_movemask_ps(cmp_lo) as u16;
    let lo_kmask: u16 = (1u16 << lo_count) - 1;
    let mut h = bits_lo & lo_kmask;
    if m > 8 {
        let hi_count = m - 8;
        let mask_hi = make_mask(hi_count);
        let dst_hi = _mm256_maskload_ps(dst.as_ptr().add(8), mask_hi);
        let src_hi = _mm256_maskload_ps(src.as_ptr().add(8), mask_hi);
        let diff_hi = _mm256_sub_ps(dst_hi, src_hi);
        let cmp_hi = _mm256_cmp_ps::<_CMP_GE_OQ>(diff_hi, _mm256_setzero_ps());
        let bits_hi = _mm256_movemask_ps(cmp_hi) as u16;
        let hi_kmask: u16 = (1u16 << hi_count) - 1;
        h |= (bits_hi & hi_kmask) << 8;
    }
    h
}

/// HashPrune reservoir for a single point.
///
/// Stores at most `l_max` entries keyed by their hash bucket. Insertion is O(1)
/// (push or swap_remove); lookups are O(l_max) linear scan, vectorised by
/// [`find_hash`] (AVX-512 8-way / AVX2 4-way / scalar fallback). The farthest
/// entry is cached so a full-reservoir reject path is O(1).
///
/// AoSoA hot slab: `hashes` is a parallel `Vec<u16>` kept in sync with
/// `entries`. find_hash scans this packed array via 32-way AVX-512 SIMD
/// (1.6-2× faster than scanning the 8-byte `entries` Vec, per microbench).
/// The hash field in `ReservoirEntry` is now redundant but kept for now to
/// minimise downstream churn.
pub(crate) struct HashPruneReservoir {
    entries: Vec<ReservoirEntry>,
    hashes: Vec<u16>,
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
            hashes: Vec::with_capacity(l_max),
            l_max,
            farthest_dist: 0,
            farthest_idx: 0,
        }
    }

    /// Create a reservoir without pre-allocating capacity.
    pub(crate) fn new_lazy(l_max: usize) -> Self {
        Self {
            entries: Vec::new(),
            hashes: Vec::new(),
            l_max,
            farthest_dist: 0,
            farthest_idx: 0,
        }
    }

    /// Update the cached farthest entry.
    #[inline]
    fn update_farthest(&mut self) {
        if self.entries.is_empty() {
            self.farthest_dist = 0;
            self.farthest_idx = 0;
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if matches!(tier(), SimdTier::Avx512) {
            unsafe { self.update_farthest_avx512() };
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

    /// SIMD argmax of the reservoir's farthest (max) distance + its index.
    /// `distance` is bits 48..64 of each 8-byte ReservoirEntry, so we load 8
    /// entries per zmm as u64 and shift. Replaces the O(l_max) scalar rescan
    /// that runs on every full-reservoir eviction (the per-eviction cost that
    /// scales linearly with l_max).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn update_farthest_avx512(&mut self) {
        use std::arch::x86_64::*;
        let n = self.entries.len();
        let ptr = self.entries.as_ptr() as *const u64;
        let chunks = n / 8;
        let mut vmax = _mm512_setzero_si512();
        for c in 0..chunks {
            let v = _mm512_loadu_si512(ptr.add(c * 8) as *const __m512i);
            vmax = _mm512_max_epu64(vmax, _mm512_srli_epi64::<48>(v));
        }
        let mut max_dist = _mm512_reduce_max_epu64(vmax) as u16;
        for i in (chunks * 8)..n {
            let d = self.entries.get_unchecked(i).distance;
            if d > max_dist {
                max_dist = d;
            }
        }
        let tgt = _mm512_set1_epi64(max_dist as i64);
        let mut max_idx = 0usize;
        'find: {
            for c in 0..chunks {
                let v = _mm512_loadu_si512(ptr.add(c * 8) as *const __m512i);
                let m = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64::<48>(v), tgt);
                if m != 0 {
                    max_idx = c * 8 + m.trailing_zeros() as usize;
                    break 'find;
                }
            }
            for i in (chunks * 8)..n {
                if self.entries.get_unchecked(i).distance == max_dist {
                    max_idx = i;
                    break 'find;
                }
            }
        }
        self.farthest_dist = max_dist;
        self.farthest_idx = max_idx;
    }

    /// Generic insert parameterized by the `find_hash` strategy.
    /// Tier-specialized wrappers (`insert_avx512`, `insert_avx2`, `insert_scalar`)
    /// pass the matching SIMD `find_hash` so the call inlines without a runtime
    /// tier check.
    ///
    /// Unsorted storage: find_hash is O(l_max) linear scan but insert/evict
    /// are O(1) via push/swap_remove.
    ///
    /// Perf: `farthest_dist` / `farthest_idx` are hoisted into locals to keep
    /// them in registers across the `find` call. The `&mut self.entries`
    /// borrow inside `find` previously invalidated their liveness, causing a
    /// stack-spill that perf annotate showed as 24% of HP cycles.
    #[inline(always)]
    fn insert_with<F: FnOnce(&[u16], u16) -> Option<usize>>(
        &mut self,
        hash: u16,
        neighbor: u32,
        distance: f32,
        find: F,
    ) -> bool {
        let dist_bf16 = f32_to_bf16(distance);

        let mut local_far_dist = self.farthest_dist;
        let mut local_far_idx = self.farthest_idx;

        if self.entries.len() >= self.l_max && dist_bf16 >= local_far_dist {
            return false;
        }

        // find() now scans the AoSoA hot `hashes` slab (1.6-2× faster).
        if let Some(idx) = find(&self.hashes, hash) {
            if dist_bf16 < self.entries[idx].distance {
                let was_farthest = idx == local_far_idx;
                self.entries[idx].neighbor = neighbor;
                self.entries[idx].distance = dist_bf16;
                // self.hashes[idx] stays the same (hash bucket unchanged).
                if was_farthest {
                    self.update_farthest();
                }
                return true;
            }
            return false;
        }

        if self.entries.len() < self.l_max {
            if self.entries.is_empty() {
                self.entries.reserve_exact(self.l_max);
                self.hashes.reserve_exact(self.l_max);
            }
            let new_idx = self.entries.len();
            self.entries.push(ReservoirEntry {
                neighbor,
                distance: dist_bf16,
                hash,
            });
            self.hashes.push(hash);
            if dist_bf16 >= local_far_dist {
                local_far_dist = dist_bf16;
                local_far_idx = new_idx;
            }
            self.farthest_dist = local_far_dist;
            self.farthest_idx = local_far_idx;
            return true;
        }

        if dist_bf16 < local_far_dist {
            self.entries[local_far_idx] = ReservoirEntry {
                neighbor,
                distance: dist_bf16,
                hash,
            };
            self.hashes[local_far_idx] = hash;
            self.update_farthest();
            return true;
        }

        false
    }

    /// Insert with runtime-dispatched find_hash. Convenience API for callers
    /// that don't already know the SIMD tier (tests, single-edge add_edge).
    /// Hot per-edge paths in `add_edges_*` use `insert_with` with a
    /// tier-specialized `find_hash` to hoist the dispatch out of the inner
    /// loop.
    #[allow(dead_code)]
    #[inline(always)]
    pub fn insert(&mut self, hash: u16, neighbor: u32, distance: f32) -> bool {
        match tier() {
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx512 => self
                .insert_with(hash, neighbor, distance, |h, t| unsafe {
                    find_hash_packed_avx512(h, t)
                }),
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx2 => self
                .insert_with(hash, neighbor, distance, |h, t| unsafe {
                    find_hash_packed_avx2(h, t)
                }),
            _ => self.insert_with(hash, neighbor, distance, find_hash_packed_scalar),
        }
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
    ///
    /// Dispatches on the runtime SIMD tier ONCE at entry; the per-edge inner
    /// loop is monomorphized to call the tier-matched `find_hash` directly,
    /// avoiding a per-edge runtime tier check.
    pub fn add_edges_batched(&self, edges: &[crate::leaf_build::Edge]) {
        if edges.is_empty() {
            return;
        }
        match tier() {
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx512 => unsafe { self.add_edges_batched_inner_avx512(edges) },
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx2 => unsafe { self.add_edges_batched_inner_avx2(edges) },
            _ => self.add_edges_batched_inner_scalar(edges),
        }
    }

    /// Generic inner body for `add_edges_batched`, parameterized by `find_hash`.
    #[inline(always)]
    fn add_edges_batched_inner<F>(&self, edges: &[crate::leaf_build::Edge], find: F)
    where
        F: Copy + Fn(&[u16], u16) -> Option<usize>,
    {
        // Cache last reservoir lock to avoid redundant lock ops for consecutive
        // edges with the same source.
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
            reservoir.insert_with(hash, edge.dst, edge.distance, find);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn add_edges_batched_inner_avx512(&self, edges: &[crate::leaf_build::Edge]) {
        self.add_edges_batched_inner(edges, |e, h| find_hash_packed_avx512(e, h));
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_edges_batched_inner_avx2(&self, edges: &[crate::leaf_build::Edge]) {
        self.add_edges_batched_inner(edges, |e, h| find_hash_packed_avx2(e, h));
    }

    fn add_edges_batched_inner_scalar(&self, edges: &[crate::leaf_build::Edge]) {
        self.add_edges_batched_inner(edges, find_hash_packed_scalar);
    }

    /// Insert leaf edges in CSR-grouped form.
    ///
    /// `group_starts[i]..group_starts[i+1]` indexes into `group_data` for
    /// edges with local source `i`. Each entry is `(local_dst, distance)`.
    /// `local_indices` maps local index → global point id.
    ///
    /// One lock acquisition per source — reservoir + sketch loaded into cache
    /// once and used for all edges of that source, eliminating ~5x of HP
    /// insert cost versus interleaved-source insertion.
    ///
    /// Dispatches on the runtime SIMD tier ONCE at entry.
    pub fn add_edges_grouped(
        &self,
        group_starts: &[u32],
        group_data: &[(u32, f32)],
        local_indices: &[u32],
    ) {
        if group_data.is_empty() {
            return;
        }
        match tier() {
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx512 => unsafe {
                self.add_edges_grouped_inner_avx512(group_starts, group_data, local_indices)
            },
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx2 => unsafe {
                self.add_edges_grouped_inner_avx2(group_starts, group_data, local_indices)
            },
            _ => self.add_edges_grouped_inner_scalar(group_starts, group_data, local_indices),
        }
    }

    #[inline(always)]
    fn add_edges_grouped_inner<F>(
        &self,
        group_starts: &[u32],
        group_data: &[(u32, f32)],
        local_indices: &[u32],
        find: F,
    ) where
        F: Copy + Fn(&[u16], u16) -> Option<usize>,
    {
        let n = local_indices.len();
        debug_assert!(group_starts.len() >= n + 1);

        for local_src in 0..n {
            let start = group_starts[local_src] as usize;
            let end = group_starts[local_src + 1] as usize;
            if start == end {
                continue;
            }
            let global_src = local_indices[local_src] as usize;
            let mut reservoir = self.reservoirs[global_src].lock();
            // SAFETY: bounds checked by `group_starts[n] == group_data.len()`
            // (caller invariant — see build_leaf_with_buffers Pass 2).
            for &(dst_local, dist) in &group_data[start..end] {
                let global_dst = local_indices[dst_local as usize];
                let hash = self
                    .sketches
                    .relative_hash(global_src, global_dst as usize);
                reservoir.insert_with(hash, global_dst, dist, find);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn add_edges_grouped_inner_avx512(
        &self,
        group_starts: &[u32],
        group_data: &[(u32, f32)],
        local_indices: &[u32],
    ) {
        self.add_edges_grouped_inner(group_starts, group_data, local_indices, |e, h| {
            find_hash_packed_avx512(e, h)
        });
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_edges_grouped_inner_avx2(
        &self,
        group_starts: &[u32],
        group_data: &[(u32, f32)],
        local_indices: &[u32],
    ) {
        self.add_edges_grouped_inner(group_starts, group_data, local_indices, |e, h| {
            find_hash_packed_avx2(e, h)
        });
    }

    fn add_edges_grouped_inner_scalar(
        &self,
        group_starts: &[u32],
        group_data: &[(u32, f32)],
        local_indices: &[u32],
    ) {
        self.add_edges_grouped_inner(
            group_starts,
            group_data,
            local_indices,
            find_hash_packed_scalar,
        );
    }

    /// CSR-grouped insert that consumes a leaf-local sketches buffer.
    /// `local_sketches` is `n × num_planes` row-major (caller gathered it
    /// during the leaf's data gather). Hash compute happens against this
    /// L1-resident buffer instead of the multi-hundred-MB global sketches —
    /// eliminates the per-edge global sketches cache miss in HP insertion.
    ///
    /// Dispatches on the runtime SIMD tier ONCE at entry; both `relative_hash`
    /// and `find_hash` inside the per-edge loop run at the matched tier with
    /// no per-call runtime check (the `#[target_feature]` on the inner fn lets
    /// the compiler inline the SIMD intrinsics into the same hot kernel).
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
        match tier() {
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx512 => unsafe {
                self.add_edges_grouped_local_sketches_inner_avx512(
                    group_starts,
                    group_data,
                    local_indices,
                    local_sketches,
                )
            },
            #[cfg(target_arch = "x86_64")]
            SimdTier::Avx2 => unsafe {
                self.add_edges_grouped_local_sketches_inner_avx2(
                    group_starts,
                    group_data,
                    local_indices,
                    local_sketches,
                )
            },
            _ => self.add_edges_grouped_local_sketches_inner_scalar(
                group_starts,
                group_data,
                local_indices,
                local_sketches,
            ),
        }
    }

    /// Generic inner body for `add_edges_grouped_local_sketches`, parameterized
    /// by `find_hash` and `relative_hash_local`. The compiler monomorphizes per
    /// tier so both SIMD ops inline into the per-edge hot loop.
    #[inline(always)]
    fn add_edges_grouped_local_sketches_inner<F, H>(
        &self,
        group_starts: &[u32],
        group_data: &[(u32, f32)],
        local_indices: &[u32],
        local_sketches: &[f32],
        find: F,
        relhash: H,
    ) where
        F: Copy + Fn(&[u16], u16) -> Option<usize>,
        H: Copy + Fn(&[f32], &[f32], usize) -> u16,
    {
        let n = local_indices.len();
        let m = self.sketches.num_planes();
        debug_assert!(group_starts.len() >= n + 1);
        debug_assert!(local_sketches.len() >= n * m);

        for local_src in 0..n {
            let start = group_starts[local_src] as usize;
            let end = group_starts[local_src + 1] as usize;
            if start == end {
                continue;
            }
            let global_src = local_indices[local_src] as usize;
            let src_sketch = &local_sketches[local_src * m..(local_src + 1) * m];
            let mut reservoir = self.reservoirs[global_src].lock();
            for &(dst_local, dist) in &group_data[start..end] {
                let global_dst = local_indices[dst_local as usize];
                let dst_sketch =
                    &local_sketches[dst_local as usize * m..(dst_local as usize + 1) * m];
                // relative_hash: bit j = (dst[j] - src[j] >= 0). Branch-free
                // SIMD: compute diff in one masked sub, then cmp_ge_mask gives
                // the bit pattern directly. Eliminates the 12-iter data-
                // dependent `if >= 0` branch chain that contributed to HP's
                // bad-spec slot share (PMU profile).
                let hash = relhash(dst_sketch, src_sketch, m);
                reservoir.insert_with(hash, global_dst, dist, find);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f", enable = "avx512bw")]
    unsafe fn add_edges_grouped_local_sketches_inner_avx512(
        &self,
        group_starts: &[u32],
        group_data: &[(u32, f32)],
        local_indices: &[u32],
        local_sketches: &[f32],
    ) {
        // SAFETY: AVX-512F enabled by `#[target_feature]`; the inlined
        // `find_hash_avx512` and `relative_hash_local_avx512` calls share the
        // same feature scope, so the compiler emits the AVX-512 codegen
        // identically to the previous `#[cfg(target_feature)]` blocks.
        self.add_edges_grouped_local_sketches_inner(
            group_starts,
            group_data,
            local_indices,
            local_sketches,
            |e, h| find_hash_packed_avx512(e, h),
            |dst, src, m| relative_hash_local_avx512(dst, src, m),
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_edges_grouped_local_sketches_inner_avx2(
        &self,
        group_starts: &[u32],
        group_data: &[(u32, f32)],
        local_indices: &[u32],
        local_sketches: &[f32],
    ) {
        // SAFETY: AVX2 enabled by `#[target_feature]`.
        self.add_edges_grouped_local_sketches_inner(
            group_starts,
            group_data,
            local_indices,
            local_sketches,
            |e, h| find_hash_packed_avx2(e, h),
            |dst, src, m| relative_hash_local_avx2(dst, src, m),
        );
    }

    fn add_edges_grouped_local_sketches_inner_scalar(
        &self,
        group_starts: &[u32],
        group_data: &[(u32, f32)],
        local_indices: &[u32],
        local_sketches: &[f32],
    ) {
        self.add_edges_grouped_local_sketches_inner(
            group_starts,
            group_data,
            local_indices,
            local_sketches,
            find_hash_packed_scalar,
            relative_hash_local_scalar,
        );
    }

    /// Copy sketches for `indices` into `out` (length must be `indices.len() * num_planes`).
    /// Used by leaf build to pre-gather a small L1-resident sketches buffer.
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
