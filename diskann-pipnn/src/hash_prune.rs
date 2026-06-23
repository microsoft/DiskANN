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
//! `L_MAX_MAX` is hard-coded at 128 to match `PiPNNConfig` default.
//! Configs with `l_max > 128` are rejected at construction time.
//! `find_hash_simd` scans `L_MAX_MAX / 32 = 4` AVX-512 chunks (128 lanes)
//! with a u128 mask accumulator.

use parking_lot::lock_api::RawMutex as RawMutexTrait;

use crate::rayon_util::ParIterInstalled;
use diskann::utils::VectorRepr;
use diskann_vector::bf16::{bf16_to_f32, f32_to_bf16};
use diskann_vector::lsh::LshSketches;
use rayon::prelude::*;

/// Owned slab allocated via direct `mmap(MAP_PRIVATE | MAP_ANONYMOUS)`. The
/// kernel backs the range with its zero-page until first write, so we get
/// true lazy faulting for the AoSoA cold slabs — mimalloc's `eager_commit`
/// would otherwise pre-fault every page during construction (the +1.46 s
/// LSH-init tax measured pre-fix).
#[cfg(target_os = "linux")]
struct MmapSlab<T> {
    ptr: *mut T,
    len: usize,
}

#[cfg(target_os = "linux")]
unsafe impl<T: Send> Send for MmapSlab<T> {}
#[cfg(target_os = "linux")]
unsafe impl<T: Sync> Sync for MmapSlab<T> {}

#[cfg(target_os = "linux")]
impl<T> MmapSlab<T> {
    fn new_zeroed(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: std::ptr::NonNull::<T>::dangling().as_ptr(),
                len: 0,
            };
        }
        let bytes = len * std::mem::size_of::<T>();
        // SAFETY: MAP_ANONYMOUS gives a zero-backed VA region; PROT_RW makes
        // it readable/writable. Pages allocate on first write only.
        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            assert_ne!(
                ptr,
                libc::MAP_FAILED,
                "MmapSlab mmap failed for {} bytes",
                bytes
            );
            Self {
                ptr: ptr as *mut T,
                len,
            }
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline]
    fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

#[cfg(target_os = "linux")]
impl<T> Drop for MmapSlab<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: ptr was returned by mmap with this byte count.
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.bytes());
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl<T> std::ops::Deref for MmapSlab<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: ptr+len describe a valid initialized slice (zero-init).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Fallback slab for non-Linux: regular Vec. Eager-fault behavior tracks the
/// host allocator. Only Linux gets the mmap fast path.
#[cfg(not(target_os = "linux"))]
struct MmapSlab<T>(Vec<T>);

#[cfg(not(target_os = "linux"))]
impl<T: Copy + Default> MmapSlab<T> {
    fn new_zeroed(len: usize) -> Self {
        Self(vec![T::default(); len])
    }
    #[inline]
    fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
    #[inline]
    fn bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
}

#[cfg(not(target_os = "linux"))]
impl<T> std::ops::Deref for MmapSlab<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

/// Compile-time bound on reservoir size. Equals `PiPNNConfig` default
/// (l_max = 128). `find_hash_simd` scans L_MAX_MAX/32 = 4 AVX-512 chunks.
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
        T::as_f32_into(&data[i * ndims..(i + 1) * ndims], out)
            .unwrap_or_else(|e| panic!("VectorRepr::as_f32_into failed during LSH sketches: {}", e));
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

// ─── Cold slabs ───────────────────────────────────────────────────────────────
//
// Each per-point reservoir lives at index `idx` across three runtime-sized
// slabs: hashes, distances, neighbors. The stride is `scan_lanes` (l_max
// rounded up to a multiple of 32 so the AVX-512 / AVX-2 find_hash scan can
// stay aligned). At l_max=64 the stride is 64 and the per-point cold cost is
// 64 * 8 = 512 B; at l_max=128 the stride is 128 and the per-point cost is
// 1024 B. No more fixed [u16; L_MAX_MAX] padding.
//
// `ColdSlotPtrs` is the lightweight view passed into `insert_locked` and the
// scan/update helpers — three raw pointers + the stride. Mutation safety is
// established by the caller via `HotSlot.lock`.

#[derive(Clone, Copy)]
pub(crate) struct ColdSlotPtrs {
    hashes: *mut u16,
    distances: *mut u16,
    neighbors: *mut u32,
    scan_lanes: usize,
}

#[inline(always)]
fn round_up_to_32(n: usize) -> usize {
    n.div_ceil(32) * 32
}

// ─── find_hash SIMD: 32-way u16 compare ───────────────────────────────────────

/// Runtime SIMD dispatch for the hashes scan. Each tier-specific kernel is
/// emitted unconditionally via `#[target_feature]`, independent of the
/// build-time `target-cpu` baseline (the workspace pins x86-64-v3 = AVX-2 by
/// default, which used to cfg-out the AVX-512 path entirely and silently
/// disable it at deployment time even when the CPU supported it).
///
/// SAFETY: `hashes` must point at `scan_lanes` valid `u16` slots. `len` must
/// be the number of meaningful entries (<= scan_lanes and <= 128).
#[inline(always)]
unsafe fn find_hash_simd(
    hashes: *const u16,
    scan_lanes: usize,
    len: u8,
    target: u16,
) -> Option<usize> {
    if len == 0 {
        return None;
    }
    #[cfg(target_arch = "x86_64")]
    {
        match crate::cpu_dispatch::tier() {
            crate::cpu_dispatch::SimdTier::Avx512 => {
                return find_hash_avx512(hashes, scan_lanes, len, target);
            }
            crate::cpu_dispatch::SimdTier::Avx2 => {
                return find_hash_avx2(hashes, scan_lanes, len, target);
            }
            crate::cpu_dispatch::SimdTier::Scalar => {}
        }
    }
    find_hash_scalar(hashes, len as usize, target)
}

/// SAFETY: caller guarantees AVX-512F + AVX-512BW at runtime (tier()==Avx512
/// requires both — see `cpu_dispatch::detect_tier`). The 16-bit lane compare
/// `vpcmpeqw` lives in AVX-512BW; production deployment targets (Skylake-X
/// and later, Zen 4) ship both. The function-level `#[target_feature]`
/// emits the AVX-512 instructions even if the build's `target-cpu` baseline
/// does not include AVX-512.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn find_hash_avx512(
    hashes: *const u16,
    scan_lanes: usize,
    len: u8,
    target: u16,
) -> Option<usize> {
    use std::arch::x86_64::*;
    let len = len as usize;
    let t = _mm512_set1_epi16(target as i16);
    let chunks = scan_lanes / 32;
    let mut combined: u128 = 0;
    for chunk in 0..chunks {
        let v = _mm512_loadu_si512(hashes.add(chunk * 32) as *const __m512i);
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

/// SAFETY: caller guarantees AVX2 at runtime. AVX2 is the workspace baseline
/// (`x86-64-v3`) so this path covers every supported deployment.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_hash_avx2(
    hashes: *const u16,
    scan_lanes: usize,
    len: u8,
    target: u16,
) -> Option<usize> {
    use std::arch::x86_64::*;
    let len = len as usize;
    let t = _mm256_set1_epi16(target as i16);
    let chunks = scan_lanes / 16;
    for chunk in 0..chunks {
        let v = _mm256_loadu_si256(hashes.add(chunk * 16) as *const __m256i);
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

/// SAFETY: `hashes` must point at `len` valid `u16` slots.
#[inline(always)]
unsafe fn find_hash_scalar(hashes: *const u16, len: usize, target: u16) -> Option<usize> {
    for i in 0..len {
        if *hashes.add(i) == target {
            return Some(i);
        }
    }
    None
}

// ─── relative_hash_local: AoSoA-friendly per-pair sketch comparison ───────────
//
// `relative_hash_local(src, dst, m)` returns the `m`-bit pattern formed by
// `sign(dst[j] - src[j])` for j in 0..m. Used as the in-leaf LSH bucket for HP
// insertion (the "local sketch" cache hits L1).
//
// Runtime SIMD dispatch via `cpu_dispatch::tier()` — the previous
// implementation cfg-gated the AVX-512 path on compile-time `target_feature =
// "avx512f"`, which was DEAD CODE in shipped binaries because the workspace
// pins x86-64-v3 (AVX-2). Switching to runtime dispatch via `#[target_feature]
// unsafe fn` emits every tier in the binary regardless of build target.

/// SAFETY: `src` / `dst` must point at `m` valid `f32` slots, `m <= 16`.
#[inline]
unsafe fn relative_hash_local(src: *const f32, dst: *const f32, m: usize) -> u16 {
    #[cfg(target_arch = "x86_64")]
    {
        match crate::cpu_dispatch::tier() {
            crate::cpu_dispatch::SimdTier::Avx512 => {
                return relative_hash_local_avx512(src, dst, m);
            }
            crate::cpu_dispatch::SimdTier::Avx2 | crate::cpu_dispatch::SimdTier::Scalar => {}
        }
    }
    let mut h: u16 = 0;
    for j in 0..m {
        let diff = *dst.add(j) - *src.add(j);
        let bit = ((!diff.is_sign_negative()) as u16) << j;
        h |= bit;
    }
    h
}

/// SAFETY: caller guarantees AVX-512F at runtime. `m <= 16` enforced upstream.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn relative_hash_local_avx512(src: *const f32, dst: *const f32, m: usize) -> u16 {
    use std::arch::x86_64::*;
    let kmask: u16 = if m >= 16 { 0xFFFF } else { (1u16 << m) - 1 };
    let dst_v = _mm512_maskz_loadu_ps(kmask, dst);
    let src_v = _mm512_maskz_loadu_ps(kmask, src);
    let diff = _mm512_sub_ps(dst_v, src_v);
    let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(diff, _mm512_setzero_ps());
    mask & kmask
}

// ─── Per-reservoir mutation helpers (caller holds lock) ───────────────────────

/// Map a bf16-rounded `f32` to an order-preserving `u16` so raw integer
/// comparison matches float ordering for ALL signs. Raw bf16-bit compares are
/// monotonic only for non-negative values; InnerProduct distances are `-dot`
/// (negative), which otherwise sort inverted and make the reservoir evict its
/// best edges. For non-negative inputs this only sets the top bit on every
/// value, so L2/Cosine orderings — and the resulting graphs — are unchanged.
/// Inverse: [`key_to_bf16`].
#[inline(always)]
fn ordered_key(distance: f32) -> u16 {
    let b = f32_to_bf16(distance);
    if b & 0x8000 != 0 {
        !b
    } else {
        b | 0x8000
    }
}

/// Inverse of [`ordered_key`]: recover the bf16 bits for distance readback.
#[inline(always)]
fn key_to_bf16(key: u16) -> u16 {
    if key & 0x8000 != 0 {
        key & 0x7FFF
    } else {
        !key
    }
}

/// SAFETY: caller holds the slot lock; pointers in `cold` are valid for
/// `scan_lanes` elements each.
#[inline]
unsafe fn update_farthest(hot: &mut HotSlot, cold: ColdSlotPtrs) {
    if hot.len == 0 {
        hot.farthest_dist = 0;
        hot.farthest_idx = 0;
        return;
    }
    let mut max_dist: u16 = 0;
    let mut max_idx: u8 = 0;
    for i in 0..hot.len as usize {
        let d = *cold.distances.add(i);
        if d > max_dist {
            max_dist = d;
            max_idx = i as u8;
        }
    }
    hot.farthest_dist = max_dist;
    hot.farthest_idx = max_idx;
}

/// SAFETY: caller holds the slot lock; pointers in `cold` are valid for
/// `scan_lanes` elements each, and `l_max <= scan_lanes`.
#[inline(always)]
unsafe fn insert_locked(
    hot: &mut HotSlot,
    cold: ColdSlotPtrs,
    hash: u16,
    neighbor: u32,
    distance: f32,
    l_max: u8,
) -> bool {
    let dist_key = ordered_key(distance);

    if hot.len >= l_max && dist_key >= hot.farthest_dist {
        return false;
    }

    if let Some(idx) = find_hash_simd(cold.hashes, cold.scan_lanes, hot.len, hash) {
        if dist_key < *cold.distances.add(idx) {
            let was_farthest = idx == hot.farthest_idx as usize;
            *cold.neighbors.add(idx) = neighbor;
            *cold.distances.add(idx) = dist_key;
            if was_farthest {
                update_farthest(hot, cold);
            }
            return true;
        }
        return false;
    }

    if hot.len < l_max {
        let new_idx = hot.len as usize;
        *cold.hashes.add(new_idx) = hash;
        *cold.distances.add(new_idx) = dist_key;
        *cold.neighbors.add(new_idx) = neighbor;
        hot.len += 1;
        if dist_key >= hot.farthest_dist {
            hot.farthest_dist = dist_key;
            hot.farthest_idx = new_idx as u8;
        }
        return true;
    }

    if dist_key < hot.farthest_dist {
        let idx = hot.farthest_idx as usize;
        *cold.hashes.add(idx) = hash;
        *cold.distances.add(idx) = dist_key;
        *cold.neighbors.add(idx) = neighbor;
        update_farthest(hot, cold);
        return true;
    }
    false
}

/// Collect the reservoir's entries sorted by distance, truncated to
/// `max_degree`. Uses a stack-allocated `[(u32, u16); L_MAX_MAX]` scratch
/// instead of two heap Vec allocs per call — at 10M points this saves 20M
/// short-lived heap allocations on the extract path.
///
/// SAFETY: caller holds the slot lock; pointers in `cold` are valid for
/// `scan_lanes` elements each.
unsafe fn get_neighbors_saturated(
    hot: &HotSlot,
    cold: ColdSlotPtrs,
    max_degree: usize,
) -> Vec<(u32, f32)> {
    let n = hot.len as usize;
    debug_assert!(n <= L_MAX_MAX);

    let mut scratch = [(0u32, 0u16); L_MAX_MAX];
    for i in 0..n {
        scratch[i] = (*cold.neighbors.add(i), *cold.distances.add(i));
    }
    scratch[..n].sort_unstable_by_key(|&(_, d)| d);
    let out_len = n.min(max_degree);
    let mut out = Vec::with_capacity(out_len);
    for &(id, d) in &scratch[..out_len] {
        out.push((id, bf16_to_f32(key_to_bf16(d))));
    }
    out
}

// ─── Test-only thin wrapper preserving the old HashPruneReservoir API ─────────

#[cfg(test)]
pub(crate) struct HashPruneReservoir {
    hot: HotSlot,
    hashes: Vec<u16>,
    distances: Vec<u16>,
    neighbors: Vec<u32>,
    scan_lanes: usize,
    l_max: u8,
}

#[cfg(test)]
impl HashPruneReservoir {
    pub(crate) fn new(l_max: usize) -> Self {
        assert!(l_max <= L_MAX_MAX);
        let scan_lanes = round_up_to_32(l_max).max(32);
        Self {
            hot: HotSlot::new_empty(),
            hashes: vec![0u16; scan_lanes],
            distances: vec![0u16; scan_lanes],
            neighbors: vec![0u32; scan_lanes],
            scan_lanes,
            l_max: l_max as u8,
        }
    }

    pub(crate) fn new_lazy(l_max: usize) -> Self {
        Self::new(l_max)
    }

    fn cold(&self) -> ColdSlotPtrs {
        ColdSlotPtrs {
            hashes: self.hashes.as_ptr() as *mut u16,
            distances: self.distances.as_ptr() as *mut u16,
            neighbors: self.neighbors.as_ptr() as *mut u32,
            scan_lanes: self.scan_lanes,
        }
    }

    pub fn insert(&mut self, hash: u16, neighbor: u32, distance: f32) -> bool {
        // SAFETY: single-threaded test wrapper; cold ptrs alias self.hashes
        // etc. but we hold &mut self so no other reference can exist for the
        // duration. insert_locked only touches the slot at index 0.
        let cold = self.cold();
        unsafe { insert_locked(&mut self.hot, cold, hash, neighbor, distance, self.l_max) }
    }

    pub fn get_neighbors_saturated(&self, max_degree: usize) -> Vec<(u32, f32)> {
        let cold = self.cold();
        // SAFETY: single-threaded test wrapper; ptrs are valid for scan_lanes.
        unsafe { get_neighbors_saturated(&self.hot, cold, max_degree) }
    }

    pub(crate) fn get_neighbors_sorted(&self) -> Vec<(u32, f32)> {
        let cold = self.cold();
        // SAFETY: single-threaded test wrapper.
        unsafe { get_neighbors_saturated(&self.hot, cold, L_MAX_MAX) }
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
    /// AoSoA hashes slab: `npoints * scan_lanes` u16.
    cold_hashes: MmapSlab<u16>,
    /// AoSoA distances slab (bf16 in u16): `npoints * scan_lanes`.
    cold_distances: MmapSlab<u16>,
    /// AoSoA neighbors slab: `npoints * scan_lanes` u32.
    cold_neighbors: MmapSlab<u32>,
    /// Per-slot stride. Equals `round_up_to_32(l_max).max(32)`. Always a
    /// multiple of 32 so the AVX-512 / AVX-2 find_hash scan stays aligned.
    scan_lanes: usize,
    sketches: LshSketches,
    max_degree: usize,
    l_max: usize,
}

// SAFETY: HotSlot has interior mutability via RawMutex. The cold slabs are
// plain bit-pattern arrays; each per-point slot is guarded by HotSlot[i].lock.
// Disjoint-index parallel access is safe.
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
            "HashPrune requires l_max ≤ {}, got {}",
            L_MAX_MAX,
            l_max
        );

        let t1 = std::time::Instant::now();
        let scan_lanes = round_up_to_32(l_max).max(32);

        // Hot slab: one HotSlot per point, contiguous. 160 MB at 10M.
        let mut hot: Vec<HotSlot> = Vec::with_capacity(npoints);
        for _ in 0..npoints {
            hot.push(HotSlot::new_empty());
        }

        // Three cold slabs, each `npoints * scan_lanes` elements, allocated
        // via mmap so the kernel keeps them zero-backed (no physical pages
        // until first write). At scan_lanes = 64 the per-point cold cost is
        // 64 * 8 = 512 B; at scan_lanes = 128 it is 1024 B. Reservoirs that
        // never fill past the avg fill don't touch the high pages.
        let total = npoints
            .checked_mul(scan_lanes)
            .expect("HashPrune slab size overflowed usize");
        let cold_hashes = MmapSlab::<u16>::new_zeroed(total);
        let cold_distances = MmapSlab::<u16>::new_zeroed(total);
        let cold_neighbors = MmapSlab::<u32>::new_zeroed(total);

        // Hint hugepages on slabs > 2 MB so DTLB pressure scales with 2 MB
        // pages instead of 4 KB. Non-fatal on failure; no-op on kernels
        // without THP.
        #[cfg(target_os = "linux")]
        {
            let hot_bytes = hot.len() * std::mem::size_of::<HotSlot>();
            // SAFETY: each slab backs a contiguous allocation of the indicated
            // byte length. madvise is non-fatal on failure.
            unsafe {
                for (ptr, bytes) in [
                    (hot.as_ptr() as *mut libc::c_void, hot_bytes),
                    (cold_hashes.as_ptr() as *mut libc::c_void, cold_hashes.bytes()),
                    (cold_distances.as_ptr() as *mut libc::c_void, cold_distances.bytes()),
                    (cold_neighbors.as_ptr() as *mut libc::c_void, cold_neighbors.bytes()),
                ] {
                    if bytes > 2 * 1024 * 1024 {
                        libc::madvise(ptr, bytes, libc::MADV_HUGEPAGE);
                    }
                }
            }
        }

        tracing::debug!(
            elapsed_secs = t1.elapsed().as_secs_f64(),
            scan_lanes,
            "reservoir allocation"
        );

        Self {
            hot,
            cold_hashes,
            cold_distances,
            cold_neighbors,
            scan_lanes,
            sketches,
            max_degree,
            l_max,
        }
    }

    /// SAFETY: idx must be in bounds for `self.hot` and `idx * scan_lanes +
    /// scan_lanes <= cold slab capacity`.
    #[inline]
    unsafe fn slot_ptrs(&self, idx: usize) -> (*mut HotSlot, ColdSlotPtrs) {
        let hot_ptr = (self.hot.as_ptr() as *mut HotSlot).add(idx);
        let off = idx * self.scan_lanes;
        let cold = ColdSlotPtrs {
            hashes: (self.cold_hashes.as_ptr() as *mut u16).add(off),
            distances: (self.cold_distances.as_ptr() as *mut u16).add(off),
            neighbors: (self.cold_neighbors.as_ptr() as *mut u32).add(off),
            scan_lanes: self.scan_lanes,
        };
        (hot_ptr, cold)
    }

    /// SAFETY: idx must be in bounds for `self.hot`.
    #[inline]
    unsafe fn lock_slot(&self, idx: usize) -> (*mut HotSlot, ColdSlotPtrs) {
        let (hot_ptr, cold) = self.slot_ptrs(idx);
        (*hot_ptr).lock.lock();
        (hot_ptr, cold)
    }

    /// SAFETY: caller must hold the lock at idx.
    #[inline]
    unsafe fn unlock_slot(&self, idx: usize) {
        let hot_ptr = (self.hot.as_ptr() as *mut HotSlot).add(idx);
        (*hot_ptr).lock.unlock();
    }

    #[inline(always)]
    fn with_locked<R>(&self, idx: usize, f: impl FnOnce(&mut HotSlot, ColdSlotPtrs) -> R) -> R {
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
        let (hot_ptr, cold) = unsafe { self.lock_slot(idx) };
        let _guard = UnlockOnDrop { hp: self, idx };
        unsafe { f(&mut *hot_ptr, cold) }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn add_edge(&self, p: usize, c: usize, distance: f32) {
        let hash = self.sketches.relative_hash(p, c);
        let l_max = self.l_max as u8;
        self.with_locked(p, |hot, cold| {
            // SAFETY: cold ptrs from with_locked are valid for scan_lanes.
            unsafe { insert_locked(hot, cold, hash, c as u32, distance, l_max) };
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
                    // SAFETY: cold ptrs from with_locked are valid for scan_lanes.
                    unsafe { insert_locked(hot, cold, hash, edge.dst, edge.distance, l_max) };
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
                    // SAFETY: cold ptrs from with_locked are valid for scan_lanes.
                    unsafe { insert_locked(hot, cold, hash, global_dst, dist, l_max) };
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
                        let off = nxt_global * self.scan_lanes;
                        // SAFETY: nxt_global < npoints; off + scan_lanes
                        // bounded by slab capacity. Prefetch is non-fatal.
                        unsafe {
                            let hot_p = self.hot.as_ptr().add(nxt_global) as *const i8;
                            _mm_prefetch::<{ _MM_HINT_T0 }>(hot_p);
                            // hashes array covers `scan_lanes` u16 = scan_lanes*2 bytes;
                            // prefetch in 64-byte cache-line strides.
                            let hashes_p = self.cold_hashes.as_ptr().add(off) as *const i8;
                            let hashes_bytes = self.scan_lanes * std::mem::size_of::<u16>();
                            let mut b = 0usize;
                            while b < hashes_bytes {
                                _mm_prefetch::<{ _MM_HINT_T0 }>(hashes_p.add(b));
                                b += 64;
                            }
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
                    debug_assert!(m <= 16, "num_planes <= 16 enforced by validate");
                    // SAFETY: m <= 16, sketches are m-elt slices.
                    let hash =
                        unsafe { relative_hash_local(src_sketch.as_ptr(), dst_sketch.as_ptr(), m) };
                    // SAFETY: cold ptrs from with_locked are valid for scan_lanes.
                    unsafe { insert_locked(hot, cold, hash, global_dst, dist, l_max) };
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

    /// Consume self and parallel-extract one per-point row from the cold slabs.
    /// `cap` truncates each reservoir; `project` maps the sorted
    /// `(neighbor, dist_f32)` pairs into the caller's output type.
    fn extract_into<R, F>(self, cap: usize, project: F) -> Vec<Vec<R>>
    where
        R: Send,
        F: Fn((u32, f32)) -> R + Sync,
    {
        let scan_lanes = self.scan_lanes;
        drop(self.sketches);
        let HashPrune {
            hot,
            cold_hashes,
            cold_distances,
            cold_neighbors,
            ..
        } = self;
        (0..hot.len())
            .into_par_iter()
            .map(|i| {
                let off = i * scan_lanes;
                let cold = ColdSlotPtrs {
                    hashes: (cold_hashes.as_ptr() as *mut u16).wrapping_add(off),
                    distances: (cold_distances.as_ptr() as *mut u16).wrapping_add(off),
                    neighbors: (cold_neighbors.as_ptr() as *mut u32).wrapping_add(off),
                    scan_lanes,
                };
                // SAFETY: i < hot.len() == npoints; off + scan_lanes within slab.
                let nbrs = unsafe { get_neighbors_saturated(&hot[i], cold, cap) };
                nbrs.into_iter().map(&project).collect()
            })
            .collect_installed()
    }

    pub fn extract_graph(self) -> Vec<Vec<u32>> {
        let cap = self.max_degree;
        self.extract_into(cap, |(id, _)| id)
    }

    pub fn extract_graph_for_prune(self) -> Vec<Vec<(u32, f32)>> {
        let cap = self.l_max;
        self.extract_into(cap, |pair| pair)
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
