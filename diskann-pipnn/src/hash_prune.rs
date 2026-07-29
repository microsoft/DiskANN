/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! HashPrune: LSH-based online pruning for merging edges from overlapping partitions.
//!
//! Storage is AoSoA hot/cold split:
//! - `hot: Vec<HotSlot>` — one 16-byte slot per point (mutex + len + farthest).
//!   Early rejection and lock acquisition only touch this slab.
//! - Three cold slabs (`cold_hashes`, `cold_distances`, `cold_neighbors`), each a
//!   single `MmapSlab` of `npoints * scan_lanes` elements. Splitting hashes /
//!   distances / neighbors into three contiguous arrays lets `find_hash` walk
//!   pure u16 hashes (32 per cache line) instead of 8-byte mixed-AoS entries.
//!
//! Each slab is one contiguous allocation, so `madvise(HUGEPAGE)` is effective
//! when the kernel actually backs THP.
//!
//! `l_max` is dynamic user input; the cold slab stride (`scan_lanes`) and the
//! hash scan width both scale with it at runtime. The only fixed
//! bound is `MAX_RESERVOIR_LEN = 255`, the structural limit of the `u8`
//! `HotSlot.len` / `farthest_idx` fields; a larger `l_max` is rejected at
//! construction time.

use parking_lot::lock_api::RawMutex as RawMutexTrait;
use std::cell::UnsafeCell;

use crate::bf16::{bf16_to_f32, f32_to_bf16};
use crate::lsh::{LshSketchError, LshSketches};
use bytemuck::Pod;
use diskann::{graph::AdjacencyList, utils::VectorRepr, ANNError, ANNResult};
use diskann_vector::prefetch_hint_all;
use diskann_wide::{
    arch::{self, Dispatched1, FTarget1, Target},
    lifetime::As,
    Architecture, SIMDMask, SIMDPartialEq, SIMDPartialOrd, SIMDVector,
};
use rayon::prelude::*;

/// Owned slab allocated via direct `mmap(MAP_PRIVATE | MAP_ANONYMOUS)`. The
/// kernel backs the range with its zero-page until first write, so we get
/// true lazy faulting for the AoSoA cold slabs rather than eagerly committing
/// the full reservoir allocation.
#[cfg(target_os = "linux")]
struct MmapSlab<T: Pod> {
    ptr: *mut T,
    len: usize,
}

#[cfg(target_os = "linux")]
// SAFETY: the allocation contains only `Pod` values and ownership transfers
// with the slab; mutation is synchronized by HashPrune's per-row locks.
unsafe impl<T: Pod + Send> Send for MmapSlab<T> {}
#[cfg(target_os = "linux")]
// SAFETY: shared access exposes only immutable pointers; HashPrune synchronizes
// every mutation to a row.
unsafe impl<T: Pod + Sync> Sync for MmapSlab<T> {}

#[cfg(target_os = "linux")]
impl<T: Pod> MmapSlab<T> {
    fn new_zeroed(len: usize) -> ANNResult<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::<T>::dangling().as_ptr(),
                len: 0,
            });
        }
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| crate::config_error(format!("slab size {len} overflows usize")))?;
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
            if ptr == libc::MAP_FAILED {
                return Err(ANNError::from(std::io::Error::last_os_error())
                    .context(format!("mmap failed for {bytes} HashPrune slab bytes")));
            }
            Ok(Self {
                ptr: ptr as *mut T,
                len,
            })
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
impl<T: Pod> Drop for MmapSlab<T> {
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
impl<T: Pod> std::ops::Deref for MmapSlab<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: ptr+len describe a valid initialized slice (zero-init).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Windows counterpart of the Linux mmap slab. `VirtualAlloc(MEM_RESERVE |
/// MEM_COMMIT, PAGE_READWRITE)` reserves a zero-backed anonymous range whose
/// pages fault in on first write — the same lazy-commit behavior as
/// `mmap(MAP_ANONYMOUS)`, so Windows gets the same peak-RSS win instead of the
/// eager-fault `Vec` fallback.
#[cfg(windows)]
mod winmem {
    // Minimal FFI to the Win32 memory API — avoids pulling in the `windows`
    // crate for four extern declarations.
    pub(super) type LPVOID = *mut core::ffi::c_void;
    pub(super) const MEM_COMMIT: u32 = 0x0000_1000;
    pub(super) const MEM_RESERVE: u32 = 0x0000_2000;
    pub(super) const MEM_RELEASE: u32 = 0x0000_8000;
    pub(super) const PAGE_READWRITE: u32 = 0x04;

    extern "system" {
        pub(super) fn VirtualAlloc(
            lpAddress: LPVOID,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> LPVOID;
        pub(super) fn VirtualFree(lpAddress: LPVOID, dwSize: usize, dwFreeType: u32) -> i32;
    }
}

#[cfg(windows)]
struct MmapSlab<T: Pod> {
    ptr: *mut T,
    len: usize,
}

#[cfg(windows)]
// SAFETY: see the Linux implementation; Windows provides the same zeroed,
// process-owned allocation semantics.
unsafe impl<T: Pod + Send> Send for MmapSlab<T> {}
#[cfg(windows)]
// SAFETY: shared mutation is synchronized by HashPrune's per-row locks.
unsafe impl<T: Pod + Sync> Sync for MmapSlab<T> {}

#[cfg(windows)]
impl<T: Pod> MmapSlab<T> {
    fn new_zeroed(len: usize) -> ANNResult<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::<T>::dangling().as_ptr(),
                len: 0,
            });
        }
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| crate::config_error(format!("slab size {len} overflows usize")))?;
        // SAFETY: MEM_RESERVE|MEM_COMMIT + PAGE_READWRITE returns a zero-backed
        // RW region; physical pages fault in on first write only. Windows
        // zero-fills committed pages, matching mmap's MAP_ANONYMOUS contract.
        unsafe {
            let ptr = winmem::VirtualAlloc(
                std::ptr::null_mut(),
                bytes,
                winmem::MEM_RESERVE | winmem::MEM_COMMIT,
                winmem::PAGE_READWRITE,
            );
            if ptr.is_null() {
                return Err(
                    ANNError::from(std::io::Error::last_os_error()).context(format!(
                        "VirtualAlloc failed for {bytes} HashPrune slab bytes"
                    )),
                );
            }
            Ok(Self {
                ptr: ptr as *mut T,
                len,
            })
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline]
    #[allow(dead_code)] // parity with the Linux slab; madvise (Linux-only) is the sole caller
    fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

#[cfg(windows)]
impl<T: Pod> Drop for MmapSlab<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: ptr came from VirtualAlloc; MEM_RELEASE requires dwSize=0.
            unsafe {
                winmem::VirtualFree(self.ptr as winmem::LPVOID, 0, winmem::MEM_RELEASE);
            }
        }
    }
}

#[cfg(windows)]
impl<T: Pod> std::ops::Deref for MmapSlab<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: ptr+len describe a valid initialized slice (zero-init).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Fallback slab for platforms that are neither Linux nor Windows: regular Vec.
/// Eager-fault behavior tracks the host allocator.
#[cfg(not(any(target_os = "linux", windows)))]
struct MmapSlab<T: Pod>(Vec<T>);

#[cfg(not(any(target_os = "linux", windows)))]
impl<T: Pod + Default> MmapSlab<T> {
    fn new_zeroed(len: usize) -> ANNResult<Self> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(ANNError::opaque)
            .map_err(|error| error.context(format!("reserving {len} HashPrune slab elements")))?;
        values.resize_with(len, T::default);
        Ok(Self(values))
    }
    #[inline]
    fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }
    #[inline]
    fn bytes(&self) -> usize {
        self.0.len() * std::mem::size_of::<T>()
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
impl<T: Pod> std::ops::Deref for MmapSlab<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

/// Structural upper bound on per-node reservoir length: `HotSlot.len` and
/// `farthest_idx` are `u8`, so a reservoir can hold at most 255 entries. This
/// is an overflow guard, NOT the reservoir size — the cold slab stride
/// (`scan_lanes`) is sized to the runtime `l_max`, so the list scales with the
/// user's `l_max` up to this bound. `find_hash_simd` scans `scan_lanes / 32`
/// chunks, also runtime-sized.
pub(crate) const MAX_RESERVOIR_LEN: usize = u8::MAX as usize;

/// Compute LSH sketches over `data` (row-major `npoints × ndims` of `T`).
fn sketches_from_data<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    num_planes: usize,
    seed: u64,
) -> ANNResult<LshSketches> {
    LshSketches::try_new(npoints, ndims, num_planes, seed, |i, out| {
        T::as_f32_into(&data[i * ndims..(i + 1) * ndims], out)
    })
    .map_err(|error| match error {
        LshSketchError::InvalidPlaneCount { actual, max } => {
            crate::config_error(format!("num_hash_planes ({actual}) must be in 1..={max}"))
        }
        LshSketchError::ShapeOverflow { rows, columns } => ANNError::log_index_error(format!(
            "LSH matrix shape {rows} x {columns} overflows usize"
        )),
        LshSketchError::Allocation(error) => ANNError::opaque(error),
        LshSketchError::Fill(error) => error.into(),
    })
}

// ─── HotSlot: 16-byte per-point mutex + cached fields ─────────────────────────

#[repr(C, align(16))]
struct HotSlot {
    lock: parking_lot::RawMutex,
    len: u8,
    farthest_idx: u8,
    _pad0: u8,
    farthest_dist: u16,
    _pad1: [u8; 10],
}

#[repr(transparent)]
struct LockedHotSlot(UnsafeCell<HotSlot>);

impl LockedHotSlot {
    fn new() -> Self {
        Self(UnsafeCell::new(HotSlot::new_empty()))
    }

    fn get(&self) -> *mut HotSlot {
        self.0.get()
    }
}

// SAFETY: every mutable access to the contained HotSlot is guarded by its
// embedded RawMutex. Read-only extraction happens only after HashPrune is
// consumed, when no mutation can remain.
unsafe impl Sync for LockedHotSlot {}

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
// 1024 B. No fixed-size padding — the stride is the runtime l_max.
//
// `ColdSlotPtrs` is the lightweight view passed into `insert_locked` and the
// scan/update helpers — three raw pointers + the stride. Mutation safety is
// established by the caller via `HotSlot.lock`.

#[derive(Clone, Copy)]
struct ColdSlotPtrs {
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

#[derive(Clone, Copy)]
struct FindHashArgs {
    hashes: *const u16,
    scan_lanes: usize,
    len: u8,
    target: u16,
}

#[derive(Clone, Copy)]
struct RelativeHashArgs {
    src: *const f32,
    dst: *const f32,
    len: usize,
}

type FindHash = Dispatched1<Option<usize>, As<FindHashArgs>>;
type RelativeHash = Dispatched1<u16, As<RelativeHashArgs>>;

struct FindHashKernel;
struct RelativeHashKernel;
struct SelectFindHash;
struct SelectRelativeHash;

impl<A> Target<A, FindHash> for SelectFindHash
where
    A: Architecture,
    FindHashKernel: FTarget1<A, Option<usize>, FindHashArgs>,
{
    fn run(self, arch: A) -> FindHash {
        arch.dispatch1::<FindHashKernel, Option<usize>, As<FindHashArgs>>()
    }
}

impl<A> Target<A, RelativeHash> for SelectRelativeHash
where
    A: Architecture,
    RelativeHashKernel: FTarget1<A, u16, RelativeHashArgs>,
{
    fn run(self, arch: A) -> RelativeHash {
        arch.dispatch1::<RelativeHashKernel, u16, As<RelativeHashArgs>>()
    }
}

fn select_find_hash() -> FindHash {
    arch::dispatch(SelectFindHash)
}

fn select_relative_hash() -> RelativeHash {
    arch::dispatch(SelectRelativeHash)
}

impl<A> FTarget1<A, Option<usize>, FindHashArgs> for FindHashKernel
where
    A: Architecture,
    A::i16x32: SIMDPartialEq,
{
    fn run(arch: A, args: FindHashArgs) -> Option<usize> {
        find_hash_simd::<A::i16x32>(arch, args)
    }
}

impl<A> FTarget1<A, u16, RelativeHashArgs> for RelativeHashKernel
where
    A: Architecture,
    A::f32x16: SIMDPartialOrd + std::ops::Sub<Output = A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    fn run(arch: A, args: RelativeHashArgs) -> u16 {
        relative_hash_simd::<A::f32x16>(arch, args)
    }
}

/// Hashes are compared for equality only, so the lanes are loaded as `i16`
/// (diskann-wide has no `u16` vector type) — the bit patterns, and therefore
/// the equality result, are identical.
fn find_hash_simd<F>(arch: F::Arch, args: FindHashArgs) -> Option<usize>
where
    F: SIMDVector<Scalar = i16> + SIMDPartialEq,
{
    let len = args.len as usize;
    let target = F::splat(arch, args.target as i16);
    let chunks = len.div_ceil(F::LANES).min(args.scan_lanes / F::LANES);
    for chunk in 0..chunks {
        // SAFETY: every full load stays inside the padded `scan_lanes` row.
        let values = unsafe { F::load_simd(arch, args.hashes.add(chunk * F::LANES).cast::<i16>()) };
        if let Some(offset) = values.eq_simd(target).first() {
            let lane = chunk * F::LANES + offset;
            if lane < len {
                return Some(lane);
            }
        }
    }
    None
}

/// Bit `j` of the returned hash is `dst[j] - src[j] >= 0.0`; equality,
/// including signed zero, hashes as non-negative on every backend.
fn relative_hash_simd<F>(arch: F::Arch, args: RelativeHashArgs) -> u16
where
    F: SIMDVector<Scalar = f32> + SIMDPartialOrd + std::ops::Sub<Output = F>,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    debug_assert!(args.len <= F::LANES);
    debug_assert!(F::LANES <= u16::BITS as usize);

    // SAFETY: the first `len` lanes are valid and masked loads do not access
    // inactive lanes. HashPrune limits sketches to 16 planes.
    let dst = unsafe { F::load_simd_first(arch, args.dst, args.len) };
    // SAFETY: as above.
    let src = unsafe { F::load_simd_first(arch, args.src, args.len) };
    let bits = u64::from(
        (dst - src)
            .ge_simd(F::splat(arch, 0.0))
            .bitmask()
            .to_underlying(),
    );
    let active = ((1_u32 << args.len) - 1) as u16;
    bits as u16 & active
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
        // SAFETY: guaranteed by this function's contract.
        let d = unsafe { *cold.distances.add(i) };
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
    find_hash: FindHash,
) -> bool {
    let dist_key = ordered_key(distance);

    if hot.len >= l_max && dist_key >= hot.farthest_dist {
        return false;
    }

    if let Some(idx) = find_hash.call(FindHashArgs {
        hashes: cold.hashes,
        scan_lanes: cold.scan_lanes,
        len: hot.len,
        target: hash,
    }) {
        // SAFETY: `idx < hot.len <= cold.scan_lanes`.
        if dist_key < unsafe { *cold.distances.add(idx) } {
            let was_farthest = idx == hot.farthest_idx as usize;
            // SAFETY: as above; the caller holds the slot lock.
            unsafe {
                *cold.neighbors.add(idx) = neighbor;
                *cold.distances.add(idx) = dist_key;
            }
            if was_farthest {
                // SAFETY: this function's contract provides the same invariants.
                unsafe { update_farthest(hot, cold) };
            }
            return true;
        }
        return false;
    }

    if hot.len < l_max {
        let new_idx = hot.len as usize;
        // SAFETY: `new_idx < l_max <= cold.scan_lanes`; the caller holds the lock.
        unsafe {
            *cold.hashes.add(new_idx) = hash;
            *cold.distances.add(new_idx) = dist_key;
            *cold.neighbors.add(new_idx) = neighbor;
        }
        hot.len += 1;
        if dist_key >= hot.farthest_dist {
            hot.farthest_dist = dist_key;
            hot.farthest_idx = new_idx as u8;
        }
        return true;
    }

    if dist_key < hot.farthest_dist {
        let idx = hot.farthest_idx as usize;
        // SAFETY: `idx < hot.len <= cold.scan_lanes`; the caller holds the lock.
        unsafe {
            *cold.hashes.add(idx) = hash;
            *cold.distances.add(idx) = dist_key;
            *cold.neighbors.add(idx) = neighbor;
            update_farthest(hot, cold);
        }
        return true;
    }
    false
}

/// Collect the reservoir's entries sorted by distance, truncated to `cap`.
/// A thread-local scratch `Vec`, sized to the reservoir's runtime fill and
/// reused across calls, avoids per-row allocation during extraction.
///
/// SAFETY: caller holds the slot lock; `distances` and `neighbors` are valid
/// for `hot.len` elements.
unsafe fn collect_sorted_neighbors(
    hot: &HotSlot,
    distances: *const u16,
    neighbors: *const u32,
    cap: usize,
) -> Vec<(u32, f32)> {
    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<(u32, u16)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let n = hot.len as usize;
    SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.clear();
        scratch.reserve(n);
        for i in 0..n {
            // SAFETY: guaranteed by this function's contract.
            scratch.push(unsafe { (*neighbors.add(i), *distances.add(i)) });
        }
        scratch.sort_unstable_by_key(|&(_, d)| d);
        let out_len = n.min(cap);
        let mut out = Vec::with_capacity(out_len);
        for &(id, d) in &scratch[..out_len] {
            out.push((id, bf16_to_f32(key_to_bf16(d))));
        }
        out
    })
}

/// Collect the reservoir's neighbor ids, truncated to `cap`, WITHOUT sorting.
/// Reservoir order is intentionally not preserved. Reading only `neighbors`
/// lets the caller drop the hashes and distances slabs before extraction; any
/// ordering required by a later graph-finalization policy belongs to that caller.
///
/// SAFETY: caller holds the slot lock (or owns the reservoir); `neighbors` is
/// valid for `hot.len` elements.
#[inline]
unsafe fn collect_neighbor_ids(hot: &HotSlot, neighbors: *const u32, cap: usize) -> Vec<u32> {
    let out_len = (hot.len as usize).min(cap);
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        // SAFETY: guaranteed by this function's contract.
        out.push(unsafe { *neighbors.add(i) });
    }
    out
}

// ─── HashPrune ────────────────────────────────────────────────────────────────

pub(crate) struct HashPrune {
    hot: Vec<LockedHotSlot>,
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
    l_max: usize,
    find_hash: FindHash,
    relative_hash: RelativeHash,
}

// SAFETY: HotSlot has interior mutability via RawMutex. The cold slabs are
// plain bit-pattern arrays; each per-point slot is guarded by HotSlot[i].lock.
// Disjoint-index parallel access is safe.
unsafe impl Send for HashPrune {}
// SAFETY: the same per-row lock protects all mutation through shared
// references, and immutable sketches are safe to share.
unsafe impl Sync for HashPrune {}

impl HashPrune {
    pub(crate) fn new<T: VectorRepr + Send + Sync>(
        data: &[T],
        npoints: usize,
        ndims: usize,
        num_planes: usize,
        l_max: usize,
        seed: u64,
    ) -> ANNResult<Self> {
        if !(1..=MAX_RESERVOIR_LEN).contains(&l_max) {
            return Err(crate::config_error(format!(
                "HashPrune l_max ({l_max}) must be in 1..={MAX_RESERVOIR_LEN}"
            )));
        }

        let t0 = std::time::Instant::now();
        let sketches = sketches_from_data(data, npoints, ndims, num_planes, seed)?;
        tracing::debug!(
            elapsed_secs = t0.elapsed().as_secs_f64(),
            "sketch computation"
        );
        let t1 = std::time::Instant::now();
        let scan_lanes = round_up_to_32(l_max).max(32);

        // Hot slab: one HotSlot per point, contiguous.
        let mut hot: Vec<LockedHotSlot> = Vec::new();
        hot.try_reserve_exact(npoints)
            .map_err(ANNError::opaque)
            .map_err(|error| error.context(format!("reserving {npoints} HashPrune rows")))?;
        for _ in 0..npoints {
            hot.push(LockedHotSlot::new());
        }

        // Three cold slabs, each `npoints * scan_lanes` elements, allocated
        // via mmap so the kernel keeps them zero-backed (no physical pages
        // until first write). At scan_lanes = 64 the per-point cold cost is
        // 64 * 8 = 512 B; at scan_lanes = 128 it is 1024 B. Reservoirs that
        // never fill past the avg fill don't touch the high pages.
        let total = npoints.checked_mul(scan_lanes).ok_or_else(|| {
            crate::config_error(format!(
                "HashPrune slab shape {npoints} x {scan_lanes} overflows usize"
            ))
        })?;
        let cold_hashes = MmapSlab::<u16>::new_zeroed(total)?;
        let cold_distances = MmapSlab::<u16>::new_zeroed(total)?;
        let cold_neighbors = MmapSlab::<u32>::new_zeroed(total)?;

        // Hint hugepages on slabs > 2 MB so DTLB pressure scales with 2 MB
        // pages instead of 4 KB. Non-fatal on failure; no-op on kernels
        // without THP. Linux-only: the Windows MEM_LARGE_PAGES equivalent must
        // be requested at VirtualAlloc time AND needs SeLockMemoryPrivilege
        // (off by default), so a failure would abort the slab rather than
        // silently fall back — not worth it for a DTLB hint.
        #[cfg(target_os = "linux")]
        {
            let hot_bytes = hot.len() * std::mem::size_of::<LockedHotSlot>();
            // SAFETY: each slab backs a contiguous allocation of the indicated
            // byte length. madvise is non-fatal on failure.
            unsafe {
                for (ptr, bytes) in [
                    (hot.as_ptr() as *mut libc::c_void, hot_bytes),
                    (
                        cold_hashes.as_ptr() as *mut libc::c_void,
                        cold_hashes.bytes(),
                    ),
                    (
                        cold_distances.as_ptr() as *mut libc::c_void,
                        cold_distances.bytes(),
                    ),
                    (
                        cold_neighbors.as_ptr() as *mut libc::c_void,
                        cold_neighbors.bytes(),
                    ),
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

        Ok(Self {
            hot,
            cold_hashes,
            cold_distances,
            cold_neighbors,
            scan_lanes,
            sketches,
            l_max,
            find_hash: select_find_hash(),
            relative_hash: select_relative_hash(),
        })
    }

    /// Locks the per-slot mutex at `idx`, runs `f` with mutable access to the
    /// hot slot and its cold-slab pointers, and unlocks on return or panic.
    #[inline(always)]
    fn with_locked<R>(&self, idx: usize, f: impl FnOnce(&mut HotSlot, ColdSlotPtrs) -> R) -> R {
        struct UnlockOnDrop {
            hot_ptr: *mut HotSlot,
        }
        impl Drop for UnlockOnDrop {
            fn drop(&mut self) {
                // SAFETY: only constructed after locking this slot's mutex.
                unsafe { (*self.hot_ptr).lock.unlock() };
            }
        }
        assert!(idx < self.hot.len(), "HashPrune row index out of bounds");
        // SAFETY: idx bounds-checked above, so `idx * scan_lanes + scan_lanes`
        // is within each cold slab's capacity. UnlockOnDrop unlocks on panic.
        unsafe {
            let hot_ptr = self.hot[idx].get();
            let off = idx * self.scan_lanes;
            let cold = ColdSlotPtrs {
                hashes: (self.cold_hashes.as_ptr() as *mut u16).add(off),
                distances: (self.cold_distances.as_ptr() as *mut u16).add(off),
                neighbors: (self.cold_neighbors.as_ptr() as *mut u32).add(off),
                scan_lanes: self.scan_lanes,
            };
            (*hot_ptr).lock.lock();
            let _guard = UnlockOnDrop { hot_ptr };
            f(&mut *hot_ptr, cold)
        }
    }

    /// Merge one leaf's CSR edge list into the global reservoirs.
    ///
    /// Sketch layout and gathering are HashPrune implementation details; the
    /// caller only lends a reusable buffer to avoid per-leaf allocation.
    pub(crate) fn add_leaf_edges(
        &self,
        point_ids: &[u32],
        edge_offsets: &[u32],
        edges: &[(u32, f32)],
        sketch_scratch: &mut Vec<f32>,
    ) {
        if edges.is_empty() {
            return;
        }
        self.add_leaf_edges_with_scratch(point_ids, edge_offsets, edges, sketch_scratch);
    }

    fn add_leaf_edges_with_scratch(
        &self,
        point_ids: &[u32],
        edge_offsets: &[u32],
        edges: &[(u32, f32)],
        sketch_scratch: &mut Vec<f32>,
    ) {
        let n = point_ids.len();
        let m = self.sketches.num_planes();
        let l_max = self.l_max as u8;
        debug_assert_eq!(edge_offsets.len(), n + 1);
        let sketch_len = n * m;
        sketch_scratch.resize(sketch_len, 0.0);
        self.gather_sketches(point_ids, &mut sketch_scratch[..sketch_len]);

        for local_src in 0..n {
            let start = edge_offsets[local_src] as usize;
            let end = edge_offsets[local_src + 1] as usize;
            if start == end {
                continue;
            }
            let global_src = point_ids[local_src] as usize;

            // Prefetch the next non-empty source's hot and cold slots.
            if let Some(next) = (local_src + 1..n)
                .find(|&i| edge_offsets[i] != edge_offsets[i + 1])
                .map(|i| point_ids[i] as usize)
            {
                let off = next * self.scan_lanes;
                prefetch_hint_all(std::slice::from_ref(&self.hot[next]));
                prefetch_hint_all(&self.cold_hashes[off..off + self.scan_lanes]);
            }

            let src_sketch = &sketch_scratch[local_src * m..(local_src + 1) * m];
            self.with_locked(global_src, |hot, cold| {
                for &(dst_local, dist) in &edges[start..end] {
                    let global_dst = point_ids[dst_local as usize];
                    let dst_sketch =
                        &sketch_scratch[dst_local as usize * m..(dst_local as usize + 1) * m];
                    debug_assert!(m <= 16, "num_planes <= 16 enforced by validate");
                    let hash = self.relative_hash.call(RelativeHashArgs {
                        src: src_sketch.as_ptr(),
                        dst: dst_sketch.as_ptr(),
                        len: m,
                    });
                    // SAFETY: cold ptrs from with_locked are valid for scan_lanes.
                    unsafe {
                        insert_locked(hot, cold, hash, global_dst, dist, l_max, self.find_hash)
                    };
                }
            });
        }
    }

    fn gather_sketches(&self, indices: &[u32], out: &mut [f32]) {
        let m = self.sketches.num_planes();
        let src = self.sketches.sketches();
        debug_assert_eq!(out.len(), indices.len() * m);
        for (i, &idx) in indices.iter().enumerate() {
            let g = idx as usize;
            out[i * m..(i + 1) * m].copy_from_slice(&src[g * m..(g + 1) * m]);
        }
    }

    /// Extract the nearest `max_degree` candidates retained by HashPrune.
    #[allow(clippy::disallowed_methods)] // build_graph installs the caller-owned pool.
    pub(crate) fn into_nearest_lists(self, max_degree: usize) -> Vec<AdjacencyList<u32>> {
        let scan_lanes = self.scan_lanes;
        drop(self.sketches);
        let HashPrune {
            hot,
            cold_hashes,
            cold_distances,
            cold_neighbors,
            ..
        } = self;
        drop(cold_hashes);
        (0..hot.len())
            .into_par_iter()
            .map(|i| {
                let off = i * scan_lanes;
                // SAFETY: extraction owns every row, so no mutation remains.
                let hot = unsafe { &*hot[i].get() };
                // SAFETY: i < hot.len() == npoints, so off + scan_lanes is
                // within both cold slabs.
                let nbrs = unsafe {
                    collect_sorted_neighbors(
                        hot,
                        cold_distances.as_ptr().wrapping_add(off),
                        cold_neighbors.as_ptr().wrapping_add(off),
                        max_degree,
                    )
                };
                let ids = nbrs.into_iter().map(|(id, _)| id).collect();
                // A neighbor always has the same relative hash for this source;
                // insertion replaces an existing hash slot instead of appending.
                AdjacencyList::from_vec_trusted(ids)
            })
            .collect()
    }

    /// Extract each point's full reservoir as candidate IDs. Drops the hashes
    /// and distances slabs (2/3 of the reservoir) before materializing the copy,
    /// so only the neighbors slab overlaps it.
    #[allow(clippy::disallowed_methods)] // build_graph installs the caller-owned pool.
    pub(crate) fn into_candidate_lists(self) -> Vec<AdjacencyList<u32>> {
        let cap = self.l_max;
        let scan_lanes = self.scan_lanes;
        drop(self.sketches);
        let HashPrune {
            hot,
            cold_hashes,
            cold_distances,
            cold_neighbors,
            ..
        } = self;
        // Neither the hashes (LSH dedup index) nor the distances (bf16
        // keep-closer key) are read again — free them before the copy so the
        // reservoir+copy overlap is just the neighbors slab.
        drop(cold_hashes);
        drop(cold_distances);
        (0..hot.len())
            .into_par_iter()
            .map(|i| {
                let neighbors = cold_neighbors.as_ptr().wrapping_add(i * scan_lanes);
                // SAFETY: extraction owns every row; no mutation remains.
                let hot = unsafe { &*hot[i].get() };
                // SAFETY: i < hot.len() == npoints; the row spans scan_lanes
                // slots and hot.len <= l_max <= scan_lanes.
                let ids = unsafe { collect_neighbor_ids(hot, neighbors, cap) };
                // Reservoir slots have unique hashes, and one neighbor cannot
                // produce two hashes for the same source.
                AdjacencyList::from_vec_trusted(ids)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
