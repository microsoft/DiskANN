/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! HashPrune: LSH-based online pruning for merging edges from overlapping partitions.
//!
//! ```text
//! dataset ──> random-hyperplane sketches (one sketch per point)
//!                                  │
//! leaf-local symmetric CSR ──> gather leaf sketches
//!                                  │
//!                                  v
//!                     relative hash(source, target)
//!                                  │
//!                                  v
//!              lock source reservoir ──> update bounded candidates
//!                                  │
//!                    consume HashPrune at stage exit
//!                       ┌──────────┴──────────┐
//!                       v                     v
//!              nearest lists         unsorted candidate lists
//!                                      (for RobustPrune)
//! ```
//!
//! Each source reservoir retains at most one neighbor per relative hash bucket:
//!
//! | Existing state | Incoming edge | Action |
//! | --- | --- | --- |
//! | matching hash | closer | replace that bucket |
//! | matching hash | not closer | reject |
//! | free bucket | any distance | append |
//! | full | closer than cached farthest | replace farthest |
//! | full | not closer | reject before scanning hashes |
//!
//! Storage is AoSoA hot/cold split:
//! - `hot: Vec<LockedHotSlot>` — one 16-byte slot per point with a mutex beside
//!   an `UnsafeCell<HotSlot>` containing len/farthest state.
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

use super::{
    bf16::{bf16_to_f32, f32_to_bf16},
    lsh::{LshSketchError, LshSketches},
};
use bytemuck::Pod;
use crate::{graph::AdjacencyList, utils::VectorRepr, ANNError, ANNResult};
use diskann_vector::{prefetch_hint_all, prefetch_hint_all_raw};
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
// SAFETY: the slab uniquely owns its mmap region until Drop; moving the slab
// transfers that ownership, and `T: Send` permits its initialized values to move.
unsafe impl<T: Pod + Send> Send for MmapSlab<T> {}
#[cfg(target_os = "linux")]
// SAFETY: safe shared access exposes only `*const T` or `&[T]`, and `T: Sync`.
// Raw mutation is possible only inside this module and requires an unsafe block;
// HashPrune places the slab in UnsafeCell before performing such mutation.
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
            // SAFETY: this slab still uniquely owns the mmap base pointer and exact
            // byte count established by `new_zeroed`; `self.len > 0` excludes the
            // dangling zero-length representation.
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
        // SAFETY: `new_zeroed` stores an aligned mmap base covering
        // `len * size_of::<T>()` live bytes; anonymous pages are zero-initialized,
        // and every zero bit pattern is valid because `T: Pod`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Windows counterpart of the Linux mmap slab. `VirtualAlloc(MEM_RESERVE |
/// MEM_COMMIT, PAGE_READWRITE)` reserves a zero-backed anonymous range whose
/// pages fault in on first write — the same lazy-commit behavior as
/// `mmap(MAP_ANONYMOUS)`, rather than the fallback `Vec`'s eager initialization.
#[cfg(windows)]
mod winmem {
    // Minimal FFI to the Win32 memory API — avoids pulling in the `windows`
    // crate for four extern declarations.
    pub(super) type Lpvoid = *mut core::ffi::c_void;
    pub(super) const MEM_COMMIT: u32 = 0x0000_1000;
    pub(super) const MEM_RESERVE: u32 = 0x0000_2000;
    pub(super) const MEM_RELEASE: u32 = 0x0000_8000;
    pub(super) const PAGE_READWRITE: u32 = 0x04;

    unsafe extern "system" {
        pub(super) fn VirtualAlloc(
            lpAddress: Lpvoid,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> Lpvoid;
        pub(super) fn VirtualFree(lpAddress: Lpvoid, dwSize: usize, dwFreeType: u32) -> i32;
    }
}

#[cfg(windows)]
struct MmapSlab<T: Pod> {
    ptr: *mut T,
    len: usize,
}

#[cfg(windows)]
// SAFETY: the slab uniquely owns its VirtualAlloc region until Drop; moving
// transfers that ownership, and `T: Send` permits its initialized values to move.
unsafe impl<T: Pod + Send> Send for MmapSlab<T> {}
#[cfg(windows)]
// SAFETY: safe shared access exposes only `*const T` or `&[T]`, and `T: Sync`.
// HashPrune places the slab in UnsafeCell before any raw mutation.
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
            // SAFETY: this slab still uniquely owns the VirtualAlloc base pointer;
            // MEM_RELEASE requires and receives `dwSize = 0`.
            unsafe {
                winmem::VirtualFree(self.ptr as winmem::Lpvoid, 0, winmem::MEM_RELEASE);
            }
        }
    }
}

#[cfg(windows)]
impl<T: Pod> std::ops::Deref for MmapSlab<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `new_zeroed` stores an aligned VirtualAlloc base covering
        // `len * size_of::<T>()` live bytes; Windows zero-initializes the region,
        // and every zero bit pattern is valid because `T: Pod`.
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
            .map_err(ANNError::new)
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
            super::config_error(format!("num_hash_planes ({actual}) must be in 1..={max}"))
        }
        LshSketchError::ShapeOverflow { rows, columns } => ANNError::message(format!(
            "LSH matrix shape {rows} x {columns} overflows usize"
        )),
        LshSketchError::Allocation(error) => ANNError::new(error),
        LshSketchError::Fill(error) => error.into(),
    })
}

// ─── HotSlot: 16-byte per-point mutex + cached fields ─────────────────────────

#[repr(C)]
struct HotSlot {
    len: u8,
    farthest_idx: u8,
    farthest_dist: u16,
    _pad: [u8; 10],
}

#[repr(C, align(16))]
struct LockedHotSlot {
    lock: parking_lot::RawMutex,
    state: UnsafeCell<HotSlot>,
}

impl LockedHotSlot {
    fn new() -> Self {
        Self {
            lock: <parking_lot::RawMutex as RawMutexTrait>::INIT,
            state: UnsafeCell::new(HotSlot::new_empty()),
        }
    }

    fn get(&self) -> *mut HotSlot {
        self.state.get()
    }

    fn with_state<R>(&self, f: impl FnOnce(&mut HotSlot) -> R) -> R {
        struct UnlockOnDrop<'a>(&'a parking_lot::RawMutex);
        impl Drop for UnlockOnDrop<'_> {
            fn drop(&mut self) {
                // SAFETY: the guard is created only after acquiring this mutex.
                unsafe { self.0.unlock() };
            }
        }

        self.lock.lock();
        let _guard = UnlockOnDrop(&self.lock);
        // SAFETY: the mutex is separate from `state`, so contending threads may
        // access the lock while this exclusive state reference is live.
        f(unsafe { &mut *self.state.get() })
    }
}

// SAFETY: `lock` guards every mutable access to `state`. Read-only extraction
// happens only after HashPrune is consumed, when no mutation can remain.
unsafe impl Sync for LockedHotSlot {}

impl HotSlot {
    const fn new_empty() -> Self {
        Self {
            len: 0,
            farthest_idx: 0,
            farthest_dist: 0,
            _pad: [0; 10],
        }
    }
}

const _: () = assert!(std::mem::size_of::<LockedHotSlot>() == 16);

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
        // SAFETY: production args are created inside `insert_locked` from a
        // `ColdSlotPtrs` hash segment valid for `scan_lanes` elements. `chunks`
        // is capped at `scan_lanes / F::LANES`, so this full load stays inside it.
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

    // SAFETY: the production constructor takes `src` and `dst` from slices of
    // exactly `len` sketch values. HashPrune validates `len <= 16 <= F::LANES`,
    // and masked loads do not access inactive lanes.
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

/// # Safety
///
/// The source slot lock is held, `hot.len <= cold.scan_lanes`, and the first
/// `hot.len` entries of every cold array are initialized.
#[inline]
unsafe fn update_farthest(hot: &mut HotSlot, cold: ColdSlotPtrs) {
    if hot.len == 0 {
        hot.farthest_dist = 0;
        hot.farthest_idx = 0;
        return;
    }
    // Distance is stored as bf16, so distinct candidates frequently tie after
    // quantization. The residual hash is already a seeded, ID-order-independent
    // discriminator; the neighbor ID is only the final total-order fallback.
    // This preserves the paper's history-independence guarantee under ties
    // without systematically favoring low dataset IDs.
    let mut max_idx: u8 = 0;
    // SAFETY: `hot.len > 0` and all active slots are initialized.
    let mut max_key = unsafe { (*cold.distances, *cold.hashes, *cold.neighbors) };
    for i in 1..hot.len as usize {
        // SAFETY: guaranteed by this function's contract.
        let key = unsafe {
            (
                *cold.distances.add(i),
                *cold.hashes.add(i),
                *cold.neighbors.add(i),
            )
        };
        if key > max_key {
            max_key = key;
            max_idx = i as u8;
        }
    }
    hot.farthest_dist = max_key.0;
    hot.farthest_idx = max_idx;
}

/// # Safety
///
/// The source slot lock is held; each cold pointer is valid for `scan_lanes`
/// elements; `hot.len <= l_max <= scan_lanes`; and the first `hot.len` entries
/// of every cold array are initialized.
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

    if hot.len >= l_max {
        let farthest = hot.farthest_idx as usize;
        // SAFETY: a full reservoir has `farthest < hot.len` initialized slots.
        let farthest_key = unsafe {
            (
                hot.farthest_dist,
                *cold.hashes.add(farthest),
                *cold.neighbors.add(farthest),
            )
        };
        if (dist_key, hash, neighbor) >= farthest_key {
            return false;
        }
    }

    if let Some(idx) = find_hash.call(FindHashArgs {
        hashes: cold.hashes,
        scan_lanes: cold.scan_lanes,
        len: hot.len,
        target: hash,
    }) {
        // SAFETY: `idx < hot.len <= cold.scan_lanes`.
        let current_key = unsafe { (*cold.distances.add(idx), *cold.neighbors.add(idx)) };
        if (dist_key, neighbor) < current_key {
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
        let becomes_farthest = if hot.len == 0 {
            true
        } else {
            let farthest = hot.farthest_idx as usize;
            // SAFETY: `farthest < hot.len` identifies an initialized slot.
            let farthest_key = unsafe {
                (
                    hot.farthest_dist,
                    *cold.hashes.add(farthest),
                    *cold.neighbors.add(farthest),
                )
            };
            (dist_key, hash, neighbor) > farthest_key
        };
        // SAFETY: `new_idx < l_max <= cold.scan_lanes`; the caller holds the lock.
        unsafe {
            *cold.hashes.add(new_idx) = hash;
            *cold.distances.add(new_idx) = dist_key;
            *cold.neighbors.add(new_idx) = neighbor;
        }
        hot.len += 1;
        if becomes_farthest {
            hot.farthest_dist = dist_key;
            hot.farthest_idx = new_idx as u8;
        }
        return true;
    }

    // The full-reservoir early rejection above proved that the incoming
    // `(distance, residual hash, ID)` key is better than the cached farthest key.
    let idx = hot.farthest_idx as usize;
    // SAFETY: `idx < hot.len <= cold.scan_lanes`; the caller holds the lock.
    unsafe {
        *cold.hashes.add(idx) = hash;
        *cold.distances.add(idx) = dist_key;
        *cold.neighbors.add(idx) = neighbor;
        update_farthest(hot, cold);
    }
    true
}

/// Collect the reservoir's entries sorted by distance, truncated to `cap`.
/// A Rayon-owned scratch `Vec`, sized to the reservoir's runtime fill and
/// reused within one extraction job, avoids per-reservoir allocation.
///
/// # Safety
///
/// Mutation is excluded by the slot lock or unique ownership, and `distances`
/// and `neighbors` each point to at least `hot.len` initialized entries.
unsafe fn collect_sorted_neighbors(
    hot: &HotSlot,
    distances: *const u16,
    neighbors: *const u32,
    cap: usize,
    scratch: &mut Vec<(u32, u16)>,
) -> Vec<(u32, f32)> {
    let n = hot.len as usize;
    scratch.clear();
    scratch.reserve(n);
    for i in 0..n {
        // SAFETY: guaranteed by this function's contract.
        scratch.push(unsafe { (*neighbors.add(i), *distances.add(i)) });
    }
    scratch.sort_unstable_by_key(|&(id, distance)| (distance, id));
    let out_len = n.min(cap);
    let mut out = Vec::with_capacity(out_len);
    for &(id, d) in &scratch[..out_len] {
        out.push((id, bf16_to_f32(key_to_bf16(d))));
    }
    out
}

/// Collect the reservoir's neighbor ids, truncated to `cap`, WITHOUT sorting.
/// Reservoir order is intentionally not preserved. Reading only `neighbors`
/// lets the caller drop the hashes and distances slabs before extraction; any
/// ordering required by a later graph-finalization policy belongs to that caller.
///
/// # Safety
///
/// Mutation is excluded by the slot lock or unique ownership, and `neighbors`
/// points to at least `hot.len` initialized entries.
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

/// Global bounded reservoirs shared by parallel leaf workers.
///
/// Row `i` consists of `hot[i]` and the `i * scan_lanes` range in each cold
/// slab. The point mutex is the sole synchronization boundary: callers never
/// hold two point locks, and sketch computation/extraction happens without a
/// lock. Consuming extraction proves that no writer can remain.
pub(crate) struct HashPrune {
    hot: Vec<LockedHotSlot>,
    /// AoSoA hashes slab: `npoints * scan_lanes` u16.
    cold_hashes: UnsafeCell<MmapSlab<u16>>,
    /// AoSoA distances slab (bf16 in u16): `npoints * scan_lanes`.
    cold_distances: UnsafeCell<MmapSlab<u16>>,
    /// AoSoA neighbors slab: `npoints * scan_lanes` u32.
    cold_neighbors: UnsafeCell<MmapSlab<u32>>,
    /// Per-slot stride. Equals `l_max.next_multiple_of(32).max(32)`. Always a
    /// multiple of 32 so the AVX-512 / AVX-2 find_hash scan stays aligned.
    scan_lanes: usize,
    sketches: LshSketches,
    l_max: usize,
    find_hash: FindHash,
    relative_hash: RelativeHash,
}

// SAFETY: every mutable hot/cold access is inside an UnsafeCell and guarded by
// the corresponding `LockedHotSlot::lock`. Different locks cover disjoint cold
// ranges, and consuming extraction proves no writer remains.
unsafe impl Send for HashPrune {}
// SAFETY: the same per-point lock protects mutation through shared HashPrune
// references; immutable sketches are safe to share.
unsafe impl Sync for HashPrune {}

impl HashPrune {
    /// Precompute immutable sketches and allocate one lazy-backed reservoir
    /// per dataset point.
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
        let scan_lanes = l_max.next_multiple_of(32).max(32);

        // Hot slab: one HotSlot per point, contiguous.
        let mut hot: Vec<LockedHotSlot> = Vec::new();
        hot.try_reserve_exact(npoints)
            .map_err(ANNError::new)
            .map_err(|error| error.context(format!("reserving {npoints} HashPrune reservoirs")))?;
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
            cold_hashes: UnsafeCell::new(cold_hashes),
            cold_distances: UnsafeCell::new(cold_distances),
            cold_neighbors: UnsafeCell::new(cold_neighbors),
            scan_lanes,
            sketches,
            l_max,
            find_hash: select_find_hash(),
            relative_hash: select_relative_hash(),
        })
    }

    /// Locks the per-slot mutex at `idx`, runs `f` with mutable access to the
    /// hot state and its cold-slab pointers, and unlocks on return or panic.
    #[inline(always)]
    fn with_locked<R>(&self, idx: usize, f: impl FnOnce(&mut HotSlot, ColdSlotPtrs) -> R) -> R {
        assert!(idx < self.hot.len(), "HashPrune point index out of bounds");
        let off = idx * self.scan_lanes;
        // SAFETY: `idx` is bounds-checked, each slab has
        // `hot.len() * scan_lanes` elements, and its UnsafeCell permits mutation
        // through this shared HashPrune reference while the point lock is held.
        let cold = unsafe {
            ColdSlotPtrs {
                hashes: (*self.cold_hashes.get()).as_ptr().cast_mut().add(off),
                distances: (*self.cold_distances.get()).as_ptr().cast_mut().add(off),
                neighbors: (*self.cold_neighbors.get()).as_ptr().cast_mut().add(off),
                scan_lanes: self.scan_lanes,
            }
        };
        self.hot[idx].with_state(|hot| f(hot, cold))
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
        if sketch_scratch.len() < sketch_len {
            sketch_scratch.resize(sketch_len, 0.0);
        }
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
                // SAFETY: `next` is a dataset point ID, so this raw range is the
                // complete padded hash segment for that point. Raw prefetch avoids
                // creating a shared slice while another worker mutates the segment.
                unsafe {
                    let hashes = (*self.cold_hashes.get()).as_ptr().add(off);
                    prefetch_hint_all_raw(
                        hashes.cast(),
                        self.scan_lanes * std::mem::size_of::<u16>(),
                    );
                }
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
                    // SAFETY: `with_locked` holds this source's lock for the
                    // closure and supplies its exact `scan_lanes` cold segments.
                    // `l_max` was validated at construction and `insert_locked`
                    // maintains initialized entries through `hot.len`.
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
        let cold_hashes = cold_hashes.into_inner();
        let cold_distances = cold_distances.into_inner();
        let cold_neighbors = cold_neighbors.into_inner();
        drop(cold_hashes);
        (0..hot.len())
            .into_par_iter()
            .map_init(Vec::new, |scratch, i| {
                let off = i * scan_lanes;
                // SAFETY: indexing proves `i` names a live slot, and consuming
                // `self` gives this extraction unique ownership, so no mutable
                // access can overlap the returned state reference.
                let hot = unsafe { &*hot[i].get() };
                // SAFETY: construction allocated `npoints * scan_lanes` entries;
                // this loop keeps `i < npoints`, and insertion maintains
                // `hot.len <= l_max <= scan_lanes` initialized entries.
                let nbrs = unsafe {
                    collect_sorted_neighbors(
                        hot,
                        cold_distances.as_ptr().wrapping_add(off),
                        cold_neighbors.as_ptr().wrapping_add(off),
                        max_degree,
                        scratch,
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
        let cold_hashes = cold_hashes.into_inner();
        let cold_distances = cold_distances.into_inner();
        let cold_neighbors = cold_neighbors.into_inner();
        // Neither the hashes (LSH dedup index) nor the distances (bf16
        // keep-closer key) are read again — free them before the copy so the
        // reservoir+copy overlap is just the neighbors slab.
        drop(cold_hashes);
        drop(cold_distances);
        (0..hot.len())
            .into_par_iter()
            .map(|i| {
                let neighbors = cold_neighbors.as_ptr().wrapping_add(i * scan_lanes);
                // SAFETY: indexing proves `i` names a live slot, and consuming
                // `self` gives this extraction unique ownership, so no mutable
                // access can overlap the returned state reference.
                let hot = unsafe { &*hot[i].get() };
                // SAFETY: construction allocated `npoints * scan_lanes` entries;
                // this loop keeps `i < npoints`, and insertion maintains
                // `hot.len <= l_max <= scan_lanes` initialized entries.
                let ids = unsafe { collect_neighbor_ids(hot, neighbors, cap) };
                // Reservoir slots have unique hashes, and one neighbor cannot
                // produce two hashes for the same source.
                AdjacencyList::from_vec_trusted(ids)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Reservoir {
        hot: HotSlot,
        hashes: Vec<u16>,
        distances: Vec<u16>,
        neighbors: Vec<u32>,
        scan_lanes: usize,
        l_max: u8,
    }

    impl Reservoir {
        fn new(l_max: usize) -> Self {
            assert!(l_max <= MAX_RESERVOIR_LEN);
            let scan_lanes = l_max.next_multiple_of(32).max(32);
            Self {
                hot: HotSlot::new_empty(),
                hashes: vec![0; scan_lanes],
                distances: vec![0; scan_lanes],
                neighbors: vec![0; scan_lanes],
                scan_lanes,
                l_max: l_max as u8,
            }
        }

        fn cold(&self) -> ColdSlotPtrs {
            ColdSlotPtrs {
                hashes: self.hashes.as_ptr() as *mut u16,
                distances: self.distances.as_ptr() as *mut u16,
                neighbors: self.neighbors.as_ptr() as *mut u32,
                scan_lanes: self.scan_lanes,
            }
        }

        fn insert(&mut self, hash: u16, neighbor: u32, distance: f32) -> bool {
            let cold = self.cold();
            // SAFETY: the test owns the reservoir and holds its only mutable reference.
            unsafe {
                insert_locked(
                    &mut self.hot,
                    cold,
                    hash,
                    neighbor,
                    distance,
                    self.l_max,
                    select_find_hash(),
                )
            }
        }

        fn neighbors(&self) -> Vec<(u32, f32)> {
            let cold = self.cold();
            let mut scratch = Vec::new();
            // SAFETY: the test owns the reservoir; all cold slabs span scan_lanes entries.
            unsafe {
                collect_sorted_neighbors(
                    &self.hot,
                    cold.distances,
                    cold.neighbors,
                    usize::MAX,
                    &mut scratch,
                )
            }
        }

        fn len(&self) -> usize {
            self.hot.len as usize
        }

        fn is_empty(&self) -> bool {
            self.hot.len == 0
        }
    }

    #[test]
    fn hot_slot_contention_serializes_state_mutation() {
        let slot = LockedHotSlot::new();
        let start = std::sync::Barrier::new(3);

        std::thread::scope(|scope| {
            for _ in 0..2 {
                let slot = &slot;
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    for _ in 0..16 {
                        slot.with_state(|hot| hot.farthest_dist += 1);
                    }
                });
            }
            start.wait();
        });

        assert_eq!(slot.with_state(|hot| hot.farthest_dist), 32);
    }

    fn add_edge(hp: &HashPrune, src: usize, dst: usize, distance: f32) {
        let m = hp.sketches.num_planes();
        let sketches = hp.sketches.sketches();
        let hash = hp.relative_hash.call(RelativeHashArgs {
            src: sketches[src * m..(src + 1) * m].as_ptr(),
            dst: sketches[dst * m..(dst + 1) * m].as_ptr(),
            len: m,
        });
        let l_max = hp.l_max as u8;
        hp.with_locked(src, |hot, cold| {
            // SAFETY: with_locked guards the reservoir and supplies valid cold-slab pointers.
            unsafe { insert_locked(hot, cold, hash, dst as u32, distance, l_max, hp.find_hash) };
        });
    }

    fn assert_sketch_source_type_matches_f32<T>(
        label: &str,
        convert: impl Fn(u8) -> T,
        reference: impl Fn(u8) -> f32,
    ) where
        T: VectorRepr + Send + Sync,
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let points = 5;
        for dimensions in [1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let raw: Vec<u8> = (0..points * dimensions)
                .map(|index| ((index * 7 + index / dimensions * 3) % 23) as u8)
                .collect();
            let converted: Vec<T> = raw.iter().copied().map(&convert).collect();
            let f32_data: Vec<f32> = raw.iter().copied().map(&reference).collect();
            for planes in [1, 8, 16] {
                let (actual, expected) = pool.install(|| {
                    (
                        sketches_from_data(&converted, points, dimensions, planes, 42).unwrap(),
                        sketches_from_data(&f32_data, points, dimensions, planes, 42).unwrap(),
                    )
                });
                assert_eq!(
                    actual.sketches(),
                    expected.sketches(),
                    "{label} dimensions={dimensions} planes={planes}"
                );
            }
        }
    }

    // Source conversion.

    #[test]
    fn f16_sketch_conversion_matches_f32_across_dimensions_and_planes() {
        assert_sketch_source_type_matches_f32(
            "f16",
            |value| half::f16::from_f32(value as f32),
            |value| value as f32,
        );
    }

    #[test]
    fn u8_sketch_conversion_matches_f32_across_dimensions_and_planes() {
        assert_sketch_source_type_matches_f32("u8", |value| value, |value| value as f32);
    }

    #[test]
    fn i8_sketch_conversion_matches_f32_across_dimensions_and_planes() {
        assert_sketch_source_type_matches_f32(
            "i8",
            |value| value as i8 - 11,
            |value| (value as i8 - 11) as f32,
        );
    }

    // Dispatched hash primitives.

    #[test]
    fn relative_hash_matches_numeric_reference() {
        let dispatched = select_relative_hash();

        let src = [
            1.0, -2.0, 0.0, 7.5, -0.0, 3.25, -9.0, 4.0, 8.0, -1.5, 2.0, 0.0, 6.0, -3.0, 5.5, -7.25,
        ];
        let dst = [
            1.0, -3.0, 0.5, 7.0, 0.0, 3.25, -8.0, -4.0, 9.0, -1.5, -2.0, -0.0, 5.0, -2.0, 5.5, -8.0,
        ];

        for m in 0..=16 {
            let mut expected = 0u16;
            for j in 0..m {
                let diff: f32 = dst[j] - src[j];
                expected |= ((diff >= 0.0) as u16) << j;
            }

            let actual = dispatched.call(RelativeHashArgs {
                src: src.as_ptr(),
                dst: dst.as_ptr(),
                len: m,
            });
            assert_eq!(actual, expected, "m={m}");
        }
    }

    #[test]
    fn relative_hash_defines_signed_zero_and_nan_buckets() {
        let src = [0.0; 4];
        let dst = [
            0.0,
            -0.0,
            f32::from_bits(0x7FC0_0000),
            f32::from_bits(0xFFC0_0000),
        ];

        assert_eq!(
            select_relative_hash().call(RelativeHashArgs {
                src: src.as_ptr(),
                dst: dst.as_ptr(),
                len: dst.len(),
            }),
            0b0011
        );
    }

    #[test]
    fn find_hash_handles_padded_boundaries_and_all_bit_patterns() {
        let dispatched = select_find_hash();

        for target in [0, 0xF00D] {
            for len in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 254, 255] {
                let scan_lanes = len.max(1).next_multiple_of(32);
                let mut hashes = vec![target; scan_lanes];
                hashes[..len].fill(0x8001);
                let args = |hashes: &[u16]| FindHashArgs {
                    hashes: hashes.as_ptr(),
                    scan_lanes,
                    len: len as u8,
                    target,
                };

                assert_eq!(dispatched.call(args(&hashes)), None, "len={len}");
                for index in [0, len / 2, len.saturating_sub(1)] {
                    if index < len {
                        hashes[index] = target;
                        assert_eq!(dispatched.call(args(&hashes)), Some(index), "len={len}");
                        hashes[index] = 0x8001;
                    }
                }
            }
        }
    }

    // Storage and configuration.

    #[test]
    fn slab_is_zeroed_and_reports_its_bytes() {
        let slab = MmapSlab::<u32>::new_zeroed(4).unwrap();
        assert_eq!(slab.bytes(), 4 * std::mem::size_of::<u32>());
        assert_eq!(slab.len(), 4);
        assert!(!slab.as_ptr().is_null());
        assert_eq!(&*slab, &[0; 4]);
    }

    #[test]
    fn accepts_structural_l_max_boundaries() {
        let data = [0.0_f32];
        let low = HashPrune::new(&data, 1, 1, 1, 1, 42).unwrap();
        assert_eq!(low.l_max, 1);
        assert_eq!(low.scan_lanes, 32);

        let high = HashPrune::new(&data, 1, 1, 1, MAX_RESERVOIR_LEN, 42).unwrap();
        assert_eq!(high.l_max, MAX_RESERVOIR_LEN);
        assert_eq!(high.scan_lanes, 256);
    }

    #[test]
    fn rejects_l_max_outside_structural_boundaries() {
        for l_max in [0, MAX_RESERVOIR_LEN + 1] {
            let result = HashPrune::new(&[0.0_f32], 1, 1, 1, l_max, 42);
            let error = match result {
                Ok(_) => panic!("l_max={l_max} must be rejected"),
                Err(error) => error,
            };
            assert!(format!("{error:?}").contains(&format!("l_max ({l_max})")));
        }
    }

    #[test]
    fn ordered_key_roundtrips_bf16_order_for_all_signs() {
        let values = [
            f32::NEG_INFINITY,
            -100.0,
            -0.0,
            0.0,
            0.25,
            100.0,
            f32::INFINITY,
        ];
        let keys: Vec<_> = values.iter().copied().map(ordered_key).collect();
        assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
        for (value, key) in values.into_iter().zip(keys) {
            assert_eq!(
                bf16_to_f32(key_to_bf16(key)),
                bf16_to_f32(f32_to_bf16(value))
            );
        }
    }

    // Leaf ingestion and scratch reuse.

    #[test]
    fn batched_leaf_edges_match_single_edge_reference() {
        let data = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let batched = HashPrune::new(&data, 4, 2, 8, 8, 42).unwrap();
        let reference = HashPrune::new(&data, 4, 2, 8, 8, 42).unwrap();
        let point_ids = [0, 1, 2, 3];
        let offsets = [0, 3, 6, 9, 12];
        let edges = [
            (1, 1.0),
            (2, 1.0),
            (3, 2.0),
            (0, 1.0),
            (2, 2.0),
            (3, 1.0),
            (0, 1.0),
            (1, 2.0),
            (3, 1.0),
            (0, 2.0),
            (1, 1.0),
            (2, 1.0),
        ];
        let mut scratch = Vec::new();

        batched.add_leaf_edges(&point_ids, &offsets, &edges, &mut scratch);
        for source in 0..point_ids.len() {
            for &(target, distance) in
                &edges[offsets[source] as usize..offsets[source + 1] as usize]
            {
                add_edge(&reference, source, target as usize, distance);
            }
        }
        let canonicalize = |lists: Vec<crate::graph::AdjacencyList<u32>>| {
            lists
                .into_iter()
                .map(|candidates| {
                    let mut ids = candidates.to_vec();
                    ids.sort_unstable();
                    ids
                })
                .collect::<Vec<_>>()
        };
        let actual = canonicalize(batched.into_candidate_lists());
        let expected = canonicalize(reference.into_candidate_lists());

        assert_eq!(actual, expected);
        assert!(actual.iter().all(|candidates| !candidates.is_empty()));
    }

    #[test]
    fn leaf_edges_grow_then_reuse_sketch_scratch() {
        let data = [0.0_f32, 1.0, 2.0, 3.0];
        let hp = HashPrune::new(&data, 4, 1, 8, 4, 42).unwrap();
        let mut scratch = vec![99.0; 1];

        hp.add_leaf_edges(&[0, 1], &[0, 1, 2], &[(1, 1.0), (0, 1.0)], &mut scratch);
        assert_eq!(scratch.len(), 16);
        let capacity = scratch.capacity();

        hp.add_leaf_edges(&[2, 3], &[0, 1, 2], &[(1, 1.0), (0, 1.0)], &mut scratch);
        assert_eq!(scratch.len(), 16);
        assert_eq!(scratch.capacity(), capacity);

        hp.add_leaf_edges(&[0, 1], &[0, 0, 0], &[], &mut scratch);
        assert_eq!(scratch.len(), 16);
        assert_eq!(scratch.capacity(), capacity);
        assert!(
            hp.into_candidate_lists()
                .iter()
                .all(|candidates| candidates.len() == 1)
        );
    }

    // Reservoir replacement and ordering policy.

    #[test]
    fn full_reservoir_evicts_the_farthest_candidate() {
        let mut reservoir = Reservoir::new(3);
        assert!(reservoir.is_empty());

        assert!(reservoir.insert(0, 1, 1.0));
        assert!(reservoir.insert(1, 2, 2.0));
        assert!(reservoir.insert(2, 3, 3.0));
        assert!(reservoir.insert(3, 4, 0.5));

        assert_eq!(reservoir.len(), 3);
        assert_eq!(reservoir.neighbors(), [(4, 0.5), (1, 1.0), (2, 2.0)]);
    }

    #[test]
    fn same_hash_keeps_only_the_closest_candidate() {
        let mut reservoir = Reservoir::new(5);

        assert!(reservoir.insert(0, 1, 3.0));
        assert!(reservoir.insert(0, 2, 2.0));
        assert!(reservoir.insert(0, 3, 1.0));
        assert!(!reservoir.insert(0, 4, 5.0));

        assert_eq!(reservoir.len(), 1);
        assert_eq!(reservoir.neighbors(), [(3, 1.0)]);
    }

    #[test]
    fn equal_distances_are_ordered_by_neighbor_id() {
        let mut res = Reservoir::new(5);
        res.insert(0, 1, 1.0);
        res.insert(1, 2, 1.0);
        res.insert(2, 3, 1.0);
        assert_eq!(res.len(), 3);
        assert_eq!(res.neighbors(), [(1, 1.0), (2, 1.0), (3, 1.0)]);
    }

    #[test]
    fn same_hash_bf16_ties_are_history_independent() {
        for order in [[0, 1], [1, 0]] {
            let candidates = [(7, 20, 1.0), (7, 10, 1.0)];
            let mut reservoir = Reservoir::new(2);
            for index in order {
                let (hash, neighbor, distance) = candidates[index];
                reservoir.insert(hash, neighbor, distance);
            }
            assert_eq!(reservoir.neighbors(), [(10, 1.0)], "order={order:?}");
        }
    }

    #[test]
    fn full_reservoir_bf16_ties_are_history_independent() {
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let candidates = [(1, 30, 1.0), (2, 10, 1.0), (3, 20, 1.0)];
        for order in permutations {
            let mut reservoir = Reservoir::new(2);
            for index in order {
                let (hash, neighbor, distance) = candidates[index];
                reservoir.insert(hash, neighbor, distance);
            }
            let mut actual = reservoir.neighbors();
            actual.sort_unstable_by_key(|&(neighbor, _)| neighbor);
            assert_eq!(actual, [(10, 1.0), (30, 1.0)], "order={order:?}");
        }
    }

    // Concurrency and consuming extraction.

    #[test]
    #[allow(clippy::disallowed_methods)]
    fn parallel_insertion_matches_serial_neighbor_lists() {
        use rayon::prelude::*;

        let data = vec![0.0f32; 100 * 4];
        let parallel = HashPrune::new(&data, 100, 4, 4, 10, 42).unwrap();
        let serial = HashPrune::new(&data, 100, 4, 4, 10, 42).unwrap();

        (0..50).into_par_iter().for_each(|source| {
            add_edge(&parallel, source, (source + 1) % 100, 1.0);
            add_edge(&parallel, (source + 1) % 100, source, 1.0);
        });
        for source in 0..50 {
            add_edge(&serial, source, (source + 1) % 100, 1.0);
            add_edge(&serial, (source + 1) % 100, source, 1.0);
        }

        assert_eq!(parallel.into_nearest_lists(5), serial.into_nearest_lists(5));
    }

    #[test]
    fn extraction_returns_full_candidates_and_truncates_to_nearest_degree() {
        #[rustfmt::skip]
        let data = [
             0.0,  0.0,
             1.0,  0.0,
             0.0,  1.0,
            -1.0,  0.0,
             0.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
             1.0, -1.0,
        ];
        let full = HashPrune::new(&data, 8, 2, 16, 10, 42).unwrap();
        let nearest = HashPrune::new(&data, 8, 2, 16, 10, 42).unwrap();
        for target in 1..8 {
            add_edge(&full, 0, target, target as f32);
            add_edge(&nearest, 0, target, target as f32);
        }

        let mut full_ids = full.into_candidate_lists()[0].to_vec();
        full_ids.sort_unstable();
        assert_eq!(full_ids, (1..8).collect::<Vec<_>>());
        assert_eq!(&*nearest.into_nearest_lists(2)[0], &[1, 2]);
    }

    #[test]
    fn farthest_cache_updates_after_repeated_evictions() {
        let mut reservoir = Reservoir::new(3);
        reservoir.insert(0, 10, 5.0);
        reservoir.insert(1, 11, 4.0);
        reservoir.insert(2, 12, 3.0);
        assert!(reservoir.insert(3, 13, 2.0));
        assert!(reservoir.insert(4, 14, 1.0));

        assert_eq!(reservoir.neighbors(), [(14, 1.0), (13, 2.0), (12, 3.0)]);
    }

    #[test]
    fn sorted_extraction_handles_an_early_farthest_slot() {
        let mut reservoir = Reservoir::new(4);
        reservoir.insert(5, 1, 1.0);
        reservoir.insert(10, 2, 3.0);
        reservoir.insert(15, 3, 2.0);
        reservoir.insert(3, 4, 0.5);

        assert_eq!(
            reservoir.neighbors(),
            [(4, 0.5), (1, 1.0), (3, 2.0), (2, 3.0)]
        );
    }
}
