/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Merge leaf edges into bounded per-point reservoirs.
//!
//! For edge `source → target`, relative-hash bit `j` records whether the target's
//! projection on hyperplane `j` is at least the source's projection. The hash
//! groups edges with similar residual directions.
//!
//! A source reservoir keeps at most one neighbor for each relative hash. A closer
//! edge replaces the edge for that direction. A full reservoir accepts only an
//! edge below its farthest total key.
//!
//! Each source owns one lock. The lock protects its reservoir metadata and its
//! rows in the hash, distance, and neighbor arrays. `l_max` sets the logical
//! reservoir length. The `u8` metadata limits this value to 255.

use parking_lot::lock_api::RawMutex as RawMutexTrait;
use std::cell::UnsafeCell;

use super::{bf16::f32_to_bf16, lsh::LshSketches};
use crate::{ANNError, ANNResult, graph::AdjacencyList, utils::VectorRepr};
use bytemuck::Pod;
use diskann_utils::views::MatrixView;
use diskann_vector::{prefetch_hint_all, prefetch_hint_all_raw};
use diskann_wide::{
    Architecture, SIMDMask, SIMDPartialEq, SIMDPartialOrd, SIMDVector,
    arch::{self, Dispatched1, FTarget1, Target},
    lifetime::As,
};
use rayon::prelude::*;

/// Owned zero-initialized slab from `mmap(MAP_PRIVATE | MAP_ANONYMOUS)`.
#[cfg(target_os = "linux")]
struct MmapSlab<T: Pod> {
    ptr: *mut T,
    len: usize,
}

#[cfg(target_os = "linux")]
// SAFETY: the slab uniquely owns its mmap region until `drop`. Moving the slab
// transfers that ownership. `T: Send` permits transfer of initialized values.
unsafe impl<T: Pod + Send> Send for MmapSlab<T> {}
#[cfg(target_os = "linux")]
// SAFETY: shared access exposes only `*const T`. `T: Sync` permits shared access
// to initialized values. HashPrune uses `UnsafeCell` and a point lock for writes.
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
            .ok_or_else(|| super::config_error(format!("slab size {len} overflows usize")))?;
        // SAFETY: `MAP_ANONYMOUS` returns zero-initialized memory.
        // `PROT_READ | PROT_WRITE` permits all accesses used by this slab.
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

/// Owned zero-initialized slab from `VirtualAlloc`.
#[cfg(windows)]
mod winmem {
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
// SAFETY: the slab uniquely owns its `VirtualAlloc` region until `drop`. Moving
// the slab transfers that ownership. `T: Send` permits transfer of initialized values.
unsafe impl<T: Pod + Send> Send for MmapSlab<T> {}
#[cfg(windows)]
// SAFETY: shared access exposes only `*const T`. `T: Sync` permits shared access
// to initialized values. HashPrune uses `UnsafeCell` and a point lock for writes.
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
            .ok_or_else(|| super::config_error(format!("slab size {len} overflows usize")))?;
        // SAFETY: `MEM_RESERVE | MEM_COMMIT` returns zero-initialized memory.
        // `PAGE_READWRITE` permits all accesses used by this slab.
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

/// Owned zero-initialized slab for platforms without `mmap` or `VirtualAlloc`.
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
}

/// Largest reservoir length that fits in `ReservoirState`.
///
/// `ReservoirState.len` and `ReservoirState.farthest_idx` are `u8`. Runtime `l_max` selects the
/// actual length. Values above this bound are invalid.
pub(crate) const MAX_RESERVOIR_LEN: usize = u8::MAX as usize;

#[repr(C)]
struct ReservoirState {
    len: u8,
    farthest_idx: u8,
    farthest_dist: u16,
    _pad: [u8; 10],
}

#[repr(C, align(16))]
struct LockedReservoirState {
    lock: parking_lot::RawMutex,
    state: UnsafeCell<ReservoirState>,
}

impl LockedReservoirState {
    fn new() -> Self {
        Self {
            lock: <parking_lot::RawMutex as RawMutexTrait>::INIT,
            state: UnsafeCell::new(ReservoirState::new_empty()),
        }
    }

    fn state_ptr(&self) -> *mut ReservoirState {
        self.state.get()
    }

    fn with_locked_state<R>(&self, f: impl FnOnce(&mut ReservoirState) -> R) -> R {
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
unsafe impl Sync for LockedReservoirState {}

impl ReservoirState {
    const fn new_empty() -> Self {
        Self {
            len: 0,
            farthest_idx: 0,
            farthest_dist: 0,
            _pad: [0; 10],
        }
    }
}

const _: [(); 16] = [(); std::mem::size_of::<LockedReservoirState>()];

// These pointers name one source reservoir's hash, distance, and neighbor rows.
// The caller must hold that source lock before it writes through a pointer.

#[derive(Clone, Copy)]
struct ReservoirRows {
    hashes: *mut u16,
    distances: *mut u16,
    neighbors: *mut u32,
    row_stride: usize,
}

#[derive(Clone, Copy)]
struct FindHashArgs {
    hashes: *const u16,
    row_stride: usize,
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

/// Find an existing relative-direction bucket in one source reservoir.
///
/// The SIMD backend has no `u16` vector. An `i16` load keeps each hash bit
/// pattern, so equality gives the same result.
fn find_hash_simd<F>(arch: F::Arch, args: FindHashArgs) -> Option<usize>
where
    F: SIMDVector<Scalar = i16> + SIMDPartialEq,
{
    let len = args.len as usize;
    let target = F::splat(arch, args.target as i16);
    let chunks = len.div_ceil(F::LANES).min(args.row_stride / F::LANES);
    for chunk in 0..chunks {
        // SAFETY: `insert_reservoir_edge` supplies a hash row with `row_stride` elements.
        // `chunks <= row_stride / F::LANES`, so this full load stays in the row.
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

/// Return the relative hash for two sketches.
///
/// Bit `j` is one when `dst[j] - src[j] >= 0.0`. Equality and signed zero set
/// the bit on every architecture.
fn relative_hash_simd<F>(arch: F::Arch, args: RelativeHashArgs) -> u16
where
    F: SIMDVector<Scalar = f32> + SIMDPartialOrd + std::ops::Sub<Output = F>,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    // SAFETY: `src` and `dst` each contain `len` values. Construction checks
    // `len <= 16 <= F::LANES`. The masked load does not read inactive lanes.
    let dst = unsafe { F::load_simd_first(arch, args.dst, args.len) };
    // SAFETY: `dst` and `src` have the same checked length.
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

/// Convert a bf16 distance to an order-preserving `u16` key.
///
/// Raw bf16 bits are monotonic only for non-negative values. Inner-product
/// distance can be negative. This transform preserves the total numeric order
/// for both signs.
#[inline(always)]
fn ordered_distance_key(distance: f32) -> u16 {
    let b = f32_to_bf16(distance);
    if b & 0x8000 != 0 { !b } else { b | 0x8000 }
}

/// Update the cached farthest entry for one reservoir.
///
/// # Safety
///
/// The caller holds the source lock. `state.len <= rows.row_stride`. The first
/// `state.len` entries of all three reservoir rows are initialized.
#[inline]
unsafe fn update_farthest(state: &mut ReservoirState, rows: ReservoirRows) {
    if state.len == 0 {
        state.farthest_dist = 0;
        state.farthest_idx = 0;
        return;
    }
    // The total key is `(distance, residual hash, neighbor ID)`. The residual
    // hash resolves equal bf16 distances. The ID resolves the remaining ties.
    let mut max_idx: u8 = 0;
    // SAFETY: `state.len > 0` and all active slots are initialized.
    let mut max_key = unsafe { (*rows.distances, *rows.hashes, *rows.neighbors) };
    for i in 1..state.len as usize {
        // SAFETY: `i < state.len <= rows.row_stride`, and all active entries are
        // initialized.
        let key = unsafe {
            (
                *rows.distances.add(i),
                *rows.hashes.add(i),
                *rows.neighbors.add(i),
            )
        };
        if key > max_key {
            max_key = key;
            max_idx = i as u8;
        }
    }
    state.farthest_dist = max_key.0;
    state.farthest_idx = max_idx;
}

/// Insert one edge into a locked reservoir.
///
/// The function replaces a matching hash only when the new edge has a smaller
/// total key. A full reservoir accepts only a key below its farthest key.
///
/// # Safety
///
/// The caller holds the source lock. Each pointer in `rows` is valid for
/// `row_stride` elements. `state.len <= l_max <= row_stride`. The first
/// `state.len` entries of all three rows are initialized.
#[inline(always)]
unsafe fn insert_reservoir_edge(
    state: &mut ReservoirState,
    rows: ReservoirRows,
    hash: u16,
    neighbor: u32,
    distance: f32,
    l_max: u8,
    find_hash: FindHash,
) -> bool {
    let dist_key = ordered_distance_key(distance);

    if state.len >= l_max {
        let farthest = state.farthest_idx as usize;
        // SAFETY: a full reservoir has `farthest < state.len` initialized slots.
        let farthest_key = unsafe {
            (
                state.farthest_dist,
                *rows.hashes.add(farthest),
                *rows.neighbors.add(farthest),
            )
        };
        if (dist_key, hash, neighbor) >= farthest_key {
            return false;
        }
    }

    if let Some(idx) = find_hash.call(FindHashArgs {
        hashes: rows.hashes,
        row_stride: rows.row_stride,
        len: state.len,
        target: hash,
    }) {
        // SAFETY: `idx < state.len <= rows.row_stride`.
        let current_key = unsafe { (*rows.distances.add(idx), *rows.neighbors.add(idx)) };
        if (dist_key, neighbor) < current_key {
            let was_farthest = idx == state.farthest_idx as usize;
            // SAFETY: `idx < state.len <= rows.row_stride`. The entry is
            // initialized, and the caller holds the source lock.
            unsafe {
                *rows.neighbors.add(idx) = neighbor;
                *rows.distances.add(idx) = dist_key;
            }
            if was_farthest {
                // SAFETY: the caller still holds the source lock. The reservoir rows
                // and initialized prefix are unchanged.
                unsafe { update_farthest(state, rows) };
            }
            return true;
        }
        return false;
    }

    if state.len < l_max {
        let new_idx = state.len as usize;
        let becomes_farthest = if state.len == 0 {
            true
        } else {
            let farthest = state.farthest_idx as usize;
            // SAFETY: `farthest < state.len` identifies an initialized slot.
            let farthest_key = unsafe {
                (
                    state.farthest_dist,
                    *rows.hashes.add(farthest),
                    *rows.neighbors.add(farthest),
                )
            };
            (dist_key, hash, neighbor) > farthest_key
        };
        // SAFETY: `new_idx < l_max <= rows.row_stride`; the caller holds the lock.
        unsafe {
            *rows.hashes.add(new_idx) = hash;
            *rows.distances.add(new_idx) = dist_key;
            *rows.neighbors.add(new_idx) = neighbor;
        }
        state.len += 1;
        if becomes_farthest {
            state.farthest_dist = dist_key;
            state.farthest_idx = new_idx as u8;
        }
        return true;
    }

    // The full-reservoir early rejection above proved that the incoming
    // `(distance, residual hash, ID)` key is better than the cached farthest key.
    let idx = state.farthest_idx as usize;
    // SAFETY: `idx < state.len <= rows.row_stride`; the caller holds the lock.
    unsafe {
        *rows.hashes.add(idx) = hash;
        *rows.distances.add(idx) = dist_key;
        *rows.neighbors.add(idx) = neighbor;
        update_farthest(state, rows);
    }
    true
}

/// Return at most `cap` neighbor IDs in distance order.
///
/// `scratch` belongs to one Rayon extraction job and is reused for its rows.
///
/// # Safety
///
/// The caller must exclude mutation with the source lock or unique ownership.
/// `distances` and `neighbors` each point to `state.len` initialized entries.
unsafe fn collect_nearest_ids(
    state: &ReservoirState,
    distances: *const u16,
    neighbors: *const u32,
    cap: usize,
    scratch: &mut Vec<(u32, u16)>,
) -> Vec<u32> {
    let n = state.len as usize;
    scratch.clear();
    scratch.reserve(n);
    for i in 0..n {
        // SAFETY: `i < n == state.len`, and both arrays have an initialized entry
        // at `i`.
        scratch.push(unsafe { (*neighbors.add(i), *distances.add(i)) });
    }
    scratch.sort_unstable_by_key(|&(id, distance)| (distance, id));
    scratch[..n.min(cap)].iter().map(|&(id, _)| id).collect()
}

/// Return at most `cap` neighbor IDs without sorting them.
///
/// The caller does not depend on reservoir order. This function reads only the
/// neighbor row.
///
/// # Safety
///
/// The caller must exclude mutation with the source lock or unique ownership.
/// `neighbors` points to `state.len` initialized entries.
#[inline]
unsafe fn collect_neighbor_ids(
    state: &ReservoirState,
    neighbors: *const u32,
    cap: usize,
) -> Vec<u32> {
    let out_len = (state.len as usize).min(cap);
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        // SAFETY: `i < out_len <= state.len`, and the neighbor entry is initialized.
        out.push(unsafe { *neighbors.add(i) });
    }
    out
}

/// Bounded point reservoirs shared by parallel leaf workers.
///
/// Source point `i` owns `states[i]` and row `i` in each reservoir array.
/// Its lock protects the metadata and all three rows. A worker holds at most one
/// source lock. Extraction consumes `HashPrune`, so no writer can remain.
pub(crate) struct HashPrune {
    states: Vec<LockedReservoirState>,
    hash_rows: UnsafeCell<MmapSlab<u16>>,
    distance_rows: UnsafeCell<MmapSlab<u16>>,
    neighbor_rows: UnsafeCell<MmapSlab<u32>>,
    row_stride: usize,
    sketches: LshSketches,
    l_max: usize,
    find_hash: FindHash,
    relative_hash: RelativeHash,
}

// SAFETY: each mutable reservoir row is inside `UnsafeCell` and guarded by the
// matching source lock. Different source locks protect disjoint rows. Consuming
// extraction proves that no writer remains.
unsafe impl Send for HashPrune {}
// SAFETY: the same per-point lock protects mutation through shared HashPrune
// references; immutable sketches are safe to share.
unsafe impl Sync for HashPrune {}

impl HashPrune {
    /// Create one empty direction reservoir and LSH sketch for each dataset point.
    pub(crate) fn new<T: VectorRepr + Send + Sync>(
        data: MatrixView<'_, T>,
        num_planes: usize,
        l_max: usize,
        seed: u64,
    ) -> ANNResult<Self> {
        if !(1..=MAX_RESERVOIR_LEN).contains(&l_max) {
            return Err(super::config_error(format!(
                "HashPrune l_max ({l_max}) must be in 1..={MAX_RESERVOIR_LEN}"
            )));
        }

        let npoints = data.nrows();
        let t0 = std::time::Instant::now();
        let sketches = LshSketches::try_new(data, num_planes, seed)?;
        tracing::debug!(
            elapsed_secs = t0.elapsed().as_secs_f64(),
            "sketch computation"
        );
        let t1 = std::time::Instant::now();
        let row_stride = l_max.next_multiple_of(32).max(32);

        let mut states: Vec<LockedReservoirState> = Vec::new();
        states
            .try_reserve_exact(npoints)
            .map_err(ANNError::new)
            .map_err(|error| error.context(format!("reserving {npoints} HashPrune reservoirs")))?;
        for _ in 0..npoints {
            states.push(LockedReservoirState::new());
        }

        // Each reservoir array has one `row_stride` row for each source point.
        let total = npoints.checked_mul(row_stride).ok_or_else(|| {
            super::config_error(format!(
                "HashPrune slab shape {npoints} x {row_stride} overflows usize"
            ))
        })?;
        let hash_rows = MmapSlab::<u16>::new_zeroed(total)?;
        let distance_rows = MmapSlab::<u16>::new_zeroed(total)?;
        let neighbor_rows = MmapSlab::<u32>::new_zeroed(total)?;

        #[cfg(target_os = "linux")]
        {
            let state_bytes = states.len() * std::mem::size_of::<LockedReservoirState>();
            // SAFETY: each pointer names a contiguous allocation of `bytes`.
            // `madvise` does not read or write the allocation.
            unsafe {
                for (ptr, bytes) in [
                    (states.as_ptr() as *mut libc::c_void, state_bytes),
                    (hash_rows.as_ptr() as *mut libc::c_void, hash_rows.bytes()),
                    (
                        distance_rows.as_ptr() as *mut libc::c_void,
                        distance_rows.bytes(),
                    ),
                    (
                        neighbor_rows.as_ptr() as *mut libc::c_void,
                        neighbor_rows.bytes(),
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
            row_stride,
            "reservoir allocation"
        );

        Ok(Self {
            states,
            hash_rows: UnsafeCell::new(hash_rows),
            distance_rows: UnsafeCell::new(distance_rows),
            neighbor_rows: UnsafeCell::new(neighbor_rows),
            row_stride,
            sketches,
            l_max,
            find_hash: arch::dispatch(SelectFindHash),
            relative_hash: arch::dispatch(SelectRelativeHash),
        })
    }

    /// Lock one source point's reservoir while `f` reads or updates it.
    ///
    /// RAII unlocks the point when the closure exits.
    ///
    /// Returns an error when `idx` is outside the reservoir array or its row
    /// offset overflows `usize`.
    #[inline(always)]
    fn with_locked_reservoir<R>(
        &self,
        idx: usize,
        f: impl FnOnce(&mut ReservoirState, ReservoirRows) -> R,
    ) -> ANNResult<R> {
        let slot = self.states.get(idx).ok_or_else(|| {
            ANNError::message(format!(
                "HashPrune point ID {idx} is outside {} reservoirs",
                self.states.len()
            ))
        })?;
        let off = idx.checked_mul(self.row_stride).ok_or_else(|| {
            ANNError::message(format!(
                "HashPrune row offset {idx} x {} overflows usize",
                self.row_stride
            ))
        })?;
        // SAFETY: `idx` is in bounds. Each array has
        // `states.len() * row_stride` elements. `UnsafeCell` permits these writes,
        // and `with_locked_state` holds the source lock for the closure.
        let rows = unsafe {
            ReservoirRows {
                hashes: (*self.hash_rows.get()).as_ptr().cast_mut().add(off),
                distances: (*self.distance_rows.get()).as_ptr().cast_mut().add(off),
                neighbors: (*self.neighbor_rows.get()).as_ptr().cast_mut().add(off),
                row_stride: self.row_stride,
            }
        };
        Ok(slot.with_locked_state(|state| f(state, rows)))
    }

    /// Merge one leaf's CSR edges into the point reservoirs.
    ///
    /// `point_ids` maps leaf-local positions to dataset IDs. `edge_offsets` and
    /// `edges` form a CSR matrix with leaf-local targets. `sketch_scratch` stores
    /// the gathered sketches for this leaf.
    ///
    /// Returns an error for an invalid CSR shape, point ID, or local target.
    pub(crate) fn add_leaf_edges(
        &self,
        point_ids: &[u32],
        edge_offsets: &[u32],
        edges: &[(u32, f32)],
        sketch_scratch: &mut Vec<f32>,
    ) -> ANNResult<()> {
        let n = point_ids.len();
        let expected_offsets = n.checked_add(1).ok_or_else(|| {
            ANNError::message(format!("HashPrune point count {n} overflows usize"))
        })?;
        if edge_offsets.len() != expected_offsets {
            return Err(ANNError::message(format!(
                "HashPrune expected {expected_offsets} edge offsets, got {}",
                edge_offsets.len()
            )));
        }
        if edges.is_empty() {
            return Ok(());
        }

        let m = self.sketches.num_planes();
        let l_max = self.l_max as u8;
        let sketch_len = n.checked_mul(m).ok_or_else(|| {
            ANNError::message(format!("HashPrune sketch shape {n} x {m} overflows usize"))
        })?;
        if sketch_scratch.len() < sketch_len {
            sketch_scratch.resize(sketch_len, 0.0);
        }
        self.gather_sketches(point_ids, &mut sketch_scratch[..sketch_len])?;

        for local_src in 0..n {
            let start = edge_offsets[local_src] as usize;
            let end = edge_offsets[local_src + 1] as usize;
            if start > end || end > edges.len() {
                return Err(ANNError::message(format!(
                    "HashPrune CSR range {start}..{end} is outside {} edges",
                    edges.len()
                )));
            }
            if start == end {
                continue;
            }
            let global_src = point_ids[local_src] as usize;

            if let Some(next) = (local_src + 1..n)
                .find(|&i| edge_offsets[i] != edge_offsets[i + 1])
                .map(|i| point_ids[i] as usize)
            {
                let off = next * self.row_stride;
                prefetch_hint_all(std::slice::from_ref(&self.states[next]));
                // SAFETY: `next` is a dataset point ID, so this raw range is the
                // complete padded hash segment for that point. Raw prefetch avoids
                // creating a shared slice while another worker mutates the segment.
                unsafe {
                    let hashes = (*self.hash_rows.get()).as_ptr().add(off);
                    prefetch_hint_all_raw(
                        hashes.cast(),
                        self.row_stride * std::mem::size_of::<u16>(),
                    );
                }
            }

            let src_sketch = &sketch_scratch[local_src * m..(local_src + 1) * m];
            self.with_locked_reservoir(global_src, |state, rows| -> ANNResult<()> {
                for &(dst_local, dist) in &edges[start..end] {
                    let dst_index = dst_local as usize;
                    let global_dst = *point_ids.get(dst_index).ok_or_else(|| {
                        ANNError::message(format!(
                            "HashPrune local target {dst_local} is outside {n} leaf points"
                        ))
                    })?;
                    let dst_sketch = &sketch_scratch[dst_index * m..(dst_index + 1) * m];
                    let hash = self.relative_hash.call(RelativeHashArgs {
                        src: src_sketch.as_ptr(),
                        dst: dst_sketch.as_ptr(),
                        len: m,
                    });
                    // SAFETY: `with_locked_reservoir` holds this source's lock for the
                    // closure and supplies its three exact reservoir rows.
                    // `l_max` was validated at construction and `insert_reservoir_edge`
                    // maintains initialized entries through `state.len`.
                    unsafe {
                        insert_reservoir_edge(
                            state,
                            rows,
                            hash,
                            global_dst,
                            dist,
                            l_max,
                            self.find_hash,
                        )
                    };
                }
                Ok(())
            })??;
        }
        Ok(())
    }

    fn gather_sketches(&self, indices: &[u32], out: &mut [f32]) -> ANNResult<()> {
        let m = self.sketches.num_planes();
        let expected = indices.len().checked_mul(m).ok_or_else(|| {
            ANNError::message(format!(
                "HashPrune sketch shape {} x {m} overflows usize",
                indices.len()
            ))
        })?;
        if out.len() != expected {
            return Err(ANNError::message(format!(
                "HashPrune expected {expected} gathered sketch values, got {}",
                out.len()
            )));
        }
        let src = self.sketches.sketches();
        for (i, &idx) in indices.iter().enumerate() {
            let start = (idx as usize).checked_mul(m).ok_or_else(|| {
                ANNError::message(format!(
                    "HashPrune sketch offset {idx} x {m} overflows usize"
                ))
            })?;
            let end = start.checked_add(m).ok_or_else(|| {
                ANNError::message(format!("HashPrune sketch row {idx} overflows usize"))
            })?;
            let source = src.get(start..end).ok_or_else(|| {
                ANNError::message(format!("HashPrune point ID {idx} has no sketch row"))
            })?;
            let output_start = i * m;
            out[output_start..output_start + m].copy_from_slice(source);
        }
        Ok(())
    }

    /// Consume the reservoirs and return at most `max_degree` nearest IDs per point.
    #[allow(clippy::disallowed_methods)] // build_graph installs the caller-owned pool.
    pub(crate) fn into_nearest_lists(self, max_degree: usize) -> Vec<AdjacencyList<u32>> {
        let row_stride = self.row_stride;
        drop(self.sketches);
        let HashPrune {
            states,
            hash_rows,
            distance_rows,
            neighbor_rows,
            ..
        } = self;
        let hash_rows = hash_rows.into_inner();
        let distance_rows = distance_rows.into_inner();
        let neighbor_rows = neighbor_rows.into_inner();
        drop(hash_rows);
        (0..states.len())
            .into_par_iter()
            .map_init(Vec::new, |scratch, i| {
                let off = i * row_stride;
                // SAFETY: indexing proves that `i` names a live slot. This method
                // consumes `self`, so no writer can overlap this state reference.
                let state = unsafe { &*states[i].state_ptr() };
                // SAFETY: construction allocated `npoints * row_stride` entries;
                // this loop keeps `i < npoints`, and insertion maintains
                // `state.len <= l_max <= row_stride` initialized entries.
                let ids = unsafe {
                    collect_nearest_ids(
                        state,
                        distance_rows.as_ptr().wrapping_add(off),
                        neighbor_rows.as_ptr().wrapping_add(off),
                        max_degree,
                        scratch,
                    )
                };
                // A neighbor always has the same relative hash for this source;
                // insertion replaces an existing hash slot instead of appending.
                AdjacencyList::from_vec_trusted(ids)
            })
            .collect()
    }

    /// Consume the reservoirs and return all retained IDs without sorting them.
    #[allow(clippy::disallowed_methods)] // build_graph installs the caller-owned pool.
    pub(crate) fn into_candidate_lists(self) -> Vec<AdjacencyList<u32>> {
        let cap = self.l_max;
        let row_stride = self.row_stride;
        drop(self.sketches);
        let HashPrune {
            states,
            hash_rows,
            distance_rows,
            neighbor_rows,
            ..
        } = self;
        let hash_rows = hash_rows.into_inner();
        let distance_rows = distance_rows.into_inner();
        let neighbor_rows = neighbor_rows.into_inner();
        // Extraction reads only neighbor IDs. Drop the hash and distance arrays
        // before the code creates the output lists.
        drop(hash_rows);
        drop(distance_rows);
        (0..states.len())
            .into_par_iter()
            .map(|i| {
                let neighbors = neighbor_rows.as_ptr().wrapping_add(i * row_stride);
                // SAFETY: indexing proves that `i` names a live slot. This method
                // consumes `self`, so no writer can overlap this state reference.
                let state = unsafe { &*states[i].state_ptr() };
                // SAFETY: construction allocated `npoints * row_stride` entries;
                // this loop keeps `i < npoints`, and insertion maintains
                // `state.len <= l_max <= row_stride` initialized entries.
                let ids = unsafe { collect_neighbor_ids(state, neighbors, cap) };
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

    fn hash_prune<T: VectorRepr>(
        data: &[T],
        points: usize,
        dimensions: usize,
        planes: usize,
        l_max: usize,
    ) -> ANNResult<HashPrune> {
        HashPrune::new(
            MatrixView::try_from(data, points, dimensions).unwrap(),
            planes,
            l_max,
            42,
        )
    }

    struct Reservoir {
        state: ReservoirState,
        hashes: Vec<u16>,
        distances: Vec<u16>,
        neighbors: Vec<u32>,
        row_stride: usize,
        l_max: u8,
    }

    impl Reservoir {
        fn new(l_max: usize) -> Self {
            assert!(l_max <= MAX_RESERVOIR_LEN);
            let row_stride = l_max.next_multiple_of(32).max(32);
            Self {
                state: ReservoirState::new_empty(),
                hashes: vec![0; row_stride],
                distances: vec![0; row_stride],
                neighbors: vec![0; row_stride],
                row_stride,
                l_max: l_max as u8,
            }
        }

        fn rows(&self) -> ReservoirRows {
            ReservoirRows {
                hashes: self.hashes.as_ptr() as *mut u16,
                distances: self.distances.as_ptr() as *mut u16,
                neighbors: self.neighbors.as_ptr() as *mut u32,
                row_stride: self.row_stride,
            }
        }

        fn insert(&mut self, hash: u16, neighbor: u32, distance: f32) -> bool {
            let rows = self.rows();
            // SAFETY: the test owns the reservoir and holds its only mutable reference.
            unsafe {
                insert_reservoir_edge(
                    &mut self.state,
                    rows,
                    hash,
                    neighbor,
                    distance,
                    self.l_max,
                    arch::dispatch(SelectFindHash),
                )
            }
        }

        fn neighbors(&self) -> Vec<(u32, f32)> {
            let mut entries: Vec<_> = self
                .neighbors
                .iter()
                .copied()
                .zip(self.distances.iter().copied())
                .take(self.len())
                .collect();
            entries.sort_unstable_by_key(|&(id, distance)| (distance, id));
            entries
                .into_iter()
                .map(|(id, key)| {
                    let bits = if key & 0x8000 != 0 {
                        key & 0x7fff
                    } else {
                        !key
                    };
                    (id, f32::from_bits((bits as u32) << 16))
                })
                .collect()
        }

        fn len(&self) -> usize {
            self.state.len as usize
        }

        fn is_empty(&self) -> bool {
            self.state.len == 0
        }
    }

    #[test]
    fn reservoir_lock_serializes_state_mutation() {
        let slot = LockedReservoirState::new();
        let start = std::sync::Barrier::new(3);

        std::thread::scope(|scope| {
            for _ in 0..2 {
                let slot = &slot;
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    for _ in 0..16 {
                        slot.with_locked_state(|state| state.farthest_dist += 1);
                    }
                });
            }
            start.wait();
        });

        assert_eq!(slot.with_locked_state(|state| state.farthest_dist), 32);
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
        hp.with_locked_reservoir(src, |state, rows| {
            // SAFETY: `with_locked_reservoir` holds the source lock and supplies valid reservoir rows.
            unsafe {
                insert_reservoir_edge(state, rows, hash, dst as u32, distance, l_max, hp.find_hash)
            };
        })
        .unwrap();
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
                        LshSketches::try_new(
                            MatrixView::try_from(converted.as_slice(), points, dimensions).unwrap(),
                            planes,
                            42,
                        )
                        .unwrap(),
                        LshSketches::try_new(
                            MatrixView::try_from(f32_data.as_slice(), points, dimensions).unwrap(),
                            planes,
                            42,
                        )
                        .unwrap(),
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
        let dispatched = arch::dispatch(SelectRelativeHash);

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
            arch::dispatch(SelectRelativeHash).call(RelativeHashArgs {
                src: src.as_ptr(),
                dst: dst.as_ptr(),
                len: dst.len(),
            }),
            0b0011
        );
    }

    #[test]
    fn find_hash_handles_padded_boundaries_and_all_bit_patterns() {
        let dispatched = arch::dispatch(SelectFindHash);

        for target in [0, 0xF00D] {
            for len in [0usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 254, 255] {
                let row_stride = len.max(1).next_multiple_of(32);
                let mut hashes = vec![target; row_stride];
                hashes[..len].fill(0x8001);
                let args = |hashes: &[u16]| FindHashArgs {
                    hashes: hashes.as_ptr(),
                    row_stride,
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
        assert!(!slab.as_ptr().is_null());
        // SAFETY: this test uniquely owns a live four-element slab.
        let values = unsafe { std::slice::from_raw_parts(slab.as_ptr(), 4) };
        assert_eq!(values, &[0; 4]);
    }

    #[test]
    fn accepts_structural_l_max_boundaries() {
        let data = [0.0_f32];
        let low = hash_prune(&data, 1, 1, 1, 1).unwrap();
        assert_eq!(low.l_max, 1);
        assert_eq!(low.row_stride, 32);

        let high = hash_prune(&data, 1, 1, 1, MAX_RESERVOIR_LEN).unwrap();
        assert_eq!(high.l_max, MAX_RESERVOIR_LEN);
        assert_eq!(high.row_stride, 256);
    }

    #[test]
    fn rejects_l_max_outside_structural_boundaries() {
        for l_max in [0, MAX_RESERVOIR_LEN + 1] {
            let result = hash_prune(&[0.0_f32], 1, 1, 1, l_max);
            let error = match result {
                Ok(_) => panic!("l_max={l_max} must be rejected"),
                Err(error) => error,
            };
            assert!(format!("{error:?}").contains(&format!("l_max ({l_max})")));
        }
    }

    #[test]
    fn ordered_distance_key_preserves_bf16_order_for_all_signs() {
        let values = [
            f32::NEG_INFINITY,
            -100.0,
            -0.0,
            0.0,
            0.25,
            100.0,
            f32::INFINITY,
        ];
        let keys: Vec<_> = values.iter().copied().map(ordered_distance_key).collect();
        assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    // Leaf ingestion and scratch reuse.

    #[test]
    fn batched_leaf_edges_match_single_edge_reference() {
        let data = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let batched = hash_prune(&data, 4, 2, 8, 8).unwrap();
        let reference = hash_prune(&data, 4, 2, 8, 8).unwrap();
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

        batched
            .add_leaf_edges(&point_ids, &offsets, &edges, &mut scratch)
            .unwrap();
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
    fn leaf_edges_reject_invalid_csr_and_point_ids() {
        let data = [0.0_f32, 1.0];
        let hp = hash_prune(&data, 2, 1, 1, 2).unwrap();
        let mut scratch = Vec::new();

        assert!(hp.add_leaf_edges(&[0], &[0], &[], &mut scratch).is_err());
        assert!(
            hp.add_leaf_edges(&[2], &[0, 1], &[(0, 1.0)], &mut scratch)
                .is_err()
        );
        assert!(
            hp.add_leaf_edges(&[0, 1], &[0, 1, 1], &[(2, 1.0)], &mut scratch)
                .is_err()
        );
        assert!(
            hp.add_leaf_edges(&[0, 1], &[0, 1, 0], &[(1, 1.0)], &mut scratch)
                .is_err()
        );
    }

    #[test]
    fn leaf_edges_grow_then_reuse_sketch_scratch() {
        let data = [0.0_f32, 1.0, 2.0, 3.0];
        let hp = hash_prune(&data, 4, 1, 8, 4).unwrap();
        let mut scratch = vec![99.0; 1];

        hp.add_leaf_edges(&[0, 1], &[0, 1, 2], &[(1, 1.0), (0, 1.0)], &mut scratch)
            .unwrap();
        assert_eq!(scratch.len(), 16);
        let capacity = scratch.capacity();

        hp.add_leaf_edges(&[2, 3], &[0, 1, 2], &[(1, 1.0), (0, 1.0)], &mut scratch)
            .unwrap();
        assert_eq!(scratch.len(), 16);
        assert_eq!(scratch.capacity(), capacity);

        hp.add_leaf_edges(&[0, 1], &[0, 0, 0], &[], &mut scratch)
            .unwrap();
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
        let parallel = hash_prune(&data, 100, 4, 4, 10).unwrap();
        let serial = hash_prune(&data, 100, 4, 4, 10).unwrap();

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
        let full = hash_prune(&data, 8, 2, 16, 10).unwrap();
        let nearest = hash_prune(&data, 8, 2, 16, 10).unwrap();
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
