/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
//! # vector
//!
//! This crate contains SIMD accelerated functions for operating on vector data. Note that the name 'vector'
//! does not exclusively mean embedding vectors, but any array of data appropriate for SIMD. Therefor, aside
//! from fast implementations of distance for real vectors, this crate also includes things like SIMD
//! accelerated contains for slices.
#![cfg_attr(
    not(test),
    warn(
        clippy::panic,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::undocumented_unsafe_blocks
    )
)]

mod half;
pub use half::Half;

mod traits;
pub use traits::{
    DistanceFunction, DistanceFunctionMut, Norm, PreprocessedDistanceFunction, PureDistanceFunction,
};

mod value;
pub use value::{MathematicalValue, SimilarityScore};

mod unaligned;
pub use unaligned::{AsUnaligned, UnalignedSlice};

pub mod contains;
pub mod conversion;
pub mod distance;
pub mod norm;
pub mod sparse;

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
))]
const CACHE_LINE_SIZE: usize = 64;

/// Prefetch the cache line containing `ptr`.
///
/// # Safety
///
/// `ptr` must point to a byte within a live allocation.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn prefetch_cache_line(_ptr: *const i8) {
    #[cfg(not(miri))]
    {
        use std::arch::x86_64::*;
        // SAFETY: Inherited from caller and intrinsic is cfg-gated to supported targets.
        unsafe { _mm_prefetch(_ptr, _MM_HINT_T0) };
    }
}

/// Prefetch the cache line containing `ptr`.
///
/// # Safety
///
/// `ptr` must point to a byte within a live allocation.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn prefetch_cache_line(_ptr: *const i8) {
    #[cfg(not(miri))]
    {
        // SAFETY: The `prfm` hint does not dereference the pointer in Rust memory model terms.
        unsafe {
            std::arch::asm!(
                "prfm pldl2keep, [{ptr}]",
                ptr = in(reg) _ptr,
                options(nostack, preserves_flags, readonly),
            );
        }
    }
}

/// Prefetch the cache line containing `ptr`.
///
/// # Safety
///
/// `ptr` must point to a byte within a live allocation.
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
)))]
#[inline(always)]
pub unsafe fn prefetch_cache_line(_ptr: *const i8) {}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
))]
#[inline(always)]
unsafe fn prefetch_exactly<const N: usize>(ptr: *const i8) {
    for i in 0..N {
        // SAFETY: Pointer arithmetic stays within the caller-provided prefetch range.
        unsafe { prefetch_cache_line(ptr.add(i * CACHE_LINE_SIZE)) };
    }
}

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
))]
#[inline(always)]
unsafe fn prefetch_at_most<const N: usize>(ptr: *const i8, bytes: usize) {
    for i in 0..N {
        if CACHE_LINE_SIZE * i >= bytes {
            break;
        }

        // SAFETY: `i` is bounded by `bytes`, so computed addresses remain in range.
        unsafe { prefetch_cache_line(ptr.add(i * CACHE_LINE_SIZE)) };
    }
}

/// Prefetch the given vector in chunks of 64 bytes, which is a cache line size.
/// Only the first `MAX_CACHE_LINES` chunks will be prefetched.
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
))]
#[inline]
pub fn prefetch_hint_max<const MAX_CACHE_LINES: usize, T>(vec: &[T]) {
    let vecsize = std::mem::size_of_val(vec);
    if vecsize >= MAX_CACHE_LINES * CACHE_LINE_SIZE {
        // SAFETY: Pointer originates from `vec` and prefetch count is bounded.
        unsafe { prefetch_exactly::<MAX_CACHE_LINES>(vec.as_ptr().cast()) }
    } else {
        // SAFETY: Pointer originates from `vec`; `bytes` bound caps traversal.
        unsafe { prefetch_at_most::<MAX_CACHE_LINES>(vec.as_ptr().cast(), vecsize) }
    }
}

/// Prefetch the given vector in chunks of 64 bytes, which is a cache line size.
/// The entire vector will be prefetched.
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
))]
#[inline]
pub fn prefetch_hint_all<T>(vec: &[T]) {
    let vecsize = std::mem::size_of_val(vec);
    let num_prefetch_blocks = vecsize.div_ceil(CACHE_LINE_SIZE);
    let vec_ptr = vec.as_ptr() as *const i8;
    for d in 0..num_prefetch_blocks {
        // SAFETY: Loop bounds are derived from `vec` size and this is a prefetch hint.
        unsafe { prefetch_cache_line(vec_ptr.add(d * CACHE_LINE_SIZE)) };
    }
}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
)))]
pub fn prefetch_hint_max<const MAX_CACHE_LINES: usize, T>(_vec: &[T]) {}

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
)))]
pub fn prefetch_hint_all<T>(_vec: &[T]) {}

#[cfg(test)]
mod test_util;
