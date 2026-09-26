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

cfg_if::cfg_if! {
    // x86-64 guarantees SSE2; `_mm_prefetch` needs only SSE.
    if #[cfg(target_arch = "x86_64")] {
        const CACHE_LINE_SIZE: usize = 64;

        #[inline(always)]
        unsafe fn prefetch_exactly<const N: usize>(ptr: *const i8) {
            use std::arch::x86_64::*;
            for i in 0..N {
                // SAFETY: the caller guarantees that all `N` computed addresses are
                // inside the allocation.
                unsafe { _mm_prefetch(ptr.add(i * CACHE_LINE_SIZE), _MM_HINT_T0) };
            }
        }

        #[inline(always)]
        unsafe fn prefetch_at_most<const N: usize>(ptr: *const i8, bytes: usize) {
            use std::arch::x86_64::*;
            for i in 0..N {
                if CACHE_LINE_SIZE * i >= bytes {
                    break;
                }
                // SAFETY: the loop uses only offsets below `bytes`.
                unsafe { _mm_prefetch(ptr.add(i * CACHE_LINE_SIZE), _MM_HINT_T0) };
            }
        }

        /// Prefetch the given vector in chunks of 64 bytes, which is a cache line size.
        /// Only the first `MAX_BLOCKS` chunks will be prefetched.
        #[inline]
        pub fn prefetch_hint_max<const MAX_CACHE_LINES: usize, T>(vec: &[T]) {
            let vecsize = std::mem::size_of_val(vec);
            if vecsize >= MAX_CACHE_LINES * 64 {
                // SAFETY: the slice contains every address passed to prefetch.
                unsafe { prefetch_exactly::<MAX_CACHE_LINES>(vec.as_ptr().cast()) }
            } else {
                // SAFETY: the slice covers `vecsize` bytes.
                unsafe { prefetch_at_most::<MAX_CACHE_LINES>(vec.as_ptr().cast(), vecsize) }
            }
        }

        /// Prefetch a raw byte range without creating a slice.
        ///
        /// # Safety
        ///
        /// `ptr` must identify an allocation of at least `bytes` bytes. The allocation
        /// must remain live for this call. The function creates no Rust reference.
        /// The caller controls concurrent mutation of the range.
        #[inline]
        pub unsafe fn prefetch_hint_all_raw(ptr: *const u8, bytes: usize) {
            use std::arch::x86_64::*;

            for offset in (0..bytes).step_by(CACHE_LINE_SIZE) {
                // SAFETY: the caller guarantees the byte range, and `offset < bytes`.
                unsafe { _mm_prefetch(ptr.add(offset).cast(), _MM_HINT_T0) };
            }
        }

        /// Prefetch the given vector in chunks of 64 bytes, which is a cache line size.
        /// The entire vector will be prefetched.
        #[inline]
        pub fn prefetch_hint_all<T>(vec: &[T]) {
            // SAFETY: the slice remains live and covers exactly `size_of_val(vec)` bytes.
            unsafe { prefetch_hint_all_raw(vec.as_ptr().cast(), std::mem::size_of_val(vec)) }
        }
    } else {
        pub fn prefetch_hint_max<const MAX_CACHE_LINES: usize, T>(_vec: &[T]) {}

        /// Accept a raw prefetch range and do nothing.
        ///
        /// # Safety
        ///
        /// The pointer contract is the same as the x86-64 implementation.
        pub unsafe fn prefetch_hint_all_raw(_ptr: *const u8, _bytes: usize) {}

        pub fn prefetch_hint_all<T>(_vec: &[T]) {}
    }
}

#[cfg(test)]
mod test_util;
