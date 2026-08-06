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
                // SAFETY: the caller guarantees `N` cache-line starts are in bounds.
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
                // SAFETY: the loop only visits cache-line starts below `bytes`.
                unsafe { _mm_prefetch(ptr.add(i * CACHE_LINE_SIZE), _MM_HINT_T0) };
            }
        }

        /// Prefetch the given vector in chunks of 64 bytes, which is a cache line size.
        /// Only the first `MAX_BLOCKS` chunks will be prefetched.
        #[inline]
        pub fn prefetch_hint_max<const MAX_CACHE_LINES: usize, T>(vec: &[T]) {
            let vecsize = std::mem::size_of_val(vec);
            if vecsize >= MAX_CACHE_LINES * 64 {
                // SAFETY: the slice covers every prefetched cache-line start.
                unsafe { prefetch_exactly::<MAX_CACHE_LINES>(vec.as_ptr().cast()) }
            } else {
                // SAFETY: the slice covers `vecsize` bytes.
                unsafe { prefetch_at_most::<MAX_CACHE_LINES>(vec.as_ptr().cast(), vecsize) }
            }
        }

        /// Prefetch cache-line starts in a raw byte range without creating a slice.
        ///
        /// # Safety
        ///
        /// `ptr` must point to an allocation that remains live for `bytes` bytes during
        /// this call. The function creates no Rust reference, so the range may be
        /// concurrently mutated under the caller's synchronization policy.
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

        /// No-op raw-range prefetch on architectures without a prefetch implementation.
        ///
        /// # Safety
        ///
        /// The pointer contract matches the x86-64 implementation.
        pub unsafe fn prefetch_hint_all_raw(_ptr: *const u8, _bytes: usize) {}

        pub fn prefetch_hint_all<T>(_vec: &[T]) {}
    }
}

#[cfg(test)]
mod test_util;
