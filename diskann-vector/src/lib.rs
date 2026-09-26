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
    if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
        const CACHE_LINE_SIZE: usize = 64;

        #[inline(always)]
        unsafe fn prefetch_exactly<const N: usize>(ptr: *const i8) {
            use std::arch::x86_64::*;
            for i in 0..N {
                _mm_prefetch(ptr.add(i * CACHE_LINE_SIZE), _MM_HINT_T0);
            }
        }

        #[inline(always)]
        unsafe fn prefetch_at_most<const N: usize>(ptr: *const i8, bytes: usize) {
            use std::arch::x86_64::*;
            for i in 0..N {
                if CACHE_LINE_SIZE * i >= bytes {
                    break;
                }
                _mm_prefetch(ptr.add(i * CACHE_LINE_SIZE), _MM_HINT_T0);
            }
        }

        /// Prefetch the given vector in chunks of 64 bytes, which is a cache line size.
        /// Only the first `MAX_BLOCKS` chunks will be prefetched.
        #[inline]
        pub fn prefetch_hint_max<const MAX_CACHE_LINES: usize, T>(vec: &[T]) {
            let vecsize = std::mem::size_of_val(vec);
            if vecsize >= MAX_CACHE_LINES * 64 {
                // SAFETY: Pointer is in-bounds and use of the intrinsic is cfg gated.
                unsafe { prefetch_exactly::<MAX_CACHE_LINES>(vec.as_ptr().cast()) }
            } else {
                // SAFETY: Pointer is in-bounds and use of the intrinsic is cfg gated.
                unsafe { prefetch_at_most::<MAX_CACHE_LINES>(vec.as_ptr().cast(), vecsize) }
            }
        }

        /// Prefetch the given vector in chunks of 64 bytes, which is a cache line size.
        /// The entire vector will be prefetched.
        #[inline]
        pub fn prefetch_hint_all<T>(vec: &[T]) {
            use std::arch::x86_64::*;

            let vecsize = std::mem::size_of_val(vec);
            let num_prefetch_blocks = vecsize.div_ceil(64);
            let vec_ptr = vec.as_ptr() as *const i8;
            for d in 0..num_prefetch_blocks {
                // SAFETY: Pointer is in-bounds and use of the intrinsic is gated by the
                // `cfg`-guard on this function.
                unsafe {
                    std::arch::x86_64::_mm_prefetch(vec_ptr.add(d * CACHE_LINE_SIZE), _MM_HINT_T0);
                }
            }        }
    } else if #[cfg(target_arch = "aarch64")] {
        use std::arch::asm;

        const CACHE_LINE_SIZE: usize = 64;

        #[inline(always)]
        unsafe fn prefetch_l1(ptr: *const i8) {
            // SAFETY: The `prfm` hint does not dereference the pointer in Rust memory model
            // terms, and this function is only compiled on aarch64.
            unsafe {
                asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) ptr,
                    options(nostack, preserves_flags),
                );
            }
        }

        #[inline(always)]
        unsafe fn prefetch_exactly<const N: usize>(ptr: *const i8) {
            for i in 0..N {
                // SAFETY: Pointer arithmetic stays within the caller-provided prefetch range and
                // `prefetch_l1` is aarch64-only.
                unsafe {
                    prefetch_l1(ptr.add(i * CACHE_LINE_SIZE));
                }
            }
        }

        #[inline(always)]
        unsafe fn prefetch_at_most<const N: usize>(
            ptr: *const i8,
            bytes: usize,
        ) {
            for i in 0..N {
                if CACHE_LINE_SIZE * i >= bytes {
                    break;
                }

                // SAFETY: `i` is bounded by `bytes`, so the computed address remains within the
                // caller-provided prefetch range and `prefetch_l1` is aarch64-only.
                unsafe {
                    prefetch_l1(ptr.add(i * CACHE_LINE_SIZE));
                }
            }
        }

        #[inline]
        pub fn prefetch_hint_max<const MAX_CACHE_LINES: usize, T>(vec: &[T]) {
            let vecsize = std::mem::size_of_val(vec);

            if vecsize >= MAX_CACHE_LINES * CACHE_LINE_SIZE {
                // SAFETY: Pointer originates from `vec` and prefetch count is bounded.
                unsafe {
                    prefetch_exactly::<MAX_CACHE_LINES>(vec.as_ptr().cast())
                }
            } else {
                // SAFETY: Pointer originates from `vec`; bytes bound ensures addresses are valid.
                unsafe {
                    prefetch_at_most::<MAX_CACHE_LINES>(
                        vec.as_ptr().cast(),
                        vecsize,
                    )
                }
            }
        }

        #[inline]
        pub fn prefetch_hint_all<T>(vec: &[T]) {
            let vecsize = std::mem::size_of_val(vec);
            let num_prefetch_blocks = vecsize.div_ceil(CACHE_LINE_SIZE);
            let vec_ptr = vec.as_ptr() as *const i8;

            for d in 0..num_prefetch_blocks {
                // SAFETY: Loop bounds are derived from `vec` length and this is a prefetch hint.
                unsafe {
                    prefetch_l1(vec_ptr.add(d * CACHE_LINE_SIZE));
                }
            }
        }

    } else {
        pub fn prefetch_hint_max<const MAX_CACHE_LINES: usize, T>(_vec: &[T]) {}
        pub fn prefetch_hint_all<T>(_vec: &[T]) {}
    }
}

#[cfg(test)]
mod test_util;
