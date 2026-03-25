// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Cache-aware block-transposed SIMD implementation for multi-vector distance computation.
//!
//! This module provides a SIMD-accelerated implementation that uses block-transposed
//! memory layout for **query** vectors (instead of documents), with documents remaining
//! in row-major format.
//!
//! # Module Organization
//!
//! - [`CacheAwareKernel`] — unsafe trait that each element type implements (this file).
//! - `kernel` — generic 5-level cache-aware tiling loop (`tiled_reduce`).
//! - `f32_kernel` — f32 SIMD micro-kernel, entry point ([`cache_aware_chamfer`]),
//!   query wrapper ([`QueryBlockTransposedRef`]), and `MaxSim` / `Chamfer` trait impls.
//!
//! # Cache-Aware Tiling Strategy
//!
//! This approach uses a reducing-GEMM pattern modeled after high-performance BLAS
//! implementations:
//!
//! - **L2 cache**: Tiles of the transposed query ("A") are sized to fit in L2.
//! - **L1 cache**: Tiles of the document ("B") plus one micro-panel of A are sized
//!   to fit in L1.
//! - **Micro-kernel**: An `A_PANEL × B_PANEL` micro-kernel (e.g. 16×4 for f32)
//!   processes a panel of query vectors against a panel of document vectors per
//!   invocation, accumulating max-IP into a scratch buffer. The panel sizes are
//!   determined by the [`CacheAwareKernel`] implementation for each element type.
//!
//! # Memory Layout
//!
//! - **Query**: Block-transposed (`GROUP` vectors per block, dimensions contiguous
//!   within each block). The block size is determined by the kernel's `A_PANEL`.
//! - **Document**: Row-major (standard [`MatRef`](crate::multi_vector::MatRef) format).

mod f32_kernel;
mod kernel;

pub use f32_kernel::{QueryBlockTransposedRef, cache_aware_chamfer};

// ── Cache budget constants ───────────────────────────────────────

/// Approximate usable L1 data cache in bytes (conservative estimate).
const L1_CACHE: usize = 48_000;

/// Approximate usable L2 cache in bytes (conservative estimate).
const L2_CACHE: usize = 1_250_000;

/// Fraction of L2 reserved for the A tile. The remainder accommodates B streaming
/// traffic and incidental cache pollution.
const L2_A_TILE_BUDGET: usize = L2_CACHE / 2;

/// Fraction of L1 available for the B tile. The A micro-panel is subtracted at
/// runtime since it depends on K; this is the total L1 budget before that subtraction.
const L1_B_TILE_BUDGET: usize = L1_CACHE * 3 / 4;

// ── CacheAwareKernel trait ───────────────────────────────────────

/// Trait abstracting a SIMD micro-kernel for the cache-aware tiling loop.
///
/// Each implementation provides the element types, panel geometry, and the actual
/// SIMD micro-kernel body. The generic [`tiled_reduce`](kernel::tiled_reduce) function
/// handles the 5-level cache-aware loop nest and calls into the kernel via this trait.
///
/// # Safety
///
/// Implementors must ensure that `full_panel` and `remainder_dispatch` only
/// read/write within the bounds described by their pointer arguments and the
/// `k` / panel-size contracts.
pub(crate) unsafe trait CacheAwareKernel {
    /// Element type stored in the block-transposed query ("A" side).
    type QueryElem: Copy;
    /// Element type stored in the row-major document ("B" side).
    type DocElem: Copy;

    /// Number of query rows processed per micro-kernel invocation.
    /// Determined by SIMD register width for the element type.
    const A_PANEL: usize;
    /// Number of document rows processed per micro-kernel invocation
    /// (broadcast unroll factor).
    const B_PANEL: usize;

    /// Process one full `A_PANEL × B_PANEL` micro-panel pair.
    ///
    /// # Safety
    ///
    /// * `a` must point to `A_PANEL * k` contiguous `QueryElem` values.
    /// * `b` must point to `B_PANEL` rows of `k` contiguous `DocElem` values.
    /// * `r` must point to at least `A_PANEL` writable `f32` values.
    unsafe fn full_panel(
        arch: diskann_wide::arch::Current,
        a: *const Self::QueryElem,
        b: *const Self::DocElem,
        k: usize,
        r: *mut f32,
    );

    /// Dispatch for `1..(B_PANEL-1)` remainder document rows.
    ///
    /// # Safety
    ///
    /// Same pointer contracts as `full_panel`, but `b` points to `remainder`
    /// rows instead of `B_PANEL` rows.
    unsafe fn remainder_dispatch(
        arch: diskann_wide::arch::Current,
        remainder: usize,
        a: *const Self::QueryElem,
        b: *const Self::DocElem,
        k: usize,
        r: *mut f32,
    );
}
