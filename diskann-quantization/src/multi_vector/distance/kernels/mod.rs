// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Block-transposed SIMD kernels for multi-vector distance computation.
//!
//! This module provides a SIMD-accelerated implementation that uses block-transposed
//! memory layout for **query** vectors (instead of documents), with documents remaining
//! in row-major format.
//!
//! # Module Organization
//!
//! - `Kernel<A>` — unsafe trait parameterized on architecture (this file).
//! - `tiled_reduce` — generic 5-level tiling loop (`tiled_reduce`).
//! - `f32` — f32 micro-kernel family: V3/V4 (x86_64), Scalar/Neon (all platforms).
//!   Entry point (`chamfer_kernel`), `MaxSim` / `Chamfer` trait impls.
//!
//! # Tiling Strategy
//!
//! This approach uses a reducing-GEMM pattern modeled after high-performance BLAS
//! implementations:
//!
//! - **L2 cache**: Tiles of A (the transposed query) are sized to fit in L2.
//! - **L1 cache**: Tiles of B (the document) plus one micro-panel of A are sized
//!   to fit in L1.
//! - **Micro-kernel**: An `A_PANEL × B_PANEL` micro-kernel (e.g. 16×4 for f32 on V3)
//!   processes a panel of A rows against a panel of B rows per invocation,
//!   accumulating max-IP into a scratch buffer. The panel sizes are determined
//!   by the `Kernel<A>` implementation for each element type.
//!
//! # Memory Layout
//!
//! - **Query**: Block-transposed (`GROUP` vectors per block, dimensions contiguous
//!   within each block). The block size is determined by the kernel's `A_PANEL`.
//! - **Document**: Row-major (standard [`MatRef`](crate::multi_vector::MatRef) format).

mod f32;
mod tiled_reduce;

// ── Cache size detection stubs ───────────────────────────────────

/// Detect the L1 data cache size in bytes.
///
/// Returns `None` until platform detection is wired in (e.g. via
/// `diskann_platform::get_l1d_cache_size()`).
fn detect_l1d_cache() -> Option<usize> {
    None
}

/// Detect the L2 cache size in bytes.
///
/// Returns `None` until platform detection is wired in (e.g. via
/// `diskann_platform::get_l2_cache_size()`).
fn detect_l2_cache() -> Option<usize> {
    None
}

// ── Cache budget helpers ─────────────────────────────────────────

/// Approximate usable L1 data cache in bytes.
fn l1_cache() -> usize {
    detect_l1d_cache().unwrap_or(48_000)
}

/// Approximate usable L2 cache in bytes.
fn l2_cache() -> usize {
    detect_l2_cache().unwrap_or(1_250_000)
}

/// Fraction of L2 reserved for the A tile. The remainder accommodates B streaming
/// traffic and incidental cache pollution.
fn l2_a_tile_budget() -> usize {
    l2_cache() / 2
}

/// Fraction of L1 available for the B tile. The A micro-panel is subtracted at
/// runtime since it depends on K; this is the total L1 budget before that subtraction.
fn l1_b_tile_budget() -> usize {
    l1_cache() * 3 / 4
}

// ── Kernel trait ─────────────────────────────────────────────────

/// Trait abstracting a SIMD micro-kernel for the tiling loop.
///
/// The architecture parameter `A` enables different implementations for each
/// micro-architecture (e.g. V3 for AVX2+FMA, V4 for AVX-512, Neon for ARM).
///
/// Each implementation provides the element types, panel geometry, and the actual
/// SIMD micro-kernel body. The generic `tiled_reduce` function
/// handles the 5-level loop nest and calls into the kernel via this trait.
///
/// # Safety
///
/// Implementors must ensure that `full_panel` and `remainder_dispatch` only
/// read/write within the bounds described by their pointer arguments and the
/// `k` / panel-size contracts.
pub(crate) unsafe trait Kernel<A: diskann_wide::Architecture> {
    /// Element type for the A side (e.g. query vectors).
    type AElem: Copy;
    /// Element type for the B side (e.g. document vectors).
    type BElem: Copy;

    /// Number of A rows processed per micro-kernel invocation.
    const A_PANEL: usize;
    /// Number of B rows processed per micro-kernel invocation.
    const B_PANEL: usize;

    /// Process one full `A_PANEL × B_PANEL` micro-panel pair.
    ///
    /// # Safety
    ///
    /// * `a` must point to `A_PANEL * k` contiguous `AElem` values.
    /// * `b` must point to `B_PANEL * k` contiguous `BElem` values.
    /// * `r` must point to at least `A_PANEL` writable `f32` values.
    unsafe fn full_panel(
        arch: A,
        a: *const Self::AElem,
        b: *const Self::BElem,
        k: usize,
        r: *mut f32,
    );

    /// Dispatch for `1..(B_PANEL-1)` remainder B rows.
    ///
    /// # Safety
    ///
    /// Same pointer contracts as `full_panel`, but `b` points to `remainder`
    /// rows instead of `B_PANEL` rows.
    unsafe fn remainder_dispatch(
        arch: A,
        remainder: usize,
        a: *const Self::AElem,
        b: *const Self::BElem,
        k: usize,
        r: *mut f32,
    );
}
