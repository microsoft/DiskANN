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
//! - [`Kernel<A>`] — unsafe trait parameterized on architecture (this file).
//! - [`layouts`] — layout markers, [`ConvertTo`](layouts::ConvertTo) conversion
//!   trait, and `DescribeLayout` bridge.
//! - [`tiled_reduce`](tiled_reduce) — generic 5-level tiling loop.
//! - [`f32`] — f32 micro-kernel family: V3 (x86_64), Scalar (portable). V4 delegates to V3; Neon delegates to Scalar.
//!   Entry point ([`max_ip_kernel`](f32::max_ip_kernel)).
//! - [`f16`] — f16 entry point reusing the f32 micro-kernel family with
//!   tile-level f16→f32 conversion via [`ConvertTo`](layouts::ConvertTo).
//!   Entry point ([`max_ip_kernel`](f16::max_ip_kernel)).
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

pub(super) mod f16;
pub(super) mod f32;
mod layouts;
mod tiled_reduce;

/// Detect the L1 data cache size in bytes.
///
/// Returns `None` until platform detection is wired in (e.g. via
/// `diskann_platform::get_l1d_cache_size()`).
///
// TODO: Wire to `diskann_platform` or env-var override.
fn detect_l1d_cache() -> Option<usize> {
    None
}

/// Detect the L2 cache size in bytes.
///
/// Returns `None` until platform detection is wired in (e.g. via
/// `diskann_platform::get_l2_cache_size()`).
///
// TODO: Wire to `diskann_platform` or env-var override.
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

/// SIMD micro-kernel trait for the [`tiled_reduce`](tiled_reduce::tiled_reduce) loop.
///
/// Each implementation provides the micro-kernel body and the layout types
/// that describe the element format consumed by `full_panel` /
/// `partial_panel`. Tile-level conversion between storage layouts and
/// kernel layouts is handled externally by [`ConvertTo`](layouts::ConvertTo)
/// implementations; the kernel itself only sees already-converted data.
///
/// # Safety
///
/// Implementors must ensure that:
/// - `full_panel` and `partial_panel` access only within the bounds
///   described by their pointer arguments and the `k`/panel-size contracts.
unsafe trait Kernel<A: diskann_wide::Architecture> {
    /// Layout consumed by the A (left / query) side of the micro-kernel.
    type Left: layouts::Layout;
    /// Layout consumed by the B (right / document) side of the micro-kernel.
    type Right: layouts::Layout;

    /// Number of A rows processed per micro-kernel invocation.
    const A_PANEL: usize;
    /// Number of B rows processed per micro-kernel invocation.
    const B_PANEL: usize;

    /// Process one full `A_PANEL × B_PANEL` micro-panel pair.
    ///
    /// # Safety
    ///
    /// * `a` must point to `A_PANEL * k` contiguous elements of
    ///   `<Self::Left as Layout>::Element`.
    /// * `b` must point to `B_PANEL * k` contiguous elements of
    ///   `<Self::Right as Layout>::Element`.
    /// * `r` must point to at least `A_PANEL` writable `f32` values.
    unsafe fn full_panel(
        arch: A,
        a: *const <Self::Left as layouts::Layout>::Element,
        b: *const <Self::Right as layouts::Layout>::Element,
        k: usize,
        r: *mut f32,
    );

    /// Dispatch for `1..(B_PANEL-1)` remainder B rows.
    ///
    /// # Safety
    ///
    /// * `a` must point to `A_PANEL * k` contiguous elements of
    ///   `<Self::Left as Layout>::Element`.
    /// * `b` must point to `remainder * k` contiguous elements of
    ///   `<Self::Right as Layout>::Element`.
    /// * `r` must point to at least `A_PANEL` writable `f32` values.
    unsafe fn partial_panel(
        arch: A,
        remainder: usize,
        a: *const <Self::Left as layouts::Layout>::Element,
        b: *const <Self::Right as layouts::Layout>::Element,
        k: usize,
        r: *mut f32,
    );
}
