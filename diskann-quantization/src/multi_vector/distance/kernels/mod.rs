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
//! - [`tiled_reduce`](tiled_reduce) — generic 5-level tiling loop.
//! - [`f32`] — f32 micro-kernel family: V3/V4 (x86_64), Scalar/Neon.
//!   Entry point ([`chamfer_kernel`](f32::chamfer_kernel)),
//!   `MaxSim` / `Chamfer` trait impls.
//! - [`f16`] — f16 micro-kernel family: same architecture matrix.
//!   Lazily unpacks f16→f32 in `prepare_a`/`prepare_b`, then delegates
//!   to the f32 micro-kernel. Entry point
//!   ([`chamfer_kernel_f16`](f16::chamfer_kernel_f16)),
//!   `MaxSim` / `Chamfer` trait impls.
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

mod f16;
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
/// Each implementation provides the element types, panel geometry, preparation
/// hooks, and the actual SIMD micro-kernel body. The generic `tiled_reduce`
/// function handles the 5-level loop nest and calls into the kernel via this
/// trait.
///
/// # Panel preparation
///
/// The `prepare_a` and `prepare_b` hooks allow kernels to convert or repack
/// panel data before the micro-kernel sees it. `tiled_reduce` calls
/// `prepare_a` once per A micro-panel (Loop 3) and `prepare_b` once per B
/// micro-panel (Loop 4). Staging buffers for those conversions are owned by
/// the kernel itself, allocated once in [`new`](Self::new) and reused across
/// panels.
///
/// For identity cases (e.g. `AElem == APrepared`), the implementation may
/// return the `src` pointer directly without copying, achieving zero
/// overhead. Such kernels remain zero-sized types with a no-op `new`.
///
/// # Partial panels and zero-padding
///
/// Both `prepare_a` and `prepare_b` accept a `rows` parameter indicating the
/// actual number of rows in the panel. For the last panel on either dimension,
/// `rows` may be less than `A_PANEL` / `B_PANEL`. Implementations that
/// convert into an internal buffer **must** zero-pad the remaining
/// `(PANEL - rows) * k` elements so the micro-kernel can process a full-sized
/// panel without reading garbage. The scratch buffer entries corresponding to
/// the padded rows are written but unused by the caller.
///
/// # Aliasing contract
///
/// `prepare_a` returns a `*const APrepared` that may point into `self`.
/// The subsequent call to `prepare_b(&mut self, ...)` re-borrows `self`
/// mutably, but implementations **must not** write to the A staging buffer
/// during `prepare_b`. This ensures the pointer returned by `prepare_a`
/// remains valid through `full_panel` / `remainder_dispatch`.
///
/// # Safety
///
/// Implementors must ensure that:
/// - `prepare_a` and `prepare_b` only read from `src` within the bounds
///   `rows * k`, and if they write to an internal buffer, the output is
///   valid for `{A,B}_PANEL * k` reads of `{A,B}Prepared`.
/// - If `prepare_a` / `prepare_b` returns `src` directly, that pointer must
///   be valid for `{A,B}_PANEL * k` reads of `{A,B}Prepared`.
/// - `prepare_b` must not write to the A staging buffer.
/// - `full_panel` and `remainder_dispatch` only read/write within the bounds
///   described by their pointer arguments and the `k` / panel-size contracts.
pub(crate) unsafe trait Kernel<A: diskann_wide::Architecture> {
    /// Element type for the A side (e.g. query vectors in storage format).
    type AElem: Copy;
    /// Element type for the B side (e.g. document vectors in storage format).
    type BElem: Copy;
    /// Element type for the A side after panel preparation.
    type APrepared: Copy;
    /// Element type for the B side after panel preparation.
    type BPrepared: Copy;

    /// Number of A rows processed per micro-kernel invocation.
    const A_PANEL: usize;
    /// Number of B rows processed per micro-kernel invocation.
    const B_PANEL: usize;

    /// Construct a new kernel instance, allocating any staging buffers needed
    /// for the contraction dimension `k`.
    ///
    /// Identity kernels (where `AElem == APrepared` and `BElem == BPrepared`)
    /// should remain zero-sized and ignore `k`.
    fn new(k: usize) -> Self;

    /// Prepare one A micro-panel for the micro-kernel.
    ///
    /// Called once per A micro-panel in Loop 3, before the B-panel iteration.
    /// The result is reused across all B-panels in the tile, amortizing
    /// conversion cost.
    ///
    /// For full panels `rows == A_PANEL`. For the last panel when the A
    /// dimension is not a multiple of `A_PANEL`, `rows < A_PANEL`.
    /// Implementations that write to an internal buffer must zero-pad the
    /// remaining `(A_PANEL - rows) * k` elements.
    ///
    /// # Safety
    ///
    /// * `src` must point to `rows * k` contiguous `AElem` values.
    /// * `rows` must be in `1..=A_PANEL`.
    /// * The returned pointer must be valid for `A_PANEL * k` reads of
    ///   `APrepared`.
    unsafe fn prepare_a(
        &mut self,
        arch: A,
        src: *const Self::AElem,
        rows: usize,
        k: usize,
    ) -> *const Self::APrepared;

    /// Prepare one B micro-panel for the micro-kernel.
    ///
    /// Called once per B micro-panel in Loop 4. The result is reused across all
    /// A-panels within the current B-panel iteration, amortizing conversion cost.
    ///
    /// For full panels `rows == B_PANEL`. For the remainder panel at the tail of
    /// the B dimension, `rows < B_PANEL`.
    ///
    /// # Safety
    ///
    /// * `src` must point to `rows * k` contiguous `BElem` values.
    /// * `rows` must be in `1..=B_PANEL`.
    /// * The returned pointer must be valid for `rows * k` reads of `BPrepared`.
    unsafe fn prepare_b(
        &mut self,
        arch: A,
        src: *const Self::BElem,
        rows: usize,
        k: usize,
    ) -> *const Self::BPrepared;

    /// Process one full `A_PANEL × B_PANEL` micro-panel pair.
    ///
    /// # Safety
    ///
    /// * `a` must point to `A_PANEL * k` contiguous `APrepared` values.
    /// * `b` must point to `B_PANEL * k` contiguous `BPrepared` values.
    /// * `r` must point to at least `A_PANEL` writable `f32` values.
    unsafe fn full_panel(
        arch: A,
        a: *const Self::APrepared,
        b: *const Self::BPrepared,
        k: usize,
        r: *mut f32,
    );

    /// Dispatch for `1..(B_PANEL-1)` remainder B rows.
    ///
    /// Both A and B are in prepared format — `prepare_b` is called with the
    /// actual remainder row count before this method, so implementors can
    /// reuse the same micro-kernel as `full_panel` with a smaller UNROLL.
    ///
    /// # Safety
    ///
    /// * `a` must point to `A_PANEL * k` contiguous `APrepared` values.
    /// * `b` must point to `remainder * k` contiguous `BPrepared` values.
    /// * `r` must point to at least `A_PANEL` writable `f32` values.
    unsafe fn remainder_dispatch(
        arch: A,
        remainder: usize,
        a: *const Self::APrepared,
        b: *const Self::BPrepared,
        k: usize,
        r: *mut f32,
    );
}
