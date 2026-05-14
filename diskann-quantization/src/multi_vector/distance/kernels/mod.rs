// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Block-transposed SIMD kernels for multi-vector distance computation.
//!
//! This module provides a SIMD-accelerated implementation that uses block-transposed
//! memory layout for **query** vectors (instead of documents), with documents remaining
//! in row-major format.
//!
//! # Memory Layout
//!
//! - **Query**: Block-transposed (`GROUP` vectors per block, dimensions contiguous
//!   within each block). The block size is determined by the kernel's `A_PANEL`.
//! - **Document**: Row-major (standard [`MatRef`](crate::multi_vector::MatRef) format).
//!
//! The [`Kernel<A>`] trait + [`tiled_reduce`] + [`layouts`] are public so
//! external crates can plug new micro-kernels into the existing orchestrator.

pub(super) mod f16;
pub(super) mod f32;
pub mod layouts;
mod reduce;
mod tiled_reduce;

pub use tiled_reduce::tiled_reduce;

// ── Tile budget ──────────────────────────────────────────────────

/// Cache budgets fed to the tile planner.
///
/// `Default` returns the production budgets derived from hardcoded L1/L2
/// cache-size estimates and fixed fractions. Researchers benchmarking with
/// non-default cache assumptions can construct a custom [`TileBudget`] via
/// [`TileBudget::new`] and pass it to [`tiled_reduce`].
#[derive(Debug, Clone, Copy)]
pub struct TileBudget {
    /// L2 budget in bytes reserved for A tiles.
    l2_a: usize,
    /// L1 budget in bytes reserved for B tiles (before A-panel subtraction).
    l1_b: usize,
}

impl TileBudget {
    /// Construct a [`TileBudget`] with explicit L2 and L1 byte budgets.
    ///
    /// `l2_a` is the budget the tile planner uses to size A tiles; `l1_b`
    /// is the budget for B tiles (one A micro-panel is subtracted at
    /// runtime since both must coexist in L1 during the inner loop).
    pub fn new(l2_a: usize, l1_b: usize) -> Self {
        Self { l2_a, l1_b }
    }
}

impl Default for TileBudget {
    // TODO: Replace hardcoded fallbacks with detected cache sizes
    // (e.g. via `diskann_platform`, env-var override, or runtime query).
    fn default() -> Self {
        const L2_CACHE: usize = 1_250_000; // 1.25 MB fallback
        const L1_CACHE: usize = 48_000; // 48 KB fallback

        Self {
            // 50% of L2 for A tiles; remainder for B streaming + pollution.
            l2_a: L2_CACHE / 2,
            // 75% of L1 for B tiles; A micro-panel subtracted at runtime.
            l1_b: L1_CACHE * 3 / 4,
        }
    }
}

// ── Kernel trait ─────────────────────────────────────────────────

/// SIMD micro-kernel for the [`tiled_reduce`] loop.
///
/// The kernel only sees already-converted data: storage-layout to
/// kernel-layout conversion is handled at tile boundaries by
/// [`ConvertTo`](layouts::ConvertTo), so implementors can assume their input
/// pointers reference `<Self::Left as Layout>::Element` /
/// `<Self::Right as Layout>::Element` directly.
///
/// # Invariant
///
/// When pairing this kernel with the owning storage type
/// [`BlockTransposed<T, GROUP>`](crate::multi_vector::BlockTransposed) via
/// [`tiled_reduce`], the storage's `GROUP` const must equal this kernel's
/// [`A_PANEL`](Self::A_PANEL). The library's own f32 / f16 paths enforce
/// this with a `const { assert!(...) }` in `max_ip_kernel`; external
/// implementors must uphold it manually.
///
/// # Safety
///
/// Implementors must respect the per-method `# Safety` contracts on
/// [`full_panel`](Self::full_panel) and [`partial_panel`](Self::partial_panel).
/// Implementations should be validated under Miri: construct arch tokens
/// via `*::new_checked_miri()` in tests and gate Miri-unsupported
/// intrinsics with `#[cfg(not(miri))]`.
pub unsafe trait Kernel<A: diskann_wide::Architecture> {
    /// Layout consumed by the A (left / query) side of the micro-kernel.
    type Left: layouts::Layout;
    /// Layout consumed by the B (right / document) side of the micro-kernel.
    type Right: layouts::Layout;

    /// Number of A rows processed per micro-kernel invocation.
    ///
    /// Callers of [`tiled_reduce`] must guarantee
    /// `a_padded_nrows % A_PANEL == 0`.
    const A_PANEL: usize;
    /// Number of B rows processed per micro-kernel invocation.
    const B_PANEL: usize;

    /// Process one full `A_PANEL × B_PANEL` micro-panel pair.
    ///
    /// # Safety
    ///
    /// * `a` must point to `A_PANEL * k` contiguous elements of
    ///   `<Self::Left as Layout>::Element`, properly aligned for that layout.
    /// * `b` must point to `B_PANEL * k` contiguous elements of
    ///   `<Self::Right as Layout>::Element`, properly aligned for that layout.
    /// * `r` must point to at least `A_PANEL` writable `f32` values.
    /// * `k > 0`.
    /// * The caller must invoke this from within an `arch.run3` (or
    ///   equivalent) so that target_feature is active for the entire body.
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
    /// Same as [`full_panel`](Self::full_panel) except `b` points to
    /// `remainder * k` contiguous elements and `1 <= remainder < B_PANEL`.
    unsafe fn partial_panel(
        arch: A,
        remainder: usize,
        a: *const <Self::Left as layouts::Layout>::Element,
        b: *const <Self::Right as layouts::Layout>::Element,
        k: usize,
        r: *mut f32,
    );
}
