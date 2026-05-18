// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Layout markers and tile-level conversion traits.
//!
//! - [`Layout`] — marker trait: memory layout + element type.
//! - [`BlockTransposed`] / [`RowMajor`] — zero-sized layout markers.
//! - [`DescribeLayout`] — bridges matrix types to layout markers.
//! - [`ConvertTo`] — tile-level conversion (blanket identity + f16→f32).

use core::marker::PhantomData;

use diskann_vector::conversion::SliceCast;
use diskann_wide::Architecture;
use diskann_wide::arch::Target2;

// ── Layout trait ─────────────────────────────────────

/// Memory layout and element type marker for tile data.
pub(super) trait Layout {
    type Element: Copy;
}

// ── Layout markers ───────────────────────────────────

/// Block-transposed tile layout: `GROUP` rows per block, `PACK` columns
/// interleaved. Matches [`BlockTransposedRef`](crate::multi_vector::BlockTransposedRef).
pub(super) struct BlockTransposed<T, const GROUP: usize, const PACK: usize = 1>(PhantomData<T>);

impl<T, const GROUP: usize, const PACK: usize> BlockTransposed<T, GROUP, PACK> {
    pub(super) fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T, const GROUP: usize, const PACK: usize> Copy for BlockTransposed<T, GROUP, PACK> {}

impl<T, const GROUP: usize, const PACK: usize> Clone for BlockTransposed<T, GROUP, PACK> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Copy, const GROUP: usize, const PACK: usize> Layout for BlockTransposed<T, GROUP, PACK> {
    type Element = T;
}

/// Dense row-major tile layout. Matches [`MatRef<Standard<T>>`](crate::multi_vector::MatRef).
pub(super) struct RowMajor<T>(PhantomData<T>);

impl<T> RowMajor<T> {
    pub(super) fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Copy for RowMajor<T> {}

impl<T> Clone for RowMajor<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Copy> Layout for RowMajor<T> {
    type Element = T;
}

// ── DescribeLayout ───────────────────────────────────

/// Bridges a concrete matrix type to its [`Layout`] marker, enabling
/// type inference of [`ConvertTo`] parameters at call sites.
pub(super) trait DescribeLayout {
    type Layout: Layout;

    fn layout(&self) -> Self::Layout;
}

impl<T: Copy, const GROUP: usize, const PACK: usize> DescribeLayout
    for crate::multi_vector::BlockTransposedRef<'_, T, GROUP, PACK>
{
    type Layout = BlockTransposed<T, GROUP, PACK>;

    fn layout(&self) -> Self::Layout {
        BlockTransposed::new()
    }
}

impl<T: Copy> DescribeLayout for crate::multi_vector::MatRef<'_, crate::multi_vector::Standard<T>> {
    type Layout = RowMajor<T>;

    fn layout(&self) -> Self::Layout {
        RowMajor::new()
    }
}

// ── ConvertTo trait ──────────────────────────────────

/// Tile-level conversion from layout `Self` to layout `To`.
///
/// The blanket identity impl covers every layout converting to itself
/// with `Buffer = ()` and zero cost. Explicit impls handle f16→f32 via
/// [`SliceCast`].
///
/// # Safety
///
/// Implementors must ensure:
/// - `convert` reads at most `rows * k` source elements.
/// - `convert` writes only within `buf`.
/// - The returned pointer is valid until the next `&mut` access to `buf`.
pub(super) unsafe trait ConvertTo<A: Architecture, To: Layout>: Layout {
    /// Staging buffer for converted tile data (`()` for identity conversions).
    type Buffer;

    /// Allocate a buffer for up to `max_tile_rows` rows of dimension `k`.
    fn new_buffer(&self, max_tile_rows: usize, k: usize) -> Self::Buffer;

    /// Convert `rows` rows of source data into `buf`, returning a read pointer.
    ///
    /// # Safety
    ///
    /// * `src` must point to `rows * k` valid elements in `Self`'s layout.
    /// * `buf` must come from [`new_buffer`](Self::new_buffer) with the
    ///   same `k` and a `max_tile_rows >= rows`.
    unsafe fn convert(
        &self,
        buf: &mut Self::Buffer,
        arch: A,
        src: *const Self::Element,
        rows: usize,
        k: usize,
    ) -> *const To::Element;
}

// ── Blanket identity ─────────────────────────────────

/// Identity conversion: every layout converts to itself at zero cost.
// SAFETY: Identity conversion reads nothing beyond `src` and writes
// nothing into `buf`. The returned pointer is exactly `src`, which is
// valid for the lifetime guaranteed by the caller.
unsafe impl<A: Architecture, L: Layout> ConvertTo<A, L> for L {
    type Buffer = ();

    fn new_buffer(&self, _max_tile_rows: usize, _k: usize) {}

    unsafe fn convert(
        &self,
        _buf: &mut (),
        _arch: A,
        src: *const L::Element,
        _rows: usize,
        _k: usize,
    ) -> *const L::Element {
        src
    }
}

// ── f16 → f32 conversions ────────────────────────────

/// Block-transposed f16 → block-transposed f32 (element-wise, layout-preserving).
// SAFETY: `SliceCast` converts exactly `rows * k` f16 values from `src`
// into `rows * k` f32 values in `buf`. The returned pointer is
// `buf.as_ptr()`, valid until the next `&mut` access to `buf`.
unsafe impl<A, const GROUP: usize, const PACK: usize>
    ConvertTo<A, BlockTransposed<f32, GROUP, PACK>> for BlockTransposed<half::f16, GROUP, PACK>
where
    A: Architecture,
    SliceCast<f32, half::f16>: for<'a> Target2<A, (), &'a mut [f32], &'a [half::f16]>,
{
    type Buffer = Vec<f32>;

    fn new_buffer(&self, max_tile_rows: usize, k: usize) -> Vec<f32> {
        vec![0.0f32; max_tile_rows * k]
    }

    unsafe fn convert(
        &self,
        buf: &mut Vec<f32>,
        arch: A,
        src: *const half::f16,
        rows: usize,
        k: usize,
    ) -> *const f32 {
        let count = rows * k;
        // SAFETY: Caller guarantees `src` points to `count` contiguous f16 values.
        let src_slice = unsafe { std::slice::from_raw_parts(src, count) };
        arch.run2(SliceCast::new(), &mut buf[..count], src_slice);
        buf.as_ptr()
    }
}

/// Row-major f16 → row-major f32 (element-wise, layout-preserving).
// SAFETY: Same as block-transposed variant — element-wise, layout-preserving.
unsafe impl<A> ConvertTo<A, RowMajor<f32>> for RowMajor<half::f16>
where
    A: Architecture,
    SliceCast<f32, half::f16>: for<'a> Target2<A, (), &'a mut [f32], &'a [half::f16]>,
{
    type Buffer = Vec<f32>;

    fn new_buffer(&self, max_tile_rows: usize, k: usize) -> Vec<f32> {
        vec![0.0f32; max_tile_rows * k]
    }

    unsafe fn convert(
        &self,
        buf: &mut Vec<f32>,
        arch: A,
        src: *const half::f16,
        rows: usize,
        k: usize,
    ) -> *const f32 {
        let count = rows * k;
        // SAFETY: Caller guarantees `src` points to `count` contiguous f16 values.
        let src_slice = unsafe { std::slice::from_raw_parts(src, count) };
        arch.run2(SliceCast::new(), &mut buf[..count], src_slice);
        buf.as_ptr()
    }
}
