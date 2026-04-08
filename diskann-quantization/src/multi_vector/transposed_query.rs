// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Architecture-opaque transposed query types with runtime dispatch.
//!
//! These types hide the block-transposed `GROUP` parameter behind an enum,
//! allowing callers to work with transposed queries without knowing which
//! micro-architecture was detected at runtime.
//!
//! # Usage
//!
//! ```
//! use diskann_quantization::multi_vector::{
//!     transpose_query, MatRef, Standard, Chamfer,
//! };
//! use diskann_vector::PureDistanceFunction;
//!
//! let query_data = [1.0f32, 0.0, 0.0, 1.0];
//! let doc_data = [1.0f32, 0.0, 0.0, 1.0];
//!
//! let query = MatRef::new(Standard::new(2, 2).unwrap(), &query_data).unwrap();
//! let doc = MatRef::new(Standard::new(2, 2).unwrap(), &doc_data).unwrap();
//!
//! // Transpose — runtime detects arch, picks optimal GROUP
//! let tq = transpose_query(query.as_matrix_view());
//!
//! // Distance — runtime dispatches to matching micro-kernel
//! let dist = Chamfer::evaluate(tq.as_ref(), doc);
//! assert_eq!(dist, -2.0);
//! ```

use diskann_utils::views::MatrixView;

use super::block_transposed::{BlockTransposed, BlockTransposedRef};

// ── Opaque transposed query types ────────────────────────────────

/// An owning block-transposed query matrix with runtime-selected `GROUP`.
///
/// Created by [`transpose_query`] or [`transpose_query_f16`]. The `GROUP`
/// constant is hidden inside the enum — callers never see it.
#[derive(Debug)]
pub enum TransposedQuery<T: Copy> {
    /// Block size 8 (Scalar / Neon).
    Group8(BlockTransposed<T, 8>),
    /// Block size 16 (x86_64 AVX2+ / AVX-512).
    Group16(BlockTransposed<T, 16>),
}

/// An immutable view of a [`TransposedQuery`] with runtime-selected `GROUP`.
///
/// This is `Copy` and lightweight — it borrows the underlying storage.
#[derive(Debug, Clone, Copy)]
pub enum TransposedQueryRef<'a, T: Copy> {
    /// Block size 8 (Scalar / Neon).
    Group8(BlockTransposedRef<'a, T, 8>),
    /// Block size 16 (x86_64 AVX2+ / AVX-512).
    Group16(BlockTransposedRef<'a, T, 16>),
}

// ── TransposedQuery methods ──────────────────────────────────────

impl<T: Copy> TransposedQuery<T> {
    /// Number of logical (non-padded) query vectors.
    #[inline]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Group8(bt) => bt.nrows(),
            Self::Group16(bt) => bt.nrows(),
        }
    }

    /// Dimensionality of each query vector.
    #[inline]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Group8(bt) => bt.ncols(),
            Self::Group16(bt) => bt.ncols(),
        }
    }

    /// Total available rows (padded to GROUP boundary).
    #[inline]
    pub fn available_rows(&self) -> usize {
        match self {
            Self::Group8(bt) => bt.available_rows(),
            Self::Group16(bt) => bt.available_rows(),
        }
    }

    /// Borrow as an immutable [`TransposedQueryRef`].
    #[inline]
    pub fn as_ref(&self) -> TransposedQueryRef<'_, T> {
        match self {
            Self::Group8(bt) => TransposedQueryRef::Group8(bt.as_view()),
            Self::Group16(bt) => TransposedQueryRef::Group16(bt.as_view()),
        }
    }
}

// ── TransposedQueryRef methods ───────────────────────────────────

impl<T: Copy> TransposedQueryRef<'_, T> {
    /// Number of logical (non-padded) query vectors.
    #[inline]
    pub fn nrows(&self) -> usize {
        match self {
            Self::Group8(bt) => bt.nrows(),
            Self::Group16(bt) => bt.nrows(),
        }
    }

    /// Dimensionality of each query vector.
    #[inline]
    pub fn ncols(&self) -> usize {
        match self {
            Self::Group8(bt) => bt.ncols(),
            Self::Group16(bt) => bt.ncols(),
        }
    }

    /// Total available rows (padded to GROUP boundary).
    #[inline]
    pub fn available_rows(&self) -> usize {
        match self {
            Self::Group8(bt) => bt.available_rows(),
            Self::Group16(bt) => bt.available_rows(),
        }
    }
}

// ── Transpose factories ──────────────────────────────────────────

/// Block-transpose a row-major f32 query matrix, selecting the optimal `GROUP`
/// for the current CPU at runtime.
///
/// Returns an owning [`TransposedQuery`] that can be reused across multiple
/// distance calls via [`TransposedQuery::as_ref`].
pub fn transpose_query(query: MatrixView<'_, f32>) -> TransposedQuery<f32> {
    diskann_wide::arch::dispatch1_no_features(TransposeF32, query)
}

/// Block-transpose a row-major f16 query matrix, selecting the optimal `GROUP`
/// for the current CPU at runtime.
pub fn transpose_query_f16(query: MatrixView<'_, half::f16>) -> TransposedQuery<half::f16> {
    diskann_wide::arch::dispatch1_no_features(TransposeF16, query)
}

// TODO: transpose_query_into / transpose_query_f16_into (buffer-reuse) —
// deferred to follow-up; requires writing block-transposed data into a
// caller-owned buffer without allocating.

// ── Allocating transpose dispatcher ──────────────────────────────

#[derive(Clone, Copy)]
struct TransposeF32;

impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::Scalar,
        TransposedQuery<f32>,
        MatrixView<'_, f32>,
    > for TransposeF32
{
    fn run(
        self,
        _arch: diskann_wide::arch::Scalar,
        query: MatrixView<'_, f32>,
    ) -> TransposedQuery<f32> {
        TransposedQuery::Group8(BlockTransposed::<f32, 8>::from_matrix_view(query))
    }
}

#[cfg(target_arch = "x86_64")]
impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::x86_64::V3,
        TransposedQuery<f32>,
        MatrixView<'_, f32>,
    > for TransposeF32
{
    fn run(
        self,
        _arch: diskann_wide::arch::x86_64::V3,
        query: MatrixView<'_, f32>,
    ) -> TransposedQuery<f32> {
        TransposedQuery::Group16(BlockTransposed::<f32, 16>::from_matrix_view(query))
    }
}

#[cfg(target_arch = "x86_64")]
impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::x86_64::V4,
        TransposedQuery<f32>,
        MatrixView<'_, f32>,
    > for TransposeF32
{
    fn run(
        self,
        _arch: diskann_wide::arch::x86_64::V4,
        query: MatrixView<'_, f32>,
    ) -> TransposedQuery<f32> {
        TransposedQuery::Group16(BlockTransposed::<f32, 16>::from_matrix_view(query))
    }
}

#[cfg(target_arch = "aarch64")]
impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::aarch64::Neon,
        TransposedQuery<f32>,
        MatrixView<'_, f32>,
    > for TransposeF32
{
    fn run(
        self,
        _arch: diskann_wide::arch::aarch64::Neon,
        query: MatrixView<'_, f32>,
    ) -> TransposedQuery<f32> {
        TransposedQuery::Group8(BlockTransposed::<f32, 8>::from_matrix_view(query))
    }
}

// ── Allocating transpose dispatcher (f16) ────────────────────────

#[derive(Clone, Copy)]
struct TransposeF16;

impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::Scalar,
        TransposedQuery<half::f16>,
        MatrixView<'_, half::f16>,
    > for TransposeF16
{
    fn run(
        self,
        _arch: diskann_wide::arch::Scalar,
        query: MatrixView<'_, half::f16>,
    ) -> TransposedQuery<half::f16> {
        TransposedQuery::Group8(BlockTransposed::<half::f16, 8>::from_matrix_view(query))
    }
}

#[cfg(target_arch = "x86_64")]
impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::x86_64::V3,
        TransposedQuery<half::f16>,
        MatrixView<'_, half::f16>,
    > for TransposeF16
{
    fn run(
        self,
        _arch: diskann_wide::arch::x86_64::V3,
        query: MatrixView<'_, half::f16>,
    ) -> TransposedQuery<half::f16> {
        TransposedQuery::Group16(BlockTransposed::<half::f16, 16>::from_matrix_view(query))
    }
}

#[cfg(target_arch = "x86_64")]
impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::x86_64::V4,
        TransposedQuery<half::f16>,
        MatrixView<'_, half::f16>,
    > for TransposeF16
{
    fn run(
        self,
        _arch: diskann_wide::arch::x86_64::V4,
        query: MatrixView<'_, half::f16>,
    ) -> TransposedQuery<half::f16> {
        TransposedQuery::Group16(BlockTransposed::<half::f16, 16>::from_matrix_view(query))
    }
}

#[cfg(target_arch = "aarch64")]
impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::aarch64::Neon,
        TransposedQuery<half::f16>,
        MatrixView<'_, half::f16>,
    > for TransposeF16
{
    fn run(
        self,
        _arch: diskann_wide::arch::aarch64::Neon,
        query: MatrixView<'_, half::f16>,
    ) -> TransposedQuery<half::f16> {
        TransposedQuery::Group8(BlockTransposed::<half::f16, 8>::from_matrix_view(query))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f32_view(nrows: usize, ncols: usize) -> (Vec<f32>, MatrixView<'static, f32>) {
        let data = vec![1.0f32; nrows * ncols];
        // SAFETY: we leak a clone of the vec so the slice lives for 'static.
        let slice: &'static [f32] = Vec::leak(data.clone());
        let view = MatrixView::try_from(slice, nrows, ncols).unwrap();
        (data, view)
    }

    fn make_f16_view(
        nrows: usize,
        ncols: usize,
    ) -> (Vec<half::f16>, MatrixView<'static, half::f16>) {
        let data = vec![half::f16::from_f32(1.0); nrows * ncols];
        let slice: &'static [half::f16] = Vec::leak(data.clone());
        let view = MatrixView::try_from(slice, nrows, ncols).unwrap();
        (data, view)
    }

    #[test]
    fn transpose_query_dimensions() {
        let (_data, view) = make_f32_view(5, 8);
        let tq = transpose_query(view);

        assert_eq!(tq.nrows(), 5);
        assert_eq!(tq.ncols(), 8);
        assert!(tq.available_rows() >= 5);
        assert_eq!(tq.available_rows() % 8, 0);
    }

    #[test]
    fn transpose_query_as_ref_preserves_dimensions() {
        let (_data, view) = make_f32_view(5, 8);
        let tq = transpose_query(view);
        let tq_ref = tq.as_ref();

        assert_eq!(tq_ref.nrows(), tq.nrows());
        assert_eq!(tq_ref.ncols(), tq.ncols());
        assert_eq!(tq_ref.available_rows(), tq.available_rows());
    }

    #[test]
    fn transpose_query_f16_dimensions() {
        let (_data, view) = make_f16_view(5, 8);
        let tq = transpose_query_f16(view);

        assert_eq!(tq.nrows(), 5);
        assert_eq!(tq.ncols(), 8);
        assert!(tq.available_rows() >= 5);
        assert_eq!(tq.available_rows() % 8, 0);
    }

    #[test]
    fn transpose_query_f16_as_ref_preserves_dimensions() {
        let (_data, view) = make_f16_view(5, 8);
        let tq = transpose_query_f16(view);
        let tq_ref = tq.as_ref();

        assert_eq!(tq_ref.nrows(), tq.nrows());
        assert_eq!(tq_ref.ncols(), tq.ncols());
        assert_eq!(tq_ref.available_rows(), tq.available_rows());
    }

    #[test]
    fn transpose_query_single_vector() {
        let (_data, view) = make_f32_view(1, 4);
        let tq = transpose_query(view);

        assert_eq!(tq.nrows(), 1);
        assert_eq!(tq.ncols(), 4);
        assert!(tq.available_rows() >= 1);
    }

    #[test]
    fn transpose_query_ref_is_copy() {
        let (_data, view) = make_f32_view(3, 4);
        let tq = transpose_query(view);
        let r1 = tq.as_ref();
        let r2 = r1;
        // Both copies are valid and equal.
        assert_eq!(r1.nrows(), r2.nrows());
        assert_eq!(r1.ncols(), r2.ncols());
    }
}
