/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Distance layers indexing.
//!
//! An important assumption made by this module is that the data within each layer is
//! uniformly sized: each entry occupies the same number of bytes. Furthermore, the data
//! to be stored may not assume any particular alignment. Implementations will strive to
//! achieve a reasonable alignment, but this may not be relied on.
//!
//! # Query Distance Specialization
//!
//! The design of this module allows aggressive optimization of graph search kernels via
//! the [`Search`] and [`QueryVisitor`] pairs of traits.
//!
//! Implementations of [`Search`] can pass a [`QueryDistance`] kernel specialized to
//! a specific geometry (dimensionality or metric type) which upstream [`QueryVisitor`]
//! will fuse into larger kernels. While this allows for high performance graph kernels,
//! some considerations should be taken into account:
//!
//! 1. For correctness purposes, upstream callers cannot do any kind of caching. As such,
//!    the dispatch layer used to select the kernel passed to the [`QueryVisitor`] should
//!    be relatively efficient.
//!
//! 2. Keep the number of specializations bounded for compile time reasons.

use diskann::ANNResult;

use crate::num::Bytes;

mod full;
pub use full::{Full, FullPrecision};

/// Base layer for data representations.
pub trait Layer: Send + Sync + 'static {
    /// Return the number of bytes needed by this layer representation.
    ///
    /// To be well-behaved, this function must be idempotent.
    fn bytes(&self) -> Bytes;
}

/// Store an element of type `T` into a raw byte buffer.
///
/// Implementations may assume that `bytes.len()` is equal to [`Layer::bytes`].
pub trait Set<T>: Layer {
    /// Write into the stored representation.
    fn set(&self, element: T, bytes: &mut [u8]) -> ANNResult<()>;
}

/// A distance computation on raw byte slices.
///
/// When paired with [`Layer`] via helpers like [`AsDistance`], implementations may assume
/// that `x` and `y` have length [`Layer::bytes`].
///
/// No alignment guarantees are made for `x` and `y`, though in practice they are likely
/// to be aligned to 32 or 64 bytes.
pub trait Distance: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, x: &[u8], y: &[u8]) -> ANNResult<f32>;
}

/// Return a [`Distance`] function for a [`Layer`].
pub trait AsDistance: Send + Sync + std::fmt::Debug {
    fn as_distance(&self) -> &dyn Distance;
}

/// A unary query distance on raw byte slices.
///
/// When paired with [`Layer`] via helpers like [`Search`], implementations may assume
/// that `x` has length [`Layer::bytes`].
///
/// No alignment guarantees are made for `x`, though in practice it is likely to be
/// aligned to 32 or 64 bytes.
pub trait QueryDistance: Send + Sync + std::fmt::Debug {
    fn evaluate(&self, x: &[u8]) -> ANNResult<f32>;
}

/// Enable search over vectors defined by a [`Layer`].
pub trait Search: Send + Sync + 'static {
    /// The type of the query. This should be equivalent to the generic parameter in
    /// [`Set`], but needs to be replicated here due to limitations in the current trait
    /// design.
    type Query<'a>;

    /// Create a distance computer specialized for `query` and provide it to `visitor`.
    fn query_distance<'a, V>(&'a self, query: Self::Query<'a>, visitor: V) -> ANNResult<V::Output>
    where
        V: QueryVisitor<'a>;
}

/// Specialize a kernel around a [`QueryDistance`] implementation.
pub trait QueryVisitor<'a>: Sized {
    /// The type of the type-erased output.
    type Output;

    /// Specialize [`Self::Output`] for `distance`.
    fn visit<T>(self, distance: T) -> Self::Output
    where
        T: QueryDistance + 'a;

    /// Specialize [`Self::Output`] for `distance` accepting a hint that `distance` has been
    /// specialized to work on data elements of exactly `BYTES` bytes long.
    ///
    /// This can be used to tailor surrounding code (e.g. software prefetches) for exactly
    /// the length of the data being processed.
    fn visit_sized<const BYTES: usize, T>(self, distance: T) -> Self::Output
    where
        T: QueryDistance + 'a,
    {
        self.visit(distance)
    }
}

/// A insert-specific specialization of [`Search`].
///
/// Note that the bounds for this trait are unnecessarily complicated, but rely on changes
/// to `diskann` to full resolve.
pub trait Insert: Search + for<'a> Set<Self::Query<'a>> + AsDistance {
    /// A specialization of [`Search::query_distance`] targeting vector insert specifically.
    fn insert_distance<'a, V>(&'a self, query: Self::Query<'a>, visitor: V) -> ANNResult<V::Output>
    where
        V: QueryVisitor<'a>,
    {
        self.query_distance(query, visitor)
    }
}
