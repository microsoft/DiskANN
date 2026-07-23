/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Vertex-id abstraction for the bf-tree provider.

use std::marker::PhantomData;

use diskann::utils::VectorId;
use diskann::{ANNError, ANNResult};

/// Identifier type usable as a `BfTreeProvider` vertex id.
///
/// This bundles the bounds the core algorithm requires of an id ([`VectorId`],
/// mandated by `HasId::Id` and `DataProvider::InternalId`) with the index
/// arithmetic the provider needs: converting an id *to* a `usize` index
/// ([`as_index`](BfTreeId::as_index), used to key the per-vector stores) and
/// *from* a zero-based index ([`from_index`](BfTreeId::from_index)). The
/// provider mints ids densely from `0..total`, so it needs a way to build an
/// `I` from a counter.
///
/// Implemented for `u32` (the default, capping at ~4.29B vertices) and `u64`
/// (for billion-scale-and-beyond, larger-than-memory datasets). On a 64-bit
/// target `u64` covers every representable `usize`, so its conversions never
/// fail.
pub trait BfTreeId: VectorId {
    /// Build an id from a zero-based index, truncating on overflow.
    ///
    /// Only call this for indices already known to fit (e.g. ids drawn from
    /// `0..total`, which the provider guarantees fit by construction).
    fn from_index(index: usize) -> Self;

    /// Build an id from a zero-based index, returning `None` if it does not fit.
    fn try_from_index(index: usize) -> Option<Self>;

    /// Convert this id to its zero-based `usize` index.
    ///
    /// The provider uses the identity map, so an id *is* its own index. This is
    /// lossless on the 64-bit targets diskann supports.
    fn as_index(&self) -> usize;

    /// An iterator over the dense id range `0..total`.
    ///
    /// The conversion is monomorphized and inlined per id type (no stored
    /// function pointer), and the iterator preserves the exact-size and
    /// double-ended properties of the underlying index range.
    #[inline]
    fn id_range(total: usize) -> IdRange<Self> {
        IdRange::new(total)
    }
}

impl BfTreeId for u32 {
    #[inline(always)]
    fn from_index(index: usize) -> Self {
        index as u32
    }

    #[inline(always)]
    fn try_from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok()
    }

    #[inline(always)]
    fn as_index(&self) -> usize {
        *self as usize
    }
}

impl BfTreeId for u64 {
    #[inline(always)]
    fn from_index(index: usize) -> Self {
        index as u64
    }

    #[inline(always)]
    fn try_from_index(index: usize) -> Option<Self> {
        u64::try_from(index).ok()
    }

    #[inline(always)]
    fn as_index(&self) -> usize {
        *self as usize
    }
}

/// A dense id iterator yielding `I::from_index(0..total)`.
///
/// Wrapping a `Range<usize>` and mapping inside [`Iterator::next`] keeps the
/// conversion a zero-sized, inlinable function item — unlike `Range::map` with a
/// `fn(usize) -> I` pointer, which forces an indirect call per element — while
/// still delegating length and reverse iteration to the inner range.
#[derive(Debug, Clone)]
pub struct IdRange<I> {
    inner: std::ops::Range<usize>,
    _marker: PhantomData<fn() -> I>,
}

impl<I: BfTreeId> IdRange<I> {
    #[inline]
    fn new(total: usize) -> Self {
        Self {
            inner: 0..total,
            _marker: PhantomData,
        }
    }
}

impl<I: BfTreeId> Iterator for IdRange<I> {
    type Item = I;

    #[inline]
    fn next(&mut self) -> Option<I> {
        self.inner.next().map(I::from_index)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<I: BfTreeId> DoubleEndedIterator for IdRange<I> {
    #[inline]
    fn next_back(&mut self) -> Option<I> {
        self.inner.next_back().map(I::from_index)
    }
}

impl<I: BfTreeId> ExactSizeIterator for IdRange<I> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Validate that a provider holding `total` ids can represent every id in `0..total`.
///
/// `BfTreeProvider::iter` mints ids densely via the infallible (truncating)
/// [`BfTreeId::from_index`]; callers must guarantee the range fits in `I`. This check
/// enforces that guarantee up front (at construction and load) so the truncating
/// conversion can never silently wrap a real id.
pub(crate) fn validate_id_capacity<I: BfTreeId>(total: usize) -> ANNResult<()> {
    if let Some(last) = total.checked_sub(1) {
        if I::try_from_index(last).is_none() {
            return Err(ANNError::log_index_error(format!(
                "provider capacity of {total} ids exceeds the maximum representable by the \
                 {}-byte vertex id type",
                std::mem::size_of::<I>()
            )));
        }
    }
    Ok(())
}
