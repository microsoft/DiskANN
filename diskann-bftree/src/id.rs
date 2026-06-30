/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Vertex-id abstraction for the bf-tree provider.

use diskann::utils::{IntoUsize, VectorId};
use diskann::{ANNError, ANNResult};

/// Identifier type usable as a `BfTreeProvider` vertex id.
///
/// This bundles the bounds the core algorithm requires of an id ([`VectorId`])
/// with the ability to convert *to* `usize` ([`IntoUsize`], used to key the
/// per-vector stores) and *from* a zero-based index. The provider mints ids
/// densely from `0..total`, so it needs a way to build an `I` from a counter.
///
/// Implemented for `u32` (the default, capping at ~4.29B vertices) and `u64`
/// (for billion-scale-and-beyond, larger-than-memory datasets). On a 64-bit
/// target `u64` covers every representable `usize`, so its conversions never
/// fail.
pub trait BfTreeId: VectorId + IntoUsize {
    /// Build an id from a zero-based index, truncating on overflow.
    ///
    /// Only call this for indices already known to fit (e.g. ids drawn from
    /// `0..total`, which the provider guarantees fit by construction).
    fn from_index(index: usize) -> Self;

    /// Build an id from a zero-based index, returning `None` if it does not fit.
    fn try_from_index(index: usize) -> Option<Self>;
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
