/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`FlatPostProcess`] — terminal stage of the flat search pipeline.

use diskann_utils::future::SendFuture;

use crate::{
    error::StandardError, flat::FlatIterator, graph::SearchOutputBuffer, neighbor::Neighbor,
    provider::HasId,
};

/// Post-process the survivor candidates produced by a flat search and
/// write them into an output buffer.
///
/// This is the flat counterpart to [`crate::graph::glue::SearchPostProcess`]. Processors
/// receive `&mut S` so they can consult any iterator-owned lookup state (e.g., an
/// `Id -> rich-record` table built up during the scan) when assembling outputs.
///
/// The `O` type parameter lets callers pick the output element type (raw `(Id, f32)`
/// pairs, fully hydrated hits etc.).
pub trait FlatPostProcess<S, T, O = <S as HasId>::Id>
where
    S: FlatIterator,
    T: ?Sized,
{
    /// Errors yielded by [`Self::post_process`].
    type Error: StandardError;

    /// Consume `candidates` (in distance order) and write at most `k` results into
    /// `output`. Returns the number of results written.
    fn post_process<I, B>(
        &self,
        iter: &mut S,
        query: &T,
        candidates: I,
        output: &mut B,
    ) -> impl SendFuture<Result<usize, Self::Error>>
    where
        I: Iterator<Item = Neighbor<S::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized;
}

/// A trivial [`FlatPostProcess`] that copies each `(Id, distance)` pair straight into the
/// output buffer.
#[derive(Debug, Default, Clone, Copy)]
pub struct CopyFlatIds;

impl<S, T> FlatPostProcess<S, T> for CopyFlatIds
where
    S: FlatIterator,
    T: ?Sized,
{
    type Error = crate::error::Infallible;

    fn post_process<I, B>(
        &self,
        _iter: &mut S,
        _query: &T,
        candidates: I,
        output: &mut B,
    ) -> impl SendFuture<Result<usize, Self::Error>>
    where
        I: Iterator<Item = Neighbor<<S as HasId>::Id>> + Send,
        B: SearchOutputBuffer<<S as HasId>::Id> + Send + ?Sized,
    {
        let count = output.extend(candidates.map(|n| (n.id, n.distance)));
        std::future::ready(Ok(count))
    }
}
