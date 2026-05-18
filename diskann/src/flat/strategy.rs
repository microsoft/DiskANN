/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Core flat-search traits: [`DistancesUnordered`] and [`SearchStrategy`].

use std::fmt::Debug;

use diskann_utils::future::SendFuture;
use diskann_vector::PreprocessedDistanceFunction;

use crate::{
    error::{StandardError, ToRanked},
    provider::{DataProvider, HasId},
};

/// Fused iterate-and-score primitive over the elements of a flat index.
///
/// Implementations drive an entire scan over the underlying data, scoring each element
/// with the supplied computer `C` and invoking `f` with the resulting `(id, distance)`
/// pair. The associated [`Self::ElementRef`] is the reference shape on which `C` must
/// be able to compute distances.
pub trait DistancesUnordered<C>: HasId + Send + Sync
where
    C: for<'a> PreprocessedDistanceFunction<Self::ElementRef<'a>, f32>,
{
    /// Lifetime is intentionally unconstrained so it can appear under HRTB without
    /// inducing a `'static` bound on `Self`.
    type ElementRef<'a>;

    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Drive the entire scan, scoring each element with `computer` and invoking `f`
    /// with the resulting `(id, distance)` pair.
    fn distances_unordered<F>(
        &mut self,
        computer: &C,
        f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + FnMut(<Self as HasId>::Id, f32);
}

/// Per-call configuration that knows how to construct a per-query
/// [`DistancesUnordered`] visitor for a provider, and the [`Self::QueryComputer`] used
/// to score each element during the scan.
pub trait SearchStrategy<P, T>: Send + Sync
where
    P: DataProvider,
{
    /// The reference element shape on which [`Self::QueryComputer`] computes
    /// distances.
    type ElementRef<'a>;

    /// The concrete query-computer type.
    type QueryComputer: for<'a> PreprocessedDistanceFunction<Self::ElementRef<'a>, f32>
        + Send
        + Sync
        + 'static;

    /// The error type for [`Self::build_query_computer`].
    type QueryComputerError: StandardError;

    /// The visitor type produced by [`Self::create_visitor`].
    type Visitor<'a>: for<'b> DistancesUnordered<
            Self::QueryComputer,
            ElementRef<'b> = Self::ElementRef<'b>,
            Id = P::InternalId,
        >
    where
        Self: 'a,
        P: 'a;

    /// The error type for [`Self::create_visitor`].
    type Error: StandardError;

    /// Construct a fresh visitor over `provider` for the given request `context`.
    fn create_visitor<'a>(
        &'a self,
        provider: &'a P,
        context: &'a P::Context,
    ) -> Result<Self::Visitor<'a>, Self::Error>;

    /// Construct the per-query computer.
    fn build_query_computer(
        &self,
        query: T,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError>;
}
