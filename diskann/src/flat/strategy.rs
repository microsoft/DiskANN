/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`FlatSearchStrategy`] — glue between [`DataProvider`] and per-query [`FlatIterator`]s.

use diskann_vector::PreprocessedDistanceFunction;

use crate::{error::StandardError, flat::OnElementsUnordered, provider::DataProvider};

/// Per-call configuration that knows how to construct a [`FlatIterator`] for a provider
/// and how to pre-process queries of type `T` into a distance computer.
///
/// `FlatSearchStrategy` is the flat counterpart to [`crate::graph::glue::SearchStrategy`].
/// A strategy instance is stateless config — typically constructed at the call site, used
/// for one search, and dropped.
///
/// # Why two methods?
///
/// - [`Self::create_callback`] is query-independent and may be called multiple times per
///   request (e.g., once per parallel query in a batched search).
/// - [`Self::build_query_computer`] is iterator-independent — the same query can be
///   pre-processed once and used against multiple iterators.
///
/// Both methods may borrow from the strategy itself.
///
/// # Type parameters
///
/// - `Provider`: the [`DataProvider`] that backs the index.
/// - `T`: the query type. Often `[E]` for vector queries; can be any `?Sized` type.
pub trait FlatSearchStrategy<P, T>: Send + Sync
where
    P: DataProvider,
    T: ?Sized,
{
    /// The iterator type produced by [`Self::create_callback`]. Borrows from `self` and the
    /// provider.
    type Callback<'a>: OnElementsUnordered
    where
        Self: 'a,
        P: 'a;

    /// The query computer produced by [`Self::build_query_computer`].
    ///
    /// The HRTB on `ElementRef` ensures the same computer can score every element yielded
    /// by every lifetime of `Iter`. Two lifetimes are needed: `'a` for the iterator
    /// instance and `'b` for the reborrowed element.
    type QueryComputer: for<'a, 'b> PreprocessedDistanceFunction<
            <Self::Callback<'a> as OnElementsUnordered>::ElementRef<'b>,
            f32,
        > + Send
        + Sync
        + 'static;

    /// The error type for both factory methods.
    type Error: StandardError;

    /// Construct a fresh iterator over `provider` for the given request `context`.
    ///
    /// This is where lock acquisition, snapshot pinning, and any other per-query setup
    /// should happen. The returned callback object owns whatever borrows / guards it needs to
    /// remain valid until it is dropped.
    fn create_callback<'a>(
        &'a self,
        provider: &'a P,
        context: &'a P::Context,
    ) -> Result<Self::Callback<'a>, Self::Error>;

    /// Pre-process a query into a [`Self::QueryComputer`] usable for distance computation
    /// against any iterator produced by [`Self::create_callback`].
    fn build_query_computer(&self, query: &T) -> Result<Self::QueryComputer, Self::Error>;
}
