/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`SearchStrategy`] — glue between [`DataProvider`] and per-query [`crate::flat::FlatIterator`]s.

use crate::{error::StandardError, flat::DistancesUnordered, provider::DataProvider};

/// Per-call configuration that knows how to construct a per-query
/// [`DistancesUnordered<T>`] visitor for a provider.
///
/// `SearchStrategy` is the flat counterpart to [`crate::graph::glue::SearchStrategy`]
/// (disambiguated by module path). A strategy instance is stateless config — typically
/// constructed at the call site, used for one search, and dropped.
///
/// The strategy itself is a pure factory; the visitor it produces carries the
/// query-preprocessing capability via [`crate::provider::BuildQueryComputer<T>`] (a
/// super-trait of [`DistancesUnordered<T>`]).
///
/// # Type parameters
///
/// - `P`: the [`DataProvider`] that backs the index.
/// - `T`: the query type that the query computer is constructed using.
pub trait SearchStrategy<P, T>: Send + Sync
where
    P: DataProvider,
{
    /// The visitor type produced by [`Self::create_visitor`]. Borrows from `self` and the
    /// provider.
    ///
    /// The visitor implements both the streaming [`DistancesUnordered<T>`] primitive and
    /// the query preprocessor [`crate::provider::BuildQueryComputer<T>`].
    type Visitor<'a>: DistancesUnordered<T>
    where
        Self: 'a,
        P: 'a;

    /// The error type for [`Self::create_visitor`].
    type Error: StandardError;

    /// Construct a fresh visitor over `provider` for the given request `context`.
    ///
    /// This is where lock acquisition, snapshot pinning, and any other per-query setup
    /// should happen. The returned visitor owns whatever borrows / guards it needs to
    /// remain valid until it is dropped.
    fn create_visitor<'a>(
        &'a self,
        provider: &'a P,
        context: &'a P::Context,
    ) -> Result<Self::Visitor<'a>, Self::Error>;
}
