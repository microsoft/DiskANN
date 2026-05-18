/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`SearchStrategy`] — glue between [`DataProvider`] and per-query
//! [`DistancesUnordered`] visitors.

use diskann_vector::PreprocessedDistanceFunction;

use crate::{
    error::StandardError, flat::DistancesUnordered, provider::DataProvider,
};

/// Per-call configuration that knows how to construct a per-query
/// [`DistancesUnordered`] visitor for a provider, and the [`Self::QueryComputer`] used
/// to score each element during the scan.
pub trait SearchStrategy<P, T>: Send + Sync
where
    P: DataProvider,
{
    /// The reference element shape on which [`Self::QueryComputer`] computes
    /// distances.  The visitor's [`DistancesUnordered::ElementRef`] is 
    /// constrained to equal this for every lifetime.
    type ElementRef<'a>;

    /// The concrete query-computer type.
    type QueryComputer: for<'a> PreprocessedDistanceFunction<Self::ElementRef<'a>, f32>
        + Send
        + Sync
        + 'static;

    /// The error type for [`Self::build_query_computer`].
    type QueryComputerError: StandardError;

    /// The visitor type produced by [`Self::create_visitor`]. Borrows from `self` and
    /// the provider, scans elements of shape [`Self::ElementRef`], and consumes a
    /// [`Self::QueryComputer`].
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

#[cfg(test)]
mod tests {
    use crate::{
        flat::test::provider::{self as flat_provider, Strategy},
        graph::test::synthetic::Grid,
    };

    use super::SearchStrategy;

    /// `create_visitor` produces independent visitors on successive calls, and a
    /// computer can be built from each.
    #[test]
    fn exercise_create_visitor() {
        let provider = flat_provider::Provider::grid(Grid::Two, 3);
        let context = flat_provider::Context::new();
        let strategy = Strategy::new();

        let v1 = strategy.create_visitor(&provider, &context).unwrap();
        let v2 = strategy.create_visitor(&provider, &context).unwrap();

        // The two visitors must occupy distinct stack slots — i.e. holding `v1` does
        // not preclude constructing `v2`.
        let _ = (&v1, &v2);

        // A query computer can be built independently of any visitor.
        let _c1 = strategy.build_query_computer(&[1.0_f32, 0.0][..]).unwrap();
        let _c2 = strategy.build_query_computer(&[0.0_f32, 1.0][..]).unwrap();
    }
}
