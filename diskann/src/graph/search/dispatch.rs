/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Core search dispatch trait.

use diskann_utils::future::SendFuture;

use crate::{ANNResult, graph::index::DiskANNIndex, provider::DataProvider};

/// Trait for search parameter types that execute their own search logic.
///
/// Each search type (graph search, range search, etc.) implements
/// this trait to define its complete search behavior. The [`DiskANNIndex::search`]
/// method delegates to the `dispatch` method.
pub trait SearchDispatch<DP, S, T: ?Sized, O, OB: ?Sized>
where
    DP: DataProvider,
{
    /// The result type returned by this search.
    type Output;

    /// Execute the search operation with full search logic.
    fn dispatch<'a>(
        &'a self,
        index: &'a DiskANNIndex<DP>,
        strategy: &'a S,
        context: &'a DP::Context,
        query: &'a T,
        output: &'a mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>;
}
