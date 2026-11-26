/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Unified search execution framework.
//!
//! This module provides the primary search interface for DiskANN. All search types
//! are represented as parameter structs that implement [`Search`] or [`FilteredSearch`],
//! which contain the complete search logic.

use diskann_utils::future::SendFuture;

use crate::{
    ANNResult,
    graph::{
        SearchOutputBuffer,
        glue::{FilteredAccessor, SearchAccessor, SearchPostProcess},
    },
};

mod filtered_range_search;
mod inline_filter_search;
mod knn_search;
pub(crate) mod multihop_filter_search;
mod range_search;

mod paged;
pub use paged::PagedSearch;

pub use filtered_range_search::FilteredRange;
pub use inline_filter_search::{AdaptiveL, InlineFilterSearch};
pub use knn_search::{Knn, KnnSearchError, RecordedKnn};
pub use multihop_filter_search::MultihopFilterSearch;
pub use range_search::{Range, RangeBuilder, RangeSearchError};

// Feature-gated diverse search.
#[cfg(feature = "experimental_diversity_search")]
mod diverse_search;

#[cfg(feature = "experimental_diversity_search")]
pub use diverse_search::{Diverse, DiverseSearchParams};

pub mod record;
pub(crate) mod scratch;

/// Perform a search using a [`SearchAccessor`], passing the result to `output` after
/// post-processing.
///
/// Implementations provided by `diskann` return [`Poll::Pending`](std::task::Poll::Pending)
/// only when an accessor or postprocessor operations returns `Poll::Pending`.
///
/// # Error Handling
///
/// Any encountered critical error should be returned. The state of the [`SearchOutputBuffer`]
/// is unspecified if an error is returned.
pub trait Search<I> {
    /// The result of search - typically summary statistics.
    type Output;

    /// Run search.
    fn search<A, O, PP, OB>(
        self,
        accessor: &mut A,
        processor: PP,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        A: SearchAccessor<Id = I>,
        O: Send,
        PP: SearchPostProcess<A, O> + Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized;
}

/// Perform a filtered search using a [`FilteredAccessor`], passing the result to `output`
/// after post-processing. Implementations should ensure that only candidates yielded as
/// [`crate::graph::glue::Accept`] are passed to the postprocessor.
///
/// Implementations provided by `diskann` return [`Poll::Pending`](std::task::Poll::Pending)
/// only when an accessor or postprocessor operations returns `Poll::Pending`.
///
/// # Error Handling
///
/// Any encountered critical error should be returned. The state of the [`SearchOutputBuffer`]
/// is unspecified if an error is returned.
pub trait FilteredSearch<I> {
    /// The result of search - typically summary statistics.
    type Output;

    /// Run filtered search.
    fn filtered_search<A, O, PP, OB>(
        self,
        accessor: &mut A,
        processor: PP,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        A: FilteredAccessor<Id = I>,
        O: Send,
        PP: SearchPostProcess<A, O> + Send,
        OB: SearchOutputBuffer<O> + Send + ?Sized;
}
