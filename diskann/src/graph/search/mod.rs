/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Unified search execution framework.
//!
//! This module provides the primary search interface for DiskANN. All search types
//! are represented as parameter structs that implement [`Search`], which
//! contains the complete search logic.
//!
//! # Usage
//!
//! ```ignore
//! use diskann::graph::{
//!     neighbor::{BackInserter, Neighbor},
//!     search::{Knn, Range, MultihopFilterSearch},
//!     Search,
//! };
//!
//! // Standard k-NN search: use a fixed-capacity output buffer
//! let params = Knn::new(10, 100, None)?;
//! let mut knn_storage = [Neighbor::default(); 10];
//! let mut knn_output = BackInserter::new(&mut knn_storage);
//! let stats = index
//!     .search(params, &strategy, &context, &query, &mut knn_output)
//!     .await?;
//!
//! // Range search: use a growable Vec buffer for an unknown number of results
//! let params = Range::new(100, 0.5)?;
//! let mut range_output: Vec<Neighbor<_>> = Vec::new();
//! let stats = index
//!     .search(params, &strategy, &context, &query, &mut range_output)
//!     .await?;
//! ```

use diskann_utils::future::SendFuture;

use crate::{
    ANNResult,
    graph::{
        self,
        glue::{FilteredAccessor, SearchAccessor},
    },
};

mod filtered_range_search;
mod inline_filter_search;
mod knn_search;
pub(crate) mod multihop_filter_search;
mod range_search;

mod paged;
pub use paged::PagedSearch;

pub mod record;
pub(crate) mod scratch;

pub trait Search {
    type Output;

    fn search<A, O, PP, OB>(
        self,
        accessor: &mut A,
        processor: PP,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        A: SearchAccessor,
        O: Send,
        PP: graph::glue::SearchPostProcess<A, O> + Send + Sync,
        OB: graph::search_output_buffer::SearchOutputBuffer<O> + Send + ?Sized;
}

pub trait SearchFor<I> {
    type Output;

    fn search_for<A, O, PP, OB>(
        self,
        accessor: &mut A,
        processor: PP,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        A: SearchAccessor<Id = I>,
        O: Send,
        PP: graph::glue::SearchPostProcess<A, O> + Send + Sync,
        OB: graph::search_output_buffer::SearchOutputBuffer<O> + Send + ?Sized;
}

impl<I, T> SearchFor<I> for T
where
    T: Search,
{
    type Output = T::Output;

    fn search_for<A, O, PP, OB>(
        self,
        accessor: &mut A,
        processor: PP,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        A: SearchAccessor<Id = I>,
        O: Send,
        PP: graph::glue::SearchPostProcess<A, O> + Send + Sync,
        OB: graph::search_output_buffer::SearchOutputBuffer<O> + Send + ?Sized,
    {
        self.search(accessor, processor, output)
    }
}

pub trait FilteredSearch {
    type Output;

    fn filtered_search<A, O, PP, OB>(
        self,
        accessor: &mut A,
        processor: PP,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        A: FilteredAccessor,
        O: Send,
        PP: graph::glue::SearchPostProcess<A, O> + Send + Sync,
        OB: graph::search_output_buffer::SearchOutputBuffer<O> + Send + ?Sized;
}

pub use filtered_range_search::FilteredRange;
pub use inline_filter_search::{AdaptiveL, InlineFilterSearch};
pub use knn_search::{Knn, KnnSearchError};
pub use multihop_filter_search::MultihopFilterSearch;
pub use range_search::{Range, RangeBuilder, RangeSearchError};

// Feature-gated diverse search.
#[cfg(feature = "experimental_diversity_search")]
mod diverse_search;

#[cfg(feature = "experimental_diversity_search")]
pub use diverse_search::Diverse;

#[cfg(feature = "experimental_diversity_search")]
pub use diverse_search::DiverseSearchParams;
