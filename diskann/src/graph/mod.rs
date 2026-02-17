/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod search_output_buffer;
pub use search_output_buffer::{
    BufferState, IdDistance, IdDistanceAssociatedData, SearchOutputBuffer,
};

pub mod adjacencylist;
pub use adjacencylist::AdjacencyList;

pub mod config;
pub use config::Config;

pub mod index;
pub use index::DiskANNIndex;

mod start_point;
pub use start_point::{SampleableForStart, StartPointStrategy};

mod misc;
pub use misc::{ConsolidateKind, InplaceDeleteMethod};

#[cfg(feature = "experimental_diversity_search")]
pub use misc::DiverseSearchParams;

pub mod glue;
pub mod search;

// Re-export unified search interface as the primary API.
pub use search::{
    Knn, KnnSearchError, MultihopSearch, Range, RangeSearchError, RangeSearchOutput,
    RecordedKnn, Search,
};

#[cfg(feature = "experimental_diversity_search")]
pub use search::Diverse;

mod internal;

// Integration tests and test providers.
#[cfg(any(test, feature = "testing"))]
pub mod test;
