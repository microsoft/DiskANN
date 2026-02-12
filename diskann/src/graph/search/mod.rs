/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Unified search execution framework.
//!
//! This module provides the primary search interface for DiskANN. All search types
//! are represented as parameter structs that implement [`SearchDispatch`], which
//! contains the complete search logic.
//!
//! # Usage
//!
//! ```ignore
//! use diskann::graph::{GraphSearch, RangeSearch, MultihopSearch, SearchDispatch};
//!
//! // Standard graph search
//! let params = GraphSearch::new(10, 100, None)?;
//! let stats = index.search(&strategy, &context, &query, &params, &mut output).await?;
//!
//! // Range search
//! let params = RangeSearch::new(100, 0.5)?;
//! let result = index.search(&strategy, &context, &query, &params, &mut ()).await?;
//! println!("Found {} points within radius", result.ids.len());
//! ```

mod dispatch;
mod graph_search;
mod multihop_search;
mod range_search;

pub mod record;
pub(crate) mod scratch;

// Re-export the core dispatch trait.
pub use dispatch::SearchDispatch;

// Re-export search parameter types.
pub use graph_search::{GraphSearch, RecordedGraphSearch};
pub use multihop_search::MultihopSearch;
pub use range_search::{RangeSearch, RangeSearchOutput};

// Feature-gated diverse search.
#[cfg(feature = "experimental_diversity_search")]
mod diverse_search;

#[cfg(feature = "experimental_diversity_search")]
pub use diverse_search::DiverseSearch;
