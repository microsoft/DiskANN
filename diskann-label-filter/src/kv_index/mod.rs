/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Generic inverted index implementation.
//!
//! This module provides a complete implementation of the InvertedIndex, PostingListProvider,
//! and QueryEvaluator traits that works with any KvStore backend.

mod error;
mod generic_index;
mod inverted_index_provider_impl;
mod posting_list_accessor_impl;
mod query_evaluator_impl;

// Re-export public types
pub use error::{IndexError, QueryError, Result};
pub use generic_index::GenericIndex;
