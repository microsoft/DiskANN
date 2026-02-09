/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Pipelined search module implementing the PipeANN algorithm.
//!
//! This module provides a pipelined disk search that overlaps IO and compute
//! within a single query, using io_uring for non-blocking IO on Linux.

#[cfg(target_os = "linux")]
mod pipelined_reader;
#[cfg(target_os = "linux")]
pub use pipelined_reader::PipelinedReader;

#[cfg(target_os = "linux")]
mod pipelined_search;

#[cfg(target_os = "linux")]
mod pipelined_searcher;
#[cfg(target_os = "linux")]
pub use pipelined_searcher::PipelinedSearcher;
