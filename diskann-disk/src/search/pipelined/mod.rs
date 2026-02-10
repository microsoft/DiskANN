/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Pipelined search module implementing the PipeANN algorithm.
//!
//! This module provides a pipelined disk search that overlaps IO and compute
//! within a single query, using io_uring for non-blocking IO on Linux.
//!
//! # Safety
//!
//! This search implementation is designed for **read-only search on completed
//! (static) disk indices**. It bypasses the synchronized `DiskProvider` path and
//! reads raw sectors directly via O_DIRECT, so it must NOT be used concurrently
//! with index modifications (build, insert, delete). For search during streaming
//! operations, use `DiskIndexSearcher` (beam search) instead.

#[cfg(target_os = "linux")]
mod pipelined_reader;
#[cfg(target_os = "linux")]
pub use pipelined_reader::PipelinedReader;
#[cfg(target_os = "linux")]
pub use pipelined_reader::PipelinedReaderConfig;
#[cfg(target_os = "linux")]
pub use pipelined_reader::MAX_IO_CONCURRENCY;

#[cfg(target_os = "linux")]
mod pipelined_search;

#[cfg(target_os = "linux")]
mod pipelined_searcher;
#[cfg(target_os = "linux")]
pub use pipelined_searcher::PipelinedSearcher;
