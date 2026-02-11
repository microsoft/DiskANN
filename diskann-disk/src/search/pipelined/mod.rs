/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Pipelined IO reader for disk search using io_uring.
//!
//! Provides [`PipelinedReader`] for non-blocking sector reads with O_DIRECT,
//! used by [`PipelinedDiskAccessor`](super::provider::pipelined_accessor::PipelinedDiskAccessor)
//! to overlap IO and compute within a single query.

#[cfg(target_os = "linux")]
mod pipelined_reader;
#[cfg(target_os = "linux")]
pub use pipelined_reader::PipelinedReader;
#[cfg(target_os = "linux")]
pub use pipelined_reader::PipelinedReaderConfig;
#[cfg(target_os = "linux")]
pub use pipelined_reader::MAX_IO_CONCURRENCY;
