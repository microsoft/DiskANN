/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk-specific storage operations.
//!
//! This module contains storage utilities and types
//! specific to disk index operations.

pub mod disk_index_reader;

mod disk_index_writer;
pub use disk_index_writer::DiskIndexWriter;

mod cached_reader;
pub use cached_reader::CachedReader;

mod cached_writer;
pub use cached_writer::CachedWriter;

pub mod quant;

pub mod api;
