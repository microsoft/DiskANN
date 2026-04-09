/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk-specific utilities and helper functions.
//!
//! This module contains utility functions and helpers
//! specific to disk index operations.

pub mod partition;
pub use partition::partition_with_ram_budget;

pub mod kmeans;
pub use kmeans::{compute_vecs_l2sq, k_means_clustering, k_meanspp_selecting_pivots, run_lloyds};

pub mod instrumentation;

pub mod aligned_file_reader;
pub use aligned_file_reader::AlignedFileReaderFactory;
#[cfg(any(feature = "virtual_storage", test))]
pub use aligned_file_reader::VirtualAlignedReaderFactory;

pub mod statistics;
pub use statistics::{get_mean_stats, get_percentile_stats, get_sum_stats, QueryStatistics};
