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

pub mod instrumentation;

pub mod aligned_file_reader;
pub use aligned_file_reader::AlignedFileReaderFactory;
#[cfg(any(feature = "virtual_storage", test))]
pub use aligned_file_reader::VirtualAlignedReaderFactory;

pub mod statistics;
pub use statistics::{get_mean_stats, get_percentile_stats, get_sum_stats, QueryStatistics};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify that key types are accessible
        let _ = core::any::type_name::<AlignedFileReaderFactory>();
        let _ = core::any::type_name::<QueryStatistics>();
    }
}
