/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
mod perf_logger;
pub use perf_logger::{BuildMergedVamanaIndexCheckpoint, DiskIndexBuildCheckpoint, PerfLogger};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify that key types are accessible
        let _ = core::any::type_name::<BuildMergedVamanaIndexCheckpoint>();
        let _ = core::any::type_name::<DiskIndexBuildCheckpoint>();
        let _ = core::any::type_name::<PerfLogger>();
    }
}
