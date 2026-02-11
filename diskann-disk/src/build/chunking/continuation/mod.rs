/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod chunking_config;
pub use chunking_config::ChunkingConfig;

mod continuation_tracker;
pub use continuation_tracker::{
    ContinuationCheckerTraitClone, ContinuationGrant, ContinuationTrackerTrait,
    NaiveContinuationTracker,
};

pub mod utils;
pub use utils::{process_while_resource_is_available, process_while_resource_is_available_async};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify that key types are accessible
        let _ = core::any::type_name::<ChunkingConfig>();
        let _ = core::any::type_name::<ContinuationGrant>();
        let _ = core::any::type_name::<NaiveContinuationTracker>();
    }
}
