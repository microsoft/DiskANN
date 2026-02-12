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
