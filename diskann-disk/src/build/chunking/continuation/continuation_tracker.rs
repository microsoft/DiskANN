/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::time::Duration;

// ContinuationGrant struct used for ochestrating chunkable operations.
pub enum ContinuationGrant {
    Continue,
    Yield(Duration),
    Stop,
}

// Contituation grant trait to be implemented by the client.
// This trait is used to get a continuation grant during chunk intervals while doing chunkable operations.
pub trait ContinuationTrackerTrait: Send + Sync + ContinuationCheckerTraitClone {
    fn get_continuation_grant(&self) -> ContinuationGrant;
}

// Naive implementation of the ContinuationGrantProviderTrait
// This implementation always returns ContinuationGrant::Continue.
#[derive(Default, Clone)]
pub struct NaiveContinuationTracker {}

impl ContinuationTrackerTrait for NaiveContinuationTracker {
    fn get_continuation_grant(&self) -> ContinuationGrant {
        ContinuationGrant::Continue
    }
}

pub trait ContinuationCheckerTraitClone {
    fn clone_box(&self) -> Box<dyn ContinuationTrackerTrait>;
}

impl<T> ContinuationCheckerTraitClone for T
where
    T: 'static + ContinuationTrackerTrait + Clone,
{
    fn clone_box(&self) -> Box<dyn ContinuationTrackerTrait> {
        Box::new(self.clone())
    }
}
