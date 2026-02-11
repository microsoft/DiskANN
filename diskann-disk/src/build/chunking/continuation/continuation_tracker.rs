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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuation_grant_variants() {
        // Test enum variants can be created
        let _continue = ContinuationGrant::Continue;
        let _yield = ContinuationGrant::Yield(Duration::from_millis(100));
        let _stop = ContinuationGrant::Stop;
    }

    #[test]
    fn test_naive_continuation_tracker_default() {
        let tracker = NaiveContinuationTracker::default();
        // Verify it always returns Continue
        match tracker.get_continuation_grant() {
            ContinuationGrant::Continue => assert!(true),
            _ => panic!("Expected Continue"),
        }
    }

    #[test]
    fn test_naive_continuation_tracker_clone() {
        let tracker = NaiveContinuationTracker::default();
        let cloned = tracker.clone();
        
        // Both should return Continue
        match cloned.get_continuation_grant() {
            ContinuationGrant::Continue => assert!(true),
            _ => panic!("Expected Continue"),
        }
    }

    #[test]
    fn test_naive_continuation_tracker_clone_box() {
        let tracker = NaiveContinuationTracker::default();
        let boxed = tracker.clone_box();
        
        // The boxed version should also return Continue
        match boxed.get_continuation_grant() {
            ContinuationGrant::Continue => assert!(true),
            _ => panic!("Expected Continue"),
        }
    }

    #[test]
    fn test_continuation_tracker_trait_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NaiveContinuationTracker>();
    }
}
