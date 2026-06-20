/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{error::Error, thread::sleep};

use tracing::info;

use super::continuation_tracker::{ContinuationGrant, ContinuationTrackerTrait};
use crate::build::chunking::checkpoint::Progress;

/// This takes an operation with an iterator of oprands,
/// and processes the oprands using the operation in a loop,
/// until the continuation_checker asks it to stop.
/// The continuation_checker is used to get continuation grants between processing each operation.
/// The clean_up function is called after the loop is broken and before exit.
/// The function returns a Progress enum, which indicates the number of operations executed.
pub fn process_while_resource_is_available<Action, ParamIter, Param, E>(
    mut action: Action,
    params: ParamIter,
    continuation_checker: Box<dyn ContinuationTrackerTrait>,
) -> Result<Progress, E>
where
    ParamIter: Iterator<Item = Param>,
    Action: FnMut(Param) -> Result<(), E>,
    E: Error,
{
    for (idx, param) in params.enumerate() {
        loop {
            match continuation_checker.get_continuation_grant() {
                ContinuationGrant::Continue => {
                    info!("Continue processing.");
                    action(param)?;
                    break;
                }
                ContinuationGrant::Yield(duration) => {
                    info!(
                        "Continuation checker asks to yield for {} ms.",
                        duration.as_millis()
                    );
                    sleep(duration);
                }
                ContinuationGrant::Stop => {
                    info!("Continuation checker asks to stop. Breaking the loop.");
                    return Ok(Progress::Processed(idx));
                }
            }
        }
    }

    Ok(Progress::Completed)
}

/// Asynchronous version of [`process_while_resource_is_available`].
///
/// Takes an async operation with an iterator of operands and processes them in a loop
/// until the continuation_checker signals to stop.
pub async fn process_while_resource_is_available_async<Action, ParamIter, Param, Fut, E>(
    mut action: Action,
    params: ParamIter,
    continuation_checker: Box<dyn ContinuationTrackerTrait>,
) -> Result<Progress, E>
where
    ParamIter: Iterator<Item = Param>,
    Action: FnMut(Param) -> Fut,
    Fut: core::future::Future<Output = Result<(), E>>,
    E: Error,
{
    for (idx, param) in params.enumerate() {
        loop {
            match continuation_checker.get_continuation_grant() {
                ContinuationGrant::Continue => {
                    info!("Continue processing.");
                    action(param).await?;
                    break;
                }
                ContinuationGrant::Yield(duration) => {
                    info!(
                        "Continuation checker asks to yield for {} ms.",
                        duration.as_millis()
                    );
                    sleep(duration);
                }
                ContinuationGrant::Stop => {
                    info!("Continuation checker asks to stop. Breaking the loop.");
                    return Ok(Progress::Processed(idx));
                }
            }
        }
    }

    Ok(Progress::Completed)
}

#[cfg(test)]
mod tests {
    use super::super::continuation_tracker::NaiveContinuationTracker;
    use super::*;
    use std::fmt;

    #[derive(Debug)]
    struct TestError;

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestError")
        }
    }

    impl Error for TestError {}

    #[test]
    fn test_process_while_resource_is_available_completes() {
        let checker = Box::new(NaiveContinuationTracker::default());
        let items = vec![1, 2, 3, 4, 5];
        let mut processed = Vec::new();

        let result = process_while_resource_is_available(
            |item| {
                processed.push(item);
                Ok::<(), TestError>(())
            },
            items.into_iter(),
            checker,
        );

        assert!(result.is_ok());
        match result.unwrap() {
            Progress::Completed => assert_eq!(processed, vec![1, 2, 3, 4, 5]),
            _ => panic!("Expected Completed"),
        }
    }

    #[test]
    fn test_process_while_resource_is_available_empty_iter() {
        let checker = Box::new(NaiveContinuationTracker::default());
        let items: Vec<i32> = vec![];

        let result = process_while_resource_is_available(
            |_item| Ok::<(), TestError>(()),
            items.into_iter(),
            checker,
        );

        assert!(result.is_ok());
        match result.unwrap() {
            Progress::Completed => {}
            _ => panic!("Expected Completed"),
        }
    }

    /// A tracker that returns Stop after `stop_after` Continue grants.
    #[derive(Clone)]
    struct StopAfterTracker {
        count: std::sync::Arc<std::sync::Mutex<usize>>,
        stop_after: usize,
    }

    impl ContinuationTrackerTrait for StopAfterTracker {
        fn get_continuation_grant(&self) -> ContinuationGrant {
            let mut count = self.count.lock().unwrap();
            if *count >= self.stop_after {
                ContinuationGrant::Stop
            } else {
                *count += 1;
                ContinuationGrant::Continue
            }
        }
    }

    #[test]
    fn test_process_while_resource_is_available_stops_early() {
        let tracker = StopAfterTracker {
            count: std::sync::Arc::new(std::sync::Mutex::new(0)),
            stop_after: 3,
        };
        let items = vec![10, 20, 30, 40, 50];
        let mut processed = Vec::new();

        let result = process_while_resource_is_available(
            |item| {
                processed.push(item);
                Ok::<(), TestError>(())
            },
            items.into_iter(),
            Box::new(tracker),
        );

        assert!(result.is_ok());
        match result.unwrap() {
            Progress::Processed(idx) => {
                assert_eq!(idx, 3); // stopped before processing item at index 3
                assert_eq!(processed, vec![10, 20, 30]);
            }
            _ => panic!("Expected Processed"),
        }
    }

    /// A tracker that yields once (with a tiny duration), then continues.
    #[derive(Clone)]
    struct YieldOnceThenContinueTracker {
        yielded: std::sync::Arc<std::sync::Mutex<bool>>,
    }

    impl ContinuationTrackerTrait for YieldOnceThenContinueTracker {
        fn get_continuation_grant(&self) -> ContinuationGrant {
            let mut yielded = self.yielded.lock().unwrap();
            if !*yielded {
                *yielded = true;
                ContinuationGrant::Yield(std::time::Duration::from_millis(1))
            } else {
                // After yielding once, always continue
                ContinuationGrant::Continue
            }
        }
    }

    #[test]
    fn test_process_while_resource_is_available_yield_then_continue() {
        let tracker = YieldOnceThenContinueTracker {
            yielded: std::sync::Arc::new(std::sync::Mutex::new(false)),
        };
        let items = vec![1, 2];
        let mut processed = Vec::new();

        let result = process_while_resource_is_available(
            |item| {
                processed.push(item);
                Ok::<(), TestError>(())
            },
            items.into_iter(),
            Box::new(tracker),
        );

        assert!(result.is_ok());
        // After yielding, it should have continued and processed all items
        match result.unwrap() {
            Progress::Completed => assert_eq!(processed, vec![1, 2]),
            _ => panic!("Expected Completed"),
        }
    }

    #[test]
    fn test_process_while_resource_is_available_action_error() {
        let checker = Box::new(NaiveContinuationTracker::default());
        let items = vec![1, 2, 3];

        let result = process_while_resource_is_available(
            |item| {
                if item == 2 {
                    Err(TestError)
                } else {
                    Ok(())
                }
            },
            items.into_iter(),
            checker,
        );

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_while_resource_is_available_async_stops_early() {
        let tracker = StopAfterTracker {
            count: std::sync::Arc::new(std::sync::Mutex::new(0)),
            stop_after: 2,
        };
        let items = vec![1, 2, 3, 4, 5];
        let processed = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));

        let result = process_while_resource_is_available_async(
            |item| {
                let processed = processed.clone();
                async move {
                    processed.lock().await.push(item);
                    Ok::<(), TestError>(())
                }
            },
            items.into_iter(),
            Box::new(tracker),
        )
        .await;

        assert!(result.is_ok());
        match result.unwrap() {
            Progress::Processed(idx) => {
                assert_eq!(idx, 2);
                let processed = processed.lock().await;
                assert_eq!(*processed, vec![1, 2]);
            }
            _ => panic!("Expected Processed"),
        }
    }

    #[tokio::test]
    async fn test_process_while_resource_is_available_async_completes() {
        let checker = Box::new(NaiveContinuationTracker::default());
        let items = vec![1, 2, 3];
        let processed = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));

        let result = process_while_resource_is_available_async(
            |item| {
                let processed = processed.clone();
                async move {
                    processed.lock().await.push(item);
                    Ok::<(), TestError>(())
                }
            },
            items.into_iter(),
            checker,
        )
        .await;

        assert!(result.is_ok());
        match result.unwrap() {
            Progress::Completed => {
                let processed = processed.lock().await;
                assert_eq!(*processed, vec![1, 2, 3]);
            }
            _ => panic!("Expected Completed"),
        }
    }
}
