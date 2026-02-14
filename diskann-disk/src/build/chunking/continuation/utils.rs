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
    use super::*;
    use super::super::continuation_tracker::NaiveContinuationTracker;
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
            Progress::Completed => assert!(true),
            _ => panic!("Expected Completed"),
        }
    }

    #[tokio::test]
    async fn test_process_while_resource_is_available_async_completes() {
        let checker = Box::new(NaiveContinuationTracker::default());
        let items = vec![1, 2, 3];
        let mut processed = Vec::new();
        
        let result = process_while_resource_is_available_async(
            |item| {
                processed.push(item);
                async { Ok::<(), TestError>(()) }
            },
            items.into_iter(),
            checker,
        ).await;
        
        assert!(result.is_ok());
        match result.unwrap() {
            Progress::Completed => assert_eq!(processed, vec![1, 2, 3]),
            _ => panic!("Expected Completed"),
        }
    }
}
