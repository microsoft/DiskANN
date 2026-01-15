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
