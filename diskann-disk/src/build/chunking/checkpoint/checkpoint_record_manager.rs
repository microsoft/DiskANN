/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use tracing::info;

use super::{Progress, WorkStage};

/// This trait provides functionalities to get and set checkpoint records
/// ..for tracking the progress and state in a chunkable index build process.
/// it needs to be marked as send and sync because it will be used in a multi-threaded environment.
/// However, during the index build process DiskANN will not have parallel requests trying to access the checkpoint record.
pub trait CheckpointManager: Send + Sync + CheckpointManagerClone {
    /// Gets the resumption point for a given work stage.
    ///
    /// Returns the offset where processing should resume for the specified stage.
    /// Returns None if:
    /// - No checkpoint exists for the stage
    /// - The stage has already completed
    fn get_resumption_point(&self, stage: WorkStage) -> ANNResult<Option<usize>>;

    /// Updates the checkpoint record with progress information
    ///
    /// # Arguments
    ///
    /// * `progress` - The current progress (Completed or Processed amount)
    /// * `next_stage` - If provided and progress is Completed, advances to this stage.
    fn update(&mut self, progress: Progress, next_stage: WorkStage) -> ANNResult<()>;

    /// Marks the checkpoint as invalid for current stage.
    ///
    /// When a checkpoint is marked as invalid:
    /// - Future calls to get_resumption_point(curent_stage) will return offset 0
    /// - This forces processing to restart from the beginning of the stage
    /// - Protects against partial/incomplete work if a crash occurs
    fn mark_as_invalid(&mut self) -> ANNResult<()>;
}

pub trait CheckpointManagerExt {
    fn execute_stage<F, S, U>(
        &mut self,
        stage: WorkStage,
        next_stage: WorkStage,
        operation: F,
        skip_handler: S,
    ) -> ANNResult<U>
    where
        F: FnOnce() -> ANNResult<U>,
        S: FnOnce() -> ANNResult<U>;
}

impl<T: ?Sized> CheckpointManagerExt for T
where
    T: CheckpointManager,
{
    fn execute_stage<F, S, U>(
        &mut self,
        stage: WorkStage,
        next_stage: WorkStage,
        operation: F,
        skip_handler: S,
    ) -> ANNResult<U>
    where
        F: FnOnce() -> ANNResult<U>,
        S: FnOnce() -> ANNResult<U>,
    {
        match self.get_resumption_point(stage)? {
            Some(_) => {
                let result = operation()?;
                self.update(Progress::Completed, next_stage)?;
                Ok(result)
            }
            None => {
                info!("[Stage:{:?}] Skip stage - invalid checkpoint", stage);
                skip_handler()
            }
        }
    }
}

/// This trait is used to clone the Box<dyn CheckpointRecordManager>
pub trait CheckpointManagerClone {
    fn clone_box(&self) -> Box<dyn CheckpointManager>;
}

impl<T> CheckpointManagerClone for T
where
    T: 'static + CheckpointManager + Clone,
{
    fn clone_box(&self) -> Box<dyn CheckpointManager> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::super::NaiveCheckpointRecordManager;
    use super::*;

    /// A test-only manager that always skips every stage (returns None for every
    /// resumption-point query), simulating a checkpoint that has already advanced
    /// past every queried stage.
    #[derive(Default, Clone)]
    struct AlwaysSkipManager;

    impl CheckpointManager for AlwaysSkipManager {
        fn get_resumption_point(&self, _stage: WorkStage) -> ANNResult<Option<usize>> {
            Ok(None)
        }

        fn update(&mut self, _progress: Progress, _next_stage: WorkStage) -> ANNResult<()> {
            Ok(())
        }

        fn mark_as_invalid(&mut self) -> ANNResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_checkpoint_manager_ext_execute_stage_with_resumption() {
        let mut manager = NaiveCheckpointRecordManager;
        let mut executed = false;

        let result = manager.execute_stage(
            WorkStage::Start,
            WorkStage::End,
            || {
                executed = true;
                Ok(42)
            },
            || Ok(0),
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert!(executed);
    }

    #[test]
    fn test_checkpoint_manager_ext_execute_stage_skip_when_stage_already_done() {
        // When the checkpoint has already advanced past a stage, execute_stage must
        // call skip_handler instead of operation.
        let mut manager = AlwaysSkipManager;
        let mut operation_called = false;
        let mut skip_called = false;

        let result = manager.execute_stage(
            WorkStage::Start,
            WorkStage::QuantizeFPV,
            || {
                operation_called = true;
                Ok(1)
            },
            || {
                skip_called = true;
                Ok(0)
            },
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0, "skip_handler return value should be used");
        assert!(!operation_called, "operation should NOT be called when stage is already done");
        assert!(skip_called, "skip_handler should be called when stage is already done");
    }

    #[test]
    fn test_checkpoint_manager_ext_execute_stage_operation_failure_does_not_advance() {
        // When the operation callback returns an error, execute_stage must propagate
        // the error without calling update (i.e. the stage must not be advanced).
        let mut manager = NaiveCheckpointRecordManager;
        let mut skip_called = false;

        let result: ANNResult<i32> = manager.execute_stage(
            WorkStage::Start,
            WorkStage::QuantizeFPV,
            || Err(diskann::ANNError::log_index_error("simulated failure".to_string())),
            || {
                skip_called = true;
                Ok(-1)
            },
        );

        assert!(result.is_err(), "error from operation must be propagated");
        assert!(!skip_called, "skip_handler must not be called when operation fails");
    }

    #[test]
    fn test_checkpoint_manager_clone_box() {
        let manager = NaiveCheckpointRecordManager;
        let boxed = manager.clone_box();

        // The boxed version should work the same
        let result = boxed.get_resumption_point(WorkStage::Start);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(0));
    }
}
