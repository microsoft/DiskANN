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
