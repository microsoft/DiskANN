/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use diskann::ANNResult;

use super::{CheckpointManager, Progress, WorkStage};

// Context for managing checkpoint operations during various processing stages.
pub struct CheckpointContext<'a> {
    checkpoint_manager: &'a dyn CheckpointManager,
    current_stage: WorkStage,
    next_stage: WorkStage,
}

impl<'a> CheckpointContext<'a> {
    pub fn new(
        checkpoint_manager: &'a dyn CheckpointManager,
        current_stage: WorkStage,
        next_stage: WorkStage,
    ) -> Self {
        Self {
            checkpoint_manager,
            current_stage,
            next_stage,
        }
    }

    pub fn current_stage(&self) -> WorkStage {
        self.current_stage
    }

    pub fn to_owned(&self) -> OwnedCheckpointContext {
        OwnedCheckpointContext::new(
            self.checkpoint_manager.clone_box(),
            self.current_stage,
            self.next_stage,
        )
    }

    pub fn get_resumption_point(&self) -> ANNResult<Option<usize>> {
        self.checkpoint_manager
            .get_resumption_point(self.current_stage)
    }
}

/// Context for managing checkpoint operations with an owned checkpoint manager
pub struct OwnedCheckpointContext {
    checkpoint_manager: Box<dyn CheckpointManager>,
    current_stage: WorkStage,
    next_stage: WorkStage,
}

impl OwnedCheckpointContext {
    pub fn new(
        checkpoint_manager: Box<dyn CheckpointManager>,
        current_stage: WorkStage,
        next_stage: WorkStage,
    ) -> Self {
        Self {
            checkpoint_manager,
            current_stage,
            next_stage,
        }
    }

    pub fn current_stage(&self) -> WorkStage {
        self.current_stage
    }

    pub fn get_resumption_point(&mut self) -> ANNResult<Option<usize>> {
        self.checkpoint_manager
            .get_resumption_point(self.current_stage)
    }

    pub fn update(&mut self, progress: Progress) -> ANNResult<()> {
        self.checkpoint_manager.update(progress, self.next_stage)
    }

    pub fn mark_as_invalid(&mut self) -> ANNResult<()> {
        self.checkpoint_manager.mark_as_invalid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::NaiveCheckpointRecordManager;

    #[test]
    fn test_checkpoint_context_new() {
        let manager = NaiveCheckpointRecordManager::default();
        let context = CheckpointContext::new(&manager, WorkStage::Start, WorkStage::End);
        
        assert_eq!(context.current_stage(), WorkStage::Start);
    }

    #[test]
    fn test_checkpoint_context_current_stage() {
        let manager = NaiveCheckpointRecordManager::default();
        let context = CheckpointContext::new(&manager, WorkStage::QuantizeFPV, WorkStage::InMemIndexBuild);
        
        assert_eq!(context.current_stage(), WorkStage::QuantizeFPV);
    }

    #[test]
    fn test_checkpoint_context_get_resumption_point() {
        let manager = NaiveCheckpointRecordManager::default();
        let context = CheckpointContext::new(&manager, WorkStage::Start, WorkStage::End);
        
        let result = context.get_resumption_point();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(0));
    }

    #[test]
    fn test_checkpoint_context_to_owned() {
        let manager = NaiveCheckpointRecordManager::default();
        let context = CheckpointContext::new(&manager, WorkStage::Start, WorkStage::End);
        
        let owned = context.to_owned();
        assert_eq!(owned.current_stage(), WorkStage::Start);
    }

    #[test]
    fn test_owned_checkpoint_context_new() {
        let manager = Box::new(NaiveCheckpointRecordManager::default());
        let context = OwnedCheckpointContext::new(manager, WorkStage::TrainBuildQuantizer, WorkStage::PartitionData);
        
        assert_eq!(context.current_stage(), WorkStage::TrainBuildQuantizer);
    }

    #[test]
    fn test_owned_checkpoint_context_update() {
        let manager = Box::new(NaiveCheckpointRecordManager::default());
        let mut context = OwnedCheckpointContext::new(manager, WorkStage::Start, WorkStage::End);
        
        let result = context.update(Progress::Completed);
        assert!(result.is_ok());
    }

    #[test]
    fn test_owned_checkpoint_context_mark_as_invalid() {
        let manager = Box::new(NaiveCheckpointRecordManager::default());
        let mut context = OwnedCheckpointContext::new(manager, WorkStage::Start, WorkStage::End);
        
        let result = context.mark_as_invalid();
        assert!(result.is_ok());
    }
}
