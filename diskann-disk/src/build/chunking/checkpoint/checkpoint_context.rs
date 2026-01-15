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
