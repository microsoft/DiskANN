/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;

use super::{CheckpointManager, WorkStage};

/// A naive implementation of the CheckpointRecordManagerTrait
/// .. that always returns the default checkpoint record.
/// This implementation is used for keeping the backward compatibility
/// .. with the previous version of DiskANN for both DiskIndex and InMemoryIndex.
#[derive(Default, Clone)]
pub struct NaiveCheckpointRecordManager;

impl CheckpointManager for NaiveCheckpointRecordManager {
    fn get_resumption_point(&self, _stage: WorkStage) -> ANNResult<Option<usize>> {
        Ok(Some(0))
    }

    fn update(
        &mut self,
        _progress: super::Progress,
        _next_stage: super::WorkStage,
    ) -> ANNResult<()> {
        Ok(())
    }

    fn mark_as_invalid(&mut self) -> ANNResult<()> {
        Ok(())
    }
}
