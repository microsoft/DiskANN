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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::Progress;

    #[test]
    fn test_naive_checkpoint_record_manager_default() {
        let manager = NaiveCheckpointRecordManager::default();
        // Test get_resumption_point always returns Some(0)
        let result = manager.get_resumption_point(WorkStage::Start);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(0));
    }

    #[test]
    fn test_naive_checkpoint_record_manager_get_resumption_point() {
        let manager = NaiveCheckpointRecordManager::default();
        
        // Test with various stages
        for stage in [WorkStage::Start, WorkStage::End, WorkStage::QuantizeFPV] {
            let result = manager.get_resumption_point(stage);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), Some(0));
        }
    }

    #[test]
    fn test_naive_checkpoint_record_manager_update() {
        let mut manager = NaiveCheckpointRecordManager::default();
        
        // Update should always succeed
        let result = manager.update(Progress::Completed, WorkStage::End);
        assert!(result.is_ok());
        
        let result = manager.update(Progress::Processed(100), WorkStage::InMemIndexBuild);
        assert!(result.is_ok());
    }

    #[test]
    fn test_naive_checkpoint_record_manager_mark_as_invalid() {
        let mut manager = NaiveCheckpointRecordManager::default();
        
        // mark_as_invalid should always succeed
        let result = manager.mark_as_invalid();
        assert!(result.is_ok());
    }

    #[test]
    fn test_naive_checkpoint_record_manager_clone() {
        let manager = NaiveCheckpointRecordManager::default();
        let cloned = manager.clone();
        
        // Both should behave the same
        let result1 = manager.get_resumption_point(WorkStage::Start);
        let result2 = cloned.get_resumption_point(WorkStage::Start);
        
        assert_eq!(result1.unwrap(), result2.unwrap());
    }
}
