/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::{self, File},
    io::Write,
};

use diskann::{ANNError, ANNResult};
use diskann_providers::{storage::FileStorageProvider, utils::file_exists};

use super::{CheckpointManager, CheckpointRecord, Progress, WorkStage};

// A specific implementation of CheckpointRecordManagerTrait that uses file storage to save the checkpoint record.
// Library users can have their own implementation of CheckpointRecordManagerTrait.
// This implementation does not support concurrent access to the checkpoint record file.
#[derive(Clone)]
pub struct CheckpointRecordManagerWithFileStorage {
    checkpoint_record_identifier: String,
}

impl CheckpointRecordManagerWithFileStorage {
    // Create a new CheckpointRecordManagerWithFileStorage with the given index prefix and an identifier.
    // The identifer can be generated from the hash of the index build parameters
    // to ensure we don't resume from a checkpoint that is not compatible
    // with the current index build parameters set.
    pub fn new(index_prefix: &str, identifier: u64) -> Self {
        Self {
            checkpoint_record_identifier: format!("{index_prefix}_{identifier}.checkpoint"),
        }
    }

    fn get_checkpoint_record(&self) -> ANNResult<CheckpointRecord> {
        if !file_exists(&FileStorageProvider, &self.checkpoint_record_identifier) {
            Ok(CheckpointRecord::new())
        } else {
            let buffer = fs::read(&self.checkpoint_record_identifier)?;
            Ok(bincode::deserialize(&buffer).map_err(|e| {
                ANNError::log_serde_error(
                    "Error deserializing checkpoint record data.".to_string(),
                    *e,
                )
            })?)
        }
    }

    fn set_checkpoint_record(&mut self, checkpoint_record: CheckpointRecord) -> ANNResult<()> {
        let serialized = bincode::serialize(&checkpoint_record).map_err(|e| {
            ANNError::log_serde_error("Error serializing checkpoint record data.".to_string(), *e)
        })?;

        if file_exists(&FileStorageProvider, &self.checkpoint_record_identifier) {
            fs::remove_file(&self.checkpoint_record_identifier)?;
        }

        let mut writer = File::create(&self.checkpoint_record_identifier)?;
        writer.write_all(&serialized)?;
        Ok(())
    }

    // This method is used for testing purposes only.
    pub fn has_completed(&self) -> ANNResult<bool> {
        Ok(self.get_checkpoint_record()?.get_work_stage() == WorkStage::End)
    }
}

impl CheckpointManager for CheckpointRecordManagerWithFileStorage {
    fn get_resumption_point(&self, stage: WorkStage) -> ANNResult<Option<usize>> {
        Ok(self.get_checkpoint_record()?.get_resumption_point(stage))
    }

    fn mark_as_invalid(&mut self) -> ANNResult<()> {
        self.set_checkpoint_record(self.get_checkpoint_record()?.mark_as_invalid())
    }

    fn update(&mut self, progress: Progress, next_stage: super::WorkStage) -> ANNResult<()> {
        match progress {
            Progress::Completed => self.set_checkpoint_record(
                self.get_checkpoint_record()?
                    .advance_work_type(next_stage)?,
            ),
            Progress::Processed(progress) => {
                self.set_checkpoint_record(self.get_checkpoint_record()?.update_progress(progress))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::build::chunking::checkpoint::checkpoint_record_manager::CheckpointManagerExt;

    // Helper function to clean up checkpoint files after tests
    fn clean_checkpoint_file(prefix: &str, identifier: u64) {
        let checkpoint_file = format!("{prefix}_{identifier}.checkpoint");
        if std::path::Path::new(&checkpoint_file).exists() {
            // There is a possible race between checking that the file exists and removing
            // the file here, but since this is test code, that is unlikely.
            fs::remove_file(&checkpoint_file).unwrap();
        }
    }

    #[test]
    fn test_checkpoint_manager_interruption_and_resumption() -> ANNResult<()> {
        let temp_dir = tempdir()?;
        let index_prefix = temp_dir
            .path()
            .join("test_checkpoint")
            .to_str()
            .unwrap()
            .to_string();
        let identifier = 42;

        // Clean up any existing files
        clean_checkpoint_file(&index_prefix, identifier);

        // Define a helper function to process a stage with interruption and resumption
        fn process_stage(
            index_prefix: &str,
            identifier: u64,
            current_stage: WorkStage,
            next_stage: WorkStage,
            progress: usize,
        ) -> ANNResult<()> {
            // Start and interrupt the current stage
            let mut manager = CheckpointRecordManagerWithFileStorage::new(index_prefix, identifier);
            assert_eq!(manager.get_resumption_point(current_stage)?, Some(0));
            manager.update(Progress::Processed(progress), next_stage)?;

            // Simulate a restart
            let mut manager = CheckpointRecordManagerWithFileStorage::new(index_prefix, identifier);
            assert_eq!(manager.get_resumption_point(current_stage)?, Some(progress));

            // Resume and complete the current stage
            manager.update(Progress::Completed, next_stage)?;

            Ok(())
        }

        // Define the workflow stages and their progress values
        let stages = [
            (WorkStage::Start, 0),
            (WorkStage::TrainBuildQuantizer, 0),
            (WorkStage::QuantizeFPV, 10),
            (WorkStage::InMemIndexBuild, 20),
            (WorkStage::PartitionData, 30),
            (WorkStage::BuildIndicesOnShards(0), 50),
            (WorkStage::BuildIndicesOnShards(1), 50),
            (WorkStage::BuildIndicesOnShards(2), 50),
            (WorkStage::MergeIndices, 50),
            (WorkStage::WriteDiskLayout, 50),
        ];

        let next_stages = [
            WorkStage::TrainBuildQuantizer,
            WorkStage::QuantizeFPV,
            WorkStage::InMemIndexBuild,
            WorkStage::PartitionData,
            WorkStage::BuildIndicesOnShards(0),
            WorkStage::BuildIndicesOnShards(1),
            WorkStage::BuildIndicesOnShards(2),
            WorkStage::MergeIndices,
            WorkStage::WriteDiskLayout,
            WorkStage::End,
        ];

        for (stage, next_stage) in stages.iter().zip(next_stages.iter()) {
            process_stage(&index_prefix, identifier, stage.0, *next_stage, stage.1)?;
        }

        // Verify workflow is complete
        let manager = CheckpointRecordManagerWithFileStorage::new(&index_prefix, identifier);
        assert!(manager.has_completed()?);

        // Clean up test files
        clean_checkpoint_file(&index_prefix, identifier);

        Ok(())
    }

    #[test]
    fn test_mark_as_invalid_persists_and_returns_zero_progress_on_resume() -> ANNResult<()> {
        // mark_as_invalid must be durably written so that after a restart the same
        // stage returns progress 0 (forcing a restart from the beginning of that stage).
        let temp_dir = tempdir()?;
        let index_prefix = temp_dir
            .path()
            .join("test_invalid")
            .to_str()
            .unwrap()
            .to_string();
        let identifier = 99;

        clean_checkpoint_file(&index_prefix, identifier);

        // Record partial progress on Start.
        let mut manager = CheckpointRecordManagerWithFileStorage::new(&index_prefix, identifier);
        manager.update(Progress::Processed(50), WorkStage::TrainBuildQuantizer)?;
        assert_eq!(manager.get_resumption_point(WorkStage::Start)?, Some(50));

        // Mark the stage as invalid (e.g. crash during work).
        manager.mark_as_invalid()?;

        // After a simulated restart the resumption point for the same stage must be 0,
        // not 50, so that the stage is retried from the beginning.
        let restarted = CheckpointRecordManagerWithFileStorage::new(&index_prefix, identifier);
        assert_eq!(
            restarted.get_resumption_point(WorkStage::Start)?,
            Some(0),
            "invalid checkpoint must restart the stage from offset 0"
        );

        clean_checkpoint_file(&index_prefix, identifier);
        Ok(())
    }

    #[test]
    fn test_execute_stage_skips_already_completed_stage() -> ANNResult<()> {
        // When the checkpoint has already moved past a stage, execute_stage must
        // invoke skip_handler rather than operation for that stage.
        let temp_dir = tempdir()?;
        let index_prefix = temp_dir
            .path()
            .join("test_skip")
            .to_str()
            .unwrap()
            .to_string();
        let identifier = 100;

        clean_checkpoint_file(&index_prefix, identifier);

        // Advance past Start -> checkpoint is now at QuantizeFPV.
        let mut manager = CheckpointRecordManagerWithFileStorage::new(&index_prefix, identifier);
        manager.update(Progress::Completed, WorkStage::QuantizeFPV)?;

        // Querying the old stage (Start) must now return None.
        assert_eq!(manager.get_resumption_point(WorkStage::Start)?, None);

        let mut operation_called = false;
        let mut skip_called = false;

        let result = manager.execute_stage(
            WorkStage::Start,
            WorkStage::QuantizeFPV,
            || {
                operation_called = true;
                Ok(1_i32)
            },
            || {
                skip_called = true;
                Ok(0_i32)
            },
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
        assert!(
            !operation_called,
            "operation must NOT be called for a stage that has already been completed"
        );
        assert!(
            skip_called,
            "skip_handler must be called for a stage that has already been completed"
        );

        clean_checkpoint_file(&index_prefix, identifier);
        Ok(())
    }
}
