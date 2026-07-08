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

    #[test]
    fn test_has_completed_false_when_no_file() -> ANNResult<()> {
        let temp_dir = tempdir()?;
        let index_prefix = temp_dir
            .path()
            .join("nonexistent_index")
            .to_str()
            .unwrap()
            .to_string();
        let manager = CheckpointRecordManagerWithFileStorage::new(&index_prefix, 999);
        assert!(!manager.has_completed()?);
        Ok(())
    }

    #[test]
    fn test_mark_as_invalid() -> ANNResult<()> {
        let temp_dir = tempdir()?;
        let index_prefix = temp_dir
            .path()
            .join("test_invalid")
            .to_str()
            .unwrap()
            .to_string();
        let identifier = 77;

        let mut manager = CheckpointRecordManagerWithFileStorage::new(&index_prefix, identifier);
        // Advance to a later stage with some progress
        manager.update(Progress::Completed, WorkStage::QuantizeFPV)?;
        manager.update(Progress::Processed(42), WorkStage::InMemIndexBuild)?;

        // Verify we can resume from progress=42
        let manager2 = CheckpointRecordManagerWithFileStorage::new(&index_prefix, identifier);
        assert_eq!(
            manager2.get_resumption_point(WorkStage::QuantizeFPV)?,
            Some(42)
        );

        // Mark as invalid - progress resets to 0 (is_valid=false => progress read as 0)
        let mut manager3 = CheckpointRecordManagerWithFileStorage::new(&index_prefix, identifier);
        manager3.mark_as_invalid()?;
        assert_eq!(
            manager3.get_resumption_point(WorkStage::QuantizeFPV)?,
            Some(0)
        );

        Ok(())
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

        Ok(())
    }
}
