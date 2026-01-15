/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use serde::{Deserialize, Serialize};
use tracing::info;

use super::WorkStage;

/// Represents a checkpoint record in the index build process.
/// The checkpoint record can be marked as in-valid to indicate that the exising intermediate data should be discarded.
/// This can happen because of a crash or an unexpected shutdown during the in-memory index build.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CheckpointRecord {
    /// The work type represents the current stage of the index build process.
    stage: WorkStage,

    /// Indicates if the checkpoint record is dirty.
    is_valid: bool,

    progress: usize,
}

impl Default for CheckpointRecord {
    fn default() -> Self {
        CheckpointRecord::new()
    }
}

impl CheckpointRecord {
    /// Create a new CheckpointRecord with the work type set to Start.
    pub fn new() -> CheckpointRecord {
        CheckpointRecord {
            stage: WorkStage::Start,
            is_valid: true,
            progress: 0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.is_valid
    }

    pub fn get_resumption_point(&self, stage: WorkStage) -> Option<usize> {
        if self.stage == stage {
            info!(
                "The resumption point is at {} for stage {:?}",
                self.progress, stage
            );
            Some(if self.is_valid { self.progress } else { 0 })
        } else {
            info!(
                "Failed to get resumption point for {:?} since the current stage is {:?}.",
                stage, self.stage
            );
            None
        }
    }

    // Advance the work type to the next stage in the index build process.
    // This method is used in each individual step of the index build process
    // ..t o update the checkpoint record.
    pub fn advance_work_type(&self, next_stage: WorkStage) -> ANNResult<CheckpointRecord> {
        info!(
            "Advancing work type from {:?} to {:?}.",
            self.stage, next_stage
        );
        Ok(CheckpointRecord {
            stage: next_stage,
            is_valid: true,
            progress: 0,
        })
    }

    // Mark the checkpoint record as invalid.
    pub fn mark_as_invalid(&self) -> CheckpointRecord {
        CheckpointRecord {
            stage: self.stage,
            is_valid: false,
            progress: self.progress,
        }
    }

    // Update the progress of the current work type.
    pub fn update_progress(&self, progress: usize) -> CheckpointRecord {
        info!("Updating progress to {:?}={}", self.stage, progress);
        CheckpointRecord {
            stage: self.stage,
            is_valid: true,
            progress,
        }
    }

    #[allow(unused)]
    // This function is used for testing purposes only.
    pub(crate) fn get_work_stage(&self) -> WorkStage {
        self.stage
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
    enum LegacyWorkStage {
        QuantizeFPV,
        InMemIndexBuild,
        WriteDiskLayout,
        End,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct LegacyCheckpointRecord {
        stage: LegacyWorkStage,
        is_valid: bool,
        progress: usize,
    }

    #[rstest]
    #[case(LegacyWorkStage::QuantizeFPV, WorkStage::QuantizeFPV, true, 0)]
    #[case(
        LegacyWorkStage::InMemIndexBuild,
        WorkStage::InMemIndexBuild,
        false,
        42
    )]
    #[case(
        LegacyWorkStage::WriteDiskLayout,
        WorkStage::WriteDiskLayout,
        true,
        100
    )]
    #[case(LegacyWorkStage::End, WorkStage::End, false, 0)]
    fn test_backward_compatibility(
        #[case] legacy_stage: LegacyWorkStage,
        #[case] stage: WorkStage,
        #[case] is_valid: bool,
        #[case] progress: usize,
    ) {
        // Test backward compatibility: Newer code (current) reading older data format (legacy)
        let legacy_record = LegacyCheckpointRecord {
            stage: legacy_stage,
            is_valid,
            progress,
        };
        let serialized = bincode::serialize(&legacy_record).unwrap();
        let deserialized: CheckpointRecord = bincode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.stage, stage);
        assert_eq!(deserialized.is_valid, is_valid);
        assert_eq!(deserialized.progress, progress);
    }

    #[rstest]
    #[case(WorkStage::QuantizeFPV, LegacyWorkStage::QuantizeFPV, true, 10)]
    #[case(
        WorkStage::InMemIndexBuild,
        LegacyWorkStage::InMemIndexBuild,
        false,
        30
    )]
    #[case(WorkStage::WriteDiskLayout, LegacyWorkStage::WriteDiskLayout, true, 80)]
    #[case(WorkStage::End, LegacyWorkStage::End, false, 0)]
    fn test_forward_compatibility(
        #[case] current_stage: WorkStage,
        #[case] expected_legacy_stage: LegacyWorkStage,
        #[case] is_valid: bool,
        #[case] progress: usize,
    ) {
        // Test forward compatibility: Older code (legacy) reading newer data format (current)
        // This simulates rolling back to an older version after using a newer version
        let current_record = CheckpointRecord {
            stage: current_stage,
            is_valid,
            progress,
        };

        let serialized = bincode::serialize(&current_record).unwrap();

        // Legacy code should still be able to deserialize common enum variants
        let deserialized: LegacyCheckpointRecord = bincode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.stage, expected_legacy_stage);
        assert_eq!(deserialized.is_valid, is_valid);
        assert_eq!(deserialized.progress, progress);
    }

    #[rstest]
    #[case(WorkStage::PartitionData, true, 25)]
    #[case(WorkStage::BuildIndicesOnShards(0), true, 75)]
    #[case(WorkStage::BuildIndicesOnShards(10), true, 75)]
    #[case(WorkStage::MergeIndices, false, 75)]

    fn test_rolling_back_with_new_variants(
        #[case] stage: WorkStage,
        #[case] is_valid: bool,
        #[case] progress: usize,
    ) {
        // When rolling back to older versions, newer variants should fail to deserialize
        // This is expected behavior and we should test for it
        let current_record = CheckpointRecord {
            stage,
            is_valid,
            progress,
        };

        let serialized = bincode::serialize(&current_record).unwrap();

        // Legacy code should fail to deserialize newer enum variants
        // This is expected behavior - we're testing that it fails
        let result: Result<LegacyCheckpointRecord, bincode::Error> =
            bincode::deserialize(&serialized);
        assert!(
            result.is_err(),
            "Legacy code should not be able to deserialize newer enum variants"
        );
    }
}
