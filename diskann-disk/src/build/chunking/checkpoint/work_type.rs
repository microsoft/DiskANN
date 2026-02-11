/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
// WorkStage implements the partial ordering trait,
// ..which is used to decide which stage should the index build be resumed from.
pub enum WorkStage {
    QuantizeFPV,
    InMemIndexBuild,
    WriteDiskLayout,
    // End is not an actual step but used to indicate that the index build
    // ..process is completed.
    End,
    PartitionData,
    BuildIndicesOnShards(usize),
    MergeIndices,
    TrainBuildQuantizer,
    // Start is not an actual step but used to indicate the beginning of the index build process.
    Start,
    // Always add new stages at the end of the enum to avoid breaking the serialization order.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_stage_enum_variants() {
        // Test all enum variants can be created
        let _ = WorkStage::QuantizeFPV;
        let _ = WorkStage::InMemIndexBuild;
        let _ = WorkStage::WriteDiskLayout;
        let _ = WorkStage::End;
        let _ = WorkStage::PartitionData;
        let _ = WorkStage::BuildIndicesOnShards(5);
        let _ = WorkStage::MergeIndices;
        let _ = WorkStage::TrainBuildQuantizer;
        let _ = WorkStage::Start;
    }

    #[test]
    fn test_work_stage_equality() {
        assert_eq!(WorkStage::Start, WorkStage::Start);
        assert_eq!(WorkStage::End, WorkStage::End);
        assert_eq!(WorkStage::BuildIndicesOnShards(3), WorkStage::BuildIndicesOnShards(3));
        assert_ne!(WorkStage::Start, WorkStage::End);
        assert_ne!(WorkStage::BuildIndicesOnShards(1), WorkStage::BuildIndicesOnShards(2));
    }

    #[test]
    fn test_work_stage_clone() {
        let stage = WorkStage::QuantizeFPV;
        let cloned = stage.clone();
        assert_eq!(stage, cloned);
    }

    #[test]
    fn test_work_stage_copy() {
        let stage = WorkStage::InMemIndexBuild;
        let copied = stage;
        assert_eq!(stage, copied);
    }

    #[test]
    fn test_work_stage_debug() {
        let stage = WorkStage::Start;
        let debug_str = format!("{:?}", stage);
        assert!(debug_str.contains("Start"));
    }

    #[test]
    fn test_work_stage_serialization() {
        let stage = WorkStage::BuildIndicesOnShards(42);
        let serialized = bincode::serialize(&stage).unwrap();
        let deserialized: WorkStage = bincode::deserialize(&serialized).unwrap();
        assert_eq!(stage, deserialized);
    }
}
