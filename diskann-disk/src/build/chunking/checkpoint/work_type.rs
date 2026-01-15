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
