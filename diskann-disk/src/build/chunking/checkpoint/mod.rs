/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod checkpoint_record_manager;
pub use checkpoint_record_manager::{
    CheckpointManager, CheckpointManagerClone, CheckpointManagerExt,
};

mod checkpoint_record;
pub use checkpoint_record::CheckpointRecord;

mod checkpoint_record_manager_with_file;
pub use checkpoint_record_manager_with_file::CheckpointRecordManagerWithFileStorage;

mod naive_checkpoint_record_manager;
pub use naive_checkpoint_record_manager::NaiveCheckpointRecordManager;

mod progress;
pub use progress::Progress;

mod work_type;
pub use work_type::WorkStage;

mod checkpoint_context;
pub use checkpoint_context::{CheckpointContext, OwnedCheckpointContext};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify that key types are accessible
        let _ = core::any::type_name::<CheckpointRecord>();
        let _ = core::any::type_name::<CheckpointRecordManagerWithFileStorage>();
        let _ = core::any::type_name::<NaiveCheckpointRecordManager>();
        let _ = core::any::type_name::<Progress>();
        let _ = core::any::type_name::<WorkStage>();
        let _ = core::any::type_name::<CheckpointContext>();
        let _ = core::any::type_name::<OwnedCheckpointContext>();
    }
}
