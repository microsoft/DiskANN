/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
pub mod disk_index_build_parameter;
pub use disk_index_build_parameter::{DiskIndexBuildParameters, MemoryBudget, NumPQChunks};

pub mod filter_parameter;

pub mod quantization_types;
pub use quantization_types::QuantizationType;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify that key types are accessible
        let _ = core::any::type_name::<DiskIndexBuildParameters>();
        let _ = core::any::type_name::<MemoryBudget>();
        let _ = core::any::type_name::<NumPQChunks>();
        let _ = core::any::type_name::<QuantizationType>();
    }
}
