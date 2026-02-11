/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod graph_layout_version;
pub use graph_layout_version::GraphLayoutVersion;

mod graph_metadata;
pub use graph_metadata::GraphMetadata;

mod graph_header;
pub use graph_header::GraphHeader;

mod cache;
pub use cache::{Cache, CachingStrategy};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify that key types are accessible
        let _ = core::any::type_name::<GraphLayoutVersion>();
        let _ = core::any::type_name::<GraphMetadata>();
        let _ = core::any::type_name::<GraphHeader>();
        let _ = core::any::type_name::<CachingStrategy>();
    }
}
