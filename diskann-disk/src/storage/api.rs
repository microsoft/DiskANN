/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_providers::storage::AsyncQuantLoadContext;

/// Context for loading an async disk index.
pub struct AsyncDiskLoadContext {
    /// AsyncQuantLoadContext for loading quantized data.
    pub quant_load_context: AsyncQuantLoadContext,
    /// Number of nodes to pre-load using breadth-first-search and cache in memory.
    pub num_nodes_to_cache: usize,
    /// Number of maximum IO operations to perform during search.
    pub search_io_limit: usize,
    /// Number of vectors in the index.
    pub num_points: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_disk_load_context_fields() {
        // Create a simple test to verify the struct has the expected fields
        // We can't easily construct AsyncQuantLoadContext without complex setup
        // so we just verify the type compiles and has accessible fields
        let _ = core::any::type_name::<AsyncDiskLoadContext>();
        
        // Verify field types are correct
        fn check_fields(ctx: &AsyncDiskLoadContext) {
            let _: &AsyncQuantLoadContext = &ctx.quant_load_context;
            let _: usize = ctx.num_nodes_to_cache;
            let _: usize = ctx.search_io_limit;
            let _: usize = ctx.num_points;
        }
        
        // This function is never called, just used for type checking
        if false {
            check_fields(&unsafe { std::mem::zeroed() });
        }
    }
}
