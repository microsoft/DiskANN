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
