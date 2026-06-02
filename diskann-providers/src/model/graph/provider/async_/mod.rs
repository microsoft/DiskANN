/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
pub mod experimental;

pub mod common;
pub use common::{PrefetchCacheLineLevel, StartPoints, VectorGuard};

pub(crate) mod postprocess;
// Re-export from parent module for backward compatibility.
// The algorithm is not async-specific and lives in provider::determinant_diversity.
pub mod distances;
pub use super::determinant_diversity_post_process;
pub mod memory_vector_provider;
pub use memory_vector_provider::MemoryVectorProviderAsync;

pub mod memory_quant_vector_provider;
pub use memory_quant_vector_provider::MemoryQuantVectorProviderAsync;

pub mod simple_neighbor_provider;
pub use simple_neighbor_provider::SimpleNeighborProviderAsync;

pub mod table_delete_provider;
pub use table_delete_provider::TableDeleteProviderAsync;

pub mod fast_memory_vector_provider;
pub use fast_memory_vector_provider::FastMemoryVectorProviderAsync;

pub mod fast_memory_quant_vector_provider;
pub use fast_memory_quant_vector_provider::FastMemoryQuantVectorProviderAsync;

// The default `inmem` data provider for the async index.
pub mod inmem;
