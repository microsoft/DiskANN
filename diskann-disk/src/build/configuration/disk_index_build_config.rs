/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared disk-build inputs and algorithm-specific graph settings.

use std::num::NonZeroUsize;

use diskann::graph::config::Config;
use diskann_vector::distance::Metric;

use super::{
    DEFAULT_DATA_COMPRESSION_CHUNK_VECTOR_COUNT, MemoryBudget, NumPQChunks, QuantizationType,
};

/// Shared build policies; dataset shape is read from the input file.
#[derive(Clone, Debug)]
pub struct DiskIndexBuildConfig {
    /// Dataset metric.
    pub metric: Metric,
    /// Worker count; zero selects the system default.
    pub num_threads: usize,
    /// Search-time PQ code size.
    pub search_pq_chunks: NumPQChunks,
    /// Number of vectors compressed per batch.
    pub compression_chunk_vector_count: usize,
    /// Seed for search-PQ training and medoid sampling.
    pub seed: Option<u64>,
    /// Parameters used only by the selected graph algorithm.
    pub graph: GraphBuildConfig,
}

/// Graph-specific configuration; search-PQ settings stay outside it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum GraphBuildConfig {
    /// Vamana insertion and build-provider settings.
    Vamana {
        /// Vamana search and pruning policy.
        graph: Config,
        /// Budget used to select one-shot or sharded construction.
        memory_budget: MemoryBudget,
        /// Vamana build representation, independent of search-time PQ.
        build_quantization: QuantizationType,
        /// Vector-provider prefetch lookahead.
        prefetch_lookahead: Option<NonZeroUsize>,
    },
    /// PiPNN partitioning and pruning policy.
    #[cfg(feature = "pipnn")]
    Pipnn(diskann::graph::pipnn::Config),
}

impl DiskIndexBuildConfig {
    /// Configures a disk build with default compression batching and unseeded PQ sampling.
    pub fn new(
        metric: Metric,
        num_threads: usize,
        search_pq_chunks: NumPQChunks,
        graph: GraphBuildConfig,
    ) -> Self {
        Self {
            metric,
            num_threads,
            search_pq_chunks,
            graph,
            compression_chunk_vector_count: DEFAULT_DATA_COMPRESSION_CHUNK_VECTOR_COUNT,
            seed: None,
        }
    }

    /// Seeds search-PQ training and medoid sampling; graph-specific seeds stay independent.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the number of vectors compressed per batch.
    pub fn with_data_compression_chunk_vector_count(mut self, count: usize) -> Self {
        self.compression_chunk_vector_count = count;
        self
    }
}
