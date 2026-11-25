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

/// Build policies shared across graph construction algorithms.
///
/// Dataset shape is read from the input file.
#[derive(Clone, Copy, Debug)]
pub struct SharedBuildConfig {
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
}

impl SharedBuildConfig {
    /// Configures shared disk-build policies with default compression batching
    /// and unseeded PQ sampling.
    pub fn new(metric: Metric, num_threads: usize, search_pq_chunks: NumPQChunks) -> Self {
        Self {
            metric,
            num_threads,
            search_pq_chunks,
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

/// Shared build policies paired with algorithm-specific graph settings `G`.
#[derive(Clone, Debug)]
pub struct DiskIndexBuildConfig<G = VamanaBuildConfig> {
    /// Build policies shared across graph construction algorithms.
    pub shared: SharedBuildConfig,
    /// Parameters used only by the selected graph algorithm.
    pub graph: G,
}

/// Vamana settings for one-shot and sharded disk builds.
#[derive(Clone, Debug)]
pub struct VamanaBuildConfig {
    /// Vamana search and pruning policy.
    pub graph: Config,
    /// Budget used to select one-shot or sharded construction.
    pub memory_budget: MemoryBudget,
    /// Vamana build representation, independent of search-time PQ.
    pub build_quantization: QuantizationType,
    /// Vector-provider prefetch lookahead.
    pub prefetch_lookahead: Option<NonZeroUsize>,
}

impl<G> DiskIndexBuildConfig<G> {
    /// Configures a disk build with default compression batching and unseeded PQ sampling.
    pub fn new(
        metric: Metric,
        num_threads: usize,
        search_pq_chunks: NumPQChunks,
        graph: G,
    ) -> Self {
        Self {
            shared: SharedBuildConfig::new(metric, num_threads, search_pq_chunks),
            graph,
        }
    }

    /// Pairs an existing [`SharedBuildConfig`] with algorithm-specific graph settings.
    pub fn from_shared(shared: SharedBuildConfig, graph: G) -> Self {
        Self { shared, graph }
    }

    /// Seeds search-PQ training and medoid sampling; graph-specific seeds stay independent.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.shared = self.shared.with_seed(seed);
        self
    }

    /// Sets the number of vectors compressed per batch.
    pub fn with_data_compression_chunk_vector_count(mut self, count: usize) -> Self {
        self.shared = self.shared.with_data_compression_chunk_vector_count(count);
        self
    }
}
