/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared disk-build inputs and algorithm-specific graph settings.

use std::num::NonZeroUsize;

use diskann::graph::config::Config;
use diskann_providers::model::{
    IndexConfiguration, graph::provider::async_::PrefetchCacheLineLevel,
};
use diskann_vector::distance::Metric;

use super::{
    DEFAULT_DATA_COMPRESSION_CHUNK_VECTOR_COUNT, DiskIndexBuildParameters, MemoryBudget,
    NumPQChunks, QuantizationType,
};

/// Shared inputs for disk construction, with one selected graph algorithm.
#[derive(Clone, Debug)]
pub struct DiskIndexBuildConfig {
    /// Dataset metric.
    pub metric: Metric,
    /// Stored scalar elements per row.
    pub dim: usize,
    /// Dataset row count.
    pub num_points: usize,
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

/// Graph-specific configuration; dataset and search-PQ settings stay outside it.
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
        /// Initial candidates retained by the Vamana provider.
        frozen_points: NonZeroUsize,
        /// Vector-provider prefetch lookahead.
        prefetch_lookahead: Option<NonZeroUsize>,
        /// Vector-provider prefetch granularity.
        prefetch_cache_line_level: Option<PrefetchCacheLineLevel>,
    },
    /// PiPNN partitioning and pruning policy.
    #[cfg(feature = "pipnn")]
    Pipnn(diskann::graph::pipnn::Config),
}

impl DiskIndexBuildConfig {
    /// Configures a disk build with default compression batching and unseeded PQ sampling.
    pub fn new(
        metric: Metric,
        dim: usize,
        num_points: usize,
        num_threads: usize,
        search_pq_chunks: NumPQChunks,
        graph: GraphBuildConfig,
    ) -> Self {
        Self {
            metric,
            dim,
            num_points,
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

impl From<(IndexConfiguration, DiskIndexBuildParameters)> for DiskIndexBuildConfig {
    fn from((index, parameters): (IndexConfiguration, DiskIndexBuildParameters)) -> Self {
        Self {
            metric: index.dist_metric,
            dim: index.dim,
            num_points: index.max_points,
            num_threads: index.num_threads,
            search_pq_chunks: parameters.search_pq_chunks(),
            compression_chunk_vector_count: parameters.data_compression_chunk_vector_count(),
            seed: index.random_seed,
            graph: GraphBuildConfig::Vamana {
                graph: index.config,
                memory_budget: parameters.build_memory_limit(),
                build_quantization: *parameters.build_quantization(),
                frozen_points: index.num_frozen_pts,
                prefetch_lookahead: index.prefetch_lookahead,
                prefetch_cache_line_level: index.prefetch_cache_line_level,
            },
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BuildParameters {
    pub metric: Metric,
    pub dim: usize,
    pub num_points: usize,
    pub num_threads: usize,
    pub search_pq_chunks: NumPQChunks,
    pub compression_chunk_vector_count: usize,
    pub seed: Option<u64>,
}

impl From<&DiskIndexBuildConfig> for BuildParameters {
    fn from(config: &DiskIndexBuildConfig) -> Self {
        Self {
            metric: config.metric,
            dim: config.dim,
            num_points: config.num_points,
            num_threads: config.num_threads,
            search_pq_chunks: config.search_pq_chunks,
            compression_chunk_vector_count: config.compression_chunk_vector_count,
            seed: config.seed,
        }
    }
}

impl BuildParameters {
    pub(crate) fn index_configuration(
        &self,
        graph: Config,
        frozen_points: NonZeroUsize,
        prefetch_lookahead: Option<NonZeroUsize>,
        prefetch_cache_line_level: Option<PrefetchCacheLineLevel>,
    ) -> IndexConfiguration {
        IndexConfiguration {
            config: graph,
            dist_metric: self.metric,
            dim: self.dim,
            max_points: self.num_points,
            num_threads: self.num_threads,
            random_seed: self.seed,
            num_frozen_pts: frozen_points,
            prefetch_lookahead,
            prefetch_cache_line_level,
        }
    }
}
