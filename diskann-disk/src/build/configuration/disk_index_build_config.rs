/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Configuration selected once for disk graph construction.

use diskann_providers::model::IndexConfiguration;
use diskann_vector::distance::Metric;

#[cfg(feature = "pipnn")]
use super::disk_index_build_parameter::DEFAULT_DATA_COMPRESSION_CHUNK_VECTOR_COUNT;
use super::{DiskIndexBuildParameters, NumPQChunks};

/// Owns the selected algorithm's configuration, without parameters for the other algorithm.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum DiskIndexBuildConfig {
    /// Existing Vamana configuration, including its build representation and RAM budget.
    Vamana {
        /// Vamana provider and graph settings.
        index: IndexConfiguration,
        /// Disk-search compression and Vamana build settings.
        parameters: DiskIndexBuildParameters,
    },
    /// PiPNN graph settings and the shared disk-build inputs.
    #[cfg(feature = "pipnn")]
    Pipnn {
        /// PiPNN partitioning and pruning policy.
        graph: diskann::graph::pipnn::Config,
        /// Dataset metric.
        metric: Metric,
        /// Number of stored scalar elements per row.
        dim: usize,
        /// Dataset row count.
        num_points: usize,
        /// Worker count; zero selects the system default.
        num_threads: usize,
        /// Search-time PQ code size.
        search_pq_chunks: NumPQChunks,
        /// Number of vectors compressed per batch.
        compression_chunk_vector_count: usize,
        /// Seed for search-PQ training and medoid sampling.
        seed: Option<u64>,
    },
}

impl DiskIndexBuildConfig {
    /// Configures PiPNN with default disk compression batching and unseeded PQ sampling.
    #[cfg(feature = "pipnn")]
    pub fn pipnn(
        metric: Metric,
        dim: usize,
        num_points: usize,
        num_threads: usize,
        search_pq_chunks: NumPQChunks,
        graph: diskann::graph::pipnn::Config,
    ) -> Self {
        Self::Pipnn {
            graph,
            metric,
            dim,
            num_points,
            num_threads,
            search_pq_chunks,
            compression_chunk_vector_count: DEFAULT_DATA_COMPRESSION_CHUNK_VECTOR_COUNT,
            seed: None,
        }
    }

    /// Seeds disk-search PQ and medoid sampling; graph-specific seeds remain independent.
    pub fn with_seed(mut self, seed: u64) -> Self {
        match &mut self {
            Self::Vamana { index, .. } => index.random_seed = Some(seed),
            #[cfg(feature = "pipnn")]
            Self::Pipnn { seed: value, .. } => *value = Some(seed),
        }
        self
    }

    /// Sets the number of vectors compressed per batch.
    pub fn with_data_compression_chunk_vector_count(mut self, count: usize) -> Self {
        match &mut self {
            Self::Vamana { parameters, .. } => {
                *parameters = parameters.with_data_compression_chunk_vector_count(count);
            }
            #[cfg(feature = "pipnn")]
            Self::Pipnn {
                compression_chunk_vector_count,
                ..
            } => *compression_chunk_vector_count = count,
        }
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BuildParameters {
    pub metric: Metric,
    pub dim: usize,
    pub num_points: usize,
    pub num_threads: usize,
    pub search_pq_chunks: NumPQChunks,
    pub compression_chunk_vector_count: usize,
    pub seed: Option<u64>,
}

impl From<(&IndexConfiguration, &DiskIndexBuildParameters)> for BuildParameters {
    fn from((index, parameters): (&IndexConfiguration, &DiskIndexBuildParameters)) -> Self {
        Self {
            metric: index.dist_metric,
            dim: index.dim,
            num_points: index.max_points,
            num_threads: index.num_threads,
            search_pq_chunks: parameters.search_pq_chunks(),
            compression_chunk_vector_count: parameters.data_compression_chunk_vector_count(),
            seed: index.random_seed,
        }
    }
}
