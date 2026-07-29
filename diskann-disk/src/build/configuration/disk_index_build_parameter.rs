/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Parameters for disk index construction.
use std::num::NonZeroUsize;

use diskann::ANNError;
#[cfg(feature = "pipnn")]
use diskann::ANNResult;
use thiserror::Error;

#[cfg(feature = "pipnn")]
use super::PiPNNParameters;
use super::{BuildAlgorithm, QuantizationType};

use crate::error::{diskann_error, ErrorKind};

/// GB to bytes ratio.
pub const BYTES_IN_GB: f64 = 1024_f64 * 1024_f64 * 1024_f64;

/// Disk sector length in bytes. This is used as the offset alignment and
/// smallest block size when reading/writing index data from/to disk.
pub const DISK_SECTOR_LEN: usize = 4096;

const DEFAULT_DATA_COMPRESSION_CHUNK_VECTOR_COUNT: usize = 25_000;

/// Errors returned when validating PQ chunk parameters.
#[derive(Debug, Error, PartialEq)]
#[error("Budget must be greater than zero")]
pub struct InvalidMemBudget;

impl From<InvalidMemBudget> for ANNError {
    fn from(value: InvalidMemBudget) -> Self {
        diskann_error!(ErrorKind::IndexConfigError("MemoryBudget"), value)
    }
}

/// Memory budget for building the disk index.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MemoryBudget {
    bytes: NonZeroUsize,
}

impl MemoryBudget {
    /// Create a memory budget from gibibytes.
    pub fn try_from_gb(gib: f64) -> Result<Self, InvalidMemBudget> {
        let bytes_f = (gib * BYTES_IN_GB).round() as usize;
        let bytes = NonZeroUsize::new(bytes_f).ok_or(InvalidMemBudget)?;

        Ok(Self { bytes })
    }

    /// Returns the budget in bytes.
    pub fn in_bytes(self) -> usize {
        self.bytes.get()
    }
}

/// Errors returned when validating PQ chunk parameters.
#[derive(Debug, Error, PartialEq)]
pub enum PQChunksError {
    /// Provided dimension was zero.
    #[error("Dimension must be greater than zero")]
    DimensionIsZero,
    /// Requested PQ chunk count falls outside the valid range for the dimension.
    #[error("Number of PQ chunks must be within [1, {dim}], received {num_chunks}")]
    OutOfRange {
        /// Requested PQ chunk count.
        num_chunks: usize,
        /// Dimension used to validate the chunk count.
        dim: usize,
    },
}

impl From<PQChunksError> for ANNError {
    fn from(value: PQChunksError) -> Self {
        diskann_error!(ErrorKind::IndexConfigError("NumPQChunks"), value)
    }
}

/// Validated PQ chunk count used during disk index construction.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct NumPQChunks(NonZeroUsize);

impl NumPQChunks {
    /// Create a validated PQ chunk count.
    pub fn new_with(num_chunks: usize, dim: usize) -> Result<Self, PQChunksError> {
        if dim == 0 {
            return Err(PQChunksError::DimensionIsZero);
        }

        let num_chunks = NonZeroUsize::new(num_chunks).ok_or(PQChunksError::DimensionIsZero)?;

        if num_chunks.get() > dim {
            return Err(PQChunksError::OutOfRange {
                dim,
                num_chunks: num_chunks.get(),
            });
        }

        Ok(Self(num_chunks))
    }

    /// Get the raw chunk count.
    pub fn get(self) -> usize {
        self.0.into()
    }
}

/// Parameters specific for disk index construction.
#[derive(Clone, PartialEq, Debug)]
pub struct DiskIndexBuildParameters {
    /// Limit on graph-construction memory. PiPNN falls back to Vamana when its
    /// estimated one-shot peak exceeds this value.
    build_memory_limit: MemoryBudget,

    /// Number of PQ chunks stored in-memory for search and to be generated during build.
    search_pq_chunks: NumPQChunks,

    /// QuantizationType used to instantiate quantized DataProvider for DiskANN Index during build.
    build_quantization: QuantizationType,

    /// Number of vectors processed per data-compression chunk.
    data_compression_chunk_vector_count: usize,

    /// Which graph construction algorithm to use.
    build_algorithm: BuildAlgorithm,
}

impl DiskIndexBuildParameters {
    /// Create new build parameters from already validated components.
    pub fn new(
        build_memory_limit: MemoryBudget,
        build_quantization: QuantizationType,
        search_pq_chunks: NumPQChunks,
    ) -> Self {
        Self {
            build_memory_limit,
            search_pq_chunks,
            build_quantization,
            data_compression_chunk_vector_count: DEFAULT_DATA_COMPRESSION_CHUNK_VECTOR_COUNT,
            build_algorithm: BuildAlgorithm::default(),
        }
    }

    /// Create parameters for one-shot PiPNN graph construction.
    ///
    /// PiPNN uses the common search-PQ and disk-layout pipeline. The memory
    /// budget selects Vamana when the estimated PiPNN peak does not fit.
    #[cfg(feature = "pipnn")]
    pub fn new_pipnn(
        build_memory_limit: MemoryBudget,
        search_pq_chunks: NumPQChunks,
        config: PiPNNParameters,
    ) -> Self {
        Self {
            build_memory_limit,
            search_pq_chunks,
            build_quantization: QuantizationType::FP,
            data_compression_chunk_vector_count: DEFAULT_DATA_COMPRESSION_CHUNK_VECTOR_COUNT,
            build_algorithm: BuildAlgorithm::PiPNN(config),
        }
    }

    /// Set the number of vectors processed per data-compression chunk.
    pub fn with_data_compression_chunk_vector_count(
        mut self,
        data_compression_chunk_vector_count: usize,
    ) -> Self {
        self.data_compression_chunk_vector_count = data_compression_chunk_vector_count;
        self
    }

    /// Get the configured memory budget for index building.
    pub fn build_memory_limit(&self) -> MemoryBudget {
        self.build_memory_limit
    }

    /// Get quantization type used to instantiate quantized DataProvider for DiskANN Index during build
    pub fn build_quantization(&self) -> &QuantizationType {
        &self.build_quantization
    }

    /// Get user specified PQ chunks count for in-memory search data.
    pub fn search_pq_chunks(&self) -> NumPQChunks {
        self.search_pq_chunks
    }

    /// Get the number of vectors processed per data-compression chunk.
    pub fn data_compression_chunk_vector_count(&self) -> usize {
        self.data_compression_chunk_vector_count
    }

    /// Get the graph-construction algorithm.
    pub fn build_algorithm(&self) -> &BuildAlgorithm {
        &self.build_algorithm
    }

    #[cfg(feature = "pipnn")]
    pub(crate) fn pipnn_config(&self) -> Option<diskann_pipnn::PiPNNConfig> {
        match &self.build_algorithm {
            BuildAlgorithm::PiPNN(config) => Some(config.into()),
            BuildAlgorithm::Vamana => None,
        }
    }

    #[cfg(feature = "pipnn")]
    pub(crate) fn use_vamana_if_pipnn_exceeds(
        &mut self,
        npoints: usize,
        dimensions: usize,
        element_size: usize,
        num_threads: usize,
    ) -> ANNResult<usize> {
        let parameters = match &self.build_algorithm {
            BuildAlgorithm::PiPNN(parameters) => parameters,
            BuildAlgorithm::Vamana => {
                return Err(ANNError::log_index_error(
                    "memory selection requires PiPNN parameters",
                ));
            }
        };
        let estimate =
            estimate_pipnn_peak_memory(parameters, npoints, dimensions, element_size, num_threads);
        if estimate.is_none_or(|bytes| bytes > self.build_memory_limit.in_bytes()) {
            self.build_algorithm = BuildAlgorithm::Vamana;
        }
        Ok(estimate.unwrap_or(usize::MAX))
    }
}

#[cfg(feature = "pipnn")]
fn estimate_pipnn_peak_memory(
    config: &PiPNNParameters,
    npoints: usize,
    dimensions: usize,
    element_size: usize,
    num_threads: usize,
) -> Option<usize> {
    const PROPORTIONAL_HEADROOM_PERCENT: u128 = 108;
    const PROCESS_HEADROOM: u128 = 16 * 1024 * 1024;
    const PER_WORKER_HEADROOM: u128 = 28 * 1024 * 1024;

    let workers = if num_threads == 0 {
        std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
    } else {
        num_threads
    } as u128;
    let copies = config
        .fanout
        .iter()
        .try_fold(config.replicas as u128, |copies, &fanout| {
            copies.checked_mul(fanout as u128)
        })?;
    let dataset = (dimensions as u128).checked_mul(element_size as u128)?;
    let leaf_ids = copies.checked_mul(size_of::<u32>() as u128)?;
    let offered = copies.checked_mul(config.k as u128)?.checked_mul(2)?;
    let candidate_capacity = offered.checked_next_power_of_two()?;
    let candidate_storage = candidate_capacity.checked_mul(size_of::<u32>() as u128)?;
    let candidate_metadata =
        size_of::<std::sync::Mutex<diskann::graph::AdjacencyList<u32>>>() as u128;

    let partition_per_point = dataset
        .checked_add(16)?
        .checked_add(leaf_ids.checked_mul(2)?)?;
    let leaf_per_point = dataset
        .checked_add(candidate_metadata)?
        .checked_add(candidate_storage)?
        .checked_add(leaf_ids)?;
    let finalization_per_point = dataset
        .checked_add(candidate_metadata)?
        .checked_add(size_of::<Vec<u32>>() as u128)?
        .checked_add(candidate_storage)?;

    let leaf_size = config.c_max.min(npoints).max(1) as u128;
    let leaf_scratch = leaf_size
        .checked_mul(dimensions as u128)?
        .checked_mul(size_of::<f32>() as u128)?
        .checked_add(leaf_size.checked_mul(leaf_size)?.checked_mul(5)?)?
        .checked_add(leaf_size.checked_mul(config.k as u128)?.checked_mul(24)?)?
        .checked_add(leaf_size.checked_mul(20)?)?
        .checked_add(68)?;
    let partition_rows = (npoints as u128).min(1_024);
    let partition_leaders = (npoints as u128).min(1_000);
    let partition_scratch = partition_leaders
        .checked_mul(dimensions as u128)?
        .checked_mul(size_of::<f32>() as u128)?
        .checked_add(
            partition_rows
                .checked_mul(dimensions as u128)?
                .checked_mul(size_of::<f32>() as u128)?,
        )?
        .checked_add(
            partition_rows
                .checked_mul(partition_leaders)?
                .checked_mul(size_of::<f32>() as u128)?,
        )?;

    let points = npoints as u128;
    let structural = points
        .checked_mul(partition_per_point)?
        .checked_add(workers.checked_mul(partition_scratch)?)?
        .max(
            points
                .checked_mul(leaf_per_point)?
                .checked_add(workers.checked_mul(leaf_scratch)?)?,
        )
        .max(points.checked_mul(finalization_per_point)?);
    let proportional = structural
        .checked_mul(PROPORTIONAL_HEADROOM_PERCENT)?
        .div_ceil(100);
    let allocator_floor = structural
        .checked_add(PROCESS_HEADROOM)?
        .checked_add(workers.checked_mul(PER_WORKER_HEADROOM)?)?;
    usize::try_from(proportional.max(allocator_floor)).ok()
}

#[cfg(test)]
mod dataset_test {
    use diskann::ANNError;

    use crate::error::{error_kind, ErrorKind};

    use super::*;

    #[test]
    fn memory_budget_converts_units() {
        let budget = MemoryBudget::try_from_gb(2.0).unwrap();
        assert_eq!(budget.in_bytes() as f64, 2.0 * BYTES_IN_GB);
        assert!(MemoryBudget::try_from_gb(0.0).is_err());
    }

    #[test]
    fn build_with_num_of_pq_chunks_should_work() {
        let memory_budget = MemoryBudget::try_from_gb(2.0).unwrap();
        let num_pq_chunks = NumPQChunks::new_with(20, 128).unwrap();

        let result = DiskIndexBuildParameters::new(
            memory_budget,
            QuantizationType::default(),
            num_pq_chunks,
        );

        assert_eq!(result.search_pq_chunks().get(), num_pq_chunks.get());
        assert_eq!(
            result.data_compression_chunk_vector_count(),
            DEFAULT_DATA_COMPRESSION_CHUNK_VECTOR_COUNT
        );
    }

    #[test]
    fn data_compression_chunk_vector_count_can_be_configured() {
        let memory_budget = MemoryBudget::try_from_gb(2.0).unwrap();
        let num_pq_chunks = NumPQChunks::new_with(20, 128).unwrap();

        let result = DiskIndexBuildParameters::new(
            memory_budget,
            QuantizationType::default(),
            num_pq_chunks,
        )
        .with_data_compression_chunk_vector_count(10_000);

        assert_eq!(result.data_compression_chunk_vector_count(), 10_000);
    }

    #[test]
    fn disk_index_build_parameters_try_new_handles_invalid() {
        // Test valid parameters
        let memory_budget = MemoryBudget::try_from_gb(1.0).unwrap();
        let pq_chunks = NumPQChunks::new_with(1, 128).unwrap();
        let params =
            DiskIndexBuildParameters::new(memory_budget, QuantizationType::default(), pq_chunks);

        assert_eq!(
            params.build_memory_limit().in_bytes() as f64,
            1.0 * BYTES_IN_GB
        );

        // Test invalid parameters
        assert!(MemoryBudget::try_from_gb(0.0).is_err());

        let err = MemoryBudget::try_from_gb(-1.0)
            .map_err(ANNError::from)
            .unwrap_err();
        assert_eq!(
            error_kind(&err),
            ErrorKind::IndexConfigError("MemoryBudget")
        );
    }

    #[test]
    fn num_pq_chunks_new_rejects_invalid_values() {
        assert!(NumPQChunks::new_with(0, 128).is_err());
        assert!(NumPQChunks::new_with(129, 128).is_err());
        assert!(NumPQChunks::new_with(1, 0).is_err());
    }

    #[test]
    fn num_pq_chunks_new_accepts_valid_values() {
        let chunks = NumPQChunks::new_with(64, 128).unwrap();
        assert_eq!(chunks.get(), 64);
    }

    #[cfg(feature = "pipnn")]
    #[test]
    fn new_pipnn_uses_the_common_disk_pipeline_parameters() {
        let pq = NumPQChunks::new_with(1, 128).unwrap();
        let parameters = PiPNNParameters::default();
        let config = diskann_pipnn::PiPNNConfig::from(&parameters);
        let budget = MemoryBudget::try_from_gb(2.0).unwrap();
        let params = DiskIndexBuildParameters::new_pipnn(budget, pq, parameters);

        assert_eq!(params.pipnn_config(), Some(config));
        assert_eq!(params.search_pq_chunks(), pq);
        assert_eq!(
            params.data_compression_chunk_vector_count(),
            DEFAULT_DATA_COMPRESSION_CHUNK_VECTOR_COUNT
        );
        assert!(matches!(params.build_algorithm(), BuildAlgorithm::PiPNN(_)));
    }

    #[cfg(feature = "pipnn")]
    #[test]
    fn pipnn_memory_estimate_covers_measured_bigann10m_peak() {
        const MEASURED_PEAK: usize = 8_656_004 * 1024;
        let parameters = PiPNNParameters {
            c_max: 512,
            c_min: 64,
            p_samp: 0.01,
            fanout: vec![10, 3],
            k: 2,
            replicas: 1,
        };

        let estimate = estimate_pipnn_peak_memory(&parameters, 10_000_000, 128, 2, 16).unwrap();

        assert!(estimate >= MEASURED_PEAK);
        assert!(estimate <= MEASURED_PEAK * 120 / 100);
    }

    #[cfg(feature = "pipnn")]
    #[test]
    fn pipnn_falls_back_to_vamana_above_budget() {
        let budget = MemoryBudget::try_from_gb(1.0).unwrap();
        let pq = NumPQChunks::new_with(1, 128).unwrap();
        let mut params =
            DiskIndexBuildParameters::new_pipnn(budget, pq, PiPNNParameters::default());

        let estimate = params
            .use_vamana_if_pipnn_exceeds(10_000_000, 128, 2, 16)
            .unwrap();

        assert!(estimate > budget.in_bytes());
        assert!(matches!(params.build_algorithm(), BuildAlgorithm::Vamana));
        assert_eq!(params.build_quantization(), &QuantizationType::FP);
    }
}
