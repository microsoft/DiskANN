/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Parameters for disk index construction.
use std::num::NonZeroUsize;

use diskann::ANNError;
use thiserror::Error;

use super::QuantizationType;

/// GB to bytes ratio.
pub const BYTES_IN_GB: f64 = 1024_f64 * 1024_f64 * 1024_f64;

/// Disk sector length in bytes. This is used as the offset alignment and
/// smallest block size when reading/writing index data from/to disk.
pub const DISK_SECTOR_LEN: usize = 4096;

/// Errors returned when validating PQ chunk parameters.
#[derive(Debug, Error, PartialEq)]
#[error("Budget must be greater than zero")]
pub struct InvalidMemBudget;

impl From<InvalidMemBudget> for ANNError {
    fn from(value: InvalidMemBudget) -> Self {
        ANNError::log_index_config_error("MemoryBudget".to_string(), format!("{value:?}"))
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
        ANNError::log_index_config_error("NumPQChunks".to_string(), format!("{value:?}"))
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
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DiskIndexBuildParameters {
    /// Limit on the memory allowed for building the index.
    build_memory_limit: MemoryBudget,

    /// Number of PQ chunks stored in-memory for search and to be generated during build.
    search_pq_chunks: NumPQChunks,

    /// QuantizationType used to instantiate quantized DataProvider for DiskANN Index during build.
    build_quantization: QuantizationType,
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
        }
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
}

#[cfg(test)]
mod dataset_test {
    use diskann::{ANNError, ANNErrorKind};

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
        assert_eq!(err.kind(), ANNErrorKind::IndexConfigError);
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
}
