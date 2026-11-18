/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Parameters for disk index construction.
use std::num::NonZeroUsize;

use diskann::ANNError;
use thiserror::Error;

use crate::error::{ErrorKind, diskann_error};

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

#[cfg(test)]
mod dataset_test {
    use diskann::ANNError;

    use crate::error::{ErrorKind, error_kind};

    use super::*;

    #[test]
    fn memory_budget_converts_units() {
        let budget = MemoryBudget::try_from_gb(2.0).unwrap();
        assert_eq!(budget.in_bytes() as f64, 2.0 * BYTES_IN_GB);
        assert!(MemoryBudget::try_from_gb(0.0).is_err());
    }

    #[test]
    fn negative_memory_budget_returns_config_error() {
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
}
