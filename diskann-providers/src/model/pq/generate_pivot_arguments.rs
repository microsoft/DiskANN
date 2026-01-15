/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Arguments for generate pq pivot

use diskann::ANNError;
use thiserror::Error;

/// Represents the configuration parameters required to generate pivots for Product Quantization (PQ).
///
/// PQ is a technique used to compress high-dimensional vectors into a lower-dimensional space for
/// efficient similarity search. This struct defines the parameters needed for training and
/// partitioning the data during PQ.
///
/// # Fields
///
/// * `num_train` - The number of training samples used to train the k-means clustering algorithm.
///   A larger number of samples can improve clustering accuracy but increases computational cost.
///
/// * `dim` - The dimensionality of each input data vector. This specifies the number of features
///   in each vector that will be processed during PQ.
///
/// * `num_centers` - The number of cluster centers (or centroids) to generate during the k-means
///   clustering process. Each center represents a prototype for the data in its cluster.
///
/// * `num_pq_chunks` - The number of chunks into which each vector is divided. Each chunk is independently
///   quantized to a single byte, defining the compression ratio. For example, if the dimensionality
///   (`dim`) is 128 and `num_pq_chunks` is 8, each chunk will span 16 dimensions.
///
/// * `max_k_means_reps` - The maximum number of iterations for the k-means clustering algorithm.
///   Increasing this value can improve clustering quality at the cost of additional computation time.
///
/// * `translate_to_center` - A boolean flag indicating whether the data should be translated
///   (centered) to the origin before clustering. Centering can improve clustering performance by
///   reducing variance caused by the global offset of the data.
#[derive(Debug, Clone)]
pub struct GeneratePivotArguments {
    num_train: usize,
    dim: usize,
    num_centers: usize,
    num_pq_chunks: usize,
    max_k_means_reps: usize,
    translate_to_center: bool,
}

#[derive(Error, Debug, PartialEq)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum GeneratePivotArgumentsError {
    #[error("number of chunks {num_pq_chunks} more than dimension {dim}")]
    NumChunksMoreThanDim { num_pq_chunks: usize, dim: usize },

    #[error("invalid number of chunks 0 reatively to dimension")]
    NumChunksIsZero,

    #[error("vector dimension {0} is greater than i32::MAX_VALUE")]
    DimGreaterThanI32MaxValue(usize),

    #[error("number of vectors {0} is greater than i32::MAX_VALUE")]
    NumTrainGreaterThanI32MaxValue(usize),
}

// Compatibility with ANNError.
impl From<GeneratePivotArgumentsError> for ANNError {
    #[track_caller]
    fn from(value: GeneratePivotArgumentsError) -> Self {
        ANNError::log_pq_error(value)
    }
}

impl GeneratePivotArguments {
    /// Constructor
    pub fn new(
        num_train: usize,
        dim: usize,
        num_centers: usize,
        num_pq_chunks: usize,
        max_k_means_reps: usize,
        translate_to_center: bool,
    ) -> Result<Self, GeneratePivotArgumentsError> {
        if num_pq_chunks > dim {
            return Err(GeneratePivotArgumentsError::NumChunksMoreThanDim { num_pq_chunks, dim });
        }

        if num_pq_chunks == 0 {
            return Err(GeneratePivotArgumentsError::NumChunksIsZero);
        }

        if dim > (i32::MAX as usize) {
            // cblas::sgemm takes i32 parameters, so we need to ensure dim can be cast to i32 without truncation
            return Err(GeneratePivotArgumentsError::DimGreaterThanI32MaxValue(dim));
        }

        if num_train > (i32::MAX as usize) {
            // cblas::sgemm takes i32 parameters, so we need to ensure num_train can be cast to i32 without truncation
            return Err(GeneratePivotArgumentsError::NumTrainGreaterThanI32MaxValue(
                num_train,
            ));
        }

        Ok(Self {
            num_train,
            dim,
            num_centers,
            num_pq_chunks,
            max_k_means_reps,
            translate_to_center,
        })
    }

    /// Get number of training vectors
    pub fn num_train(&self) -> usize {
        self.num_train
    }

    /// Get dimension of training vectors
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get number of centers
    pub fn num_centers(&self) -> usize {
        self.num_centers
    }

    /// Get number of PQ chunks
    pub fn num_pq_chunks(&self) -> usize {
        self.num_pq_chunks
    }

    /// Get maximum number of K-means repetitions
    pub fn max_k_means_reps(&self) -> usize {
        self.max_k_means_reps
    }

    /// Get whether to translate to center
    pub fn translate_to_center(&self) -> bool {
        self.translate_to_center
    }
}

#[cfg(test)]
mod arguments_test {
    use diskann::{ANNErrorKind, ANNResult};

    use super::*;

    #[test]
    fn num_chunks_exceeds_dim() {
        let num_train = 10;
        let dim = 5;
        let num_centers = 2;
        let num_pq_chunks = dim + 1; // number of chunks more than dimension.
        let max_k_means_reps = 10;

        let result = GeneratePivotArguments::new(
            num_train,
            dim,
            num_centers,
            num_pq_chunks,
            max_k_means_reps,
            true,
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            GeneratePivotArgumentsError::NumChunksMoreThanDim { num_pq_chunks, dim }
        );
    }

    #[test]
    fn num_chunks_is_zero() {
        let num_train = 10;
        let dim = 5;
        let num_centers = 2;
        let num_pq_chunks = 0; // number of chunks is zero.
        let max_k_means_reps = 10;

        let result = GeneratePivotArguments::new(
            num_train,
            dim,
            num_centers,
            num_pq_chunks,
            max_k_means_reps,
            true,
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            GeneratePivotArgumentsError::NumChunksIsZero
        );
    }

    #[test]
    fn num_dim_exceeds_i32_max() {
        let num_train = 10;
        let dim = i32::MAX as usize + 1; // vector dimension is greater than i32::MAX_VALUE.
        let num_centers = 2;
        let num_pq_chunks = 2;
        let max_k_means_reps = 10;

        let result = GeneratePivotArguments::new(
            num_train,
            dim,
            num_centers,
            num_pq_chunks,
            max_k_means_reps,
            true,
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            GeneratePivotArgumentsError::DimGreaterThanI32MaxValue(dim)
        );
    }

    #[test]
    fn num_train_exceeds_i32_max() {
        let num_train = i32::MAX as usize + 1; // number of vectors is greater than i32::MAX_VALUE.
        let dim = 5;
        let num_centers = 2;
        let num_pq_chunks = 2;
        let max_k_means_reps = 10;

        let result = GeneratePivotArguments::new(
            num_train,
            dim,
            num_centers,
            num_pq_chunks,
            max_k_means_reps,
            true,
        );

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            GeneratePivotArgumentsError::NumTrainGreaterThanI32MaxValue(num_train)
        );
    }

    #[test]
    fn compatibility_with_ann_error_test() {
        let result = compatibility_helper();

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), ANNErrorKind::PQError,);
    }

    fn compatibility_helper() -> ANNResult<()> {
        Err(GeneratePivotArgumentsError::NumChunksIsZero)?
    }
}
