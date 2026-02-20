/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{utils::VectorRepr, ANNResult};
use diskann_utils::views::{MatrixView, MutMatrixView};

/// [`CompressionStage`] defines the stage of compression
/// used by the compressor to determine how to instantiate the quantizer
/// (e.g., whether to initialize codebooks for a product quantizer).
/// Passed to [`QuantCompressor::new_at_stage`] when creating quantizer instances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompressionStage {
    Start,
    Resume,
}

/// [`QuantCompressor`] defines the interface for quantizer with [`QuantDataGenerator`]
///
/// This trait serves as a general wrapper for different quantizers, allowing them to be
/// used interchangeably with QuantDataGenerator. Any type implementing this trait
/// can be used to compress vector data during the data generation process.
///
/// # Type Parameters
/// - `T`: The data type of the input vectors. Must impl Copy + Into<f32> + Pod + Sync
///   so that the [`QuantDataGenerator`] can parallelize computation, call compress_into and read from data file.
///
/// # Associated Types
/// - [`CompressorContext`]: An overloadable type that provides initialization parameters for the compressor
///
/// # Methods
/// - `new_at_stage`: Constructs a new compressor instance with the provided context for the stage of compression.
/// - `compress`: Compresses a batch of vectors into the output buffer.
/// - `compressed_bytes`: Returns the size in bytes of each compressed vector
pub trait QuantCompressor<T>: Sized + Sync
where
    T: VectorRepr,
{
    type CompressorContext;

    fn new_at_stage(stage: CompressionStage, context: &Self::CompressorContext) -> ANNResult<Self>;
    fn compress(&self, vector: MatrixView<f32>, output: MutMatrixView<u8>) -> ANNResult<()>;
    fn compressed_bytes(&self) -> usize;
}
