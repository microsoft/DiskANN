/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{utils::VectorRepr, ANNResult};
use diskann_utils::views::{MatrixView, MutMatrixView};

/// [`QuantCompressor`] defines the interface for quantizers used by
/// [`super::QuantDataGenerator`].
///
/// This trait serves as a general wrapper for different quantizers, allowing them to be
/// used interchangeably with [`super::QuantDataGenerator`]. Any type implementing this trait
/// can be used to compress vector data during the data generation process.
///
/// # Type Parameters
/// - `T`: The data type of the input vectors. Must implement `Copy + Into<f32> + Pod + Sync`
///   so that [`super::QuantDataGenerator`] can parallelize computation, call `compress_into`,
///   and read from the data file.
///
/// # Associated Types
/// - [`Self::CompressorContext`]: An overloadable type that provides initialization parameters
///   for the compressor.
/// - [`Self::Prepared`]: The ready-to-use compressor produced by [`Self::prepare`].
///
/// # Methods
/// - `new`: Constructs a new compressor instance with the provided context.
/// - `prepare`: Returns a ready-to-use compressor.
pub trait QuantCompressor<'a, T>: Sized
where
    T: VectorRepr,
{
    type CompressorContext: 'a;

    type Prepared: PreparedCompressor + Sync;

    fn new(context: &'a Self::CompressorContext) -> Self;

    /// Returns an error if preparation fails.
    fn prepare(&self) -> ANNResult<Self::Prepared>;
}

/// A quantizer that is ready to compress vectors.
///
/// # Methods
/// - `compress`: Compresses a batch of vectors into the output buffer.
/// - `compressed_bytes`: Returns the size in bytes of each compressed vector.
pub trait PreparedCompressor {
    fn compress(&self, vector: MatrixView<f32>, output: MutMatrixView<u8>) -> ANNResult<()>;
    fn compressed_bytes(&self) -> usize;
}
