/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{utils::VectorRepr, ANNResult};
use diskann_utils::views::{MatrixView, MutMatrixView};

/// A quantizer constructed once and shared across compression batches.
pub trait QuantCompressor<T>: Sized + Sync
where
    T: VectorRepr,
{
    type CompressorContext;

    /// Construct a quantizer, including any training and persistence it requires.
    ///
    /// # Errors
    /// Returns an error if construction fails.
    fn new(context: &Self::CompressorContext) -> ANNResult<Self>;

    fn compress(&self, vector: MatrixView<f32>, output: MutMatrixView<u8>) -> ANNResult<()>;
    fn compressed_bytes(&self) -> usize;
}
