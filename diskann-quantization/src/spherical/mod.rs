/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod iface;
mod pairwise;
pub(crate) mod quantizer;
mod vectors;

#[doc(hidden)]
#[cfg(feature = "codegen")]
pub mod __codegen;

/////////////
// Exports //
/////////////

pub use pairwise::{Pairwise1Bit, Pairwise1BitScratch, PairwiseSIMDSchema};

pub use quantizer::{CompressionError, PreScale, SphericalQuantizer, TrainError};
#[cfg(feature = "flatbuffers")]
pub use vectors::InvalidMetric;
pub use vectors::{
    CompensatedCosine, CompensatedIP, CompensatedSquaredL2, Data, DataMeta, DataMetaError,
    DataMetaF32, DataMut, DataRef, FullQuery, FullQueryMeta, FullQueryMut, FullQueryRef, Query,
    QueryMeta, QueryMut, QueryRef, SupportedMetric, UnsupportedMetric,
};
