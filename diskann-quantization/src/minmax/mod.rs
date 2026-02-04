/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # MinMax Quantization
//!
//! MinMax quantization provides memory-efficient vector compression by converting
//! floating-point values to small n-bit integers on a per-vector basis.
//!
//! ## Core Concept
//!
//! Each vector is independently quantized using the formula:
//! ```math
//! X' = round((X - s) * (2^n - 1) / c).clamp(0, 2^n - 1)
//! ```
//! where `s` is a shift value and `c` is a scaling parameter computed from the
//! range of values.
//!
//! For most bit widths (>1), given a positive scaling parameter `grid_scale : f32`,
//! these are computed as:
//! ```math
//! - m = (max_i X[i] + min_i X[i]) / 2.0
//! - w = max_i X[i] - min_i X[i]
//!
//! - s = m - w * grid_scale
//! - c = 2 * w * grid_scale
//! ```
//! For 1-bit quantization, to avoid outliers, `s` and `c` are derived differently:
//!   i) Values are first split into two groups: those below and above the mean.
//!  ii) `s` is the average of values below the mean.
//! iii) `c` is the difference between the average of values above the mean and `s`.
//!
//! This encoding is similar to scalar quantization, but, since both 's' and 'c'
//! are computed on a per-vector basis, this allows this quantization mechanism
//! to be applied in a **streaming setting**; making it qualitatively different
//! than scalar quantization.
//!
//! ## Module Components
//!
//! - [`MinMaxQuantizer`]: Handles vector encoding and decoding
//! - [`Data`]: Stores quantized vectors with compensation parameters
//! - Distance functions:
//!   - [`MinMaxIP`]: Inner product distance for quantized vectors.
//!   - [`MinMaxL2Squared`]: L2 (Euclidean) distance for quantized vectors.
//!   - [`MinMaxCosine`]: Cosine similarity for quantized vectors.
//!   - [`MinMaxCosineNormalized`]: Cosine similarity for quantized vectors assuming the
//!     original full-precision vectors were normalized.
//!
//! To reconstruct the original vector, the inverse operation is applied:
//! ```math
//! X = X' * c / (2^n - 1) + s
//! ```
mod multi;
mod quantizer;
mod recompress;
mod vectors;

/////////////
// Exports //
/////////////

pub use multi::{MinMaxKernel, MinMaxMeta};
pub use quantizer::{L2Loss, MinMaxQuantizer};
pub use recompress::{RecompressError, Recompressor};
pub use vectors::{
    Data, DataMutRef, DataRef, DecompressError, FullQuery, FullQueryMeta, MetaParseError,
    MinMaxCompensation, MinMaxCosine, MinMaxCosineNormalized, MinMaxIP, MinMaxL2Squared,
};
