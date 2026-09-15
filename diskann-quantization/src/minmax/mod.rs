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
//!
//! ## Multi-vector Support
//!
//! [`MinMaxMeta`] and [`MinMaxKernel`] support storing and computing distances between
//! multi-vector representations that use MinMax quantization.
//!
//! ```rust
//! use std::num::NonZeroUsize;
//! use diskann_quantization::{
//!     algorithms::{transforms::NullTransform, Transform},
//!     minmax::{MinMaxMeta, MinMaxQuantizer},
//!     multi_vector::{
//!         distance::{Chamfer, MaxSim, QueryMatRef},
//!         Defaulted, Mat, MatRef, Standard,
//!     },
//!     num::Positive,
//!     CompressInto,
//! };
//! use diskann_utils::{Reborrow, ReborrowMut};
//! use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
//!
//! const NBITS: usize = 8;
//! let dim = 4;
//! let num_query_vectors = 2;
//! let num_doc_vectors = 3;
//!
//! // Create a MinMax quantizer (using NullTransform for simplicity)
//! let quantizer = MinMaxQuantizer::new(
//!     Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
//!     Positive::new(1.0).unwrap(),
//! );
//!
//! // Full-precision query multi-vector (2 vectors × 4 dimensions)
//! let query_data: Vec<f32> = vec![
//!     1.0, 0.0, 0.0, 0.0,  // query vector 0
//!     0.0, 1.0, 0.0, 0.0,  // query vector 1
//! ];
//! let query_input = MatRef::new(
//!     Standard::new(num_query_vectors, dim).unwrap(), &query_data
//! ).unwrap();
//!
//! // Full-precision document multi-vector (3 vectors × 4 dimensions)
//! let doc_data: Vec<f32> = vec![
//!     0.5, 0.5, 0.0, 0.0,  // doc vector 0
//!     1.0, 0.0, 0.0, 0.0,  // doc vector 1
//!     0.0, 0.0, 1.0, 0.0,  // doc vector 2
//! ];
//! let doc_input = MatRef::new(
//!     Standard::new(num_doc_vectors, dim).unwrap(), &doc_data
//! ).unwrap();
//!
//! // Create owned matrices for quantized output using Mat::new
//! let mut query_out: Mat<MinMaxMeta<NBITS>> =
//!     Mat::new(MinMaxMeta::new(num_query_vectors, dim), Defaulted).unwrap();
//! let mut doc_out: Mat<MinMaxMeta<NBITS>> =
//!     Mat::new(MinMaxMeta::new(num_doc_vectors, dim), Defaulted).unwrap();
//!
//! // Quantize both multi-vectors
//! quantizer.compress_into(query_input, query_out.reborrow_mut()).unwrap();
//! quantizer.compress_into(doc_input, doc_out.reborrow_mut()).unwrap();
//!
//! // Get immutable views via reborrow for distance computation
//! let query_mv = query_out.reborrow();
//! let doc_mv = doc_out.reborrow();
//!
//! // Compute MaxSim: per-query-vector max similarities
//! let mut scores = vec![0.0f32; num_query_vectors];
//! MaxSim::new(&mut scores).evaluate(query_mv.into(), doc_mv);
//! // scores[i] = min over all doc vectors of distance(query[i], doc[j])
//!
//! // Compute Chamfer distance (sum of MaxSim scores)
//! let chamfer = Chamfer::evaluate(query_mv.into(), doc_mv);
//! ```
mod multi;
mod quantizer;
mod recompress;
mod vectors;

/////////////
// Exports //
/////////////

pub use multi::{MinMaxErase, MinMaxKernel, MinMaxMaxSimKernel, MinMaxMeta, build_minmax_max_sim};
pub use quantizer::{L2Loss, MinMaxQuantizer};
pub use recompress::{RecompressError, Recompressor};
pub use vectors::{
    Data, DataMutRef, DataRef, DecompressError, FullQuery, FullQueryMeta, FullQueryMut,
    FullQueryRef, MetaParseError, MinMaxCompensation, MinMaxCosine, MinMaxCosineNormalized,
    MinMaxIP, MinMaxL2Squared,
};
