// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Multi-vector support for MinMax quantized vectors.
//!
//! This module provides wrappers for storing and computing distances between
//! multi-vector representations that use MinMax quantization.
//!
//! # Overview
//!
//! - [`MinMaxMeta`]: Metadata type for MinMax quantized multi-vectors
//! - [`MinMaxKernel`]: Kernel for computing MaxSim/Chamfer distance with MinMax vectors
//! - [`FullQueryMatRef`]: Multi-vector reference for full-precision query, enabling
//!   distances with quantized document.
//!
//! # Example
//!
//! ```rust
//! use std::num::NonZeroUsize;
//! use diskann_quantization::{
//!     algorithms::{transforms::NullTransform, Transform},
//!     minmax::{FullQueryMatRef, FullQueryMeta, MinMaxMeta, MinMaxQuantizer},
//!     multi_vector::{
//!         distance::{Chamfer, MaxSim, QueryMatRef},
//!         Defaulted, Mat, MatRef, SliceMatRepr, Standard,
//!     },
//!     num::Positive,
//!     CompressInto,
//! };
//! use diskann_utils::{Reborrow, ReborrowMut};
//! use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! const NBITS: usize = 8;
//! let dim = 4;
//! let num_query_vectors = 2;
//! let num_doc_vectors = 3;
//!
//! // Create a MinMax quantizer (using NullTransform for simplicity)
//! let quantizer = MinMaxQuantizer::new(
//!     Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
//!     Positive::new(1.0)?,
//! );
//!
//! // Full-precision query multi-vector (2 vectors × 4 dimensions)
//! let query_data: Vec<f32> = vec![
//!     1.0, 0.0, 0.0, 0.0,  // query vector 0
//!     0.0, 1.0, 0.0, 0.0,  // query vector 1
//! ];
//! let query_input = MatRef::new(
//!     Standard::new(num_query_vectors, dim)?, &query_data
//! )?;
//!
//! // Compress query into FullQueryMatRef (keeps f32, adds precomputed metadata)
//! let repr = SliceMatRepr::<f32, FullQueryMeta>::new(num_query_vectors, dim)?;
//! let mut query_mat = Mat::new(repr, Defaulted)?;
//! quantizer.compress_into(query_input, query_mat.reborrow_mut())?;
//!
//! // Full-precision document multi-vector (3 vectors × 4 dimensions)
//! let doc_data: Vec<f32> = vec![
//!     0.5, 0.5, 0.0, 0.0,  // doc vector 0
//!     1.0, 0.0, 0.0, 0.0,  // doc vector 1
//!     0.0, 0.0, 1.0, 0.0,  // doc vector 2
//! ];
//! let doc_input = MatRef::new(
//!     Standard::new(num_doc_vectors, dim)?, &doc_data
//! )?;
//!
//! // Quantize the document multi-vector
//! let meta = MinMaxMeta::new(num_doc_vectors, dim);
//! let mut doc_out: Mat<MinMaxMeta<NBITS>> = Mat::new(meta, Defaulted)?;
//! quantizer.compress_into(doc_input, doc_out.reborrow_mut())?;
//!
//! // Wrap query as FullQueryMatRef for asymmetric distance
//! let query: FullQueryMatRef<'_> = query_mat.reborrow().into();
//! let doc = doc_out.reborrow();
//!
//! // Compute MaxSim: per-query-vector max similarities
//! let mut scores = vec![0.0f32; num_query_vectors];
//! MaxSim::new(&mut scores)?.evaluate(query, doc);
//!
//! // Compute Chamfer distance (sum of MaxSim scores)
//! let chamfer = Chamfer::evaluate(query, doc);
//! # Ok(())
//! # }
//! ```

mod max_sim;
mod meta;

pub use max_sim::{FullQueryMatRef, MinMaxKernel};
pub use meta::MinMaxMeta;
