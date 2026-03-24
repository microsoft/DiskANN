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
//!
//! # Example
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
//! MaxSim::new(&mut scores).unwrap().evaluate(query_mv.into(), doc_mv);
//! // scores[i] = min over all doc vectors of distance(query[i], doc[j])
//!
//! // Compute Chamfer distance (sum of MaxSim scores)
//! let chamfer = Chamfer::evaluate(query_mv.into(), doc_mv);
//! ```

mod max_sim;
mod meta;

pub use max_sim::MinMaxKernel;
pub use meta::MinMaxMeta;
