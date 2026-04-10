// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Distance computation for multi-vector representations.
//!
//! Provides asymmetric distance primitives for multi-vector search:
//!
//! - [`MaxSim`]: Per-query-vector maximum similarities.
//! - [`Chamfer`]: Sum of MaxSim scores (asymmetric Chamfer distance).
//! - [`QueryComputer`]: Architecture-dispatched query computer backed by
//!   SIMD-accelerated block-transposed kernels.
//!
//! The fallback path uses a double-loop kernel over
//! [`InnerProduct`](diskann_vector::distance::InnerProduct). The optimised
//! path (via [`QueryComputer`]) uses block-transposed layout with
//! cache-tiled SIMD micro-kernels.
//!
//! # Example
//!
//! ```
//! use diskann_quantization::multi_vector::{
//!     distance::{Chamfer, MaxSim, QueryMatRef},
//!     MatRef, Standard,
//! };
//! use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
//!
//! // Query: 2 vectors of dim 3 (wrapped as QueryMatRef)
//! let query_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
//! let query: QueryMatRef<_> = MatRef::new(
//!     Standard::new(2, 3).unwrap(),
//!     &query_data,
//! ).unwrap().into();
//!
//! // Doc: 2 vectors of dim 3
//! let doc_data = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0];
//! let doc = MatRef::new(
//!     Standard::new(2, 3).unwrap(),
//!     &doc_data,
//! ).unwrap();
//!
//! // Chamfer distance (sum of max similarities)
//! let chamfer_dist = Chamfer::evaluate(query, doc);
//!
//! // MaxSim (per-query-vector scores)
//! let mut scores = vec![0.0f32; 2];
//! let mut max_sim = MaxSim::new(&mut scores).unwrap();
//! max_sim.evaluate(query, doc);
//! // scores[0] = -1.0 (query[0] matches doc[0]: negated max inner product)
//! // scores[1] =  0.0 (query[1] has no good match: max IP was 0)
//! ```

mod fallback;
mod kernels;
mod max_sim;
mod query_computer;

pub use fallback::QueryMatRef;
pub use max_sim::{Chamfer, MaxSim, MaxSimError};
pub use query_computer::QueryComputer;
