// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Distance computation for multi-vector representations.
//!
//! Provides asymmetric distance primitives for multi-vector search:
//!
//! - [`MaxSim`]: Per-query-vector maximum similarities.
//! - [`Chamfer`]: Sum of MaxSim scores (asymmetric Chamfer distance).
//!
//! Both are currently implemented using a simple double-loop kernel over
//! [`InnerProduct`](diskann_vector::distance::InnerProduct).
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
//!     Standard::new(2, 3),
//!     &query_data,
//! ).unwrap().into();
//!
//! // Doc: 2 vectors of dim 3
//! let doc_data = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0];
//! let doc = MatRef::new(
//!     Standard::new(2, 3),
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
//! // scores[0] = -1.0 (query[0] matches doc[0])
//! // scores[1] =  0.0 (query[1] has no good match)
//! ```

mod max_sim;
mod simple;

pub use max_sim::{Chamfer, MaxSim, MaxSimError};
pub use simple::QueryMatRef;
