/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Distance computation for multi-vector representations.
//!
//! The fallback path uses a double-loop kernel over
//! [`InnerProduct`](diskann_vector::distance::InnerProduct); the factory
//! returns cache-tiled SIMD kernels selected by [`MaxSimIsa`].
//!
//! # Example
//!
//! ```
//! use diskann_quantization::multi_vector::{
//!     distance::{Chamfer, MaxSim, Query},
//! };
//! use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
//! use diskann_utils::views::rowmajor::{Ref as MatRef};
//!
//! // Query: 2 vectors of dim 3 (wrapped as Query)
//! let query_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
//! let query = Query(MatRef::try_from_data(&query_data, 2, 3).unwrap());
//!
//! // Doc: 2 vectors of dim 3
//! let doc_data = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0];
//! let doc = MatRef::try_from_data(&doc_data, 2, 3).unwrap();
//!
//! // Chamfer distance (sum of max similarities)
//! let chamfer_dist = Chamfer::evaluate(query, doc);
//!
//! // MaxSim (per-query-vector scores)
//! let mut scores = vec![0.0f32; 2];
//! let mut max_sim = MaxSim::new(&mut scores);
//! max_sim.evaluate(query, doc);
//! // scores[0] = -1.0 (query[0] matches doc[0]: negated max inner product)
//! // scores[1] =  0.0 (query[1] has no good match: max IP was 0)
//! ```

mod factory;
mod fallback;
mod isa;
mod kernel;
mod max_sim;
mod projected_eigen;

pub use factory::{MaxSimElement, build_max_sim};
pub use fallback::Query;
pub use isa::{MaxSimIsa, NotSupported};
pub use kernel::{BoxErase, Erase, MaxSimKernel};
pub use max_sim::{Chamfer, MaxSim, MaxSimError};
pub use projected_eigen::ProjectedEigen;
