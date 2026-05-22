// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Multi-vector distance computation.
//!
//! Use [`build_max_sim`] to construct a kernel for a given ISA and query
//! matrix. The kernel implements [`MaxSimKernel<T>`] and computes per-query
//! max-similarity scores against document matrices.
//!
//! [`Chamfer`] and [`MaxSim`] provide the non-factory (reference) path.
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
//! let query_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
//! let query: QueryMatRef<_> = MatRef::new(
//!     Standard::new(2, 3).unwrap(),
//!     &query_data,
//! ).unwrap().into();
//!
//! let doc_data = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0];
//! let doc = MatRef::new(
//!     Standard::new(2, 3).unwrap(),
//!     &doc_data,
//! ).unwrap();
//!
//! let chamfer_dist = Chamfer::evaluate(query, doc);
//!
//! let mut scores = vec![0.0f32; 2];
//! let mut max_sim = MaxSim::new(&mut scores).unwrap();
//! max_sim.evaluate(query, doc);
//! ```

mod factory;
mod fallback;
mod isa;
mod kernel;
mod kernels;
mod max_sim;

pub use factory::{MaxSimElement, build_max_sim};
pub use fallback::QueryMatRef;
pub use isa::{MaxSimIsa, NotSupported};
pub use kernel::{BoxErase, Erase, MaxSimKernel};
pub use max_sim::{Chamfer, MaxSim, MaxSimError};
