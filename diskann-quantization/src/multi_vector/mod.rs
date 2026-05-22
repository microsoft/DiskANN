/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Multi-vector matrix types and distance functions.
//!
//! Row-major matrix abstractions for multi-vector representations, where each
//! entity is encoded as multiple embedding vectors (e.g., per-token embeddings).
//!
//! Use [`build_max_sim`] to construct ISA-optimized kernels for computing
//! MaxSim distances. [`Chamfer`] and [`MaxSim`] provide the reference path.
//!
//! # Example
//!
//! ```
//! use diskann_quantization::multi_vector::{
//!     distance::QueryMatRef,
//!     Chamfer, Mat, MatRef, MaxSim, Standard,
//! };
//! use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
//!
//! let query_data = [1.0f32, 0.0, 0.0, 1.0];
//! let doc_data = [1.0f32, 0.0, 0.0, 1.0];
//!
//! let query: QueryMatRef<_> = MatRef::new(
//!     Standard::new(2, 2).unwrap(),
//!     &query_data,
//! ).unwrap().into();
//! let doc = MatRef::new(Standard::new(2, 2).unwrap(), &doc_data).unwrap();
//!
//! let distance = Chamfer::evaluate(query, doc);
//! assert_eq!(distance, -2.0);
//! ```

pub mod block_transposed;
pub mod distance;
pub(crate) mod matrix;

pub use block_transposed::{BlockTransposed, BlockTransposedMut, BlockTransposedRef};
pub use distance::{
    BoxErase, Chamfer, Erase, MaxSim, MaxSimElement, MaxSimError, MaxSimIsa, MaxSimKernel,
    NotSupported, QueryMatRef, build_max_sim,
};
pub use matrix::{
    Defaulted, LayoutError, Mat, MatMut, MatRef, NewCloned, NewMut, NewOwned, NewRef, Overflow,
    Repr, ReprMut, ReprOwned, SliceError, Standard,
};
