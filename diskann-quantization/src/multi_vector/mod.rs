/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Multi-vector matrix types and distance functions.
//!
//! Row-major matrix abstractions for multi-vector representations, where each
//! entity is encoded as multiple embedding vectors (e.g., per-token embeddings).
//!
//! # Example
//!
//! ```
//! use diskann_quantization::multi_vector::{
//!     distance::QueryMatRef,
//!     Chamfer, Mat, MatMut, MatRef, MaxSim, RowMajor,
//! };
//! use diskann_utils::ReborrowMut;
//! use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
//!
//! // Create an owned matrix (2 vectors, dim 3, initialized to 0.0)
//! let mut owned = Mat::new(RowMajor::new(2, 3).unwrap(), 0.0f32).unwrap();
//! assert_eq!(owned.num_vectors(), 2);
//!
//! // Modify via mutable view
//! let mut view = owned.reborrow_mut();
//! if let Some(row) = view.get_row_mut(0) {
//!    row[0] = 1.0;
//! }
//!
//! // Create views from slices
//! let query_data = [1.0f32, 0.0, 0.0, 1.0];
//! let doc_data = [1.0f32, 0.0, 0.0, 1.0];
//!
//! // Wrap query as QueryMatRef for type-safe asymmetric distance
//! let query: QueryMatRef<_> = MatRef::new(
//!     RowMajor::new(2, 2).unwrap(),
//!     &query_data,
//! ).unwrap().into();
//! let doc = MatRef::new(RowMajor::new(2, 2).unwrap(), &doc_data).unwrap();
//!
//! // Chamfer distance (sum of max similarities)
//! let distance = Chamfer::evaluate(query, doc);
//! assert_eq!(distance, -2.0); // Perfect match: -1.0 per vector
//!
//! // MaxSim (per-query-vector scores)
//! let mut scores = vec![0.0f32; 2];
//! let mut max_sim = MaxSim::new(&mut scores);
//! max_sim.evaluate(query, doc);
//! assert_eq!(scores[0], -1.0);
//! assert_eq!(scores[1], -1.0);
//! ```

pub mod block_transposed;
pub mod distance;
pub(crate) use diskann_utils::matrix;

pub use block_transposed::{BlockTransposed, BlockTransposedMut, BlockTransposedRef};
pub use distance::{
    BoxErase, Chamfer, Erase, MaxSim, MaxSimElement, MaxSimError, MaxSimIsa, MaxSimKernel,
    NotSupported, ProjectedEigen, QueryMatRef, build_max_sim,
};
pub use diskann_utils::matrix::{
    Defaulted, LayoutError, Mat, MatMut, MatRef, NewCloned, NewMut, NewOwned, NewRef, Overflow,
    Repr, ReprMut, ReprOwned, SliceError, RowMajor,
};
