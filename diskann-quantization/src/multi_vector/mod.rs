// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Multi-vector matrix types and distance functions.
//!
//! Row-major matrix abstractions for multi-vector representations, where each
//! entity is encoded as multiple embedding vectors (e.g., per-token embeddings).
//!
//! # Core Types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`Mat`] | Owning matrix that manages its own memory |
//! | [`MatRef`] | Immutable borrowed view  |
//! | [`MatMut`] | Mutable borrowed view |
//! | [`Repr`] | Trait defining row layout (e.g., [`Standard`]) |
//! | [`QueryMatRef`] | Query wrapper for asymmetric distances |
//! | [`MaxSim`] | Per-query-vector max similarity computation |
//! | [`Chamfer`] | Asymmetric Chamfer distance (sum of MaxSim) |
//!
//! # Example
//!
//! ```
//! use diskann_quantization::multi_vector::{
//!     distance::QueryMatRef,
//!     Chamfer, Mat, MatMut, MatRef, MaxSim, Standard,
//! };
//! use diskann_utils::ReborrowMut;
//! use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
//!
//! // Create an owned matrix (2 vectors, dim 3, initialized to 0.0)
//! let mut owned = Mat::new(Standard::new(2, 3).unwrap(), 0.0f32).unwrap();
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
//!     Standard::new(2, 2).unwrap(),
//!     &query_data,
//! ).unwrap().into();
//! let doc = MatRef::new(Standard::new(2, 2).unwrap(), &doc_data).unwrap();
//!
//! // Chamfer distance (sum of max similarities)
//! let distance = Chamfer::evaluate(query, doc);
//! assert_eq!(distance, -2.0); // Perfect match: -1.0 per vector
//!
//! // MaxSim (per-query-vector scores)
//! let mut scores = vec![0.0f32; 2];
//! let mut max_sim = MaxSim::new(&mut scores).unwrap();
//! max_sim.evaluate(query, doc);
//! assert_eq!(scores[0], -1.0);
//! assert_eq!(scores[1], -1.0);
//! ```

pub mod distance;
pub(crate) mod matrix;

pub use distance::{Chamfer, MaxSim, MaxSimError, QueryMatRef};
pub use matrix::{
    Defaulted, LayoutError, Mat, MatMut, MatRef, NewCloned, NewMut, NewOwned, NewRef, Overflow,
    Repr, ReprMut, ReprOwned, SliceError, Standard,
};
