// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

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
//! let mut max_sim = MaxSim::new(&mut scores);
//! max_sim.evaluate(query, doc);
//! // scores[0] = -1.0 (query[0] matches doc[0]: negated max inner product)
//! // scores[1] =  0.0 (query[1] has no good match: max IP was 0)
//! ```

mod factory;
mod fallback;
mod isa;
mod kernel;
mod kernels;
mod max_sim;
mod projected_eigen;

mod temp;

pub use factory::{MaxSimElement, build_max_sim};
pub use fallback::QueryMatRef;
pub use isa::{MaxSimIsa, NotSupported};
pub use kernel::{BoxErase, Erase, MaxSimKernel};
pub use max_sim::{Chamfer, MaxSim, MaxSimError};
pub use projected_eigen::ProjectedEigen;

/// Standalone POC entry for the 4-bit MinMax *staged* multi-vector MaxSim kernel
/// (V3/AVX2 only) — not yet unified into [`MaxSimIsa`]/[`build_max_sim`].
#[cfg(target_arch = "x86_64")]
pub use kernels::{QuantStagedDocs, QuantStagedQuery};

/// Standalone POC entry for the coarse `Tiler`/`Tile` rebuild (V3/AVX2 only) — the
/// 4-bit MinMax kernel where tilers own movement and the kernel is pinned on the
/// A/B panel types. The i8 A/B baseline is `QuantStaged*`.
#[cfg(target_arch = "x86_64")]
pub use kernels::{QuantTiledDocs, QuantTiledQuery};

/// Standalone POC entry for the coarse tiler's **f16** path (V3/AVX2 only) — each
/// tile is widened f16→f32 into a reused buffer, then an f32 store kernel + identity
/// postprocess reuse the same tiled pipeline.
#[cfg(target_arch = "x86_64")]
pub use kernels::{QuantTiledF16Docs, QuantTiledF16Query};
