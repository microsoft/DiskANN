/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! `diskann-graphivf` — a hybrid graph + clustered-IVF approximate nearest
//! neighbor index.
//!
//! The index has two parts:
//!
//! 1. An in-memory full-precision DiskANN graph built over a set of cluster
//!    centroids (one centroid per cluster).
//! 2. An on-disk file holding, for every cluster, the corpus vectors assigned to
//!    that cluster laid out contiguously so a single read fetches a whole list.
//!
//! Build: sample the corpus, run k-means to obtain centroids, build a graph over
//! the centroids, assign every corpus point to its nearest centroid via graph
//! search, then stream the per-cluster inverted lists to disk.
//!
//! Search: find the `nlist` nearest centroids via graph search, fetch those
//! lists from disk in one batched read, and exhaustively score the query against
//! the fetched vectors to produce the top-k.
//!
//! The inverted-list vectors can be stored in any [`VectorRepr`] element type
//! ([`GraphIvfIndex`]'s type parameter, default `f32`; [`Half`] for `f16`, and
//! `i8`/`u8` are also supported). The `f32` query is encoded into the stored
//! element type and preprocessed once into a distance scorer reused across every
//! candidate, via the shared SIMD distance kernels. The centroid graph is always
//! full-precision `f32`. Cosine similarity is implemented by L2-normalizing
//! vectors at build and query time (spherical reduction to L2).

// Retained for reference / future re-integration; not currently wired into the
// index or search path.
#[allow(dead_code)]
mod cache;
mod centroids;
mod cluster;
mod error;
mod index;
mod online;
mod params;
mod profile;
mod storage;

pub use diskann::utils::VectorRepr;
pub use diskann_vector::Half;
pub use error::{GraphIvfError, Result};
pub use index::{CentroidInit, GraphIvfIndex, Searcher};
pub use online::{BuildTelemetry, OnlineClusterer, OnlineParams, SeedStrategy, SplitEvent};
pub use params::{
    AssignMethod, BuildParams, EmptyClusterPolicy, GraphParams, Metric, SearchParams,
};
pub use profile::{BuildProfile, SearchProfile};
