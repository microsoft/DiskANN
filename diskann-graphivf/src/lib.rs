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
//! Two build paths produce the same on-disk format and share the flushed-index
//! search path:
//!
//! * **Static** ([`GraphIvfIndex::build`]): sample the corpus, run k-means to
//!   obtain a fixed number of centroids, build a graph over them, assign every
//!   corpus point to its nearest centroid, then stream the lists to disk.
//! * **Online** ([`OnlineClusterer`]): accept insert and delete batches, route
//!   inserts through the centroid graph, split overfull clusters with local
//!   reassignment, and dissolve underfull clusters onto nearby survivors. The
//!   live cluster count emerges from the stream. See `ONLINE.md` for the
//!   algorithm.
//!
//! Before flush, [`OnlineClusterer::searcher`] opens an [`OnlineSearcher`] over
//! the current in-memory partition. [`OnlineSearcher::search_into`] reuses a
//! caller-owned result buffer and reports per-query [`OnlineSearchStats`].
//! Mutations prepare validation, routing, clustering, and candidate search
//! before changing live state. A failure after an irreversible graph operation
//! poisons the clusterer and later mutation, live-search, and flush requests
//! return [`GraphIvfError::Poisoned`].
//!
//! After flush, [`Searcher`] finds the `nlist` nearest centroids via graph
//! search, fetches those lists from disk in one batched read, and exhaustively
//! scores the query against the fetched vectors to produce the top-k.
//!
//! The inverted-list vectors can be stored in any [`VectorRepr`] element type
//! ([`GraphIvfIndex`]'s type parameter, default `f32`; [`Half`] for `f16`,
//! `MinMaxElement<8>` for 8-bit MinMax-quantized rows, and `i8`/`u8` are also
//! supported). Queries are supplied in that same stored type and preprocessed
//! once into a distance scorer reused across every candidate, via the shared
//! SIMD distance kernels. The centroid graph is always full-precision `f32`.
//! Cosine uses the spherical reduction to L2. A plain static build normalizes
//! its corpus copy; pre-encoded static rows and online rows are stored verbatim
//! and must be normalized beforehand. Callers supply normalized queries.

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
pub use online::{
    BuildTelemetry, MergeEvent, OnlineClusterer, OnlineSearchStats, OnlineSearcher, SeedStrategy,
    SplitEvent,
};
pub use params::{
    AssignMethod, BuildParams, EmptyClusterPolicy, GraphParams, Metric, OnlineParams, SearchParams,
    DEFAULT_CENTROID_SEARCH_ALPHA, MIN_CENTROID_SEARCH_L,
};
pub use profile::{BuildProfile, SearchProfile};
