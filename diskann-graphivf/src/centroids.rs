/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! In-memory full-precision DiskANN graph over the cluster centroids.
//!
//! Thin wrappers around the `diskann` / `diskann-providers` in-memory index that
//! build a graph over a set of centroids and run k-NN over it.

use std::sync::Arc;

use diskann::{
    graph::{
        config::{Builder, MaxDegree},
        search::Knn,
        search_output_buffer::{IdDistance, SearchOutputBuffer},
        strategy::FullPrecision,
        AdjacencyList,
    },
    provider::{DefaultContext, Delete},
    utils::ONE,
    ANNError,
};
use diskann_providers::{
    index::diskann_async::{new_index, MemoryIndex},
    model::graph::provider::async_::{
        common::{NoDeletes, TableBasedDeletes},
        inmem::{DefaultProviderParameters, SetStartPoints},
        TableDeleteProviderAsync,
    },
};
use diskann_utils::views::Matrix;
use diskann_vector::distance::Metric as VectorMetric;
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokio::runtime::Runtime;

use crate::{params::GraphParams, GraphIvfError, Result};

/// Build an in-memory full-precision graph over `centroids` (row `i` is centroid
/// `i`, returned as internal/external id `i`).
///
/// `metric` is the distance used for graph construction and navigation. Build-
/// time callers (centroid assignment) pass [`VectorMetric::L2`]; the search-time
/// graph (rebuilt on load) may use a different metric — e.g.
/// [`VectorMetric::InnerProduct`] for a MIPS index — so queries navigate to the
/// centroids that maximize that search metric. The centroid data is assumed to
/// already be in the internal representation expected by the metric (e.g.
/// L2-normalized for cosine).
pub(crate) fn build(
    centroids: Matrix<f32>,
    graph: &GraphParams,
    num_threads: usize,
    metric: VectorMetric,
) -> Result<MemoryIndex<f32>> {
    let num_clusters = centroids.nrows();
    let dim = centroids.ncols();

    let config = Builder::new_with(
        graph.degree,
        MaxDegree::slack(graph.slack),
        graph.l_build,
        metric.into(),
        |b| {
            b.alpha(graph.alpha);
        },
    )
    .build()
    .map_err(ANNError::from)?;

    let params = DefaultProviderParameters {
        max_points: num_clusters,
        frozen_points: ONE,
        dim,
        metric,
        prefetch_lookahead: None,
        prefetch_cache_line_level: None,
        max_degree: config.max_degree_u32().get(),
    };

    let index = new_index::<f32, _>(config, params, NoDeletes)?;

    // Pick a random centroid as the graph's start point. The mean of all
    // centroids is a poor start point under inner product: it has the smallest
    // norm, so no centroid selects it as an IP-nearest neighbor and it ends up
    // with zero out-edges, causing navigation to dead-end. A real centroid is
    // always well connected. The RNG is seeded deterministically for
    // reproducibility.
    let mut rng = StdRng::seed_from_u64(num_clusters as u64);
    let start = rng.random_range(0..num_clusters);
    index
        .provider()
        .set_start_points(std::iter::once(centroids.row(start)))?;

    let ids: Arc<[u32]> = (0..num_clusters as u32).collect();
    let batch = Arc::new(centroids);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_threads)
        .build()
        .map_err(ANNError::from)?;

    runtime.block_on(index.multi_insert::<_, Matrix<f32>>(
        FullPrecision,
        &DefaultContext,
        batch,
        ids,
    ))?;

    Ok(index)
}

/// Run a k-NN search over the centroid graph, writing the resulting centroid ids
/// and distances into `ids_out` / `dist_out`. Returns the number of results
/// written (`<= ids_out.len()`).
///
/// `ids_out` and `dist_out` must have the same length, which is the requested
/// `k`. `l` is the centroid-graph search-list size and must be `>= k`.
pub(crate) fn search(
    index: &MemoryIndex<f32>,
    runtime: &Runtime,
    query: &[f32],
    l: usize,
    ids_out: &mut [u32],
    dist_out: &mut [f32],
) -> Result<usize> {
    debug_assert_eq!(ids_out.len(), dist_out.len());
    let k = ids_out.len();
    let knn = Knn::new(k, l, None).map_err(ANNError::from)?;

    let mut buffer = IdDistance::new(ids_out, dist_out);
    runtime.block_on(index.search(knn, &FullPrecision, &DefaultContext, query, &mut buffer))?;

    Ok(buffer.current_len())
}

/// A centroid graph that supports incremental insertion and soft deletion of
/// centroids (used by the online split/reassign clusterer).
///
/// Unlike [`build`], which produces an immutable graph sized exactly to its
/// centroids, this graph is built with spare capacity and a delete table so new
/// centroids can be inserted and split centroids soft-deleted in place. Soft
/// deletes leave the slot occupied (no reuse), so the capacity must cover every
/// id ever allocated over the clusterer's lifetime.
pub(crate) type MutableCentroidGraph = MemoryIndex<f32, TableDeleteProviderAsync>;

/// Build a [`MutableCentroidGraph`] over `centroids` with room for `capacity`
/// total centroid ids.
///
/// The initial centroids receive ids `0..centroids.nrows()`; ids
/// `centroids.nrows()..capacity` are free for later [`insert_centroid`] calls.
/// `metric` is the navigation/assignment metric (callers pass
/// [`VectorMetric::L2`], matching batch clustering).
///
/// # Panics / Errors
///
/// Returns an error if `capacity < centroids.nrows()` or graph construction
/// fails.
pub(crate) fn build_mutable(
    centroids: Matrix<f32>,
    graph: &GraphParams,
    num_threads: usize,
    capacity: usize,
    metric: VectorMetric,
) -> Result<MutableCentroidGraph> {
    let num_clusters = centroids.nrows();
    let dim = centroids.ncols();
    if capacity < num_clusters {
        return Err(GraphIvfError::invalid(format!(
            "graph capacity ({capacity}) is smaller than the initial centroid count ({num_clusters})"
        )));
    }

    let config = Builder::new_with(
        graph.degree,
        MaxDegree::slack(graph.slack),
        graph.l_build,
        metric.into(),
        |b| {
            b.alpha(graph.alpha);
        },
    )
    .build()
    .map_err(ANNError::from)?;

    let params = DefaultProviderParameters {
        max_points: capacity,
        frozen_points: ONE,
        dim,
        metric,
        prefetch_lookahead: None,
        prefetch_cache_line_level: None,
        max_degree: config.max_degree_u32().get(),
    };

    let index = new_index::<f32, _>(config, params, TableBasedDeletes)?;

    // Use a real centroid as the frozen start point (see `build` for rationale).
    let mut rng = StdRng::seed_from_u64(num_clusters as u64);
    let start = rng.random_range(0..num_clusters);
    index
        .provider()
        .set_start_points(std::iter::once(centroids.row(start)))?;

    let ids: Arc<[u32]> = (0..num_clusters as u32).collect();
    let batch = Arc::new(centroids);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_threads)
        .build()
        .map_err(ANNError::from)?;

    runtime.block_on(index.multi_insert::<_, Matrix<f32>>(
        FullPrecision,
        &DefaultContext,
        batch,
        ids,
    ))?;

    Ok(index)
}

/// Insert a new centroid `vec` under external id `id` into a
/// [`MutableCentroidGraph`]. `id` must be unused and within the graph's
/// capacity.
pub(crate) fn insert_centroid(
    index: &MutableCentroidGraph,
    runtime: &Runtime,
    id: u32,
    vec: &[f32],
) -> Result<()> {
    runtime.block_on(index.insert(&FullPrecision, &DefaultContext, &id, vec))?;
    Ok(())
}

/// Soft-delete the centroid with external id `id` from a
/// [`MutableCentroidGraph`]. The centroid is no longer returned by
/// [`search_mut`], though its slot remains occupied.
pub(crate) fn delete_centroid(
    index: &MutableCentroidGraph,
    runtime: &Runtime,
    id: u32,
) -> Result<()> {
    runtime.block_on(index.provider().delete(&DefaultContext, &id))?;
    Ok(())
}

/// Overwrite `out` with the current graph neighbors (out-edges) of centroid
/// `id`. Neighbors may include soft-deleted centroids; callers filter those out.
pub(crate) fn neighbors(index: &MutableCentroidGraph, id: u32, out: &mut Vec<u32>) -> Result<()> {
    let mut adj: AdjacencyList<u32> = AdjacencyList::new();
    index
        .provider()
        .neighbors()
        .get_neighbors_sync(id as usize, &mut adj)?;
    out.clear();
    out.extend_from_slice(&adj);
    Ok(())
}

/// Like [`search`], but over a [`MutableCentroidGraph`]. Soft-deleted centroids
/// are skipped automatically.
pub(crate) fn search_mut(
    index: &MutableCentroidGraph,
    runtime: &Runtime,
    query: &[f32],
    l: usize,
    ids_out: &mut [u32],
    dist_out: &mut [f32],
) -> Result<usize> {
    debug_assert_eq!(ids_out.len(), dist_out.len());
    let k = ids_out.len();
    let knn = Knn::new(k, l, None).map_err(ANNError::from)?;

    let mut buffer = IdDistance::new(ids_out, dist_out);
    runtime.block_on(index.search(knn, &FullPrecision, &DefaultContext, query, &mut buffer))?;

    Ok(buffer.current_len())
}
