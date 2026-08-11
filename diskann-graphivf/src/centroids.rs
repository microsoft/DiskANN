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
        Config,
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

/// Construct the graph [`Config`] and provider parameters shared by the
/// immutable ([`build`]) and mutable ([`build_mutable`]) centroid graphs.
///
/// `dim` is the centroid dimension and `capacity` the number of id slots to
/// reserve — equal to the centroid count for an immutable graph, larger for a
/// mutable one that must accommodate later insertions.
fn centroid_graph_config(
    graph: &GraphParams,
    dim: usize,
    capacity: usize,
    metric: VectorMetric,
) -> Result<(Config, DefaultProviderParameters)> {
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
    Ok((config, params))
}

/// Deterministically pick a centroid to use as the graph's frozen start point.
///
/// The mean of all centroids is a poor start point under inner product: it has
/// the smallest norm, so no centroid selects it as an IP-nearest neighbor and it
/// ends up with zero out-edges, causing navigation to dead-end. A real centroid
/// is always well connected. The RNG is seeded deterministically for
/// reproducibility.
fn start_point_index(num_clusters: usize) -> usize {
    let mut rng = StdRng::seed_from_u64(num_clusters as u64);
    rng.random_range(0..num_clusters)
}

/// Build an immutable in-memory full-precision graph over `centroids` (row `i`
/// is centroid `i`, returned as internal/external id `i`), sized exactly to its
/// centroids.
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
    let (config, params) = centroid_graph_config(graph, centroids.ncols(), num_clusters, metric)?;
    let index = new_index::<f32, _>(config, params, NoDeletes)?;

    let start = start_point_index(num_clusters);
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
    if capacity < num_clusters {
        return Err(GraphIvfError::invalid(format!(
            "graph capacity ({capacity}) is smaller than the initial centroid count ({num_clusters})"
        )));
    }
    let (config, params) = centroid_graph_config(graph, centroids.ncols(), capacity, metric)?;
    let index = new_index::<f32, _>(config, params, TableBasedDeletes)?;

    let start = start_point_index(num_clusters);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(data: Vec<f32>, nrows: usize, ncols: usize) -> Matrix<f32> {
        Matrix::try_from(data.into_boxed_slice(), nrows, ncols).unwrap()
    }

    /// Four centroids at the corners of a 10x10 square, with spare capacity.
    fn square(capacity: usize) -> (MutableCentroidGraph, Runtime) {
        let cents = mat(vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0], 4, 2);
        let graph = build_mutable(
            cents,
            &GraphParams::default(),
            2,
            capacity,
            VectorMetric::L2,
        )
        .unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        (graph, rt)
    }

    fn nearest(graph: &MutableCentroidGraph, rt: &Runtime, q: &[f32], k: usize) -> Vec<u32> {
        let mut ids = vec![0u32; k];
        let mut dist = vec![0.0f32; k];
        let n = search_mut(graph, rt, q, k.max(16), &mut ids, &mut dist).unwrap();
        ids.truncate(n);
        ids
    }

    /// A soft-deleted centroid is no longer returned by search, and a fresh id
    /// can be inserted afterwards.
    #[test]
    fn delete_then_insert_fresh_id() {
        let (graph, rt) = square(8);
        delete_centroid(&graph, &rt, 1).unwrap();
        assert!(!nearest(&graph, &rt, &[10.0, 0.0], 4).contains(&1));

        insert_centroid(&graph, &rt, 4, &[9.0, 1.0]).unwrap();
        assert!(nearest(&graph, &rt, &[10.0, 0.0], 4).contains(&4));
    }

    /// Centroid ids are **not** reusable: re-inserting the id of a soft-deleted
    /// centroid silently appears to succeed, but the centroid is never returned
    /// by a search — the tombstone wins.
    ///
    /// This is why [`CentroidTable`] hands out ids monotonically and
    /// `centroid_capacity` must cover every id ever allocated, not just the
    /// peak live count. Recycling ids would silently lose clusters.
    ///
    /// [`CentroidTable`]: crate::online::OnlineClusterer
    #[test]
    fn deleted_centroid_ids_are_not_reusable() {
        let (graph, rt) = square(8);
        delete_centroid(&graph, &rt, 1).unwrap();

        // Re-inserting a retired id reports success ...
        insert_centroid(&graph, &rt, 1, &[-5.0, -5.0]).unwrap();

        // ... but the centroid is unreachable: a query sitting exactly on it
        // returns every *other* centroid instead.
        let found = nearest(&graph, &rt, &[-5.0, -5.0], 4);
        assert!(
            !found.contains(&1),
            "id reuse unexpectedly worked; revisit the monotonic id allocation \
             in OnlineClusterer (found {found:?})"
        );
        assert!(
            !found.is_empty(),
            "the surviving centroids remain reachable"
        );
    }
}
