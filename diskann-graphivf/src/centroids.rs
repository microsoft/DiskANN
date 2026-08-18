/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! In-memory structures for locating the nearest cluster centroids.
//!
//! Two of them: a DiskANN graph over the centroids (thin wrappers around the
//! `diskann` / `diskann-providers` in-memory index) and an exact scan over a
//! packed copy of them ([`exact`]). [`CentroidSource`] and its online
//! counterpart pick between the two so that no caller has to.

mod exact;

pub(crate) use exact::{DenseCentroids, ExactMetric};

use std::sync::Arc;

use diskann::{
    graph::{
        config::{Builder, MaxDegree},
        search::Knn,
        search_output_buffer::{IdDistance, SearchOutputBuffer},
        strategy::FullPrecision,
        Config, InplaceDeleteMethod,
    },
    provider::{DataProvider, DefaultAccessor, DefaultContext, Delete},
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
    utils::{ParallelIteratorInPool, RayonThreadPool},
};
use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::Metric as VectorMetric;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use tokio::runtime::Runtime;

use crate::{
    params::{CentroidRouting, GraphParams},
    GraphIvfError, Result,
};

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
    let knn = Knn::new(l, None).map_err(ANNError::from)?;

    let mut buffer = IdDistance::new(ids_out, dist_out);
    runtime.block_on(index.search(knn, &FullPrecision, &DefaultContext, query, &mut buffer))?;

    Ok(buffer.current_len())
}

/// A fixed set of centroids together with the structure used to find the
/// nearest of them.
///
/// This is what a built [`GraphIvfIndex`](crate::GraphIvfIndex) holds, and the
/// only place the [`CentroidSearch`] choice is interpreted: everything
/// downstream calls [`search`](Self::search) and never learns which mode is in
/// effect. Both variants are handle-sized, so a per-thread searcher clones one
/// rather than rebuilding it.
#[derive(Clone)]
pub(crate) enum CentroidSource {
    /// Navigate a DiskANN graph over the centroids.
    Graph {
        index: MemoryIndex<f32>,
        /// Search-list size for the per-point walk in [`assign`](Self::assign).
        assign_l: usize,
    },
    /// Scan every centroid exactly.
    Exact {
        centroids: Arc<DenseCentroids>,
        metric: ExactMetric,
    },
}

impl CentroidSource {
    /// Prepare `centroids` for lookup under `routing`.
    ///
    /// In [`CentroidRouting::Exact`] no graph is constructed — the centroids are
    /// simply packed — which is also why loading an index is markedly cheaper in
    /// that mode.
    ///
    /// `metric` is the distance centroids are ranked by. Build-time callers
    /// (centroid assignment) pass [`VectorMetric::L2`]; the search-time source
    /// may use a different metric — e.g. [`VectorMetric::InnerProduct`] for a
    /// MIPS index. The centroid data is assumed to already be in the internal
    /// representation the metric expects (e.g. L2-normalized for cosine).
    pub(crate) fn new(
        routing: CentroidRouting,
        centroids: Matrix<f32>,
        num_threads: usize,
        metric: VectorMetric,
    ) -> Result<Self> {
        match routing {
            CentroidRouting::Graph { graph, assign_l } => Ok(Self::Graph {
                index: build(centroids, &graph, num_threads, metric)?,
                assign_l,
            }),
            CentroidRouting::Exact => Ok(Self::Exact {
                centroids: Arc::new(DenseCentroids::from_matrix(&centroids)),
                metric: ExactMetric::for_navigation(metric)?,
            }),
        }
    }

    /// The `ids_out.len()` nearest centroids to `query`, ascending by distance,
    /// returning how many were written.
    ///
    /// `l` is the graph search-list size and is ignored by an exact source,
    /// which has no beam to widen.
    pub(crate) fn search(
        &self,
        runtime: &Runtime,
        query: &[f32],
        l: usize,
        ids_out: &mut [u32],
        dist_out: &mut [f32],
    ) -> Result<usize> {
        match self {
            Self::Graph { index, .. } => search(index, runtime, query, l, ids_out, dist_out),
            Self::Exact { centroids, metric } => {
                centroids.search(*metric, query, ids_out, dist_out)
            }
        }
    }

    /// Assign every row of `work` to its nearest centroid.
    ///
    /// An exact source scores the whole batch against the whole centroid set as
    /// one tiled matrix multiply; a graph source walks the graph once per point,
    /// parallelized across chunks of points, with each worker driving the
    /// (in-memory) search from its own current-thread runtime.
    pub(crate) fn assign(
        &self,
        work: MatrixView<'_, f32>,
        pool: &RayonThreadPool,
    ) -> Result<Vec<u32>> {
        let mut assignments = vec![0u32; work.nrows()];
        match self {
            Self::Exact { centroids, metric } => {
                let mut distances = vec![0.0f32; work.nrows()];
                centroids.search_batch(*metric, work, 1, &mut assignments, &mut distances, pool)?;
            }
            Self::Graph { index, assign_l } => {
                const CHUNK: usize = 256;
                assignments
                    .par_chunks_mut(CHUNK)
                    .enumerate()
                    .try_for_each_in_pool(pool.as_ref(), |(ci, out)| -> Result<()> {
                        let runtime = tokio::runtime::Builder::new_current_thread()
                            .build()
                            .map_err(ANNError::from)?;
                        let mut ids = [0u32; 1];
                        let mut dist = [0.0f32; 1];
                        for (j, slot) in out.iter_mut().enumerate() {
                            let point = work.row(ci * CHUNK + j);
                            search(index, &runtime, point, *assign_l, &mut ids, &mut dist)?;
                            *slot = ids[0];
                        }
                        Ok(())
                    })?;
            }
        }
        Ok(assignments)
    }
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

/// Soft-delete the centroid with external id `id` and repair the graph around
/// it, so the tombstoned slot is no longer reachable from live centroids.
///
/// [`delete_centroid`] only flips the delete bit: the slot keeps its in-edges,
/// and a walk still spends frontier entries on it. Because ids are never
/// reused, a long-running clusterer accumulates roughly one tombstone per
/// split and searches degrade as the beam fills with dead nodes. This variant
/// additionally rewires each in-neighbor onto the `num_to_replace` nearest
/// members of the dying centroid's own neighborhood and drops its out-edges.
///
/// Deletion is performed by the call itself, so this *replaces*
/// [`delete_centroid`] rather than following it (calling both is harmless —
/// the delete is idempotent — and is what the dissolve path relies on to mark
/// a whole victim set before repairing it).
///
/// Uses [`InplaceDeleteMethod::TwoHopAndOneHop`].
pub(crate) fn inplace_delete_centroid(
    index: &MutableCentroidGraph,
    runtime: &Runtime,
    id: u32,
    num_to_replace: usize,
) -> Result<()> {
    runtime.block_on(index.inplace_delete(
        FullPrecision,
        &DefaultContext,
        &id,
        num_to_replace,
        InplaceDeleteMethod::TwoHopAndOneHop,
    ))?;
    Ok(())
}

/// Out-edge census of the live centroid graph.
///
/// Search traverses tombstoned centroids like any other node and only discards
/// them when results are copied out, so dead adjacency entries dilute the
/// candidate list rather than disconnecting the graph. This measures how much
/// of the graph's out-degree is still live, which decides whether the dead
/// entries can simply be dropped or have to be replaced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdjacencyCensus {
    /// Live centroids inspected.
    pub nodes: usize,
    /// Out-edges across those centroids, live and tombstoned.
    pub out_edges: u64,
    /// Out-edges pointing at a live centroid.
    pub live_out_edges: u64,
    /// Live centroids with no live out-edge at all — dead ends for navigation
    /// if the tombstoned entries were dropped.
    pub starved: usize,
}

impl AdjacencyCensus {
    /// Share of out-edges pointing at a live centroid, in `0.0..=1.0`.
    pub fn live_fraction(&self) -> f64 {
        if self.out_edges == 0 {
            return 1.0;
        }
        self.live_out_edges as f64 / self.out_edges as f64
    }

    /// Mean live out-degree per inspected centroid.
    pub fn mean_live_degree(&self) -> f64 {
        if self.nodes == 0 {
            return 0.0;
        }
        self.live_out_edges as f64 / self.nodes as f64
    }
}

/// Census the out-edges of the centroids in `live`.
///
/// Reads adjacency lists and delete bits only — no distance computations — so
/// the cost is `O(|live| * degree)` status checks.
pub(crate) fn adjacency_census(
    index: &MutableCentroidGraph,
    runtime: &Runtime,
    live: impl Iterator<Item = u32>,
) -> Result<AdjacencyCensus> {
    let provider = index.provider();
    let mut accessor = provider.default_accessor();
    let mut census = AdjacencyCensus::default();

    runtime.block_on(async {
        for id in live {
            let internal = provider.to_internal_id(&DefaultContext, &id)?;
            let partitioned = index
                .get_undeleted_neighbors(&DefaultContext, &mut accessor, internal)
                .await?;

            census.nodes += 1;
            census.live_out_edges += partitioned.undeleted.len() as u64;
            census.out_edges += (partitioned.undeleted.len() + partitioned.deleted.len()) as u64;
            if partitioned.undeleted.is_empty() {
                census.starved += 1;
            }
        }
        Ok::<_, ANNError>(())
    })?;

    Ok(census)
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
    let knn = Knn::new(l, None).map_err(ANNError::from)?;

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
    /// This is why the online [`CentroidRegistry`] hands out ids monotonically
    /// and `centroid_capacity` must cover every id ever allocated, not just the
    /// peak live count. Recycling ids would silently lose clusters.
    ///
    /// [`CentroidRegistry`]: crate::online::OnlineClusterer
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

    /// An in-place delete removes the centroid from search while leaving the
    /// rest of the graph navigable, including successors inserted at the dying
    /// centroid's location — the split path's ordering.
    #[test]
    fn inplace_delete_keeps_successors_reachable() {
        let (graph, rt) = square(8);

        // Split the corner at (10, 10): publish two children on top of it, then
        // retire the parent.
        insert_centroid(&graph, &rt, 4, &[9.0, 10.0]).unwrap();
        insert_centroid(&graph, &rt, 5, &[11.0, 10.0]).unwrap();
        inplace_delete_centroid(&graph, &rt, 3, 2).unwrap();

        let found = nearest(&graph, &rt, &[10.0, 10.0], 6);
        assert!(!found.contains(&3), "the parent is gone (found {found:?})");
        assert!(
            found.contains(&4) && found.contains(&5),
            "both children are reachable (found {found:?})"
        );
        for id in [0u32, 1, 2] {
            assert!(
                nearest(&graph, &rt, &[10.0, 10.0], 6).contains(&id),
                "centroid {id} survived the repair"
            );
        }
    }
}
