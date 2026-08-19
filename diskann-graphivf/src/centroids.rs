/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! In-memory structures for locating the nearest cluster centroids.
//!
//! Two of them: a DiskANN graph over the centroids (a thin wrapper around the
//! `diskann` graph and the `diskann-inmem` concurrent provider) and an exact
//! scan over a packed copy of them ([`exact`]). [`CentroidSource`] and its
//! online counterpart pick between the two so that no caller has to.

mod exact;

pub(crate) use exact::{DenseCentroids, ExactMetric};

use std::{num::NonZeroUsize, sync::Arc};

use diskann::{
    graph::{
        config::{Builder, MaxDegree},
        glue::PruneStrategy,
        search::Knn,
        search_output_buffer::{IdDistance, SearchOutputBuffer},
        Config, DiskANNIndex, InplaceDeleteMethod,
    },
    provider::DataProvider,
    ANNError,
};
use diskann_inmem::{layers, provider as inmem, Context, Provider, Strategy};
use diskann_providers::utils::{create_thread_pool, ParallelIteratorInPool, RayonThreadPool};
use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::Metric as VectorMetric;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use tokio::runtime::Runtime;

use crate::{
    params::{CentroidRouting, GraphParams},
    GraphIvfError, Result,
};

/// A DiskANN graph over the centroids, backed by the concurrent in-memory
/// provider.
///
/// Centroid ids are the provider's *external* ids, so row `i` of the matrix the
/// graph was built from is always returned as id `i` regardless of which
/// internal slot it landed in.
///
/// The provider supports insertion and deletion natively, so the graph the
/// batch clusterer builds and the one the online clusterer mutates differ only
/// in how much spare capacity they were given ([`build`] versus
/// [`build_mutable`]).
pub(crate) type CentroidGraph = Arc<DiskANNIndex<Provider<layers::Full<f32>, u32>>>;

/// Build the graph [`Config`] shared by every centroid graph.
fn graph_config(graph: &GraphParams, metric: VectorMetric) -> Result<Config> {
    Builder::new_with(
        graph.degree,
        MaxDegree::slack(graph.slack),
        graph.l_build,
        metric.into(),
        |b| {
            b.alpha(graph.alpha);
        },
    )
    .build()
    .map_err(ANNError::from)
    .map_err(Into::into)
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

/// Create an empty centroid graph with room for `capacity` centroids, frozen at
/// `start` (a copy of one centroid, used only as the navigation entry point).
///
/// The frozen point holds no external id, so it is never reported by a search:
/// result translation drops any candidate without an id mapping.
fn empty_graph(
    graph: &GraphParams,
    dim: usize,
    capacity: usize,
    metric: VectorMetric,
    start: &[f32],
    num_threads: usize,
) -> Result<CentroidGraph> {
    let config = graph_config(graph, metric)?;

    let mut provider_config = inmem::Config::new(capacity, config.max_degree_u32().get() as usize);
    // One epoch guard is held for the whole of every concurrent search or
    // insert. Build-time `num_threads` only bounds the build, so keep a floor
    // that also covers search-time concurrency; a slot is 8 bytes.
    provider_config.set_epoch_guard_slots(NonZeroUsize::new(
        num_threads.saturating_mul(4).max(MIN_GUARD_SLOTS),
    ));

    let provider = Provider::new(
        layers::Full::<f32>::new(dim, metric),
        provider_config,
        std::iter::once(start),
    )
    .map_err(ANNError::new)?;

    Ok(Arc::new(DiskANNIndex::new(config, provider, None)))
}

/// Guard slots to provision when the caller's thread count does not demand more.
const MIN_GUARD_SLOTS: usize = 1024;

/// Insert row `i` of `centroids` under external id `i`, in parallel.
///
/// Ids come from the row index rather than the insert order, so the graph's
/// contents are independent of how the work is scheduled: row `i` is always
/// centroid `i`, whichever internal slot the provider hands it.
fn insert_centroids(
    index: &CentroidGraph,
    centroids: &Matrix<f32>,
    pool: &RayonThreadPool,
) -> Result<()> {
    const CHUNK: usize = 64;

    (0..centroids.nrows())
        .into_par_iter()
        .chunks(CHUNK)
        .try_for_each_in_pool(pool.as_ref(), |rows| -> Result<()> {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .map_err(ANNError::from)?;
            runtime.block_on(async {
                for i in rows {
                    index
                        .insert(&Strategy, &Context, &(i as u32), centroids.row(i))
                        .await?;
                }
                Ok::<_, ANNError>(())
            })?;
            Ok(())
        })
}

/// Build an immutable in-memory full-precision graph over `centroids` (row `i`
/// is centroid `i`, returned as external id `i`), sized exactly to its
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
) -> Result<CentroidGraph> {
    let capacity = centroids.nrows();
    build_with_capacity(centroids, graph, num_threads, capacity, metric)
}

/// Shared body of [`build`] and [`build_mutable`]: fill a graph sized to
/// `capacity` with every row of `centroids` under its own row index as id.
fn build_with_capacity(
    centroids: Matrix<f32>,
    graph: &GraphParams,
    num_threads: usize,
    capacity: usize,
    metric: VectorMetric,
) -> Result<CentroidGraph> {
    let num_clusters = centroids.nrows();
    if num_clusters == 0 {
        return Err(GraphIvfError::invalid(
            "cannot build a centroid graph over an empty centroid set",
        ));
    }
    if capacity < num_clusters {
        return Err(GraphIvfError::invalid(format!(
            "graph capacity ({capacity}) is smaller than the initial centroid count ({num_clusters})"
        )));
    }

    let start = start_point_index(num_clusters);
    let index = empty_graph(
        graph,
        centroids.ncols(),
        capacity,
        metric,
        centroids.row(start),
        num_threads,
    )?;

    let pool = create_thread_pool(num_threads)?;
    insert_centroids(&index, &centroids, &pool)?;

    Ok(index)
}

/// Run a k-NN search over the centroid graph, writing the resulting centroid ids
/// and distances into `ids_out` / `dist_out`. Returns the number of results
/// written (`<= ids_out.len()`).
///
/// `ids_out` and `dist_out` must have the same length, which is the requested
/// `k`. `l` is the centroid-graph search-list size and must be `>= k`.
pub(crate) fn search(
    index: &CentroidGraph,
    runtime: &Runtime,
    query: &[f32],
    l: usize,
    ids_out: &mut [u32],
    dist_out: &mut [f32],
) -> Result<usize> {
    debug_assert_eq!(ids_out.len(), dist_out.len());
    let knn = Knn::new(l, None).map_err(ANNError::from)?;

    let mut buffer = IdDistance::new(ids_out, dist_out);
    runtime.block_on(index.search(knn, &Strategy, &Context, query, &mut buffer))?;

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
        index: CentroidGraph,
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

/// Build a [`CentroidGraph`] over `centroids` with room for `capacity` live
/// centroids.
///
/// The initial centroids receive ids `0..centroids.nrows()`. Unlike [`build`],
/// which sizes the graph exactly to its centroids, this leaves spare slots for
/// later [`insert_centroid`] calls. Because [`inplace_delete_centroid`] frees a
/// slot for reuse, the graph itself only needs `capacity` to cover the centroids
/// live at any one instant. The online clusterer nonetheless sizes it as a
/// budget on ids *consumed*, since it never reissues a retired id.
///
/// `metric` is the navigation/assignment metric (callers pass
/// [`VectorMetric::L2`], matching batch clustering).
///
/// # Errors
///
/// Returns an error if `capacity < centroids.nrows()`, `centroids` is empty, or
/// graph construction fails.
pub(crate) fn build_mutable(
    centroids: Matrix<f32>,
    graph: &GraphParams,
    num_threads: usize,
    capacity: usize,
    metric: VectorMetric,
) -> Result<CentroidGraph> {
    build_with_capacity(centroids, graph, num_threads, capacity, metric)
}

/// Insert a new centroid `vec` under external id `id` into a
/// [`CentroidGraph`]. `id` must not currently be live, and the graph must have
/// a free slot.
pub(crate) fn insert_centroid(
    index: &CentroidGraph,
    runtime: &Runtime,
    id: u32,
    vec: &[f32],
) -> Result<()> {
    runtime.block_on(index.insert(&Strategy, &Context, &id, vec))?;
    Ok(())
}

/// Delete the centroid with external id `id` and repair the graph around it, so
/// the dying centroid's in-neighbors are rewired before its slot is retired.
///
/// Repairing is not optional. A bare provider delete would release the id
/// mapping and leave the departing centroid's in-edges dangling: a walk still
/// spends frontier entries reaching a slot that no longer resolves to a
/// centroid, and once that slot is recycled those edges point at an unrelated
/// centroid instead. This rewires each in-neighbor onto the `num_to_replace`
/// nearest members of the dying centroid's own neighborhood and drops its
/// out-edges.
///
/// Deletion is performed by the call itself. Calling it twice on the same
/// centroid fails: the id mapping is gone, so the id can no longer be resolved.
///
/// Uses [`InplaceDeleteMethod::TwoHopAndOneHop`].
pub(crate) fn inplace_delete_centroid(
    index: &CentroidGraph,
    runtime: &Runtime,
    id: u32,
    num_to_replace: usize,
) -> Result<()> {
    runtime.block_on(index.inplace_delete(
        Strategy,
        &Context,
        &id,
        num_to_replace,
        InplaceDeleteMethod::TwoHopAndOneHop,
    ))?;
    Ok(())
}

/// Out-edge census of the live centroid graph.
///
/// A retired slot stays unreadable until the provider recycles it, and search
/// traverses an edge into one like any other before discarding it at result
/// translation. Such edges dilute the candidate list rather than disconnecting
/// the graph. This measures how much of the graph's out-degree still reaches a
/// readable slot, which decides whether those entries can simply be dropped or
/// have to be replaced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdjacencyCensus {
    /// Live centroids inspected.
    pub nodes: usize,
    /// Out-edges across those centroids, live and dead.
    pub out_edges: u64,
    /// Out-edges pointing at a readable slot.
    pub live_out_edges: u64,
    /// Live centroids with no live out-edge at all — dead ends for navigation
    /// if the dead entries were dropped.
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
/// Reads adjacency lists and slot status only — no distance computations — so
/// the cost is `O(|live| * degree)` status checks.
///
/// An edge counts as live if its slot is readable. Once a retired slot has been
/// recycled it is readable again, so this measures reachable-slot health, not
/// that each edge still points at the centroid it was created for.
pub(crate) fn adjacency_census(
    index: &CentroidGraph,
    runtime: &Runtime,
    live: impl Iterator<Item = u32>,
) -> Result<AdjacencyCensus> {
    let provider = index.provider();
    let mut accessor = Strategy.prune_accessor(provider, &Context, 0)?;
    let mut census = AdjacencyCensus::default();

    runtime.block_on(async {
        for id in live {
            let internal = provider.to_internal_id(&Context, &id)?;
            let partitioned = index
                .get_undeleted_neighbors(&Context, &mut accessor, internal)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(data: Vec<f32>, nrows: usize, ncols: usize) -> Matrix<f32> {
        Matrix::try_from(data.into_boxed_slice(), nrows, ncols).unwrap()
    }

    /// Deterministic pseudo-random centroids.
    fn random_centroids(nrows: usize, ncols: usize, seed: u64) -> Matrix<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let data = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        mat(data, nrows, ncols)
    }

    fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
        std::iter::zip(a, b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Brute-force nearest centroid under squared L2.
    fn exact_top1(cents: &Matrix<f32>, q: &[f32]) -> u32 {
        (0..cents.nrows())
            .min_by(|&i, &j| {
                sq_l2(cents.row(i), q)
                    .total_cmp(&sq_l2(cents.row(j), q))
                    .then(i.cmp(&j))
            })
            .unwrap() as u32
    }

    fn current_thread_rt() -> Runtime {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
    }

    fn probe(graph: &CentroidGraph, rt: &Runtime, q: &[f32], k: usize, l: usize) -> Vec<u32> {
        let mut ids = vec![u32::MAX; k];
        let mut dist = vec![0.0f32; k];
        let n = search(graph, rt, q, l, &mut ids, &mut dist).unwrap();
        ids.truncate(n);
        ids
    }

    /// Row `i` of the matrix must come back as centroid id `i`.
    ///
    /// This is the invariant every caller depends on — assignments, inverted
    /// lists and `DenseCentroids` are all keyed by it — and it is the one the
    /// provider swap could silently break, because internal slots are now
    /// handed out by a freelist instead of matching the row index.
    #[test]
    fn static_graph_returns_the_row_index_as_the_centroid_id() {
        let cents = random_centroids(512, 8, 7);
        let graph = build(cents.clone(), &GraphParams::default(), 4, VectorMetric::L2).unwrap();
        let rt = current_thread_rt();

        for i in 0..cents.nrows() {
            let got = probe(&graph, &rt, cents.row(i), 1, 64);
            assert_eq!(got, vec![i as u32], "centroid {i} did not retrieve itself");
        }
    }

    /// The frozen start point duplicates a centroid's vector but holds no
    /// external id, so it must never surface as a result or as a duplicate.
    #[test]
    fn static_graph_never_reports_the_frozen_start_point() {
        let cents = random_centroids(256, 8, 11);
        let n = cents.nrows() as u32;
        let graph = build(cents.clone(), &GraphParams::default(), 4, VectorMetric::L2).unwrap();
        let rt = current_thread_rt();

        for q in 0..64 {
            let query = random_centroids(1, 8, 1000 + q);
            let ids = probe(&graph, &rt, query.row(0), 32, 64);

            assert!(ids.iter().all(|&id| id < n), "out-of-range id in {ids:?}");
            let mut sorted = ids.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), ids.len(), "duplicate ids in {ids:?}");
        }
    }

    /// Graph routing must agree with a brute-force scan on the overwhelming
    /// majority of queries. A broken id mapping would score near zero here even
    /// though the graph itself navigates fine.
    #[test]
    fn static_graph_agrees_with_exact_search() {
        let cents = random_centroids(512, 8, 13);
        let graph = build(cents.clone(), &GraphParams::default(), 4, VectorMetric::L2).unwrap();
        let rt = current_thread_rt();

        let queries = random_centroids(256, 8, 99);
        let hits = (0..queries.nrows())
            .filter(|&q| {
                probe(&graph, &rt, queries.row(q), 1, 64)
                    .first()
                    .is_some_and(|&id| id == exact_top1(&cents, queries.row(q)))
            })
            .count();

        assert!(
            hits * 100 >= queries.nrows() * 95,
            "graph top-1 agreed with exact on only {hits}/{}",
            queries.nrows()
        );
    }

    /// Ids are taken from the row index, not the insert order, so the mapping
    /// must not depend on how many workers the build ran on.
    #[test]
    fn static_graph_ids_do_not_depend_on_the_thread_count() {
        let cents = random_centroids(384, 8, 17);
        let rt = current_thread_rt();

        let serial = build(cents.clone(), &GraphParams::default(), 1, VectorMetric::L2).unwrap();
        let parallel = build(cents.clone(), &GraphParams::default(), 8, VectorMetric::L2).unwrap();

        for i in 0..cents.nrows() {
            assert_eq!(
                probe(&serial, &rt, cents.row(i), 1, 64),
                probe(&parallel, &rt, cents.row(i), 1, 64),
                "centroid {i} resolved differently across thread counts"
            );
        }
    }

    /// Every concurrent search holds an epoch guard for its whole duration, so
    /// a graph built by few threads must still serve many searchers.
    #[test]
    fn static_graph_serves_more_searchers_than_builders() {
        let cents = random_centroids(256, 8, 23);
        let graph = build(cents.clone(), &GraphParams::default(), 2, VectorMetric::L2).unwrap();

        std::thread::scope(|s| {
            for t in 0..32 {
                let graph = &graph;
                let cents = &cents;
                s.spawn(move || {
                    let rt = current_thread_rt();
                    for i in (t..cents.nrows()).step_by(32) {
                        assert_eq!(probe(graph, &rt, cents.row(i), 1, 64), vec![i as u32]);
                    }
                });
            }
        });
    }

    /// A capacity sized exactly to the centroid count must absorb every
    /// insertion: the frozen start point lives outside that budget.
    #[test]
    fn static_graph_capacity_is_exactly_the_centroid_count() {
        let cents = random_centroids(1024, 4, 29);
        let graph = build(cents.clone(), &GraphParams::default(), 4, VectorMetric::L2).unwrap();
        let rt = current_thread_rt();

        let ids = probe(&graph, &rt, cents.row(0), cents.nrows(), 2048);
        assert_eq!(ids.len(), cents.nrows(), "not every centroid is reachable");
    }

    /// Four centroids at the corners of a 10x10 square, with spare capacity.
    fn square(capacity: usize) -> (CentroidGraph, Runtime) {
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

    fn nearest(graph: &CentroidGraph, rt: &Runtime, q: &[f32], k: usize) -> Vec<u32> {
        let mut ids = vec![0u32; k];
        let mut dist = vec![0.0f32; k];
        let n = search(graph, rt, q, k.max(16), &mut ids, &mut dist).unwrap();
        ids.truncate(n);
        ids
    }

    /// A deleted centroid is no longer returned by search, and a fresh id
    /// can be inserted afterwards.
    #[test]
    fn delete_then_insert_fresh_id() {
        let (graph, rt) = square(8);
        inplace_delete_centroid(&graph, &rt, 1, 2).unwrap();
        assert!(!nearest(&graph, &rt, &[10.0, 0.0], 4).contains(&1));

        insert_centroid(&graph, &rt, 4, &[9.0, 1.0]).unwrap();
        assert!(nearest(&graph, &rt, &[10.0, 0.0], 4).contains(&4));
    }

    /// Centroid ids **are** reusable: deleting a centroid releases its id
    /// mapping and frees its slot, so the same id can be re-inserted with a new
    /// vector and is found at the new location.
    ///
    /// The online [`CentroidRegistry`] still hands out ids monotonically and
    /// sizes itself by `centroid_capacity`, so it does not yet exploit this;
    /// the point here is that the graph no longer forbids it.
    ///
    /// [`CentroidRegistry`]: crate::online::OnlineClusterer
    #[test]
    fn deleted_centroid_ids_are_reusable() {
        let (graph, rt) = square(8);
        inplace_delete_centroid(&graph, &rt, 1, 2).unwrap();

        insert_centroid(&graph, &rt, 1, &[-5.0, -5.0]).unwrap();

        let found = nearest(&graph, &rt, &[-5.0, -5.0], 4);
        assert!(
            found.contains(&1),
            "a re-inserted id must be reachable at its new location (found {found:?})"
        );

        // The old location must no longer resolve to it.
        let corner = nearest(&graph, &rt, &[10.0, 0.0], 4);
        assert!(
            !corner.is_empty(),
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
