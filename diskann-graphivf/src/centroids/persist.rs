/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Capture a built centroid graph's adjacency and rebuild an equivalent graph
//! from it, so that loading an index does not have to re-run graph construction.
//!
//! Adjacency is captured in *centroid* id space rather than provider-internal
//! slot ids. Internal slots are handed out by a concurrent freelist, so they
//! depend on insert scheduling and on the online clusterer's churn history;
//! neither survives a save. Centroid ids are positional and do survive, which
//! makes a snapshot reproducible and lets the online flush renumber live
//! centroids into a dense range as it writes them.

use std::collections::HashMap;

use diskann::{
    graph::{
        glue::{PruneStrategy, SearchAccessor, SearchStrategy},
        AdjacencyList,
    },
    provider::{DataProvider, Guard, NeighborAccessor, NeighborAccessorMut, SetElement},
    ANNError,
};
use diskann_inmem::{layers, Context, Provider, Strategy};
use diskann_utils::views::Matrix;
use diskann_vector::distance::Metric as VectorMetric;
use tokio::runtime::Runtime;

use super::{empty_graph, graph_config, start_point_index, CentroidGraph};
use crate::{params::GraphParams, GraphIvfError, Result};

/// Stands in for the frozen start point in a persisted adjacency list.
///
/// The start point holds no centroid id (that is what keeps searches from ever
/// reporting it), so it cannot be named positionally like every other node. It
/// is nonetheless reachable *from* other nodes and is the entry point every walk
/// begins at, so its edges have to round-trip along with the rest.
///
/// Centroid ids are dense positions below the centroid count, so the maximum
/// `u32` can never collide with one.
const START_POINT: u32 = u32::MAX;

/// A centroid graph's adjacency, in centroid id space.
///
/// `adjacency[i]` holds the out-edges of centroid `i`; `start` holds those of
/// the frozen start point. Entries are centroid ids, or [`START_POINT`].
///
/// Adjacency is just a set of edges, so it carries no distance of its own: the
/// metric belongs to whoever walks it. [`restore`] therefore navigates the saved
/// graph with whatever metric the caller asks for, which may differ from the one
/// that shaped it — clustering always runs in squared L2, while a loaded
/// inner-product index searches by inner product.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphSnapshot {
    /// Out-edges of centroid `i`, for `i` in `0..num_centroids`.
    pub(crate) adjacency: Vec<Vec<u32>>,
    /// Out-edges of the frozen start point.
    pub(crate) start: Vec<u32>,
}

impl GraphSnapshot {
    /// Number of centroids the snapshot describes.
    pub(crate) fn num_centroids(&self) -> usize {
        self.adjacency.len()
    }
}

/// Capture the adjacency of `index`, renumbering the centroids named by `ids`
/// into the dense range `0..ids.len()`.
///
/// `ids[p]` is the graph's current id for the centroid that becomes centroid `p`
/// in the snapshot. A batch build passes the identity; the online clusterer
/// passes its live centroids in flush order, which is exactly the renumbering it
/// applies to the inverted lists, so graph and lists stay in agreement.
///
/// `probe` is a throwaway vector of the graph's dimension. Obtaining the frozen
/// start point's id goes through a search accessor, which has to be handed a
/// query to build its distance computer; no search is run against it. Callers
/// have the centroid matrix to hand, so any row of it will do.
///
/// Edges into slots that are neither a listed centroid nor the start point are
/// dropped. Those are centroids retired by online deletion: the graph keeps
/// their edges until the slot is recycled (see
/// [`adjacency_census`](super::adjacency_census)), but they name nothing that
/// still exists, so they cannot be written down. This is why a snapshot of a
/// heavily churned graph is sparser than a freshly built one.
///
/// # Errors
///
/// Returns an error if `ids` is empty or repeats an id, if any id is not live in
/// the graph, or if the graph does not have exactly one start point.
pub(crate) fn snapshot(
    index: &CentroidGraph,
    runtime: &Runtime,
    ids: &[u32],
    probe: &[f32],
) -> Result<GraphSnapshot> {
    if ids.is_empty() {
        return Err(GraphIvfError::invalid(
            "cannot snapshot a centroid graph with no live centroids",
        ));
    }

    let provider = index.provider();

    // Internal slot -> snapshot position. Built up front so edge translation is
    // a lookup that cannot be confused by a slot that has since been recycled:
    // anything absent from this map is not part of the snapshot, full stop.
    let mut position = HashMap::with_capacity(ids.len() + 1);
    for (pos, &id) in ids.iter().enumerate() {
        let internal = provider.to_internal_id(&Context, &id)?;
        if position.insert(internal, pos as u32).is_some() {
            return Err(GraphIvfError::invalid(format!(
                "centroid id {id} is listed more than once"
            )));
        }
    }

    runtime.block_on(async {
        let start_internal = sole_start_point(provider, probe).await?;
        if position.insert(start_internal, START_POINT).is_some() {
            return Err(GraphIvfError::invalid(
                "the frozen start point also carries a centroid id",
            ));
        }

        let mut accessor = Strategy.prune_accessor(provider, &Context, 0)?;
        let mut edges = AdjacencyList::new();

        let mut adjacency = Vec::with_capacity(ids.len());
        for &id in ids {
            let internal = provider.to_internal_id(&Context, &id)?;
            accessor.get_neighbors(internal, &mut edges).await?;
            adjacency.push(translate(&edges, &position));
        }

        accessor.get_neighbors(start_internal, &mut edges).await?;
        let start = translate(&edges, &position);

        Ok(GraphSnapshot { adjacency, start })
    })
}

/// The internal slot of the graph's frozen start point.
///
/// The start point is only reachable through a search accessor, which has to be
/// handed a query to build its distance computer. `probe` is that query and
/// nothing is ever searched with it.
///
/// # Errors
///
/// Returns an error unless there is exactly one start point: both [`snapshot`]
/// and [`restore`] treat it as a single distinguished node, so a graph with more
/// would not round-trip.
async fn sole_start_point(
    provider: &Provider<layers::Full<f32>, u32>,
    probe: &[f32],
) -> Result<u32> {
    let accessor = Strategy
        .search_accessor(provider, &Context, probe)
        .map_err(ANNError::new)?;
    match accessor.starting_points().await?[..] {
        [only] => Ok(only),
        ref many => Err(GraphIvfError::invalid(format!(
            "expected a centroid graph with exactly one start point, found {}",
            many.len()
        ))),
    }
}

/// Map internal slots to snapshot ids, discarding slots that are not in the
/// snapshot (see [`snapshot`]).
fn translate(edges: &AdjacencyList<u32>, position: &HashMap<u32, u32>) -> Vec<u32> {
    edges
        .iter()
        .filter_map(|internal| position.get(internal).copied())
        .collect()
}

/// Rebuild a centroid graph over `centroids` with the adjacency in `snapshot`,
/// navigated by `metric`.
///
/// The result holds the edges that were saved: same centroids under the same
/// ids, same edges, same frozen start point. Only the internal slot numbering
/// may differ, and nothing outside the provider can observe it.
///
/// `metric` is the distance searches will rank by and need not be the one the
/// graph was built under — a saved graph is a set of edges, and loading is not
/// the place to silently substitute a different one. An inner-product index
/// therefore navigates the same L2-clustered graph it was built with, rather
/// than getting a differently shaped graph than the one that was saved.
///
/// Points are added without any graph construction and their edges are then
/// written verbatim, so this costs one insert and one adjacency write per
/// centroid instead of a search and a prune.
///
/// # Errors
///
/// Returns an error if the snapshot does not describe exactly the rows of
/// `centroids`, or if any adjacency list is longer than `graph` allows — the
/// provider would silently truncate an over-long list, which would leave a graph
/// that is quietly worse than the one that was saved.
pub(crate) fn restore(
    centroids: Matrix<f32>,
    graph: &GraphParams,
    num_threads: usize,
    metric: VectorMetric,
    snapshot: &GraphSnapshot,
) -> Result<CentroidGraph> {
    let num_clusters = centroids.nrows();
    if num_clusters == 0 {
        return Err(GraphIvfError::invalid(
            "cannot restore a centroid graph over an empty centroid set",
        ));
    }
    if snapshot.num_centroids() != num_clusters {
        return Err(GraphIvfError::malformed(format!(
            "persisted centroid graph covers {} centroids but {num_clusters} were loaded",
            snapshot.num_centroids()
        )));
    }

    let config = graph_config(graph, metric)?;
    let degree_cap = config.max_degree_u32().get() as usize;
    let over_capacity = snapshot
        .adjacency
        .iter()
        .chain(std::iter::once(&snapshot.start))
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    if over_capacity > degree_cap {
        return Err(GraphIvfError::malformed(format!(
            "persisted centroid graph has an out-degree of {over_capacity}, above the {degree_cap} \
             allowed by its recorded graph parameters"
        )));
    }

    let index = empty_graph(
        graph,
        centroids.ncols(),
        num_clusters,
        metric,
        centroids.row(start_point_index(num_clusters)),
        num_threads,
    )?;
    let provider = index.provider();

    current_thread_runtime()?.block_on(async {
        // Centroid id -> internal slot, plus the start point under `START_POINT`,
        // so an edge can be translated back with a single lookup.
        let mut internal = HashMap::with_capacity(num_clusters + 1);
        for i in 0..num_clusters {
            let guard = provider
                .set_element(&Context, &(i as u32), centroids.row(i))
                .await?;
            internal.insert(i as u32, guard.id());
        }

        let start_internal = sole_start_point(provider, centroids.row(0)).await?;
        internal.insert(START_POINT, start_internal);

        let mut accessor = Strategy.prune_accessor(provider, &Context, 0)?;
        let mut edges = Vec::with_capacity(degree_cap);
        for (i, neighbors) in snapshot.adjacency.iter().enumerate() {
            let id = i as u32;
            resolve(neighbors, &internal, &mut edges)?;
            accessor.set_neighbors(internal[&id], &edges).await?;
        }
        resolve(&snapshot.start, &internal, &mut edges)?;
        accessor.set_neighbors(start_internal, &edges).await?;

        Ok::<_, GraphIvfError>(())
    })?;

    Ok(index)
}

/// Translate persisted ids back to internal slots, into `out`.
fn resolve(neighbors: &[u32], internal: &HashMap<u32, u32>, out: &mut Vec<u32>) -> Result<()> {
    out.clear();
    for id in neighbors {
        let slot = internal.get(id).ok_or_else(|| {
            GraphIvfError::malformed(format!(
                "persisted centroid graph has an edge to unknown centroid {id}"
            ))
        })?;
        out.push(*slot);
    }
    Ok(())
}

/// A current-thread runtime to drive the provider calls that [`snapshot`] and
/// [`restore`] make.
///
/// Both are single-threaded walks — every centroid is visited exactly once, with
/// no search to parallelize — so they need a runtime only because the provider
/// API is async.
pub(super) fn current_thread_runtime() -> Result<Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(ANNError::from)
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::centroids::{build, search};

    const NUM_CENTROIDS: usize = 200;
    const DIM: usize = 8;

    fn random_matrix(nrows: usize, ncols: usize, seed: u64) -> Matrix<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect();
        Matrix::try_from(data.into_boxed_slice(), nrows, ncols).unwrap()
    }

    fn runtime() -> Runtime {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
    }

    fn identity(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    fn take(index: &CentroidGraph, centroids: &Matrix<f32>) -> GraphSnapshot {
        snapshot(
            index,
            &runtime(),
            &identity(centroids.nrows()),
            centroids.row(0),
        )
        .unwrap()
    }

    /// A restored graph must be the graph that was saved: same edges, and the
    /// same answers. Re-snapshotting is the sharper of the two checks, since it
    /// compares every edge rather than just the ones a query happens to walk.
    #[test]
    fn restore_reproduces_the_saved_graph() {
        let centroids = random_matrix(NUM_CENTROIDS, DIM, 7);
        let params = GraphParams::default();
        let original = build(
            random_matrix(NUM_CENTROIDS, DIM, 7),
            &params,
            2,
            VectorMetric::L2,
        )
        .unwrap();

        let saved = take(&original, &centroids);
        assert_eq!(saved.num_centroids(), NUM_CENTROIDS);
        assert!(
            saved.adjacency.iter().any(|n| !n.is_empty()),
            "a built graph must have edges"
        );

        let restored = restore(
            random_matrix(NUM_CENTROIDS, DIM, 7),
            &params,
            2,
            VectorMetric::L2,
            &saved,
        )
        .unwrap();

        assert_eq!(take(&restored, &centroids), saved);

        // ...and the same answers, which is what callers actually observe.
        let rt = runtime();
        let queries = random_matrix(32, DIM, 99);
        for q in 0..queries.nrows() {
            let mut want = [u32::MAX; 5];
            let mut want_d = [0.0f32; 5];
            let mut got = [u32::MAX; 5];
            let mut got_d = [0.0f32; 5];
            let n = search(&original, &rt, queries.row(q), 32, &mut want, &mut want_d).unwrap();
            let m = search(&restored, &rt, queries.row(q), 32, &mut got, &mut got_d).unwrap();
            assert_eq!((n, want), (m, got), "query {q} diverged after restore");
        }
    }

    /// A saved graph is a set of edges, not a distance. Restoring it under a
    /// different metric must hand back those same edges, navigated the new way,
    /// rather than quietly substituting a differently shaped graph.
    #[test]
    fn restore_under_another_metric_keeps_the_saved_edges() {
        let centroids = random_matrix(NUM_CENTROIDS, DIM, 11);
        let params = GraphParams::default();
        let index = build(
            random_matrix(NUM_CENTROIDS, DIM, 11),
            &params,
            2,
            VectorMetric::L2,
        )
        .unwrap();
        let saved = take(&index, &centroids);

        let restored = restore(
            random_matrix(NUM_CENTROIDS, DIM, 11),
            &params,
            2,
            VectorMetric::InnerProduct,
            &saved,
        )
        .unwrap();

        assert_eq!(take(&restored, &centroids), saved);
    }

    /// A snapshot that does not describe exactly the centroids being loaded is
    /// not a graph over them.
    #[test]
    fn restore_rejects_a_centroid_count_mismatch() {
        let centroids = random_matrix(NUM_CENTROIDS, DIM, 13);
        let params = GraphParams::default();
        let index = build(
            random_matrix(NUM_CENTROIDS, DIM, 13),
            &params,
            2,
            VectorMetric::L2,
        )
        .unwrap();
        let saved = take(&index, &centroids);

        assert!(restore(
            random_matrix(NUM_CENTROIDS - 1, DIM, 13),
            &params,
            2,
            VectorMetric::L2,
            &saved,
        )
        .is_err());
    }

    /// An adjacency list longer than the graph parameters allow would be
    /// truncated by the provider, quietly costing edges. It has to be refused.
    #[test]
    fn restore_rejects_an_over_capacity_adjacency_list() {
        let params = GraphParams::default();
        let too_many = params.degree * 10;
        let saved = GraphSnapshot {
            adjacency: vec![(0..too_many as u32).collect(); NUM_CENTROIDS],
            start: Vec::new(),
        };

        assert!(restore(
            random_matrix(NUM_CENTROIDS, DIM, 17),
            &params,
            2,
            VectorMetric::L2,
            &saved,
        )
        .is_err());
    }

    /// Edges naming a centroid that is not in the snapshot cannot be resolved to
    /// a slot, and must fail rather than be dropped.
    #[test]
    fn restore_rejects_an_edge_to_an_unknown_centroid() {
        let saved = GraphSnapshot {
            adjacency: (0..NUM_CENTROIDS)
                .map(|i| if i == 0 { vec![9_999] } else { Vec::new() })
                .collect(),
            start: Vec::new(),
        };

        assert!(restore(
            random_matrix(NUM_CENTROIDS, DIM, 19),
            &GraphParams::default(),
            2,
            VectorMetric::L2,
            &saved,
        )
        .is_err());
    }

    /// The snapshot renumbers the centroids it is given, which is what lets the
    /// online flush pack live centroids into a dense range.
    #[test]
    fn snapshot_renumbers_to_the_requested_order() {
        let centroids = random_matrix(NUM_CENTROIDS, DIM, 23);
        let index = build(
            random_matrix(NUM_CENTROIDS, DIM, 23),
            &GraphParams::default(),
            2,
            VectorMetric::L2,
        )
        .unwrap();

        // Keep every other centroid, in reverse order.
        let ids: Vec<u32> = (0..NUM_CENTROIDS as u32).rev().step_by(2).collect();
        let subset = snapshot(&index, &runtime(), &ids, centroids.row(0)).unwrap();

        assert_eq!(subset.num_centroids(), ids.len());
        // Every surviving edge names a position in the new numbering, never an
        // id from the old one.
        for neighbors in subset
            .adjacency
            .iter()
            .chain(std::iter::once(&subset.start))
        {
            for &n in neighbors {
                assert!(
                    (n as usize) < ids.len() || n == START_POINT,
                    "edge {n} is outside the renumbered range"
                );
            }
        }
    }

    #[test]
    fn snapshot_rejects_an_empty_id_list() {
        let index = build(
            random_matrix(NUM_CENTROIDS, DIM, 29),
            &GraphParams::default(),
            2,
            VectorMetric::L2,
        )
        .unwrap();
        let probe = vec![0.0f32; DIM];
        assert!(snapshot(&index, &runtime(), &[], &probe).is_err());
    }
}
