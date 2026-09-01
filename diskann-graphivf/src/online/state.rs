/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::Matrix;
use tokio::runtime::Runtime;

use crate::{
    centroids::{self, AdjacencyCensus, CentroidGraph, DenseCentroids, ExactMetric, GraphSnapshot},
    GraphIvfError, Result,
};

use super::UNASSIGNED;

/// Substitute edges offered to each node that loses a link when a centroid is
/// deleted in place.
///
/// This is a per-repaired-node fan-out, not a count of deletions: every
/// in-neighbor lost exactly one edge, and the proposed replacements still have
/// to survive pruning. Two matches the arity of a split, where the parent is
/// succeeded by exactly two children.
const REPLACE_EDGES: usize = 2;

/// The single owner of the live centroids and of the structure used to find
/// them.
///
/// Vectors live in a [`DenseCentroids`] mirror, which keeps them packed and
/// contiguous while still addressing them by id. That costs nothing over a
/// sparse id-indexed array — one `u32` per id slot instead of a pointer plus a
/// separate allocation per centroid — and it is what makes an exact scan of the
/// centroids a matrix multiply rather than a pointer chase.
///
/// A centroid id is permanent and is never reused after retirement, so the id
/// map is sized to the whole budget up front.
///
/// Structural updates go through this type so callers cannot update the vectors
/// without also updating whatever navigates them. Which of the two navigation
/// modes is in effect is fixed for the life of the clusterer and is not visible
/// to callers: routing, splitting, merging, and query-time cluster selection all
/// go through the same [`search`](Self::search).
pub(super) struct CentroidRegistry {
    /// Packed, live-only vectors, addressable by centroid id.
    dense: DenseCentroids,
    /// Total number of id slots (live + retired + free).
    capacity: usize,
    /// Next unused centroid id.
    next_id: u32,
    /// Maintained graph over the live centroids, or `None` when they are
    /// scanned exactly instead. Exact mode keeps no state of its own — there is
    /// no second copy of the centroids and no graph to hold in sync.
    graph: Option<CentroidGraph>,
}

impl CentroidRegistry {
    /// Create a registry with `capacity` id slots, seeding ids
    /// `0..initial.nrows()` from `initial` and leaving the remaining slots free.
    ///
    /// `graph` must already contain those same initial centroids under the same
    /// ids; pass `None` to navigate by exact scan instead.
    pub(super) fn new(
        initial: &Matrix<f32>,
        capacity: usize,
        graph: Option<CentroidGraph>,
    ) -> Self {
        let mut dense = DenseCentroids::with_capacity(initial.ncols(), capacity);
        for i in 0..initial.nrows() {
            dense.push(i as u32, initial.row(i));
        }
        Self {
            dense,
            capacity,
            next_id: initial.nrows() as u32,
            graph,
        }
    }

    /// Total number of id slots (live + retired + free).
    #[cfg(test)]
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of live (non-retired) centroids.
    pub(super) fn live_count(&self) -> usize {
        self.dense.len()
    }

    /// Whether `id` is a live centroid.
    pub(super) fn is_live(&self, id: u32) -> bool {
        self.dense.contains(id)
    }

    /// The vector of centroid `id`, or `None` if retired or out of range.
    pub(super) fn get(&self, id: u32) -> Option<&[f32]> {
        self.dense.get(id)
    }

    /// Number of ids still available for new centroids.
    pub(super) fn alloc_budget(&self) -> usize {
        self.capacity.saturating_sub(self.next_id as usize)
    }

    /// Iterate over live centroids as `(id, vector)` pairs, in ascending id
    /// order.
    pub(super) fn iter_live(&self) -> impl Iterator<Item = (u32, &[f32])> {
        self.dense.iter_by_id()
    }

    /// Ids of the live centroids, in ascending order.
    pub(super) fn live_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.dense.ids_by_id()
    }

    /// Return the next `count` ids without changing the registry.
    ///
    /// # Errors
    ///
    /// Returns an error if the id budget would be exhausted.
    fn reserve_ids(&self, count: usize) -> Result<Vec<u32>> {
        if count > self.alloc_budget() {
            return Err(GraphIvfError::invalid(
                "centroid capacity exceeded; increase centroid_capacity",
            ));
        }
        Ok((self.next_id..self.next_id + count as u32).collect())
    }

    /// Publish centroids under ids previously returned by [`reserve_ids`].
    /// This is infallible so the vectors can be updated only after all fallible
    /// centroid-graph mutations have succeeded.
    ///
    /// [`reserve_ids`]: Self::reserve_ids
    fn commit_reserved(&mut self, ids: &[u32], vectors: Vec<Box<[f32]>>) {
        debug_assert_eq!(ids.len(), vectors.len());
        debug_assert_eq!(ids.first().copied(), Some(self.next_id));
        for (&id, vector) in ids.iter().zip(vectors) {
            self.dense.push(id, &vector);
        }
        self.next_id += ids.len() as u32;
    }

    /// Densely pack the live centroids into a contiguous `k x dim` matrix and
    /// return `(remap, matrix)`, where `remap[old_id]` is the new dense index of
    /// a live centroid and [`UNASSIGNED`] for a retired id.
    ///
    /// The dense order is ascending id, not the mirror's internal row order:
    /// this numbering is written to disk, so it has to be a function of the live
    /// set alone and not of the order retirements happened to occur in.
    pub(super) fn densify(&self) -> Result<(Vec<u32>, Matrix<f32>)> {
        let dim = self.dense.dim();
        let mut remap = vec![UNASSIGNED; self.capacity];
        let mut cbuf = Vec::with_capacity(self.live_count() * dim);
        for (new, (old, vector)) in self.iter_live().enumerate() {
            remap[old as usize] = new as u32;
            cbuf.extend_from_slice(vector);
        }
        let mat = Matrix::try_from(cbuf.into_boxed_slice(), self.live_count(), dim)
            .map_err(|_| GraphIvfError::invalid("centroid matrix shape mismatch"))?;
        Ok((remap, mat))
    }

    /// Capture the centroid graph's adjacency, renumbered into the same dense
    /// order [`densify`](Self::densify) produces, so the saved graph and the
    /// saved centroid matrix agree on every id.
    ///
    /// Returns `None` when navigating exactly: there is no graph to capture.
    ///
    /// Edges into centroids retired by earlier deletions are dropped, since they
    /// name nothing that is being written. A graph that has seen heavy churn
    /// therefore saves sparser than it ran.
    pub(super) fn snapshot_graph(
        &self,
        runtime: &Runtime,
        probe: &[f32],
    ) -> Result<Option<GraphSnapshot>> {
        match &self.graph {
            None => Ok(None),
            Some(graph) => {
                let ids: Vec<u32> = self.live_ids().collect();
                centroids::snapshot(graph, runtime, &ids, probe).map(Some)
            }
        }
    }

    /// The `ids.len()` nearest live centroids to `query`, ascending by
    /// distance, returning how many were written.
    ///
    /// `l` is the graph search-list size and is ignored in exact mode, which has
    /// no beam to widen.
    pub(super) fn search(
        &self,
        runtime: &Runtime,
        query: &[f32],
        l: usize,
        ids: &mut [u32],
        distances: &mut [f32],
    ) -> Result<usize> {
        match &self.graph {
            Some(graph) => centroids::search(graph, runtime, query, l, ids, distances),
            None => self.exact_search(query, ids, distances),
        }
    }

    /// Like [`search`](Self::search), but always ranks against every live
    /// centroid, whichever navigation mode is configured.
    ///
    /// This is the ranking the centroid graph is trying to reproduce, which is
    /// what makes it both the reference for centroid-selection recall and the
    /// recovery path when a graph walk exhausts its frontier without reaching
    /// any live centroid. It costs a full pass over the centroid set, so it
    /// does not belong on a hot query path.
    pub(super) fn exact_search(
        &self,
        query: &[f32],
        ids: &mut [u32],
        distances: &mut [f32],
    ) -> Result<usize> {
        // Online clustering is squared-L2 throughout, matching a batch build,
        // whatever metric candidate scoring later uses.
        self.dense.search(ExactMetric::SqL2, query, ids, distances)
    }

    /// The nearest packed live centroid accepted by `keep`.
    ///
    /// Used only after bounded graph candidates fail, or as the primary path
    /// when exact routing is configured.
    pub(super) fn closest_live_where(
        &self,
        query: &[f32],
        keep: impl FnMut(u32) -> bool,
    ) -> Option<u32> {
        self.dense.closest_where(query, keep)
    }

    /// Out-edge health of the centroid graph, or `None` when no graph is
    /// maintained.
    pub(super) fn adjacency_census(&self, runtime: &Runtime) -> Result<Option<AdjacencyCensus>> {
        match &self.graph {
            Some(graph) => centroids::adjacency_census(graph, runtime, self.live_ids()).map(Some),
            None => Ok(None),
        }
    }

    /// Publish `children` and retire `parents` as one structural update.
    ///
    /// The vectors are changed only after every graph operation succeeds. A
    /// graph failure can still leave the graph partially updated, so the caller
    /// must poison the enclosing clusterer before calling this method.
    pub(super) fn apply_split(
        &mut self,
        runtime: &Runtime,
        parents: &[u32],
        children: Vec<Box<[f32]>>,
    ) -> Result<Vec<u32>> {
        debug_assert!(parents.iter().all(|&id| self.is_live(id)));
        let ids = self.reserve_ids(children.len())?;
        if let Some(graph) = &self.graph {
            for (&id, vector) in ids.iter().zip(&children) {
                centroids::insert_centroid(graph, runtime, id, vector)?;
            }
            // The children must already be in the graph: they sit where the
            // parent sat, so they are the replacements the in-place delete below
            // rewires the parent's in-neighbors onto.
            for &parent in parents {
                centroids::inplace_delete_centroid(graph, runtime, parent, REPLACE_EDGES)?;
            }
        }

        self.commit_reserved(&ids, children);
        for &parent in parents {
            self.dense.remove(parent);
        }
        Ok(ids)
    }

    /// Retire all `victims` from the graph and the vector store.
    ///
    /// As with [`apply_split`](Self::apply_split), the caller must already have
    /// poisoned the clusterer because a partial graph failure is irreversible.
    pub(super) fn retire_all(&mut self, runtime: &Runtime, victims: &[u32]) -> Result<()> {
        debug_assert!(victims.iter().all(|&id| self.is_live(id)));
        if let Some(graph) = &self.graph {
            // Repair-and-remove one victim at a time. Dissolved clusters are
            // spatially correlated — a sparse region empties together — so a
            // victim's neighborhood is largely other victims, and an early
            // repair can rewire in-neighbors onto a victim that is itself about
            // to go. That self-corrects: those in-neighbors are rewired again
            // when the victim they were pointed at is retired, and by the last
            // victim every other one is already gone, so nothing is left
            // pointing into the retired set.
            for &victim in victims {
                centroids::inplace_delete_centroid(graph, runtime, victim, REPLACE_EDGES)?;
            }
        }
        for &victim in victims {
            self.dense.remove(victim);
        }
        Ok(())
    }
}

/// A point removed from an inverted list and awaiting placement.
#[derive(Debug, Clone, Copy)]
pub(super) struct DetachedPoint {
    pub(super) id: u32,
    pub(super) previous: u32,
}

/// The IVF point↔centroid mapping: the id-indexed inverted lists and the
/// reverse per-point assignment.
pub(super) struct IvfPartition {
    /// `lists[c]` holds the ids of points currently assigned to centroid `c`;
    /// empty for retired ids.
    lists: Vec<Vec<u32>>,
    /// `assignments[pid]` is the centroid id of point `pid`, or [`UNASSIGNED`]
    /// before it is inserted.
    assignments: Vec<u32>,
}

impl IvfPartition {
    /// Create a partition with `capacity` empty inverted lists over `num_points`
    /// initially unassigned points.
    pub(super) fn new(capacity: usize, num_points: usize) -> Self {
        Self {
            lists: (0..capacity).map(|_| Vec::new()).collect(),
            assignments: vec![UNASSIGNED; num_points],
        }
    }

    /// Attach a previously absent point to a centroid.
    pub(super) fn attach_new(&mut self, pid: u32, cid: u32) {
        debug_assert_eq!(self.assignments[pid as usize], UNASSIGNED);
        self.lists[cid as usize].push(pid);
        self.assignments[pid as usize] = cid;
    }

    /// Place a detached point on `cid`, returning whether it moved.
    pub(super) fn attach_detached(&mut self, point: DetachedPoint, cid: u32) -> bool {
        debug_assert_eq!(self.assignments[point.id as usize], UNASSIGNED);
        self.lists[cid as usize].push(point.id);
        self.assignments[point.id as usize] = cid;
        point.previous != cid
    }

    /// The current assignment of point `pid`.
    pub(super) fn assignment(&self, pid: u32) -> u32 {
        self.assignments[pid as usize]
    }

    /// The members (point ids) currently in centroid `cid`'s inverted list.
    pub(super) fn members(&self, cid: u32) -> &[u32] {
        &self.lists[cid as usize]
    }

    /// Number of points in centroid `cid`'s inverted list.
    pub(super) fn list_len(&self, cid: u32) -> usize {
        self.lists[cid as usize].len()
    }

    /// Remove `victims` from centroid `cid`'s inverted list, resetting their
    /// reverse assignments to [`UNASSIGNED`] so the points may be re-inserted
    /// later.
    ///
    /// `victims` must be sorted, deduplicated, and all currently assigned to
    /// `cid`. The list is filtered in a single pass regardless of how many
    /// points are being removed, which is what keeps a batched delete linear in
    /// list length rather than `O(victims * list_len)`.
    pub(super) fn remove_sorted(&mut self, cid: u32, victims: &[u32]) {
        debug_assert!(victims.windows(2).all(|w| w[0] < w[1]));
        for &pid in victims {
            debug_assert_eq!(self.assignments[pid as usize], cid);
            self.assignments[pid as usize] = UNASSIGNED;
        }
        let assignments = &self.assignments;
        self.lists[cid as usize].retain(|&pid| assignments[pid as usize] != UNASSIGNED);
    }

    /// Detach all members of `cid`, appending them to `out` with their previous
    /// assignment and leaving both directions of the partition consistent.
    pub(super) fn detach_members_into(&mut self, cid: u32, out: &mut Vec<DetachedPoint>) {
        let members = std::mem::take(&mut self.lists[cid as usize]);
        out.reserve(members.len());
        for pid in members {
            debug_assert_eq!(self.assignments[pid as usize], cid);
            self.assignments[pid as usize] = UNASSIGNED;
            out.push(DetachedPoint {
                id: pid,
                previous: cid,
            });
        }
    }

    /// Move selected currently assigned points to new centroids in one pass per
    /// touched source list. Entries whose target equals their current centroid
    /// are ignored.
    pub(super) fn relocate(&mut self, destinations: &[(u32, u32)]) -> usize {
        let mut moves = Vec::with_capacity(destinations.len());
        let mut touched = Vec::new();
        for &(pid, target) in destinations {
            let source = self.assignment(pid);
            if source == target {
                continue;
            }
            debug_assert_ne!(source, UNASSIGNED);
            moves.push((pid, source, target));
            touched.push(source);
            self.assignments[pid as usize] = UNASSIGNED;
        }
        touched.sort_unstable();
        touched.dedup();
        for source in touched {
            let assignments = &self.assignments;
            self.lists[source as usize].retain(|&pid| assignments[pid as usize] != UNASSIGNED);
        }
        for &(pid, _, target) in &moves {
            self.attach_new(pid, target);
        }
        moves.len()
    }
}
