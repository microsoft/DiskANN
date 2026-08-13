/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::Matrix;
use tokio::runtime::Runtime;

use crate::{
    centroids::{self, MutableCentroidGraph},
    cluster::sq_l2,
    GraphIvfError, Result,
};

use super::UNASSIGNED;

/// Id-indexed store of centroid vectors with soft deletion.
///
/// A centroid id is permanent and is never reused after retirement, so the
/// table is sized to the whole id budget up front. A `None` slot is retired or
/// not yet allocated; a `Some` slot is live.
pub(super) struct CentroidTable {
    dim: usize,
    /// Centroid vectors indexed by id; `None` marks a retired centroid.
    vecs: Vec<Option<Box<[f32]>>>,
    /// Number of live (non-retired) centroids.
    live_count: usize,
    /// Next unused centroid id.
    next_id: u32,
}

impl CentroidTable {
    /// Create a table with `capacity` id slots, seeding ids `0..initial.nrows()`
    /// from `initial` and leaving the remaining slots free.
    pub(super) fn new(initial: &Matrix<f32>, capacity: usize) -> Self {
        let initial_k = initial.nrows();
        let mut vecs: Vec<Option<Box<[f32]>>> = Vec::with_capacity(capacity);
        for i in 0..initial_k {
            vecs.push(Some(initial.row(i).to_vec().into_boxed_slice()));
        }
        for _ in initial_k..capacity {
            vecs.push(None);
        }
        Self {
            dim: initial.ncols(),
            vecs,
            live_count: initial_k,
            next_id: initial_k as u32,
        }
    }

    /// Total number of id slots (live + retired + free).
    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.vecs.len()
    }

    /// Number of live (non-retired) centroids.
    fn live_count(&self) -> usize {
        self.live_count
    }

    /// Whether `id` is a live centroid.
    fn is_live(&self, id: u32) -> bool {
        self.vecs.get(id as usize).is_some_and(Option::is_some)
    }

    /// The vector of centroid `id`, or `None` if retired or out of range.
    fn get(&self, id: u32) -> Option<&[f32]> {
        self.vecs.get(id as usize).and_then(|s| s.as_deref())
    }

    /// Number of ids still available for new centroids.
    fn alloc_budget(&self) -> usize {
        self.vecs.len().saturating_sub(self.next_id as usize)
    }

    /// Return the next `count` ids without changing the table.
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
    /// This is infallible so the table can be updated only after all fallible
    /// centroid-graph mutations have succeeded.
    ///
    /// [`reserve_ids`]: Self::reserve_ids
    fn commit_reserved(&mut self, ids: &[u32], vectors: Vec<Box<[f32]>>) {
        debug_assert_eq!(ids.len(), vectors.len());
        debug_assert_eq!(ids.first().copied(), Some(self.next_id));
        for (&id, vector) in ids.iter().zip(vectors) {
            debug_assert_eq!(vector.len(), self.dim);
            debug_assert!(self.vecs[id as usize].is_none());
            self.vecs[id as usize] = Some(vector);
        }
        self.next_id += ids.len() as u32;
        self.live_count += ids.len();
    }

    /// Retire centroid `id` (soft delete). Its slot stays occupied but empty; a
    /// no-op if `id` is already retired.
    fn retire(&mut self, id: u32) {
        if self.vecs[id as usize].take().is_some() {
            self.live_count -= 1;
        }
    }

    /// Iterate over live centroids as `(id, vector)` pairs, in ascending id
    /// order.
    fn iter_live(&self) -> impl Iterator<Item = (u32, &[f32])> {
        self.vecs
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| slot.as_deref().map(|v| (id as u32, v)))
    }

    /// Ids of the live centroids, in ascending order.
    fn live_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.vecs
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| slot.as_ref().map(|_| id as u32))
    }

    /// Squared-L2 distance from `query` to its `k`-th nearest live centroid, or
    /// `None` if fewer than `k` centroids are live.
    ///
    /// This is the exact cutoff the centroid graph is trying to reproduce, so it
    /// serves as the reference for centroid-selection recall. Comparing against
    /// a distance rather than an id set counts a tie as correct instead of
    /// penalizing whichever tie-break the graph happened to take.
    fn kth_nearest_distance(&self, query: &[f32], k: usize) -> Option<f32> {
        if k == 0 {
            return None;
        }
        let mut distances: Vec<f32> = self.iter_live().map(|(_, v)| sq_l2(query, v)).collect();
        if distances.len() < k {
            return None;
        }
        let (_, kth, _) = distances.select_nth_unstable_by(k - 1, f32::total_cmp);
        Some(*kth)
    }

    /// Brute-force nearest live centroid to `point` by squared-L2, or `None` if
    /// no live centroid exists.
    fn nearest(&self, point: &[f32]) -> Option<u32> {
        let mut best = None;
        let mut best_d = f32::INFINITY;
        for (id, v) in self.iter_live() {
            let d = sq_l2(point, v);
            if d < best_d {
                best_d = d;
                best = Some(id);
            }
        }
        best
    }

    /// Densely pack the live centroids into a contiguous `k x dim` matrix and
    /// return `(remap, matrix)`, where `remap[old_id]` is the new dense index of
    /// a live centroid and [`UNASSIGNED`] for a retired id.
    fn densify(&self) -> Result<(Vec<u32>, Matrix<f32>)> {
        let live: Vec<usize> = (0..self.vecs.len())
            .filter(|&c| self.vecs[c].is_some())
            .collect();
        let k = live.len();
        let mut remap = vec![UNASSIGNED; self.vecs.len()];
        let mut cbuf = vec![0.0f32; k * self.dim];
        for (new, &old) in live.iter().enumerate() {
            remap[old] = new as u32;
            cbuf[new * self.dim..(new + 1) * self.dim]
                .copy_from_slice(self.vecs[old].as_ref().expect("live"));
        }
        let mat = Matrix::try_from(cbuf.into_boxed_slice(), k, self.dim)
            .map_err(|_| GraphIvfError::invalid("centroid matrix shape mismatch"))?;
        Ok((remap, mat))
    }
}

/// The single owner of the two representations of live centroids.
///
/// The vector table is used by clustering and brute-force fallback, while the
/// mutable graph is used for navigation. Structural updates go through this
/// type so callers cannot update one representation without the other.
pub(super) struct CentroidRegistry {
    table: CentroidTable,
    graph: MutableCentroidGraph,
}

impl CentroidRegistry {
    pub(super) fn new(table: CentroidTable, graph: MutableCentroidGraph) -> Self {
        Self { table, graph }
    }

    #[cfg(test)]
    pub(super) fn capacity(&self) -> usize {
        self.table.capacity()
    }

    pub(super) fn live_count(&self) -> usize {
        self.table.live_count()
    }

    pub(super) fn is_live(&self, id: u32) -> bool {
        self.table.is_live(id)
    }

    pub(super) fn get(&self, id: u32) -> Option<&[f32]> {
        self.table.get(id)
    }

    pub(super) fn alloc_budget(&self) -> usize {
        self.table.alloc_budget()
    }

    pub(super) fn iter_live(&self) -> impl Iterator<Item = (u32, &[f32])> {
        self.table.iter_live()
    }

    pub(super) fn live_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.table.live_ids()
    }

    pub(super) fn nearest(&self, point: &[f32]) -> Option<u32> {
        self.table.nearest(point)
    }

    pub(super) fn kth_nearest_distance(&self, query: &[f32], k: usize) -> Option<f32> {
        self.table.kth_nearest_distance(query, k)
    }

    pub(super) fn densify(&self) -> Result<(Vec<u32>, Matrix<f32>)> {
        self.table.densify()
    }

    pub(super) fn search(
        &self,
        runtime: &Runtime,
        query: &[f32],
        l: usize,
        ids: &mut [u32],
        distances: &mut [f32],
    ) -> Result<usize> {
        centroids::search_mut(&self.graph, runtime, query, l, ids, distances)
    }

    /// Publish `children` and retire `parents` as one structural update.
    ///
    /// The table is changed only after every graph operation succeeds. A graph
    /// failure can still leave the graph partially updated, so the caller must
    /// poison the enclosing clusterer before calling this method.
    pub(super) fn apply_split(
        &mut self,
        runtime: &Runtime,
        parents: &[u32],
        children: Vec<Box<[f32]>>,
    ) -> Result<Vec<u32>> {
        debug_assert!(parents.iter().all(|&id| self.is_live(id)));
        let ids = self.table.reserve_ids(children.len())?;
        for (&id, vector) in ids.iter().zip(&children) {
            centroids::insert_centroid(&self.graph, runtime, id, vector)?;
        }
        for &parent in parents {
            centroids::delete_centroid(&self.graph, runtime, parent)?;
        }

        self.table.commit_reserved(&ids, children);
        for &parent in parents {
            self.table.retire(parent);
        }
        Ok(ids)
    }

    /// Retire all `victims` from the graph and table.
    ///
    /// As with [`apply_split`](Self::apply_split), the caller must already have
    /// poisoned the clusterer because a partial graph failure is irreversible.
    pub(super) fn retire_all(&mut self, runtime: &Runtime, victims: &[u32]) -> Result<()> {
        debug_assert!(victims.iter().all(|&id| self.is_live(id)));
        for &victim in victims {
            centroids::delete_centroid(&self.graph, runtime, victim)?;
        }
        for &victim in victims {
            self.table.retire(victim);
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
}
