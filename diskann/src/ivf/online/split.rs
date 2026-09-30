/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Split planning: gather each split's region, fit children, and place every affected
//! point.

use std::ops::Range;

use diskann_utils::views::Matrix;
use rand::{Rng, rngs::StdRng};

use super::{
    batch::StagedBatch,
    check_reserved,
    gather::Members,
    index_error,
    kernels::{LloydScratch, lloyd, nearest},
};
use crate::{
    ANNResult,
    error::ErrorExt,
    ivf::{
        DynamicIvfConfig,
        dynamic::{CentroidIndex, MaintenanceAccessor},
        grouped::Csr,
        update::{Appends, CentroidBlock, Reassignments, SplitInsert, Splits},
    },
    utils::VectorId,
};

/// A list admitted for splitting, with its size in the accessor's view.
#[derive(Debug, Clone, Copy)]
pub(in crate::ivf) struct Parent<L> {
    pub(in crate::ivf) list: L,
    pub(in crate::ivf) len: usize,
}

/// Everything one split batch needs, gathered before any computation.
///
/// Lists are indexed densely: `lists[..parents]` are the split parents in ascending
/// order and `lists[parents..]` are their distinct neighbors in ascending order.
/// Region `r` is parent `lists[r]`, which children `2r` and `2r + 1` replace.
/// Neighbor `n` is `lists[parents + n]`.
pub(in crate::ivf) struct SplitPlan<Id, L> {
    batch: StagedBatch<Id, L>,
    lists: Vec<L>,
    parents: usize,
    /// Each region's neighbors.
    neighbors: Csr<usize>,
    /// Each neighbor's regions, ascending.
    regions: Csr<usize>,
    /// Row `n` is the centroid of neighbor `n`.
    neighbor_centroids: Matrix<f32>,
    /// Existing members of every list, indexed like `lists`.
    members: Members<Id>,
    /// Reserved ids for the children, two per region.
    child_ids: Vec<L>,
}

impl<Id: VectorId, L: VectorId> SplitPlan<Id, L> {
    /// Perform every accessor call the split needs.
    ///
    /// Reserves two child ids per parent, selects each parent's nearest surviving
    /// lists as its neighbors, and reads the members and canonical vectors of every
    /// parent and neighbor.
    ///
    /// # Errors
    ///
    /// Fails if an accessor call fails or the accessor's view is inconsistent.
    pub(in crate::ivf) async fn gather<A, T>(
        accessor: &mut A,
        config: &DynamicIvfConfig,
        batch: StagedBatch<Id, L>,
        parents: Vec<Parent<L>>,
    ) -> ANNResult<Self>
    where
        A: MaintenanceAccessor<T, Id = Id, ListId = L>,
        T: Send,
    {
        let count = 2 * parents.len();
        let child_ids = accessor
            .reserve_list_ids(count)
            .await
            .escalate("split must reserve child list ids")?;
        check_reserved(accessor.centroids(), &child_ids, count)?;

        let dim = accessor.dim();
        let mut lists: Vec<L> = parents.iter().map(|parent| parent.list).collect();
        let centroids = accessor.centroids();

        // Every parent is retired, so no parent is ever a neighbor.
        let mut by_region: Csr<L> = Csr::default();
        for &parent in &lists {
            let anchor = centroids
                .centroid(parent)
                .ok_or_else(|| index_error(format!("split parent {parent} is not live")))?;
            let selected = centroids
                .select(anchor, config.reassign_neighbors.saturating_add(1))
                .await
                .escalate("split must find neighbor lists")?;
            by_region.push(
                selected
                    .selected()
                    .iter()
                    .map(|selected| selected.id)
                    .filter(|id| lists.binary_search(id).is_err())
                    .take(config.reassign_neighbors),
            );
        }

        let mut distinct = by_region.values().to_vec();
        distinct.sort_unstable();
        distinct.dedup();

        let mut neighbor_centroids = Matrix::new(0.0f32, distinct.len(), dim);
        for (row, &list) in distinct.iter().enumerate() {
            let centroid = centroids
                .centroid(list)
                .ok_or_else(|| index_error(format!("neighbor list {list} is not live")))?;
            if centroid.len() != dim {
                return Err(index_error(format!(
                    "centroid {list} has dimension {}, expected {dim}",
                    centroid.len()
                )));
            }
            neighbor_centroids.row_mut(row).copy_from_slice(centroid);
        }

        let mut neighbors = Csr::default();
        for region in 0..by_region.num_groups() {
            // Every neighbor is in `distinct`, so each search succeeds.
            neighbors.push(
                by_region
                    .group(region)
                    .iter()
                    .filter_map(|list| distinct.binary_search(list).ok()),
            );
        }
        let regions = neighbors.transpose(distinct.len());
        lists.extend_from_slice(&distinct);

        let members = Members::read(accessor, &lists).await?;
        for (region, parent) in parents.iter().enumerate() {
            let len = members.ids(region).len();
            if len != parent.len {
                return Err(index_error(format!(
                    "split parent {} has {len} members, but its metadata reports {}",
                    parent.list, parent.len
                )));
            }
        }

        Ok(Self {
            batch,
            lists,
            parents: parents.len(),
            neighbors,
            regions,
            neighbor_centroids,
            members,
            child_ids,
        })
    }

    /// Fit the children, place every affected point, and build the update.
    ///
    /// Each parent splits into two children fitted by 2-means over its existing
    /// members and the batch points routed to it. Those points then move to the
    /// nearest of the parent's neighbors and children. A point of a neighbor list moves
    /// to a child of a region containing that list only if the child is closer than
    /// the list's centroid (as in SPFresh); otherwise it stays.
    ///
    /// # Errors
    ///
    /// Fails if a parent has fewer than two points.
    pub(in crate::ivf) fn solve(
        mut self,
        iterations: usize,
        rng: &mut StdRng,
    ) -> ANNResult<SplitInsert<Id, L>> {
        let centers = self.fit_children(iterations, rng)?;
        let children = CentroidBlock::new(std::mem::take(&mut self.child_ids), centers);
        let mut routes = std::mem::take(&mut self.batch.routes);
        let destinations = self.place(&children, &mut routes)?;
        Ok(self.package(children, &routes, &destinations))
    }

    /// Fit two children per region; rows `2r` and `2r + 1` split region `r`.
    fn fit_children(&self, iterations: usize, rng: &mut StdRng) -> ANNResult<Matrix<f32>> {
        let dim = self.batch.vectors.ncols();
        let mut centers = Matrix::new(0.0f32, 2 * self.parents, dim);
        let mut scratch = LloydScratch::default();
        for region in 0..self.parents {
            let span = self.members.ids(region).len() + self.staged(region).len();
            if span < 2 {
                return Err(index_error(format!(
                    "split parent {} has fewer than two points",
                    self.lists[region]
                )));
            }
            let a = rng.random_range(0..span);
            let b = (a + rng.random_range(1..span)) % span;

            let rows = &mut centers.as_mut_slice()[2 * region * dim..2 * (region + 1) * dim];
            let (first, second) = rows.split_at_mut(dim);
            first.copy_from_slice(self.point(region, a));
            second.copy_from_slice(self.point(region, b));
            lloyd(
                || self.points(region),
                rows,
                dim,
                iterations.max(1),
                &mut scratch,
            );
        }
        Ok(centers)
    }

    /// Choose a list for every point of every region list.
    ///
    /// Returns the existing members' lists, aligned with their positions, and
    /// overwrites the routes of the batch points it places.
    fn place(&self, children: &CentroidBlock<L>, routes: &mut [L]) -> ANNResult<Vec<L>> {
        let mut destinations = vec![L::default(); self.members.total()];
        let child_ids = children.ids();
        let child_vectors = children.vectors();
        let mut candidates: Vec<(L, &[f32])> = Vec::new();

        for region in 0..self.parents {
            candidates.clear();
            for &neighbor in self.neighbors.group(region) {
                candidates.push(self.neighbor(neighbor));
            }
            for row in [2 * region, 2 * region + 1] {
                candidates.push((child_ids[row], child_vectors.row(row)));
            }
            self.assign(region, &candidates, &mut destinations, routes)?;
        }

        // The list's own centroid comes first, so ties keep the point in place.
        for neighbor in 0..self.regions.num_groups() {
            candidates.clear();
            candidates.push(self.neighbor(neighbor));
            for &region in self.regions.group(neighbor) {
                for row in [2 * region, 2 * region + 1] {
                    candidates.push((child_ids[row], child_vectors.row(row)));
                }
            }
            self.assign(
                self.parents + neighbor,
                &candidates,
                &mut destinations,
                routes,
            )?;
        }
        Ok(destinations)
    }

    /// Place every point of `list` on its nearest candidate.
    fn assign(
        &self,
        list: usize,
        candidates: &[(L, &[f32])],
        destinations: &mut [L],
        routes: &mut [L],
    ) -> ANNResult<()> {
        let no_candidates = || index_error("split placement has no candidate lists");
        for (destination, vector) in destinations[self.members.range(list)]
            .iter_mut()
            .zip(self.members.vectors(list))
        {
            *destination = nearest(vector, candidates).ok_or_else(no_candidates)?;
        }
        for &position in self.staged(list) {
            routes[position] =
                nearest(self.batch.vectors.row(position), candidates).ok_or_else(no_candidates)?;
        }
        Ok(())
    }

    /// Assemble the update from the chosen lists.
    fn package(
        &self,
        children: CentroidBlock<L>,
        routes: &[L],
        destinations: &[L],
    ) -> SplitInsert<Id, L> {
        let parents = self.lists[..self.parents].to_vec();
        SplitInsert::new(
            Appends::new(&self.batch.ids, routes),
            Splits::new(parents, children, self.moves(0..self.parents, destinations)),
            Reassignments::new(self.moves(self.parents..self.lists.len(), destinations)),
        )
    }

    /// `(list, members, destinations)` for every list in `lists`.
    fn moves<'s>(
        &'s self,
        lists: Range<usize>,
        destinations: &'s [L],
    ) -> impl Iterator<Item = (L, &'s [Id], &'s [L])> {
        lists.map(move |list| {
            (
                self.lists[list],
                self.members.ids(list),
                &destinations[self.members.range(list)],
            )
        })
    }

    /// Neighbor `neighbor`'s list id and centroid.
    fn neighbor(&self, neighbor: usize) -> (L, &[f32]) {
        (
            self.lists[self.parents + neighbor],
            self.neighbor_centroids.row(neighbor),
        )
    }

    /// Batch positions routed to `list`.
    fn staged(&self, list: usize) -> &[usize] {
        self.batch.by_route.get(self.lists[list])
    }

    /// Canonical vectors of `list`'s existing members followed by its batch points.
    fn points(&self, list: usize) -> impl Iterator<Item = &[f32]> {
        self.members.vectors(list).chain(
            self.staged(list)
                .iter()
                .map(|&position| self.batch.vectors.row(position)),
        )
    }

    /// The `index`-th vector of [`Self::points`].
    fn point(&self, list: usize, index: usize) -> &[f32] {
        let members = self.members.range(list);
        if index < members.len() {
            self.members.vector(members.start + index)
        } else {
            self.batch
                .vectors
                .row(self.staged(list)[index - members.len()])
        }
    }
}
