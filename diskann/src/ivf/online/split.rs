/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Split planning: gather each split's region, fit children, and place every affected
//! point.

use diskann_utils::views::Matrix;
use rand::{Rng, rngs::StdRng};

use super::{
    batch::StagedBatch,
    check_reserved, index_error,
    kernels::{LloydScratch, lloyd, nearest},
    read_rows,
};
use crate::{
    ANNResult,
    error::ErrorExt,
    ivf::{
        DynamicIvfConfig,
        dynamic::{CentroidIndex, MaintenanceAccessor, Provider},
        grouped::Csr,
        update::{CentroidDelta, InsertionUpdate, PointDelta},
    },
    utils::VectorId,
};

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
    /// Existing members of every list, grouped like `lists`.
    member_ids: Csr<Id>,
    /// Row `i` is the canonical vector of `member_ids.values()[i]`.
    member_vectors: Matrix<f32>,
    /// Reserved ids for the children, two per region.
    child_ids: Vec<L>,
}

impl<Id: VectorId, L: VectorId> SplitPlan<Id, L> {
    /// Perform every accessor call the split needs.
    ///
    /// Reserves two child ids per parent, selects each parent's nearest surviving
    /// lists as its neighbors, and reads the members and canonical vectors of every
    /// parent and neighbor. `parents` must be ascending.
    ///
    /// # Errors
    ///
    /// Fails if an accessor call fails or the accessor's view is inconsistent.
    pub(in crate::ivf) async fn gather<P, A>(
        accessor: &mut A,
        config: &DynamicIvfConfig,
        batch: StagedBatch<Id, L>,
        parents: Vec<L>,
    ) -> ANNResult<Self>
    where
        P: Provider<InternalId = Id, ListId = L>,
        A: MaintenanceAccessor<P>,
    {
        let count = 2 * parents.len();
        let child_ids = accessor
            .reserve_list_ids(count)
            .await
            .escalate("split must reserve child list ids")?;
        check_reserved(accessor.centroids(), &child_ids, count)?;

        let dim = accessor.dim();
        let centroids = accessor.centroids();

        // Every parent is retired, so no parent is ever a neighbor.
        let mut by_region: Csr<L> = Csr::default();
        for &parent in &parents {
            let anchor = centroids
                .centroid(parent)
                .ok_or_else(|| index_error(format!("split parent {parent} is not live")))?;
            let selected = centroids
                .select(anchor, config.reassign_neighbors.saturating_add(1))
                .await
                .escalate("split must find neighbor lists")?;
            by_region.push(
                selected
                    .iter()
                    .map(|selected| selected.id)
                    .filter(|id| parents.binary_search(id).is_err())
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

        let num_parents = parents.len();
        let mut lists = parents;
        lists.extend_from_slice(&distinct);

        let mut member_ids = Csr::default();
        for &list in &lists {
            let members = accessor
                .read_members(list)
                .await
                .escalate("split must read list members")?;
            member_ids.push(members.iter().copied());
        }
        let member_vectors = read_rows(accessor, member_ids.values(), dim).await?;

        Ok(Self {
            batch,
            lists,
            parents: num_parents,
            neighbors,
            regions,
            neighbor_centroids,
            member_ids,
            member_vectors,
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
    ) -> ANNResult<InsertionUpdate<Id, L>> {
        let children = self.fit_children(iterations, rng)?;
        let mut routes = std::mem::take(&mut self.batch.routes);
        let destinations = self.place(&children, &mut routes)?;
        Ok(self.package(&children, &routes, &destinations))
    }

    /// Fit two children per region; rows `2r` and `2r + 1` split region `r`.
    fn fit_children(&self, iterations: usize, rng: &mut StdRng) -> ANNResult<Matrix<f32>> {
        let dim = self.batch.vectors.ncols();
        let mut centers = Matrix::new(0.0f32, 2 * self.parents, dim);
        let mut scratch = LloydScratch::default();
        for region in 0..self.parents {
            let span = self.member_ids.group(region).len() + self.staged(region).len();
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
    /// Row `r` of `children` is the centroid of child `r`. Returns the existing
    /// members' lists, aligned with their positions, and overwrites the routes of the
    /// batch points it places.
    fn place(&self, children: &Matrix<f32>, routes: &mut [L]) -> ANNResult<Vec<L>> {
        let mut destinations = vec![L::default(); self.member_ids.values().len()];
        let child = |row: usize| (self.child_ids[row], children.row(row));
        let mut candidates: Vec<(L, &[f32])> = Vec::new();

        for region in 0..self.parents {
            candidates.clear();
            for &neighbor in self.neighbors.group(region) {
                candidates.push(self.neighbor(neighbor));
            }
            candidates.extend([child(2 * region), child(2 * region + 1)]);
            self.assign(region, &candidates, &mut destinations, routes)?;
        }

        // The list's own centroid comes first, so ties keep the point in place.
        for neighbor in 0..self.regions.num_groups() {
            candidates.clear();
            candidates.push(self.neighbor(neighbor));
            for &region in self.regions.group(neighbor) {
                candidates.extend([child(2 * region), child(2 * region + 1)]);
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
        for (destination, vector) in destinations[self.member_ids.range(list)]
            .iter_mut()
            .zip(self.member_rows(list))
        {
            *destination = nearest(vector, candidates).ok_or_else(no_candidates)?;
        }
        for &position in self.staged(list) {
            routes[position] =
                nearest(self.batch.vectors.row(position), candidates).ok_or_else(no_candidates)?;
        }
        Ok(())
    }

    /// Assemble the update: install the children, retire the parents, append the
    /// batch, and move every existing point whose list changed.
    fn package(
        &self,
        children: &Matrix<f32>,
        routes: &[L],
        destinations: &[L],
    ) -> InsertionUpdate<Id, L> {
        let installs = self
            .child_ids
            .iter()
            .zip(children.row_iter())
            .map(|(&id, centroid)| CentroidDelta::Install {
                id,
                centroid: centroid.into(),
            });
        let retires = self.lists[..self.parents]
            .iter()
            .map(|&id| CentroidDelta::Retire { id });

        let appends = self
            .batch
            .ids
            .iter()
            .zip(routes)
            .map(|(&id, &to)| PointDelta::Append { id, to });
        // A parent is never a destination, so all its members move.
        let moves = self.lists.iter().enumerate().flat_map(|(list, &from)| {
            self.member_ids
                .group(list)
                .iter()
                .zip(&destinations[self.member_ids.range(list)])
                .filter(move |&(_, &to)| to != from)
                .map(move |(&id, &to)| PointDelta::Move { id, from, to })
        });

        InsertionUpdate::new(
            installs.chain(retires).collect(),
            appends.chain(moves).collect(),
        )
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

    /// Canonical vectors of `list`'s existing members.
    fn member_rows(&self, list: usize) -> impl Iterator<Item = &[f32]> {
        self.member_ids
            .range(list)
            .map(|row| self.member_vectors.row(row))
    }

    /// Canonical vectors of `list`'s existing members followed by its batch points.
    fn points(&self, list: usize) -> impl Iterator<Item = &[f32]> {
        self.member_rows(list).chain(
            self.staged(list)
                .iter()
                .map(|&position| self.batch.vectors.row(position)),
        )
    }

    /// The `index`-th vector of [`Self::points`].
    fn point(&self, list: usize, index: usize) -> &[f32] {
        let members = self.member_ids.range(list);
        if index < members.len() {
            self.member_vectors.row(members.start + index)
        } else {
            self.batch
                .vectors
                .row(self.staged(list)[index - members.len()])
        }
    }
}
