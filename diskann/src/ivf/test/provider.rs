/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! In-memory provider for exercising the dynamic IVF insert path.
//!
//! The provider stores plain owned state and mutates it in place through the
//! exclusive borrow lent to its maintenance accessor. It writes vector rows in
//! reverse order to exercise out-of-order reads, and [`Faults`] injects inconsistent
//! responses.

use std::collections::{BTreeMap, BTreeSet};

use diskann_utils::{future::SendFuture, views::MutMatrixView};

use crate::{
    ANNError, ANNErrorKind,
    ivf::{
        dynamic::{
            Apply, CentroidIndex, ListMetadata, ListProvider, MaintenanceAccessor,
            MaintenanceStrategy, SelectedList,
        },
        update::{Appends, Bootstrap, SplitInsert},
    },
    provider::{DataProvider, DefaultContext, HasId, NoopGuard},
};

fn error(message: impl Into<String>) -> ANNError {
    ANNError::message(ANNErrorKind::IndexError, message.into())
}

fn squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Inconsistent responses the provider can inject.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Faults {
    /// Leave the last row of every `read_vectors` call unwritten.
    pub(crate) skip_vector: bool,
    /// Write NaN into the last row of every `read_vectors` call.
    pub(crate) nan_vector: bool,
    /// Return the first staged id again from every later `stage_insert`.
    pub(crate) repeat_stage: bool,
    /// Reserve list ids starting from zero, which may be live.
    pub(crate) reuse_lists: bool,
}

/// Exact centroid catalog.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Centroids(BTreeMap<u32, Box<[f32]>>);

impl CentroidIndex for Centroids {
    type ListId = u32;
    type Error = ANNError;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn centroid(&self, id: u32) -> Option<&[f32]> {
        self.0.get(&id).map(|centroid| &**centroid)
    }

    fn select(
        &self,
        query: &[f32],
        nprobe: usize,
    ) -> impl SendFuture<Result<Vec<SelectedList<u32>>, ANNError>> {
        let mut selected: Vec<_> = self
            .0
            .iter()
            .map(|(&id, centroid)| SelectedList {
                id,
                distance: squared(query, centroid),
            })
            .collect();
        selected.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
        selected.truncate(nprobe);
        std::future::ready(Ok(selected))
    }
}

/// An in-memory dynamic IVF provider. Point `i` has internal id `i`.
#[derive(Debug, Clone)]
pub(crate) struct Provider {
    dim: usize,
    /// External id and canonical vector of every committed point.
    points: Vec<(u32, Box<[f32]>)>,
    /// Current list of every committed point.
    assignment: Vec<u32>,
    lists: BTreeMap<u32, Vec<u32>>,
    centroids: Centroids,
    retired: BTreeSet<u32>,
    next_list: u32,
    faults: Faults,
}

impl Provider {
    /// An empty provider for vectors of dimension `dim`.
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            dim,
            points: Vec::new(),
            assignment: Vec::new(),
            lists: BTreeMap::new(),
            centroids: Centroids::default(),
            retired: BTreeSet::new(),
            next_list: 0,
            faults: Faults::default(),
        }
    }

    /// A provider whose list `i` has centroid `lists[i].0` and members `lists[i].1`,
    /// which receive consecutive internal and external ids.
    pub(crate) fn from_lists(dim: usize, lists: &[(&[f32], &[&[f32]])]) -> Self {
        let mut provider = Self::new(dim);
        for (list, &(centroid, members)) in (0..).zip(lists) {
            provider.centroids.0.insert(list, centroid.into());
            let ids = members
                .iter()
                .map(|&vector| {
                    let id = provider.points.len() as u32;
                    provider.points.push((id, vector.into()));
                    provider.assignment.push(list);
                    id
                })
                .collect();
            provider.lists.insert(list, ids);
            provider.next_list = list + 1;
        }
        provider
    }

    /// Inject `faults` into subsequent reads.
    pub(crate) fn with_faults(mut self, faults: Faults) -> Self {
        self.faults = faults;
        self
    }

    /// Whether both providers hold the same points, lists, and centroids.
    pub(crate) fn same_contents(&self, other: &Self) -> bool {
        self.points == other.points
            && self.assignment == other.assignment
            && self.lists == other.lists
            && self.centroids == other.centroids
            && self.retired == other.retired
    }

    /// Live list ids, ascending.
    pub(crate) fn lists(&self) -> Vec<u32> {
        self.lists.keys().copied().collect()
    }

    /// Number of retired lists.
    pub(crate) fn num_retired(&self) -> usize {
        self.retired.len()
    }

    /// Number of committed points.
    pub(crate) fn len(&self) -> usize {
        self.points.len()
    }

    /// Centroid of a live list.
    pub(crate) fn centroid(&self, list: u32) -> Option<&[f32]> {
        self.centroids.centroid(list)
    }

    /// Vectors of a live list's members, sorted.
    pub(crate) fn member_vectors(&self, list: u32) -> Vec<Vec<f32>> {
        let mut vectors: Vec<Vec<f32>> = self.lists[&list]
            .iter()
            .map(|&id| self.points[id as usize].1.to_vec())
            .collect();
        vectors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vectors
    }

    /// The list holding the point whose vector is `vector`.
    pub(crate) fn list_of(&self, vector: &[f32]) -> Option<u32> {
        self.points
            .iter()
            .position(|(_, point)| &**point == vector)
            .map(|id| self.assignment[id])
    }

    /// Assert every structural invariant of the partition.
    pub(crate) fn check(&self) {
        assert_eq!(self.lists.len(), self.centroids.0.len());
        let mut seen = vec![0usize; self.points.len()];
        for (&list, members) in &self.lists {
            assert!(
                self.centroids.0.contains_key(&list),
                "list {list} has no centroid"
            );
            assert!(!self.retired.contains(&list), "list {list} is retired");
            for &id in members {
                seen[id as usize] += 1;
                assert_eq!(self.assignment[id as usize], list, "point {id}");
            }
        }
        assert!(
            seen.iter().all(|&count| count == 1),
            "every point must be in exactly one list"
        );
        assert!(self.centroids.0.values().all(|c| c.len() == self.dim));
        assert!(self.points.iter().all(|(_, v)| v.len() == self.dim));
    }
}

impl DataProvider for Provider {
    type Context = DefaultContext;
    type InternalId = u32;
    type ExternalId = u32;
    type Error = ANNError;
    type Guard = NoopGuard<u32>;

    fn to_internal_id(&self, _: &DefaultContext, gid: &u32) -> Result<u32, ANNError> {
        self.points
            .iter()
            .position(|(external, _)| external == gid)
            .map(|id| id as u32)
            .ok_or_else(|| error(format!("no point with external id {gid}")))
    }

    fn to_external_id(&self, _: &DefaultContext, id: u32) -> Result<u32, ANNError> {
        self.points
            .get(id as usize)
            .map(|(external, _)| *external)
            .ok_or_else(|| error(format!("no point {id}")))
    }
}

impl ListProvider for Provider {
    type ListId = u32;
}

/// Maintenance accessor over an exclusively borrowed [`Provider`].
pub(crate) struct Accessor<'a> {
    provider: &'a mut Provider,
    /// External id and vector of points staged by this accessor, in id order.
    staged: Vec<(u32, Box<[f32]>)>,
}

impl Accessor<'_> {
    fn vector(&self, id: u32) -> Result<&[f32], ANNError> {
        let id = id as usize;
        let committed = self.provider.points.len();
        let point = if id < committed {
            self.provider.points.get(id)
        } else {
            self.staged.get(id - committed)
        };
        point
            .map(|(_, vector)| &**vector)
            .ok_or_else(|| error(format!("no point {id}")))
    }

    fn commit_staged(&mut self) {
        for point in self.staged.drain(..) {
            self.provider.points.push(point);
            self.provider.assignment.push(u32::MAX);
        }
    }

    fn install(&mut self, id: u32, vector: &[f32]) -> Result<(), ANNError> {
        if self.provider.retired.contains(&id) || self.provider.lists.contains_key(&id) {
            return Err(error(format!("list id {id} is reused")));
        }
        self.provider.centroids.0.insert(id, vector.into());
        self.provider.lists.insert(id, Vec::new());
        Ok(())
    }

    fn place(&mut self, list: u32, ids: &[u32]) -> Result<(), ANNError> {
        self.provider
            .lists
            .get_mut(&list)
            .ok_or_else(|| error(format!("list {list} is not live")))?
            .extend_from_slice(ids);
        for &id in ids {
            let slot = self
                .provider
                .assignment
                .get_mut(id as usize)
                .ok_or_else(|| error(format!("no point {id}")))?;
            *slot = list;
        }
        Ok(())
    }
}

impl HasId for Accessor<'_> {
    type Id = u32;
}

impl<'b> MaintenanceAccessor<&'b [f32]> for Accessor<'_> {
    type ExternalId = u32;
    type ListId = u32;
    type Centroids = Centroids;
    type Error = ANNError;

    fn dim(&self) -> usize {
        self.provider.dim
    }

    fn centroids(&self) -> &Centroids {
        &self.provider.centroids
    }

    async fn list_metadata(&mut self, list: u32) -> Result<ListMetadata, ANNError> {
        let len = self
            .provider
            .lists
            .get(&list)
            .ok_or_else(|| error(format!("list {list} is not live")))?
            .len();
        Ok(ListMetadata { len })
    }

    async fn read_members(&mut self, list: u32) -> Result<&[u32], ANNError> {
        self.provider
            .lists
            .get(&list)
            .map(Vec::as_slice)
            .ok_or_else(|| error(format!("list {list} is not live")))
    }

    async fn read_vectors(
        &mut self,
        ids: &[u32],
        mut out: MutMatrixView<'_, f32>,
    ) -> Result<(), ANNError> {
        let faults = self.provider.faults;
        for (row, &id) in ids.iter().enumerate().rev() {
            let last = row + 1 == ids.len();
            if last && faults.skip_vector {
                continue;
            }
            let out = out.row_mut(row);
            out.copy_from_slice(self.vector(id)?);
            if last && faults.nan_vector {
                out[0] = f32::NAN;
            }
        }
        Ok(())
    }

    async fn stage_insert(&mut self, id: &u32, element: &'b [f32]) -> Result<u32, ANNError> {
        if element.len() != self.provider.dim {
            return Err(error(format!("point {id} has the wrong dimension")));
        }
        let known =
            |points: &[(u32, Box<[f32]>)]| points.iter().any(|(external, _)| external == id);
        if known(&self.provider.points) || known(&self.staged) {
            return Err(error(format!("external id {id} already exists")));
        }
        let committed = self.provider.points.len();
        let internal = if self.provider.faults.repeat_stage && !self.staged.is_empty() {
            committed
        } else {
            committed + self.staged.len()
        };
        self.staged.push((*id, element.into()));
        Ok(internal as u32)
    }

    async fn reserve_list_ids(&mut self, count: usize) -> Result<Vec<u32>, ANNError> {
        if self.provider.faults.reuse_lists {
            return Ok((0..count as u32).collect());
        }
        let start = self.provider.next_list;
        self.provider.next_list += count as u32;
        Ok((start..self.provider.next_list).collect())
    }
}

impl Apply<Bootstrap<u32>> for Accessor<'_> {
    type Error = ANNError;

    async fn apply(mut self, update: Bootstrap<u32>) -> Result<(), ANNError> {
        for (id, vector) in update.centroids().iter() {
            self.install(id, vector)?;
        }
        Ok(())
    }
}

impl Apply<Appends<u32, u32>> for Accessor<'_> {
    type Error = ANNError;

    async fn apply(mut self, update: Appends<u32, u32>) -> Result<(), ANNError> {
        self.commit_staged();
        for (list, ids) in update.iter() {
            self.place(list, ids)?;
        }
        Ok(())
    }
}

impl Apply<SplitInsert<u32, u32>> for Accessor<'_> {
    type Error = ANNError;

    async fn apply(mut self, update: SplitInsert<u32, u32>) -> Result<(), ANNError> {
        self.commit_staged();
        let (appends, splits, reassignments) = update.into_parts();

        // Install every child first, since any part of the update may target one.
        for split in splits.iter() {
            for (child, centroid) in split.children().into_iter().zip(split.centroids()) {
                self.install(child, centroid)?;
            }
        }
        for split in splits.iter() {
            let parent = split.parent();
            let mut members = self
                .provider
                .lists
                .remove(&parent)
                .ok_or_else(|| error(format!("split parent {parent} is not live")))?;
            self.provider.centroids.0.remove(&parent);
            self.provider.retired.insert(parent);

            let mut evacuated: Vec<u32> = split
                .evacuations()
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect();
            members.sort_unstable();
            evacuated.sort_unstable();
            if members != evacuated {
                return Err(error(format!(
                    "split of {parent} does not evacuate its members"
                )));
            }
            for (list, ids) in split.evacuations() {
                self.place(list, ids)?;
            }
        }
        for (transfer, ids) in reassignments.iter() {
            let source = self
                .provider
                .lists
                .get_mut(&transfer.source())
                .ok_or_else(|| error(format!("list {} is not live", transfer.source())))?;
            let before = source.len();
            source.retain(|id| !ids.contains(id));
            if before - source.len() != ids.len() {
                return Err(error(format!(
                    "reassigned points are not all in list {}",
                    transfer.source()
                )));
            }
            self.place(transfer.destination(), ids)?;
        }
        for (list, ids) in appends.iter() {
            self.place(list, ids)?;
        }
        Ok(())
    }
}

/// Strategy lending a [`Provider`] to one [`Accessor`].
#[derive(Debug, Default)]
pub(crate) struct Strategy;

impl<'a> MaintenanceStrategy<'a, Provider, &[f32]> for Strategy {
    type MaintenanceAccessor = Accessor<'a>;
    type Error = ANNError;

    fn maintenance_accessor(
        &'a self,
        provider: &'a mut Provider,
        _context: &'a DefaultContext,
    ) -> Result<Accessor<'a>, ANNError> {
        Ok(Accessor {
            provider,
            staged: Vec::new(),
        })
    }
}
