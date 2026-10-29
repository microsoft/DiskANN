/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Small benchmark-only provider: unique u32 IDs, uncompressed f32 rows, staged writes.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use diskann::{
    ivf::{
        pending::{Membership, PendingLists},
        traits::{
            BuildAccessor, BuildStrategy, CoarseAccessor, InsertStrategy, ListAccessor, Provider,
            SelectedList,
        },
        workingset::{Batch, ListView, PointRef, WorkingSet},
        SearchProvider,
    },
    provider::DefaultContext,
    ANNError, ANNResult,
};
use diskann_utils::{
    future::SendFuture,
    views::rowmajor::{self, MatrixMut},
};
use diskann_vector::{distance::SquaredL2, PureDistanceFunction};

pub(super) struct List {
    pub(super) centroid: Vec<f32>,
    pub(super) members: Vec<u32>,
}

pub(super) struct InMemProvider {
    pub(super) dim: usize,
    pub(super) points: BTreeMap<u32, Vec<f32>>,
    pub(super) lists: BTreeMap<u32, List>,
    pub(super) next_list: u32,
}

impl InMemProvider {
    pub(super) fn new(dim: usize) -> Self {
        Self {
            dim,
            points: BTreeMap::new(),
            lists: BTreeMap::new(),
            next_list: 0,
        }
    }
}

impl Provider for InMemProvider {
    type Context = DefaultContext;
    type InternalId = u32;
    type ExternalId = u32;
    type ListId = u32;
    type Error = ANNError;

    fn to_internal_id(&self, _: &DefaultContext, id: &u32) -> ANNResult<u32> {
        self.points
            .contains_key(id)
            .then_some(*id)
            .ok_or_else(|| ANNError::message("unknown point"))
    }
    fn to_external_id(&self, context: &DefaultContext, id: u32) -> ANNResult<u32> {
        self.to_internal_id(context, &id)
    }
}

fn select<'a>(
    rows: impl Iterator<Item = (u32, &'a [f32])>,
    query: &[f32],
    nprobe: usize,
) -> Vec<SelectedList<u32>> {
    let mut selected: Vec<_> = rows
        .map(|(id, vector)| SelectedList {
            id,
            distance: SquaredL2::evaluate(query, vector),
        })
        .collect();
    selected.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
    selected.truncate(nprobe);
    selected
}

impl SearchProvider for InMemProvider {
    fn dim(&self) -> usize {
        self.dim
    }

    fn search_lists<'a>(
        &'a self,
        _: &'a DefaultContext,
        query: &'a [f32],
        nprobe: usize,
    ) -> impl SendFuture<ANNResult<WorkingSet<'a, u32, u32>>> {
        async move {
            if self.lists.is_empty() {
                return Err(ANNError::message("IVF index is not initialized"));
            }
            let selected = select(
                self.lists
                    .iter()
                    .map(|(&id, list)| (id, list.centroid.as_slice())),
                query,
                nprobe,
            );
            selected
                .into_iter()
                .map(|selected| {
                    let list = &self.lists[&selected.id];
                    let points = list
                        .members
                        .iter()
                        .map(|&id| {
                            let vector = self
                                .points
                                .get(&id)
                                .ok_or_else(|| ANNError::message("missing point"))?;
                            Ok(PointRef {
                                id,
                                vector: vector.as_slice(),
                            })
                        })
                        .collect::<ANNResult<Vec<_>>>()?;
                    Ok(ListView {
                        id: selected.id,
                        points,
                    })
                })
                .collect()
        }
    }
}

pub(super) struct Build<'p> {
    store: &'p mut InMemProvider,
    pending: PendingLists<u32, u32>,
    created: BTreeMap<u32, Vec<f32>>,
    batch: Option<Arc<Batch<u32>>>,
    next_list: u32,
}

impl Build<'_> {
    fn stored_members(&self, list: u32) -> ANNResult<&[u32]> {
        match self.store.lists.get(&list) {
            Some(list) => Ok(&list.members),
            None if self.created.contains_key(&list) => Ok(&[]),
            None => Err(ANNError::message("unknown list")),
        }
    }
}

pub(super) struct Coarse<'a> {
    stored: &'a BTreeMap<u32, List>,
    created: &'a BTreeMap<u32, Vec<f32>>,
    pending: &'a PendingLists<u32, u32>,
}

impl CoarseAccessor for Coarse<'_> {
    type ListId = u32;
    fn len(&self) -> usize {
        self.stored
            .keys()
            .chain(self.created.keys())
            .filter(|&&id| !self.pending.is_retired(id))
            .count()
    }
    fn select(
        &mut self,
        query: &[f32],
        nprobe: usize,
    ) -> impl SendFuture<ANNResult<Vec<SelectedList<u32>>>> {
        async move {
            Ok(select(
                self.stored
                    .iter()
                    .map(|(&id, list)| (id, list.centroid.as_slice()))
                    .chain(self.created.iter().map(|(&id, c)| (id, c.as_slice())))
                    .filter(|(id, _)| !self.pending.is_retired(*id)),
                query,
                nprobe,
            ))
        }
    }
}

impl ListAccessor for &mut Build<'_> {
    type InternalId = u32;
    type ListId = u32;
    fn list_size(&mut self, list: u32) -> impl SendFuture<ANNResult<usize>> {
        async move {
            self.pending
                .list_size(list, self.stored_members(list)?.len())
        }
    }
    fn append_members(&mut self, list: u32, members: Vec<u32>) -> impl SendFuture<ANNResult<()>> {
        async move {
            self.stored_members(list)?;
            self.pending.append(list, members)
        }
    }
    fn set_members(&mut self, list: u32, members: Vec<u32>) -> impl SendFuture<ANNResult<()>> {
        async move {
            self.stored_members(list)?;
            self.pending.set(list, members)
        }
    }
    fn create_list(
        &mut self,
        centroid: &[f32],
        members: Vec<u32>,
    ) -> impl SendFuture<ANNResult<u32>> {
        async move {
            if centroid.len() != self.store.dim || !centroid.iter().all(|x| x.is_finite()) {
                return Err(ANNError::message("invalid centroid"));
            }
            let id = self.next_list;
            self.next_list = id
                .checked_add(1)
                .ok_or_else(|| ANNError::message("list ID overflow"))?;
            self.created.insert(id, centroid.to_vec());
            self.pending.set(id, members)?;
            Ok(id)
        }
    }
    fn retire_list(&mut self, list: u32) -> impl SendFuture<ANNResult<()>> {
        async move {
            self.stored_members(list)?;
            self.pending.retire(list)
        }
    }
}

impl<'p> BuildAccessor for Build<'p> {
    type InternalId = u32;
    type ListId = u32;
    type Coarse<'a>
        = Coarse<'a>
    where
        Self: 'a;
    type Lists<'a>
        = &'a mut Self
    where
        Self: 'a;
    fn dim(&self) -> usize {
        self.store.dim
    }
    fn coarse(&mut self) -> Self::Coarse<'_> {
        Coarse {
            stored: &self.store.lists,
            created: &self.created,
            pending: &self.pending,
        }
    }
    fn lists(&mut self) -> Self::Lists<'_> {
        self
    }
    fn fill(&mut self, lists: &[u32]) -> impl SendFuture<ANNResult<WorkingSet<'_, u32, u32>>> {
        async move {
            let mut view = Vec::with_capacity(lists.len());
            for &list in lists {
                let mut points = Vec::new();
                for id in self.pending.members(list, self.stored_members(list)?)? {
                    let vector = self
                        .batch
                        .as_ref()
                        .and_then(|batch| batch.get(id))
                        .or_else(|| self.store.points.get(&id).map(Vec::as_slice))
                        .ok_or_else(|| ANNError::message("missing point"))?;
                    points.push(PointRef { id, vector });
                }
                view.push(ListView { id: list, points });
            }
            Ok(view)
        }
    }
    fn finish(self) -> impl SendFuture<ANNResult<()>> {
        async move {
            if let Some(batch) = self.batch {
                for point in batch.points() {
                    self.store.points.insert(point.id, point.vector.to_vec());
                }
            }
            for (id, centroid) in self.created {
                self.store.lists.insert(
                    id,
                    List {
                        centroid,
                        members: Vec::new(),
                    },
                );
            }
            for (id, change) in self.pending.into_changes() {
                if let Membership::Retire = change {
                    self.store.lists.remove(&id);
                } else {
                    let list = self
                        .store
                        .lists
                        .get_mut(&id)
                        .ok_or_else(|| ANNError::message("missing list"))?;
                    match change {
                        Membership::Append(ids) => list.members.extend(ids),
                        Membership::Replace(ids) => list.members = ids,
                        Membership::Retire => unreachable!(),
                    }
                }
            }
            self.store.next_list = self.next_list;
            Ok(())
        }
    }
}

pub(super) struct Strategy;
impl<'a> BuildStrategy<'a, InMemProvider> for Strategy {
    type Accessor = Build<'a>;
    type Error = ANNError;
    fn build_accessor(
        &'a self,
        store: &'a mut InMemProvider,
        _: &'a DefaultContext,
    ) -> ANNResult<Build<'a>> {
        let next_list = store.next_list;
        Ok(Build {
            store,
            pending: PendingLists::default(),
            created: BTreeMap::new(),
            batch: None,
            next_list,
        })
    }
}
impl<'a, 'v> InsertStrategy<'a, InMemProvider, &'v [f32]> for Strategy {
    fn prepare(
        &self,
        accessor: &mut Self::Accessor,
        points: &[(u32, &'v [f32])],
    ) -> impl SendFuture<ANNResult<Arc<Batch<u32>>>> {
        async move {
            let mut vectors = rowmajor::Owned::try_from_element(points.len(), accessor.dim(), 0.0)?;
            let mut ids = Vec::with_capacity(points.len());
            let mut seen = BTreeSet::new();
            for ((id, vector), out) in points.iter().zip(vectors.rows_mut()) {
                if vector.len() != out.len()
                    || !vector.iter().all(|x| x.is_finite())
                    || accessor.store.points.contains_key(id)
                    || !seen.insert(*id)
                {
                    return Err(ANNError::message("invalid or duplicate input point"));
                }
                ids.push(*id);
                out.copy_from_slice(vector);
            }
            let batch = Arc::new(Batch::new(ids, vectors)?);
            accessor.batch = Some(Arc::clone(&batch));
            Ok(batch)
        }
    }
}
