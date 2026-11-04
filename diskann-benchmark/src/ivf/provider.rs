/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Coordinates are loaded once. A build accessor stages memberships and publishes both
//! split and reassignment together; a dropped accessor leaves the store unchanged.

use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    ops::Range,
    sync::Arc,
};

use diskann::{
    ivf::{
        region::{self, View},
        traits::{BuildAccessor, BuildStrategy, HasIds, Nearest},
    },
    provider::DefaultContext,
    ANNError, ANNResult,
};
use diskann_utils::{
    future::SendFuture,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use diskann_vector::{distance::SquaredL2, PureDistanceFunction};
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::data::Rows;

#[derive(Clone, Debug)]
pub(super) struct List {
    pub(super) centroid: Vec<f32>,
    pub(super) members: Vec<u32>,
}

pub(super) struct Store {
    pub(super) points: Arc<Rows>,
    pub(super) lists: BTreeMap<u32, List>,
    pub(super) inserted: usize,
    next_list: u32,
}

pub(super) struct InMemProvider {
    store: RwLock<Store>,
}

impl InMemProvider {
    pub(super) fn new(points: Arc<Rows>) -> ANNResult<Self> {
        let first = points
            .first()
            .filter(|row| !row.is_empty())
            .ok_or_else(|| ANNError::message("empty IVF dataset"))?;
        // Seed one empty list; routing inserts every point, including the first row.
        let lists = BTreeMap::from([(
            0,
            List {
                centroid: first.clone(),
                members: Vec::new(),
            },
        )]);
        Ok(Self {
            store: RwLock::new(Store {
                points,
                lists,
                inserted: 0,
                next_list: 1,
            }),
        })
    }

    pub(super) fn read(&self) -> ANNResult<RwLockReadGuard<'_, Store>> {
        self.store.try_read().map_err(ANNError::new)
    }
}

impl HasIds for InMemProvider {
    type Id = u32;
    type ListId = u32;
}

pub(super) struct Build<'a> {
    store: RwLockWriteGuard<'a, Store>,
    changes: BTreeMap<u32, Option<List>>,
    next_list: u32,
    end: usize,
}

impl HasIds for Build<'_> {
    type Id = u32;
    type ListId = u32;
}

impl<'a> BuildStrategy<'a, Range<usize>, DefaultContext> for InMemProvider {
    type Accessor = Build<'a>;
    type Error = ANNError;

    fn build_accessor(&'a self, _: &'a DefaultContext) -> ANNResult<Build<'a>> {
        let store = self.store.try_write().map_err(ANNError::new)?;
        let next_list = store.next_list;
        let end = store.inserted;
        Ok(Build {
            store,
            changes: BTreeMap::new(),
            next_list,
            end,
        })
    }
}

pub(super) struct Coarse<'a> {
    stored: &'a BTreeMap<u32, List>,
    changes: &'a BTreeMap<u32, Option<List>>,
}

impl Coarse<'_> {
    fn lists(&self) -> impl Iterator<Item = (u32, &List)> {
        self.stored
            .iter()
            .filter(|(id, _)| !self.changes.contains_key(id))
            .map(|(&id, list)| (id, list))
            .chain(
                self.changes
                    .iter()
                    .filter_map(|(&id, list)| list.as_ref().map(|list| (id, list))),
            )
    }

    fn get(&self, id: u32) -> Option<&List> {
        self.changes
            .get(&id)
            .map_or_else(|| self.stored.get(&id), Option::as_ref)
    }
}

pub(super) fn select<'a>(
    lists: impl Iterator<Item = (u32, &'a List)>,
    query: &[f32],
    nprobe: usize,
    distance: impl Fn(&[f32], &[f32]) -> f32,
) -> Vec<(u32, f32)> {
    let mut selected: Vec<_> = lists
        .map(|(id, list)| (id, distance(query, list.centroid.as_slice())))
        .collect();
    let compare = |a: &(u32, f32), b: &(u32, f32)| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0));
    if selected.len() > nprobe {
        selected.select_nth_unstable_by(nprobe, compare);
        selected.truncate(nprobe);
    }
    selected.sort_unstable_by(compare);
    selected
}

impl Nearest for Coarse<'_> {
    type Id = u32;

    fn find<P, F>(
        &self,
        id: u32,
        nprobe: NonZeroUsize,
        pred: P,
        mut emit: F,
    ) -> impl SendFuture<ANNResult<usize>>
    where
        P: Fn(u32) -> bool + Send,
        F: FnMut(u32, f32) + Send,
    {
        async move {
            let query = &self
                .get(id)
                .ok_or_else(|| ANNError::message("unknown IVF list"))?
                .centroid;
            let selected = select(
                self.lists().filter(|(id, _)| pred(*id)),
                query,
                nprobe.get(),
                |query, centroid| SquaredL2::evaluate(query, centroid),
            );
            for &(id, distance) in &selected {
                emit(id, distance);
            }
            Ok(selected.len())
        }
    }
}

struct Source {
    list: List,
    vectors: rowmajor::Owned<f32>,
}

pub(super) struct Region {
    sources: BTreeMap<u32, Source>,
}

impl View<u32, u32> for Region {
    type VectorsRef<'a> = rowmajor::Ref<'a, f32>;
    type Vectors<'a> = rowmajor::Ref<'a, f32>;

    fn get_list(&self, id: u32) -> Option<region::List<'_, u32, u32>> {
        self.sources.get(&id).map(|source| region::List {
            id,
            centroid: &source.list.centroid,
            members: &source.list.members,
        })
    }

    fn get_vectors(&self, id: u32) -> Option<Self::Vectors<'_>> {
        self.sources.get(&id).map(|source| source.vectors.as_view())
    }
}

pub(super) struct Update<'a> {
    stored: &'a BTreeMap<u32, List>,
    changes: &'a mut BTreeMap<u32, Option<List>>,
    filled: BTreeSet<u32>,
    next_list: &'a mut u32,
    dim: usize,
}

impl Update<'_> {
    fn edit(&mut self, id: u32) -> ANNResult<&mut List> {
        if !self.filled.contains(&id) {
            return Err(ANNError::message("IVF update targets an unfilled list"));
        }
        self.changes
            .entry(id)
            .or_insert_with(|| self.stored.get(&id).cloned())
            .as_mut()
            .ok_or_else(|| ANNError::message("IVF update targets a retired list"))
    }
}

impl region::Update<u32, u32> for Update<'_> {
    fn append_members(&mut self, id: u32, members: &[u32]) -> ANNResult<()> {
        if !members.is_empty() {
            self.edit(id)?.members.extend_from_slice(members);
        }
        Ok(())
    }

    fn remove_members(&mut self, id: u32, members: &[u32]) -> ANNResult<()> {
        if !members.is_empty() {
            let removed: BTreeSet<_> = members.iter().copied().collect();
            self.edit(id)?.members.retain(|id| !removed.contains(id));
        }
        Ok(())
    }

    fn retire(&mut self, id: u32) -> ANNResult<()> {
        self.edit(id)?;
        self.changes.insert(id, None);
        Ok(())
    }

    fn create(&mut self, centroid: &[f32], members: &[u32]) -> ANNResult<u32> {
        if centroid.len() != self.dim || !centroid.iter().all(|x| x.is_finite()) {
            return Err(ANNError::message("invalid IVF centroid"));
        }
        let id = *self.next_list;
        *self.next_list = id
            .checked_add(1)
            .ok_or_else(|| ANNError::message("list ID overflow"))?;
        self.changes.insert(
            id,
            Some(List {
                centroid: centroid.to_vec(),
                members: members.to_vec(),
            }),
        );
        Ok(id)
    }

    fn finish(self) -> impl SendFuture<ANNResult<()>> {
        async { Ok(()) }
    }
}

impl BuildAccessor<Range<usize>> for Build<'_> {
    type VectorsRef<'a> = rowmajor::Ref<'a, f32>;
    type Nearest<'a>
        = Coarse<'a>
    where
        Self: 'a;
    type Region<'a>
        = Region
    where
        Self: 'a;
    type Update<'a>
        = Update<'a>
    where
        Self: 'a;

    fn nearest(&mut self) -> Coarse<'_> {
        Coarse {
            stored: &self.store.lists,
            changes: &self.changes,
        }
    }

    fn route<F>(
        &mut self,
        points: Range<usize>,
        mut emit: F,
    ) -> impl SendFuture<ANNResult<NonZeroUsize>>
    where
        F: FnMut(u32, NonZeroUsize) + Send,
    {
        async move {
            let count = NonZeroUsize::new(points.len())
                .ok_or_else(|| ANNError::message("empty IVF batch"))?;
            if points.start != self.end || points.end > self.store.points.len() {
                return Err(ANNError::message(
                    "IVF batches must cover consecutive new IDs",
                ));
            }
            let mut routed = BTreeMap::<u32, Vec<u32>>::new();
            for id in points.clone() {
                let vector = &self.store.points[id];
                let list = self
                    .store
                    .lists
                    .iter()
                    .map(|(&id, list)| {
                        let distance: f32 =
                            SquaredL2::evaluate(vector.as_slice(), list.centroid.as_slice());
                        (id, distance)
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
                    .ok_or_else(|| ANNError::message("IVF provider has no live centroid"))?
                    .0;
                routed.entry(list).or_default().push(id as u32);
            }
            for (id, members) in routed {
                let list = self
                    .changes
                    .entry(id)
                    .or_insert_with(|| self.store.lists.get(&id).cloned())
                    .as_mut()
                    .ok_or_else(|| ANNError::message("invalid routing destination"))?;
                list.members.extend(members);
                emit(
                    id,
                    NonZeroUsize::new(list.members.len())
                        .ok_or_else(|| ANNError::message("empty routed list"))?,
                );
            }
            self.end = points.end;
            Ok(count)
        }
    }

    fn fill<Itr>(&mut self, lists: Itr) -> impl SendFuture<ANNResult<(Region, Update<'_>)>>
    where
        Itr: Iterator<Item = u32> + Clone + Send + Sync,
    {
        async move {
            let dim = self.store.points[0].len();
            let mut sources = BTreeMap::new();
            for id in lists {
                let list = self
                    .changes
                    .get(&id)
                    .map_or_else(|| self.store.lists.get(&id), Option::as_ref)
                    .ok_or_else(|| ANNError::message("cannot fill a missing IVF list"))?
                    .clone();
                let mut vectors = rowmajor::Owned::from_element(list.members.len(), dim, 0.0);
                for (&member, row) in list.members.iter().zip(vectors.rows_mut()) {
                    if member as usize >= self.end {
                        return Err(ANNError::message("unpublished IVF point"));
                    }
                    row.copy_from_slice(&self.store.points[member as usize]);
                }
                sources.insert(id, Source { list, vectors });
            }
            let filled = sources.keys().copied().collect();
            Ok((
                Region { sources },
                Update {
                    stored: &self.store.lists,
                    changes: &mut self.changes,
                    filled,
                    next_list: &mut self.next_list,
                    dim,
                },
            ))
        }
    }

    fn finish(mut self) -> impl SendFuture<ANNResult<()>> {
        async move {
            for (id, change) in self.changes {
                match change {
                    Some(list) => {
                        self.store.lists.insert(id, list);
                    }
                    None => {
                        self.store.lists.remove(&id);
                    }
                }
            }
            self.store.inserted = self.end;
            self.store.next_list = self.next_list;
            Ok(())
        }
    }
}
