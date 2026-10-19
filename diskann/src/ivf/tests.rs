/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    future::Future,
    num::NonZeroUsize,
};

use diskann_utils::{
    future::SendFuture,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use tokio::sync::{Mutex, MutexGuard};

use super::{
    Config, IVFIndex, InsertStats, ReassignCount, clustering,
    pending::{Membership, PendingLists},
    region::{RegionUpdate, RegionView, View},
    traits::{BuildAccessor, BuildStrategy, HasIds, Router, RouterAndNearest, SetElements},
};
use crate::{ANNError, ANNResult, provider::DefaultContext};

#[derive(Debug, Clone)]
struct StoredList {
    centroid: Vec<f32>,
    members: Vec<u32>,
}

#[derive(Debug, Default)]
struct Store {
    dim: usize,
    points: BTreeMap<u32, Vec<f32>>,
    lists: BTreeMap<u32, StoredList>,
    next_list: u32,
    finishes: usize,
    fills: usize,
    filled_lists: Vec<u32>,
    fail_nearest: bool,
    fail_fill: bool,
    fail_create: bool,
    fail_set_members: bool,
    fail_retire: bool,
    fail_finish: bool,
}

impl Store {
    fn to_internal_id(&self, _: &DefaultContext, id: &u32) -> ANNResult<u32> {
        id.checked_add(10_000)
            .filter(|id| self.points.contains_key(id))
            .ok_or_else(|| ANNError::message("point is not visible"))
    }

    fn to_external_id(&self, _: &DefaultContext, id: u32) -> ANNResult<u32> {
        if !self.points.contains_key(&id) {
            return Err(ANNError::message("point is not visible"));
        }
        id.checked_sub(10_000)
            .ok_or_else(|| ANNError::message("invalid internal point ID"))
    }
}

struct Build<'p, T> {
    store: MutexGuard<'p, Store>,
    pending: PendingLists<u32, u32>,
    staged: BTreeMap<u32, Vec<T>>,
    sources: BTreeMap<u32, Source>,
    fills: Cell<usize>,
    filled_lists: Vec<u32>,
}

struct Coarse<'a> {
    stored: &'a BTreeMap<u32, StoredList>,
    fail_nearest: bool,
}

impl Coarse<'_> {
    fn select<T: Copy + Into<f32>>(&self, query: &[T], nprobe: usize) -> Vec<(u32, f32)> {
        let mut selected: Vec<_> = self
            .stored
            .iter()
            .map(|(&id, list)| {
                let distance: f32 = query
                    .iter()
                    .zip(&list.centroid)
                    .map(|(&value, &center)| {
                        let delta = value.into() - center;
                        delta * delta
                    })
                    .sum();
                (id, distance)
            })
            .collect();
        selected.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        selected.truncate(nprobe);
        selected
    }
}

impl<T: Copy + Into<f32> + Sync> Router<(u32, &[T])> for Coarse<'_> {
    type Id = u32;

    fn len(&self) -> usize {
        self.stored.len()
    }

    fn route(
        &self,
        query: &(u32, &[T]),
        nprobe: usize,
    ) -> impl SendFuture<ANNResult<Vec<(u32, f32)>>> {
        async move { Ok(self.select(query.1, nprobe)) }
    }
}

impl<T: Copy + Into<f32> + Sync> RouterAndNearest<(u32, &[T])> for Coarse<'_> {
    fn nearest(&self, id: Self::Id, nprobe: usize) -> impl SendFuture<ANNResult<Vec<(u32, f32)>>> {
        async move {
            if self.fail_nearest {
                return Err(ANNError::message("injected nearest failure"));
            }
            let query = &self
                .stored
                .get(&id)
                .ok_or_else(|| ANNError::message(format!("unknown list {id}")))?
                .centroid;
            Ok(self.select(query, nprobe))
        }
    }
}

struct Source {
    centroid: Vec<f32>,
    members: Vec<u32>,
    vectors: rowmajor::Owned<f32>,
}

#[derive(Debug)]
struct Update {
    dim: usize,
    lists: BTreeSet<u32>,
    memberships: PendingLists<u32, u32>,
    created: Vec<StoredList>,
    fail_create: bool,
    fail_set_members: bool,
    fail_retire: bool,
}

impl Update {
    fn new(sources: &BTreeMap<u32, Source>, dim: usize) -> Self {
        Self {
            dim,
            lists: sources.keys().copied().collect(),
            memberships: PendingLists::default(),
            created: Vec::new(),
            fail_create: false,
            fail_set_members: false,
            fail_retire: false,
        }
    }

    fn check_list(&self, list: u32) -> ANNResult<()> {
        if self.lists.contains(&list) {
            Ok(())
        } else {
            Err(ANNError::message(format!("unknown list {list}")))
        }
    }
}

struct Region<'a> {
    sources: &'a BTreeMap<u32, Source>,
}

impl<'a> Region<'a> {
    fn new(sources: &'a BTreeMap<u32, Source>) -> Self {
        Self { sources }
    }
}

impl RegionView<u32, u32> for Region<'_> {
    fn get_view(&self, list: u32) -> Option<View<'_, u32, u32>> {
        self.sources.get(&list).map(|source| View {
            id: list,
            centroid: &source.centroid,
            members: &source.members,
        })
    }

    fn get_vectors(&self, list: u32) -> Option<rowmajor::Ref<'_, f32>> {
        self.sources
            .get(&list)
            .map(|source| source.vectors.as_view())
    }
}

impl RegionUpdate<u32, u32> for Update {
    fn set_members(&mut self, list: u32, members: &[u32]) -> ANNResult<()> {
        if self.fail_set_members {
            return Err(ANNError::message("injected set_members failure"));
        }
        self.check_list(list)?;
        self.memberships.set(list, members.to_vec())
    }

    fn retire(&mut self, list: u32) -> ANNResult<()> {
        if self.fail_retire {
            return Err(ANNError::message("injected retire failure"));
        }
        self.check_list(list)?;
        self.memberships.retire(list)
    }

    fn create(&mut self, centroid: &[f32], members: &[u32]) -> ANNResult<()> {
        if self.fail_create {
            return Err(ANNError::message("injected create failure"));
        }
        if centroid.len() != self.dim || !centroid.iter().all(|value| value.is_finite()) {
            return Err(ANNError::message("invalid centroid"));
        }
        self.created.push(StoredList {
            centroid: centroid.to_vec(),
            members: members.to_vec(),
        });
        Ok(())
    }
}

impl<T> HasIds for Build<'_, T> {
    type Id = u32;
    type ListId = u32;
}

impl<'p, T: Copy + Into<f32> + Send + Sync> BuildAccessor<(u32, &[T])> for Build<'p, T> {
    type Update = Update;
    type Router<'a>
        = Coarse<'a>
    where
        Self: 'a;
    type Region<'a>
        = Region<'a>
    where
        Self: 'a;

    fn dim(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.store.dim).unwrap()
    }

    fn router(&mut self) -> Self::Router<'_> {
        Coarse {
            stored: &self.store.lists,
            fail_nearest: self.store.fail_nearest,
        }
    }

    fn append_to<Itr>(
        &mut self,
        list: u32,
        members: Itr,
    ) -> impl SendFuture<ANNResult<NonZeroUsize>>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        async move {
            let size = self
                .store
                .lists
                .get(&list)
                .ok_or_else(|| ANNError::message(format!("unknown list {list}")))?
                .members
                .len();
            self.pending.append(list, members.collect())?;
            NonZeroUsize::new(self.pending.list_size(list, size)?)
                .ok_or_else(|| ANNError::message("appending must leave a nonempty list"))
        }
    }

    fn fill<Itr>(
        &mut self,
        lists: Itr,
    ) -> impl SendFuture<ANNResult<(Self::Region<'_>, Self::Update)>>
    where
        Itr: Iterator<Item = Self::ListId> + Clone + Send + Sync,
    {
        async move {
            self.fills.set(self.fills.get() + 1);
            if self.store.fail_fill {
                return Err(ANNError::message("injected fill failure"));
            }
            self.filled_lists.extend(lists.clone());
            self.sources.clear();
            for list in lists {
                let stored = self
                    .store
                    .lists
                    .get(&list)
                    .ok_or_else(|| ANNError::message(format!("unknown list {list}")))?;
                let members: Vec<_> = self.pending.members(list, &stored.members)?.collect();
                let mut vectors =
                    rowmajor::Owned::from_element(members.len(), self.dim().get(), 0.0);
                for (&id, row) in members.iter().zip(vectors.rows_mut()) {
                    if let Some(vector) = self.staged.get(&id) {
                        for (dst, &src) in row.iter_mut().zip(vector) {
                            *dst = src.into();
                        }
                    } else {
                        let vector = self
                            .store
                            .points
                            .get(&id)
                            .ok_or_else(|| ANNError::message(format!("missing point {id}")))?;
                        row.copy_from_slice(vector);
                    }
                }
                self.sources.insert(
                    list,
                    Source {
                        centroid: stored.centroid.clone(),
                        members,
                        vectors,
                    },
                );
            }
            let mut update = Update::new(&self.sources, self.dim().get());
            update.fail_create = self.store.fail_create;
            update.fail_set_members = self.store.fail_set_members;
            update.fail_retire = self.store.fail_retire;
            Ok((Region::new(&self.sources), update))
        }
    }

    fn finish(mut self, update: Option<Self::Update>) -> impl SendFuture<ANNResult<()>> {
        async move {
            if self.store.fail_finish {
                return Err(ANNError::message("injected finish failure"));
            }
            let created = update.as_ref().map_or(0, |update| update.created.len());
            let created =
                u32::try_from(created).map_err(|_| ANNError::message("list ID overflow"))?;
            let next_list = self
                .store
                .next_list
                .checked_add(created)
                .ok_or_else(|| ANNError::message("list ID overflow"))?;
            for (id, vector) in self.staged {
                self.store
                    .points
                    .insert(id, vector.into_iter().map(Into::into).collect());
            }
            let regional_memberships = if let Some(update) = update {
                for (id, list) in (self.store.next_list..next_list).zip(update.created) {
                    self.store.lists.insert(id, list);
                }
                Some(update.memberships)
            } else {
                None
            };
            for (id, change) in self.pending.into_changes().chain(
                regional_memberships
                    .into_iter()
                    .flat_map(PendingLists::into_changes),
            ) {
                match change {
                    Membership::Append(members) => {
                        self.store
                            .lists
                            .get_mut(&id)
                            .unwrap()
                            .members
                            .extend(members);
                    }
                    Membership::Replace(members) => {
                        self.store.lists.get_mut(&id).unwrap().members = members;
                    }
                    Membership::Retire => {
                        self.store.lists.remove(&id).unwrap();
                    }
                }
            }
            self.store.next_list = next_list;
            self.store.finishes += 1;
            self.store.fills += self.fills.get();
            self.store.filled_lists = self.filled_lists;
            Ok(())
        }
    }
}

struct Strategy {
    store: Mutex<Store>,
}

impl Strategy {
    fn new(store: Store) -> Self {
        Self {
            store: Mutex::new(store),
        }
    }

    fn store(&self) -> MutexGuard<'_, Store> {
        self.store.try_lock().unwrap()
    }
}

impl HasIds for Strategy {
    type Id = u32;
    type ListId = u32;
}

impl<'a, T: Copy + Into<f32> + Send + Sync> BuildStrategy<'a, (u32, &[T]), DefaultContext>
    for Strategy
{
    type Accessor = Build<'a, T>;
    type Error = ANNError;

    fn build_accessor(&'a self, _: &'a DefaultContext) -> ANNResult<Self::Accessor> {
        Ok(Build {
            store: self.store.try_lock().map_err(ANNError::new)?,
            pending: PendingLists::default(),
            staged: BTreeMap::new(),
            sources: BTreeMap::new(),
            fills: Cell::new(0),
            filled_lists: Vec::new(),
        })
    }
}

impl<'p, T: Copy + Into<f32> + Send + Sync> SetElements<(u32, &'p [T]), DefaultContext>
    for Build<'_, T>
{
    type Id = u32;

    async fn set_elements(
        &mut self,
        _: &DefaultContext,
        points: &[(u32, &'p [T])],
    ) -> ANNResult<Vec<u32>> {
        let mut staged = BTreeMap::new();
        let mut ids = Vec::with_capacity(points.len());
        for (external, vector) in points {
            if vector.len() != self.store.dim
                || !vector.iter().all(|&value| value.into().is_finite())
            {
                return Err(ANNError::message("invalid input vector"));
            }
            let id = external
                .checked_add(10_000)
                .ok_or_else(|| ANNError::message("internal point ID overflow"))?;
            if self.store.points.contains_key(&id)
                || self.staged.contains_key(&id)
                || staged.contains_key(&id)
            {
                return Err(ANNError::message("point already exists"));
            }
            staged.insert(id, vector.to_vec());
            ids.push(id);
        }
        self.staged.extend(staged);
        Ok(ids)
    }
}

fn matrix<const D: usize>(rows: &[[f32; D]]) -> rowmajor::Owned<f32> {
    let data: Box<[f32]> = rows.iter().flatten().copied().collect();
    rowmajor::Owned::try_from_data(data, rows.len(), D).unwrap()
}

fn index(store: Store, threshold: usize) -> (IVFIndex<DefaultContext>, Strategy) {
    index_with_neighbors(store, threshold, 0)
}

fn index_with_neighbors(
    store: Store,
    threshold: usize,
    neighbors: usize,
) -> (IVFIndex<DefaultContext>, Strategy) {
    let index = IVFIndex::new(Config {
        split_threshold: NonZeroUsize::new(threshold).unwrap(),
        reassign_count: NonZeroUsize::new(neighbors)
            .map_or(ReassignCount::None, ReassignCount::Count),
        split_iterations: NonZeroUsize::new(5).unwrap(),
        seed: 7,
    })
    .unwrap();
    (index, Strategy::new(store))
}

fn seeded() -> Store {
    Store {
        dim: 2,
        points: [
            (10_000, vec![0.0, 0.0]),
            (10_001, vec![2.0, 0.0]),
            (10_002, vec![10.0, 0.0]),
            (10_003, vec![12.0, 0.0]),
            (10_004, vec![100.0, 0.0]),
        ]
        .into(),
        lists: [
            (
                1,
                StoredList {
                    centroid: vec![1.0, 0.0],
                    members: vec![10_000, 10_001],
                },
            ),
            (
                2,
                StoredList {
                    centroid: vec![11.0, 0.0],
                    members: vec![10_002, 10_003],
                },
            ),
            (
                3,
                StoredList {
                    centroid: vec![100.0, 0.0],
                    members: vec![10_004],
                },
            ),
        ]
        .into(),
        next_list: 4,
        ..Store::default()
    }
}

fn require_send<F: Future + Send>(future: F) -> F {
    future
}

#[tokio::test]
async fn staging_preserves_native_inputs_and_id_order() {
    let strategy = Strategy::new(seeded());
    let mut build: Build<'_, u8> = strategy.build_accessor(&DefaultContext).unwrap();
    let points: [(u32, &[u8]); 3] = [(9, &[11, 0]), (7, &[1, 0]), (8, &[0, 0])];
    let ids = require_send(build.set_elements(&DefaultContext, &points))
        .await
        .unwrap();
    assert_eq!(ids, [10_009, 10_007, 10_008]);
    for (&id, point) in ids.iter().zip(&points) {
        assert_eq!(build.staged[&id], point.1);
        assert!(!build.store.points.contains_key(&id));
    }
    let router = build.router();
    for (point, expected) in points.iter().zip([2, 1, 1]) {
        assert_eq!(
            require_send(router.route(point, 1)).await.unwrap()[0].0,
            expected
        );
    }
    drop(build);
    let store = strategy.store();
    assert_eq!(store.points.len(), 5);
    assert_eq!(store.finishes, 0);
}

#[tokio::test]
async fn native_inputs_are_inserted_using_the_returned_ids() {
    let (mut index, strategy) = index(seeded(), 4);
    let points: [(u32, &[u8]); 3] = [(9, &[11, 0]), (7, &[1, 0]), (8, &[0, 0])];
    let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points))
        .await
        .unwrap();
    assert_eq!(
        stats,
        InsertStats {
            inserted: 3,
            splits: 0
        }
    );
    let store = strategy.store();
    assert_eq!(store.lists[&1].members, [10_000, 10_001, 10_007, 10_008]);
    assert_eq!(store.lists[&2].members, [10_002, 10_003, 10_009]);
    for (external, vector) in points {
        let expected: Vec<_> = vector.iter().copied().map(f32::from).collect();
        assert_eq!(store.points[&(external + 10_000)], expected);
    }
    assert_eq!(store.finishes, 1);
    assert_eq!(store.fills, 0);
}

#[tokio::test]
async fn failed_staging_leaves_no_new_elements() {
    let strategy = Strategy::new(seeded());
    for points in [
        vec![(5, &[1.0, 0.0][..]), (5, &[2.0, 0.0][..])],
        vec![(5, &[1.0, 0.0][..]), (0, &[2.0, 0.0][..])],
        vec![(5, &[1.0, 0.0][..]), (6, &[2.0][..])],
        vec![(5, &[1.0, 0.0][..]), (6, &[f32::NAN, 0.0][..])],
        vec![(5, &[1.0, 0.0][..]), (u32::MAX, &[2.0, 0.0][..])],
    ] {
        let mut build: Build<'_, f32> = strategy.build_accessor(&DefaultContext).unwrap();
        assert!(
            require_send(build.set_elements(&DefaultContext, &points))
                .await
                .is_err()
        );
        assert!(build.staged.is_empty());
        assert_eq!(build.store.points.len(), 5);
        assert_eq!(build.store.finishes, 0);
    }
}

#[tokio::test]
async fn appends_publish_actual_memberships_without_filling() {
    let (mut index, strategy) = index(
        Store {
            dim: 2,
            lists: [
                (
                    1,
                    StoredList {
                        centroid: vec![0.0, 0.0],
                        members: Vec::new(),
                    },
                ),
                (
                    2,
                    StoredList {
                        centroid: vec![10.0, 0.0],
                        members: Vec::new(),
                    },
                ),
            ]
            .into(),
            next_list: 3,
            fail_nearest: true,
            ..Store::default()
        },
        2,
    );
    let points: [(u32, &[f32]); 3] = [
        (7, &[0.0, 1.0][..]),
        (8, &[1.0, 0.0][..]),
        (9, &[10.0, 1.0][..]),
    ];
    let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points))
        .await
        .unwrap();

    assert_eq!(
        stats,
        InsertStats {
            inserted: 3,
            splits: 0
        }
    );
    let store = strategy.store();
    assert_eq!(store.lists[&1].members, [10_007, 10_008]);
    assert_eq!(store.lists[&2].members, [10_009]);
    assert_eq!(store.points[&10_007], [0.0, 1.0]);
    assert_eq!(store.to_internal_id(&DefaultContext, &7).unwrap(), 10_007);
    assert_eq!(store.to_external_id(&DefaultContext, 10_007).unwrap(), 7);
    assert_eq!(store.finishes, 1);
    assert_eq!(store.fills, 0);

    let empty: [(u32, &[f32]); 0] = [];
    assert_eq!(
        index
            .insert_batch(&strategy, &DefaultContext, &empty)
            .await
            .unwrap(),
        InsertStats::default()
    );
    assert_eq!(store.finishes, 1);
}

#[tokio::test]
async fn splitting_commits_children_and_ordinary_appends_once() {
    let mut store = seeded();
    store.fail_nearest = true;
    let (mut index, strategy) = index(store, 2);
    let points: [(u32, &[f32]); 3] = [
        (5, &[1.0, 1.0][..]),
        (6, &[11.0, 1.0][..]),
        (7, &[101.0, 0.0][..]),
    ];
    let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points))
        .await
        .unwrap();
    let store = strategy.store();

    assert_eq!(
        stats,
        InsertStats {
            inserted: 3,
            splits: 2
        }
    );
    assert_eq!(store.finishes, 1);
    assert_eq!(store.fills, 1);
    assert!(!store.lists.contains_key(&1));
    assert!(!store.lists.contains_key(&2));
    assert_eq!(store.lists.len(), 5);
    assert_eq!(store.lists[&3].members, [10_004, 10_007]);
    let mut members: Vec<_> = store
        .lists
        .values()
        .flat_map(|list| list.members.iter().copied())
        .collect();
    members.sort_unstable();
    assert_eq!(members, (10_000..10_008).collect::<Vec<_>>());

    for (&id, list) in &store.lists {
        if id == 3 {
            continue;
        }
        for point in &list.members {
            let vector = &store.points[point];
            let own: f32 = SquaredL2::evaluate(vector.as_slice(), list.centroid.as_slice());
            for (&other, candidate) in &store.lists {
                if other != 3 {
                    let distance: f32 =
                        SquaredL2::evaluate(vector.as_slice(), candidate.centroid.as_slice());
                    assert!(own <= distance);
                }
            }
        }
    }
}

#[tokio::test]
async fn neighboring_lists_exchange_points_and_pending_appends_once() {
    let mut store = seeded();
    store.points.insert(10_001, vec![10.0, 0.0]);
    store.points.insert(10_002, vec![5.0, 0.0]);
    store.points.insert(10_003, vec![10.1, 0.0]);
    store.points.insert(10_005, vec![5.0, 0.0]);
    store.points.insert(10_006, vec![5.0, 0.0]);
    store
        .lists
        .get_mut(&1)
        .unwrap()
        .members
        .extend([10_005, 10_006]);
    store.lists.get_mut(&2).unwrap().centroid = vec![5.0, 0.0];
    let (mut index, strategy) = index_with_neighbors(store, 3, 2);
    let points: [(u32, &[f32]); 2] = [(7, &[1.0, 0.0][..]), (8, &[5.2, 0.0][..])];
    let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points))
        .await
        .unwrap();
    let store = strategy.store();

    assert_eq!(
        stats,
        InsertStats {
            inserted: 2,
            splits: 1
        }
    );
    assert_eq!(store.fills, 1);
    assert_eq!(store.finishes, 1);
    assert_eq!(store.filled_lists, [1, 2]);
    assert!(!store.lists.contains_key(&1));
    assert_eq!(store.lists.len(), 4);
    assert_eq!(store.lists[&2].centroid, [5.0, 0.0]);
    assert_eq!(store.lists[&2].members, [10_005, 10_006, 10_002, 10_008]);
    assert!(store.lists[&2].members.len() > 3);
    assert!(
        store
            .lists
            .iter()
            .any(|(&id, list)| { id >= 4 && list.members.contains(&10_003) })
    );
    assert_eq!(store.lists[&3].centroid, [100.0, 0.0]);
    assert_eq!(store.lists[&3].members, [10_004]);
    let mut members: Vec<_> = store
        .lists
        .values()
        .flat_map(|list| list.members.iter().copied())
        .collect();
    members.sort_unstable();
    assert_eq!(members, (10_000..10_009).collect::<Vec<_>>());
}

#[tokio::test]
async fn neighborhood_selection_filters_parents_and_fills_the_union_once() {
    for (centers, two_candidates, three_candidates) in [
        (vec![], vec![], vec![]),
        (vec![100.0, 200.0, 300.0], vec![], vec![3]),
        (vec![0.0, 12.0, 200.0], vec![3, 4], vec![3, 4]),
    ] {
        for count in [0, 1, 2, 3, 4, usize::MAX] {
            let mut store = seeded();
            if centers.is_empty() {
                store.lists.remove(&3);
                store.points.remove(&10_004);
            } else {
                store.lists.get_mut(&3).unwrap().centroid = vec![centers[0], 0.0];
                for (id, &center) in (4..).zip(&centers[1..]) {
                    store.lists.insert(
                        id,
                        StoredList {
                            centroid: vec![center, 0.0],
                            members: Vec::new(),
                        },
                    );
                }
            }
            store.next_list = 3 + centers.len() as u32;
            store.fail_nearest = count == 0;
            let (mut index, strategy) = index_with_neighbors(store, 2, count);
            let points: [(u32, &[f32]); 2] = [(5, &[1.0, 1.0][..]), (6, &[11.0, 1.0][..])];
            let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points))
                .await
                .unwrap();
            let store = strategy.store();
            let neighbors = match (count, centers.len()) {
                (0 | 1, _) | (_, 0) => vec![],
                (2, _) => two_candidates.clone(),
                (3, _) => three_candidates.clone(),
                (4, _) => vec![3, 4],
                _ => vec![3, 4, 5],
            };
            let sources: Vec<_> = [1, 2].into_iter().chain(neighbors).collect();
            assert_eq!(store.filled_lists, sources);
            assert_eq!(store.fills, 1);
            assert_eq!(store.finishes, 1);
            assert_eq!(
                stats,
                InsertStats {
                    inserted: 2,
                    splits: 2
                }
            );
            assert!(!store.lists.contains_key(&1));
            assert!(!store.lists.contains_key(&2));
            assert_eq!(store.lists.len(), 4 + centers.len());
            for (id, &center) in (3..).zip(&centers) {
                assert_eq!(store.lists[&id].centroid, [center, 0.0]);
            }
            let mut members: Vec<_> = store
                .lists
                .values()
                .flat_map(|list| list.members.iter().copied())
                .collect();
            members.sort_unstable();
            assert_eq!(members, store.points.keys().copied().collect::<Vec<_>>());
        }
    }
}

#[test]
fn regional_assignment_moves_between_neighbors_and_retains_empty_lists() {
    let sources = [
        (
            1,
            Source {
                centroid: vec![0.0],
                members: vec![0, 1],
                vectors: matrix(&[[-10.0], [10.0]]),
            },
        ),
        (
            2,
            Source {
                centroid: vec![5.0],
                members: vec![2],
                vectors: matrix(&[[50.0]]),
            },
        ),
        (
            3,
            Source {
                centroid: vec![50.0],
                members: vec![3],
                vectors: matrix(&[[51.0]]),
            },
        ),
        (
            4,
            Source {
                centroid: vec![10.0],
                members: Vec::new(),
                vectors: matrix::<1>(&[]),
            },
        ),
    ]
    .into();
    let region = Region::new(&sources);
    let mut update = Update::new(&sources, 1);
    clustering::SplitAndAssign::new(
        clustering::TwoMeans::new(NonZeroUsize::new(5).unwrap(), NonZeroUsize::new(1).unwrap()),
        clustering::RegionalNearest,
    )
    .run(
        &region,
        &mut update,
        &BTreeSet::from([1]),
        &BTreeSet::from([2, 3, 4]),
    )
    .unwrap();

    let mut centers: Vec<_> = update.created.iter().map(|list| list.centroid[0]).collect();
    centers.sort_by(f32::total_cmp);
    assert_eq!(centers, [-10.0, 10.0]);
    let mut child_members: Vec<_> = update
        .created
        .iter()
        .flat_map(|list| list.members.iter().copied())
        .collect();
    child_members.sort_unstable();
    assert_eq!(child_members, [0, 1]);
    let mut changes: BTreeMap<_, _> = update.memberships.into_changes().collect();
    assert!(matches!(changes.remove(&1), Some(Membership::Retire)));
    for (id, expected) in [(2, vec![]), (3, vec![2, 3]), (4, vec![])] {
        let Some(Membership::Replace(members)) = changes.remove(&id) else {
            panic!("expected replacement for retained list {id}");
        };
        assert_eq!(members, expected);
    }
    assert!(changes.is_empty());
}

#[tokio::test]
async fn region_view_borrows_inputs_while_updates_are_recorded() {
    let strategy = Strategy::new(seeded());
    let update = {
        let mut build: Build<'_, f32> = strategy.build_accessor(&DefaultContext).unwrap();
        let points = [(5, &[1.0, 1.0][..]), (6, &[3.0, 1.0][..])];
        let ids = build.set_elements(&DefaultContext, &points).await.unwrap();
        assert_eq!(ids, [10_005, 10_006]);
        assert_eq!(
            build
                .append_to(1, [10_006, 10_005].into_iter())
                .await
                .unwrap()
                .get(),
            4
        );
        let update = {
            let (region, mut update) = require_send(build.fill([1].into_iter())).await.unwrap();
            let view = region.get_view(1).unwrap();
            let vectors = region.get_vectors(1).unwrap();
            let source = &region.sources[&1];
            assert_eq!(view.members, [10_000, 10_001, 10_006, 10_005]);
            assert_eq!(view.centroid.as_ptr(), source.centroid.as_ptr());
            assert_eq!(view.members.as_ptr(), source.members.as_ptr());
            assert_eq!(
                vectors.as_slice().as_ptr(),
                source.vectors.as_slice().as_ptr()
            );
            assert_eq!(vectors.row(0), [0.0, 0.0]);
            assert_eq!(vectors.row(2), points[1].1);
            assert_eq!(vectors.row(3), points[0].1);
            clustering::SplitAndAssign::new(
                clustering::TwoMeans::new(
                    NonZeroUsize::new(5).unwrap(),
                    NonZeroUsize::new(2).unwrap(),
                ),
                clustering::RegionalNearest,
            )
            .run(&region, &mut update, &BTreeSet::from([1]), &BTreeSet::new())
            .unwrap();
            assert_eq!(view.members, [10_000, 10_001, 10_006, 10_005]);
            assert_eq!(vectors.row(2), points[1].1);
            assert!(update.set_members(1, &[]).is_err());
            update
        };
        assert_eq!(build.dim().get(), 2);
        update
    };
    assert_eq!(update.created.len(), 2);
    let mut members: Vec<_> = update
        .created
        .iter()
        .flat_map(|list| list.members.iter().copied())
        .collect();
    members.sort_unstable();
    assert_eq!(members, [10_000, 10_001, 10_005, 10_006]);
    assert!(
        update
            .memberships
            .into_changes()
            .any(|(id, change)| id == 1 && matches!(change, Membership::Retire))
    );
    let store = strategy.store();
    assert_eq!(store.lists[&1].members, [10_000, 10_001]);
    assert!(!store.points.contains_key(&10_005));
    assert_eq!(store.finishes, 0);
}

#[tokio::test]
async fn failed_planning_or_publication_is_not_reported_as_success() {
    for (fail_nearest, fail_fill, fail_create, fail_set_members, fail_retire, fail_finish) in [
        (true, false, false, false, false, false),
        (false, true, false, false, false, false),
        (false, false, true, false, false, false),
        (false, false, false, true, false, false),
        (false, false, false, false, true, false),
        (false, false, false, false, false, true),
    ] {
        let mut store = seeded();
        store.fail_nearest = fail_nearest;
        store.fail_fill = fail_fill;
        store.fail_create = fail_create;
        store.fail_set_members = fail_set_members;
        store.fail_retire = fail_retire;
        store.fail_finish = fail_finish;
        let (mut index, strategy) = index_with_neighbors(store, 2, 2);
        let points: [(u32, &[f32]); 1] = [(5, &[1.0, 1.0][..])];
        let error = require_send(index.insert_batch(&strategy, &DefaultContext, &points))
            .await
            .unwrap_err();
        let expected = if fail_nearest {
            "injected nearest failure"
        } else if fail_fill {
            "injected fill failure"
        } else if fail_create {
            "injected create failure"
        } else if fail_set_members {
            "injected set_members failure"
        } else if fail_retire {
            "injected retire failure"
        } else {
            "injected finish failure"
        };
        assert!(error.to_string().contains(expected), "{error}");
        let store = strategy.store();
        assert_eq!(store.finishes, 0);
        assert_eq!(store.lists[&1].members, [10_000, 10_001]);
        assert_eq!(store.lists[&2].members, [10_002, 10_003]);
        assert_eq!(store.lists.len(), 3);
        assert!(store.to_internal_id(&DefaultContext, &5).is_err());
    }
}

#[tokio::test]
async fn uninitialized_insertion_fails() {
    let (mut index, strategy) = index(
        Store {
            dim: 2,
            next_list: 1,
            ..Store::default()
        },
        2,
    );
    let points: [(u32, &[f32]); 1] = [(7, &[0.0, 1.0][..])];
    assert!(
        index
            .insert_batch(&strategy, &DefaultContext, &points)
            .await
            .is_err()
    );
    assert_eq!(strategy.store().finishes, 0);
}

#[tokio::test]
async fn split_only_edits_explicit_parents_and_missing_sources_leave_no_edits() {
    let mut store = seeded();
    store.points.insert(10_001, vec![10.0, 0.0]);
    store.points.insert(10_002, vec![5.0, 0.0]);
    store.points.insert(10_003, vec![10.1, 0.0]);
    store.lists.get_mut(&2).unwrap().centroid = vec![5.0, 0.0];
    let strategy = Strategy::new(store);
    let mut build: Build<'_, f32> = strategy.build_accessor(&DefaultContext).unwrap();
    let points = [(5, &[5.0, 0.0][..])];
    build.set_elements(&DefaultContext, &points).await.unwrap();
    assert_eq!(
        build
            .append_to(1, [10_005].into_iter())
            .await
            .unwrap()
            .get(),
        3
    );

    let update = {
        let (region, mut update) = build.fill([1, 2].into_iter()).await.unwrap();
        assert!(
            clustering::SplitAndAssign::new(
                clustering::TwoMeans::new(
                    NonZeroUsize::new(5).unwrap(),
                    NonZeroUsize::new(2).unwrap(),
                ),
                clustering::RegionalNearest,
            )
            .run(
                &region,
                &mut update,
                &BTreeSet::from([1, 99]),
                &BTreeSet::new()
            )
            .is_err()
        );
        assert!(
            clustering::SplitAndAssign::new(
                clustering::TwoMeans::new(
                    NonZeroUsize::new(5).unwrap(),
                    NonZeroUsize::new(2).unwrap(),
                ),
                clustering::RegionalNearest,
            )
            .run(
                &region,
                &mut update,
                &BTreeSet::from([1]),
                &BTreeSet::from([99])
            )
            .is_err()
        );
        assert!(update.created.is_empty());
        assert_eq!(
            std::mem::take(&mut update.memberships)
                .into_changes()
                .count(),
            0
        );
        clustering::SplitAndAssign::new(
            clustering::TwoMeans::new(NonZeroUsize::new(5).unwrap(), NonZeroUsize::new(2).unwrap()),
            clustering::RegionalNearest,
        )
        .run(&region, &mut update, &BTreeSet::from([1]), &BTreeSet::new())
        .unwrap();
        assert_eq!(update.created.len(), 2);
        update
    };
    require_send(build.finish(Some(update))).await.unwrap();

    let store = strategy.store();
    assert_eq!(store.finishes, 1);
    assert!(!store.lists.contains_key(&1));
    assert_eq!(store.lists[&2].centroid, [5.0, 0.0]);
    assert_eq!(store.lists[&2].members, [10_002, 10_003]);
    assert_eq!(store.lists[&3].members, [10_004]);
    assert_eq!(store.lists.len(), 4);
    let mut members: Vec<_> = store
        .lists
        .values()
        .flat_map(|list| list.members.iter().copied())
        .collect();
    members.sort_unstable();
    assert_eq!(members, (10_000..10_006).collect::<Vec<_>>());
}

#[test]
fn assignment_can_cross_parent_boundaries() {
    let left = [[-100.0], [-99.0], [-10.0], [-9.0], [-0.1]];
    let right = [[0.1], [0.2], [100.0], [101.0]];
    let sources = [
        (
            1,
            Source {
                centroid: vec![-50.0],
                members: (0..left.len() as u32).collect(),
                vectors: matrix(&left),
            },
        ),
        (
            2,
            Source {
                centroid: vec![50.0],
                members: (100..100 + right.len() as u32).collect(),
                vectors: matrix(&right),
            },
        ),
    ]
    .into();
    let region = Region::new(&sources);
    let mut update = Update::new(&sources, 1);
    clustering::SplitAndAssign::new(
        clustering::TwoMeans::new(
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(1).unwrap(),
        ),
        clustering::RegionalNearest,
    )
    .run(
        &region,
        &mut update,
        &BTreeSet::from([1, 2]),
        &BTreeSet::new(),
    )
    .unwrap();
    assert!(
        update.created[2..]
            .iter()
            .any(|partition| partition.members.contains(&4))
    );
}

#[test]
fn two_means_resets_empty_children_to_zero_and_prefers_first_ties() {
    for value in [0.0, 7.0] {
        let sources = [(
            1,
            Source {
                centroid: vec![value],
                members: vec![0, 1, 2],
                vectors: matrix(&[[value]; 3]),
            },
        )]
        .into();
        let region = Region::new(&sources);
        let mut update = Update::new(&sources, 1);
        clustering::SplitAndAssign::new(
            clustering::TwoMeans::new(NonZeroUsize::new(1).unwrap(), NonZeroUsize::new(1).unwrap()),
            clustering::RegionalNearest,
        )
        .run(&region, &mut update, &BTreeSet::from([1]), &BTreeSet::new())
        .unwrap();
        assert_eq!(update.created.len(), 2);
        assert_eq!(update.created[0].centroid, [value]);
        assert_eq!(update.created[1].centroid, [0.0]);
        assert_eq!(update.created[0].members, [0, 1, 2]);
        assert!(update.created[1].members.is_empty());
    }
}

#[test]
fn split_strategies_support_in_place_fitting_and_source_local_candidates() {
    struct RowsAsCenters;

    impl clustering::Clusterer for RowsAsCenters {
        fn fit(
            &mut self,
            points: rowmajor::Ref<'_, f32>,
            mut dst: rowmajor::Mut<'_, f32>,
        ) -> ANNResult<()> {
            assert_eq!(points.nrows(), dst.nrows());
            dst.as_mut_slice().copy_from_slice(points.as_slice());
            Ok(())
        }

        fn num_centers(&self) -> NonZeroUsize {
            NonZeroUsize::new(2).unwrap()
        }

        fn dim(&self) -> NonZeroUsize {
            NonZeroUsize::new(1).unwrap()
        }
    }

    struct ParentLocal {
        owners: BTreeMap<u32, Vec<u32>>,
    }

    impl clustering::Assigner<u32, u32> for ParentLocal {
        fn assign<R, U>(
            &mut self,
            region: &R,
            update: &mut U,
            sources: &[u32],
            destinations: &[clustering::Destination<'_, u32>],
        ) -> ANNResult<()>
        where
            R: RegionView<u32, u32>,
            U: RegionUpdate<u32, u32>,
        {
            assert_eq!(sources, [1, 2, 3, 4]);
            assert_eq!(destinations.len(), 6);
            assert!(std::ptr::eq(
                destinations[4].centroid,
                region.get_view(3).unwrap().centroid
            ));
            let children = [(1, 0..2), (2, 2..4)];
            let mut memberships = vec![Vec::new(); destinations.len()];
            for &source in sources {
                let view = region.get_view(source).unwrap();
                let vectors = region.get_vectors(source).unwrap();
                let mut candidates: Vec<_> = destinations
                    .iter()
                    .enumerate()
                    .filter_map(|(index, destination)| {
                        matches!(&destination.list, clustering::Type::Exists(id) if *id == source)
                            .then_some(index)
                    })
                    .collect();
                for (parent, range) in &children {
                    if *parent == source
                        || self
                            .owners
                            .get(&source)
                            .is_some_and(|owners| owners.contains(parent))
                    {
                        candidates.extend(range.clone());
                    }
                }
                for (&id, point) in view.members.iter().zip(vectors.rows()) {
                    let destination = candidates
                        .iter()
                        .copied()
                        .min_by(|&a, &b| {
                            let left: f32 = SquaredL2::evaluate(point, destinations[a].centroid);
                            let right: f32 = SquaredL2::evaluate(point, destinations[b].centroid);
                            left.total_cmp(&right)
                        })
                        .unwrap();
                    memberships[destination].push(id);
                }
            }
            for (destination, members) in destinations.iter().zip(memberships) {
                match &destination.list {
                    clustering::Type::New => update.create(destination.centroid, &members)?,
                    clustering::Type::Exists(id) => update.set_members(*id, &members)?,
                }
            }
            for (parent, _) in children {
                update.retire(parent)?;
            }
            Ok(())
        }
    }

    let sources = [
        (
            1,
            Source {
                centroid: vec![0.0],
                members: vec![0, 1],
                vectors: matrix(&[[-10.0], [10.0]]),
            },
        ),
        (
            2,
            Source {
                centroid: vec![105.0],
                members: vec![2, 3],
                vectors: matrix(&[[100.0], [110.0]]),
            },
        ),
        (
            3,
            Source {
                centroid: vec![105.0],
                members: vec![4, 5, 6],
                vectors: matrix(&[[9.9], [100.1], [105.0]]),
            },
        ),
        (
            4,
            Source {
                centroid: vec![1000.0],
                members: vec![7],
                vectors: matrix(&[[100.2]]),
            },
        ),
    ]
    .into();
    let region = Region::new(&sources);
    let parents = BTreeSet::from([1, 2]);
    let neighbors = BTreeSet::from([3, 4]);
    let mut full = Update::new(&sources, 1);
    let mut splitter = clustering::SplitAndAssign::new(RowsAsCenters, clustering::RegionalNearest);
    splitter
        .run(&region, &mut full, &parents, &neighbors)
        .unwrap();
    let mut restricted = Update::new(&sources, 1);
    clustering::SplitAndAssign::new(
        RowsAsCenters,
        ParentLocal {
            owners: [(3, vec![1]), (4, vec![1, 2])].into(),
        },
    )
    .run(&region, &mut restricted, &parents, &neighbors)
    .unwrap();

    for update in [&full, &restricted] {
        assert_eq!(
            update
                .created
                .iter()
                .map(|list| list.centroid[0])
                .collect::<Vec<_>>(),
            [-10.0, 10.0, 100.0, 110.0]
        );
        assert_eq!(update.created[0].members, [0]);
        assert_eq!(update.created[1].members, [1, 4]);
        assert_eq!(update.created[3].members, [3]);
    }
    assert_eq!(full.created[2].members, [2, 5, 7]);
    assert_eq!(restricted.created[2].members, [2, 7]);
    let mut repeated = Update::new(&sources, 1);
    splitter
        .run(&region, &mut repeated, &parents, &neighbors)
        .unwrap();
    for (first, repeated) in full.created.iter().zip(&repeated.created) {
        assert_eq!(first.centroid, repeated.centroid);
        assert_eq!(first.members, repeated.members);
    }
    let mut smaller = Update::new(&sources, 1);
    splitter
        .run(
            &region,
            &mut smaller,
            &BTreeSet::from([1]),
            &BTreeSet::new(),
        )
        .unwrap();
    assert_eq!(smaller.created.len(), 2);
    assert_eq!(smaller.created[0].members, [0]);
    assert_eq!(smaller.created[1].members, [1]);
    for (update, expected) in [(full, vec![6]), (restricted, vec![5, 6])] {
        let mut changes: BTreeMap<_, _> = update.memberships.into_changes().collect();
        for parent in [1, 2] {
            assert!(matches!(changes.remove(&parent), Some(Membership::Retire)));
        }
        for (id, expected) in [(3, expected), (4, vec![])] {
            let Some(Membership::Replace(members)) = changes.remove(&id) else {
                panic!("expected replacement for retained list {id}");
            };
            assert_eq!(members, expected);
        }
        assert!(changes.is_empty());
    }
}

#[test]
fn numerical_strategy_failures_record_no_updates() {
    enum InvalidFit {
        Error,
        CountOverflow,
    }

    impl clustering::Clusterer for InvalidFit {
        fn fit(&mut self, _: rowmajor::Ref<'_, f32>, _: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
            match self {
                Self::Error => Err(ANNError::message("injected fitting failure")),
                Self::CountOverflow => {
                    unreachable!("centroid count overflow should be rejected before fitting")
                }
            }
        }

        fn num_centers(&self) -> NonZeroUsize {
            NonZeroUsize::new(match self {
                Self::CountOverflow => usize::MAX,
                Self::Error => 2,
            })
            .unwrap()
        }

        fn dim(&self) -> NonZeroUsize {
            NonZeroUsize::new(1).unwrap()
        }
    }

    struct InvalidAssignment;

    impl clustering::Assigner<u32, u32> for InvalidAssignment {
        fn assign<R, U>(
            &mut self,
            _: &R,
            _: &mut U,
            _: &[u32],
            _: &[clustering::Destination<'_, u32>],
        ) -> ANNResult<()>
        where
            R: RegionView<u32, u32>,
            U: RegionUpdate<u32, u32>,
        {
            Err(ANNError::message("injected assignment failure"))
        }
    }

    fn assert_unwritten(update: Update) {
        assert!(update.created.is_empty());
        assert_eq!(update.memberships.into_changes().count(), 0);
    }

    let sources = [(
        1,
        Source {
            centroid: vec![0.0],
            members: vec![0, 1],
            vectors: matrix(&[[-1.0], [1.0]]),
        },
    )]
    .into();
    let region = Region::new(&sources);
    let parents = BTreeSet::from([1]);
    let neighbors = BTreeSet::new();
    for (clusterer, expected) in [
        (InvalidFit::Error, "injected fitting failure"),
        (InvalidFit::CountOverflow, "centroid count overflow"),
    ] {
        let mut update = Update::new(&sources, 1);
        let selected = if matches!(clusterer, InvalidFit::CountOverflow) {
            BTreeSet::from([1, 2])
        } else {
            parents.clone()
        };
        let error = clustering::SplitAndAssign::new(clusterer, clustering::RegionalNearest)
            .run(&region, &mut update, &selected, &neighbors)
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        assert_unwritten(update);
    }
    let mut update = Update::new(&sources, 1);
    let error = clustering::SplitAndAssign::new(
        clustering::TwoMeans::new(NonZeroUsize::new(5).unwrap(), NonZeroUsize::new(1).unwrap()),
        InvalidAssignment,
    )
    .run(&region, &mut update, &parents, &neighbors)
    .unwrap_err();
    assert!(
        error.to_string().contains("injected assignment failure"),
        "{error}"
    );
    assert_unwritten(update);
}

#[test]
fn two_means_matches_quantization_lloyds() {
    use clustering::{Clusterer, TwoMeans};

    let points = matrix(&[[0.0, 2.0], [2.0, 4.0], [6.0, 6.0], [10.0, 12.0]]);
    for iterations in [1, 5] {
        let mut expected = points.as_view().subview(0..2).unwrap().to_rowmajor_owned();
        let _ = diskann_quantization::algorithms::kmeans::lloyds::lloyds(
            points.as_view(),
            expected.as_view_mut(),
            iterations,
        );

        let mut actual = rowmajor::Owned::from_element(2, 2, 0.0);
        TwoMeans::new(
            NonZeroUsize::new(iterations).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        )
        .fit(points.as_view(), actual.as_view_mut())
        .unwrap();
        assert_eq!(actual.as_slice(), expected.as_slice());
    }
}

#[test]
fn two_means_fits_only_the_destination_window() {
    use clustering::{Clusterer, TwoMeans};

    let points = matrix(&[[0.0, 2.0], [2.0, 4.0], [10.0, 12.0], [12.0, 14.0]]);
    let mut scratch = rowmajor::Owned::from_element(4, 2, -1.0);
    let mut clusterer = TwoMeans::new(
        NonZeroUsize::new(10).unwrap(),
        NonZeroUsize::new(2).unwrap(),
    );
    clusterer
        .fit(points.as_view(), scratch.subview_mut(1..3).unwrap())
        .unwrap();
    assert_eq!(scratch.row(0), [-1.0, -1.0]);
    assert_eq!(scratch.row(3), [-1.0, -1.0]);
    let window = scratch.subview(1..3).unwrap();
    let mut centers: Vec<_> = window.rows().collect();
    centers.sort_by(|a, b| a[0].total_cmp(&b[0]));
    assert_eq!(centers, [&[1.0, 3.0][..], &[11.0, 13.0][..]]);

    for (points, rows, dim) in [
        (matrix(&[[0.0, 2.0]]), 2, 2),
        (matrix(&[[0.0], [1.0]]), 2, 2),
        (matrix(&[[0.0, 2.0], [1.0, 3.0]]), 1, 2),
        (matrix(&[[0.0, 2.0], [1.0, 3.0]]), 2, 1),
    ] {
        let mut dst = rowmajor::Owned::from_element(rows, dim, -1.0);
        assert!(clusterer.fit(points.as_view(), dst.as_view_mut()).is_err());
        assert!(dst.as_slice().iter().all(|&value| value == -1.0));
    }
}

#[test]
fn regional_assignment_reports_missing_sources_before_writing() {
    use clustering::{Assigner, Destination, RegionalNearest, Type};

    let sources = BTreeMap::from([
        (
            1,
            Source {
                centroid: vec![0.0],
                members: vec![0, 1],
                vectors: matrix(&[[0.0], [1.0]]),
            },
        ),
        (
            2,
            Source {
                centroid: vec![2.0],
                members: vec![2],
                vectors: matrix(&[[2.0]]),
            },
        ),
    ]);
    let destinations = [Destination {
        list: Type::New,
        centroid: &[0.0],
    }];
    for ids in [vec![99], vec![1, 99], vec![1, 2, 99]] {
        let mut update = Update::new(&sources, 1);
        let region = Region::new(&sources);
        let error = RegionalNearest
            .assign(&region, &mut update, &ids, &destinations)
            .unwrap_err();
        assert!(error.to_string().contains("source is missing"), "{error}");
        assert!(update.created.is_empty());
        assert_eq!(update.memberships.into_changes().count(), 0);
    }
}
