/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    future::Future,
    sync::Arc,
};

use diskann_utils::{
    future::SendFuture,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use rand::{SeedableRng, rngs::StdRng};

use super::{
    Config, IVFIndex, InsertStats, clustering,
    pending::{Membership, PendingLists},
    region::{Batch, RegionUpdate, RegionView, View},
    traits::{BuildAccessor, BuildStrategy, HasIds, InsertStrategy, Provider, Router},
};
use crate::{ANNError, ANNResult, ivf::traits::RouterAndNearest, provider::DefaultContext};

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
    fail_fill: bool,
    fail_create: bool,
    fail_retire: bool,
    fail_finish: bool,
}

impl Provider for Store {
    type Context = DefaultContext;
    type InternalId = u32;
    type ExternalId = u32;
    type ListId = u32;
    type Error = ANNError;

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

struct Build<'p> {
    store: &'p mut Store,
    pending: PendingLists<u32, u32>,
    batch: Option<Arc<Batch<u32>>>,
    sources: BTreeMap<u32, Source>,
    fills: Cell<usize>,
}

struct Coarse<'a> {
    stored: &'a BTreeMap<u32, StoredList>,
}

impl Router for Coarse<'_> {
    type ListId = u32;

    fn len(&self) -> usize {
        self.stored.len()
    }

    fn select(&self, query: &[f32], nprobe: usize) -> impl SendFuture<ANNResult<Vec<(u32, f32)>>> {
        async move {
            let mut selected: Vec<(u32, f32)> = self
                .stored
                .iter()
                .map(|(&id, list)| (id, SquaredL2::evaluate(query, list.centroid.as_slice())))
                .collect();
            selected.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            selected.truncate(nprobe);
            Ok(selected)
        }
    }
}

impl RouterAndNearest for Coarse<'_> {
    fn nearest(
        &self,
        id: Self::ListId,
        nprobe: usize,
    ) -> impl SendFuture<ANNResult<Vec<(u32, f32)>>> {
        async move {
            let query = &self.stored.get(&id).unwrap().centroid;
            self.select(&query, nprobe).await
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

impl HasIds for Build<'_> {
    type InternalId = u32;
    type ListId = u32;
}

impl<'p> BuildAccessor for Build<'p> {
    type Update = Update;
    type Router<'a>
        = Coarse<'a>
    where
        Self: 'a;
    type Region<'a>
        = Region<'a>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.store.dim
    }

    fn router(&mut self) -> Self::Router<'_> {
        Coarse {
            stored: &self.store.lists,
        }
    }

    fn append_to(&mut self, list: u32, members: &[u32]) -> impl SendFuture<ANNResult<usize>> {
        async move {
            let size = self
                .store
                .lists
                .get(&list)
                .ok_or_else(|| ANNError::message(format!("unknown list {list}")))?
                .members
                .len();
            self.pending.append(list, members.to_vec())?;
            self.pending.list_size(list, size)
        }
    }

    fn fill(
        &mut self,
        lists: &[u32],
    ) -> impl SendFuture<ANNResult<(Self::Region<'_>, Self::Update)>> {
        async move {
            self.fills.set(self.fills.get() + 1);
            if self.store.fail_fill {
                return Err(ANNError::message("injected fill failure"));
            }
            self.sources.clear();
            for &list in lists {
                let stored = self
                    .store
                    .lists
                    .get(&list)
                    .ok_or_else(|| ANNError::message(format!("unknown list {list}")))?;
                let members: Vec<_> = self.pending.members(list, &stored.members)?.collect();
                let mut vectors = rowmajor::Owned::from_element(members.len(), self.dim(), 0.0);
                for (&id, row) in members.iter().zip(vectors.rows_mut()) {
                    let vector = self
                        .batch
                        .as_ref()
                        .and_then(|batch| batch.get(id))
                        .or_else(|| self.store.points.get(&id).map(Vec::as_slice))
                        .ok_or_else(|| ANNError::message(format!("missing point {id}")))?;
                    row.copy_from_slice(vector);
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
            let mut update = Update::new(&self.sources, self.dim());
            update.fail_create = self.store.fail_create;
            update.fail_retire = self.store.fail_retire;
            Ok((Region::new(&self.sources), update))
        }
    }

    fn finish(self, update: Option<&Self::Update>) -> impl SendFuture<ANNResult<()>> {
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
            if let Some(batch) = self.batch {
                for point in batch.points() {
                    self.store.points.insert(point.id, point.vector.to_vec());
                }
            }
            let regional_memberships = if let Some(update) = update {
                for (id, list) in (self.store.next_list..next_list).zip(update.created.clone()) {
                    self.store.lists.insert(id, list);
                }
                Some(update.memberships.clone())
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
            Ok(())
        }
    }
}

struct Strategy;

impl<'a> BuildStrategy<'a, Store> for Strategy {
    type Accessor = Build<'a>;
    type Error = ANNError;

    fn build_accessor(
        &'a self,
        store: &'a mut Store,
        _: &'a DefaultContext,
    ) -> ANNResult<Build<'a>> {
        Ok(Build {
            store,
            pending: PendingLists::default(),
            batch: None,
            sources: BTreeMap::new(),
            fills: Cell::new(0),
        })
    }
}

impl<'a, 'p> InsertStrategy<'a, Store, &'p [f32]> for Strategy {
    fn prepare(
        &self,
        accessor: &mut Self::Accessor,
        points: &[(u32, &'p [f32])],
    ) -> impl SendFuture<ANNResult<Arc<Batch<u32>>>> {
        async move {
            let mut vectors = rowmajor::Owned::from_element(points.len(), accessor.dim(), 0.0);
            let mut ids = Vec::with_capacity(points.len());
            for ((external, vector), out) in points.iter().zip(vectors.rows_mut()) {
                if vector.len() != out.len() || !vector.iter().all(|value| value.is_finite()) {
                    return Err(ANNError::message("invalid input vector"));
                }
                let id = external
                    .checked_add(10_000)
                    .ok_or_else(|| ANNError::message("internal point ID overflow"))?;
                if accessor.store.points.contains_key(&id) || ids.contains(&id) {
                    return Err(ANNError::message("point already exists"));
                }
                out.copy_from_slice(vector);
                ids.push(id);
            }
            let batch = Arc::new(Batch::new(ids, vectors)?);
            accessor.batch = Some(Arc::clone(&batch));
            Ok(batch)
        }
    }
}

fn matrix<const D: usize>(rows: &[[f32; D]]) -> rowmajor::Owned<f32> {
    let data: Box<[f32]> = rows.iter().flatten().copied().collect();
    rowmajor::Owned::try_from_data(data, rows.len(), D).unwrap()
}

fn index(store: Store, threshold: usize) -> IVFIndex<Store> {
    IVFIndex::new(
        store,
        Config {
            split_threshold: threshold,
            two_means_iterations: 5,
            seed: 7,
        },
    )
    .unwrap()
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
async fn appends_publish_actual_memberships_without_filling() {
    let mut index = index(
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
            ..Store::default()
        },
        2,
    );
    let points = [
        (7, &[0.0, 1.0][..]),
        (8, &[1.0, 0.0][..]),
        (9, &[10.0, 1.0][..]),
    ];
    let stats = require_send(index.insert_batch(&Strategy, &DefaultContext, &points))
        .await
        .unwrap();

    assert_eq!(
        stats,
        InsertStats {
            inserted: 3,
            splits: 0
        }
    );
    assert_eq!(index.provider().lists[&1].members, [10_007, 10_008]);
    assert_eq!(index.provider().lists[&2].members, [10_009]);
    assert_eq!(index.provider().points[&10_007], [0.0, 1.0]);
    assert_eq!(
        index
            .provider()
            .to_internal_id(&DefaultContext, &7)
            .unwrap(),
        10_007
    );
    assert_eq!(
        index
            .provider()
            .to_external_id(&DefaultContext, 10_007)
            .unwrap(),
        7
    );
    assert_eq!(index.provider().finishes, 1);
    assert_eq!(index.provider().fills, 0);

    let empty: [(u32, &[f32]); 0] = [];
    assert_eq!(
        index
            .insert_batch(&Strategy, &DefaultContext, &empty)
            .await
            .unwrap(),
        InsertStats::default()
    );
    assert_eq!(index.provider().finishes, 1);
}

#[tokio::test]
async fn splitting_commits_children_and_ordinary_appends_once() {
    let mut index = index(seeded(), 2);
    let points = [
        (5, &[1.0, 1.0][..]),
        (6, &[11.0, 1.0][..]),
        (7, &[101.0, 0.0][..]),
    ];
    let stats = require_send(index.insert_batch(&Strategy, &DefaultContext, &points))
        .await
        .unwrap();
    let store = index.provider();

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
async fn region_view_borrows_inputs_while_updates_are_recorded() {
    let mut store = seeded();
    let update = {
        let mut build = Strategy
            .build_accessor(&mut store, &DefaultContext)
            .unwrap();
        let points = [(5, &[1.0, 1.0][..]), (6, &[3.0, 1.0][..])];
        let batch = Strategy.prepare(&mut build, &points).await.unwrap();
        assert_eq!(build.append_to(1, &[10_006, 10_005]).await.unwrap(), 4);
        let update = {
            let (region, mut update) = require_send(build.fill(&[1])).await.unwrap();
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
            assert_eq!(vectors.row(2), batch.get(10_006).unwrap());
            assert_eq!(vectors.row(3), batch.get(10_005).unwrap());
            clustering::split(
                &region,
                &mut update,
                &[1],
                2,
                5,
                &mut StdRng::seed_from_u64(7),
            )
            .unwrap();
            assert_eq!(view.members, [10_000, 10_001, 10_006, 10_005]);
            assert_eq!(vectors.row(2), batch.get(10_006).unwrap());
            assert!(update.set_members(1, &[]).is_err());
            update
        };
        assert_eq!(build.dim(), 2);
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
    assert_eq!(store.lists[&1].members, [10_000, 10_001]);
    assert!(!store.points.contains_key(&10_005));
    assert_eq!(store.finishes, 0);
}

#[tokio::test]
async fn failed_planning_or_publication_is_not_reported_as_success() {
    for (fail_fill, fail_create, fail_retire, fail_finish) in [
        (true, false, false, false),
        (false, true, false, false),
        (false, false, true, false),
        (false, false, false, true),
    ] {
        let mut store = seeded();
        store.fail_fill = fail_fill;
        store.fail_create = fail_create;
        store.fail_retire = fail_retire;
        store.fail_finish = fail_finish;
        let mut index = index(store, 2);
        let points = [(5, &[1.0, 1.0][..])];
        let error = require_send(index.insert_batch(&Strategy, &DefaultContext, &points))
            .await
            .unwrap_err();
        let expected = if fail_fill {
            "injected fill failure"
        } else if fail_create {
            "injected create failure"
        } else if fail_retire {
            "injected retire failure"
        } else {
            "injected finish failure"
        };
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(index.provider().finishes, 0);
        assert_eq!(index.provider().lists[&1].members, [10_000, 10_001]);
        assert_eq!(index.provider().lists.len(), 3);
        assert!(
            index
                .provider()
                .to_internal_id(&DefaultContext, &5)
                .is_err()
        );
    }
}

#[tokio::test]
async fn invalid_config_and_uninitialized_insertion_fail() {
    assert!(
        IVFIndex::new(
            Store::default(),
            Config {
                split_threshold: 1,
                two_means_iterations: 5,
                seed: 7,
            },
        )
        .is_err()
    );
    let mut index = index(
        Store {
            dim: 2,
            next_list: 1,
            ..Store::default()
        },
        2,
    );
    let points = [(7, &[0.0, 1.0][..])];
    assert!(
        index
            .insert_batch(&Strategy, &DefaultContext, &points)
            .await
            .is_err()
    );
    assert_eq!(index.provider().finishes, 0);
}

#[tokio::test]
async fn split_only_edits_explicit_parents_and_missing_sources_leave_no_edits() {
    let mut store = seeded();
    store.points.insert(10_001, vec![10.0, 0.0]);
    store.points.insert(10_002, vec![5.0, 0.0]);
    store.points.insert(10_003, vec![10.1, 0.0]);
    store.lists.get_mut(&2).unwrap().centroid = vec![5.0, 0.0];
    let mut build = Strategy
        .build_accessor(&mut store, &DefaultContext)
        .unwrap();
    let points = [(5, &[5.0, 0.0][..])];
    Strategy.prepare(&mut build, &points).await.unwrap();
    assert_eq!(build.append_to(1, &[10_005]).await.unwrap(), 3);

    let update = {
        let (region, mut update) = build.fill(&[1, 2]).await.unwrap();
        assert!(
            clustering::split(
                &region,
                &mut update,
                &[1, 99],
                2,
                5,
                &mut StdRng::seed_from_u64(7)
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
        clustering::split(
            &region,
            &mut update,
            &[1],
            2,
            5,
            &mut StdRng::seed_from_u64(7),
        )
        .unwrap();
        assert_eq!(update.created.len(), 2);
        update
    };
    require_send(build.finish(Some(&update))).await.unwrap();

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
    clustering::split(
        &region,
        &mut update,
        &[1, 2],
        1,
        10,
        &mut StdRng::seed_from_u64(7),
    )
    .unwrap();
    assert!(
        update.created[2..]
            .iter()
            .any(|partition| partition.members.contains(&4))
    );
}

#[test]
fn two_means_keeps_empty_children_finite_and_prefers_first_ties() {
    let rows = [[f32::MAX], [f32::MAX], [f32::MAX]];
    let sources = [(
        1,
        Source {
            centroid: vec![f32::MAX],
            members: vec![0, 1, 2],
            vectors: matrix(&rows),
        },
    )]
    .into();
    let region = Region::new(&sources);
    let mut zero = Update::new(&sources, 1);
    clustering::split(
        &region,
        &mut zero,
        &[1],
        1,
        0,
        &mut StdRng::seed_from_u64(7),
    )
    .unwrap();
    let mut one = Update::new(&sources, 1);
    clustering::split(&region, &mut one, &[1], 1, 1, &mut StdRng::seed_from_u64(7)).unwrap();
    assert_eq!(zero.created.len(), 2);
    assert_eq!(one.created.len(), 2);
    for (zero, one) in zero.created.iter().zip(&one.created) {
        assert_eq!(zero.centroid, one.centroid);
        assert!(zero.centroid.iter().all(|value| value.is_finite()));
    }
    assert_eq!(zero.created[0].members, [0, 1, 2]);
    assert!(zero.created[1].members.is_empty());
}
