/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{cell::Cell, collections::BTreeMap, future::Future, sync::Arc};

use diskann_utils::{
    future::SendFuture,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use rand::{SeedableRng, rngs::StdRng};

use super::{
    Config, IVFIndex, InsertStats, clustering,
    pending::{Membership, PendingLists},
    traits::{
        BuildAccessor, BuildStrategy, CoarseAccessor, InsertStrategy, ListAccessor, Provider,
        SelectedList,
    },
    workingset::{Batch, ListView, PointRef, WorkingSet},
};
use crate::{ANNError, ANNResult, provider::DefaultContext};

#[derive(Debug)]
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
    created: BTreeMap<u32, Vec<f32>>,
    batch: Option<Arc<Batch<u32>>>,
    next_list: u32,
    fills: Cell<usize>,
}

impl Build<'_> {
    fn stored_members(&self, list: u32) -> ANNResult<&[u32]> {
        if let Some(stored) = self.store.lists.get(&list) {
            Ok(&stored.members)
        } else if self.created.contains_key(&list) {
            Ok(&[])
        } else {
            Err(ANNError::message(format!("unknown list {list}")))
        }
    }
}

struct Coarse<'a> {
    stored: &'a BTreeMap<u32, StoredList>,
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
            let mut selected: Vec<_> = self
                .stored
                .iter()
                .map(|(&id, list)| (id, list.centroid.as_slice()))
                .chain(
                    self.created
                        .iter()
                        .map(|(&id, vector)| (id, vector.as_slice())),
                )
                .filter(|(id, _)| !self.pending.is_retired(*id))
                .map(|(id, vector)| SelectedList {
                    id,
                    distance: SquaredL2::evaluate(query, vector),
                })
                .collect();
            selected.sort_by(|a, b| {
                a.distance
                    .total_cmp(&b.distance)
                    .then_with(|| a.id.cmp(&b.id))
            });
            selected.truncate(nprobe);
            Ok(selected)
        }
    }
}

struct Lists<'a, 'p>(&'a mut Build<'p>);

impl ListAccessor for Lists<'_, '_> {
    type InternalId = u32;
    type ListId = u32;

    fn list_size(&mut self, list: u32) -> impl SendFuture<ANNResult<usize>> {
        async move {
            self.0
                .pending
                .list_size(list, self.0.stored_members(list)?.len())
        }
    }

    fn append_members(&mut self, list: u32, members: Vec<u32>) -> impl SendFuture<ANNResult<()>> {
        async move {
            self.0.stored_members(list)?;
            self.0.pending.append(list, members)
        }
    }

    fn set_members(&mut self, list: u32, members: Vec<u32>) -> impl SendFuture<ANNResult<()>> {
        async move {
            self.0.stored_members(list)?;
            self.0.pending.set(list, members)
        }
    }

    fn create_list(
        &mut self,
        centroid: &[f32],
        members: Vec<u32>,
    ) -> impl SendFuture<ANNResult<u32>> {
        async move {
            if centroid.len() != self.0.store.dim {
                return Err(ANNError::message("wrong centroid dimension"));
            }
            let id = self.0.next_list;
            self.0.next_list += 1;
            self.0.created.insert(id, centroid.to_vec());
            self.0.pending.set(id, members)?;
            Ok(id)
        }
    }

    fn retire_list(&mut self, list: u32) -> impl SendFuture<ANNResult<()>> {
        async move {
            self.0.stored_members(list)?;
            self.0.pending.retire(list)
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
        = Lists<'a, 'p>
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
        Lists(self)
    }

    fn fill(&mut self, lists: &[u32]) -> impl SendFuture<ANNResult<WorkingSet<'_, u32, u32>>> {
        async move {
            self.fills.set(self.fills.get() + 1);
            if self.store.fail_fill {
                return Err(ANNError::message("injected fill failure"));
            }
            let mut view = Vec::with_capacity(lists.len());
            for &list in lists {
                let size = self
                    .pending
                    .list_size(list, self.stored_members(list)?.len())?;
                let mut points = Vec::with_capacity(size);
                for id in self.pending.members(list, self.stored_members(list)?)? {
                    let vector = self
                        .batch
                        .as_ref()
                        .and_then(|batch| batch.get(id))
                        .or_else(|| self.store.points.get(&id).map(Vec::as_slice))
                        .ok_or_else(|| ANNError::message(format!("missing point {id}")))?;
                    points.push(PointRef { id, vector });
                }
                view.push(ListView { id: list, points });
            }
            Ok(view)
        }
    }

    fn finish(self) -> impl SendFuture<ANNResult<()>> {
        async move {
            if self.store.fail_finish {
                return Err(ANNError::message("injected finish failure"));
            }
            if let Some(batch) = self.batch {
                for point in batch.points() {
                    self.store.points.insert(point.id, point.vector.to_vec());
                }
            }
            for (id, centroid) in self.created {
                self.store.lists.insert(
                    id,
                    StoredList {
                        centroid,
                        members: Vec::new(),
                    },
                );
            }
            for (id, change) in self.pending.into_changes() {
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
            self.store.next_list = self.next_list;
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
        let next_list = store.next_list;
        Ok(Build {
            store,
            pending: PendingLists::default(),
            created: BTreeMap::new(),
            batch: None,
            next_list,
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
async fn initialize_and_append_publish_actual_memberships() {
    let mut index = index(
        Store {
            dim: 2,
            next_list: 1,
            ..Store::default()
        },
        2,
    );
    let centers = matrix(&[[0.0, 0.0], [10.0, 0.0]]);
    require_send(index.initialize(&Strategy, &DefaultContext, centers.as_view()))
        .await
        .unwrap();
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
    assert_eq!(index.provider().finishes, 2);
    assert_eq!(index.provider().fills, 0);

    let empty: [(u32, &[f32]); 0] = [];
    assert_eq!(
        index
            .insert_batch(&Strategy, &DefaultContext, &empty)
            .await
            .unwrap(),
        InsertStats::default()
    );
    assert_eq!(index.provider().finishes, 2);
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
async fn working_set_borrows_stored_and_prepared_rows() {
    let mut store = seeded();
    {
        let mut build = Strategy
            .build_accessor(&mut store, &DefaultContext)
            .unwrap();
        let points = [(5, &[1.0, 1.0][..]), (6, &[3.0, 1.0][..])];
        let batch = Strategy.prepare(&mut build, &points).await.unwrap();
        let old_pointer = build.store.points[&10_000].as_ptr();
        build
            .lists()
            .append_members(1, vec![10_006, 10_005])
            .await
            .unwrap();
        assert_eq!(build.lists().list_size(1).await.unwrap(), 4);
        {
            let view = require_send(build.fill(&[1])).await.unwrap();
            assert_eq!(
                view[0]
                    .points
                    .iter()
                    .map(|point| point.id)
                    .collect::<Vec<_>>(),
                [10_000, 10_001, 10_006, 10_005]
            );
            assert_eq!(view[0].points[0].vector.as_ptr(), old_pointer);
            assert!(std::ptr::eq(
                view[0].points[2].vector,
                batch.get(10_006).unwrap()
            ));
            assert!(std::ptr::eq(
                view[0].points[3].vector,
                batch.get(10_005).unwrap()
            ));
        }
        build
            .lists()
            .set_members(1, vec![10_000, 10_005])
            .await
            .unwrap();
        assert_eq!(build.lists().list_size(1).await.unwrap(), 2);
        assert_eq!(build.fill(&[1]).await.unwrap()[0].points.len(), 2);
        build.lists().retire_list(1).await.unwrap();
        assert!(build.lists().list_size(1).await.is_err());
        assert!(build.fill(&[1]).await.is_err());
        assert!(build.lists().append_members(1, vec![10_001]).await.is_err());
    }
    assert_eq!(store.lists[&1].members, [10_000, 10_001]);
    assert!(!store.points.contains_key(&10_005));
    assert_eq!(store.finishes, 0);
}

#[tokio::test]
async fn failed_planning_or_publication_is_not_reported_as_success() {
    for fail_fill in [true, false] {
        let mut store = seeded();
        store.fail_fill = fail_fill;
        store.fail_finish = !fail_fill;
        let mut index = index(store, 2);
        let points = [(5, &[1.0, 1.0][..])];
        let error = require_send(index.insert_batch(&Strategy, &DefaultContext, &points))
            .await
            .unwrap_err();
        let expected = if fail_fill {
            "injected fill failure"
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
async fn invalid_initialization_and_uninitialized_insertion_fail() {
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
    for centers in [
        matrix::<2>(&[]),
        matrix(&[[0.0]]),
        matrix(&[[f32::NAN, 0.0]]),
    ] {
        assert!(
            index
                .initialize(&Strategy, &DefaultContext, centers.as_view())
                .await
                .is_err()
        );
    }
    let points = [(7, &[0.0, 1.0][..])];
    assert!(
        index
            .insert_batch(&Strategy, &DefaultContext, &points)
            .await
            .is_err()
    );
    assert_eq!(index.provider().finishes, 0);
    let centers = matrix(&[[0.0, 0.0]]);
    index
        .initialize(&Strategy, &DefaultContext, centers.as_view())
        .await
        .unwrap();
    assert!(
        index
            .initialize(&Strategy, &DefaultContext, centers.as_view())
            .await
            .is_err()
    );
    assert_eq!(index.provider().finishes, 1);
}

#[test]
fn assignment_can_cross_parent_boundaries() {
    let left = [[-100.0], [-99.0], [-10.0], [-9.0], [-0.1]];
    let right = [[0.1], [0.2], [100.0], [101.0]];
    let view = vec![
        ListView {
            id: 1u32,
            points: left
                .iter()
                .enumerate()
                .map(|(id, vector)| PointRef {
                    id: id as u32,
                    vector,
                })
                .collect(),
        },
        ListView {
            id: 2,
            points: right
                .iter()
                .enumerate()
                .map(|(id, vector)| PointRef {
                    id: 100 + id as u32,
                    vector,
                })
                .collect(),
        },
    ];
    let children = clustering::split(&view, 1, 10, &mut StdRng::seed_from_u64(7)).unwrap();
    assert!(
        children.members[2..]
            .iter()
            .any(|members| members.contains(&4))
    );
}

#[test]
fn two_means_keeps_empty_children_finite_and_prefers_first_ties() {
    let rows = [[f32::MAX], [f32::MAX], [f32::MAX]];
    let view = vec![ListView {
        id: 1u32,
        points: rows
            .iter()
            .enumerate()
            .map(|(id, row)| PointRef {
                id: id as u32,
                vector: row,
            })
            .collect(),
    }];
    let zero = clustering::split(&view, 1, 0, &mut StdRng::seed_from_u64(7)).unwrap();
    let one = clustering::split(&view, 1, 1, &mut StdRng::seed_from_u64(7)).unwrap();
    assert_eq!(zero.centroids.as_slice(), one.centroids.as_slice());
    assert!(
        zero.centroids
            .as_slice()
            .iter()
            .all(|value| value.is_finite())
    );
    assert_eq!(zero.members[0], [0, 1, 2]);
    assert!(zero.members[1].is_empty());
}
