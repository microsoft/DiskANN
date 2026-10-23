/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{cell::Cell, collections::BTreeMap, future::Future, num::NonZeroUsize};

use diskann_utils::{
    Reborrow,
    future::SendFuture,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use tokio::sync::{Mutex, MutexGuard};

use self::{
    pending::{Membership, PendingLists},
    support::*,
};
use super::{
    Config, IVFIndex, InsertStats, ReassignCount,
    build::{SplitAndAssign, assignment, clustering},
    region::{AsMatRef, RegionUpdate, RegionView, View},
    traits::{
        BuildAccessor, BuildStrategy, HasIds, NearestList, Router, SearchAccessor, SearchStrategy,
    },
};
use crate::{
    ANNError, ANNResult,
    flat::DistancesUnordered,
    graph::{IdDistance, SearchOutputBuffer},
    neighbor::{BackInserter, Neighbor},
    provider::{DefaultContext, HasId},
    utils::ONE,
};

mod pending;
mod support;

#[tokio::test]
async fn routing_callbacks_emit_ranked_results() {
    let expected: [(u32, f32); 3] = [(1, 25.0), (2, 25.0), (3, 8836.0)];
    for store in [seeded(), Store::default()] {
        let router = Coarse {
            stored: &store.lists,
            fail_route: false,
            fail_nearest: false,
        };
        for limit in [1, 2, usize::MAX] {
            let mut emitted = Vec::new();
            require_send(router.route(
                (9, &[6u8, 0][..]),
                NonZeroUsize::new(limit).unwrap(),
                |id, distance| emitted.push((*id, distance)),
            ))
            .await
            .unwrap();
            let expected = &expected[..limit.min(store.lists.len())];
            assert_eq!(emitted, expected);
        }
    }
}

#[tokio::test]
async fn nearest_callbacks_filter_before_limiting_and_count_emitted_results() {
    let store = seeded();
    let nearest = Coarse {
        stored: &store.lists,
        fail_route: false,
        fail_nearest: false,
    };
    for (limit, excluded, expected) in [
        (1, vec![1], vec![(2, 100.0_f32)]),
        (2, vec![1], vec![(2, 100.0), (3, 9801.0)]),
        (usize::MAX, vec![], vec![(1, 0.0), (2, 100.0), (3, 9801.0)]),
        (2, vec![1, 2, 3], vec![]),
    ] {
        let mut emitted = Vec::new();
        let count = require_send(nearest.find(
            1,
            NonZeroUsize::new(limit).unwrap(),
            |id| !excluded.contains(id),
            |id, distance| emitted.push((*id, distance)),
        ))
        .await
        .unwrap();
        assert_eq!(count, expected.len());
        assert_eq!(emitted, expected);
    }
}

#[tokio::test]
async fn nearest_lookup_failures_do_not_emit_results() {
    let store = seeded();
    for (fail_nearest, id, expected) in [
        (false, 99, "unknown list 99"),
        (true, 1, "injected nearest failure"),
    ] {
        let nearest = Coarse {
            stored: &store.lists,
            fail_route: false,
            fail_nearest,
        };
        let mut emitted = Vec::new();
        let error = require_send(nearest.find(
            id,
            ONE,
            |_| true,
            |id, distance| emitted.push((*id, distance)),
        ))
        .await
        .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        assert!(emitted.is_empty());
    }
}

#[tokio::test]
async fn routing_preserves_native_inputs_and_id_order() {
    let strategy = Strategy::new(seeded());
    let mut build: Build<'_, u8> = strategy.build_accessor(&DefaultContext).unwrap();
    let points: [(u32, &[u8]); 3] = [(9, &[11, 0]), (7, &[1, 0]), (8, &[0, 0])];
    let mut emitted = Vec::new();
    let inserted = require_send(build.route(&points[..], |&list, size| {
        emitted.push((list, size.get()));
    }))
    .await
    .unwrap();
    assert_eq!(inserted.get(), points.len());
    assert_eq!(emitted, [(1, 4), (2, 3)]);
    assert_eq!(
        build
            .pending
            .members(1, &build.store.lists[&1].members)
            .unwrap()
            .collect::<Vec<_>>(),
        [10_000, 10_001, 10_007, 10_008]
    );
    assert_eq!(
        build
            .pending
            .members(2, &build.store.lists[&2].members)
            .unwrap()
            .collect::<Vec<_>>(),
        [10_002, 10_003, 10_009]
    );
    for point in &points {
        let id = point.0 + 10_000;
        assert_eq!(build.staged[&id], point.1);
        assert!(!build.store.points.contains_key(&id));
    }
    let router = build.nearest();
    for (point, expected) in points.iter().zip([2, 1, 1]) {
        let mut selected = Vec::new();
        require_send(router.route(*point, ONE, |&id, _| selected.push(id)))
            .await
            .unwrap();
        assert_eq!(selected, [expected]);
    }
    drop(build);
    let store = strategy.store();
    assert_eq!(store.points.len(), 5);
    assert_eq!(store.finishes, 0);
}

#[tokio::test]
async fn routing_reports_sizes_including_previous_pending_appends() {
    let strategy = Strategy::new(seeded());
    let mut build: Build<'_, u8> = strategy.build_accessor(&DefaultContext).unwrap();
    let points: [(u32, &[u8]); 2] = [(5, &[1, 0]), (6, &[2, 0])];
    let mut emitted = Vec::new();
    for point in points {
        let inserted = require_send(build.route(std::slice::from_ref(&point), |&list, size| {
            emitted.push((list, size.get()));
        }))
        .await
        .unwrap();
        assert_eq!(inserted.get(), 1);
    }
    assert_eq!(emitted, [(1, 3), (1, 4)]);
    assert_eq!(build.pending.list_size(1, 2).unwrap(), 4);
    assert_eq!(build.store.lists[&1].members, [10_000, 10_001]);
    assert_eq!(build.store.finishes, 0);

    require_send(build.finish()).await.unwrap();
    let store = strategy.store();
    assert_eq!(store.lists[&1].members, [10_000, 10_001, 10_005, 10_006]);
    assert_eq!(store.points[&10_005], [1.0, 0.0]);
    assert_eq!(store.points[&10_006], [2.0, 0.0]);
    assert_eq!(store.finishes, 1);
}

#[tokio::test]
async fn failed_routing_leaves_no_staged_points_or_appends() {
    for fail_route in [false, true] {
        let mut store = seeded();
        store.fail_route = fail_route;
        if !fail_route {
            store.lists.clear();
        }
        let strategy = Strategy::new(store);
        let mut build: Build<'_, f32> = strategy.build_accessor(&DefaultContext).unwrap();
        let points = [(5, &[1.0, 0.0][..])];
        let mut emitted = Vec::new();
        let error = require_send(build.route(&points[..], |&list, size| {
            emitted.push((list, size.get()));
        }))
        .await
        .unwrap_err();
        let expected = if fail_route {
            "injected routing failure"
        } else {
            "IVF routing returned no live centroid"
        };
        assert!(error.to_string().contains(expected), "{error}");
        assert!(emitted.is_empty());
        assert!(build.staged.is_empty());
        assert!(build.pending.clone().into_changes().next().is_none());
        assert!(!build.store.points.contains_key(&10_005));
        assert_eq!(build.store.finishes, 0);
    }
}

#[tokio::test]
async fn native_inputs_are_inserted_using_the_assigned_ids() {
    let (mut index, strategy) = index(seeded(), 4);
    let points: [(u32, &[u8]); 3] = [(9, &[11, 0]), (7, &[1, 0]), (8, &[0, 0])];
    let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points[..]))
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
        let mut emitted = Vec::new();
        assert!(
            require_send(build.route(&points[..], |&list, size| {
                emitted.push((list, size.get()));
            }))
            .await
            .is_err()
        );
        assert!(emitted.is_empty());
        assert!(build.staged.is_empty());
        assert!(build.pending.clone().into_changes().next().is_none());
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
    let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points[..]))
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
}

#[tokio::test]
async fn empty_batches_are_rejected_without_staging_or_publication() {
    let (mut index, strategy) = index(seeded(), 2);
    let points: &[(u32, &[f32])] = &[];
    let mut emitted = Vec::new();
    {
        let mut build: Build<'_, f32> = strategy.build_accessor(&DefaultContext).unwrap();
        let error = require_send(build.route(points, |&list, size| {
            emitted.push((list, size.get()));
        }))
        .await
        .unwrap_err();
        assert!(error.to_string().contains("cannot insert an empty batch"));
        assert!(emitted.is_empty());
        assert!(build.staged.is_empty());
        assert!(build.pending.clone().into_changes().next().is_none());
    }
    let error = require_send(index.insert_batch(&strategy, &DefaultContext, points))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("cannot insert an empty batch"));
    let store = strategy.store();
    assert_eq!(store.points.len(), 5);
    assert_eq!(store.lists[&1].members, [10_000, 10_001]);
    assert_eq!(store.lists[&2].members, [10_002, 10_003]);
    assert_eq!(store.finishes, 0);
    assert_eq!(store.fills, 0);
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
    let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points[..]))
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
    let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points[..]))
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
    assert_eq!(store.filled_lists, [1, 2, 3]);
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
    for (centers, one_candidate) in [
        (vec![], vec![]),
        (vec![100.0, 200.0, 300.0], vec![3]),
        (vec![0.0, 12.0, 200.0], vec![3, 4]),
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
            let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points[..]))
                .await
                .unwrap();
            let store = strategy.store();
            let neighbors = match (count, centers.len()) {
                (0, _) | (_, 0) => vec![],
                (1, _) => one_candidate.clone(),
                (2, _) => vec![3, 4],
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

#[tokio::test]
async fn region_view_borrows_inputs_while_updates_are_recorded() {
    let strategy = Strategy::new(seeded());
    {
        let mut build: Build<'_, f32> = strategy.build_accessor(&DefaultContext).unwrap();
        let points = [(5, &[1.0, 1.0][..]), (6, &[3.0, 1.0][..])];
        let reversed = [points[1], points[0]];
        let mut emitted = Vec::new();
        build
            .route(&reversed[..], |&list, size| {
                emitted.push((list, size.get()));
            })
            .await
            .unwrap();
        assert_eq!(emitted, [(1, 4)]);
        {
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
            SplitAndAssign::new(
                clustering::TwoMeans::new(
                    NonZeroUsize::new(5).unwrap(),
                    NonZeroUsize::new(2).unwrap(),
                ),
                assignment::RegionalNearest,
            )
            .run(&region, &mut update, &[1], &[])
            .unwrap();
            assert_eq!(view.members, [10_000, 10_001, 10_006, 10_005]);
            assert_eq!(vectors.row(2), points[1].1);
            assert!(update.set_members(1, &[]).is_err());
        }
        assert_eq!(build.dim().get(), 2);
        assert_eq!(build.created.len(), 2);
        let mut members: Vec<_> = build
            .created
            .iter()
            .flat_map(|list| list.members.iter().copied())
            .collect();
        members.sort_unstable();
        assert_eq!(members, [10_000, 10_001, 10_005, 10_006]);
        assert!(build.pending.is_retired(1));
    }
    let store = strategy.store();
    assert_eq!(store.lists[&1].members, [10_000, 10_001]);
    assert!(!store.points.contains_key(&10_005));
    assert_eq!(store.finishes, 0);
}

#[tokio::test]
async fn borrowed_updates_share_pending_memberships_and_publish_on_finish() {
    let strategy = Strategy::new(seeded());
    let mut build: Build<'_, f32> = strategy.build_accessor(&DefaultContext).unwrap();
    let points = [
        (5, &[1.0, 0.0][..]),
        (6, &[11.0, 0.0][..]),
        (7, &[100.0, 0.0][..]),
    ];
    build.route(&points[..], |_, _| {}).await.unwrap();

    {
        let (region, mut update) = require_send(build.fill([1, 2].into_iter())).await.unwrap();
        let parent = region.get_view(1).unwrap();
        let neighbor = region.get_view(2).unwrap();
        assert_eq!(parent.members, [10_000, 10_001, 10_005]);
        assert_eq!(neighbor.members, [10_002, 10_003, 10_006]);
        assert!(update.set_members(3, &[]).is_err());
        assert!(update.retire(3).is_err());

        update.set_members(2, &[10_002, 10_003, 10_005]).unwrap();
        update
            .create(&[1.0, 0.0], &[10_000, 10_001, 10_006])
            .unwrap();
        update.retire(1).unwrap();

        assert_eq!(parent.members, [10_000, 10_001, 10_005]);
        assert_eq!(neighbor.members, [10_002, 10_003, 10_006]);
        assert_eq!(region.get_vectors(2).unwrap().row(2), points[1].1);
    }

    assert!(build.pending.is_retired(1));
    assert_eq!(build.pending.list_size(2, 2).unwrap(), 3);
    assert_eq!(build.created.len(), 1);
    assert_eq!(build.store.points.len(), 5);
    assert_eq!(build.store.lists[&1].members, [10_000, 10_001]);
    assert_eq!(build.store.lists[&2].members, [10_002, 10_003]);
    assert_eq!(build.store.lists.len(), 3);
    assert_eq!(build.store.next_list, 4);
    assert_eq!(build.store.finishes, 0);

    require_send(build.finish()).await.unwrap();
    let store = strategy.store();
    assert!(!store.lists.contains_key(&1));
    assert_eq!(store.lists[&2].members, [10_002, 10_003, 10_005]);
    assert_eq!(store.lists[&3].members, [10_004, 10_007]);
    assert_eq!(store.lists[&4].members, [10_000, 10_001, 10_006]);
    assert_eq!(store.lists[&4].centroid, [1.0, 0.0]);
    assert_eq!(store.points.len(), 8);
    assert_eq!(store.lists.len(), 3);
    assert_eq!(store.next_list, 5);
    assert_eq!(store.finishes, 1);
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
        let error = require_send(index.insert_batch(&strategy, &DefaultContext, &points[..]))
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
    let error = index
        .insert_batch(&strategy, &DefaultContext, &points[..])
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("IVF routing returned no live centroid"),
        "{error}"
    );
    assert_eq!(strategy.store().finishes, 0);
}

#[tokio::test]
async fn search_scans_selected_lists_and_returns_the_best_k() {
    let mut store = seeded();
    store.points.insert(10_004, vec![7.5, 0.0]);
    let (index, strategy) = index(store, 2);
    let query = [7.5f32, 0.0];
    for (nprobe, k, expected) in [
        (1, 3, vec![(10_002, 6.25_f32), (10_003, 20.25)]),
        (2, 3, vec![(10_002, 6.25), (10_003, 20.25), (10_001, 30.25)]),
        (3, 2, vec![(10_004, 0.0), (10_002, 6.25)]),
        (
            usize::MAX,
            10,
            vec![
                (10_004, 0.0),
                (10_002, 6.25),
                (10_003, 20.25),
                (10_001, 30.25),
                (10_000, 56.25),
            ],
        ),
    ] {
        let mut output = Vec::new();
        let count = require_send(index.search(
            &strategy,
            &DefaultContext,
            query.as_slice(),
            NonZeroUsize::new(k).unwrap(),
            NonZeroUsize::new(nprobe).unwrap(),
            &mut output,
        ))
        .await
        .unwrap();
        assert_eq!(count, expected.len());
        assert_eq!(
            output
                .into_iter()
                .map(Neighbor::as_tuple)
                .collect::<Vec<_>>(),
            expected
        );
    }
    let store = strategy.store();
    assert_eq!(store.points.len(), 5);
    assert_eq!(store.lists.len(), 3);
    assert_eq!(store.finishes, 0);
}

#[tokio::test]
async fn search_respects_output_buffer_capacity_and_existing_results() {
    let (index, strategy) = index(seeded(), 2);
    let query = [7.5f32, 0.0];
    let mut ids = [u32::MAX; 3];
    let mut distances = [f32::INFINITY; 3];
    {
        let mut output = IdDistance::new(&mut ids, &mut distances);
        assert!(output.push(Neighbor::new(42, -1.0)).is_available());
        for expected_count in [2, 0] {
            let count = require_send(index.search(
                &strategy,
                &DefaultContext,
                query.as_slice(),
                NonZeroUsize::new(4).unwrap(),
                NonZeroUsize::new(3).unwrap(),
                &mut output,
            ))
            .await
            .unwrap();
            assert_eq!(count, expected_count);
            assert_eq!(output.current_len(), 3);
        }
    }
    assert_eq!(ids, [42, 10_002, 10_003]);
    assert_eq!(distances, [-1.0, 6.25, 20.25]);

    let mut buffer = [];
    let mut output = BackInserter::new(&mut buffer);
    assert_eq!(
        require_send(index.search(
            &strategy,
            &DefaultContext,
            query.as_slice(),
            ONE,
            ONE,
            &mut output,
        ))
        .await
        .unwrap(),
        0
    );
}

#[tokio::test]
async fn search_returns_no_results_for_empty_regions() {
    for empty_index in [false, true] {
        let mut store = seeded();
        if empty_index {
            store.lists.clear();
            store.points.clear();
        } else {
            store.lists.get_mut(&2).unwrap().members.clear();
        }
        let (index, strategy) = index(store, 2);
        let mut output = vec![Neighbor::new(42, -1.0)];
        assert_eq!(
            require_send(index.search(
                &strategy,
                &DefaultContext,
                &[7.5f32, 0.0][..],
                ONE,
                ONE,
                &mut output,
            ))
            .await
            .unwrap(),
            0
        );
        assert_eq!(
            output
                .into_iter()
                .map(Neighbor::as_tuple)
                .collect::<Vec<_>>(),
            [(42, -1.0)]
        );
    }
}

#[tokio::test]
async fn search_failures_do_not_publish_partial_results() {
    for (case, expected) in [
        ("dimension", "invalid query vector"),
        ("nonfinite", "invalid query vector"),
        ("route", "injected routing failure"),
        ("scan", "missing point 10003"),
    ] {
        let mut store = seeded();
        if case == "route" {
            store.fail_route = true;
        }
        if case == "scan" {
            store.points.remove(&10_003);
        }
        let (index, strategy) = index(store, 2);
        let mut query = [7.5f32, 0.0];
        if case == "nonfinite" {
            query[0] = f32::NAN;
        }
        let query = if case == "dimension" {
            &query[..1]
        } else {
            &query[..]
        };
        let mut output = vec![Neighbor::new(42, -1.0)];
        let error = require_send(index.search(
            &strategy,
            &DefaultContext,
            query,
            NonZeroUsize::new(3).unwrap(),
            ONE,
            &mut output,
        ))
        .await
        .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(
            output
                .into_iter()
                .map(Neighbor::as_tuple)
                .collect::<Vec<_>>(),
            [(42, -1.0)]
        );
        assert_eq!(strategy.store().finishes, 0);
    }
}

#[tokio::test]
async fn search_supports_native_queries_after_insertion_and_splitting() {
    let (mut index, strategy) = index(seeded(), 2);
    let points = [(5, &[8u8, 0][..])];
    let stats = index
        .insert_batch(&strategy, &DefaultContext, &points[..])
        .await
        .unwrap();
    assert_eq!(stats.splits, 1);

    let mut output = Vec::new();
    assert_eq!(
        require_send(index.search(
            &strategy,
            &DefaultContext,
            &[8u8, 0][..],
            ONE,
            NonZeroUsize::new(usize::MAX).unwrap(),
            &mut output,
        ))
        .await
        .unwrap(),
        1
    );
    assert_eq!(
        output
            .into_iter()
            .map(Neighbor::as_tuple)
            .collect::<Vec<_>>(),
        [(10_005, 0.0)]
    );
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
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = Update::new(&sources, 1, &mut memberships, &mut created);
    SplitAndAssign::new(
        clustering::TwoMeans::new(NonZeroUsize::new(5).unwrap(), NonZeroUsize::new(1).unwrap()),
        assignment::RegionalNearest,
    )
    .run(&region, &mut update, &[1], &[2, 3, 4])
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
    let mut changes: BTreeMap<_, _> = std::mem::take(update.memberships).into_changes().collect();
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
async fn split_only_edits_explicit_parents_and_missing_sources_leave_no_edits() {
    let mut store = seeded();
    store.points.insert(10_001, vec![10.0, 0.0]);
    store.points.insert(10_002, vec![5.0, 0.0]);
    store.points.insert(10_003, vec![10.1, 0.0]);
    store.lists.get_mut(&1).unwrap().centroid = vec![5.0, 0.0];
    store.lists.get_mut(&2).unwrap().centroid = vec![5.0, 0.0];
    let strategy = Strategy::new(store);
    let mut build: Build<'_, f32> = strategy.build_accessor(&DefaultContext).unwrap();
    let points = [(5, &[5.0, 0.0][..])];
    let mut emitted = Vec::new();
    build
        .route(&points[..], |&list, size| {
            emitted.push((list, size.get()));
        })
        .await
        .unwrap();
    assert_eq!(emitted, [(1, 3)]);

    {
        let (region, mut update) = build.fill([1, 2].into_iter()).await.unwrap();
        assert!(
            SplitAndAssign::new(
                clustering::TwoMeans::new(
                    NonZeroUsize::new(5).unwrap(),
                    NonZeroUsize::new(2).unwrap(),
                ),
                assignment::RegionalNearest,
            )
            .run(&region, &mut update, &[1, 99], &[])
            .is_err()
        );
        assert!(
            SplitAndAssign::new(
                clustering::TwoMeans::new(
                    NonZeroUsize::new(5).unwrap(),
                    NonZeroUsize::new(2).unwrap(),
                ),
                assignment::RegionalNearest,
            )
            .run(&region, &mut update, &[1], &[99])
            .is_err()
        );
        assert!(update.created.is_empty());
        let changes: Vec<_> = update.memberships.clone().into_changes().collect();
        assert!(
            matches!(changes.as_slice(), [(1, Membership::Append(members))] if members == &[10_005])
        );
        SplitAndAssign::new(
            clustering::TwoMeans::new(NonZeroUsize::new(5).unwrap(), NonZeroUsize::new(2).unwrap()),
            assignment::RegionalNearest,
        )
        .run(&region, &mut update, &[1], &[])
        .unwrap();
        assert_eq!(update.created.len(), 2);
    }
    require_send(build.finish()).await.unwrap();

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
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = Update::new(&sources, 1, &mut memberships, &mut created);
    SplitAndAssign::new(
        clustering::TwoMeans::new(
            NonZeroUsize::new(10).unwrap(),
            NonZeroUsize::new(1).unwrap(),
        ),
        assignment::RegionalNearest,
    )
    .run(&region, &mut update, &[1, 2], &[])
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
        let mut memberships = PendingLists::default();
        let mut created = Vec::new();
        let mut update = Update::new(&sources, 1, &mut memberships, &mut created);
        SplitAndAssign::new(
            clustering::TwoMeans::new(NonZeroUsize::new(1).unwrap(), NonZeroUsize::new(1).unwrap()),
            assignment::RegionalNearest,
        )
        .run(&region, &mut update, &[1], &[])
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

    impl<R> clustering::Clusterer<u32, u32, R> for RowsAsCenters
    where
        R: for<'a> RegionView<u32, u32, VectorsRef<'a> = rowmajor::Ref<'a, f32>>,
    {
        fn fit(
            &mut self,
            sources: &[u32],
            region: &R,
            mut dst: rowmajor::Mut<'_, f32>,
        ) -> ANNResult<()> {
            assert_eq!(dst.nrows(), sources.len() * 2);
            assert_eq!(dst.ncols(), 1);
            for (&source, centers) in sources.iter().zip(dst.as_mut_slice().chunks_exact_mut(2)) {
                let vectors = region.get_vectors(source).unwrap();
                let points = vectors.reborrow();
                assert_eq!(points.nrows(), 2);
                centers.copy_from_slice(points.as_slice());
            }
            Ok(())
        }

        fn num_centers(&self, sources: usize) -> Option<usize> {
            sources.checked_mul(2)
        }

        fn dim(&self) -> NonZeroUsize {
            NonZeroUsize::new(1).unwrap()
        }
    }

    struct ParentLocal {
        owners: BTreeMap<u32, Vec<u32>>,
    }

    impl<R> assignment::Assigner<u32, u32, R> for ParentLocal
    where
        R: for<'a> RegionView<u32, u32, VectorsRef<'a> = rowmajor::Ref<'a, f32>>,
    {
        fn assign<U>(
            &mut self,
            region: &R,
            update: &mut U,
            sources: &[u32],
            destinations: &[assignment::Target<'_, u32>],
        ) -> ANNResult<()>
        where
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
                let vectors = vectors.reborrow();
                let mut candidates: Vec<_> = destinations
                    .iter()
                    .enumerate()
                    .filter_map(|(index, destination)| {
                        matches!(&destination.list, assignment::Type::Exists(id) if *id == source)
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
                    assignment::Type::New => update.create(destination.centroid, &members)?,
                    assignment::Type::Exists(id) => update.set_members(*id, &members)?,
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
    let parents = [1, 2];
    let neighbors = [3, 4];
    let mut full_memberships = PendingLists::default();
    let mut full_created = Vec::new();
    let mut full = Update::new(&sources, 1, &mut full_memberships, &mut full_created);
    let mut splitter = SplitAndAssign::new(RowsAsCenters, assignment::RegionalNearest);
    splitter
        .run::<_, _, Region<'_>, _>(&region, &mut full, &parents, &neighbors)
        .unwrap();
    let mut restricted_memberships = PendingLists::default();
    let mut restricted_created = Vec::new();
    let mut restricted = Update::new(
        &sources,
        1,
        &mut restricted_memberships,
        &mut restricted_created,
    );
    SplitAndAssign::new(
        RowsAsCenters,
        ParentLocal {
            owners: [(3, vec![1]), (4, vec![1, 2])].into(),
        },
    )
    .run::<_, _, Region<'_>, _>(&region, &mut restricted, &parents, &neighbors)
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
    let mut repeated_memberships = PendingLists::default();
    let mut repeated_created = Vec::new();
    let mut repeated = Update::new(
        &sources,
        1,
        &mut repeated_memberships,
        &mut repeated_created,
    );
    splitter
        .run::<_, _, Region<'_>, _>(&region, &mut repeated, &parents, &neighbors)
        .unwrap();
    for (first, repeated) in full.created.iter().zip(repeated.created.iter()) {
        assert_eq!(first.centroid, repeated.centroid);
        assert_eq!(first.members, repeated.members);
    }
    let mut smaller_memberships = PendingLists::default();
    let mut smaller_created = Vec::new();
    let mut smaller = Update::new(&sources, 1, &mut smaller_memberships, &mut smaller_created);
    splitter
        .run::<_, _, Region<'_>, _>(&region, &mut smaller, &[1], &[])
        .unwrap();
    assert_eq!(smaller.created.len(), 2);
    assert_eq!(smaller.created[0].members, [0]);
    assert_eq!(smaller.created[1].members, [1]);
    for (update, expected) in [(full, vec![6]), (restricted, vec![5, 6])] {
        let mut changes: BTreeMap<_, _> =
            std::mem::take(update.memberships).into_changes().collect();
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

    impl<I, L, R: RegionView<I, L>> clustering::Clusterer<I, L, R> for InvalidFit {
        fn fit(&mut self, _: &[L], _: &R, _: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
            match self {
                Self::Error => Err(ANNError::message("injected fitting failure")),
                Self::CountOverflow => {
                    unreachable!("centroid count overflow should be rejected before fitting")
                }
            }
        }

        fn num_centers(&self, sources: usize) -> Option<usize> {
            sources.checked_mul(match self {
                Self::CountOverflow => usize::MAX,
                Self::Error => 2,
            })
        }

        fn dim(&self) -> NonZeroUsize {
            NonZeroUsize::new(1).unwrap()
        }
    }

    struct InvalidAssignment;

    impl<R: RegionView<u32, u32>> assignment::Assigner<u32, u32, R> for InvalidAssignment {
        fn assign<U>(
            &mut self,
            _: &R,
            _: &mut U,
            _: &[u32],
            _: &[assignment::Target<'_, u32>],
        ) -> ANNResult<()>
        where
            U: RegionUpdate<u32, u32>,
        {
            Err(ANNError::message("injected assignment failure"))
        }
    }

    fn assert_unwritten(update: Update<'_>) {
        assert!(update.created.is_empty());
        assert_eq!(std::mem::take(update.memberships).into_changes().count(), 0);
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
    let parents = [1];
    let neighbors = [];
    for (clusterer, expected) in [
        (InvalidFit::Error, "injected fitting failure"),
        (InvalidFit::CountOverflow, "centroid count overflow"),
    ] {
        let mut memberships = PendingLists::default();
        let mut created = Vec::new();
        let mut update = Update::new(&sources, 1, &mut memberships, &mut created);
        let selected: &[u32] = if matches!(clusterer, InvalidFit::CountOverflow) {
            &[1, 2]
        } else {
            &parents
        };
        let error = SplitAndAssign::new(clusterer, assignment::RegionalNearest)
            .run(&region, &mut update, selected, &neighbors)
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        assert_unwritten(update);
    }
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = Update::new(&sources, 1, &mut memberships, &mut created);
    let error = SplitAndAssign::new(
        clustering::TwoMeans::new(ONE, ONE),
        assignment::RegionalNearest,
    )
    .run(&region, &mut update, &[], &neighbors)
    .unwrap_err();
    assert!(error.to_string().contains("at least one parent"), "{error}");
    assert_unwritten(update);

    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = Update::new(&sources, 1, &mut memberships, &mut created);
    let error = SplitAndAssign::new(
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
fn region_vector_reborrows_share_source_storage() {
    let sources = BTreeMap::from([(
        1,
        Source {
            centroid: vec![0.0],
            members: vec![0, 1],
            vectors: matrix(&[[-1.0], [1.0]]),
        },
    )]);
    let region = Region::new(&sources);
    let view = region.get_view(1).unwrap();
    assert!(std::ptr::eq(view.members, sources[&1].members.as_slice()));
    let vectors = region.get_vectors(1).unwrap();
    let reborrowed = vectors.reborrow();
    assert!(std::ptr::eq(
        reborrowed.as_slice(),
        sources[&1].vectors.as_slice()
    ));
    assert_eq!(reborrowed.row(0), [-1.0]);
    assert_eq!(reborrowed.row(1), [1.0]);
}

#[test]
fn split_driver_reuses_clusterer_with_fallible_borrowed_batches() {
    struct Batch<'a> {
        vectors: rowmajor::Ref<'a, f32>,
        reads: &'a Cell<usize>,
        fail_on: Option<usize>,
    }

    struct BatchRef<'a> {
        vectors: rowmajor::Ref<'a, f32>,
        reads: &'a Cell<usize>,
        fail_on: Option<usize>,
    }

    impl<'r> Reborrow<'r> for Batch<'_> {
        type Target = BatchRef<'r>;

        fn reborrow(&'r self) -> Self::Target {
            BatchRef {
                vectors: self.vectors.reborrow(),
                reads: self.reads,
                fail_on: self.fail_on,
            }
        }
    }

    impl AsMatRef<f32> for BatchRef<'_> {
        type Error = ANNError;

        fn as_view(&self) -> ANNResult<rowmajor::Ref<'_, f32>> {
            let count = self.reads.get() + 1;
            self.reads.set(count);
            if self.fail_on == Some(count) {
                return Err(ANNError::message("injected batch conversion failure"));
            }
            Ok(self.vectors)
        }
    }

    struct BatchRegion<'a> {
        inner: Region<'a>,
        reads: Cell<usize>,
        fail_on: Option<usize>,
    }

    impl RegionView<u32, u32> for BatchRegion<'_> {
        type VectorsRef<'a> = BatchRef<'a>;
        type Vectors<'a>
            = Batch<'a>
        where
            Self: 'a;

        fn get_view(&self, list: u32) -> Option<View<'_, u32, u32>> {
            self.inner.get_view(list)
        }

        fn get_vectors(&self, list: u32) -> Option<Self::Vectors<'_>> {
            self.inner.get_vectors(list).map(|vectors| Batch {
                vectors,
                reads: &self.reads,
                fail_on: self.fail_on,
            })
        }
    }

    let mut driver = SplitAndAssign::new(
        clustering::TwoMeans::new(ONE, ONE),
        assignment::RegionalNearest,
    );
    for fail_on in [Some(1), Some(2), None] {
        let sources = BTreeMap::from([(
            1,
            Source {
                centroid: vec![0.0],
                members: vec![0, 1],
                vectors: matrix(&[[-1.0], [1.0]]),
            },
        )]);
        let region = BatchRegion {
            inner: Region::new(&sources),
            reads: Cell::new(0),
            fail_on,
        };
        let batch = region.get_vectors(1).unwrap();
        let borrowed = batch.reborrow();
        assert!(std::ptr::eq(borrowed.reads, &region.reads));
        assert!(std::ptr::eq(
            borrowed.vectors.as_slice(),
            sources[&1].vectors.as_slice()
        ));
        let mut memberships = PendingLists::default();
        let mut created = Vec::new();
        let mut update = Update::new(&sources, 1, &mut memberships, &mut created);
        let result = driver.run(&region, &mut update, &[1], &[]);
        if let Some(count) = fail_on {
            let error = result.unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("injected batch conversion failure")
            );
            assert_eq!(region.reads.get(), count);
            assert!(created.is_empty());
            assert_eq!(memberships.into_changes().count(), 0);
        } else {
            result.unwrap();
            assert_eq!(region.reads.get(), 2);
            assert_eq!(created.len(), 2);
            assert_eq!(created[0].centroid, [-1.0]);
            assert_eq!(created[0].members, [0]);
            assert_eq!(created[1].centroid, [1.0]);
            assert_eq!(created[1].members, [1]);
            assert!(matches!(
                memberships.into_changes().collect::<Vec<_>>().as_slice(),
                [(1, Membership::Retire)]
            ));
        }
    }
}

#[test]
fn split_driver_accepts_native_batches_without_matrix_access() {
    struct NativeBatch<'a>(&'a [[u8; 1]]);

    impl<'r> Reborrow<'r> for NativeBatch<'_> {
        type Target = &'r [[u8; 1]];

        fn reborrow(&'r self) -> Self::Target {
            self.0
        }
    }

    struct NativeRegion<'a> {
        inner: Region<'a>,
        vectors: &'a [[u8; 1]],
    }

    impl RegionView<u32, u32> for NativeRegion<'_> {
        type VectorsRef<'a> = &'a [[u8; 1]];
        type Vectors<'a>
            = NativeBatch<'a>
        where
            Self: 'a;

        fn get_view(&self, list: u32) -> Option<View<'_, u32, u32>> {
            self.inner.get_view(list)
        }

        fn get_vectors(&self, list: u32) -> Option<Self::Vectors<'_>> {
            self.inner.get_view(list).map(|_| NativeBatch(self.vectors))
        }
    }

    struct NativeCenters;

    impl<R> clustering::Clusterer<u32, u32, R> for NativeCenters
    where
        R: for<'a> RegionView<u32, u32, VectorsRef<'a> = &'a [[u8; 1]]>,
    {
        fn fit(
            &mut self,
            sources: &[u32],
            region: &R,
            mut dst: rowmajor::Mut<'_, f32>,
        ) -> ANNResult<()> {
            for (&source, centers) in sources.iter().zip(dst.as_mut_slice().chunks_exact_mut(2)) {
                let vectors = region.get_vectors(source).unwrap();
                let points = vectors.reborrow();
                for (point, center) in points.iter().zip(centers) {
                    *center = point[0].into();
                }
            }
            Ok(())
        }

        fn num_centers(&self, sources: usize) -> Option<usize> {
            sources.checked_mul(2)
        }

        fn dim(&self) -> NonZeroUsize {
            ONE
        }
    }

    struct OrderedAssignment;

    impl<R: RegionView<u32, u32>> assignment::Assigner<u32, u32, R> for OrderedAssignment {
        fn assign<U>(
            &mut self,
            region: &R,
            update: &mut U,
            sources: &[u32],
            destinations: &[assignment::Target<'_, u32>],
        ) -> ANNResult<()>
        where
            U: RegionUpdate<u32, u32>,
        {
            let view = region.get_view(sources[0]).unwrap();
            for (destination, member) in destinations.iter().zip(view.members) {
                update.create(destination.centroid, std::slice::from_ref(member))?;
            }
            update.retire(sources[0])
        }
    }

    let mut driver = SplitAndAssign::new(NativeCenters, OrderedAssignment);
    for offset in [0u8, 20] {
        let sources = BTreeMap::from([(
            1,
            Source {
                centroid: vec![0.0],
                members: vec![0, 1],
                vectors: matrix(&[[0.0], [10.0]]),
            },
        )]);
        let vectors = vec![[offset], [offset + 10]];
        let region = NativeRegion {
            inner: Region::new(&sources),
            vectors: &vectors,
        };
        let batch = region.get_vectors(1).unwrap();
        assert!(std::ptr::eq(batch.reborrow(), vectors.as_slice()));
        let mut memberships = PendingLists::default();
        let mut created = Vec::new();
        let mut update = Update::new(&sources, 1, &mut memberships, &mut created);
        driver
            .run::<_, _, NativeRegion<'_>, _>(&region, &mut update, &[1], &[])
            .unwrap();
        assert_eq!(created.len(), 2);
        assert_eq!(created[0].centroid, [f32::from(offset)]);
        assert_eq!(created[0].members, [0]);
        assert_eq!(created[1].centroid, [f32::from(offset + 10)]);
        assert_eq!(created[1].members, [1]);
        assert!(matches!(
            memberships.into_changes().collect::<Vec<_>>().as_slice(),
            [(1, Membership::Retire)]
        ));
    }
}

#[test]
fn two_means_matches_quantization_lloyds() {
    use clustering::{Clusterer, TwoMeans};

    let sources = BTreeMap::from([(
        1,
        Source {
            centroid: vec![0.0, 0.0],
            members: vec![0, 1, 2, 3],
            vectors: matrix(&[[0.0, 2.0], [2.0, 4.0], [6.0, 6.0], [10.0, 12.0]]),
        },
    )]);
    let region = Region::new(&sources);
    let points = sources[&1].vectors.as_view();
    for iterations in [1, 5] {
        let mut expected = points.subview(0..2).unwrap().to_rowmajor_owned();
        let _ = diskann_quantization::algorithms::kmeans::lloyds::lloyds(
            points,
            expected.as_view_mut(),
            iterations,
        );

        let mut actual = rowmajor::Owned::from_element(2, 2, 0.0);
        TwoMeans::new(
            NonZeroUsize::new(iterations).unwrap(),
            NonZeroUsize::new(2).unwrap(),
        )
        .fit(&[1], &region, actual.as_view_mut())
        .unwrap();
        assert_eq!(actual.as_slice(), expected.as_slice());
    }
}

#[test]
fn two_means_fits_only_the_destination_window() {
    use clustering::{Clusterer, TwoMeans};

    let sources = BTreeMap::from([(
        1,
        Source {
            centroid: vec![0.0, 0.0],
            members: vec![0, 1, 2, 3],
            vectors: matrix(&[[0.0, 2.0], [2.0, 4.0], [10.0, 12.0], [12.0, 14.0]]),
        },
    )]);
    let region = Region::new(&sources);
    let mut scratch = rowmajor::Owned::from_element(4, 2, -1.0);
    let mut clusterer = TwoMeans::new(
        NonZeroUsize::new(10).unwrap(),
        NonZeroUsize::new(2).unwrap(),
    );
    clusterer
        .fit(&[1], &region, scratch.subview_mut(1..3).unwrap())
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
        let sources = BTreeMap::from([(
            1,
            Source {
                centroid: vec![0.0; points.ncols()],
                members: (0..).take(points.nrows()).collect(),
                vectors: points,
            },
        )]);
        let region = Region::new(&sources);
        let mut dst = rowmajor::Owned::from_element(rows, dim, -1.0);
        assert!(clusterer.fit(&[1], &region, dst.as_view_mut()).is_err());
        assert!(dst.as_slice().iter().all(|&value| value == -1.0));
    }
}

#[test]
fn regional_assignment_reports_missing_sources_before_writing() {
    use assignment::{Assigner, RegionalNearest, Target, Type};

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
    let destinations = [Target {
        list: Type::New,
        centroid: &[0.0],
    }];
    for ids in [vec![99], vec![1, 99], vec![1, 2, 99]] {
        let mut memberships = PendingLists::default();
        let mut created = Vec::new();
        let mut update = Update::new(&sources, 1, &mut memberships, &mut created);
        let region = Region::new(&sources);
        let error = RegionalNearest
            .assign(&region, &mut update, &ids, &destinations)
            .unwrap_err();
        assert!(error.to_string().contains("source is missing"), "{error}");
        assert!(update.created.is_empty());
        assert_eq!(std::mem::take(update.memberships).into_changes().count(), 0);
    }
}
