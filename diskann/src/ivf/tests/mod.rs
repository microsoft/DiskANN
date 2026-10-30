/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    future::Future,
    num::NonZeroUsize,
};

use diskann_utils::{
    Reborrow,
    future::SendFuture,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use hashbrown::HashSet;
use tokio::sync::{Mutex, MutexGuard};

use self::{
    pending::{Membership, PendingLists},
    support::*,
};
use super::{
    Config, IVFIndex, InsertStats, ReassignCount,
    region::{AsMatRef, List, Update, View},
    split::{
        KMeansSplit, Linked, Links, Reassign, Split,
        policies::{
            Assign, FirstPoints, NearestCentroid, SplitError,
            kmeans::{NotEnoughPoints, Seed},
        },
    },
    traits::{
        Accumulator, BuildAccessor, BuildStrategy, HasIds, Nearest, SearchAccessor, SearchStrategy,
    },
};
use crate::{
    ANNError, ANNResult,
    graph::{IdDistance, SearchOutputBuffer},
    neighbor::{BackInserter, Neighbor, NeighborPriorityQueue},
    provider::DefaultContext,
    utils::{ONE, TWO},
};

mod pending;
mod support;

#[tokio::test]
async fn routing_callbacks_emit_ranked_results() {
    let expected: [(u32, f32); 3] = [(1, 25.0), (2, 25.0), (3, 8836.0)];
    for store in [seeded(), Store::default()] {
        let router = Coarse::new(&store.lists);
        for limit in [1, 2, usize::MAX] {
            let mut emitted = Vec::new();
            require_send(router.route(
                &[6u8, 0][..],
                NonZeroUsize::new(limit).unwrap(),
                |id, distance| emitted.push((id, distance)),
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
    let nearest = Coarse::new(&store.lists);
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
            |id| !excluded.contains(&id),
            |id, distance| emitted.push((id, distance)),
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
            fail_nearest,
            ..Coarse::new(&store.lists)
        };
        let mut emitted = Vec::new();
        let error = require_send(nearest.find(
            id,
            ONE,
            |_| true,
            |id, distance| emitted.push((id, distance)),
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
    let inserted = require_send(build.route(&points[..], |list, size| {
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
        require_send(router.route(point.1, ONE, |id, _| selected.push(id)))
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
        let inserted = require_send(build.route(std::slice::from_ref(&point), |list, size| {
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

#[test]
fn membership_deltas_compose_with_pending_appends_and_removals() {
    let stored = [1, 2, 3];
    let mut pending = PendingLists::<u32, u32>::default();
    pending.append(7, vec![4, 5]).unwrap();
    pending.remove(7, &[2, 4]).unwrap();
    assert_eq!(
        pending.members(7, &stored).unwrap().collect::<Vec<_>>(),
        [1, 3, 5]
    );
    assert_eq!(pending.list_size(7, stored.len()).unwrap(), 3);

    pending.append(7, vec![2, 4]).unwrap();
    assert_eq!(
        pending.members(7, &stored).unwrap().collect::<Vec<_>>(),
        [1, 2, 3, 5, 4]
    );
    assert_eq!(pending.list_size(7, stored.len()).unwrap(), 5);

    pending.remove(7, &[1, 5]).unwrap();
    pending.remove(7, &[2, 3, 4]).unwrap();
    assert_eq!(pending.members(7, &stored).unwrap().count(), 0);
    assert_eq!(pending.list_size(7, stored.len()).unwrap(), 0);
    pending.retire(7).unwrap();
    assert!(pending.members(7, &stored).is_err());
    assert!(pending.list_size(7, stored.len()).is_err());
    assert!(pending.append(7, vec![6]).is_err());
    assert!(pending.remove(7, &[1]).is_err());
    assert!(pending.retire(7).is_err());
    assert!(matches!(
        pending.into_changes().collect::<Vec<_>>().as_slice(),
        [(7, Membership::Retire)]
    ));
}

#[test]
fn membership_sizes_reject_overflow_and_excess_removals() {
    let mut pending = PendingLists::<u32, u32>::default();
    pending.append(1, vec![4]).unwrap();
    assert!(pending.list_size(1, usize::MAX).is_err());
    pending.remove(2, &[1]).unwrap();
    assert!(pending.list_size(2, 0).is_err());
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
        let error = require_send(build.route(&points[..], |list, size| {
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
            require_send(build.route(&points[..], |list, size| {
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
        let error = require_send(build.route(points, |list, size| {
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
async fn splitting_uses_the_configured_number_of_children() {
    let strategy = Strategy::new(seeded());
    let mut index = IVFIndex::<DefaultContext>::new(Config {
        dim: TWO,
        split_threshold: TWO,
        split_k: NonZeroUsize::new(3).unwrap(),
        reassign_count: ReassignCount::None,
        split_iterations: ONE,
        seed: 7,
    })
    .unwrap();
    let points: [(u32, &[f32]); 1] = [(5, &[1.0, 1.0][..])];
    let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points[..]))
        .await
        .unwrap();
    assert_eq!(
        stats,
        InsertStats {
            inserted: 1,
            splits: 1
        }
    );

    // List 1 and the appended point are replaced by three children, one point each.
    let store = strategy.store();
    assert!(!store.lists.contains_key(&1));
    let children: Vec<_> = store
        .lists
        .iter()
        .filter(|&(&id, _)| id >= 4)
        .map(|(_, list)| list.members.clone())
        .collect();
    assert_eq!(children, [vec![10_000], vec![10_001], vec![10_005]]);
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
    assert_eq!(store.update_finishes, 1);
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
    // Parent 1 is split first, then its children 4 and 5 are filled with lists 2 and 3.
    assert_eq!(store.fills, 2);
    assert_eq!(store.update_finishes, 2);
    assert_eq!(store.finishes, 1);
    assert_eq!(store.filled_lists, [1, 4, 5, 2, 3]);
    assert!(!store.lists.contains_key(&1));
    assert_eq!(store.lists.len(), 4);
    // Points 10_005 and 10_006 move from child 4 to list 2, and 10_003 from list 2 to
    // child 5; list 2 keeps its pending append 10_008 once.
    assert_eq!(store.lists[&4].members, [10_000, 10_007]);
    assert_eq!(store.lists[&5].members, [10_001, 10_003]);
    assert_eq!(store.lists[&2].centroid, [5.0, 0.0]);
    assert_eq!(store.lists[&2].members, [10_002, 10_008, 10_005, 10_006]);
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
async fn reassignment_links_each_child_to_its_nearest_other_lists() {
    for count in [0, 1, 2, 3, 4, usize::MAX] {
        let mut store = seeded();
        store.lists.insert(
            4,
            StoredList {
                centroid: vec![200.0, 0.0],
                members: Vec::new(),
            },
        );
        store.next_list = 5;
        store.fail_nearest = count == 0;
        let (mut index, strategy) = index_with_neighbors(store, 2, count);
        let points: [(u32, &[f32]); 2] = [(5, &[0.5, 0.0][..]), (6, &[10.5, 0.0][..])];
        let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points[..]))
            .await
            .unwrap();
        let store = strategy.store();
        assert_eq!(
            stats,
            InsertStats {
                inserted: 2,
                splits: 2
            }
        );

        // Parents 1 and 2 split into children 5, 6 and 7, 8. Each child is linked to the
        // other parent's children first, then to lists 3 and 4; never to its sibling,
        // which is nearer, or to a retired parent.
        let reassigned: &[u32] = match count {
            0 => &[],
            1 | 2 => &[5, 6, 7, 8],
            3 => &[5, 6, 7, 8, 3],
            _ => &[5, 6, 7, 8, 3, 4],
        };
        let filled: Vec<_> = [1, 2].iter().chain(reassigned).copied().collect();
        assert_eq!(store.filled_lists, filled);
        assert_eq!(store.fills, if count == 0 { 1 } else { 2 });
        assert_eq!(store.update_finishes, store.fills);
        assert_eq!(store.finishes, 1);

        // Every point is already nearest its own list, so reassignment moves nothing.
        assert!(!store.lists.contains_key(&1));
        assert!(!store.lists.contains_key(&2));
        let members: Vec<_> = store
            .lists
            .iter()
            .map(|(&id, list)| (id, list.members.clone()))
            .collect();
        assert_eq!(
            members,
            [
                (3, vec![10_004]),
                (4, vec![]),
                (5, vec![10_000, 10_005]),
                (6, vec![10_001]),
                (7, vec![10_002, 10_006]),
                (8, vec![10_003]),
            ]
        );
        assert_eq!(store.lists[&3].centroid, [100.0, 0.0]);
        assert_eq!(store.lists[&4].centroid, [200.0, 0.0]);
    }
}

#[tokio::test]
async fn children_of_different_parents_exchange_points() {
    let store = Store {
        dim: 1,
        points: [
            (10_000, vec![0.0]),
            (10_001, vec![40.0]),
            (10_002, vec![62.0]),
            (10_003, vec![100.0]),
        ]
        .into(),
        lists: [
            (
                1,
                StoredList {
                    centroid: vec![45.0],
                    members: vec![10_000, 10_001],
                },
            ),
            (
                2,
                StoredList {
                    centroid: vec![81.0],
                    members: vec![10_002, 10_003],
                },
            ),
        ]
        .into(),
        next_list: 3,
        ..Store::default()
    };
    let (mut index, strategy) = index_with_neighbors(store, 2, 1);
    let points: [(u32, &[f32]); 2] = [(4, &[60.0][..]), (5, &[64.0][..])];
    let stats = require_send(index.insert_batch(&strategy, &DefaultContext, &points[..]))
        .await
        .unwrap();
    assert_eq!(
        stats,
        InsertStats {
            inserted: 2,
            splits: 2
        }
    );

    // Parent 1 splits into children 3 ([0]) and 4 ([40, 60]), and parent 2 into 5
    // ([62, 64]) and 6 ([100]). Point 60 is nearer child 5 than its own child 4.
    let store = strategy.store();
    assert_eq!(store.filled_lists, [1, 2, 3, 4, 5, 6]);
    assert_eq!(store.lists.len(), 4);
    assert_eq!(store.lists[&3].members, [10_000]);
    assert_eq!(store.lists[&4].members, [10_001]);
    assert_eq!(store.lists[&5].members, [10_002, 10_005, 10_004]);
    assert_eq!(store.lists[&6].members, [10_003]);
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
            .route(&reversed[..], |list, size| {
                emitted.push((list, size.get()));
            })
            .await
            .unwrap();
        assert_eq!(emitted, [(1, 4)]);
        {
            let (region, mut update) = require_send(build.fill([1].into_iter())).await.unwrap();
            let view = region.get_list(1).unwrap();
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
            KMeansSplit::new(
                TWO,
                NonZeroUsize::new(5).unwrap(),
                NonZeroUsize::new(2).unwrap(),
                NearestCentroid,
                FirstPoints,
            )
            .split(&region, &mut update, &[1], |_, _| {})
            .unwrap();
            assert_eq!(view.members, [10_000, 10_001, 10_006, 10_005]);
            assert_eq!(vectors.row(2), points[1].1);
            assert!(update.remove_members(1, &[10_000]).is_err());
            assert!(update.append_members(1, &[10_000]).is_err());
        }
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
        let parent = region.get_list(1).unwrap();
        let neighbor = region.get_list(2).unwrap();
        assert_eq!(parent.members, [10_000, 10_001, 10_005]);
        assert_eq!(neighbor.members, [10_002, 10_003, 10_006]);
        assert!(update.remove_members(3, &[]).is_err());
        assert!(update.append_members(3, &[]).is_err());
        assert!(update.retire(3).is_err());

        update.remove_members(2, &[10_006]).unwrap();
        update.append_members(2, &[10_005]).unwrap();
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

    {
        let (region, _) = build.fill([2].into_iter()).await.unwrap();
        assert_eq!(
            region.get_list(2).unwrap().members,
            [10_002, 10_003, 10_005]
        );
        assert_eq!(region.get_vectors(2).unwrap().row(2), points[0].1);
    }

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
async fn finished_updates_are_visible_to_later_fills_and_lookups() {
    let strategy = Strategy::new(seeded());
    let mut build: Build<'_, f32> = strategy.build_accessor(&DefaultContext).unwrap();
    let child = {
        let (_, mut update) = build.fill([1].into_iter()).await.unwrap();
        let child = update.create(&[1.0, 0.0], &[10_000, 10_001]).unwrap();
        update.retire(1).unwrap();
        require_send(update.finish()).await.unwrap();
        child
    };
    assert_eq!(child, 4);

    // Lookups can start from the created list and never return the retired one.
    let all = NonZeroUsize::new(usize::MAX).unwrap();
    let mut found = Vec::new();
    require_send(
        build
            .nearest()
            .find(child, all, |_| true, |id, _| found.push(id)),
    )
    .await
    .unwrap();
    assert_eq!(found, [4, 2, 3]);
    assert!(
        build
            .nearest()
            .find(1, all, |_| true, |_, _| {})
            .await
            .is_err()
    );

    // Fills include the created list, which later updates can edit, but not the retired one.
    assert!(build.fill([1].into_iter()).await.is_err());
    {
        let (region, mut update) = build.fill([child, 2].into_iter()).await.unwrap();
        let view = region.get_list(child).unwrap();
        assert_eq!(view.centroid, [1.0, 0.0]);
        assert_eq!(view.members, [10_000, 10_001]);
        assert_eq!(region.get_vectors(child).unwrap().row(1), [2.0, 0.0]);
        update.remove_members(child, &[10_001]).unwrap();
        update.append_members(2, &[10_001]).unwrap();
        require_send(update.finish()).await.unwrap();
    }

    require_send(build.finish()).await.unwrap();
    let store = strategy.store();
    assert!(!store.lists.contains_key(&1));
    assert_eq!(store.lists[&4].centroid, [1.0, 0.0]);
    assert_eq!(store.lists[&4].members, [10_000]);
    assert_eq!(store.lists[&2].members, [10_002, 10_003, 10_001]);
    assert_eq!(store.update_finishes, 2);
}

#[tokio::test]
async fn whole_list_transfer_appends_then_retires_without_deleting_vectors() {
    let strategy = Strategy::new(seeded());
    let mut build: Build<'_, f32> = strategy.build_accessor(&DefaultContext).unwrap();
    {
        let (region, mut update) = build.fill([1, 2].into_iter()).await.unwrap();
        update
            .append_members(2, region.get_list(1).unwrap().members)
            .unwrap();
        update.retire(1).unwrap();
    }
    require_send(build.finish()).await.unwrap();
    let store = strategy.store();
    assert!(!store.lists.contains_key(&1));
    assert_eq!(store.lists[&2].members, [10_002, 10_003, 10_000, 10_001]);
    assert_eq!(store.lists[&2].centroid, [11.0, 0.0]);
    assert_eq!(store.lists[&3].members, [10_004]);
    assert_eq!(store.points.len(), 5);
    assert_eq!(store.finishes, 1);
}

#[tokio::test]
async fn failed_planning_or_publication_is_not_reported_as_success() {
    for failure in [
        "nearest",
        "fill",
        "create",
        "remove_members",
        "append_members",
        "retire",
        "finish",
    ] {
        let mut store = seeded();
        store.points.insert(10_001, vec![11.0, 0.0]);
        store.points.insert(10_002, vec![1.0, 0.0]);
        store.fail_nearest = failure == "nearest";
        store.fail_fill = failure == "fill";
        store.fail_create = failure == "create";
        store.fail_remove_members = failure == "remove_members";
        store.fail_append_members = failure == "append_members";
        store.fail_retire = failure == "retire";
        store.fail_finish = failure == "finish";
        let (mut index, strategy) = index_with_neighbors(store, 2, 2);
        let points: [(u32, &[f32]); 1] = [(5, &[1.0, 1.0][..])];
        let error = require_send(index.insert_batch(&strategy, &DefaultContext, &points[..]))
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("injected {failure} failure")),
            "{error}"
        );
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

#[test]
fn accumulator_empty_like_preserves_search_limit_not_capacity() {
    let mut best = NeighborPriorityQueue::auto_resizable_with_search_param_l(2);
    for id in 0..3 {
        best.push(id, id as f32);
    }
    assert!(best.capacity() > best.search_l());

    let empty = best.empty_like();
    assert_eq!(empty.search_l(), 2);
    assert_eq!(empty.capacity(), 2);
    assert_eq!(empty.size(), 0);
    assert_eq!(best.size(), 3);
}

#[test]
fn accumulator_merges_partitioned_top_k() {
    let candidates = [
        (0, 9.0),
        (1, 1.0),
        (2, 4.0),
        (3, 7.0),
        (4, 3.0),
        (5, 8.0),
        (6, 5.0),
        (7, 2.0),
    ];
    for k in [1, 3, 16] {
        let mut reference = NeighborPriorityQueue::new(k);
        reference.insert(Neighbor::new(99, -1.0));
        for &(id, distance) in &candidates {
            reference.insert(Neighbor::new(id, distance));
        }
        let expected: Vec<_> = reference.iter().map(Neighbor::as_tuple).collect();

        for chunk_size in [1, 3, candidates.len()] {
            for reverse in [false, true] {
                let mut best = NeighborPriorityQueue::new(k);
                best.push(99, -1.0);
                let mut partials: Vec<_> = candidates
                    .chunks(chunk_size)
                    .map(|chunk| {
                        let mut partial = best.empty_like();
                        assert_eq!(partial.search_l(), k);
                        assert_eq!(partial.size(), 0);
                        for &(id, distance) in chunk {
                            partial.push(id, distance);
                        }
                        assert!(partial.size() <= k);
                        partial
                    })
                    .collect();
                if reverse {
                    partials.reverse();
                }
                for partial in partials {
                    best.merge(partial);
                }
                let empty = best.empty_like();
                best.merge(empty);
                assert_eq!(
                    best.iter().map(Neighbor::as_tuple).collect::<Vec<_>>(),
                    expected,
                );
            }
        }
    }
}

#[test]
fn accumulator_preserves_infinite_distances_and_ignores_nan() {
    let mut best = NeighborPriorityQueue::new(3);
    let mut left = best.empty_like();
    left.push(1, f32::INFINITY);
    left.push(2, f32::NAN);
    let mut right = best.empty_like();
    right.push(3, 1.0);
    right.push(4, f32::INFINITY);

    best.merge(left);
    best.merge(right);
    assert_eq!(
        best.iter()
            .map(|neighbor| *neighbor.distance())
            .collect::<Vec<_>>(),
        [1.0, f32::INFINITY, f32::INFINITY],
    );
    assert_eq!(
        best.iter()
            .map(|neighbor| *neighbor.id())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([1, 3, 4]),
    );
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
        ("later_scan", "missing point 10001"),
    ] {
        let mut store = seeded();
        if case == "route" {
            store.fail_route = true;
        }
        if case == "scan" {
            store.points.remove(&10_003);
        }
        if case == "later_scan" {
            store.points.remove(&10_001);
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
        let nprobe = if case == "later_scan" {
            NonZeroUsize::new(3).unwrap()
        } else {
            ONE
        };
        let mut output = vec![Neighbor::new(42, -1.0)];
        let error = require_send(index.search(
            &strategy,
            &DefaultContext,
            query,
            NonZeroUsize::new(3).unwrap(),
            nprobe,
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
fn reassignment_moves_points_only_along_links_and_keeps_empty_lists() {
    let sources = BTreeMap::from([
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
        (
            10,
            Source {
                centroid: vec![-10.0],
                members: vec![0],
                vectors: matrix(&[[-10.0]]),
            },
        ),
        (
            11,
            Source {
                centroid: vec![10.0],
                members: vec![1],
                vectors: matrix(&[[10.0]]),
            },
        ),
    ]);
    let region = Region::new(&sources);
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    let links = support::links(&[(10, 2), (10, 3), (10, 4), (11, 2), (11, 3), (11, 4)]);
    Linked::new(NearestCentroid)
        .reassign(&region, &mut update, &links)
        .unwrap();

    // Point 2 is nearest list 3, which it is not linked to, so it moves to list 11. Point 1
    // ties empty list 4 and stays; list 4 is retained without edits.
    assert!(created.is_empty());
    let changes: BTreeMap<_, _> = memberships.into_changes().collect();
    assert_eq!(changes.len(), 2);
    assert!(matches!(
        &changes[&2],
        Membership::Delta { removed, appended }
            if *removed == HashSet::from([2]) && appended.is_empty()
    ));
    assert!(matches!(
        &changes[&11],
        Membership::Delta { removed, appended } if removed.is_empty() && appended == &[2]
    ));
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
        .route(&points[..], |list, size| {
            emitted.push((list, size.get()));
        })
        .await
        .unwrap();
    assert_eq!(emitted, [(1, 3)]);

    {
        let (region, mut update) = build.fill([1, 2].into_iter()).await.unwrap();
        assert!(
            KMeansSplit::new(
                TWO,
                NonZeroUsize::new(5).unwrap(),
                NonZeroUsize::new(2).unwrap(),
                NearestCentroid,
                FirstPoints
            )
            .split(&region, &mut update, &[1, 99], |_, _| {})
            .is_err()
        );
        assert!(
            Linked::new(NearestCentroid)
                .reassign(&region, &mut update, &support::links(&[(1, 99)]))
                .is_err()
        );
        assert!(update.created.is_empty());
        let changes: Vec<_> = update.memberships.clone().into_changes().collect();
        assert!(
            matches!(changes.as_slice(), [(1, Membership::Delta { removed, appended })]
                if removed.is_empty() && appended == &[10_005])
        );
        KMeansSplit::new(
            TWO,
            NonZeroUsize::new(5).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            NearestCentroid,
            FirstPoints,
        )
        .split(&region, &mut update, &[1], |_, _| {})
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
fn parent_points_stay_within_their_own_children() {
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
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    let mut children = Vec::new();
    KMeansSplit::new(
        TWO,
        NonZeroUsize::new(10).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        NearestCentroid,
        FirstPoints,
    )
    .split(&region, &mut update, &[1, 2], |parent, child| {
        children.push((parent, child))
    })
    .unwrap();
    assert_eq!(children, [(1, 3), (1, 4), (2, 5), (2, 6)]);
    let members = |children: &[StoredList]| {
        let mut members: Vec<_> = children
            .iter()
            .flat_map(|child| child.members.iter().copied())
            .collect();
        members.sort_unstable();
        members
    };
    assert_eq!(members(&update.created[..2]), (0..5).collect::<Vec<_>>());
    assert_eq!(
        members(&update.created[2..]),
        (100..104).collect::<Vec<_>>()
    );
}

#[test]
fn assignment_is_delegated_to_the_assigner() {
    /// Records the candidates of every list and sends each point to the first one.
    struct First<'a>(&'a RefCell<Vec<Vec<usize>>>);

    impl<'v, V: AsMatRef<'v, f32>> Assign<V> for First<'_> {
        fn assign<I: Copy>(
            &self,
            members: &[I],
            points: V,
            candidates: &[(usize, &[f32])],
            mut emit: impl FnMut(I, usize),
        ) {
            assert_eq!(members.len(), points.as_view().nrows());
            let dsts = candidates.iter().map(|&(dst, _)| dst).collect();
            self.0.borrow_mut().push(dsts);
            for &id in members {
                emit(id, candidates[0].0);
            }
        }
    }

    let sources = BTreeMap::from([
        (
            1,
            Source {
                centroid: vec![0.0],
                members: vec![0, 1],
                vectors: matrix(&[[-1.0], [1.0]]),
            },
        ),
        (
            2,
            Source {
                centroid: vec![5.0],
                members: vec![2],
                vectors: matrix(&[[-1.0]]),
            },
        ),
    ]);
    let region = Region::new(&sources);
    let seen = RefCell::new(Vec::new());

    // Splitting offers a parent's own children.
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    KMeansSplit::new(TWO, ONE, ONE, First(&seen), FirstPoints)
        .split(&region, &mut update, &[1], |_, _| {})
        .unwrap();
    assert_eq!(seen.take(), [vec![0, 1]]);
    assert_eq!(created[0].members, [0, 1]);
    assert!(created[1].members.is_empty());

    // Reassignment offers a list's own centroid, then those of its links.
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    Linked::new(First(&seen))
        .reassign(&region, &mut update, &support::links(&[(1, 2)]))
        .unwrap();
    assert_eq!(seen.take(), [vec![0, 1], vec![1, 0]]);
    // Every point is sent to its own list, so nothing changes.
    assert_eq!(memberships.into_changes().count(), 0);
}

#[test]
fn k_means_fits_and_assigns_k_children_per_parent() {
    let sources = BTreeMap::from([(
        1,
        Source {
            centroid: vec![0.0],
            members: (0..6).collect(),
            vectors: matrix(&[[-10.0], [0.0], [10.0], [-9.0], [1.0], [11.0]]),
        },
    )]);
    let region = Region::new(&sources);
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    let mut children = Vec::new();
    let k = NonZeroUsize::new(3).unwrap();
    KMeansSplit::new(k, ONE, ONE, NearestCentroid, FirstPoints)
        .split(&region, &mut update, &[1], |parent, child| {
            children.push((parent, child))
        })
        .unwrap();

    assert_eq!(children, [(1, 2), (1, 3), (1, 4)]);
    let centroids: Vec<_> = created.iter().map(|child| child.centroid[0]).collect();
    assert_eq!(centroids, [-9.5, 0.5, 10.5]);
    let members: Vec<_> = created.iter().map(|child| child.members.clone()).collect();
    assert_eq!(members, [vec![0, 3], vec![1, 4], vec![2, 5]]);
    let changes: BTreeMap<_, _> = memberships.into_changes().collect();
    assert_eq!(changes.len(), 1);
    assert!(matches!(changes[&1], Membership::Retire));
}

#[test]
fn k_means_needs_k_points_per_parent() {
    let sources = BTreeMap::from([(
        1,
        Source {
            centroid: vec![0.0],
            members: vec![0, 1],
            vectors: matrix(&[[-1.0], [1.0]]),
        },
    )]);
    let region = Region::new(&sources);
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    let k = NonZeroUsize::new(3).unwrap();
    let error = KMeansSplit::new(k, ONE, ONE, NearestCentroid, FirstPoints)
        .split(&region, &mut update, &[1], |_, _| {})
        .unwrap_err();

    assert!(matches!(
        error.downcast_ref::<NotEnoughPoints>(),
        Some(NotEnoughPoints {
            actual: 2,
            required: 3
        })
    ));
    assert!(created.is_empty());
    assert_eq!(memberships.into_changes().count(), 0);
}

#[test]
fn two_means_fits_from_its_seeder() {
    /// Seeds with the parent's points in reverse order.
    struct Reversed;

    impl Seed for Reversed {
        type Error = ANNError;

        fn seed(
            &mut self,
            points: rowmajor::Ref<'_, f32>,
            mut centers: rowmajor::Mut<'_, f32>,
        ) -> ANNResult<()> {
            let rows = (0..centers.nrows()).rev();
            for (row, center) in rows.zip(centers.rows_mut()) {
                center.copy_from_slice(points.row(row));
            }
            Ok(())
        }
    }

    let sources = BTreeMap::from([(
        1,
        Source {
            centroid: vec![0.0],
            members: vec![0, 1],
            vectors: matrix(&[[-1.0], [1.0]]),
        },
    )]);
    let region = Region::new(&sources);
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    KMeansSplit::new(TWO, ONE, ONE, NearestCentroid, Reversed)
        .split(&region, &mut update, &[1], |_, _| {})
        .unwrap();

    // `FirstPoints` would put -1 first; reversed seeds keep that order through Lloyd's.
    assert_eq!(created[0].centroid, [1.0]);
    assert_eq!(created[0].members, [1]);
    assert_eq!(created[1].centroid, [-1.0]);
    assert_eq!(created[1].members, [0]);
}

#[test]
fn two_means_reports_seeding_failures_before_writing() {
    struct Fails;

    impl Seed for Fails {
        type Error = ANNError;

        fn seed(&mut self, _: rowmajor::Ref<'_, f32>, _: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
            Err(ANNError::message("injected seeding failure"))
        }
    }

    let sources = BTreeMap::from([(
        1,
        Source {
            centroid: vec![0.0],
            members: vec![0, 1],
            vectors: matrix(&[[-1.0], [1.0]]),
        },
    )]);
    let region = Region::new(&sources);
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    let error = KMeansSplit::new(TWO, ONE, ONE, NearestCentroid, Fails)
        .split(&region, &mut update, &[1], |_, _| {})
        .unwrap_err()
        .to_string();

    assert!(error.contains("injected seeding failure"), "{error}");
    assert!(error.contains("while seeding list 1"), "{error}");
    assert!(created.is_empty());
    assert_eq!(memberships.into_changes().count(), 0);
}

#[test]
fn two_means_resets_empty_children_to_zero_and_prefers_the_first_child_on_ties() {
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
        let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
        KMeansSplit::new(TWO, ONE, ONE, NearestCentroid, FirstPoints)
            .split(&region, &mut update, &[1], |_, _| {})
            .unwrap();
        assert_eq!(update.created.len(), 2);
        assert_eq!(update.created[0].centroid, [value]);
        assert_eq!(update.created[1].centroid, [0.0]);
        assert_eq!(update.created[0].members, [0, 1, 2]);
        assert!(update.created[1].members.is_empty());
    }
}

#[test]
fn reassignment_breaks_ties_by_own_list_then_link_order() {
    let sources = BTreeMap::from([
        (
            1,
            Source {
                centroid: vec![5.0],
                members: vec![0],
                vectors: matrix(&[[-1.0]]),
            },
        ),
        (
            2,
            Source {
                centroid: vec![-1.0],
                members: vec![2],
                vectors: matrix(&[[2.0]]),
            },
        ),
        (
            3,
            Source {
                centroid: vec![-1.0],
                members: vec![3],
                vectors: matrix(&[[2.0]]),
            },
        ),
    ]);
    let region = Region::new(&sources);
    for (pairs, first) in [([(1, 2), (1, 3)], 2), ([(1, 3), (1, 2)], 3)] {
        let mut memberships = PendingLists::default();
        let mut created = Vec::new();
        let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
        Linked::new(NearestCentroid)
            .reassign(&region, &mut update, &support::links(&pairs))
            .unwrap();

        // Point 0 ties lists 2 and 3 and joins the one linked first. Points 2 and 3 tie
        // their own list and list 1, and stay.
        let changes: BTreeMap<_, _> = memberships.into_changes().collect();
        assert_eq!(changes.len(), 2);
        assert!(matches!(
            &changes[&1],
            Membership::Delta { removed, appended }
                if *removed == HashSet::from([0]) && appended.is_empty()
        ));
        assert!(matches!(
            &changes[&first],
            Membership::Delta { removed, appended } if removed.is_empty() && appended == &[0]
        ));
    }
}

#[test]
fn two_means_reuses_splitter_across_regions() {
    let sources = BTreeMap::from([
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
    ]);
    let region = Region::new(&sources);
    let mut full_memberships = PendingLists::default();
    let mut full_created = Vec::new();
    let mut full = support::Update::new(&sources, 1, &mut full_memberships, &mut full_created);
    let mut splitter = KMeansSplit::new(TWO, ONE, ONE, NearestCentroid, FirstPoints);
    splitter
        .split(&region, &mut full, &[1, 2], |_, _| {})
        .unwrap();
    assert_eq!(
        full.created
            .iter()
            .map(|list| list.centroid[0])
            .collect::<Vec<_>>(),
        [-10.0, 10.0, 100.0, 110.0]
    );
    let members: Vec<_> = full
        .created
        .iter()
        .map(|list| list.members.clone())
        .collect();
    assert_eq!(members, [vec![0], vec![1], vec![2], vec![3]]);
    let mut repeated_memberships = PendingLists::default();
    let mut repeated_created = Vec::new();
    let mut repeated = support::Update::new(
        &sources,
        1,
        &mut repeated_memberships,
        &mut repeated_created,
    );
    splitter
        .split(&region, &mut repeated, &[1, 2], |_, _| {})
        .unwrap();
    for (first, repeated) in full.created.iter().zip(repeated.created.iter()) {
        assert_eq!(first.centroid, repeated.centroid);
        assert_eq!(first.members, repeated.members);
    }
    let mut smaller_memberships = PendingLists::default();
    let mut smaller_created = Vec::new();
    let mut smaller =
        support::Update::new(&sources, 1, &mut smaller_memberships, &mut smaller_created);
    splitter
        .split(&region, &mut smaller, &[1], |_, _| {})
        .unwrap();
    assert_eq!(smaller.created.len(), 2);
    assert_eq!(smaller.created[0].members, [0]);
    assert_eq!(smaller.created[1].members, [1]);
    let changes: BTreeMap<_, _> = std::mem::take(full.memberships).into_changes().collect();
    assert_eq!(changes.len(), 2);
    for parent in [1, 2] {
        assert!(matches!(changes[&parent], Membership::Retire));
    }
}

#[test]
fn empty_split_records_no_updates() {
    let sources = BTreeMap::new();
    let region = Region::new(&sources);
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    let error = KMeansSplit::new(TWO, ONE, ONE, NearestCentroid, FirstPoints)
        .split(&region, &mut update, &[], |_, _| {})
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<SplitError<u32>>(),
        Some(SplitError::EmptyParents)
    ));
    assert!(created.is_empty());
    assert_eq!(memberships.into_changes().count(), 0);
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
    let view = region.get_list(1).unwrap();
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
fn two_means_handles_borrowed_batches() {
    struct Batch<'a> {
        vectors: rowmajor::Ref<'a, f32>,
        reads: &'a Cell<usize>,
    }

    struct BatchRef<'a> {
        vectors: rowmajor::Ref<'a, f32>,
        reads: &'a Cell<usize>,
    }

    impl<'r> Reborrow<'r> for Batch<'_> {
        type Target = BatchRef<'r>;

        fn reborrow(&'r self) -> Self::Target {
            BatchRef {
                vectors: self.vectors.reborrow(),
                reads: self.reads,
            }
        }
    }

    impl<'a> AsMatRef<'a, f32> for BatchRef<'a> {
        fn as_view(self) -> rowmajor::Ref<'a, f32> {
            self.reads.set(self.reads.get() + 1);
            self.vectors
        }
    }

    struct BatchRegion<'a> {
        inner: Region<'a>,
        reads: Cell<usize>,
    }

    impl View<u32, u32> for BatchRegion<'_> {
        type VectorsRef<'a> = BatchRef<'a>;
        type Vectors<'a>
            = Batch<'a>
        where
            Self: 'a;

        fn get_list(&self, list: u32) -> Option<List<'_, u32, u32>> {
            self.inner.get_list(list)
        }

        fn get_vectors(&self, list: u32) -> Option<Self::Vectors<'_>> {
            self.inner.get_vectors(list).map(|vectors| Batch {
                vectors,
                reads: &self.reads,
            })
        }
    }

    let sources = BTreeMap::from([
        (
            1,
            Source {
                centroid: vec![0.0],
                members: vec![0, 1],
                vectors: matrix(&[[-1.0], [1.0]]),
            },
        ),
        (
            2,
            Source {
                centroid: vec![10.0],
                members: vec![2],
                vectors: matrix(&[[10.0]]),
            },
        ),
    ]);
    let region = BatchRegion {
        inner: Region::new(&sources),
        reads: Cell::new(0),
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
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    KMeansSplit::new(TWO, ONE, ONE, NearestCentroid, FirstPoints)
        .split(&region, &mut update, &[1], |_, _| {})
        .unwrap();
    // Fitting and assigning the parent each read its vectors once.
    assert_eq!(region.reads.get(), 2);
    assert_eq!(created.len(), 2);
    assert_eq!(created[0].centroid, [-1.0]);
    assert_eq!(created[0].members, [0]);
    assert_eq!(created[1].centroid, [1.0]);
    assert_eq!(created[1].members, [1]);
    let changes: BTreeMap<_, _> = memberships.into_changes().collect();
    assert!(matches!(changes[&1], Membership::Retire));
    assert_eq!(changes.len(), 1);

    // Reassignment reads each linked list's vectors once.
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    Linked::new(NearestCentroid)
        .reassign(&region, &mut update, &support::links(&[(1, 2)]))
        .unwrap();
    assert_eq!(region.reads.get(), 4);
    assert_eq!(memberships.into_changes().count(), 0);
}

#[test]
fn split_accepts_native_batches_without_matrix_access() {
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

    impl View<u32, u32> for NativeRegion<'_> {
        type VectorsRef<'a> = &'a [[u8; 1]];
        type Vectors<'a>
            = NativeBatch<'a>
        where
            Self: 'a;

        fn get_list(&self, list: u32) -> Option<List<'_, u32, u32>> {
            self.inner.get_list(list)
        }

        fn get_vectors(&self, list: u32) -> Option<Self::Vectors<'_>> {
            self.inner.get_list(list).map(|_| NativeBatch(self.vectors))
        }
    }

    struct NativeSplit;

    impl<R> Split<u32, u32, R> for NativeSplit
    where
        R: for<'a> View<u32, u32, VectorsRef<'a> = &'a [[u8; 1]]>,
    {
        type Error = ANNError;

        fn split<U, F>(
            &mut self,
            region: &R,
            update: &mut U,
            parents: &[u32],
            mut emit: F,
        ) -> ANNResult<()>
        where
            U: Update<u32, u32>,
            F: FnMut(u32, u32),
        {
            let parent = parents[0];
            let view = region.get_list(parent).unwrap();
            let vectors = region.get_vectors(parent).unwrap();
            for (point, member) in vectors.reborrow().iter().zip(view.members) {
                let child = update.create(&[f32::from(point[0])], std::slice::from_ref(member))?;
                emit(parent, child);
            }
            update.retire(parent)
        }
    }

    let mut splitter = NativeSplit;
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
        let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
        let mut children = Vec::new();
        splitter
            .split(&region, &mut update, &[1], |parent, child| {
                children.push((parent, child))
            })
            .unwrap();
        assert_eq!(children, [(1, 2), (1, 3)]);
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
    let sources = BTreeMap::from([
        (
            1,
            Source {
                centroid: vec![0.0, 0.0],
                members: vec![0, 1, 2, 3],
                vectors: matrix(&[[0.0, 2.0], [2.0, 4.0], [6.0, 6.0], [10.0, 12.0]]),
            },
        ),
        (
            2,
            Source {
                centroid: vec![100.0, 100.0],
                members: vec![4, 5, 6],
                vectors: matrix(&[[100.0, 102.0], [110.0, 112.0], [120.0, 120.0]]),
            },
        ),
    ]);
    let region = Region::new(&sources);
    for iterations in [1, 5] {
        let mut memberships = PendingLists::default();
        let mut created = Vec::new();
        let mut update = support::Update::new(&sources, 2, &mut memberships, &mut created);
        KMeansSplit::new(
            TWO,
            NonZeroUsize::new(iterations).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            NearestCentroid,
            FirstPoints,
        )
        .split(&region, &mut update, &[1, 2], |_, _| {})
        .unwrap();
        assert_eq!(created.len(), 4);
        for (source, actual) in sources.values().zip(created.chunks_exact(2)) {
            let points = source.vectors.as_view();
            let mut expected = points.subview(0..2).unwrap().to_rowmajor_owned();
            let _ = diskann_quantization::algorithms::kmeans::lloyds::lloyds(
                points,
                expected.as_view_mut(),
                iterations,
            );
            for (actual, expected) in actual.iter().zip(expected.rows()) {
                assert_eq!(actual.centroid, expected);
            }
        }
    }
}

#[test]
fn two_means_reports_insufficient_points_before_writing() {
    for points in [matrix::<2>(&[]), matrix(&[[0.0, 2.0]])] {
        let expected = points.nrows();
        let sources = BTreeMap::from([(
            1,
            Source {
                centroid: vec![0.0; points.ncols()],
                members: (0..).take(points.nrows()).collect(),
                vectors: points,
            },
        )]);
        let region = Region::new(&sources);
        let mut memberships = PendingLists::default();
        let mut created = Vec::new();
        let mut update = support::Update::new(&sources, 2, &mut memberships, &mut created);
        let error = KMeansSplit::new(
            TWO,
            ONE,
            NonZeroUsize::new(2).unwrap(),
            NearestCentroid,
            FirstPoints,
        )
        .split(&region, &mut update, &[1], |_, _| {})
        .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<NotEnoughPoints>(),
            Some(NotEnoughPoints { actual, required: 2 }) if *actual == expected
        ));
        let message = error.to_string();
        assert!(message.contains("while seeding list 1"), "{message}");
        assert!(created.is_empty());
        assert_eq!(memberships.into_changes().count(), 0);
    }
}

#[test]
fn two_means_missing_vectors_preserves_typed_error_without_updates() {
    let sources = BTreeMap::from([(
        1,
        Source {
            centroid: vec![0.0],
            members: vec![0, 1],
            vectors: matrix(&[[-1.0], [1.0]]),
        },
    )]);
    let region = Region::new(&sources);
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    let error = KMeansSplit::new(TWO, ONE, ONE, NearestCentroid, FirstPoints)
        .split(&region, &mut update, &[1, 99], |_, _| {})
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<SplitError<u32>>(),
        Some(SplitError::UnableToRetrieve(99))
    ));
    assert!(created.is_empty());
    assert_eq!(memberships.into_changes().count(), 0);
}

#[test]
fn reassignment_reports_missing_lists() {
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
    let region = Region::new(&sources);
    for pairs in [vec![(1, 99)], vec![(1, 2), (1, 99)]] {
        let mut memberships = PendingLists::default();
        let mut created = Vec::new();
        let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
        let error = Linked::new(NearestCentroid)
            .reassign(&region, &mut update, &support::links(&pairs))
            .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<SplitError<u32>>(),
            Some(SplitError::UnableToRetrieve(99))
        ));
        // List 1, the first one reassigned, links to the missing list, so nothing is written.
        assert!(created.is_empty());
        assert_eq!(memberships.into_changes().count(), 0);
    }
}

#[test]
fn reassignment_keeps_points_home_when_all_distances_overflow() {
    let sources = BTreeMap::from([
        (
            1,
            Source {
                centroid: vec![0.0],
                members: vec![0, 1],
                vectors: matrix(&[[-1.0], [1.0]]),
            },
        ),
        (
            2,
            Source {
                centroid: vec![f32::MAX],
                members: vec![2],
                vectors: matrix(&[[-f32::MAX]]),
            },
        ),
    ]);
    let region = Region::new(&sources);
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    Linked::new(NearestCentroid)
        .reassign(&region, &mut update, &support::links(&[(1, 2)]))
        .unwrap();

    // Point 2 is infinitely far from both centroids and stays in its own list.
    assert_eq!(memberships.into_changes().count(), 0);
}
