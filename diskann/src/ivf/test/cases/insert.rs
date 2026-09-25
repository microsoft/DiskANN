/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Insert behavior of [`crate::ivf::IvfIndex::insert`].
//!
//! Inserting a vector must (a) store it in the provider, (b) append it to its nearest list,
//! and (c) make it findable by a subsequent search that probes that list.

use crate::{
    ivf::test::{
        cases::{DIM, METRIC, N_LISTS, grid_index},
        harness::IvfOracleRun,
        provider::{Context, Strategy, assign_nearest_for_test},
    },
    neighbor::{BackInserter, Neighbor},
    test::tokio::current_thread_runtime,
};
use std::{num::NonZeroUsize, sync::Arc};

/// After inserting a point, a search probing all lists returns it as the exact nearest
/// neighbor of itself (distance 0).
#[test]
fn inserted_point_is_findable() {
    let index = grid_index();
    let strategy = Strategy::new(DIM, METRIC);
    let new_point = vec![3.5f32, 3.5];
    let new_id = index.provider().len() as u32;

    current_thread_runtime()
        .block_on(index.insert(&strategy, &Context::new(), &new_id, new_point.as_slice()))
        .unwrap();

    assert_eq!(index.provider().len(), (new_id + 1) as usize);

    let run = IvfOracleRun::run_sync(&index, &strategy, &new_point, 1, N_LISTS).unwrap();
    assert_eq!(run.top_k.len(), 1);
    let (found_id, dist) = run.top_k[0];
    assert_eq!(found_id, new_id);
    assert!(dist.abs() <= f32::EPSILON, "self-distance must be zero");
}

/// The inserted point lands in the list whose centroid is nearest to it.
#[test]
fn inserted_point_goes_to_nearest_list() {
    let index = grid_index();
    let strategy = Strategy::new(DIM, METRIC);
    let new_point = vec![6.0f32, 6.0];
    let new_id = index.provider().len() as u32;

    let expected_list = assign_nearest_for_test(index.provider().centroids(), &new_point, METRIC);

    current_thread_runtime()
        .block_on(index.insert(&strategy, &Context::new(), &new_id, new_point.as_slice()))
        .unwrap();

    let members = index.provider().members(expected_list);
    assert!(
        members.contains(&new_id),
        "inserted id {new_id} should be in list {expected_list}, members={members:?}",
    );
}

/// Inserting then probing only the assigned list still finds the point.
#[test]
fn inserted_point_found_with_single_probe() {
    let index = grid_index();
    let strategy = Strategy::new(DIM, METRIC);
    let new_point = vec![1.0f32, 1.0];
    let new_id = index.provider().len() as u32;

    current_thread_runtime()
        .block_on(index.insert(&strategy, &Context::new(), &new_id, new_point.as_slice()))
        .unwrap();

    let context = Context::new();
    let mut buf = vec![Neighbor::<u32>::default(); 1];
    let stats = current_thread_runtime()
        .block_on(index.knn_search(
            NonZeroUsize::new(1).unwrap(),
            1,
            &strategy,
            &context,
            new_point.as_slice(),
            &mut BackInserter::new(buf.as_mut_slice()),
        ))
        .unwrap();

    assert_eq!(stats.result_count, 1);
    assert_eq!(buf[0].id, new_id);
}

/// Many `insert` calls racing on a shared `Arc<IvfIndex>` (multi-threaded runtime) must all
/// take effect: every point gets a unique id, lands in its deterministically-assigned list,
/// and is findable at distance zero afterward.
///
/// This exercises the insert path's interior mutability under real contention -- the atomic
/// id allocator in `set_element` and the per-list `RwLock::write` in `AppendAccessor::append`
/// (with several points deliberately routed to the same list to force write contention).
#[test]
fn concurrent_inserts_all_land_and_are_findable() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("multi-thread runtime");

    let index = Arc::new(grid_index());
    let strategy = Arc::new(Strategy::new(DIM, METRIC));
    let base_len = index.provider().len();

    // Distinct off-grid points spread across all four quadrants. Several share a quadrant so
    // their appends contend on the same list's `RwLock`.
    let new_points: Vec<Vec<f32>> = vec![
        vec![0.25, 0.25], // lower-left
        vec![1.75, 0.50], // lower-left
        vec![0.50, 2.25], // lower-left
        vec![1.25, 5.75], // upper-left
        vec![0.75, 6.25], // upper-left
        vec![6.25, 1.25], // lower-right
        vec![5.75, 5.75], // upper-right
        vec![6.50, 6.50], // upper-right
        vec![5.25, 6.75], // upper-right
        vec![6.75, 5.25], // upper-right
    ];
    let n = new_points.len();

    // Deterministic routing: how many of the new points each list should receive.
    let mut expected_per_list = [0usize; N_LISTS];
    for point in &new_points {
        let list = assign_nearest_for_test(index.provider().centroids(), point, METRIC);
        expected_per_list[list as usize] += 1;
    }

    runtime.block_on({
        let index = Arc::clone(&index);
        let strategy = Arc::clone(&strategy);
        let points = new_points.clone();
        async move {
            let mut set = tokio::task::JoinSet::new();
            for (i, point) in points.into_iter().enumerate() {
                let index = Arc::clone(&index);
                let strategy = Arc::clone(&strategy);
                set.spawn(async move {
                    let external_id = (base_len + i) as u32;
                    index
                        .insert(&*strategy, &Context::new(), &external_id, point.as_slice())
                        .await
                        .expect("concurrent insert failed");
                });
            }
            while let Some(joined) = set.join_next().await {
                joined.expect("insert task panicked");
            }
        }
    });

    // (1) Every insert allocated a unique, sequential id and committed to some list: the set
    // of newly added ids is exactly `base_len .. base_len + n`, with no losses or duplicates.
    assert_eq!(index.provider().len(), base_len + n);
    let mut new_ids: Vec<u32> = (0..N_LISTS as u32)
        .flat_map(|list| index.provider().members(list))
        .filter(|id| (*id as usize) >= base_len)
        .collect();
    new_ids.sort_unstable();
    let expected_ids: Vec<u32> = (base_len as u32..(base_len + n) as u32).collect();
    assert_eq!(
        new_ids, expected_ids,
        "concurrent inserts must allocate every id exactly once with no loss",
    );

    // (2) Concurrent routing matched the deterministic per-list assignment.
    for (list, &expected) in expected_per_list.iter().enumerate() {
        let new_in_list = index
            .provider()
            .members(list as u32)
            .into_iter()
            .filter(|id| (*id as usize) >= base_len)
            .count();
        assert_eq!(
            new_in_list, expected,
            "list {list} received the wrong number of concurrent inserts",
        );
    }

    // (3) Each inserted point is findable at distance zero -- proving its vector bytes were
    // written intact (no torn or lost writes under contention).
    for point in &new_points {
        let run = IvfOracleRun::run_sync(&index, &strategy, point, 1, N_LISTS).unwrap();
        assert_eq!(run.top_k.len(), 1);
        assert!(
            run.top_k[0].1.abs() <= f32::EPSILON,
            "inserted point {point:?} must be findable at distance zero",
        );
    }
}
