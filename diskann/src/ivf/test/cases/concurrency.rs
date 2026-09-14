/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Concurrency behavior of the fine-scan accessor.
//!
//! [`ScanAccessor`](crate::ivf::test::provider::ScanAccessor) exposes two paths --
//! [`ScanMode::Sequential`] and [`ScanMode::Concurrent`] (one spawned task per probed list).
//! These tests assert the two paths produce identical results, that the concurrent path
//! actually spawned tasks, and that many searches can run concurrently over a shared index.

use std::sync::Arc;

use crate::{
    ivf::test::{
        cases::{METRIC, N_LISTS, grid_index, queries},
        harness::{IvfOracleRun, assert_same_distances},
        provider::{Context, ScanMode, Strategy},
    },
    test::tokio::current_thread_runtime,
};

/// The sequential and concurrent scan paths must return the same top-`k` for every query.
#[test]
fn sequential_and_concurrent_agree() {
    let index = grid_index();
    let dim = index.provider().dim();
    let sequential = Strategy::new(dim, METRIC).with_mode(ScanMode::Sequential);
    let concurrent = Strategy::new(dim, METRIC).with_mode(ScanMode::Concurrent);

    for query in queries() {
        for k in [1usize, 8] {
            let seq = IvfOracleRun::run_sync(&index, &sequential, &query, k, N_LISTS).unwrap();
            let con = IvfOracleRun::run_sync(&index, &concurrent, &query, k, N_LISTS).unwrap();

            assert_same_distances(&con.top_k, &seq.ground_truth);
            assert_same_distances(&con.top_k, &seq.top_k);
            assert_eq!(con.stats.cmps, seq.stats.cmps);
        }
    }
}

/// The concurrent path must route its list scans through `wrap_spawn`, so the context
/// observes one spawn per probed list.
#[tokio::test]
async fn concurrent_path_spawns_per_list() {
    let index = grid_index();
    let dim = index.provider().dim();
    let strategy = Strategy::new(dim, METRIC).with_mode(ScanMode::Concurrent);

    let context = Context::new();
    let mut buf = vec![crate::neighbor::Neighbor::<u32>::default(); 4];

    let stats = index
        .knn_search(
            std::num::NonZeroUsize::new(4).unwrap(),
            N_LISTS,
            &strategy,
            &context,
            [3.5f32, 3.5].as_slice(),
            &mut crate::neighbor::BackInserter::new(buf.as_mut_slice()),
        )
        .await
        .unwrap();

    assert_eq!(
        context.spawns(),
        N_LISTS,
        "concurrent scan should spawn one task per probed list",
    );
    assert_eq!(stats.cmps as usize, index.provider().len());
}

/// The sequential path performs no spawns.
#[tokio::test]
async fn sequential_path_does_not_spawn() {
    let index = grid_index();
    let dim = index.provider().dim();
    let strategy = Strategy::new(dim, METRIC).with_mode(ScanMode::Sequential);

    let context = Context::new();
    let mut buf = vec![crate::neighbor::Neighbor::<u32>::default(); 4];

    index
        .knn_search(
            std::num::NonZeroUsize::new(4).unwrap(),
            N_LISTS,
            &strategy,
            &context,
            [3.5f32, 3.5].as_slice(),
            &mut crate::neighbor::BackInserter::new(buf.as_mut_slice()),
        )
        .await
        .unwrap();

    assert_eq!(context.spawns(), 0);
}

/// Many concurrent searches over a shared index (on a multi-threaded runtime) must each
/// return the correct restricted top-`k`.
#[test]
fn many_concurrent_searches_share_one_index() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("multi-thread runtime");

    let index = Arc::new(grid_index());
    let dim = index.provider().dim();
    let strategy = Arc::new(Strategy::new(dim, METRIC).with_mode(ScanMode::Concurrent));

    runtime.block_on(async move {
        let mut set = tokio::task::JoinSet::new();
        for query in queries() {
            let index = Arc::clone(&index);
            let strategy = Arc::clone(&strategy);
            set.spawn(async move {
                let run = IvfOracleRun::run(&index, &strategy, &query, 8, N_LISTS - 1)
                    .await
                    .unwrap();
                assert_same_distances(&run.top_k, &run.ground_truth);
            });
        }
        while let Some(joined) = set.join_next().await {
            joined.expect("search task panicked");
        }
    });
}

// Bring the single-thread runtime helper into scope for documentation/intra-doc links.
#[allow(unused_imports)]
use current_thread_runtime as _current_thread_runtime;
