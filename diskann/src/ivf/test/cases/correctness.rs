/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Correctness of [`crate::ivf::IvfIndex::knn_search`].
//!
//! The central invariant is that an IVF search probing `nprobe` lists returns exactly the
//! brute-force top-`k` over the union of those lists' members. The harness computes that
//! restricted oracle directly, so each case can assert strict equality.

use crate::ivf::test::{
    cases::{METRIC, N_LISTS, grid_index, queries},
    harness::{IvfOracleRun, assert_same_distances, global_topk},
    provider::Strategy,
};

/// For every query and a sweep of `(k, nprobe)`, the search must equal the restricted
/// brute-force oracle.
#[test]
fn matches_restricted_brute_force() {
    let index = grid_index();
    let dim = index.provider().dim();
    let strategy = Strategy::new(dim, METRIC);

    for query in queries() {
        for nprobe in 1..=N_LISTS {
            for k in [1usize, 4, 16] {
                let run = IvfOracleRun::run_sync(&index, &strategy, &query, k, nprobe).unwrap();

                assert_same_distances(&run.top_k, &run.ground_truth);

                // `top_k_distances` is the distance projection of `top_k`.
                let projected: Vec<f32> = run.top_k.iter().map(|(_, d)| *d).collect();
                assert_eq!(run.top_k_distances, projected);

                // The fine step must compute exactly one distance per probed member.
                assert_eq!(
                    run.stats.cmps as usize, run.probed_members,
                    "query {query:?} k={k} nprobe={nprobe}: cmps must equal probed members",
                );

                // We never return more than the oracle (which is already truncated to k),
                // and never more than k.
                assert_eq!(run.top_k.len(), run.ground_truth.len());
                assert!(run.top_k.len() <= k);
            }
        }
    }
}

/// Probing every list (`nprobe == n_lists`) makes IVF exhaustive: it must equal the global
/// brute-force top-`k`.
#[test]
fn full_probe_equals_global_brute_force() {
    let index = grid_index();
    let dim = index.provider().dim();
    let strategy = Strategy::new(dim, METRIC);

    for query in queries() {
        for k in [1usize, 4, 16] {
            let run = IvfOracleRun::run_sync(&index, &strategy, &query, k, N_LISTS).unwrap();

            let global: Vec<(u32, f32)> = global_topk(index.provider(), &query, k)
                .into_iter()
                .map(|n| (n.id, n.distance))
                .collect();
            assert_same_distances(&run.top_k, &global);

            // Probing every list scans the whole dataset exactly once.
            assert_eq!(run.stats.cmps as usize, index.provider().len());
            assert_eq!(run.probed_lists.len(), N_LISTS);
        }
    }
}

/// `result_count` is `min(k, probed_members)` and matches the buffer contents.
#[test]
fn result_count_is_bounded() {
    let index = grid_index();
    let dim = index.provider().dim();
    let strategy = Strategy::new(dim, METRIC);
    let query = vec![0.0, 0.0];

    // nprobe = 1 probes a single 16-member quadrant.
    let run = IvfOracleRun::run_sync(&index, &strategy, &query, 100, 1).unwrap();
    assert_eq!(run.stats.result_count as usize, run.probed_members);
    assert_eq!(run.top_k.len(), run.probed_members);
    assert_eq!(run.stats.result_count as usize, run.top_k.len());
}

/// The provider's coarse/fine comparison counters reflect the work a search performs:
/// `n_lists` centroid comparisons (one full coarse pass) and one fine comparison per
/// probed member.
#[test]
fn provider_metrics_track_distance_work() {
    let index = grid_index();
    let dim = index.provider().dim();
    let strategy = Strategy::new(dim, METRIC);
    let query = vec![3.5f32, 3.5];

    let before = index.provider().metrics();
    let run = IvfOracleRun::run_sync(&index, &strategy, &query, 4, N_LISTS).unwrap();
    let after = index.provider().metrics();

    assert_eq!(index.provider().n_lists(), N_LISTS);
    assert_eq!(
        after.centroid_cmps - before.centroid_cmps,
        index.provider().n_lists(),
        "coarse step compares the query against every centroid once",
    );
    assert_eq!(
        after.member_cmps - before.member_cmps,
        run.probed_members,
        "fine step compares the query against every probed member once",
    );
}
