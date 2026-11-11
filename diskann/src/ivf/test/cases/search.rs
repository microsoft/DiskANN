/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    graph::{IdDistance, SearchOutputBuffer},
    ivf::{
        IVFIndex,
        test::{
            harness::{
                assert_fails, assert_neighbors, assert_scans, config, delta, line,
                scanned_brute_force, search, search_into,
            },
            provider::Call,
        },
    },
    neighbor::{BackInserter, Neighbor},
};

// The exact match at 7.5 is in the farthest list, not the nearest centroid.
const LISTS: &[(f32, &[f32])] = &[(1.0, &[0.0, 2.0]), (11.0, &[10.0, 12.0]), (100.0, &[7.5])];

fn index() -> IVFIndex {
    IVFIndex::new(config(usize::MAX, 2, 0)).unwrap()
}

/// A sentinel that cannot be mistaken for a squared-L2 search result.
fn prefilled() -> Vec<Neighbor<u32>> {
    vec![Neighbor::new(42, -1.0)]
}

async fn check_search(lists: &[(f32, &[f32])], query: f32, k: usize, nprobe: usize) {
    // Rebuild for each parameter combination so counters describe exactly one search.
    let provider = line(lists).await;
    let snapshot = provider.snapshot();
    let candidates = scanned_brute_force(&snapshot, &[query], nprobe);
    let mut output = prefilled();
    let count = search_into(&index(), &provider, &[query], k, nprobe, &mut output)
        .await
        .unwrap();

    assert_eq!(output[0].as_tuple(), (42, -1.0));
    assert_eq!(
        count,
        k.min(candidates.len()),
        "query {query}, k {k}, nprobe {nprobe}"
    );
    assert_neighbors(&output[1..], &candidates, k);
    assert_scans(
        provider.metrics(),
        nprobe.min(snapshot.lists().len()),
        candidates.len(),
    );
}

#[tokio::test]
async fn search_returns_the_best_k_of_the_scanned_lists() {
    for query in [7.5, 6.0] {
        for nprobe in [1, 2, 3, usize::MAX] {
            for k in [1, 2, 3, 10] {
                check_search(LISTS, query, k, nprobe).await;
            }
        }
    }
}

#[tokio::test]
async fn search_respects_the_remaining_output_capacity() {
    let provider = line(LISTS).await;
    let index = index();
    let mut ids = [u32::MAX; 3];
    let mut distances = [f32::INFINITY; 3];
    {
        let mut output = IdDistance::new(&mut ids, &mut distances);
        assert!(output.push(Neighbor::new(42, -1.0)).is_available());
        // A full output suppresses appends, not the underlying scan.
        for expected in [2, 0] {
            let before = provider.metrics();
            let count = search_into(&index, &provider, &[7.5], 4, 2, &mut output)
                .await
                .unwrap();
            assert_eq!(count, expected);
            assert_eq!(output.current_len(), 3);
            assert_scans(delta(before, provider.metrics()), 2, 4);
        }
    }
    assert_eq!(ids, [42, 2, 3]);
    assert_eq!(distances, [-1.0, 6.25, 20.25]);

    let mut buffer = [];
    let mut output = BackInserter::new(&mut buffer);
    let before = provider.metrics();
    assert_eq!(
        search_into(&index, &provider, &[7.5], 1, 1, &mut output)
            .await
            .unwrap(),
        0
    );
    assert_scans(delta(before, provider.metrics()), 1, 2);
}

#[tokio::test]
async fn search_handles_empty_lists_and_empty_providers() {
    let cases: [&[(f32, &[f32])]; 3] = [
        &[(1.0, &[0.0, 2.0]), (11.0, &[]), (100.0, &[100.0])],
        &[(1.0, &[]), (11.0, &[])],
        &[],
    ];
    for lists in cases {
        for nprobe in [1, usize::MAX] {
            check_search(lists, 7.5, 5, nprobe).await;
        }
    }
}

#[tokio::test]
async fn failed_search_at_every_scan_leaves_output_unchanged() {
    for nth in 0..3 {
        let provider = line(LISTS).await.with_fault(Call::Scan, nth);
        let index = index();
        let snapshot = provider.snapshot();
        let mut output = prefilled();
        assert_fails(
            search_into(&index, &provider, &[7.5], 3, usize::MAX, &mut output).await,
            "injected Scan failure",
        );
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].as_tuple(), (42, -1.0));
        let failed = provider.metrics();
        assert_scans(failed, nth + 1, 2 * nth);

        let found = search(&index, &provider, &[7.5], 3, usize::MAX)
            .await
            .unwrap();
        let candidates = scanned_brute_force(&snapshot, &[7.5], usize::MAX);
        assert_neighbors(&found, &candidates, 3);
        assert_scans(delta(failed, provider.metrics()), 3, 5);
    }
}
