/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

#[test]
fn prunes_an_overfull_list_with_vamana_robust_prune() {
    let data = vec![
        0.0f32, 0.0, // source
        1.0, 0.0, 0.0, 1.0, 0.1, 0.0, // closest
    ];
    let candidates = vec![vec![3, 1, 2], vec![], vec![], vec![]];

    let graph = prune_overfull_lists(&data, 2, candidates, 2, Metric::L2, 1.2).unwrap();

    assert!(graph[0].contains(&3));
    assert!(graph[0].len() <= 2);
}

#[test]
fn preserves_lists_within_the_degree_bound() {
    let data = vec![0.0f32; 8];
    let candidates = vec![vec![3, 1, 2], vec![], vec![], vec![]];

    let graph = prune_overfull_lists(&data, 2, candidates, 4, Metric::L2, 1.2).unwrap();

    assert_eq!(graph[0], [3, 1, 2]);
}

#[test]
fn rejects_unrepresentable_candidate_count_without_truncating() {
    let candidate_count = u16::MAX as usize + 1;
    let data = vec![0.0f32; candidate_count + 1];
    let candidates = vec![(1..=candidate_count as u32).collect()];

    let error = prune_overfull_lists(&data, 1, candidates, 1, Metric::L2, 1.2)
        .expect_err("the shared prune kernel must reject an oversized candidate set");

    assert!(matches!(
        error.downcast_ref::<prune::RobustPruneError<Infallible>>(),
        Some(prune::RobustPruneError::TooManyCandidates { actual, max })
            if *actual == candidate_count && *max == u16::MAX as usize
    ));
}
