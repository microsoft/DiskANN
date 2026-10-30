/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::HashSet;

use crate::{
    graph::{
        self, AdjacencyList,
        ext::labeled,
        search::{Knn, PathSeer},
        test::provider as test_provider,
    },
    neighbor::Neighbor,
    test::tokio::current_thread_runtime,
};

use super::multihop::build_1d_index;

#[derive(Debug)]
struct MatchIds(HashSet<u32>);

impl labeled::QueryLabelProvider<u32> for MatchIds {
    fn is_match(&self, id: u32) -> bool {
        self.0.contains(&id)
    }
}

fn run(
    index: &graph::DiskANNIndex<test_provider::Provider>,
    matches: &[u32],
    l: usize,
) -> (graph::index::SearchStats, Vec<Neighbor<u32>>) {
    let filter = MatchIds(matches.iter().copied().collect());
    let mut neighbors = Vec::new();
    let stats = current_thread_runtime()
        .block_on(index.search_with(
            PathSeer::new(Knn::new_default(l).unwrap()),
            &labeled::Filtered::new(test_provider::Strategy::new(), &filter),
            graph::glue::CopyIds,
            &test_provider::Context::new(),
            &[3.0f32][..],
            &mut neighbors,
        ))
        .unwrap();
    (stats, neighbors)
}

#[test]
fn finds_matches_two_hops_away_before_heap_is_full() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (1, vec![1.0], AdjacencyList::from_iter_untrusted([3])),
            (2, vec![2.0], AdjacencyList::from_iter_untrusted([4])),
            (3, vec![3.0], AdjacencyList::new()),
            (4, vec![4.0], AdjacencyList::new()),
        ],
        2,
    );
    let (stats, results) = run(&index, &[3, 4], 2);
    assert_eq!(stats.result_count, 2);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [3, 4]);

    let (stats, results) = run(&index, &[], 2);
    assert_eq!(stats.result_count, 0);
    assert!(results.is_empty());
}

#[test]
fn rejected_parent_prefilters_direct_neighbors_after_heap_fills() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1]),
        vec![
            (1, vec![1.0], AdjacencyList::from_iter_untrusted([2, 3])),
            (2, vec![2.0], AdjacencyList::new()),
            (3, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    let (stats, results) = run(&index, &[0, 3], 1);
    assert_eq!(stats.result_count, 1);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [3]);
    assert_eq!(stats.cmps, 2);
}
