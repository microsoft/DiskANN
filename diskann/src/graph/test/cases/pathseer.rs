/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::HashSet;

use crate::{
    graph::{
        self, AdjacencyList,
        ext::labeled,
        search::{Knn, PathSeer, PathSeerMethod},
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
    run_with_method(index, matches, l, PathSeerMethod::Original)
}

fn run_with_method(
    index: &graph::DiskANNIndex<test_provider::Provider>,
    matches: &[u32],
    l: usize,
    method: PathSeerMethod,
) -> (graph::index::SearchStats, Vec<Neighbor<u32>>) {
    let filter = MatchIds(matches.iter().copied().collect());
    run_with_filter(index, &filter, l, method)
}

fn run_with_filter(
    index: &graph::DiskANNIndex<test_provider::Provider>,
    filter: &dyn labeled::QueryLabelProvider<u32>,
    l: usize,
    method: PathSeerMethod,
) -> (graph::index::SearchStats, Vec<Neighbor<u32>>) {
    let mut neighbors = Vec::new();
    let stats = current_thread_runtime()
        .block_on(index.search_with(
            PathSeer::new(Knn::new_default(l).unwrap()).with_method(method),
            &labeled::Filtered::new(test_provider::Strategy::new(), filter),
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

#[test]
fn what_if_1_limits_matching_second_neighbors_per_parent() {
    // round(ln(8)) = 2 matching IDs per parent. Rejected IDs do not use quota.
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (
                1,
                vec![100.0],
                AdjacencyList::from_iter_untrusted([3, 4, 5, 6]),
            ),
            (
                2,
                vec![200.0],
                AdjacencyList::from_iter_untrusted([7, 8, 9, 10]),
            ),
            (3, vec![2.0], AdjacencyList::new()),
            (4, vec![3.0], AdjacencyList::new()),
            (5, vec![4.0], AdjacencyList::new()),
            (6, vec![5.0], AdjacencyList::new()),
            (7, vec![2.0], AdjacencyList::new()),
            (8, vec![6.0], AdjacencyList::new()),
            (9, vec![7.0], AdjacencyList::new()),
            (10, vec![8.0], AdjacencyList::new()),
        ],
        8,
    );
    let (stats, results) =
        run_with_method(&index, &[4, 5, 6, 8, 9, 10], 4, PathSeerMethod::WhatIf1);
    assert_eq!(
        results.iter().map(|n| *n.id()).collect::<Vec<_>>(),
        [4, 5, 8, 9]
    );
    assert_eq!(stats.cmps, 6);
}

#[test]
fn what_if_1_visited_matches_still_consume_quota() {
    let index = build_1d_index(
        0,
        10.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (
                1,
                vec![100.0],
                AdjacencyList::from_iter_untrusted([2, 3, 4]),
            ),
            (2, vec![3.0], AdjacencyList::new()),
            (3, vec![4.0], AdjacencyList::new()),
            (4, vec![5.0], AdjacencyList::new()),
        ],
        8,
    );
    let (stats, results) = run_with_method(&index, &[0, 2, 3, 4], 3, PathSeerMethod::WhatIf1);
    assert_eq!(
        results.iter().map(|n| *n.id()).collect::<Vec<_>>(),
        [2, 3, 0]
    );
    assert_eq!(stats.cmps, 3);
}

#[test]
fn what_if_1_marks_rejected_second_neighbors_visited_without_scoring() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1]),
        vec![
            (1, vec![100.0], AdjacencyList::from_iter_untrusted([2, 3])),
            (2, vec![2.0], AdjacencyList::new()),
            (3, vec![3.0], AdjacencyList::new()),
        ],
        8,
    );
    let (original_stats, _) = run(&index, &[3], 3);
    let (stats, results) = run_with_method(&index, &[3], 3, PathSeerMethod::WhatIf1);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [3]);
    // The original later scores rejected ID 2 when expanding parent 1.
    assert_eq!(original_stats.cmps, 3);
    assert_eq!(stats.cmps, 2);
}

#[test]
fn what_if_1_does_not_repeat_second_order_expansion() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (1, vec![1.0], AdjacencyList::from_iter_untrusted([3])),
            (2, vec![2.0], AdjacencyList::from_iter_untrusted([3])),
            (3, vec![3.0], AdjacencyList::from_iter_untrusted([4])),
            (4, vec![4.0], AdjacencyList::new()),
        ],
        8,
    );
    let (stats, results) = run_with_method(&index, &[3, 4], 10, PathSeerMethod::WhatIf1);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [3, 4]);
    assert_eq!(stats.cmps, 4);
    // Five candidates expand directly, and four distinct second-order parents
    // are scanned. Parent 3 is reached from both 1 and 2 but scanned only once.
    assert_eq!(stats.hops, 9);
}

#[test]
fn what_if_1_bounds_all_second_order_filter_checks_across_parents() {
    #[derive(Debug)]
    struct RecordingFilter(std::sync::Mutex<Vec<u32>>);
    impl labeled::QueryLabelProvider<u32> for RecordingFilter {
        fn is_match(&self, id: u32) -> bool {
            self.0.lock().unwrap().push(id);
            matches!(id, 5 | 7)
        }
    }
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (
                1,
                vec![100.0],
                AdjacencyList::from_iter_untrusted([3, 4, 5]),
            ),
            (2, vec![200.0], AdjacencyList::from_iter_untrusted([6, 7])),
            (3, vec![300.0], AdjacencyList::new()),
            (4, vec![400.0], AdjacencyList::new()),
            (5, vec![3.0], AdjacencyList::new()),
            (6, vec![600.0], AdjacencyList::new()),
            (7, vec![4.0], AdjacencyList::new()),
        ],
        4,
    );
    let filter = RecordingFilter(std::sync::Mutex::new(Vec::new()));
    let (stats, results) = run_with_filter(&index, &filter, 1, PathSeerMethod::WhatIf1);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [5]);
    assert_eq!(stats.cmps, 3);
    // Only 3, 4, 5, 6 are classified in the second hop. Rejected IDs consume
    // the shared budget; scoring selected ID 5 must not classify it again.
    let checks = filter.0.lock().unwrap();
    assert_eq!(
        checks
            .iter()
            .copied()
            .filter(|id| *id >= 3)
            .collect::<Vec<_>>(),
        [3, 4, 5, 6]
    );
}

#[test]
fn what_if_1_can_resume_a_budget_truncated_parent() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (
                1,
                vec![100.0],
                AdjacencyList::from_iter_untrusted([3, 4, 5]),
            ),
            (2, vec![200.0], AdjacencyList::from_iter_untrusted([6, 7])),
            (3, vec![300.0], AdjacencyList::new()),
            (4, vec![400.0], AdjacencyList::new()),
            (5, vec![3.0], AdjacencyList::from_iter_untrusted([2])),
            (6, vec![600.0], AdjacencyList::new()),
            (7, vec![4.0], AdjacencyList::new()),
        ],
        4,
    );
    let (stats, results) = run_with_method(&index, &[5, 7], 2, PathSeerMethod::WhatIf1);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [5, 7]);
    assert_eq!(stats.cmps, 4);
    // Parent 2 was cut off after ID 6 on the first expansion. Reaching it via
    // candidate 5 must classify 6 again (using budget) and then discover ID 7.
    assert_eq!(stats.hops, 6);
}

#[test]
fn what_if_2_keeps_more_than_l_unmatched_candidates() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (1, vec![2.0], AdjacencyList::new()),
            (2, vec![5.0], AdjacencyList::from_iter_untrusted([3])),
            (3, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    // With L=1, the farther rejected candidate 2 must survive candidate 1.
    let (stats, results) = run_with_method(&index, &[3], 1, PathSeerMethod::WhatIf2);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [3]);
    assert_eq!(stats.cmps, 3);
    assert_eq!(stats.hops, 4);
}

#[test]
fn what_if_2_stops_at_matching_gate_without_two_hop_expansion() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (1, vec![4.0], AdjacencyList::new()),
            (2, vec![100.0], AdjacencyList::from_iter_untrusted([3])),
            (3, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    let (stats, results) = run_with_method(&index, &[1, 3], 1, PathSeerMethod::WhatIf2);
    // Candidate 2 is beyond the matching-result gate. Its neighbor 3 is never seen.
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [1]);
    assert_eq!(stats.cmps, 2);
    assert_eq!(stats.hops, 2);
}

#[test]
fn what_if_2_scores_rejected_neighbors_even_with_full_results() {
    let index = build_1d_index(
        0,
        10.0,
        AdjacencyList::from_iter_untrusted([1]),
        vec![
            (1, vec![2.0], AdjacencyList::from_iter_untrusted([2, 3])),
            (2, vec![4.0], AdjacencyList::new()),
            (3, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    let (stats, results) = run_with_method(&index, &[0, 3], 1, PathSeerMethod::WhatIf2);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [3]);
    // Start 0 already fills results. Both rejected IDs 1 and 2 are still scored.
    assert_eq!(stats.cmps, 3);
    assert_eq!(stats.hops, 3);
}

#[test]
fn what_if_2_expands_candidates_equal_to_the_gate() {
    let index = build_1d_index(
        0,
        2.0,
        AdjacencyList::from_iter_untrusted([1]),
        vec![
            (1, vec![4.0], AdjacencyList::from_iter_untrusted([2])),
            (2, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    let (stats, results) = run_with_method(&index, &[0, 2], 1, PathSeerMethod::WhatIf2);
    // Rejected candidate 1 ties accepted start 0, so strict > must allow expansion.
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [2]);
    assert_eq!(stats.cmps, 2);
}

#[test]
fn what_if_2_keeps_only_l_nearest_matching_results() {
    let index = build_1d_index(
        0,
        100.0,
        AdjacencyList::from_iter_untrusted([1, 2, 3, 4]),
        vec![
            (1, vec![3.0], AdjacencyList::new()),
            (2, vec![4.0], AdjacencyList::new()),
            (3, vec![5.0], AdjacencyList::new()),
            (4, vec![6.0], AdjacencyList::new()),
        ],
        4,
    );
    let (stats, results) = run_with_method(&index, &[1, 2, 3], 2, PathSeerMethod::WhatIf2);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [1, 2]);
    assert_eq!(stats.result_count, 2);
    assert_eq!(stats.cmps, 4);
    assert_eq!(stats.hops, 3);
}

#[test]
fn what_if_2_exhausts_candidates_when_fewer_than_l_match() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1]),
        vec![(1, vec![1.0], AdjacencyList::from_iter_untrusted([0]))],
        2,
    );
    let (stats, results) = run_with_method(&index, &[], 2, PathSeerMethod::WhatIf2);
    assert!(results.is_empty());
    assert_eq!(stats.cmps, 1);
    assert_eq!(stats.hops, 2);
}

#[test]
fn what_if_3_finds_two_hop_matches_before_distant_parents_are_expanded() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (1, vec![4.0], AdjacencyList::new()),
            (2, vec![100.0], AdjacencyList::from_iter_untrusted([3])),
            (3, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    let (_, single_hop) = run_with_method(&index, &[1, 3], 1, PathSeerMethod::WhatIf2);
    let (stats, results) = run_with_method(&index, &[1, 3], 1, PathSeerMethod::WhatIf3);
    assert_eq!(single_hop.iter().map(|n| *n.id()).collect::<Vec<_>>(), [1]);
    // The phase is selected before direct neighbor 1 fills results. Second-hop
    // ID 3 is still scored this round, before distant parent 2 can be popped.
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [3]);
    assert_eq!(stats.cmps, 3);
}

#[test]
fn what_if_3_leaves_rejected_second_neighbors_unvisited() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1]),
        vec![
            (1, vec![100.0], AdjacencyList::from_iter_untrusted([2, 3])),
            (2, vec![2.0], AdjacencyList::new()),
            (3, vec![3.0], AdjacencyList::new()),
        ],
        4,
    );
    let (stats, results) = run_with_method(&index, &[3], 3, PathSeerMethod::WhatIf3);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [3]);
    // ID 2 is rejected during the second hop, then scored when parent 1 is popped.
    // ID 3, already scored in the second hop, must not be scored a second time.
    assert_eq!(stats.cmps, 3);
}

#[test]
fn what_if_3_shares_raw_second_order_budget_across_parents() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (1, vec![100.0], AdjacencyList::from_iter_untrusted([3, 4])),
            (2, vec![200.0], AdjacencyList::from_iter_untrusted([5])),
            (3, vec![2.0], AdjacencyList::new()),
            (4, vec![4.0], AdjacencyList::new()),
            (5, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    let (stats, results) = run_with_method(&index, &[4, 5], 1, PathSeerMethod::WhatIf3);
    // Rejected ID 3 and matching ID 4 exhaust the budget of two. Parent 2's
    // closer matching neighbor 5 remains unseen when the distance gate stops us.
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [4]);
    assert_eq!(stats.cmps, 3);
    assert_eq!(stats.hops, 3);
}

#[test]
fn what_if_3_full_results_disable_second_hop_but_not_direct_scoring() {
    let index = build_1d_index(
        0,
        10.0,
        AdjacencyList::from_iter_untrusted([1]),
        vec![
            (1, vec![2.0], AdjacencyList::from_iter_untrusted([2, 3])),
            (2, vec![4.0], AdjacencyList::new()),
            (3, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    let (stats, results) = run_with_method(&index, &[0, 3], 1, PathSeerMethod::WhatIf3);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [3]);
    // Start 0 fills results before expansion. There are no second-hop reads;
    // rejected parent 1 still scores both its rejected and matching neighbors.
    assert_eq!(stats.cmps, 3);
    assert_eq!(stats.hops, 3);
}

#[test]
fn what_if_4_has_no_second_order_expansion_even_before_results_fill() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (1, vec![4.0], AdjacencyList::new()),
            (2, vec![100.0], AdjacencyList::from_iter_untrusted([3])),
            (3, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    let (_, original) = run(&index, &[1, 3], 1);
    let (stats, results) = run_with_method(&index, &[1, 3], 1, PathSeerMethod::WhatIf4);
    assert_eq!(original.iter().map(|n| *n.id()).collect::<Vec<_>>(), [3]);
    // Original PathSeer reaches ID 3 in two hops. Here parent 2 is beyond
    // the distance gate, so its closer matching neighbor is never discovered.
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [1]);
    assert_eq!(stats.cmps, 2);
    assert_eq!(stats.hops, 2);
}

#[test]
fn parent_fail_variants_prefilter_rejected_parents_after_results_fill() {
    let index = build_1d_index(
        0,
        10.0,
        AdjacencyList::from_iter_untrusted([1]),
        vec![
            (1, vec![2.0], AdjacencyList::from_iter_untrusted([2, 3])),
            (2, vec![4.0], AdjacencyList::new()),
            (3, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    for method in [PathSeerMethod::WhatIf4] {
        let (stats, results) = run_with_method(&index, &[0, 3], 1, method);
        assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [3]);
        // Unlike method 3, rejected parent 1 prefilters ID 2 instead of scoring it.
        assert_eq!(stats.cmps, 2);
        assert_eq!(stats.hops, 3);
    }
}

#[test]
fn what_if_3_gates_direct_candidates_when_results_are_full() {
    let index = build_1d_index(
        0,
        2.0,
        AdjacencyList::from_iter_untrusted([1]),
        vec![
            (1, vec![4.0], AdjacencyList::from_iter_untrusted([2])),
            (2, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    let (_, unbounded) = run_with_method(&index, &[0, 2], 1, PathSeerMethod::WhatIf2);
    let (stats, results) = run_with_method(&index, &[0, 2], 1, PathSeerMethod::WhatIf3);
    assert_eq!(unbounded.iter().map(|n| *n.id()).collect::<Vec<_>>(), [2]);
    // Start 0 fills results. Direct candidate 1 ties its distance, so admission
    // rejects it; a gate applied only at pop (strict >) would have expanded it.
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [0]);
    assert_eq!(stats.cmps, 1);
    assert_eq!(stats.hops, 1);
}

#[test]
fn what_if_3_also_gates_scored_second_order_candidates() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (1, vec![4.0], AdjacencyList::from_iter_untrusted([3])),
            (2, vec![100.0], AdjacencyList::new()),
            (3, vec![2.0], AdjacencyList::from_iter_untrusted([4])),
            (4, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    let (stats, results) = run_with_method(&index, &[1, 3, 4], 1, PathSeerMethod::WhatIf3);
    // Direct ID 1 fills results. Matching second-hop ID 3 is scored and visited,
    // but ties the gate and cannot enter candidates to expose closer ID 4.
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [1]);
    assert_eq!(stats.cmps, 3);
    assert_eq!(stats.hops, 4);
}

#[test]
fn what_if_4_switches_from_scoring_all_neighbors_to_parent_fail() {
    let index = build_1d_index(
        0,
        0.0,
        AdjacencyList::from_iter_untrusted([1, 2]),
        vec![
            (1, vec![6.0], AdjacencyList::new()),
            (2, vec![4.0], AdjacencyList::from_iter_untrusted([3, 4])),
            (3, vec![2.0], AdjacencyList::new()),
            (4, vec![3.0], AdjacencyList::new()),
        ],
        2,
    );
    let (stats, results) = run_with_method(&index, &[1, 4], 1, PathSeerMethod::WhatIf4);
    assert_eq!(results.iter().map(|n| *n.id()).collect::<Vec<_>>(), [4]);
    // Phase 1 scores matching ID 1 and rejected ID 2. Result fullness then
    // switches rejected parent 2 to prefiltering: ID 3 is skipped, ID 4 scored.
    assert_eq!(stats.cmps, 3);
    assert_eq!(stats.hops, 3);
}
