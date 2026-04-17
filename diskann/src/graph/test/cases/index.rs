/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::helpers::{create_2d_unit_square, generate_2d_square_adjacency_list, setup_2d_square};
use crate::{
    graph::{self, AdjacencyList, DiskANNIndex, test::provider as test_provider},
    neighbor::Neighbor,
    provider::{Delete, NeighborAccessor},
};
use std::sync::Arc;

fn setup_paged_search_test() -> (
    Arc<DiskANNIndex<test_provider::Provider>>,
    test_provider::Strategy,
    test_provider::Context,
    Vec<f32>,
) {
    let adjacency_list = generate_2d_square_adjacency_list();
    let index = setup_2d_square(create_2d_unit_square(), adjacency_list, 4);
    let strategy = test_provider::Strategy::new();
    let ctx = test_provider::Context::new();
    let query_point = vec![0.1, 0.2];
    (index, strategy, ctx, query_point)
}

#[test]
fn query_label_provider_on_visit_default() {
    use crate::graph::index::{QueryLabelProvider, QueryVisitDecision};

    #[derive(Debug)]
    struct BasicValidation;

    impl QueryLabelProvider<u32> for BasicValidation {
        fn is_match(&self, id: u32) -> bool {
            id.is_multiple_of(2)
        }
    }

    let filter = BasicValidation;
    assert!(matches!(
        filter.on_visit(Neighbor::new(0, 1.0)),
        QueryVisitDecision::Accept(_)
    ));
    assert!(matches!(
        filter.on_visit(Neighbor::new(1, 1.0)),
        QueryVisitDecision::Reject
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn test_count_reachable_nodes() {
    let adjacency_list = generate_2d_square_adjacency_list();
    let index = setup_2d_square(create_2d_unit_square(), adjacency_list, 4);
    let mut accessor = index.provider().neighbors();
    let starting_point = [4];

    let initial_result = index
        .count_reachable_nodes(&starting_point, &mut accessor)
        .await
        .unwrap();

    assert_eq!(initial_result, 5);

    let ctx = test_provider::Context::default();
    let strat = test_provider::Strategy::new();

    index
        .inplace_delete(
            strat.clone(),
            &ctx,
            &3,
            3,
            graph::InplaceDeleteMethod::OneHop,
        )
        .await
        .unwrap();

    let post_delete_result = index
        .count_reachable_nodes(&starting_point, &mut accessor)
        .await
        .unwrap();

    assert_eq!(post_delete_result, 4);
}

#[tokio::test(flavor = "current_thread")]
async fn test_count_unreachable_isolated_nodes() {
    let adjacency_list = vec![
        AdjacencyList::from_iter_untrusted([]),
        AdjacencyList::from_iter_untrusted([]),
        AdjacencyList::from_iter_untrusted([]),
        AdjacencyList::from_iter_untrusted([]),
        AdjacencyList::from_iter_untrusted([]),
    ];
    let index = setup_2d_square(create_2d_unit_square(), adjacency_list, 1);
    let mut accessor = index.provider().neighbors();
    let starting_point = [4];

    let count = index
        .count_reachable_nodes(&starting_point, &mut accessor)
        .await
        .unwrap();
    assert_eq!(count, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn test_get_degree_stats() {
    let adjacency_list = generate_2d_square_adjacency_list();
    let index = setup_2d_square(create_2d_unit_square(), adjacency_list, 4);
    let mut accessor = index.provider().neighbors();
    let stats = index.get_degree_stats(&mut accessor).await.unwrap();

    assert_eq!(stats.max_degree, 2);
    assert_eq!(stats.min_degree, 2);
    assert_eq!(stats.avg_degree, 2.0);
    assert_eq!(stats.cnt_less_than_two, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn test_prune_range() {
    let adjacency_list = generate_2d_square_adjacency_list();
    let index = setup_2d_square(create_2d_unit_square(), adjacency_list, 1);
    let ctx = test_provider::Context::default();
    let strat = test_provider::Strategy::new();

    index.prune_range(&strat, &ctx, 0..4).await.unwrap();

    let accessor = index.provider().neighbors();
    let mut list = AdjacencyList::new();
    for node in 0u32..4 {
        accessor.get_neighbors(node, &mut list).await.unwrap();
        assert!(
            list.len() <= 1,
            "node {node} should have degree <= 1 after prune, got {}",
            list.len()
        );
    }

    // Start node (4) is outside the pruned range — should be unchanged.
    accessor.get_neighbors(4, &mut list).await.unwrap();
    assert_eq!(
        list.len(),
        4,
        "start node should be untouched by prune_range(0..4)"
    );
}

/// Multiple start points should union their reachable sets.
#[tokio::test(flavor = "current_thread")]
async fn test_count_reachable_nodes_multiple_starts() {
    // Two disconnected components: nodes 0,1 connected to each other via start (4),
    // nodes 2,3 connected to each other but not to 0,1 or start.
    let adjacency_list = vec![
        AdjacencyList::from_iter_untrusted([1, 4]),
        AdjacencyList::from_iter_untrusted([0, 4]),
        AdjacencyList::from_iter_untrusted([3]),
        AdjacencyList::from_iter_untrusted([2]),
        AdjacencyList::from_iter_untrusted([0, 1]),
    ];
    let index = setup_2d_square(create_2d_unit_square(), adjacency_list, 4);
    let mut accessor = index.provider().neighbors();

    // From start (4) alone: reaches 4, 0, 1 = 3
    let from_start = index
        .count_reachable_nodes(&[4], &mut accessor)
        .await
        .unwrap();
    assert_eq!(from_start, 3);

    // From node 2 alone: reaches 2, 3 = 2
    let from_two = index
        .count_reachable_nodes(&[2], &mut accessor)
        .await
        .unwrap();
    assert_eq!(from_two, 2);

    // From both: union = all 5
    let from_both = index
        .count_reachable_nodes(&[4, 2], &mut accessor)
        .await
        .unwrap();
    assert_eq!(from_both, 5);
}

/// Degree stats with mixed degrees exercises min/max/avg/cnt_less_than_two independently.
#[tokio::test(flavor = "current_thread")]
async fn test_get_degree_stats_mixed() {
    // Node 0: degree 0 (isolated), Node 1: degree 1, Node 2: degree 2, Node 3: degree 2
    let adjacency_list = vec![
        AdjacencyList::from_iter_untrusted([]),
        AdjacencyList::from_iter_untrusted([4]),
        AdjacencyList::from_iter_untrusted([3, 4]),
        AdjacencyList::from_iter_untrusted([2, 4]),
        AdjacencyList::from_iter_untrusted([1, 2, 3]),
    ];
    let index = setup_2d_square(create_2d_unit_square(), adjacency_list, 4);
    let mut accessor = index.provider().neighbors();
    let stats = index.get_degree_stats(&mut accessor).await.unwrap();

    assert_eq!(stats.max_degree, 2);
    assert_eq!(stats.min_degree, 0);
    // avg = (0 + 1 + 2 + 2) / 4 = 1.25
    assert_eq!(stats.avg_degree, 1.25);
    // nodes with degree < 2: node 0 (degree 0) and node 1 (degree 1)
    assert_eq!(stats.cnt_less_than_two, 2);
}

#[tokio::test(flavor = "current_thread")]
async fn validate_basic_paged_search() {
    let (index, strategy, ctx, query_point) = setup_paged_search_test();

    let k = 2;
    let mut state = index
        .start_paged_search(strategy, &ctx, query_point.as_slice(), k)
        .await
        .unwrap();

    let mut output = vec![Neighbor::default(); k];
    let count = index
        .next_search_results(&ctx, &mut state, k, &mut output)
        .await
        .unwrap();

    assert_eq!(count, k);
    assert_eq!(output[0].id, 0);
    assert_eq!(output[1].id, 1);
    assert!(output[..count].iter().all(|n| n.id != 4)); // ensure start point isn't in the
    // output
}

#[tokio::test(flavor = "current_thread")]
async fn validate_paged_search_multiple() {
    let (index, strategy, ctx, query_point) = setup_paged_search_test();

    let k = 1;
    let mut state = index
        .start_paged_search(strategy, &ctx, query_point.as_slice(), k)
        .await
        .unwrap();

    let mut output = vec![Neighbor::default()];
    let mut visited: Vec<u32> = Vec::new();
    for _ in 0..4 {
        let count = index
            .next_search_results(&ctx, &mut state, k, &mut output)
            .await
            .unwrap();
        assert_eq!(count, 1);
        visited.push(output[0].id);
    }

    use std::collections::HashSet;
    let unique: HashSet<_> = visited.iter().collect();
    assert_eq!(unique.len(), visited.len());
}

#[tokio::test(flavor = "current_thread")]
async fn test_paged_search_error_cases() {
    let (index, strategy, ctx, query_point) = setup_paged_search_test();

    let l_value = 2;
    let mut state = index
        .start_paged_search(strategy, &ctx, query_point.as_slice(), l_value)
        .await
        .unwrap();

    let mut output = vec![Neighbor::default(); 2];

    // k > l_value
    let result = index
        .next_search_results(&ctx, &mut state, l_value + 1, &mut output)
        .await;
    assert!(result.is_err());

    // k == 0
    let result = index
        .next_search_results(&ctx, &mut state, 0, &mut output)
        .await;
    assert!(result.is_err());

    // output buffer too small for requested k
    let mut small_output = vec![Neighbor::default(); 1];
    let result = index
        .next_search_results(&ctx, &mut state, 2, &mut small_output)
        .await;
    assert!(result.is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn validate_basic_paged_search_with_init_ids() {
    let (index, strategy, ctx, query_point) = setup_paged_search_test();

    let k = 2;

    // Seed search from nodes 0 and 1 instead of using the default start point
    let init_ids: &[u32] = &[0, 1];
    let mut state = index
        .start_paged_search_with_init_ids(strategy, &ctx, query_point.as_slice(), k, Some(init_ids))
        .await
        .unwrap();

    let mut output = vec![Neighbor::default(); k];
    let count = index
        .next_search_results(&ctx, &mut state, k, &mut output)
        .await
        .unwrap();

    assert_eq!(count, k);
    // Init IDs (0, 1) are filtered out as start points; results are from remaining nodes
    assert!(
        output[..count]
            .iter()
            .all(|n| n.id != 0 && n.id != 1 && n.id != 4)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn validate_paged_search_with_init_ids_multiple() {
    let (index, strategy, ctx, query_point) = setup_paged_search_test();

    let k = 1;

    let init_ids: &[u32] = &[0, 1];
    let mut state = index
        .start_paged_search_with_init_ids(strategy, &ctx, query_point.as_slice(), k, Some(init_ids))
        .await
        .unwrap();

    let mut output = vec![Neighbor::default()];
    let mut visited: Vec<u32> = Vec::new();
    for _ in 0..2 {
        let count = index
            .next_search_results(&ctx, &mut state, k, &mut output)
            .await
            .unwrap();
        assert_eq!(count, 1);
        visited.push(output[0].id);
    }

    use std::collections::HashSet;
    let unique: HashSet<_> = visited.iter().collect();
    assert_eq!(unique.len(), visited.len());
}

#[tokio::test(flavor = "current_thread")]
async fn test_paged_search_with_init_ids_error_cases() {
    let (index, strategy, ctx, query_point) = setup_paged_search_test();

    let l_value = 2;

    let init_ids: &[u32] = &[0, 1];
    let mut state = index
        .start_paged_search_with_init_ids(
            strategy,
            &ctx,
            query_point.as_slice(),
            l_value,
            Some(init_ids),
        )
        .await
        .unwrap();

    let mut output = vec![Neighbor::default(); 2];

    // k > l_value
    let result = index
        .next_search_results(&ctx, &mut state, l_value + 1, &mut output)
        .await;
    assert!(result.is_err());

    // k == 0
    let result = index
        .next_search_results(&ctx, &mut state, 0, &mut output)
        .await;
    assert!(result.is_err());

    // output buffer too small for requested k
    let mut small_output = vec![Neighbor::default(); 1];
    let result = index
        .next_search_results(&ctx, &mut state, 2, &mut small_output)
        .await;
    assert!(result.is_err());
}

#[tokio::test(flavor = "current_thread")]
async fn test_is_any_neighbor_deleted() {
    let adjacency_list = generate_2d_square_adjacency_list();
    let index = setup_2d_square(create_2d_unit_square(), adjacency_list, 4);
    let ctx = test_provider::Context::new();
    let mut accessor = index.provider().neighbors();

    // Before any deletion, no node should have deleted neighbors.
    let result = index
        .is_any_neighbor_deleted(&ctx, &mut accessor, 2)
        .await
        .unwrap();
    assert!(!result, "no neighbors should be deleted yet");

    // Delete node 3 — node 2 has neighbors [3, 4], so it should now return true.
    index.provider().delete(&ctx, &3).await.unwrap();

    let result = index
        .is_any_neighbor_deleted(&ctx, &mut accessor, 2)
        .await
        .unwrap();
    assert!(result, "node 2 should detect deleted neighbor 3");

    // Node 0 has neighbors [1, 4] — neither deleted, should still return false.
    let result = index
        .is_any_neighbor_deleted(&ctx, &mut accessor, 0)
        .await
        .unwrap();
    assert!(!result, "node 0 has no deleted neighbors");
}
