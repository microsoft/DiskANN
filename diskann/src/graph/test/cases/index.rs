/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::helpers::{create_2d_unit_square, generate_2d_square_adjacency_list, setup_2d_square};
use crate::{
    graph::{self, AdjacencyList, test::provider as test_provider},
    neighbor::Neighbor,
    provider::NeighborAccessor,
};

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

    let mut accessor = index.provider().neighbors();
    let mut list = AdjacencyList::new();
    for node in 0u32..4 {
        accessor.get_neighbors(node, &mut list).await.unwrap();
        assert!(
            list.len() <= 1,
            "node {node} should have degree <= 1 after prune, got {}",
            list.len()
        );
    }
}
