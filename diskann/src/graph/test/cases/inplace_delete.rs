/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{iter, sync::Arc};

use diskann_vector::distance::Metric;

use crate::{
    graph::{
        self, AdjacencyList, DiskANNIndex, InplaceDeleteMethod,
        test::provider::{self as test_provider},
    },
    provider::NeighborAccessor,
};

use super::helpers::{
    create_2d_unit_square, generate_2d_square_adjacency_list, setup_2d_square,
    setup_2d_square_using_synthetics_grid,
};

fn inplace_delete_setup() -> Arc<DiskANNIndex<test_provider::Provider>> {
    let provider_config = test_provider::Config::new(
        Metric::L2,
        10,
        test_provider::StartPoint::new(0, vec![0.0, 0.0]),
    )
    .unwrap();
    let provider = test_provider::Provider::new_from(
        provider_config,
        iter::once((0, AdjacencyList::new())),
        iter::empty(),
    )
    .unwrap();

    let index_config = graph::config::Builder::new(
        10,
        graph::config::MaxDegree::default_slack(),
        15,
        Metric::L2.into(),
    )
    .build()
    .unwrap();

    Arc::new(DiskANNIndex::new(index_config, provider, None))
}

/// Test that `inplace_delete()` succeeds on a simple index. The test provider will refuse to
/// translate internal/external IDs of deleted vectors once the `DataProvider::delete` call
/// returns. Inplace delete should still be able to complete successfully.
#[tokio::test(flavor = "current_thread")]
async fn basic_single() {
    let index = inplace_delete_setup();

    let ctx = test_provider::Context::default();
    let strat = test_provider::Strategy::new();

    for i in 1..6 {
        index
            .insert(strat.clone(), &ctx, &i, &[i as f32, i as f32])
            .await
            .unwrap();
    }

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
}

/// Test that `multi_inplace_delete()` succeeds on a simple index. The test provider will refuse to
/// translate internal/external IDs of deleted vectors once the `DataProvider::delete` call
/// returns. Inplace delete should still be able to complete successfully. As the single and multi
/// in place delete logic have slightly different code paths for ID tranlsations, we have tests
/// for both.
#[tokio::test(flavor = "current_thread")]
async fn basic_multi() {
    let index = inplace_delete_setup();

    let ctx = test_provider::Context::default();
    let strat = test_provider::Strategy::new();

    for i in 1..6 {
        index
            .insert(strat.clone(), &ctx, &i, &[i as f32, i as f32])
            .await
            .unwrap();
    }

    index
        .multi_inplace_delete(
            strat.clone(),
            &ctx,
            Arc::new([3, 4]),
            3,
            graph::InplaceDeleteMethod::OneHop,
        )
        .await
        .unwrap();
}

/// Sets up 2D square, deletes node 3 with the given method, then validates.
async fn delete_node_3_and_validate(method: InplaceDeleteMethod) {
    let adjacency_lists = generate_2d_square_adjacency_list();
    let index = setup_2d_square(create_2d_unit_square(), adjacency_lists, 4);
    let ctx = test_provider::Context::new();

    index
        .inplace_delete(test_provider::Strategy::new(), &ctx, &3, 3, method)
        .await
        .unwrap();

    let neighbors = &index.provider().neighbors();
    validate_graph_rebuild_for_simple_graph_after_3_delete(neighbors).await;
}

/// Sets up 2D square, multi-deletes nodes 2 and 3 with the given method, then validates.
async fn multi_delete_2_and_3_and_validate(method: InplaceDeleteMethod) {
    let adjacency_lists = generate_2d_square_adjacency_list();
    let index = setup_2d_square(create_2d_unit_square(), adjacency_lists, 4);
    let ctx = test_provider::Context::new();

    index
        .multi_inplace_delete(
            test_provider::Strategy::new(),
            &ctx,
            Arc::new([2, 3]),
            3,
            method,
        )
        .await
        .unwrap();

    let neighbors = &index.provider().neighbors();
    validate_graph_after_2_and_3_delete(neighbors).await;
}

#[tokio::test(flavor = "current_thread")]
async fn inplace_delete_onehop() {
    delete_node_3_and_validate(InplaceDeleteMethod::OneHop).await;
}

#[tokio::test(flavor = "current_thread")]
async fn inplace_delete_twohop_and_onehop() {
    delete_node_3_and_validate(InplaceDeleteMethod::TwoHopAndOneHop).await;
}

#[tokio::test(flavor = "current_thread")]
async fn inplace_delete_visited_and_topk() {
    delete_node_3_and_validate(InplaceDeleteMethod::VisitedAndTopK {
        k_value: 4,
        l_value: 10,
    })
    .await;
}

/// Multi-delete vertices 2 and 3 using TwoHopAndOneHop, then validate the graph.
#[tokio::test(flavor = "current_thread")]
async fn multi_inplace_delete_twohop_and_onehop() {
    multi_delete_2_and_3_and_validate(InplaceDeleteMethod::TwoHopAndOneHop).await;
}

/// Multi-delete vertices 2 and 3 using VisitedAndTopK, then validate the graph.
#[tokio::test(flavor = "current_thread")]
async fn multi_inplace_delete_visited_and_topk() {
    multi_delete_2_and_3_and_validate(InplaceDeleteMethod::VisitedAndTopK {
        k_value: 4,
        l_value: 10,
    })
    .await;
}

async fn validate_graph_rebuild_for_simple_graph_after_3_delete<N>(neighbors: &N)
where
    N: NeighborAccessor<Id = u32> + Copy,
{
    // Validate graph repair after deletion.
    // VisitedAndTopK does a full search for candidates, so it may add edges
    // beyond just replacing the deleted vertex's connections. We verify:
    // 1. Vertex 3 is absent from all neighbor lists
    // 2. Key structural properties hold
    let mut list = AdjacencyList::new();

    for node in [0u32, 1, 2, 4] {
        neighbors.get_neighbors(node, &mut list).await.unwrap();
        assert!(
            !list.contains(3),
            "node {node} should not point to deleted vertex 3, got: {:?}",
            &*list
        );
    }

    // Node 2 originally only connected to [3, 4]. After 3's deletion,
    // repair must have given it at least one live neighbor.
    neighbors.get_neighbors(2, &mut list).await.unwrap();
    assert!(
        !list.is_empty(),
        "node 2 should still have neighbors after repair"
    );
    assert!(list.contains(4), "node 2 should still point to start (4)");

    // Start node (4) should still connect to all live corners.
    neighbors.get_neighbors(4, &mut list).await.unwrap();
    list.sort();
    assert_eq!(
        &*list,
        &[0, 1, 2],
        "start node 4 should connect to live corners"
    );
}

/// Validate the graph after deleting both vertices 2 and 3.
/// Live nodes: 0, 1, 4 (start). Original adjacency:
///   0: [1, 4]  — no edges to deleted nodes
///   1: [0, 4]  — no edges to deleted nodes
///   2: [3, 4]  — deleted
///   3: [2, 4]  — deleted
///   4: [0, 1, 2, 3] — edges to 2 and 3 must be removed
async fn validate_graph_after_2_and_3_delete<N>(neighbors: &N)
where
    N: NeighborAccessor<Id = u32> + Copy,
{
    let mut list = AdjacencyList::new();

    // No live node should reference either deleted vertex.
    for node in [0u32, 1, 4] {
        neighbors.get_neighbors(node, &mut list).await.unwrap();
        assert!(
            !list.contains(2),
            "node {node} should not point to deleted vertex 2, got: {:?}",
            &*list
        );
        assert!(
            !list.contains(3),
            "node {node} should not point to deleted vertex 3, got: {:?}",
            &*list
        );
    }

    // Node 0: had [1, 4], neither deleted — should still have both.
    neighbors.get_neighbors(0, &mut list).await.unwrap();
    assert!(list.contains(1), "node 0 should still point to node 1");
    assert!(list.contains(4), "node 0 should still point to start (4)");

    // Node 1: had [0, 4], neither deleted — should still have both.
    neighbors.get_neighbors(1, &mut list).await.unwrap();
    assert!(list.contains(0), "node 1 should still point to node 0");
    assert!(list.contains(4), "node 1 should still point to start (4)");

    // Start node (4): had [0, 1, 2, 3], should now connect to only live corners.
    neighbors.get_neighbors(4, &mut list).await.unwrap();
    list.sort();
    assert_eq!(
        &*list,
        &[0, 1],
        "start node 4 should connect to remaining live corners"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn delete_isolated_node() {
    let adjacency_list = vec![
        AdjacencyList::from_iter_untrusted([1, 4]),
        AdjacencyList::from_iter_untrusted([0, 4]),
        AdjacencyList::from_iter_untrusted([]),
        AdjacencyList::from_iter_untrusted([4]),
        AdjacencyList::from_iter_untrusted([0, 1, 3]),
    ];

    let index = setup_2d_square(create_2d_unit_square(), adjacency_list, 4);
    let ctx = test_provider::Context::new();

    //capture state of neighbors pre-delete (ignoring node 2)
    let accessor = index.provider().neighbors();
    let mut list = AdjacencyList::new();
    let mut before: Vec<Vec<u32>> = Vec::new();

    for node in [0u32, 1, 3, 4] {
        accessor.get_neighbors(node, &mut list).await.unwrap();
        before.push(list.iter().copied().collect());
    }

    index
        .inplace_delete(
            test_provider::Strategy::new(),
            &ctx,
            &2,
            3,
            InplaceDeleteMethod::OneHop,
        )
        .await
        .unwrap(); // shouldn't panic

    for (i, node) in [0u32, 1, 3, 4].iter().enumerate() {
        accessor.get_neighbors(*node, &mut list).await.unwrap();
        let after: Vec<u32> = list.iter().copied().collect();
        assert_eq!(
            after, before[i],
            "node {node} neighbors should be unchanged"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn inplace_delete_two_hop_and_one_hop_wider_topology() {
    let start_id = u32::MAX;
    let index = setup_2d_square_using_synthetics_grid(3, start_id, 4);
    let ctx = test_provider::Context::new();

    index
        .inplace_delete(
            test_provider::Strategy::new(),
            &ctx,
            &4,
            3,
            InplaceDeleteMethod::TwoHopAndOneHop,
        )
        .await
        .unwrap();

    let mut accessor = index.provider().neighbors();
    let mut list = AdjacencyList::new();

    for node in 0u32..9 {
        if node == 4 {
            continue; // we deleted 4
        }

        accessor.get_neighbors(node, &mut list).await.unwrap();
        assert!(
            !list.contains(4),
            "node {node} should not reference 4, it's deleted"
        );
    }

    let reachable = index
        .count_reachable_nodes(&[start_id], &mut accessor)
        .await
        .unwrap();
    assert_eq!(reachable, 9, "8 data nodes + start should be reachable");
}

/// Multi-delete on the 3×3 grid with multi_thread to exercise parallel spawn + edge merge.
/// Deletes nodes [0, 4, 6] simultaneously, avoiding node 8 which is the start's only link.
#[tokio::test(flavor = "multi_thread")]
async fn multi_inplace_delete_wider_topology() {
    let start_id = u32::MAX;
    let index = setup_2d_square_using_synthetics_grid(3, start_id, 4);
    let ctx = test_provider::Context::new();

    let deleted: [u32; 3] = [0, 4, 6];
    index
        .multi_inplace_delete(
            test_provider::Strategy::new(),
            &ctx,
            Arc::new(deleted),
            3,
            InplaceDeleteMethod::TwoHopAndOneHop,
        )
        .await
        .unwrap();

    let mut accessor = index.provider().neighbors();
    let mut list = AdjacencyList::new();

    for node in 0u32..9 {
        if deleted.contains(&node) {
            continue; // deleted
        }
        accessor.get_neighbors(node, &mut list).await.unwrap();
        for &d in &deleted {
            assert!(
                !list.contains(d),
                "node {node} should not reference deleted node {d}"
            );
        }
    }

    let reachable = index
        .count_reachable_nodes(&[start_id], &mut accessor)
        .await
        .unwrap();
    assert_eq!(
        reachable, 7,
        "6 surviving data nodes + start should be reachable"
    );
}
