/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::future::Future;

use super::helpers::{generate_2d_square_adjacency_list, setup_2d_square};
use crate::{
    graph::{self, AdjacencyList, index::DegreeStats, test::provider as test_provider},
    provider::{Delete, NeighborAccessor},
    test::cmp::assert_eq_verbose,
};

#[tokio::test(flavor = "current_thread")]
async fn test_count_reachable_nodes() {
    let adjacency_list = generate_2d_square_adjacency_list();
    let index = setup_2d_square(adjacency_list, 4);
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
    let index = setup_2d_square(adjacency_list, 1);
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
    let index = setup_2d_square(adjacency_list, 4);
    let mut accessor = index.provider().neighbors();
    let stats = index
        .get_degree_stats(&mut accessor, index.provider().non_start_points_ids())
        .await
        .unwrap();
    let expected = DegreeStats {
        max_degree: 2,
        avg_degree: 2.0,
        min_degree: 2,
        cnt_less_than_two: 0,
    };
    assert_eq_verbose!(expected, stats);
}

#[tokio::test(flavor = "current_thread")]
async fn test_prune_range() {
    let adjacency_list = generate_2d_square_adjacency_list();
    let index = setup_2d_square(adjacency_list, 1);
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
    let index = setup_2d_square(adjacency_list, 4);
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
    let index = setup_2d_square(adjacency_list, 4);
    let mut accessor = index.provider().neighbors();
    let stats = index
        .get_degree_stats(&mut accessor, index.provider().non_start_points_ids())
        .await
        .unwrap();

    let expected = DegreeStats {
        max_degree: 2,
        avg_degree: 1.25,
        min_degree: 0,
        cnt_less_than_two: 2,
    };
    assert_eq_verbose!(expected, stats);
}

#[tokio::test(flavor = "current_thread")]
async fn test_is_any_neighbor_deleted() {
    let adjacency_list = generate_2d_square_adjacency_list();
    let index = setup_2d_square(adjacency_list, 4);
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

#[tokio::test(flavor = "current_thread")]
async fn test_drop_deleted_neighbors() {
    let adjacency_list = generate_2d_square_adjacency_list();
    let index = setup_2d_square(adjacency_list, 4);
    let ctx = test_provider::Context::new();
    let mut accessor = index.provider().neighbors();

    // Delete node 3, then drop deleted neighbors from node 2 (neighbors: [3, 4]).
    index.provider().delete(&ctx, &3).await.unwrap();

    let result = index
        .drop_deleted_neighbors(&ctx, &mut accessor, 2, false)
        .await
        .unwrap();
    assert_eq!(result, graph::ConsolidateKind::Complete);

    // Node 2 should no longer reference deleted node 3.
    let mut list = AdjacencyList::new();
    accessor.get_neighbors(2, &mut list).await.unwrap();
    assert!(
        !list.contains(3),
        "node 2 should not reference deleted node 3"
    );
    assert!(
        list.contains(4),
        "node 2 should still reference start node 4"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_drop_deleted_neighbors_only_orphans() {
    let adjacency_list = generate_2d_square_adjacency_list();
    let index = setup_2d_square(adjacency_list, 4);
    let ctx = test_provider::Context::new();
    let mut accessor = index.provider().neighbors();

    // Delete node 3 but don't clear its adjacency list — it's not an orphan.
    index.provider().delete(&ctx, &3).await.unwrap();

    let result = index
        .drop_deleted_neighbors(&ctx, &mut accessor, 2, true)
        .await
        .unwrap();
    assert_eq!(result, graph::ConsolidateKind::Complete);

    // With only_orphans=true, node 3 still has a non-empty adjacency list,
    // so it should be kept in node 2's neighbor list.
    let mut list = AdjacencyList::new();
    accessor.get_neighbors(2, &mut list).await.unwrap();
    assert!(
        list.contains(3),
        "non-orphan deleted neighbor should be retained"
    );
    assert!(
        list.contains(4),
        "node 2 should still reference start node 4"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn test_drop_deleted_neighbors_noop() {
    let adjacency_list = generate_2d_square_adjacency_list();
    let index = setup_2d_square(adjacency_list, 4);
    let ctx = test_provider::Context::new();
    let mut accessor = index.provider().neighbors();

    // No deletions — should be a no-op.
    let result = index
        .drop_deleted_neighbors(&ctx, &mut accessor, 0, false)
        .await
        .unwrap();
    assert_eq!(result, graph::ConsolidateKind::Complete);
}

fn zero_runtime_block_on<F: Future>(fut: F) -> F::Output {
    use std::{
        pin::pin,
        task::{Context, Poll, Waker},
        thread,
    };

    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut fut = pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => thread::yield_now(),
        }
    }
}

/// End-to-end multi_insert and search running without ANY async runtime.
///
/// This serves as an end-to-end proof that the DiskANN core algorithm is fully runtime-agnostic.
#[test]
fn test_zero_runtime_multi_insert_and_search() {
    use std::sync::Arc;

    use diskann_vector::distance::Metric;

    use crate::{
        graph::{
            DiskANNIndex,
            config::{Builder, IntraBatchCandidates, MaxDegree},
            search::Knn,
            test::synthetic::Grid,
        },
        neighbor::{BackInserter, Neighbor},
    };

    let grid = Grid::Two;
    let size = 3;
    let max_degree = 8;
    let start_vector = grid.start_point(size);

    let config = test_provider::Config::new(
        Metric::L2,
        max_degree,
        test_provider::StartPoint::new(u32::MAX, start_vector),
    )
    .unwrap();

    let provider = test_provider::Provider::new(config);
    let index_config =
        Builder::new_with(4, MaxDegree::new(max_degree), 50, Metric::L2.into(), |b| {
            b.intra_batch_candidates(IntraBatchCandidates::None)
                .max_minibatch_par(2);
        })
        .build()
        .unwrap();

    let index = Arc::new(DiskANNIndex::new(index_config, provider, None));
    let strategy = test_provider::Strategy::new();
    let context = test_provider::Context::new();

    let data = grid.data(size);
    let batch = Arc::new(data.to_owned());
    let ids: Arc<[u32]> = (0..data.nrows() as u32).collect();

    // Multi-insert driven entirely by zero_runtime_block_on without any Tokio runtime!
    zero_runtime_block_on(index.multi_insert::<test_provider::Strategy, _>(
        strategy.clone(),
        &context,
        batch,
        ids,
    ))
    .unwrap();

    // Search query close to node 0
    let query = data.row(0);
    let knn = Knn::new(5, None).unwrap();
    let mut neighbors = vec![Neighbor::<u32>::default(); 5];
    let stats = zero_runtime_block_on(index.search(
        knn,
        &strategy,
        &context,
        query,
        &mut BackInserter::new(neighbors.as_mut_slice()),
    ))
    .unwrap();

    assert!(stats.result_count > 0);
    assert_eq!(*neighbors[0].id(), 0);
}
