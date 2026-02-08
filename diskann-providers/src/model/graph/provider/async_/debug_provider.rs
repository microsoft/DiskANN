/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::graph::test::debug_provider as core;

use crate::model::FixedChunkPQTable;

pub use core::{
    AccessedInvalidId, Counter, Datum, DebugConfig, FullPrecision, Hybrid, HybridComputer,
    Internal, InvalidId, Panics, Quantized, Vector,
};

pub type DebugProvider = core::DebugProvider<FixedChunkPQTable>;
pub type DebugNeighborAccessor<'a> = core::DebugNeighborAccessor<'a, FixedChunkPQTable>;
pub type FullAccessor<'a> = core::FullAccessor<'a, FixedChunkPQTable>;
pub type QuantAccessor<'a> = core::QuantAccessor<'a, FixedChunkPQTable>;
pub type HybridAccessor<'a> = core::HybridAccessor<'a, FixedChunkPQTable>;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use diskann::{
        graph::{self, AdjacencyList, DiskANNIndex},
        provider::{
            Accessor, DataProvider, DefaultAccessor, DefaultContext, Delete, ElementStatus, Guard,
            NeighborAccessor, NeighborAccessorMut, SetElement,
        },
        utils::async_tools::VectorIdBoxSlice,
    };
    use diskann_vector::{PureDistanceFunction, distance::{Metric, SquaredL2}};
    use rstest::rstest;

    use super::*;
    use crate::{
        index::diskann_async::{
            tests::{
                GenerateGrid, PagedSearch, check_grid_search, populate_data, populate_graph, squish,
            },
            train_pq,
        },
        test_utils::groundtruth,
        utils,
    };

    #[tokio::test]
    async fn basic_operations() {
        let dim = 2;
        let ctx = &DefaultContext;

        let debug_config = DebugConfig {
            start_id: u32::MAX,
            start_point: vec![0.0; dim],
            max_degree: 10,
            metric: Metric::L2,
        };

        let vectors = [vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let pq_table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim), // Number of PQ chunks is bounded by the dimension.
            &mut crate::utils::create_rnd_from_seed_in_tests(0x04a8832604476965),
            1usize,
        )
        .unwrap();
        let provider = DebugProvider::new(debug_config, pq_table).unwrap();

        provider
            .set_element(ctx, &0, &[1.0, 1.0])
            .await
            .unwrap()
            .complete()
            .await;

        // internal id = external id
        assert_eq!(provider.to_internal_id(ctx, &0).unwrap(), 0);
        assert_eq!(provider.to_external_id(ctx, 0).unwrap(), 0);

        let mut accessor = FullAccessor::new(&provider);

        let res = accessor.get_element(0).await;
        assert!(res.is_ok());
        assert_eq!(provider.full_reads.get(), 1);

        let mut neighbors = AdjacencyList::new();

        let accessor = provider.default_accessor();
        let res = accessor.get_neighbors(0, &mut neighbors).await;
        assert!(res.is_ok());
        assert_eq!(provider.neighbor_reads.get(), 1);

        let accessor = provider.default_accessor();
        let res = accessor.set_neighbors(0, &[1, 2, 3]).await;
        assert!(res.is_ok());
        assert_eq!(provider.neighbor_writes.get(), 1);

        // delete and release vector 0
        let res = provider.delete(&DefaultContext, &0).await;
        assert!(res.is_ok());
        assert_eq!(
            ElementStatus::Deleted,
            provider
                .status_by_external_id(&DefaultContext, &0)
                .await
                .unwrap()
        );

        let mut accessor = FullAccessor::new(&provider);
        let res = accessor.get_element(0).await;
        assert!(res.is_ok());
        assert_eq!(provider.full_reads.get(), 2);

        let mut accessor = HybridAccessor::new(&provider);
        let res = accessor.get_element(0).await;
        assert!(res.is_ok());
        assert_eq!(provider.full_reads.get(), 3);

        // Releasing should make the element unreachable.
        let res = provider.release(&DefaultContext, 0).await;
        assert!(res.is_ok());
        assert!(
            provider
                .status_by_external_id(&DefaultContext, &0)
                .await
                .is_err()
        );
    }

    pub fn new_quant_index(
        index_config: graph::Config,
        debug_config: DebugConfig,
        pq_table: FixedChunkPQTable,
    ) -> Arc<DiskANNIndex<DebugProvider>> {
        let data = DebugProvider::new(debug_config, pq_table).unwrap();
        Arc::new(DiskANNIndex::new(index_config, data, None))
    }

    #[rstest]
    #[case(1, 100)]
    #[case(3, 7)]
    #[case(4, 5)]
    #[tokio::test]
    async fn grid_search(#[case] dim: usize, #[case] grid_size: usize) {
        let l = 10;
        let max_degree = 2 * dim;
        let num_points = (grid_size).pow(dim as u32);
        let start_id = u32::MAX;

        let index_config = graph::config::Builder::new(
            max_degree,
            graph::config::MaxDegree::default_slack(),
            l,
            (Metric::L2).into(),
        )
        .build()
        .unwrap();

        let debug_config = DebugConfig {
            start_id,
            start_point: vec![grid_size as f32; dim],
            max_degree,
            metric: Metric::L2,
        };

        let adjacency_lists = match dim {
            1 => utils::generate_1d_grid_adj_list(grid_size as u32),
            3 => utils::genererate_3d_grid_adj_list(grid_size as u32),
            4 => utils::generate_4d_grid_adj_list(grid_size as u32),
            _ => panic!("Unsupported number of dimensions"),
        };
        let mut vectors = f32::generate_grid(dim, grid_size);

        assert_eq!(adjacency_lists.len(), num_points);
        assert_eq!(vectors.len(), num_points);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim), // Number of PQ chunks is bounded by the dimension.
            &mut crate::utils::create_rnd_from_seed_in_tests(0x04a8832604476965),
            1usize,
        )
        .unwrap();

        let index = new_quant_index(index_config, debug_config, table);
        {
            let mut neighbor_accessor = index.provider().default_accessor();
            populate_data(index.provider(), &DefaultContext, &vectors).await;
            populate_graph(&mut neighbor_accessor, &adjacency_lists).await;

            // Set the adjacency list for the start point.
            neighbor_accessor
                .set_neighbors(start_id, &[num_points as u32 - 1])
                .await
                .unwrap();
        }

        // The corpus of actual vectors consists of all but the last point, which we use
        // as the start point.
        //
        // So, when we compute the corpus used during groundtruth generation, we take all
        // but this last point.
        let corpus: diskann_utils::views::Matrix<f32> =
            squish(vectors.iter().take(num_points), dim);

        let mut paged_tests = Vec::new();

        // Test with the zero query.
        let query = vec![0.0; dim];
        let gt = groundtruth(corpus.as_view(), &query, |a, b| SquaredL2::evaluate(a, b));
        paged_tests.push(PagedSearch::new(query, gt));

        // Test with the start point to ensure it is filtered out.
        let query = vectors.last().unwrap();
        let gt = groundtruth(corpus.as_view(), query, |a, b| SquaredL2::evaluate(a, b));
        paged_tests.push(PagedSearch::new(query.clone(), gt));

        // Unfortunately - this is needed for the `check_grid_search` test.
        vectors.push(index.provider().start_point_full().to_owned());
        check_grid_search(&index, &vectors, &paged_tests, FullPrecision, Quantized).await;
    }

    #[rstest]
    #[tokio::test]
    async fn grid_search_with_build(
        #[values((1, 100), (3, 7), (4, 5))] dim_and_size: (usize, usize),
    ) {
        let dim = dim_and_size.0;
        let grid_size = dim_and_size.1;
        let start_id = u32::MAX;

        let l = 10;

        // NOTE: Be careful changing `max_degree`. It needs to be high enough that the
        // graph is navigable, but low enough that the batch parallel handling inside
        // `multi_insert` is needed for the multi-insert graph to be navigable.
        //
        // With the current configured values, removing the other elements in the batch
        // from the visited set during `multi_insert` results in a graph failure.
        let max_degree = 2 * dim;

        let num_points = (grid_size).pow(dim as u32);

        let index_config = graph::config::Builder::new_with(
            max_degree,
            graph::config::MaxDegree::default_slack(),
            l,
            (Metric::L2).into(),
            |b| {
                b.max_minibatch_par(10);
            },
        )
        .build()
        .unwrap();

        let debug_config = DebugConfig {
            start_id,
            start_point: vec![grid_size as f32; dim],
            max_degree: index_config.max_degree().into(),
            metric: Metric::L2,
        };

        let mut vectors = f32::generate_grid(dim, grid_size);
        assert_eq!(vectors.len(), num_points);

        // This is a little subtle, but we need `vectors` to contain the start point as
        // its last element, but we **don't** want to include it in the index build.
        //
        // This basically means that we need to be careful with index initialization.
        vectors.push(vec![grid_size as f32; dim]);

        let table = train_pq(
            squish(vectors.iter(), dim).as_view(),
            2.min(dim), // Number of PQ chunks is bounded by the dimension.
            &mut crate::utils::create_rnd_from_seed_in_tests(0x04a8832604476965),
            1usize,
        )
        .unwrap();

        // Initialize an index for a new round of building.
        let init_index =
            || new_quant_index(index_config.clone(), debug_config.clone(), table.clone());

        // Build with full-precision single insert
        {
            let index = init_index();
            let ctx = DefaultContext;
            for (i, v) in vectors.iter().take(num_points).enumerate() {
                index
                    .insert(FullPrecision, &ctx, &(i as u32), v.as_slice())
                    .await
                    .unwrap();
            }

            // Ensure the `insert_search_accessor` API is invoked.
            assert_eq!(
                index.provider().insert_search_accessor_calls.get(),
                num_points,
                "insert should invoke `insert_search_accessor`",
            );

            check_grid_search(&index, &vectors, &[], FullPrecision, Quantized).await;
        }

        // Build with quantized single insert
        {
            let index = init_index();
            let ctx = DefaultContext;
            for (i, v) in vectors.iter().take(num_points).enumerate() {
                index
                    .insert(Quantized, &ctx, &(i as u32), v.as_slice())
                    .await
                    .unwrap();
            }

            // Ensure the `insert_search_accessor` API is invoked.
            assert_eq!(
                index.provider().insert_search_accessor_calls.get(),
                num_points,
                "insert should invoke `insert_search_accessor`",
            );

            check_grid_search(&index, &vectors, &[], FullPrecision, Quantized).await;
        }

        // Build with full-precision multi-insert
        {
            let index = init_index();
            let ctx = DefaultContext;
            let batch: Box<[_]> = vectors
                .iter()
                .take(num_points)
                .enumerate()
                .map(|(id, v)| VectorIdBoxSlice::new(id as u32, v.as_slice().into()))
                .collect();

            index
                .multi_insert(FullPrecision, &ctx, batch)
                .await
                .unwrap();

            // Ensure the `insert_search_accessor` API is invoked.
            assert_eq!(
                index.provider().insert_search_accessor_calls.get(),
                num_points,
                "multi-insert should invoke `insert_search_accessor`",
            );

            check_grid_search(&index, &vectors, &[], FullPrecision, Quantized).await;
        }

        // Build with quantized multi-insert
        {
            let index = init_index();
            let ctx = DefaultContext;
            let batch: Box<[_]> = vectors
                .iter()
                .take(num_points)
                .enumerate()
                .map(|(id, v)| VectorIdBoxSlice::new(id as u32, v.as_slice().into()))
                .collect();

            index.multi_insert(Quantized, &ctx, batch).await.unwrap();

            // Ensure the `insert_search_accessor` API is invoked.
            assert_eq!(
                index.provider().insert_search_accessor_calls.get(),
                num_points,
                "multi-insert should invoke `insert_search_accessor`",
            );

            check_grid_search(&index, &vectors, &[], FullPrecision, Quantized).await;
        }
    }
}
