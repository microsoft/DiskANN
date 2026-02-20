/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use super::{StorageReadProvider, StorageWriteProvider};
use diskann::{
    ANNError, ANNResult, graph::DiskANNIndex, provider::DataProvider, utils::VectorRepr,
};
use diskann_utils::future::AsyncFriendly;

use super::{AsyncIndexMetadata, AsyncQuantLoadContext, DiskGraphOnly, LoadWith, SaveWith};
use crate::model::{
    configuration::IndexConfiguration,
    graph::provider::async_::{
        FastMemoryQuantVectorProviderAsync, TableDeleteProviderAsync, common,
        inmem::{self, DefaultProvider, FullPrecisionStore},
    },
};

impl<U, V, D> SaveWith<AsyncIndexMetadata> for DiskANNIndex<DefaultProvider<U, V, D>>
where
    U: AsyncFriendly,
    V: AsyncFriendly,
    D: AsyncFriendly,
    DefaultProvider<U, V, D>: SaveWith<(u32, AsyncIndexMetadata), Error = ANNError>,
{
    type Ok = ();
    type Error = ANNError;

    async fn save_with<P>(&self, provider: &P, ctx_prefix: &AsyncIndexMetadata) -> ANNResult<()>
    where
        P: StorageWriteProvider,
    {
        let start_id = get_and_validate_single_starting_point(&self.data_provider)?;

        self.data_provider
            .save_with(provider, &(start_id, ctx_prefix.clone()))
            .await?;

        Ok(())
    }
}

// This implementation saves only graph and not the vector/quant data.
impl<U, V, D> SaveWith<(u32, DiskGraphOnly)> for DiskANNIndex<DefaultProvider<U, V, D>>
where
    U: AsyncFriendly,
    V: AsyncFriendly,
    D: AsyncFriendly,
    DefaultProvider<U, V, D>: SaveWith<(u32, u32, DiskGraphOnly), Error = ANNError>,
{
    type Ok = ();
    type Error = ANNError;

    async fn save_with<P>(&self, provider: &P, ctx_prefix: &(u32, DiskGraphOnly)) -> ANNResult<()>
    where
        P: StorageWriteProvider,
    {
        let start_id = get_and_validate_single_starting_point(&self.data_provider)?;

        self.data_provider
            .save_with(provider, &(start_id, ctx_prefix.0, ctx_prefix.1.clone()))
            .await?;
        Ok(())
    }
}

/// Creates a `AsyncQuantLoadContext` from an `IndexConfiguration` with the given path and disk index flag.
pub fn create_load_context(
    path: &str,
    index_config: &IndexConfiguration,
    is_disk_index: bool,
) -> ANNResult<AsyncQuantLoadContext> {
    Ok(AsyncQuantLoadContext {
        metadata: AsyncIndexMetadata::new(path),
        num_frozen_points: index_config.num_frozen_pts,
        metric: index_config.dist_metric,
        prefetch_lookahead: index_config.prefetch_lookahead.map(|x| x.get()),
        is_disk_index,
        prefetch_cache_line_level: index_config.prefetch_cache_line_level,
    })
}

impl<'a, DP> LoadWith<(&'a str, IndexConfiguration)> for DiskANNIndex<DP>
where
    DP: DataProvider<InternalId = u32> + LoadWith<AsyncQuantLoadContext, Error = ANNError>,
{
    type Error = ANNError;
    async fn load_with<P>(
        provider: &P,
        (path, index_config): &(&'a str, IndexConfiguration),
    ) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        let pq_context = create_load_context(path, index_config, false)?;

        let data_provider = DP::load_with(provider, &pq_context).await?;
        let num_threads = index_config.num_threads;
        Ok(Self::new(
            index_config.config.clone(),
            data_provider,
            NonZeroUsize::new(num_threads),
        ))
    }
}

pub async fn load_pq_index<T, P>(
    provider: &P,
    path: &str,
    config: IndexConfiguration,
) -> ANNResult<DiskANNIndex<inmem::FullPrecisionProvider<T, FastMemoryQuantVectorProviderAsync>>>
where
    P: StorageReadProvider,
    T: VectorRepr,
{
    DiskANNIndex::load_with(provider, &(path, config)).await
}

pub async fn load_pq_index_with_deletes<T, P>(
    provider: &P,
    path: &str,
    config: IndexConfiguration,
) -> ANNResult<
    DiskANNIndex<
        inmem::DefaultProvider<
            FullPrecisionStore<T>,
            FastMemoryQuantVectorProviderAsync,
            TableDeleteProviderAsync,
        >,
    >,
>
where
    P: StorageReadProvider,
    T: VectorRepr,
{
    DiskANNIndex::load_with(provider, &(path, config)).await
}

pub async fn load_fp_index<T, P, Q>(
    provider: &P,
    path: &str,
    config: IndexConfiguration,
) -> ANNResult<DiskANNIndex<inmem::FullPrecisionProvider<T, Q>>>
where
    P: StorageReadProvider,
    T: VectorRepr,
    Q: AsyncFriendly,
    inmem::FullPrecisionProvider<T, Q>: LoadWith<AsyncQuantLoadContext, Error = ANNError>,
{
    DiskANNIndex::load_with(provider, &(path, config)).await
}

pub async fn load_index<P, U, V>(
    provider: &P,
    path: &str,
    config: IndexConfiguration,
) -> ANNResult<DiskANNIndex<inmem::DefaultProvider<U, V>>>
where
    P: StorageReadProvider,
    U: AsyncFriendly,
    V: AsyncFriendly,
    inmem::DefaultProvider<U, V>: LoadWith<AsyncQuantLoadContext, Error = ANNError>,
{
    DiskANNIndex::load_with(provider, &(path, config)).await
}

pub async fn load_index_with_deletes<T, P>(
    provider: &P,
    path: &str,
    config: IndexConfiguration,
) -> ANNResult<
    DiskANNIndex<inmem::FullPrecisionProvider<T, common::NoStore, TableDeleteProviderAsync>>,
>
where
    P: StorageReadProvider,
    T: VectorRepr,
{
    DiskANNIndex::load_with(provider, &(path, config)).await
}

/// Retrieves starting points and enforces that there is exactly one starting point.
///
/// This helper function:
/// 1. Retrieves the starting points from the data provider
/// 2. Validates there is exactly one starting point
/// 3. Returns the single start point if valid
///
/// Returns an error if there are multiple starting points or no starting points.
fn get_and_validate_single_starting_point<U, V, D>(
    data_provider: &DefaultProvider<U, V, D>,
) -> ANNResult<u32> {
    let start_ids = data_provider.starting_points()?;

    let num_starting_points = start_ids.len();
    if num_starting_points > 1 {
        return Err(ANNError::log_index_error(format_args!(
            "ERROR: Save index does not support multiple starting points. Found {} starting points.",
            num_starting_points
        )));
    }

    start_ids
        .first()
        .cloned()
        .ok_or_else(|| ANNError::log_index_error("ERROR: No starting points found"))
}
///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::{num::NonZeroUsize, sync::Arc};

    use crate::storage::VirtualStorageProvider;
    use diskann::{
        graph::{AdjacencyList, config, glue::InsertStrategy},
        provider::{Accessor, SetElement},
        utils::{IntoUsize, ONE},
    };
    use diskann_utils::{
        Reborrow, test_data_root,
        views::{Matrix, MatrixView},
    };
    use diskann_vector::distance::Metric;

    use super::*;
    use crate::{
        index::diskann_async::{self, MemoryIndex},
        model::graph::provider::async_::{
            SimpleNeighborProviderAsync,
            common::{FullPrecision, NoDeletes, NoStore, TableBasedDeletes},
            inmem::{self},
        },
        utils::{create_rnd_from_seed_in_tests, file_util},
    };

    async fn build_index<DP, S>(
        index: &Arc<DiskANNIndex<DP>>,
        strategy: S,
        data: MatrixView<'_, f32>,
    ) where
        DP: DataProvider<ExternalId = u32> + SetElement<[f32]>,
        DP::Context: Default,
        S: InsertStrategy<DP, [f32]> + Clone,
    {
        let ctx = &DP::Context::default();
        for (i, v) in data.row_iter().enumerate() {
            index
                .insert(strategy.clone(), ctx, &(i as u32), v)
                .await
                .unwrap();
        }
    }

    // Our test strategy here is to basically build one main index using quantization
    // and to save that.
    //
    // We will the try reloading with the following flavors:
    // 1. Without quant, with delete set.
    // 2. Without quant, without delete set.
    // 3. With quant, with delete set.
    // 4. With quant, without delete set.
    #[tokio::test]
    async fn test_save_and_load() {
        let save_path = "/index";
        let file_path = "/sift/siftsmall_learn_256pts.fbin";
        let train_data = {
            let storage = VirtualStorageProvider::new_overlay(test_data_root());
            let (train_data, npoints, dim) = file_util::load_bin(&storage, file_path, 0).unwrap();
            Matrix::<f32>::try_from(train_data.into(), npoints, dim).unwrap()
        };

        let pq_bytes = 8;
        let pq_table = diskann_async::train_pq(
            train_data.as_view(),
            pq_bytes,
            &mut create_rnd_from_seed_in_tests(0xe3c52ef001bc7ade),
            2,
        )
        .unwrap();

        let (config, parameters) = diskann_async::simplified_builder(
            20,
            32,
            Metric::L2,
            train_data.ncols(),
            train_data.nrows(),
            |_| {},
        )
        .unwrap();

        let index = diskann_async::new_quant_index::<f32, _, _>(
            config,
            parameters,
            pq_table,
            TableBasedDeletes,
        )
        .unwrap();

        build_index(&index, FullPrecision, train_data.as_view()).await;

        // Check that all nodes are reachable.
        {
            let count = index
                .count_reachable_nodes(
                    &index.provider().starting_points().unwrap(),
                    &mut index.provider().neighbors(),
                )
                .await
                .unwrap();
            assert_eq!(count, train_data.nrows() + 1);
        }

        // Save the resulting index.
        let provider = VirtualStorageProvider::new_memory();
        index
            .save_with(&provider, &AsyncIndexMetadata::new(save_path.to_string()))
            .await
            .unwrap();

        // Convert into the full index configuration.
        let config = IndexConfiguration::new(
            Metric::L2,
            train_data.ncols(),
            train_data.nrows(),
            ONE,
            1,
            config::Builder::new(
                30,
                config::MaxDegree::default_slack(),
                20,
                Metric::L2.into(),
            )
            .build()
            .unwrap(),
        );

        let id_iter = index.data_provider.iter();

        // Without Quant, With Delete Set.
        {
            let reloaded = load_index_with_deletes::<f32, _>(&provider, save_path, config.clone())
                .await
                .unwrap();

            assert_eq!(id_iter, reloaded.data_provider.iter());
            check_accessor_equal(
                inmem::FullAccessor::new(index.provider()),
                inmem::FullAccessor::new(reloaded.provider()),
                id_iter.clone(),
            )
            .await;

            check_graphs_equal(
                &index.provider().neighbor_provider,
                &reloaded.provider().neighbor_provider,
                id_iter.clone(),
            )
        }

        // Without Quant, Without Delete Set.
        {
            let reloaded = load_fp_index::<f32, _, NoStore>(&provider, save_path, config.clone())
                .await
                .unwrap();

            assert_eq!(id_iter, reloaded.data_provider.iter());
            check_accessor_equal(
                inmem::FullAccessor::new(index.provider()),
                inmem::FullAccessor::new(reloaded.provider()),
                id_iter.clone(),
            )
            .await;

            check_graphs_equal(
                &index.provider().neighbor_provider,
                &reloaded.provider().neighbor_provider,
                id_iter.clone(),
            )
        }

        // With Quant, With Delete Set.
        {
            let reloaded =
                load_pq_index_with_deletes::<f32, _>(&provider, save_path, config.clone())
                    .await
                    .unwrap();

            assert_eq!(id_iter, reloaded.data_provider.iter());
            check_accessor_equal(
                inmem::FullAccessor::new(index.provider()),
                inmem::FullAccessor::new(reloaded.provider()),
                index.data_provider.iter(),
            )
            .await;

            check_accessor_equal(
                inmem::product::QuantAccessor::new(index.provider()),
                inmem::product::QuantAccessor::new(reloaded.provider()),
                index.data_provider.iter(),
            )
            .await;

            check_graphs_equal(
                &index.provider().neighbor_provider,
                &reloaded.provider().neighbor_provider,
                id_iter.clone(),
            )
        }

        // With Quant, Without Delete Set.
        {
            let reloaded = load_pq_index::<f32, _>(&provider, save_path, config.clone())
                .await
                .unwrap();

            assert_eq!(id_iter, reloaded.data_provider.iter());
            check_accessor_equal(
                inmem::FullAccessor::new(index.provider()),
                inmem::FullAccessor::new(reloaded.provider()),
                index.data_provider.iter(),
            )
            .await;

            check_accessor_equal(
                inmem::product::QuantAccessor::new(index.provider()),
                inmem::product::QuantAccessor::new(reloaded.provider()),
                index.data_provider.iter(),
            )
            .await;

            check_graphs_equal(
                &index.provider().neighbor_provider,
                &reloaded.provider().neighbor_provider,
                id_iter.clone(),
            )
        }
    }

    async fn check_accessor_equal<T, A, B, Itr>(mut left: A, mut right: B, itr: Itr)
    where
        A: for<'a> Accessor<Id = u32, ElementRef<'a> = &'a T>,
        B: for<'a> Accessor<Id = u32, ElementRef<'a> = &'a T>,
        T: PartialEq + std::fmt::Debug + ?Sized,
        Itr: Iterator<Item = u32>,
    {
        for i in itr {
            assert_eq!(
                left.get_element(i).await.unwrap().reborrow(),
                right.get_element(i).await.unwrap().reborrow(),
                "failed for index {}",
                i
            );
        }
    }

    fn check_graphs_equal<Itr>(
        left: &SimpleNeighborProviderAsync<u32>,
        right: &SimpleNeighborProviderAsync<u32>,
        itr: Itr,
    ) where
        Itr: Iterator<Item = u32>,
    {
        let mut lv = AdjacencyList::new();
        let mut rv = AdjacencyList::new();
        for i in itr {
            left.get_neighbors_sync(i.into_usize(), &mut lv).unwrap();
            right.get_neighbors_sync(i.into_usize(), &mut rv).unwrap();
            assert_eq!(lv, rv, "failed for index {}", i);
        }
    }

    fn create_test_index(num_start_points: usize) -> MemoryIndex<f32> {
        let (config, mut parameters) =
            diskann_async::simplified_builder(20, 32, Metric::L2, 3, 5, |_| {}).unwrap();

        parameters.frozen_points = NonZeroUsize::new(num_start_points).unwrap();
        diskann_async::new_index::<f32, _>(config, parameters, NoDeletes).unwrap()
    }

    #[tokio::test]
    async fn test_validate_single_starting_point() {
        // Test case 1: Single start point should succeed
        {
            let index = create_test_index(1);
            let result = get_and_validate_single_starting_point(&index.data_provider);
            assert!(result.is_ok(), "Failed to validate single start point");
        }

        // Test case 2: Multiple start points should fail
        {
            let index = create_test_index(2);
            let result = get_and_validate_single_starting_point(&index.data_provider);
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("not support multiple starting points")
            );
        }
    }
}
