/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use super::{StorageReadProvider, StorageWriteProvider};
use diskann::{
    ANNError, ANNResult, graph::DiskANNIndex, provider::DataProvider, utils::VectorRepr,
};
use diskann_utils::{future::AsyncFriendly, lazy_format};

use super::{
    AsyncIndexMetadata, AsyncQuantLoadContext, DiskGraphOnly, LoadWith, NativeStaticLoadContext,
    SaveWith,
};
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
        if self.data_provider.is_native_static() {
            return Err(ANNError::message("native static graphs are read-only"));
        }
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
        if self.data_provider.is_native_static() {
            return Err(ANNError::message("native static graphs are read-only"));
        }
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

/// Select the read-only native static layout with N graph records and N data vectors.
///
/// Its serialized `frozen=1` header is a compatibility marker, not an extra
/// physical vector. The ordinary [`IndexConfiguration`] loader continues to
/// interpret an appended start point as frozen.
pub struct NativeStaticIndexConfiguration {
    config: IndexConfiguration,
}

impl NativeStaticIndexConfiguration {
    /// Select the native static layout for an existing index configuration.
    ///
    /// # Errors
    ///
    /// Returns an error unless the configuration's frozen-point count is exactly one.
    pub fn new(config: IndexConfiguration) -> ANNResult<Self> {
        if config.num_frozen_pts.get() != 1 {
            return Err(ANNError::message(
                "native static graph requires exactly one serialized fake frozen point",
            ));
        }
        Ok(Self { config })
    }
}

impl<'a, DP> LoadWith<(&'a str, NativeStaticIndexConfiguration)> for DiskANNIndex<DP>
where
    DP: DataProvider<InternalId = u32> + LoadWith<NativeStaticLoadContext, Error = ANNError>,
{
    type Error = ANNError;

    async fn load_with<P>(
        provider: &P,
        (path, config): &(&'a str, NativeStaticIndexConfiguration),
    ) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        let context = NativeStaticLoadContext {
            inner: create_load_context(path, &config.config, false)?,
        };
        let data_provider = DP::load_with(provider, &context).await?;
        Ok(Self::new(
            config.config.config.clone(),
            data_provider,
            NonZeroUsize::new(config.config.num_threads),
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
        return Err(ANNError::message(lazy_format!(
            move,
            "ERROR: Save index does not support multiple starting points. Found {} starting points.",
            num_starting_points
        )));
    }

    start_ids
        .first()
        .cloned()
        .ok_or_else(|| ANNError::message("ERROR: No starting points found"))
}
///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::{io::Write, num::NonZeroUsize, sync::Arc};

    use crate::storage::VirtualStorageProvider;
    use diskann::{
        graph::{AdjacencyList, config, glue::InsertStrategy},
        provider::{DefaultContext, SetElement},
        utils::{IntoUsize, ONE},
    };
    use diskann_utils::{test_data_root, views::MatrixView};
    use diskann_vector::distance::Metric;

    use super::*;
    use crate::{
        index::diskann_async::{self, MemoryIndex},
        model::graph::provider::async_::{
            SimpleNeighborProviderAsync,
            common::{FullPrecision, NoDeletes, NoStore, TableBasedDeletes},
            inmem::SetStartPoints,
        },
        utils::create_rnd_from_seed_in_tests,
    };

    async fn build_index<DP, S>(
        index: &Arc<DiskANNIndex<DP>>,
        strategy: S,
        data: MatrixView<'_, f32>,
    ) where
        DP: DataProvider<ExternalId = u32> + for<'a> SetElement<&'a [f32]>,
        DP::Context: Default,
        S: for<'a> InsertStrategy<'a, DP, &'a [f32]> + Clone,
    {
        let ctx = &DP::Context::default();
        for (i, v) in data.row_iter().enumerate() {
            index.insert(&strategy, ctx, &(i as u32), v).await.unwrap();
        }
    }

    #[test]
    fn native_static_configuration_rejects_extra_frozen_points() {
        let graph_config =
            config::Builder::new(32, config::MaxDegree::same(), 100, Metric::L2.into())
                .build()
                .unwrap();
        for frozen in [1, 2] {
            let index_config = IndexConfiguration::new(
                Metric::L2,
                64,
                256,
                NonZeroUsize::new(frozen).unwrap(),
                1,
                graph_config.clone(),
            );
            assert_eq!(
                NativeStaticIndexConfiguration::new(index_config).is_ok(),
                frozen == 1
            );
        }
    }

    #[tokio::test]
    async fn native_static_load_keeps_last_vector_and_medoid_searchable() {
        let path = "/native_static";
        let storage = VirtualStorageProvider::new_memory();
        let graph = SimpleNeighborProviderAsync::new(2, 1, 2, 1.0);
        graph.set_neighbors_sync(0, &[1]).unwrap();
        graph.set_neighbors_sync(1, &[0, 2]).unwrap();
        graph.set_neighbors_sync(2, &[1]).unwrap();
        graph.save_direct(&storage, 1, path).unwrap();
        {
            let mut data = storage.create_for_write("/native_static.data").unwrap();
            data.write_all(&3_u32.to_le_bytes()).unwrap();
            data.write_all(&2_u32.to_le_bytes()).unwrap();
            data.write_all(&[0_u8, 0, 4, 4, 8, 8]).unwrap();
        }

        let config = IndexConfiguration::new(
            Metric::L2,
            2,
            3,
            ONE,
            1,
            config::Builder::new(32, config::MaxDegree::same(), 100, Metric::L2.into())
                .build()
                .unwrap(),
        );
        let legacy = DiskANNIndex::<inmem::FullPrecisionProvider<u8>>::load_with(
            &storage,
            &(path, config.clone()),
        )
        .await
        .unwrap();
        assert_eq!(legacy.provider().capacity(), 2);

        let mut invalid_frozen = create_load_context(path, &config, false).unwrap();
        invalid_frozen.num_frozen_points = NonZeroUsize::new(2).unwrap();
        assert!(
            inmem::FullPrecisionProvider::<u8>::load_with(
                &storage,
                &NativeStaticLoadContext {
                    inner: invalid_frozen,
                },
            )
            .await
            .is_err()
        );

        let index = DiskANNIndex::<inmem::FullPrecisionProvider<u8>>::load_with(
            &storage,
            &(
                path,
                NativeStaticIndexConfiguration::new(config.clone()).unwrap(),
            ),
        )
        .await
        .unwrap();
        assert_eq!(index.provider().capacity(), 3);
        assert_eq!(index.provider().total_points(), 3);
        assert_eq!(index.provider().starting_points().unwrap(), vec![1]);
        assert!(index.provider().is_native_static());
        assert!((index.provider().is_not_frozen())(1));
        assert!((index.provider().is_not_frozen())(2));

        for (query, expected_id) in [([8_u8, 8], 2), ([4, 4], 1)] {
            let mut ids = [u32::MAX; 1];
            let mut distances = [f32::NAN; 1];
            let mut output =
                diskann::graph::search_output_buffer::IdDistance::new(&mut ids, &mut distances);
            let result = index
                .search(
                    diskann::graph::search::Knn::new_default(3).unwrap(),
                    &FullPrecision,
                    &DefaultContext,
                    &query,
                    &mut output,
                )
                .await
                .unwrap();
            assert_eq!(result.result_count, 1);
            assert_eq!(ids[0], expected_id);
        }

        assert!(
            index
                .provider()
                .set_element(&DefaultContext, &2, &[8_u8, 8])
                .await
                .is_err()
        );
        assert!(
            index
                .provider()
                .set_start_points(std::iter::once(&[0_u8, 0][..]))
                .is_err()
        );
        assert!(
            index
                .provider()
                .neighbors()
                .set_neighbors_sync(2, &[0])
                .is_err()
        );
        assert!(
            index
                .provider()
                .neighbors()
                .append_vector_sync(2, &[0])
                .is_err()
        );
        assert!(
            index
                .provider()
                .neighbors()
                .save_direct(&storage, 1, "/blocked_graph")
                .is_err()
        );
        assert!(!storage.exists("/blocked_graph"));
        assert!(
            index
                .provider()
                .save_with(&storage, &(1, AsyncIndexMetadata::new("/blocked_provider")))
                .await
                .is_err()
        );
        assert!(!storage.exists("/blocked_provider"));
        assert!(
            index
                .save_with(&storage, &AsyncIndexMetadata::new("/blocked_index"))
                .await
                .is_err()
        );
        assert!(!storage.exists("/blocked_index"));

        storage
            .open_writer(path)
            .unwrap()
            .write_all(&0_u32.to_le_bytes())
            .unwrap();
        let err = DiskANNIndex::<inmem::FullPrecisionProvider<u8>>::load_with(
            &storage,
            &(path, NativeStaticIndexConfiguration::new(config).unwrap()),
        )
        .await
        .err()
        .unwrap();
        assert!(err.to_string().contains("complete file"));
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
            let mut reader = storage.open_reader(file_path).unwrap();
            diskann_utils::io::read_bin::<f32>(&mut reader).unwrap()
        };

        let pq_bytes = 8;
        let pq_table = diskann_async::train_pq(
            train_data.as_view(),
            pq_bytes,
            &mut create_rnd_from_seed_in_tests(0xe3c52ef001bc7ade),
            crate::utils::create_thread_pool(2).unwrap().as_ref(),
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
            index
                .provider()
                .base_vectors
                .compare_data(&reloaded.provider().base_vectors);

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
            index
                .provider()
                .base_vectors
                .compare_data(&reloaded.provider().base_vectors);

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
            index
                .provider()
                .base_vectors
                .compare_data(&reloaded.provider().base_vectors);
            index
                .provider()
                .aux_vectors
                .compare_data(&reloaded.provider().aux_vectors);

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
            index
                .provider()
                .base_vectors
                .compare_data(&reloaded.provider().base_vectors);
            index
                .provider()
                .aux_vectors
                .compare_data(&reloaded.provider().aux_vectors);

            check_graphs_equal(
                &index.provider().neighbor_provider,
                &reloaded.provider().neighbor_provider,
                id_iter.clone(),
            )
        }
    }

    fn check_graphs_equal<Itr>(
        left: &SimpleNeighborProviderAsync,
        right: &SimpleNeighborProviderAsync,
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
