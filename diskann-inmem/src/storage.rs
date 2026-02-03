/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Storage implementations for inmem providers.

use std::num::NonZeroUsize;

use diskann::{
    ANNError, ANNResult, graph::DiskANNIndex, provider::DataProvider, utils::VectorRepr,
};
use diskann_utils::future::AsyncFriendly;
use diskann_providers::storage::{
    StorageReadProvider, StorageWriteProvider, AsyncIndexMetadata, AsyncQuantLoadContext,
    DiskGraphOnly, LoadWith, SaveWith,
};
use diskann_providers::model::{
    configuration::IndexConfiguration,
    graph::provider::async_::{
        FastMemoryQuantVectorProviderAsync, TableDeleteProviderAsync, common,
    },
};

use crate::{DefaultProvider, FullPrecisionStore};

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
) -> ANNResult<DiskANNIndex<crate::FullPrecisionProvider<T, FastMemoryQuantVectorProviderAsync>>>
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
        crate::DefaultProvider<
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
) -> ANNResult<DiskANNIndex<crate::FullPrecisionProvider<T, Q>>>
where
    P: StorageReadProvider,
    T: VectorRepr,
    Q: AsyncFriendly,
    crate::FullPrecisionProvider<T, Q>: LoadWith<AsyncQuantLoadContext, Error = ANNError>,
{
    DiskANNIndex::load_with(provider, &(path, config)).await
}

pub async fn load_index<P, U, V>(
    provider: &P,
    path: &str,
    config: IndexConfiguration,
) -> ANNResult<DiskANNIndex<crate::DefaultProvider<U, V>>>
where
    P: StorageReadProvider,
    U: AsyncFriendly,
    V: AsyncFriendly,
    crate::DefaultProvider<U, V>: LoadWith<AsyncQuantLoadContext, Error = ANNError>,
{
    DiskANNIndex::load_with(provider, &(path, config)).await
}

pub async fn load_index_with_deletes<T, P>(
    provider: &P,
    path: &str,
    config: IndexConfiguration,
) -> ANNResult<
    DiskANNIndex<crate::FullPrecisionProvider<T, common::NoStore, TableDeleteProviderAsync>>,
>
where
    P: StorageReadProvider,
    T: VectorRepr,
{
    DiskANNIndex::load_with(provider, &(path, config)).await
}

/// Extracts and validates single start point from DefaultProvider.
///
/// # Errors
/// - Returns an error if the number of start points is not exactly 1
fn get_and_validate_single_starting_point<U, V, D>(
    data_provider: &DefaultProvider<U, V, D>,
) -> ANNResult<u32> {
    use diskann::provider::DataProvider;
    use diskann::ANNErrorKind;

    let start_points: Vec<u32> = data_provider.start_points();
    if start_points.len() != 1 {
        return Err(ANNError::log_error(ANNErrorKind::InvalidStartPoint, format!(
            "Index must have exactly 1 start point for saving. Found {} start points: {:?}",
            start_points.len(),
            start_points
        )));
    }
    Ok(start_points[0])
}
