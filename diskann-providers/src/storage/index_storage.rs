/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use super::StorageReadProvider;
use diskann::{ANNError, ANNResult, graph::DiskANNIndex, provider::DataProvider};

use super::{AsyncIndexMetadata, AsyncQuantLoadContext, LoadWith};
use crate::model::configuration::IndexConfiguration;

/// Creates a `AsyncQuantLoadContext` from an `IndexConfiguration` with the given path and disk index flag.
pub fn create_load_context(
    path: &str,
    index_config: &IndexConfiguration,
) -> ANNResult<AsyncQuantLoadContext> {
    Ok(AsyncQuantLoadContext {
        metadata: AsyncIndexMetadata::new(path),
        num_frozen_points: index_config.num_frozen_pts,
        metric: index_config.dist_metric,
        prefetch_lookahead: index_config.prefetch_lookahead.map(|x| x.get()),
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
        let pq_context = create_load_context(path, index_config)?;

        let data_provider = DP::load_with(provider, &pq_context).await?;
        let num_threads = index_config.num_threads;
        Ok(Self::new(
            index_config.config.clone(),
            data_provider,
            NonZeroUsize::new(num_threads),
        ))
    }
}
