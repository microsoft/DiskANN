/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use super::{StorageReadProvider, StorageWriteProvider};
use diskann::{
    ANNError, ANNResult, graph::DiskANNIndex, provider::DataProvider,
};

use super::{AsyncIndexMetadata, AsyncQuantLoadContext, DiskGraphOnly, LoadWith, SaveWith};
use crate::model::configuration::IndexConfiguration;

/// A trait for types that can provide starting points for index search.
///
/// This trait is used to generalize the `SaveWith` implementations for `DiskANNIndex`
/// so that they are not tied to a specific provider type.
pub trait HasStartingPoints {
    /// Return a vector of starting points.
    fn starting_points(&self) -> ANNResult<Vec<u32>>;
}

impl<DP> SaveWith<AsyncIndexMetadata> for DiskANNIndex<DP>
where
    DP: HasStartingPoints + SaveWith<(u32, AsyncIndexMetadata), Error = ANNError> + Send + Sync,
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
impl<DP> SaveWith<(u32, DiskGraphOnly)> for DiskANNIndex<DP>
where
    DP: HasStartingPoints + SaveWith<(u32, u32, DiskGraphOnly), Error = ANNError> + Send + Sync,
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

// Inmem-specific load functions have been moved to the `diskann-inmem` crate.
// Use `diskann_inmem::storage` module for `load_pq_index`, `load_pq_index_with_deletes`,
// `load_fp_index`, `load_index`, and `load_index_with_deletes`.

/// Retrieves starting points and enforces that there is exactly one starting point.
///
/// This helper function:
/// 1. Retrieves the starting points from the data provider
/// 2. Validates there is exactly one starting point
/// 3. Returns the single start point if valid
///
/// Returns an error if there are multiple starting points or no starting points.
fn get_and_validate_single_starting_point(
    data_provider: &impl HasStartingPoints,
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

// Tests for the inmem-specific load/save functionality have been moved to the
// `diskann-inmem` crate alongside the in-memory provider types they test.
