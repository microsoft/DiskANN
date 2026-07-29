/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PiPNN graph adapter for the common disk-build pipeline.

use diskann::{utils::VectorRepr, ANNError, ANNResult};
use diskann_pipnn::{PiPNNBuildContext, PiPNNConfig};
use diskann_providers::{
    storage::{save_adjacency_graph, StorageReadProvider, StorageWriteProvider},
    utils::{find_medoid_with_sampling, RayonThreadPoolRef, MAX_MEDOID_SAMPLE_SIZE},
};
use diskann_utils::io::{read_bin, Metadata};

use super::{u32_try_from, DiskIndexBuilder};
use crate::data_model::GraphDataType;

pub(super) fn build_graph<Data, StorageProvider>(
    builder: &DiskIndexBuilder<'_, Data, StorageProvider>,
    pool: RayonThreadPoolRef<'_>,
    config: PiPNNConfig,
) -> ANNResult<()>
where
    Data: GraphDataType<VectorIdType = u32>,
    Data::VectorDataType: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    let data_path = builder.index_writer.get_dataset_file();
    let (points, dimensions) =
        Metadata::read(&mut builder.storage_provider.open_reader(&data_path)?)?.into_dims();
    if dimensions != builder.index_configuration.dim {
        return Err(ANNError::log_dimension_mismatch_error(format!(
            "configured dimension {} does not match dataset dimension {dimensions}",
            builder.index_configuration.dim
        )));
    }
    if points != builder.index_configuration.max_points {
        return Err(ANNError::log_index_error(format!(
            "configured point count {} does not match dataset point count {points}",
            builder.index_configuration.max_points
        )));
    }

    let data =
        read_bin::<Data::VectorDataType>(&mut builder.storage_provider.open_reader(&data_path)?)?;
    let context = PiPNNBuildContext::new(
        config,
        &builder.index_configuration.config,
        builder.index_configuration.dist_metric,
        pool.as_rayon(),
    )?;
    let adjacency = diskann_pipnn::build_graph(data.as_view(), &context)?;

    let mut rng = diskann_providers::utils::create_rnd_from_optional_seed(
        builder.index_configuration.random_seed,
    );
    let (_, start_id) = find_medoid_with_sampling::<Data::VectorDataType, _>(
        &data_path,
        builder.storage_provider,
        MAX_MEDOID_SAMPLE_SIZE,
        &mut rng,
    )?;
    save_adjacency_graph(
        &adjacency,
        u32_try_from(builder.index_configuration.config.pruned_degree().get())?,
        builder.storage_provider,
        u32_try_from(start_id)?,
        &builder.index_writer.get_mem_index_file(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
