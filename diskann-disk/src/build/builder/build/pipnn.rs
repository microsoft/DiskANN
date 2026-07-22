/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PiPNN graph construction for the common disk-build pipeline.
//!
//! The common builder owns search-PQ generation and disk layout. This module
//! replaces only graph construction and adapts PiPNN adjacency lists to the
//! standard graph file.

use diskann::{utils::VectorRepr, ANNError, ANNResult};
use diskann_pipnn::{PiPNNBuildContext, PiPNNConfig};
use diskann_providers::{
    storage::{save_graph, GetAdjacencyList, StorageReadProvider, StorageWriteProvider},
    utils::{find_medoid_with_sampling, MAX_MEDOID_SAMPLE_SIZE},
};
use diskann_utils::io::{read_bin, Metadata};

use super::{u32_try_from, DiskIndexBuilder};
use crate::data_model::GraphDataType;

pub(super) fn prepare_context<Data, StorageProvider>(
    builder: &DiskIndexBuilder<'_, Data, StorageProvider>,
    config: &PiPNNConfig,
) -> ANNResult<PiPNNBuildContext>
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
    PiPNNBuildContext::new(
        config.clone(),
        builder.index_configuration.config.pruned_degree(),
        builder.index_configuration.config.alpha(),
        builder.index_configuration.dist_metric,
        builder.index_configuration.num_threads,
    )
}

pub(super) fn build_graph<Data, StorageProvider>(
    builder: &DiskIndexBuilder<'_, Data, StorageProvider>,
    context: &PiPNNBuildContext,
) -> ANNResult<()>
where
    Data: GraphDataType<VectorIdType = u32>,
    Data::VectorDataType: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    let data_path = builder.index_writer.get_dataset_file();
    let data =
        read_bin::<Data::VectorDataType>(&mut builder.storage_provider.open_reader(&data_path)?)?;
    let points = data.nrows();
    let dimensions = data.ncols();
    let adjacency =
        diskann_pipnn::builder::build_typed(data.as_slice(), points, dimensions, context)?;

    let mut rng = diskann_providers::utils::create_rnd_from_optional_seed(
        builder.index_configuration.random_seed,
    );
    let (_, start_id) = find_medoid_with_sampling::<Data::VectorDataType, _>(
        &data_path,
        builder.storage_provider,
        MAX_MEDOID_SAMPLE_SIZE,
        &mut rng,
    )?;
    let start_id = u32_try_from(start_id)?;
    let graph = PiPNNGraph {
        adjacency: &adjacency,
        max_degree: u32_try_from(builder.index_configuration.config.pruned_degree().get())?,
    };
    save_graph(
        &graph,
        builder.storage_provider,
        start_id,
        &builder.index_writer.get_mem_index_file(),
    )?;
    Ok(())
}

struct PiPNNGraph<'a> {
    adjacency: &'a [Vec<u32>],
    max_degree: u32,
}

impl GetAdjacencyList for PiPNNGraph<'_> {
    type Element = u32;
    type Item<'a>
        = &'a [u32]
    where
        Self: 'a;

    fn get_adjacency_list(&self, index: usize) -> ANNResult<Self::Item<'_>> {
        self.adjacency
            .get(index)
            .map(Vec::as_slice)
            .ok_or_else(|| ANNError::log_index_error(format_args!("missing node {index}")))
    }

    fn total(&self) -> usize {
        self.adjacency.len()
    }

    fn additional_points(&self) -> u64 {
        0
    }

    fn max_degree(&self) -> Option<u32> {
        Some(self.max_degree)
    }
}

#[cfg(test)]
mod tests;
