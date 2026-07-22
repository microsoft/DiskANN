/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PiPNN graph construction for the common disk-build pipeline.
//!
//! The common builder owns search-PQ generation and disk layout. This module
//! replaces only graph construction and adapts PiPNN adjacency lists to the
//! standard graph file.

use std::io::Read;

use diskann::{utils::VectorRepr, ANNError, ANNResult};
use diskann_pipnn::{PiPNNBuildContext, PiPNNConfig};
use diskann_providers::{
    storage::{bin::GetAdjacencyList, StorageReadProvider, StorageWriteProvider},
    utils::{find_medoid_with_sampling, MAX_MEDOID_SAMPLE_SIZE},
};
use diskann_utils::io::Metadata;

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
    if points >= u32::MAX as usize {
        return Err(ANNError::log_index_error(format_args!(
            "PiPNN dataset has {points} points, leaving no u32 ID for the frozen start point"
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
    let (points, dimensions, mut data) =
        load_with_spare_row::<Data::VectorDataType, _>(&data_path, builder.storage_provider)?;

    let mut rng = diskann_providers::utils::create_rnd_from_optional_seed(
        builder.index_configuration.random_seed,
    );
    let (start_vector, start_id) = find_medoid_with_sampling::<Data::VectorDataType, _>(
        &data_path,
        builder.storage_provider,
        MAX_MEDOID_SAMPLE_SIZE,
        &mut rng,
    )?;
    let frozen_id = u32_try_from(points)?;
    let start_id = u32_try_from(start_id)?;
    data.extend_from_slice(&start_vector);

    let adjacency = diskann_pipnn::builder::build_typed(&data, points + 1, dimensions, context)?;
    diskann_providers::storage::bin::save_graph_with_remapped_start(
        &FrozenGraph(&adjacency),
        builder.storage_provider,
        frozen_id,
        start_id,
        &builder.index_writer.get_mem_index_file(),
    )?;
    Ok(())
}

/// Load the dense dataset while reserving one row for the frozen start point.
/// The generic matrix loader has no spare capacity, which would force a second
/// full-dataset allocation when the row is appended.
fn load_with_spare_row<T, StorageProvider>(
    path: &str,
    storage: &StorageProvider,
) -> ANNResult<(usize, usize, Vec<T>)>
where
    T: VectorRepr,
    StorageProvider: StorageReadProvider,
{
    let mut reader = storage.open_reader(path)?;
    let (points, dimensions) = Metadata::read(&mut reader)?.into_dims();
    let elements = points
        .checked_mul(dimensions)
        .ok_or_else(|| ANNError::log_index_error("dataset shape overflows usize"))?;
    let expected_bytes = elements
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| ANNError::log_index_error("dataset byte size overflows usize"))?;
    let available_bytes = storage.get_length(path)?.saturating_sub(8);
    if available_bytes < expected_bytes as u64 {
        return Err(ANNError::log_index_error(format_args!(
            "dataset declares {expected_bytes} payload bytes, but only {available_bytes} are available"
        )));
    }
    let capacity = elements
        .checked_add(dimensions)
        .ok_or_else(|| ANNError::log_index_error("dataset capacity overflows usize"))?;

    let mut data = Vec::new();
    data.try_reserve_exact(capacity)
        .map_err(|error| ANNError::log_index_error(error.to_string()))?;
    data.resize(elements, T::default());
    reader.read_exact(bytemuck::must_cast_slice_mut(&mut data))?;
    Ok((points, dimensions, data))
}

/// Serialization view for a graph with one trailing frozen row.
struct FrozenGraph<'a>(&'a [Vec<u32>]);

impl GetAdjacencyList for FrozenGraph<'_> {
    type Element = u32;
    type Item<'a>
        = &'a [u32]
    where
        Self: 'a;

    fn get_adjacency_list(&self, index: usize) -> ANNResult<Self::Item<'_>> {
        self.0
            .get(index)
            .map(Vec::as_slice)
            .ok_or_else(|| ANNError::log_index_error(format_args!("missing node {index}")))
    }

    fn total(&self) -> usize {
        self.0.len()
    }

    fn additional_points(&self) -> u64 {
        1
    }

    fn max_degree(&self) -> Option<u32> {
        None
    }
}

#[cfg(test)]
mod tests;
