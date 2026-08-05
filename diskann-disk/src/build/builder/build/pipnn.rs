/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Adapter from provider-independent PiPNN adjacency to the common disk index format.
//!
//! The core crate deliberately knows nothing about dataset files, medoids, graph
//! headers, or serialization. This adapter owns that boundary:
//!
//! 1. verify on-disk dataset metadata against the requested index configuration;
//! 2. load the contiguous matrix required by batch construction;
//! 3. run PiPNN in the caller-provided Rayon pool;
//! 4. compute the production start node with the existing medoid policy; and
//! 5. serialize adjacency with the same header/layout used by Vamana.
//!
//! ```text
//! dataset file ──> metadata check ──> MatrixView ──> diskann::graph::pipnn ──> adjacency
//!      │                                                              │
//!      └──────────────────> sampled medoid ────────────────────────────┤
//!                                                                     v
//!                                                        canonical graph writer
//! ```
//!
//! There is no PiPNN-specific disk graph format. Keeping serialization here means
//! search and loading cannot distinguish which builder produced the graph.

use diskann::graph::pipnn::{PiPNNBuildContext, PiPNNConfig};
use diskann::{utils::VectorRepr, ANNError, ANNResult};
use diskann_providers::{
    storage::{save_adjacency_graph, StorageReadProvider, StorageWriteProvider},
    utils::{find_medoid_with_sampling, RayonThreadPoolRef, MAX_MEDOID_SAMPLE_SIZE},
};
use diskann_utils::io::{read_bin, Metadata};

use super::{u32_try_from, DiskIndexBuilder};
use crate::data_model::GraphDataType;

/// Build PiPNN adjacency and persist it through the canonical disk graph writer.
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
    // Validate metadata before allocating/loading the full matrix. A mismatch
    // here otherwise turns a configuration error into a later shape failure.
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

    // PiPNN is a batch algorithm: materialize the matrix once, while all
    // partition and leaf scratch stays inside the supplied pool and is released
    // before the outer disk pipeline continues.
    let data =
        read_bin::<Data::VectorDataType>(&mut builder.storage_provider.open_reader(&data_path)?)?;
    let context = PiPNNBuildContext::new(
        config,
        &builder.index_configuration.config,
        builder.index_configuration.dist_metric,
        pool.as_rayon(),
    )?;
    let adjacency = diskann::graph::pipnn::build_graph(data.as_view(), &context)?;

    // Start-node policy belongs to the persisted index, not the core graph
    // constructor. Reuse the production sampled medoid implementation so the
    // serialized header has the same semantics as a Vamana-built index.
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
