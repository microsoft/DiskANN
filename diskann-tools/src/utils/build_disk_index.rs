/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{
    graph::config,
    utils::{IntoUsize, ONE},
    ANNError, ANNResult,
};
use diskann_disk::{
    build::{
        builder::build::DiskIndexBuilder,
        chunking::{checkpoint::CheckpointManager, continuation::ChunkingConfig},
    },
    disk_index_build_parameter::{
        DiskIndexBuildParameters, MemoryBudget, NumPQChunks, DISK_SECTOR_LEN,
    },
    storage::DiskIndexWriter,
    QuantizationType,
};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::{graph::traits::GraphDataType, IndexConfiguration},
    utils::{load_metadata_from_file, Timer},
};
use diskann_vector::distance::Metric;
use opentelemetry::global::BoxedSpan;
#[cfg(feature = "perf_test")]
use opentelemetry::{
    trace::{Span, Tracer},
    KeyValue,
};

pub struct ChunkingParameters {
    pub chunking_config: ChunkingConfig,
    pub checkpoint_record_manager: Box<dyn CheckpointManager>,
}

/// A simple struct to contain the underlying dimension of the data and
/// its full-precision vector dimension.
///
/// * `dim` is the length of the vector when represented with the underlying datatype
/// * `full_dim` is the length of the vector when converted to a full-precision slice, i.e. [f32]
///
/// # Notes
///
/// These values are the same when using primitive data types to represent the vectors
/// such as `half::f16` or `f32`, however, for quantized vectors used in place of
/// full-preicision vectors such as [`common::MinMaxElement`] these might be different.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DimensionValues {
    dim: usize,
    full_dim: usize,
}
impl DimensionValues {
    pub fn new(dim: usize, full_dim: usize) -> Self {
        Self { dim, full_dim }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn full_dim(&self) -> usize {
        self.full_dim
    }
}

pub struct BuildDiskIndexParameters<'a> {
    pub metric: Metric,
    pub data_path: &'a str,
    pub r: u32,
    pub l: u32,
    pub index_path_prefix: &'a str,
    pub num_threads: usize,
    pub num_of_pq_chunks: usize,
    pub index_build_ram_limit_gb: f64,
    pub build_quantization_type: QuantizationType,
    pub chunking_parameters: Option<ChunkingParameters>,
    pub dim_values: DimensionValues,
}

/// The main function to build a disk index
pub fn build_disk_index<Data, StorageProviderType>(
    storage_provider: &StorageProviderType,
    parameters: BuildDiskIndexParameters,
) -> ANNResult<()>
where
    Data: GraphDataType<VectorIdType = u32>,
    StorageProviderType: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageProviderType as StorageReadProvider>::Reader: std::marker::Send,
{
    let build_parameters = DiskIndexBuildParameters::new(
        MemoryBudget::try_from_gb(parameters.index_build_ram_limit_gb)?,
        parameters.build_quantization_type,
        NumPQChunks::new_with(
            parameters.num_of_pq_chunks,
            parameters.dim_values.full_dim(),
        )?,
    );

    let config = config::Builder::new_with(
        parameters.r.into_usize(),
        config::MaxDegree::default_slack(),
        parameters.l.into_usize(),
        parameters.metric.into(),
        |b| {
            b.saturate_after_prune(true);
        },
    )
    .build()?;

    let metadata = load_metadata_from_file(storage_provider, parameters.data_path)?;

    if metadata.ndims != parameters.dim_values.dim() {
        return Err(ANNError::log_index_config_error(
            format!("{:?}", parameters.dim_values),
            format!("dim_values must match with data_dim {}", metadata.ndims),
        ));
    }

    let index_configuration = IndexConfiguration::new(
        parameters.metric,
        metadata.ndims,
        metadata.npoints,
        ONE,
        parameters.num_threads,
        config,
    )
    .with_pseudo_rng();

    let disk_index_writer = DiskIndexWriter::new(
        parameters.data_path.to_string(),
        parameters.index_path_prefix.to_string(),
        Option::None,
        DISK_SECTOR_LEN,
    )?;

    let mut disk_index = match parameters.chunking_parameters {
        Some(chunking_parameters) => {
            let chunking_config = chunking_parameters.chunking_config;
            let checkpoint_record_manager = chunking_parameters.checkpoint_record_manager;
            DiskIndexBuilder::<Data, StorageProviderType>::new_with_chunking_config(
                storage_provider,
                build_parameters,
                index_configuration,
                disk_index_writer,
                chunking_config,
                checkpoint_record_manager,
            )
        }
        None => DiskIndexBuilder::<Data, StorageProviderType>::new(
            storage_provider,
            build_parameters,
            index_configuration,
            disk_index_writer,
        ),
    }?;

    let mut _span: BoxedSpan;
    #[cfg(feature = "perf_test")]
    {
        let tracer = opentelemetry::global::tracer("");

        // Start a span for the search iteration.
        _span = tracer.start("index-build statistics".to_string());
    }

    let timer = Timer::new();
    disk_index.build()?;

    let diff = timer.elapsed();
    println!("Indexing time: {} seconds", diff.as_secs_f64());

    #[cfg(feature = "perf_test")]
    {
        _span.set_attribute(KeyValue::new("total_time", diff.as_secs_f64()));
        _span.set_attribute(KeyValue::new("total_comparisons", 0i64));
        _span.set_attribute(KeyValue::new("search_hops", 0i64));
        _span.end();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use diskann::ANNErrorKind;
    use diskann_providers::storage::VirtualStorageProvider;
    use vfs::MemoryFS;

    use super::*;
    use crate::utils::GraphDataInt8Vector;

    #[test]
    fn test_build_disk_index_with_num_of_pq_chunks() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let parameters = BuildDiskIndexParameters {
            metric: Metric::L2,
            data_path: "test_data_path",
            r: 10,
            l: 20,
            index_path_prefix: "test_index_path_prefix",
            num_threads: 4,
            num_of_pq_chunks: 8,
            index_build_ram_limit_gb: 1.0,
            build_quantization_type: QuantizationType::FP,
            chunking_parameters: None,
            dim_values: DimensionValues::new(128, 128),
        };

        let result = build_disk_index::<GraphDataInt8Vector, VirtualStorageProvider<MemoryFS>>(
            &storage_provider,
            parameters,
        );
        assert!(result.is_err());
        assert_ne!(result.unwrap_err().kind(), ANNErrorKind::IndexConfigError);
    }

    #[test]
    fn test_build_disk_index_with_zero_num_of_pq_chunks() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let parameters = BuildDiskIndexParameters {
            metric: Metric::L2,
            data_path: "test_data_path",
            r: 10,
            l: 20,
            index_path_prefix: "test_index_path_prefix",
            num_threads: 4,
            num_of_pq_chunks: 0,
            index_build_ram_limit_gb: 1.0,
            build_quantization_type: QuantizationType::FP,
            chunking_parameters: None,
            dim_values: DimensionValues::new(128, 128),
        };

        let result = build_disk_index::<GraphDataInt8Vector, VirtualStorageProvider<MemoryFS>>(
            &storage_provider,
            parameters,
        );
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), ANNErrorKind::IndexConfigError);
    }
}
