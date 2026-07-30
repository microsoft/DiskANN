/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};
use std::fmt;

use diskann::{
    graph::config,
    utils::{VectorRepr, ONE},
};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_disk::{
    build::builder::build::DiskIndexBuilder,
    data_model::{AdHoc, CachingStrategy},
    disk_index_build_parameter::{
        DiskIndexBuildParameters, MemoryBudget, NumPQChunks, DISK_SECTOR_LEN,
    },
    search::{
        pq_kmeans_router::{PqKmeansRouterBuildParams, PqKmeansRouterData},
        provider::{
            aligned_file_reader::AlignedFileReaderFactory,
            disk_vertex_provider_factory::DiskVertexProviderFactory,
        },
        traits::VertexProviderFactory,
    },
    storage::{disk_index_reader::DiskIndexReader, DiskIndexWriter},
};
use diskann_providers::storage::{
    get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, FileStorageProvider,
    StorageReadProvider, StorageWriteProvider,
};
use diskann_providers::{model::IndexConfiguration, utils::load_metadata_from_file};
use diskann_vector::distance::Metric;
use opentelemetry::global;
use opentelemetry::trace::Tracer;
use opentelemetry_sdk::trace::SdkTracerProvider;
use scopeguard::defer;

use crate::{
    disk_index::json_spancollector::JsonSpanCollector,
    inputs::disk::{DiskIndexBuild, PqKmeansRouterBuild},
};

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct DiskBuildStats {
    build_time: MicroSeconds,
    span_metrics: serde_json::Value,
}

impl DiskBuildStats {
    pub(super) fn new(build_time: MicroSeconds, span_metrics: serde_json::Value) -> Self {
        Self {
            build_time,
            span_metrics,
        }
    }

    pub(super) fn build_time_seconds(&self) -> f64 {
        self.build_time.as_seconds()
    }
}

impl fmt::Display for DiskBuildStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let build_time_seconds = self.build_time.as_seconds();
        writeln!(f, "Build time: {:.3}s", build_time_seconds)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct PqKmeansRouterBuildStats {
    build_time: MicroSeconds,
    num_points: usize,
    num_pq_chunks: usize,
    num_representatives: usize,
    artifact_bytes: u64,
}

impl fmt::Display for PqKmeansRouterBuildStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "PQ-kmeans router build time: {:.3}s",
            self.build_time.as_seconds()
        )?;
        writeln!(f, "Points: {}", self.num_points)?;
        writeln!(f, "PQ chunks: {}", self.num_pq_chunks)?;
        writeln!(f, "Representatives: {}", self.num_representatives)?;
        writeln!(f, "Artifact bytes: {}", self.artifact_bytes)
    }
}

pub(super) fn build_pq_kmeans_router(
    params: &PqKmeansRouterBuild,
) -> anyhow::Result<PqKmeansRouterBuildStats> {
    let start = std::time::Instant::now();
    let index_reader = DiskIndexReader::new(
        get_pq_pivot_file(&params.load_path),
        get_compressed_pq_file(&params.load_path),
        &FileStorageProvider,
    )?;
    let vertex_provider_factory =
        DiskVertexProviderFactory::<AdHoc<f32>, AlignedFileReaderFactory>::from_disk_index_path(
            get_disk_index_file(&params.load_path),
            CachingStrategy::None,
        )?;
    let graph_header = vertex_provider_factory.get_header()?;
    let pq_data = index_reader.get_pq_data();
    let router_data = PqKmeansRouterData::build_from_pq_data(
        pq_data.as_ref(),
        PqKmeansRouterBuildParams {
            num_representatives: params.num_representatives,
            training_sample_size: params.training_sample_size,
            max_iterations: params.max_iterations,
        },
        Some(graph_header.metadata().medoid as u32),
    )?;
    router_data.save_to_path(&params.artifact)?;
    let artifact_bytes = std::fs::metadata(&params.artifact)?.len();
    let build_time = start.elapsed().into();

    Ok(PqKmeansRouterBuildStats {
        build_time,
        num_points: router_data.num_points,
        num_pq_chunks: router_data.num_pq_chunks,
        num_representatives: router_data.representative_ids.len(),
        artifact_bytes,
    })
}

pub(super) fn build_disk_index<T, StorageProviderType>(
    storage_provider: &StorageProviderType,
    params: &DiskIndexBuild,
) -> anyhow::Result<DiskBuildStats>
where
    T: VectorRepr,
    StorageProviderType: StorageReadProvider + StorageWriteProvider + 'static,
    <StorageProviderType as StorageReadProvider>::Reader: std::marker::Send,
{
    let previous_tracer_provider = global::tracer_provider();
    let span_collector = {
        let collector = JsonSpanCollector::new();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(collector.clone())
            .build();
        global::set_tracer_provider(provider.clone());
        Some((collector, provider))
    };
    defer! {
        global::set_tracer_provider(previous_tracer_provider);
    }

    let metric: Metric = params.distance.into();
    let config = config::Builder::new_with(
        params.max_degree,
        config::MaxDegree::default_slack(),
        params.l_build,
        metric.into(),
        |b| {
            b.saturate_after_prune(true);
        },
    )
    .build()?;

    let data_path = params.data.to_string_lossy().to_string();

    let metadata = load_metadata_from_file(storage_provider, &data_path)?;

    let build_parameters = DiskIndexBuildParameters::new(
        MemoryBudget::try_from_gb(params.build_ram_limit_gb)?,
        params.quantization_type,
        NumPQChunks::new_with(params.num_pq_chunks.get(), metadata.ndims())?,
    );

    let index_configuration = IndexConfiguration::new(
        metric,
        metadata.ndims(),
        metadata.npoints(),
        ONE,
        params.num_threads,
        config,
    )
    .with_pseudo_rng();

    let disk_index_writer = DiskIndexWriter::new(
        data_path,
        params.save_path.clone(),
        Option::None,
        DISK_SECTOR_LEN,
    )?;

    let mut disk_index = DiskIndexBuilder::<AdHoc<T>, StorageProviderType>::new(
        storage_provider,
        build_parameters,
        index_configuration,
        disk_index_writer,
    )?;

    let span = {
        let tracer = opentelemetry::global::tracer("benchmark");
        tracer.start("disk-index-build")
    };

    let start = std::time::Instant::now();
    disk_index.build()?;
    let total_time: MicroSeconds = start.elapsed().into();

    drop(span);
    let span_metrics = if let Some((collector, provider)) = span_collector {
        provider.shutdown()?;
        collector.to_hierarchical_json()
    } else {
        serde_json::json!({ "span_data": [] })
    };

    Ok(DiskBuildStats::new(total_time, span_metrics))
}
