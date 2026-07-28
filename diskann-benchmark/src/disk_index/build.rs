/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};
use std::{fmt, fs::File, io::BufWriter, path::Path};

use diskann::{
    graph::config,
    utils::{VectorRepr, ONE},
    ANNError,
};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_disk::{
    build::builder::build::DiskIndexBuilder,
    data_model::AdHoc,
    disk_index_build_parameter::{
        DiskIndexBuildParameters, MemoryBudget, NumPQChunks, DISK_SECTOR_LEN,
    },
    search::ivf_pq_router::{
        build_ivf_pq_router_data, write_ivf_pq_router_binary, IvfPqRouterBuildParams,
    },
    storage::DiskIndexWriter,
};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{model::IndexConfiguration, utils::load_metadata_from_file};
use diskann_vector::distance::Metric;
use opentelemetry::global;
use opentelemetry::trace::Tracer;
use opentelemetry_sdk::trace::SdkTracerProvider;
use scopeguard::defer;

use crate::{
    disk_index::json_spancollector::JsonSpanCollector,
    inputs::disk::{DiskIndexBuild, IvfPqRouterBuildConfig},
    utils::datafiles,
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
    if let Some(ivf_pq_router_build) = &params.ivf_pq_router_build {
        build_ivf_pq_router_artifact::<T>(params, ivf_pq_router_build)?;
    }
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

fn build_ivf_pq_router_artifact<T>(
    params: &DiskIndexBuild,
    config: &IvfPqRouterBuildConfig,
) -> anyhow::Result<()>
where
    T: VectorRepr,
{
    let (training_data, num_points, dim) =
        load_training_data_as_f32::<T>(&params.data, params.dim, "IVF+PQ router")?;
    let pool = diskann_providers::utils::create_thread_pool(params.num_threads)?;
    let build_params = IvfPqRouterBuildParams {
        num_centroids: config.num_centroids.get(),
        max_iterations: config.max_iterations.get(),
        seed: config.seed,
        fallback_medoid: None,
        training_sample_size: config.training_sample_size.map(|value| value.get()),
    };
    let artifact = build_ivf_pq_router_data(
        &training_data,
        num_points,
        dim,
        &build_params,
        pool.as_ref(),
    )?;

    let file = File::create(&config.save_path).map_err(|err| {
        anyhow::anyhow!(
            "failed to create IVF+PQ router artifact {}: {err}",
            config.save_path
        )
    })?;
    let mut writer = BufWriter::new(file);
    if is_binary_artifact_path(Path::new(&config.save_path)) {
        write_ivf_pq_router_binary(&mut writer, &artifact)?;
    } else {
        serde_json::to_writer_pretty(writer, &artifact)?;
    }
    Ok(())
}

fn load_training_data_as_f32<T>(
    data: &diskann_benchmark_runner::files::InputFile,
    expected_dim: usize,
    artifact_label: &str,
) -> anyhow::Result<(Vec<f32>, usize, usize)>
where
    T: VectorRepr,
{
    let dataset: diskann_utils::views::Matrix<T> =
        datafiles::load_dataset(datafiles::BinFile(data)).map_err(|err| {
            anyhow::anyhow!(
                "failed to load {} training data {}: {err}",
                artifact_label,
                data.display()
            )
        })?;
    if dataset.ncols() != expected_dim {
        anyhow::bail!(
            "{} training data dimension {} does not match build dim {}",
            artifact_label,
            dataset.ncols(),
            expected_dim
        );
    }

    let num_points = dataset.nrows();
    let dim = dataset.ncols();
    let value_count = num_points
        .checked_mul(dim)
        .ok_or_else(|| anyhow::anyhow!("{} training data shape overflowed", artifact_label))?;
    let mut training_data = vec![0.0; value_count];
    for row in 0..num_points {
        let dst = &mut training_data[row * dim..(row + 1) * dim];
        T::as_f32_into(dataset.row(row), dst)
            .map_err(|err| anyhow::Error::from(Into::<ANNError>::into(err)))?;
    }

    Ok((training_data, num_points, dim))
}

fn is_binary_artifact_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("bin"))
}
