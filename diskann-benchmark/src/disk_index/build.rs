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
    data_model::AdHoc,
    disk_index_build_parameter::{
        DiskIndexBuildParameters, MemoryBudget, NumPQChunks, DISK_SECTOR_LEN,
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
    inputs::disk::{DiskIndexBuild, SearchQuantization},
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

    let num_pq_chunks = match params.search_quantization {
        SearchQuantization::PQ => params.num_pq_chunks.get(),
        SearchQuantization::SphericalQ3 => 1,
    };
    let build_parameters = DiskIndexBuildParameters::new(
        MemoryBudget::try_from_gb(params.build_ram_limit_gb)?,
        params.quantization_type,
        NumPQChunks::new_with(num_pq_chunks, metadata.ndims())?,
    )
    .with_search_quantization(params.search_quantization.into());

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

#[cfg(test)]
mod tests {
    use std::{io::Write, num::NonZeroUsize};

    use diskann_benchmark_runner::files::InputFile;
    use diskann_disk::{storage::quant::SearchData, QuantizationType};
    use diskann_providers::storage::{get_disk_index_file, FileStorageProvider};
    use diskann_utils::{
        io::write_bin,
        views::rowmajor::{self, Matrix},
    };

    use super::*;
    use crate::inputs::{
        disk::{DiskIndexOperation, DiskIndexSource},
        Example,
    };

    #[test]
    fn search_quantization_controls_pq_chunk_validation() {
        let storage = FileStorageProvider;
        let tempdir = tempfile::tempdir().unwrap();
        let data_path = tempdir.path().join("data.bin");
        let values = (0..37 * 13)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) / 7.0)
            .collect::<Vec<_>>();
        let data = rowmajor::Owned::try_from_data(values.into_boxed_slice(), 37, 13).unwrap();
        let mut writer = storage
            .create_for_write(&data_path.to_string_lossy())
            .unwrap();
        write_bin(data.as_view(), &mut writer).unwrap();
        writer.flush().unwrap();
        drop(writer);

        let DiskIndexSource::Build(mut params) = DiskIndexOperation::example().source else {
            panic!("expected build example");
        };
        params.data = InputFile::new(data_path);
        params.dim = 13;
        params.num_threads = 1;
        params.quantization_type = QuantizationType::FP;
        params.num_pq_chunks = NonZeroUsize::new(128).unwrap();
        params.save_path = tempdir.path().join("index").to_string_lossy().into_owned();

        params.search_quantization = SearchQuantization::PQ;
        let error = build_disk_index::<f32, _>(&storage, &params).unwrap_err();
        assert!(error
            .to_string()
            .contains("Number of PQ chunks must be within [1, 13]"));
        assert!(!storage.exists(&get_disk_index_file(&params.save_path)));

        params.search_quantization = SearchQuantization::SphericalQ3;
        build_disk_index::<f32, _>(&storage, &params).unwrap();
        let search_data = SearchData::load(
            &params.save_path,
            params.search_quantization.into(),
            &storage,
        )
        .unwrap();
        assert_eq!(search_data.num_points(), 37);
        assert!(storage.exists(&get_disk_index_file(&params.save_path)));
        for path in SearchQuantization::PQ.paths(&params.save_path) {
            assert!(!storage.exists(&path));
        }
    }
}
