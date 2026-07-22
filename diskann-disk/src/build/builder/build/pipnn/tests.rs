/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{graph::config, utils::ONE, ANNResult};
use diskann_providers::{
    model::IndexConfiguration,
    storage::{get_disk_index_file, StorageWriteProvider, VirtualStorageProvider},
};
use diskann_utils::{io::write_bin, views::MatrixView};
use diskann_vector::distance::Metric;

use super::*;
use crate::{
    build::{
        builder::build::DiskIndexBuilder,
        chunking::{
            checkpoint::{CheckpointManager, Progress, WorkStage},
            continuation::ChunkingConfig,
        },
        configuration::{MemoryBudget, NumPQChunks},
    },
    data_model::AdHoc,
    storage::DiskIndexWriter,
    DiskIndexBuildParameters,
};

#[derive(Clone)]
struct RejectCheckpointManager;

impl CheckpointManager for RejectCheckpointManager {
    fn get_resumption_point(&self, _stage: WorkStage) -> ANNResult<Option<usize>> {
        panic!("PiPNN must not read checkpoint state")
    }

    fn update(&mut self, _progress: Progress, _next_stage: WorkStage) -> ANNResult<()> {
        panic!("PiPNN must not write checkpoint state")
    }

    fn mark_as_invalid(&mut self) -> ANNResult<()> {
        panic!("PiPNN must not invalidate checkpoint state")
    }
}

#[test]
fn load_with_spare_row_rejects_short_payload() {
    let storage = VirtualStorageProvider::new_memory();
    let mut writer = storage.create_for_write("/short.fbin").unwrap();
    std::io::Write::write_all(&mut writer, &2_u32.to_le_bytes()).unwrap();
    std::io::Write::write_all(&mut writer, &3_u32.to_le_bytes()).unwrap();
    std::io::Write::write_all(&mut writer, bytemuck::cast_slice(&[1.0f32; 5])).unwrap();
    drop(writer);

    let error = load_with_spare_row::<f32, _>("/short.fbin", &storage).unwrap_err();
    assert!(format!("{error:?}").contains("declares 24 payload bytes"));
}

#[test]
fn pipnn_disk_build_rejects_configuration_dataset_mismatch() {
    let storage = VirtualStorageProvider::new_memory();
    write_bin(
        MatrixView::try_from(&[0.0f32; 16][..], 2, 8).unwrap(),
        &mut storage.create_for_write("/data.fbin").unwrap(),
    )
    .unwrap();

    let params = DiskIndexBuildParameters::new_pipnn(
        MemoryBudget::try_from_gb(10_000.0).unwrap(),
        NumPQChunks::new_with(4, 4).unwrap(),
        diskann_pipnn::PiPNNConfig::default(),
    );
    let graph_config = config::Builder::new_with(
        4,
        config::MaxDegree::default_slack(),
        8,
        Metric::L2.into(),
        |_| {},
    )
    .build()
    .unwrap();
    let index_config = IndexConfiguration::new(Metric::L2, 4, 3, ONE, 1, graph_config);
    let writer = DiskIndexWriter::new("/data.fbin".into(), "/index".into(), None, 4096).unwrap();
    let mut builder =
        DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, params, index_config, writer).unwrap();

    let error = builder.build().unwrap_err();
    assert!(format!("{error:?}").contains("configured dimension 4"));
    assert!(!storage.exists("/index_pq_compressed.bin"));
}

#[test]
fn pipnn_disk_build_reserves_a_u32_id_for_the_frozen_point() {
    let storage = VirtualStorageProvider::new_memory();
    let mut data = storage.create_for_write("/data.fbin").unwrap();
    std::io::Write::write_all(&mut data, &u32::MAX.to_le_bytes()).unwrap();
    std::io::Write::write_all(&mut data, &1_u32.to_le_bytes()).unwrap();
    drop(data);

    let params = DiskIndexBuildParameters::new_pipnn(
        MemoryBudget::try_from_gb(10_000.0).unwrap(),
        NumPQChunks::new_with(1, 1).unwrap(),
        diskann_pipnn::PiPNNConfig::default(),
    );
    let graph_config = config::Builder::new_with(
        4,
        config::MaxDegree::default_slack(),
        8,
        Metric::L2.into(),
        |_| {},
    )
    .build()
    .unwrap();
    let index_config =
        IndexConfiguration::new(Metric::L2, 1, u32::MAX as usize, ONE, 1, graph_config);
    let writer = DiskIndexWriter::new("/data.fbin".into(), "/index".into(), None, 4096).unwrap();
    let mut builder =
        DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, params, index_config, writer).unwrap();

    let error = builder.build().unwrap_err();
    assert!(format!("{error:?}").contains("leaving no u32 ID for the frozen start point"));
    assert!(!storage.exists("/index_pq_compressed.bin"));
}

#[test]
fn pipnn_disk_build_is_one_shot_and_does_not_use_checkpoints() {
    let storage = VirtualStorageProvider::new_memory();
    let points = 256;
    let dimensions = 8;
    let data: Vec<f32> = (0..points * dimensions)
        .map(|i| ((i * 17) % 251) as f32)
        .collect();
    write_bin(
        MatrixView::try_from(data.as_slice(), points, dimensions).unwrap(),
        &mut storage.create_for_write("/data.fbin").unwrap(),
    )
    .unwrap();

    let params = DiskIndexBuildParameters::new_pipnn(
        MemoryBudget::try_from_gb(1.0).unwrap(),
        NumPQChunks::new_with(dimensions, dimensions).unwrap(),
        diskann_pipnn::PiPNNConfig {
            c_max: 512,
            c_min: 64,
            p_samp: 0.01,
            fanout: vec![10, 3],
            k: 2,
            replicas: 1,
            l_max: 72,
            num_hash_planes: 14,
            final_prune: true,
            skip_hash_prune: false,
        },
    );
    let graph_config = config::Builder::new_with(
        32,
        config::MaxDegree::default_slack(),
        50,
        Metric::L2.into(),
        |_| {},
    )
    .build()
    .unwrap();
    let index_config =
        IndexConfiguration::new(Metric::L2, dimensions, points, ONE, 1, graph_config)
            .with_pseudo_rng_from_seed(42);
    let writer = DiskIndexWriter::new("/data.fbin".into(), "/index".into(), None, 4096).unwrap();

    let mut builder = DiskIndexBuilder::<AdHoc<f32>, _>::new_with_chunking_config(
        &storage,
        params,
        index_config,
        writer,
        ChunkingConfig::default(),
        Box::new(RejectCheckpointManager),
    )
    .unwrap();

    builder.build().unwrap();
    assert!(storage.exists(&get_disk_index_file("/index")));
}

#[test]
fn pipnn_disk_build_falls_back_to_vamana_above_memory_limit() {
    let storage = VirtualStorageProvider::new_memory();
    let points = 256;
    let dimensions = 8;
    let data: Vec<f32> = (0..points * dimensions)
        .map(|i| ((i * 17) % 251) as f32)
        .collect();
    write_bin(
        MatrixView::try_from(data.as_slice(), points, dimensions).unwrap(),
        &mut storage.create_for_write("/data.fbin").unwrap(),
    )
    .unwrap();

    let params = DiskIndexBuildParameters::new_pipnn(
        MemoryBudget::try_from_gb(0.0001).unwrap(),
        NumPQChunks::new_with(dimensions, dimensions).unwrap(),
        diskann_pipnn::PiPNNConfig {
            c_max: 512,
            c_min: 64,
            p_samp: 0.01,
            fanout: vec![10, 3],
            k: 2,
            replicas: 1,
            l_max: 72,
            num_hash_planes: 14,
            final_prune: true,
            skip_hash_prune: false,
        },
    );
    let graph_config = config::Builder::new_with(
        32,
        config::MaxDegree::default_slack(),
        50,
        Metric::L2.into(),
        |builder| {
            builder.alpha(1.3);
        },
    )
    .build()
    .unwrap();
    let index_config =
        IndexConfiguration::new(Metric::L2, dimensions, points, ONE, 1, graph_config)
            .with_pseudo_rng_from_seed(42);
    let writer = DiskIndexWriter::new("/data.fbin".into(), "/index".into(), None, 4096).unwrap();

    let mut builder =
        DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, params, index_config, writer).unwrap();

    assert!(matches!(
        builder.disk_build_param.build_algorithm(),
        crate::BuildAlgorithm::Vamana
    ));
    assert_eq!(
        builder.disk_build_param.build_quantization(),
        &crate::QuantizationType::FP
    );
    assert_eq!(builder.index_configuration.config.pruned_degree().get(), 32);
    assert_eq!(builder.index_configuration.config.l_build().get(), 50);
    assert_eq!(builder.index_configuration.config.alpha(), 1.3);
    builder.build().unwrap();
    assert!(storage.exists(&get_disk_index_file("/index")));
}

#[test]
fn pipnn_disk_build_rejects_invalid_config_before_fallback() {
    let storage = VirtualStorageProvider::new_memory();
    let params = DiskIndexBuildParameters::new_pipnn(
        MemoryBudget::try_from_gb(0.0001).unwrap(),
        NumPQChunks::new_with(1, 1).unwrap(),
        diskann_pipnn::PiPNNConfig {
            c_max: 0,
            ..diskann_pipnn::PiPNNConfig::default()
        },
    );
    let graph_config = config::Builder::new_with(
        4,
        config::MaxDegree::default_slack(),
        8,
        Metric::L2.into(),
        |_| {},
    )
    .build()
    .unwrap();
    let index_config = IndexConfiguration::new(Metric::L2, 1, 1, ONE, 1, graph_config);
    let writer = DiskIndexWriter::new("/data.fbin".into(), "/index".into(), None, 4096).unwrap();

    let error = match DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, params, index_config, writer)
    {
        Ok(_) => panic!("invalid PiPNN config must not silently fall back to Vamana"),
        Err(error) => error,
    };

    assert!(format!("{error:?}").contains("c_max must be > 0"));
}
