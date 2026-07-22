/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{graph::config, utils::ONE};
use diskann_providers::{
    model::IndexConfiguration,
    storage::{
        get_disk_index_file, StorageReadProvider, StorageWriteProvider, VirtualStorageProvider,
    },
};
use diskann_utils::{io::write_bin, views::MatrixView};
use diskann_vector::distance::Metric;

use super::*;
use crate::{
    build::{
        builder::build::DiskIndexBuilder,
        configuration::{MemoryBudget, NumPQChunks},
    },
    data_model::AdHoc,
    storage::DiskIndexWriter,
    DiskIndexBuildParameters,
};

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
    assert!(storage.exists("/index_pq_compressed.bin"));
}

#[test]
fn pipnn_disk_build_uses_the_common_pipeline() {
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

    let mut builder =
        DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, params, index_config, writer).unwrap();

    builder.build().unwrap();
    assert!(storage.exists(&get_disk_index_file("/index")));
}

#[test]
fn pipnn_graph_contains_only_real_points() {
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

    let pipnn = diskann_pipnn::PiPNNConfig {
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
    };
    let params = DiskIndexBuildParameters::new_pipnn(
        MemoryBudget::try_from_gb(1.0).unwrap(),
        NumPQChunks::new_with(dimensions, dimensions).unwrap(),
        pipnn.clone(),
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
    let builder =
        DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, params, index_config, writer).unwrap();

    let context = prepare_context(&builder, &pipnn).unwrap();
    build_graph(&builder, &context).unwrap();

    let mut header = [0_u8; 24];
    std::io::Read::read_exact(
        &mut storage
            .open_reader(&builder.index_writer.get_mem_index_file())
            .unwrap(),
        &mut header,
    )
    .unwrap();
    assert_eq!(u32::from_le_bytes(header[8..12].try_into().unwrap()), 32);
    assert!(u32::from_le_bytes(header[12..16].try_into().unwrap()) < points as u32);
    assert_eq!(u64::from_le_bytes(header[16..24].try_into().unwrap()), 0);
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
