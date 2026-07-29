/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{graph::config, utils::ONE};
use diskann_providers::utils::create_thread_pool;
use diskann_providers::{
    model::IndexConfiguration,
    storage::{
        get_disk_index_file, StorageReadProvider, StorageWriteProvider, VirtualStorageProvider,
    },
};
use diskann_utils::{io::write_bin, views::MatrixView};
use diskann_vector::distance::Metric;
use vfs::MemoryFS;

use crate::{
    build::{
        builder::{
            build::DiskIndexBuilder,
            core::{determine_build_strategy, IndexBuildStrategy},
        },
        configuration::{MemoryBudget, NumPQChunks, PiPNNParameters},
    },
    data_model::AdHoc,
    storage::DiskIndexWriter,
    DiskIndexBuildParameters,
};

fn pipnn() -> PiPNNParameters {
    PiPNNParameters {
        c_max: 512,
        c_min: 64,
        p_samp: 0.01,
        fanout: vec![10, 3],
        k: 2,
        replicas: 1,
    }
}

fn write_data(storage: &VirtualStorageProvider<MemoryFS>, points: usize, dimensions: usize) {
    let data: Vec<f32> = (0..points * dimensions)
        .map(|index| ((index * 17) % 251) as f32)
        .collect();
    write_bin(
        MatrixView::try_from(data.as_slice(), points, dimensions).unwrap(),
        &mut storage.create_for_write("/data.fbin").unwrap(),
    )
    .unwrap();
}

fn graph_config(degree: usize, alpha: f32) -> diskann::graph::Config {
    config::Builder::new_with(
        degree,
        config::MaxDegree::default_slack(),
        50,
        Metric::L2.into(),
        |builder| {
            builder.alpha(alpha);
        },
    )
    .build()
    .unwrap()
}

fn builder<'a>(
    storage: &'a VirtualStorageProvider<MemoryFS>,
    points: usize,
    dimensions: usize,
    budget_gib: f64,
    alpha: f32,
    parameters: PiPNNParameters,
) -> DiskIndexBuilder<'a, AdHoc<f32>, VirtualStorageProvider<MemoryFS>> {
    let params = DiskIndexBuildParameters::new_pipnn(
        MemoryBudget::try_from_gb(budget_gib).unwrap(),
        NumPQChunks::new_with(dimensions, dimensions).unwrap(),
        parameters,
    );
    let config = IndexConfiguration::new(
        Metric::L2,
        dimensions,
        points,
        ONE,
        1,
        graph_config(32, alpha),
    )
    .with_pseudo_rng_from_seed(42);
    let writer = DiskIndexWriter::new("/data.fbin".into(), "/index".into(), None, 4096).unwrap();
    DiskIndexBuilder::new(storage, params, config, writer).unwrap()
}

#[test]
fn pipnn_disk_build_rejects_configuration_dataset_mismatch() {
    let storage = VirtualStorageProvider::new_memory();
    write_data(&storage, 2, 8);
    let params = DiskIndexBuildParameters::new_pipnn(
        MemoryBudget::try_from_gb(10_000.0).unwrap(),
        NumPQChunks::new_with(4, 4).unwrap(),
        PiPNNParameters::default(),
    );
    let config = IndexConfiguration::new(Metric::L2, 4, 3, ONE, 1, graph_config(4, 1.2));
    let writer = DiskIndexWriter::new("/data.fbin".into(), "/index".into(), None, 4096).unwrap();
    let mut builder =
        DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, params, config, writer).unwrap();

    let error = builder.build().unwrap_err();
    assert!(format!("{error:?}").contains("configured dimension 4"));
    assert!(storage.exists("/index_pq_compressed.bin"));
}

#[test]
fn pipnn_disk_build_uses_common_pipeline() {
    let storage = VirtualStorageProvider::new_memory();
    let (points, dimensions) = (256, 8);
    write_data(&storage, points, dimensions);
    let mut builder = builder(&storage, points, dimensions, 1.0, 1.2, pipnn());

    builder.build().unwrap();

    assert!(storage.exists(&get_disk_index_file("/index")));
    assert!(storage.exists("/index_pq_compressed.bin"));
}

#[test]
fn pipnn_graph_adapter_writes_real_point_header() {
    let storage = VirtualStorageProvider::new_memory();
    let (points, dimensions) = (256, 8);
    write_data(&storage, points, dimensions);
    let parameters = pipnn();
    let builder = builder(&storage, points, dimensions, 1.0, 1.2, parameters.clone());
    let pool = create_thread_pool(1).unwrap();

    super::build_graph(&builder, pool.as_ref(), (&parameters).into()).unwrap();

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
fn pipnn_disk_build_falls_back_to_complete_vamana_pipeline() {
    let storage = VirtualStorageProvider::new_memory();
    let (points, dimensions) = (256, 8);
    write_data(&storage, points, dimensions);
    let mut builder = builder(&storage, points, dimensions, 0.000001, 1.3, pipnn());

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
    assert!(matches!(
        determine_build_strategy::<AdHoc<f32>>(
            &builder.index_configuration,
            builder.disk_build_param.build_memory_limit().in_bytes() as f64,
            builder.disk_build_param.build_quantization(),
        ),
        IndexBuildStrategy::Merged
    ));

    builder.build().unwrap();
    assert!(storage.exists(&get_disk_index_file("/index")));
}

#[test]
fn pipnn_disk_build_rejects_invalid_config_before_fallback() {
    let storage = VirtualStorageProvider::new_memory();
    let invalid = PiPNNParameters {
        c_max: 0,
        ..PiPNNParameters::default()
    };
    let params = DiskIndexBuildParameters::new_pipnn(
        MemoryBudget::try_from_gb(0.0001).unwrap(),
        NumPQChunks::new_with(1, 1).unwrap(),
        invalid,
    );
    let config = IndexConfiguration::new(Metric::L2, 1, 1, ONE, 1, graph_config(4, 1.2));
    let writer = DiskIndexWriter::new("/data.fbin".into(), "/index".into(), None, 4096).unwrap();

    let error = match DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, params, config, writer) {
        Ok(_) => panic!("invalid PiPNN config must not silently fall back to Vamana"),
        Err(error) => error,
    };

    assert!(format!("{error:?}").contains("c_max must be greater than zero"));
}
