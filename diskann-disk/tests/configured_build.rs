/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk builds selected by configuration through one public entry point.

#![cfg(feature = "pipnn")]

use std::io::Read;

use diskann_disk::{
    build::{
        builder::build::DiskIndexBuilder,
        configuration::{DiskIndexBuildConfig, NumPQChunks},
    },
    data_model::{AdHoc, GraphHeader},
    storage::DiskIndexWriter,
};
use diskann_providers::storage::{
    StorageReadProvider, StorageWriteProvider, VirtualStorageProvider,
};
use diskann_utils::{io::write_bin, views::rowmajor};
use diskann_vector::distance::Metric;
use vfs::MemoryFS;

const DATA: &str = "/data.fbin";
const SEED: u64 = 42;

fn storage() -> VirtualStorageProvider<MemoryFS> {
    let storage = VirtualStorageProvider::new_memory();
    // Points at 0, 1, and 4 on a line; padding keeps the stored dimension eight.
    let rows = [
        [0.0_f32; 8],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    write_bin(
        rowmajor::Ref::try_from_data(rows.as_flattened(), 3, 8).unwrap(),
        &mut storage.create_for_write(DATA).unwrap(),
    )
    .unwrap();
    storage
}

fn writer(prefix: &str) -> DiskIndexWriter {
    DiskIndexWriter::new(DATA.into(), prefix.into(), None, 4096).unwrap()
}

fn read(storage: &VirtualStorageProvider<MemoryFS>, path: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    storage
        .open_reader(path)
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}

#[cfg(feature = "pipnn")]
#[test]
fn pipnn_configuration_builds_exact_neighbors_and_search_pq_without_vamana_parameters() {
    use diskann::graph::pipnn;
    use std::num::NonZeroUsize;

    // With leaf_k=2 all three points are candidates. R=1 keeps the nearest
    // point: 0 -> 1, 1 -> 0, 4 -> 1. No Vamana L or RAM budget is supplied.
    let expected_neighbors = [1_u32, 0, 1];
    let storage = storage();
    let graph = pipnn::Config::builder(NonZeroUsize::new(1).unwrap())
        .seed(SEED)
        .build()
        .unwrap();
    let config = DiskIndexBuildConfig::pipnn(
        Metric::L2,
        8,
        3,
        1,
        NumPQChunks::new_with(1, 8).unwrap(),
        graph,
    )
    .with_seed(SEED);

    DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, config, writer("/pipnn"))
        .unwrap()
        .build()
        .unwrap();

    let bytes = read(&storage, "/pipnn_disk.index");
    let header = GraphHeader::try_from(&bytes[8..]).unwrap();
    assert_eq!(header.metadata().num_pts, 3);
    assert_eq!(header.metadata().dims, 8);
    assert_eq!(header.metadata().node_len, 40);
    for (point, expected) in expected_neighbors.iter().enumerate() {
        let row = 4096 + point * 40;
        let degree = u32::from_le_bytes(bytes[row + 32..row + 36].try_into().unwrap());
        let neighbor = u32::from_le_bytes(bytes[row + 36..row + 40].try_into().unwrap());
        assert_eq!(degree, 1, "point {point}");
        assert_eq!(neighbor, *expected, "point {point}");
    }
    // Two u32 header fields and one compressed byte per point.
    assert_eq!(read(&storage, "/pipnn_pq_compressed.bin").len(), 8 + 3);
    assert!(storage.exists("/pipnn_pq_pivots.bin"));
}
