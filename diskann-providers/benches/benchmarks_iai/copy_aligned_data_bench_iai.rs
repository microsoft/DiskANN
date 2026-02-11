/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::{BufWriter, Write};

use diskann::ANNResult;
use diskann_providers::{
    model::PQCompressedData,
    storage::{StorageReadProvider, StorageWriteProvider, VirtualStorageProvider},
    utils::{copy_aligned_data, write_metadata},
};
use iai_callgrind::black_box;
use rand::Rng;
use tempfile::TempDir;

pub const TEST_DATA_PATH: &str = "test_aligned_data.bin";
pub const BENCHMARK_ID: &str = "copy_aligned_data";

iai_callgrind::library_benchmark_group!(
    name = benchmark_copy_aligned_data_bench_iai;
    benchmarks = benchmark_copy_aligned_data_iai,
);

#[iai_callgrind::library_benchmark]
pub fn benchmark_copy_aligned_data_iai() {
    let tmp_dir = TempDir::with_prefix(BENCHMARK_ID).expect("Failed to create temporary directory");
    #[expect(
        clippy::disallowed_methods,
        reason = "Use physical file system rather than memory for testing the actual disk read/write"
    )]
    let storage_provider = VirtualStorageProvider::new_physical(tmp_dir.path());

    let num_points = 1_000_000;
    let num_pq_chunks = 192;
    //Write random data to file
    let mut writer = storage_provider.create_for_write(TEST_DATA_PATH).unwrap();
    generate_random_data(&mut writer, num_points, num_pq_chunks).unwrap();

    let mut pq_compressed_data = PQCompressedData::new(num_points, num_pq_chunks).unwrap();

    let mut reader = storage_provider.open_reader(TEST_DATA_PATH).unwrap();
    copy_aligned_data(
        black_box(&mut reader),
        black_box(pq_compressed_data.into_dto()),
        black_box(0),
    )
    .unwrap();
}

fn generate_random_data<Writer: Write>(
    writer: &mut Writer,
    npts: usize,
    dims: usize,
) -> ANNResult<()> {
    let mut writer = BufWriter::new(writer);
    write_metadata(&mut writer, npts, dims)?;

    let mut rng = diskann_providers::utils::create_rnd_in_tests();
    let data: Vec<u8> = (0..dims).map(|_| rng.random()).collect();
    for _ in 0..npts {
        writer.write_all(&data)?;
    }

    writer.flush()?;
    Ok(())
}
