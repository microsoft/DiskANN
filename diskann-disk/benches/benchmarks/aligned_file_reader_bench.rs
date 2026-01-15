/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::time::Duration;

use criterion::Criterion;
use diskann_disk::utils::aligned_file_reader::{
    traits::{AlignedFileReader, AlignedReaderFactory},
    AlignedFileReaderFactory, AlignedRead,
};
use diskann_providers::common::AlignedBoxWithSlice;

pub const TEST_INDEX_PATH: &str =
    "../test_data/disk_index_misc/disk_index_siftsmall_learn_256pts_R4_L50_A1.2_aligned_reader_test.index";

// MAX_IO_CONCURRENCY copied from the LinuxAlignedFileReader
const MAX_IO_CONCURRENCY: usize = 128;

/// Benchmark for all AlignedFileReaders (Linux and Windows).  Will run the specific reader
/// for the current OS.
///
/// # Run this before making your code change
/// cargo bench --bench bench_main -p disk-index -- --save-baseline prior_to_change
///
/// # Run this after making your code change to generate comparison metrics
/// cargo bench --bench bench_main -p disk-index -- --baseline prior_to_change
pub fn benchmark_aligned_file_reader(c: &mut Criterion) {
    // Get OS-specific aligned file reader
    let mut reader = AlignedFileReaderFactory::new(TEST_INDEX_PATH.to_string())
        .build()
        .unwrap();

    let read_length = 512;
    let num_read = MAX_IO_CONCURRENCY * 100; // The LinuxAlignedFileReader batches reads according to MAX_IO_CONCURRENCY.  Make sure we have many batches to handle.
    let mut aligned_mem = AlignedBoxWithSlice::<u8>::new(read_length * num_read, 512).unwrap();

    // create and add AlignedReads to the vector
    let mut mem_slices = aligned_mem
        .split_into_nonoverlapping_mut_slices(0..aligned_mem.len(), read_length)
        .unwrap();

    // Read the same data from disk over and over again.  We guarantee that it is not all zeros.
    let mut aligned_reads: Vec<AlignedRead<'_, u8>> = mem_slices
        .iter_mut()
        .map(|slice| AlignedRead::new(0, slice).unwrap())
        .collect();

    let mut group = c.benchmark_group("aligned_file_reader");
    group.sample_size(500);
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("Read using AlignedFileReader", |bencher| {
        bencher.iter(|| {
            let result = reader.read(&mut aligned_reads);

            // Make sure read completed successfully
            assert!(result.is_ok());
        })
    });
}
