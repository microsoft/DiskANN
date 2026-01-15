/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Read;

use diskann::ANNResult;
use diskann_providers::storage::StorageReadProvider;
use tracing::info;

use super::traits::AlignedFileReader;
use crate::utils::aligned_file_reader::AlignedRead;

pub struct StorageProviderAlignedFileReader {
    data: Vec<u8>,
}

impl StorageProviderAlignedFileReader {
    pub fn new(
        storage_provider: &impl StorageReadProvider,
        file_name: &str,
    ) -> ANNResult<StorageProviderAlignedFileReader> {
        info!("Loading data from {}", file_name);
        let file_length = storage_provider.get_length(file_name)?;

        let mut data = vec![0u8; file_length as usize];
        storage_provider
            .open_reader(file_name)?
            .read_exact(&mut data)?;

        Ok(StorageProviderAlignedFileReader { data })
    }
}

impl AlignedFileReader for StorageProviderAlignedFileReader {
    fn read(&mut self, read_requests: &mut [AlignedRead<u8>]) -> ANNResult<()> {
        for read in read_requests {
            let offset = read.offset();
            let len = read.aligned_buf().len();
            let aligned_buf = read.aligned_buf_mut();
            aligned_buf.copy_from_slice(&self.data[offset as usize..offset as usize + len]);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{Seek, SeekFrom},
    };

    use diskann_providers::storage::VirtualStorageProvider;
    use diskann_utils::test_data_root;
    use vfs::PhysicalFS;

    use super::*;
    use diskann_providers::common::AlignedBoxWithSlice;

    fn test_index_path() -> String {
        test_data_root()
            .join("disk_index_misc/disk_index_siftsmall_learn_256pts_R4_L50_A1.2_aligned_reader_test.index")
            .to_string_lossy()
            .to_string()
    }

    fn setup_reader() -> StorageProviderAlignedFileReader {
        let storage_provider = VirtualStorageProvider::new(PhysicalFS::new("/"));
        StorageProviderAlignedFileReader::new(&storage_provider, &test_index_path()).unwrap()
    }

    #[test]
    fn test_new_aligned_file_reader() {
        let reader = setup_reader();
        assert!(!(reader.data.is_empty()));
    }

    #[test]
    fn test_read() {
        let mut reader = setup_reader();

        let read_length = 512;
        let num_read = 10;
        let mut aligned_mem = AlignedBoxWithSlice::<u8>::new(read_length * num_read, 512).unwrap();

        // create and add AlignedReads to the vector
        let mut mem_slices = aligned_mem
            .split_into_nonoverlapping_mut_slices(0..aligned_mem.len(), read_length)
            .unwrap();

        let mut aligned_reads: Vec<AlignedRead<'_, u8>> = mem_slices
            .iter_mut()
            .enumerate()
            .map(|(i, slice)| {
                let offset = (i * read_length) as u64;
                AlignedRead::new(offset, slice).unwrap()
            })
            .collect();

        let result = reader.read(&mut aligned_reads);
        assert!(result.is_ok());

        // Assert that the actual data is correct.
        let mut file = File::open(test_index_path()).unwrap();
        for current_read in aligned_reads {
            let offset = current_read.offset();
            let mut expected = vec![0; current_read.aligned_buf().len()];
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.read_exact(&mut expected).unwrap();

            assert_eq!(
                expected,
                current_read.aligned_buf(),
                "aligned_buf did not contain the expected data"
            );
        }
    }
}
