/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Utilities for reading and writing data from the storage layer with generic reader/writer.
//! This is a replacement for the functions file_util.rs with generic reader/writer.

use std::io::{Read, Seek, Write};

use bytemuck::Pod;
use diskann::ANNResult;
use diskann_utils::{
    io::{Metadata, ReadBinError, SaveBinError, read_bin, write_bin},
    views::rowmajor,
};

/// Load a list of type T data from a stream.
/// # Arguments
/// * `reader` - a stream reader.
/// * `offset` - start offset of the data.
pub fn read_bin_from<T: Pod>(
    reader: &mut (impl Read + Seek),
    offset: usize,
) -> Result<rowmajor::Owned<T>, ReadBinError> {
    reader.seek(std::io::SeekFrom::Start(offset as u64))?;
    read_bin(reader)
}

/// Write a matrix at the given byte offset.
pub fn write_bin_from<T: Pod>(
    data: rowmajor::Ref<'_, T>,
    writer: &mut (impl Write + Seek),
    offset: usize,
) -> Result<usize, SaveBinError> {
    writer.seek(std::io::SeekFrom::Start(offset as u64))?;
    write_bin(data, writer)
}

/// Save the byte array to storage.
pub fn save_bytes<Writer: Write + Seek>(
    writer: &mut Writer,
    data: &[u8],
    npts: usize,
    ndims: usize,
    offset: usize,
) -> ANNResult<usize> {
    writer.seek(std::io::SeekFrom::Start(offset as u64))?;
    Metadata::new(npts, ndims)?.write(writer)?;
    writer.write_all(data)?;
    writer.flush()?;

    Ok(data.len() + 2 * std::mem::size_of::<u32>())
}

#[cfg(test)]
mod storage_util_test {
    use crate::storage::{StorageReadProvider, StorageWriteProvider, VirtualStorageProvider};
    use diskann_utils::views::rowmajor::Matrix;
    use std::io::SeekFrom;
    use tempfile::tempfile;

    use super::*;

    #[test]
    fn test_read_bin_from() {
        let file_name = "/read_bin_from";
        let data = vec![0u64, 1u64, 2u64];
        let storage_provider = VirtualStorageProvider::new_memory();
        let view = rowmajor::Ref::column_vector(data.as_slice());
        let bytes_written = write_bin_from(
            view,
            &mut storage_provider.create_for_write(file_name).unwrap(),
            0,
        )
        .unwrap();
        assert_eq!(bytes_written, 32);

        let loaded =
            read_bin_from::<u64>(&mut storage_provider.open_reader(file_name).unwrap(), 0).unwrap();
        assert_eq!(loaded.as_view(), view);
        storage_provider.delete(file_name).unwrap();
    }

    #[test]
    fn test_read_bin_from_offset_test() {
        let offset: usize = 32;
        let file_name = "/read_bin_from_offset_test";
        let data = vec![0u64, 1u64, 2u64];
        let storage_provider = VirtualStorageProvider::new_memory();
        let view = rowmajor::Ref::column_vector(data.as_slice());
        let bytes_written = write_bin_from(
            view,
            &mut storage_provider.create_for_write(file_name).unwrap(),
            offset,
        )
        .unwrap();
        assert_eq!(bytes_written, 32);

        let loaded = read_bin_from::<u64>(
            &mut storage_provider.open_reader(file_name).unwrap(),
            offset,
        )
        .unwrap();
        assert_eq!(loaded.as_view(), view);
        storage_provider.delete(file_name).unwrap();
    }

    #[test]
    fn write_bin_from_test() {
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = data.len();
        let dims = 1;
        let mut file = tempfile().unwrap();
        let view = rowmajor::Ref::column_vector(data.as_slice());
        let bytes_written = write_bin_from(view, &mut file, 0).unwrap();
        assert_eq!(bytes_written, 32);

        let mut buffer = vec![];
        file.seek(SeekFrom::Start(0)).unwrap();
        let metadata = Metadata::read(&mut file).unwrap();

        file.read_to_end(&mut buffer).unwrap();
        let data_read: Vec<u64> = buffer
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();

        assert_eq!(num_pts, metadata.npoints());
        assert_eq!(dims, metadata.ndims());
        assert_eq!(data, data_read);
    }
}
