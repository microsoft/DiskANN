/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Utilities for reading and writing data from the storage layer with generic reader/writer.
//! This is a replacement for the functions file_util.rs with generic reader/writer.

use std::io::{BufReader, Read, Seek, Write};

use bytemuck::Pod;
use byteorder::{LittleEndian, ReadBytesExt};
use diskann::ANNResult;
use diskann_utils::{
    io::{Metadata, ReadBinError, SaveBinError, read_bin, write_bin},
    views::{Matrix, MatrixView},
};

/// Load a list of vector ids from the stream.
pub fn load_vector_ids<Reader: Read>(reader: &mut Reader) -> std::io::Result<(usize, Vec<u32>)> {
    // The first 4 bytes are the number of vector ids.
    // The rest of the file are the vector ids in the format of usize.
    // The vector ids are sorted in ascending order.
    let mut reader = BufReader::new(reader);
    let num_ids = reader.read_u32::<LittleEndian>()? as usize;

    let mut ids = Vec::with_capacity(num_ids);
    for _ in 0..num_ids {
        let id = reader.read_u32::<LittleEndian>()?;
        ids.push(id);
    }

    Ok((num_ids, ids))
}

/// Load a list of type T data from a stream.
/// # Arguments
/// * `reader` - a stream reader.
/// * `offset` - start offset of the data.
pub fn read_bin_from<T: Pod>(
    reader: &mut (impl Read + Seek),
    offset: usize,
) -> Result<Matrix<T>, ReadBinError> {
    reader.seek(std::io::SeekFrom::Start(offset as u64))?;
    read_bin(reader)
}

/// Write a matrix at the given byte offset.
pub fn write_bin_from<T: Pod>(
    data: MatrixView<'_, T>,
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

/// Save vector data to stream with aligned dimension.
/// # Arguments
/// * `writer` - A writer to write the data to storage system.
/// * `data` - information data
/// * `npts` - number of points
/// * `ndims` - point dimension
/// * `aligned_dim` - aligned dimension
/// * `offset` - data offset in file
pub fn save_data_in_base_dimensions<T: Default + Copy + bytemuck::Pod, Writer: Write + Seek>(
    writer: &mut Writer,
    data: &[T],
    npts: usize,
    ndims: usize,
    aligned_dim: usize,
    offset: usize,
) -> ANNResult<usize> {
    let bytes_written = 2 * std::mem::size_of::<u32>() + npts * ndims * (std::mem::size_of::<T>());

    writer.seek(std::io::SeekFrom::Start(offset as u64))?;
    Metadata::new(npts, ndims)?.write(writer)?;

    for i in 0..npts {
        let start = i * aligned_dim;
        let end = start + ndims;
        let vector_slice = &data[start..end];
        // Casting Pod type to bytes always succeeds (u8 has alignment of 1)
        let bytes: &[u8] = bytemuck::must_cast_slice(vector_slice);
        writer.write_all(bytes)?;
    }
    writer.flush()?;
    Ok(bytes_written)
}

#[cfg(test)]
mod storage_util_test {
    use crate::storage::{StorageReadProvider, StorageWriteProvider, VirtualStorageProvider};
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::SeekFrom;
    use tempfile::tempfile;

    use super::*;
    pub const DIM_8: usize = 8;

    #[test]
    fn load_vector_ids_test() {
        let file_name = "/load_vector_ids_test";
        let ids = vec![0u32, 1u32, 2u32];
        let num_ids = ids.len();
        let storage_provider = VirtualStorageProvider::new_memory();
        {
            let mut writer = storage_provider.create_for_write(file_name).unwrap();
            writer.write_u32::<LittleEndian>(num_ids as u32).unwrap();
            for item in ids.iter() {
                writer.write_u32::<LittleEndian>(*item).unwrap();
            }
        }

        let load_data =
            load_vector_ids(&mut storage_provider.open_reader(file_name).unwrap()).unwrap();
        assert_eq!(load_data, (num_ids, ids));
        storage_provider
            .delete(file_name)
            .expect("Should be able to delete sample file");
    }

    #[test]
    fn test_read_bin_from() {
        let file_name = "/read_bin_from";
        let data = vec![0u64, 1u64, 2u64];
        let storage_provider = VirtualStorageProvider::new_memory();
        let view = MatrixView::column_vector(data.as_slice());
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
        let view = MatrixView::column_vector(data.as_slice());
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
    fn save_data_in_base_dimensions_test() {
        //npoints=2, dim=8
        let data: [u8; 72] = [
            2, 0, 0, 0, 8, 0, 0, 0, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
            0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40,
            0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x10, 0x41, 0x00, 0x00,
            0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 0x00, 0x00, 0x50, 0x41,
            0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x00, 0x80, 0x41,
        ];
        let num_points = 2;
        let dim = DIM_8;
        let data_file = "/save_data_in_base_dimensions_test.data";
        let storage_provider = VirtualStorageProvider::new_memory();
        match save_data_in_base_dimensions(
            &mut storage_provider.create_for_write(data_file).unwrap(),
            &data,
            num_points,
            dim,
            DIM_8,
            0,
        ) {
            Ok(num) => {
                assert!(storage_provider.exists(data_file));
                assert_eq!(
                    num,
                    2 * std::mem::size_of::<u32>() + num_points * dim * std::mem::size_of::<u8>()
                );
                storage_provider
                    .delete(data_file)
                    .expect("Failed to delete file");
            }
            Err(e) => {
                storage_provider
                    .delete(data_file)
                    .expect("Failed to delete file");
                panic!("{}", e)
            }
        }
    }

    #[test]
    fn write_bin_from_test() {
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = data.len();
        let dims = 1;
        let mut file = tempfile().unwrap();
        let view = MatrixView::column_vector(data.as_slice());
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
