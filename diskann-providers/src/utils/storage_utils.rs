/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Utilities for reading and writing data from the storage layer with generic reader/writer.
//! This is a replacement for the functions file_util.rs with generic reader/writer.

use std::{
    convert::TryInto,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    mem,
};

use bytemuck::Pod;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use diskann::{ANNError, ANNErrorKind, ANNResult, utils::read_exact_into};
use diskann_wide::{LoHi, SplitJoin};
use thiserror::Error;
use tracing::info;

use crate::utils::DatasetDto;

const DEFAULT_BUF_SIZE: usize = 1024 * 1024;

/// Metadata containing number of points and dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metadata {
    pub npoints: usize,
    pub ndims: usize,
}

/// Error type for metadata I/O operations
#[derive(Debug, Error)]
pub enum MetadataError<T, U> {
    #[error("num points conversion")]
    NumPoints(#[source] T),
    #[error("dim conversion")]
    Dim(#[source] U),
    #[error("writing binary results")]
    Write(#[source] std::io::Error),
}

impl<T, U> From<MetadataError<T, U>> for ANNError
where
    T: std::error::Error + Send + Sync + 'static,
    U: std::error::Error + Send + Sync + 'static,
{
    #[track_caller]
    fn from(err: MetadataError<T, U>) -> Self {
        ANNError::new(ANNErrorKind::IOError, err)
    }
}

/// Read binary metadata header (number of points and dimension) from a reader.
///
/// Reads 8 bytes total:
/// - First 4 bytes: number of points (u32, little-endian)
/// - Next 4 bytes: number of dimensions (u32, little-endian)
///
/// # Returns
/// * `Ok(Metadata)` - Metadata containing number of points and dimensions
/// * `Err(io::Error)` - If reading fails
pub fn read_metadata<Reader: Read>(reader: &mut Reader) -> std::io::Result<Metadata> {
    let raw = reader.read_u64::<LittleEndian>()?;
    let bytes: [u8; 8] = bytemuck::cast(raw);
    let LoHi {
        lo: npts_bytes,
        hi: ndims_bytes,
    } = bytes.split();
    let npoints = u32::from_le_bytes(npts_bytes) as usize;
    let ndims = u32::from_le_bytes(ndims_bytes) as usize;
    Ok(Metadata { npoints, ndims })
}

/// Write binary metadata header (number of points and dimension) to a writer.
///
/// Writes 8 bytes total:
/// - First 4 bytes: number of points (u32, little-endian)
/// - Next 4 bytes: number of dimensions (u32, little-endian)
///
/// This unified function accepts both `u32` and `usize` values, handling conversion appropriately:
/// - `u32` values are written directly (no conversion overhead)
/// - `usize` values are safely converted using `TryInto<u32>` (returns error on overflow)
///
/// # Returns
/// * `Ok(usize)` - Number of bytes written (always 8)
/// * `Err(MetadataError)` - If writing fails or conversion fails (usize > u32::MAX)
pub fn write_metadata<Writer: Write, N, D>(
    writer: &mut Writer,
    npts: N,
    ndims: D,
) -> Result<usize, MetadataError<N::Error, D::Error>>
where
    N: TryInto<u32>,
    D: TryInto<u32>,
    N::Error: std::error::Error + 'static,
    D::Error: std::error::Error + 'static,
{
    let npts_u32 = npts.try_into().map_err(MetadataError::NumPoints)?;
    let ndims_u32 = ndims.try_into().map_err(MetadataError::Dim)?;

    let bytes: [u8; 8] = LoHi::new(npts_u32.to_le_bytes(), ndims_u32.to_le_bytes()).join();
    writer.write_all(&bytes).map_err(MetadataError::Write)?;

    Ok(2 * std::mem::size_of::<u32>())
}

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

/// Copies data from a reader into a dataset with alignment.
/// This function reads vector data and aligns it within the given dataset.
///
/// # Arguments
/// * `reader` - A mutable reference to a type implementing the `Read` trait, where the data is read from.
/// * `dataset_dto` - Destination dataset DTO to which the data is copied. It must have the correct rounded dimension.
/// * `pts_offset` - Offset of points. Data will be loaded after this point in the dataset.
///
/// # Returns
/// * `npts` - Number of points read from the reader.
/// * `dim` - Point dimension read from the reader.
#[cfg(target_endian = "little")]
pub fn copy_aligned_data<T: Default + bytemuck::Pod, Reader: Read>(
    reader: &mut Reader,
    dataset_dto: DatasetDto<T>,
    pts_offset: usize,
) -> std::io::Result<(usize, usize)> {
    let mut reader = BufReader::with_capacity(DEFAULT_BUF_SIZE, reader);

    let metadata = read_metadata(&mut reader)?;
    let (npts, dim) = (metadata.npoints, metadata.ndims);
    let rounded_dim = dataset_dto.rounded_dim;
    let offset = pts_offset * rounded_dim;

    for i in 0..npts {
        let data_slice =
            &mut dataset_dto.data[offset + i * rounded_dim..offset + i * rounded_dim + dim];

        // Casting Pod type to bytes always succeeds (u8 has alignment of 1)
        let byte_slice: &mut [u8] = bytemuck::must_cast_slice_mut(data_slice);
        reader.read_exact(byte_slice)?;

        let remaining = &mut dataset_dto.data
            [offset + i * rounded_dim + dim..offset + i * rounded_dim + rounded_dim];
        remaining.fill_with(Default::default);
    }

    Ok((npts, dim))
}

/// Load a list of type T data from a stream.
/// # Arguments
/// * `reader` - a stream reader.
/// * `offset` - start offset of the data.
pub fn load_bin<T: Pod + Default, Reader: Read + Seek>(
    reader: &mut Reader,
    offset: usize,
) -> std::io::Result<(Vec<T>, usize, usize)> {
    let mut reader = BufReader::new(reader);
    reader.seek(std::io::SeekFrom::Start(offset as u64))?;
    let metadata = read_metadata(&mut reader)?;
    let (npts, dim) = (metadata.npoints, metadata.ndims);

    let size = npts * dim * std::mem::size_of::<T>();

    let buf: Vec<T> = read_exact_into(&mut reader, npts * dim)?;
    info!(
        "bin: #pts = {}, #dims = {}, offset = {} size = {}B",
        npts, dim, offset, size
    );

    Ok((buf, npts, dim))
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
    write_metadata(writer, npts, ndims)?;
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
    write_metadata(writer, npts, ndims)?;

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

macro_rules! save_bin {
    ($name:ident, $t:ty, $write_func:ident) => {
        /// Write data into the storage system.
        pub fn $name<W: Write + Seek>(
            writer: &mut W,
            data: &[$t],
            num_pts: usize,
            dims: usize,
            offset: usize,
        ) -> ANNResult<usize> {
            writer.seek(SeekFrom::Start(offset as u64))?;
            let bytes_written = num_pts * dims * mem::size_of::<$t>() + 2 * mem::size_of::<u32>();

            write_metadata(writer, num_pts, dims)?;
            info!(
                "bin: #pts = {}, #dims = {}, size = {}B",
                num_pts, dims, bytes_written
            );

            for item in data.iter() {
                writer.$write_func::<LittleEndian>(*item)?;
            }

            writer.flush()?;

            info!("Finished writing bin.");
            Ok(bytes_written)
        }
    };
}

save_bin!(save_bin_f32, f32, write_f32);
save_bin!(save_bin_u64, u64, write_u64);
save_bin!(save_bin_u32, u32, write_u32);

#[cfg(test)]
mod storage_util_test {
    use crate::storage::{StorageReadProvider, StorageWriteProvider, VirtualStorageProvider};
    use tempfile::tempfile;

    use super::*;
    pub const DIM_8: usize = 8;

    #[test]
    fn read_metadata_test() {
        let file_name = "/test_read_metadata_test.bin";
        let data = [200, 0, 0, 0, 128, 0, 0, 0]; // 200 and 128 in little endian bytes (u32)
        let storage_provider = VirtualStorageProvider::new_memory();
        {
            let mut file = storage_provider
                .create_for_write(file_name)
                .expect("Could not create file");
            file.write_all(&data)
                .expect("Should be able to write sample file");
        }

        let mut reader = storage_provider.open_reader(file_name).unwrap();
        match read_metadata(&mut reader) {
            Ok(metadata) => {
                assert_eq!(metadata.npoints, 200);
                assert_eq!(metadata.ndims, 128);
            }
            Err(_e) => {}
        }
        storage_provider
            .delete(file_name)
            .expect("Should be able to delete sample file");
    }

    #[test]
    fn read_metadata_i32_compatibility_test() {
        // Test that read_metadata (u32) can read data written as i32
        let file_name = "/test_read_metadata_i32_compat.bin";
        let npts = 200i32;
        let dims = 128i32;
        let storage_provider = VirtualStorageProvider::new_memory();
        {
            let mut file = storage_provider
                .create_for_write(file_name)
                .expect("Could not create file");
            // Write as i32 (old format)
            file.write_i32::<LittleEndian>(npts).unwrap();
            file.write_i32::<LittleEndian>(dims).unwrap();
        }

        // Read as u32 (new format)
        let mut reader = storage_provider.open_reader(file_name).unwrap();
        let metadata = read_metadata(&mut reader).unwrap();

        assert_eq!(metadata.npoints, 200);
        assert_eq!(metadata.ndims, 128);

        storage_provider
            .delete(file_name)
            .expect("Should be able to delete sample file");
    }

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
    fn load_bin_test() {
        let file_name = "/load_bin_test";
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = data.len();
        let dims = 1;
        let storage_provider = VirtualStorageProvider::new_memory();
        let bytes_written = save_bin_u64(
            &mut storage_provider.create_for_write(file_name).unwrap(),
            &data,
            num_pts,
            dims,
            0,
        )
        .unwrap();
        assert_eq!(bytes_written, 32);

        let (load_data, load_num_pts, load_dims) =
            load_bin::<u64, _>(&mut storage_provider.open_reader(file_name).unwrap(), 0).unwrap();
        assert_eq!(load_num_pts, num_pts);
        assert_eq!(load_dims, dims);
        assert_eq!(load_data, data);
        storage_provider.delete(file_name).unwrap();
    }

    #[test]
    fn load_bin_offset_test() {
        let offset: usize = 32;
        let file_name = "/load_bin_offset_test";
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = data.len();
        let dims = 1;
        let storage_provider = VirtualStorageProvider::new_memory();
        let bytes_written = save_bin_u64(
            &mut storage_provider.create_for_write(file_name).unwrap(),
            &data,
            num_pts,
            dims,
            offset,
        )
        .unwrap();
        assert_eq!(bytes_written, 32);

        let (load_data, load_num_pts, load_dims) = load_bin::<u64, _>(
            &mut storage_provider.open_reader(file_name).unwrap(),
            offset,
        )
        .unwrap();
        assert_eq!(load_num_pts, num_pts);
        assert_eq!(load_dims, dims);
        assert_eq!(load_data, data);
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
    fn save_bin_test() {
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = data.len();
        let dims = 1;
        let mut file = tempfile().unwrap();
        let bytes_written = save_bin_u64::<_>(&mut file, &data, num_pts, dims, 0).unwrap();
        assert_eq!(bytes_written, 32);

        let mut buffer = vec![];
        file.seek(SeekFrom::Start(0)).unwrap();
        let metadata = read_metadata(&mut file).unwrap();

        file.read_to_end(&mut buffer).unwrap();
        let data_read: Vec<u64> = buffer
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            .collect();

        assert_eq!(num_pts, metadata.npoints);
        assert_eq!(dims, metadata.ndims);
        assert_eq!(data, data_read);
    }

    #[test]
    fn write_metadata_unified_test() {
        let mut buffer = Vec::new();

        // Test with u32 values (no conversion)
        let result = write_metadata(&mut buffer, 200u32, 128u32);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 8);

        // Test with usize values (safe conversion)
        buffer.clear();
        let result = write_metadata(&mut buffer, 200usize, 128usize);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 8);

        // Test mixed types
        buffer.clear();
        let result = write_metadata(&mut buffer, 200usize, 128u32);
        assert!(result.is_ok());

        // Verify the written data
        let mut cursor = std::io::Cursor::new(&buffer);
        let metadata = read_metadata(&mut cursor).unwrap();
        assert_eq!(metadata.npoints, 200);
        assert_eq!(metadata.ndims, 128);
    }

    #[test]
    fn metadata_error_types_test() {
        // Test NumPoints error
        let large_value = u32::MAX as usize + 1;
        let result = write_metadata(&mut Vec::new(), large_value, 128usize);
        assert!(matches!(result, Err(MetadataError::NumPoints(_))));

        // Test Dim error
        let result = write_metadata(&mut Vec::new(), 128usize, large_value);
        assert!(matches!(result, Err(MetadataError::Dim(_))));

        // Test Write error
        struct FailingWriter;
        impl std::io::Write for FailingWriter {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "fail",
                ))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let result = write_metadata(&mut FailingWriter, 200u32, 128u32);
        assert!(matches!(result, Err(MetadataError::Write(_))));
    }

    #[test]
    fn metadata_error_to_ann_error_test() {
        use diskann::{ANNError, ANNErrorKind};

        // Test MetadataError -> ANNError conversion
        let large_value = u32::MAX as usize + 1;
        let result = write_metadata(&mut Vec::new(), large_value, 128usize);
        let metadata_err = result.unwrap_err();
        let ann_error: ANNError = metadata_err.into();

        assert_eq!(ann_error.kind(), ANNErrorKind::IOError);

        // Check that the error message contains information about the conversion
        let error_str = ann_error.to_string();
        assert!(error_str.contains("num points conversion"));
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn test_copy_aligned_data() -> std::io::Result<()> {
        let mut data = Vec::with_capacity(24);
        data.extend_from_slice(&(2_i32.to_le_bytes()));
        data.extend_from_slice(&(2_i32.to_le_bytes()));
        data.extend_from_slice(&(1_f32.to_le_bytes()));
        data.extend_from_slice(&(2_f32.to_le_bytes()));
        data.extend_from_slice(&(3_f32.to_le_bytes()));
        data.extend_from_slice(&(4_f32.to_le_bytes()));

        let mut reader = Cursor::new(data);

        let rounded_dim = 4;
        let mut aligned_data = vec![0f32; 2 * rounded_dim];
        let dataset_dto = DatasetDto::<f32> {
            data: &mut aligned_data,
            rounded_dim,
        };

        let (npts, dim) = copy_aligned_data(&mut reader, dataset_dto, 0)?;

        assert_eq!(npts, 2);
        assert_eq!(dim, 2);

        assert_eq!(aligned_data, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);

        Ok(())
    }
}
