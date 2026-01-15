/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! File operations

use std::{
    io,
    io::{BufReader, Read, Seek},
    mem::size_of,
};

use crate::storage::{StorageReadProvider, StorageWriteProvider};
use byteorder::{LittleEndian, ReadBytesExt};
use diskann::{ANNError, ANNResult, utils::IntoUsize};
use diskann_utils::views::Matrix;
use tracing::info;

use crate::{
    common::AlignedBoxWithSlice,
    utils::{
        DatasetDto, copy_aligned_data,
        storage_utils::{Metadata, read_metadata},
    },
};

/// Read metadata of data file.
pub fn load_metadata_from_file<ReadProvider: StorageReadProvider>(
    storage_provider: &ReadProvider,
    file_name: &str,
) -> std::io::Result<Metadata> {
    let mut file = storage_provider.open_reader(file_name)?;
    read_metadata(&mut file)
}

/// Read metadata from data content. Use include_bytes! marco to get reference of a byte array.
pub fn load_metadata_from_bytes(bytes: &[u8]) -> std::io::Result<Metadata> {
    let mut cursor = std::io::Cursor::new(bytes);
    read_metadata(&mut cursor)
}

/// Read the deleted vertex ids from file.
pub fn load_ids_to_delete_from_file<ReadProvider: StorageReadProvider>(
    storage_provider: &ReadProvider,
    file_name: &str,
) -> std::io::Result<Vec<u32>> {
    // The first 4 bytes are the number of vector ids.
    // The rest of the file are the vector ids in the format of usize.
    // The vector ids are sorted in ascending order.
    let file = storage_provider.open_reader(file_name)?;
    let mut reader = BufReader::new(file);
    let num_ids = reader.read_u32::<LittleEndian>()? as usize;

    let mut ids = Vec::with_capacity(num_ids);
    for _ in 0..num_ids {
        let id = reader.read_u32::<LittleEndian>()?;
        ids.push(id);
    }

    Ok(ids)
}

/// Copy aligned data from a file to a dataset
///
/// # Arguments
///
/// * `storage_provider` - A reference to an implementation of `StorageReadProvider` to read the file.
/// * `bin_file` - The filename where the data is stored.
/// * `dataset_dto` - The destination dataset DTO to which the data is copied.
/// * `pts_offset` - The offset of points. Data will be loaded after this point in the dataset.
///
/// # Returns
///
/// * A `Result` containing a tuple with:
///     * `npts` - The number of points read from the `bin_file`.
///     * `dim` - The point dimension read from the `bin_file`.
pub fn copy_aligned_data_from_file<T: Default + Copy + bytemuck::Pod>(
    storage_provider: &impl StorageReadProvider,
    bin_file: &str,
    dataset_dto: DatasetDto<T>,
    pts_offset: usize,
) -> std::io::Result<(usize, usize)> {
    let mut reader = storage_provider.open_reader(bin_file)?;
    copy_aligned_data(&mut reader, dataset_dto, pts_offset)
}

/// Loads a binary file with aligned memory and returns the data along with metadata.
///
/// # Parameters
/// - `storage_provider`: A reference to an object that implements the `StorageReadProvider` trait, used to read the binary file.
/// - `bin_file`: A string slice that holds the name of the binary file to be read.
///
/// # Returns
/// - `ANNResult<(AlignedBoxWithSlice<T>, usize, usize, usize)>`: A result containing a tuple with the aligned data, number of points, dimensions, and rounded dimensions.
///
/// # Errors
/// - Returns an `ANNError` if there is a file size mismatch or if the requested memory size cannot be allocated.
///
/// # Type Parameters
/// - `T`: The type of the elements in the binary file. Must implement `Default`, `Copy`, `Sized`, and `bytemuck::Pod`.
#[inline]
pub fn load_aligned_bin<T: Default + Copy + Sized + bytemuck::Pod>(
    storage_provider: &impl StorageReadProvider,
    bin_file: &str,
) -> ANNResult<(AlignedBoxWithSlice<T>, usize, usize, usize)> {
    let size_of_t = size_of::<T>();
    let (npts, dim, file_size): (usize, usize, usize);
    {
        info!("Reading (with alignment) bin file: {bin_file}");
        file_size = storage_provider.get_length(bin_file)? as usize;

        let mut file = storage_provider.open_reader(bin_file)?;
        let metadata = read_metadata(&mut file)?;
        (npts, dim) = (metadata.npoints, metadata.ndims);
    }

    let rounded_dim = dim.next_multiple_of(8);
    let expected_actual_file_size = npts * dim * size_of_t + 2 * size_of::<u32>();

    if file_size != expected_actual_file_size {
        return Err(ANNError::log_index_error(format_args!(
            "ERROR: File size mismatch. Actual size is {} while expected size is {}
        npts = {}, #dims = {}, aligned_dim = {}",
            file_size, expected_actual_file_size, npts, dim, rounded_dim
        )));
    }

    info!("Metadata: #pts = {npts}, #dims = {dim}, aligned_dim = {rounded_dim}...");

    let alloc_size = npts * rounded_dim;
    let alignment = 8 * size_of_t;
    info!(
        "allocating aligned memory of {} bytes... ",
        alloc_size * size_of_t
    );
    if !(alloc_size * size_of_t).is_multiple_of(alignment) {
        return Err(ANNError::log_index_error(format_args!(
            "Requested memory size is not a multiple of {}. Can not be allocated.",
            alignment
        )));
    }

    let mut data = AlignedBoxWithSlice::<T>::new(alloc_size, alignment)?;
    let dto = DatasetDto {
        data: &mut data,
        rounded_dim,
    };

    info!("done. Copying data to mem_aligned buffer...");

    let (_, _) = copy_aligned_data_from_file(storage_provider, bin_file, dto, 0)?;

    Ok((data, npts, dim, rounded_dim))
}

/// Open a file to write
/// # Arguments
/// * `writer` - mutable File reference
/// * `file_name` - file name
#[inline]
pub fn open_file_to_write<StorageProvider: StorageWriteProvider>(
    storage_provider: &StorageProvider,
    file_name: &str,
) -> std::io::Result<StorageProvider::Writer> {
    storage_provider.create_for_write(file_name)
}

/// Check whether file exists or not
pub fn file_exists<StorageProvider: StorageReadProvider>(
    storage_provider: &StorageProvider,
    filename: &str,
) -> bool {
    storage_provider.exists(filename)
}

/// Read data file
/// # Arguments
/// * `bin_file` - filename where the data is
/// * `file_offset` - data offset in file
/// * `data` - information data
/// * `npts` - number of points
/// * `ndims` - point dimension
pub fn load_bin<T: Copy + bytemuck::Pod, StorageReader: StorageReadProvider>(
    storage_read_provider: &StorageReader,
    bin_file: &str,
    file_offset: usize,
) -> std::io::Result<(Vec<T>, usize, usize)> {
    let mut reader = storage_read_provider.open_reader(bin_file)?;
    reader.seek(std::io::SeekFrom::Start(file_offset as u64))?;
    let metadata = read_metadata(&mut reader)?;
    let (npts, dim) = (metadata.npoints, metadata.ndims);

    let size = npts * dim * std::mem::size_of::<T>();
    let mut buf = vec![0u8; size];
    reader.read_exact(&mut buf)?;

    let data: &[T] = bytemuck::cast_slice(&buf);

    Ok((data.to_vec(), npts, dim))
}

/// Read data file
/// # Arguments
/// * `bin_file` - filename where the data is
///
/// File structure:
/// * Header: [npts (4 bytes), dim (4 bytes), total_vecs (4 bytes)]
/// * Vector length information: [len1 (4 bytes), len2 (4 bytes), ..., len npts (4 bytes)]
/// * Data: [vec1 (len1 * dimension bytes), vec2 (len2 * dimension bytes), ..., vec npts (len npts * dimension bytes)]
///
/// Returns the header information along with the loaded vectors as a vec of vecs
#[allow(clippy::type_complexity)]
pub fn load_multivec_bin<T: Copy + bytemuck::Pod + Default, StorageReader: StorageReadProvider>(
    storage_read_provider: &StorageReader,
    bin_file: &str,
) -> ANNResult<(Vec<Matrix<T>>, usize, usize, usize)> {
    let mut reader = storage_read_provider.open_reader(bin_file)?;

    let (num_points, dimension, total_results) = {
        let mut buffer = [0u8; std::mem::size_of::<u32>()];

        reader.read_exact(&mut buffer)?;
        let num_points = u32::from_le_bytes(buffer).into_usize();

        reader.read_exact(&mut buffer)?;
        let dimension = u32::from_le_bytes(buffer).into_usize();

        reader.read_exact(&mut buffer)?;
        let total_results = u32::from_le_bytes(buffer).into_usize();

        (num_points, dimension, total_results)
    };

    let mut vec_lengths = vec![0u32; num_points];
    let mut is_any_vector_zero_length = false;
    reader.read_exact(bytemuck::must_cast_slice_mut::<u32, u8>(&mut vec_lengths))?;
    vec_lengths.iter_mut().for_each(|x| {
        *x = u32::from_le(*x);
        if *x == 0 {
            is_any_vector_zero_length = true;
        }
    });

    if is_any_vector_zero_length {
        return Err(ANNError::log_index_error(format_args!(
            "Vector length cannot be zero"
        )));
    }

    // compute sum of vector lengths and check that it's equal to total_results
    let sum_vec_lengths: usize = vec_lengths.iter().map(|&x| x as usize).sum();
    if sum_vec_lengths != total_results {
        return Err(ANNError::log_index_error(format_args!(
            "Sum of vector lengths ({}) does not match total_results ({})",
            sum_vec_lengths, total_results
        )));
    }

    let mut all_vectors: Vec<Matrix<T>> = Vec::with_capacity(num_points);

    for &length in &vec_lengths {
        let mut vectors = Matrix::<T>::new(T::default(), length as usize, dimension);
        reader.read_exact(bytemuck::must_cast_slice_mut::<T, u8>(
            vectors.as_mut_slice(),
        ))?;

        all_vectors.push(vectors);
    }

    Ok((all_vectors, num_points, dimension, total_results))
}

/// Get file size
pub fn get_file_size<StorageReader: StorageReadProvider>(
    storage_read_provider: &StorageReader,
    filename: &str,
) -> io::Result<u64> {
    storage_read_provider.get_length(filename)
}

#[cfg(test)]
mod file_util_test {
    use crate::storage::{StorageWriteProvider, VirtualStorageProvider};
    use vfs::{FileSystem, MemoryFS, SeekAndWrite};

    use super::*;
    use crate::utils::save_bin_u64;

    #[test]
    fn get_file_size_test() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::default());
        let file_name = "/test_get_file_size_test.bin";

        storage_provider
            .create_for_write(file_name)
            .unwrap()
            .write_all(b"test get file size!")
            .expect("Write did not succeed");

        let result = get_file_size(&storage_provider, file_name);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 19);
        storage_provider
            .delete(file_name)
            .expect("Should be able to delete temp file");
    }

    #[test]
    fn load_metadata_from_bytes_test() {
        let data = [200, 0, 0, 0, 128, 0, 0, 0]; // 200 and 128 in little endian bytes (u32)

        let result = load_metadata_from_bytes(&data);
        assert!(result.is_ok());
        let metadata = result.unwrap();
        assert_eq!(metadata.npoints, 200);
        assert_eq!(metadata.ndims, 128);
    }

    #[test]
    fn load_metadata_test() {
        let filesystem = MemoryFS::new();
        let storage_provider = VirtualStorageProvider::new(filesystem);
        let file_name = "/test_load_metadata_test.bin";
        let data = [200, 0, 0, 0, 128, 0, 0, 0]; // 200 and 128 in little endian bytes
        {
            let mut writer = storage_provider.create_for_write(file_name).unwrap();
            writer
                .write_all(&data)
                .expect("Failed to write sample file");
        }
        match load_metadata_from_file(&storage_provider, file_name) {
            Ok(metadata) => {
                assert!(metadata.npoints == 200);
                assert!(metadata.ndims == 128);
            }
            Err(_e) => {}
        }
        storage_provider
            .delete(file_name)
            .expect("Failed to delete file");
    }

    #[test]
    fn load_data_test() {
        let filesystem = MemoryFS::new();
        let storage_provider = VirtualStorageProvider::new(filesystem);

        let file_name = "/test_load_data_test.bin";
        //npoints=2, dim=8, 2 vectors [1.0;8] [2.0;8]
        let data: [u8; 72] = [
            2, 0, 0, 0, 8, 0, 0, 0, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
            0x40, 0x40, 0x00, 0x00, 0x80, 0x40, 0x00, 0x00, 0xa0, 0x40, 0x00, 0x00, 0xc0, 0x40,
            0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x10, 0x41, 0x00, 0x00,
            0x20, 0x41, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x40, 0x41, 0x00, 0x00, 0x50, 0x41,
            0x00, 0x00, 0x60, 0x41, 0x00, 0x00, 0x70, 0x41, 0x00, 0x00, 0x80, 0x41,
        ];
        {
            let mut writer = storage_provider.create_for_write(file_name).unwrap();
            writer
                .write_all(&data)
                .expect("Failed to write sample file");
        }

        // Create aligned buffer for 2 points, 8 dimensions each, rounded to 8
        let rounded_dim = 8;
        let num_points = 2;
        let mut dataset = AlignedBoxWithSlice::<f32>::new(
            num_points * rounded_dim,
            std::mem::size_of::<f32>() * 8,
        )
        .unwrap();

        let dto = DatasetDto {
            data: &mut dataset,
            rounded_dim,
        };

        match copy_aligned_data_from_file(&storage_provider, file_name, dto, 0) {
            Ok((num_points, dim)) => {
                storage_provider
                    .delete(file_name)
                    .expect("Failed to delete file");
                assert_eq!(num_points, 2);
                assert_eq!(dim, 8);
                assert_eq!(dataset.as_slice().len(), 16);

                // Check the first vector: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                let first_vector = &dataset.as_slice()[0..8];
                assert_eq!(first_vector, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

                // Check the second vector: [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
                let second_vector = &dataset.as_slice()[8..16];
                assert_eq!(
                    second_vector,
                    &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
                );
            }
            Err(e) => {
                storage_provider
                    .delete(file_name)
                    .expect("Failed to delete file");
                panic!("{}", e)
            }
        }
    }

    #[test]
    fn open_file_to_write_test() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::new());
        let file_name = "/test_open_file_to_write_test.bin";
        {
            let mut writer: Box<dyn SeekAndWrite + Send> =
                storage_provider.create_for_write(file_name).unwrap();
            let data = [200, 0, 0, 0, 128, 0, 0, 0];
            writer.write(&data).expect("Failed to write sample file");
        }
        open_file_to_write(&storage_provider, file_name).unwrap();

        storage_provider
            .delete(file_name)
            .expect("Failed to delete file");
    }

    #[test]
    fn load_bin_test() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::new());

        let file_name = "/load_bin_test";
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = data.len();
        let dims = 1;
        {
            let mut file_write = storage_provider.create_for_write(file_name).unwrap();
            let bytes_written = save_bin_u64(&mut file_write, &data, num_pts, dims, 0).unwrap();
            assert_eq!(bytes_written, 32);
        }

        let (load_data, load_num_pts, load_dims) =
            load_bin::<u64, VirtualStorageProvider<MemoryFS>>(&storage_provider, file_name, 0)
                .unwrap();
        assert_eq!(load_num_pts, num_pts);
        assert_eq!(load_dims, dims);
        assert_eq!(load_data, data);
        storage_provider
            .filesystem()
            .remove_file(file_name)
            .unwrap();
    }

    #[test]
    fn load_bin_offset_test() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::new());

        let offset: usize = 32;
        let file_name = "/load_bin_offset_test";
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = data.len();
        let dims = 1;
        {
            let mut file_write = storage_provider.create_for_write(file_name).unwrap();
            let bytes_written =
                save_bin_u64(&mut file_write, &data, num_pts, dims, offset).unwrap();
            assert_eq!(bytes_written, 32);
        }

        let (load_data, load_num_pts, load_dims) =
            load_bin::<u64, VirtualStorageProvider<MemoryFS>>(&storage_provider, file_name, offset)
                .unwrap();
        assert_eq!(load_num_pts, num_pts);
        assert_eq!(load_dims, dims);
        assert_eq!(load_data, data);
        storage_provider
            .filesystem()
            .remove_file(file_name)
            .unwrap();
    }

    #[test]
    fn load_multivec_bin_test() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::new());

        let file_name = "/load_multivec_bin_test";
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = 2;
        let total_vecs = 3;
        let lengths = vec![2u32, 1u32]; // first multivec has 2 vectors, second has 1 vector
        let dims = 1;
        {
            let mut file_write = storage_provider.create_for_write(file_name).unwrap();

            // Write header
            file_write
                .write_all(&(num_pts as u32).to_le_bytes())
                .unwrap();
            file_write.write_all(&(dims as u32).to_le_bytes()).unwrap(); // dimension = 1
            file_write
                .write_all(&(total_vecs as u32).to_le_bytes())
                .unwrap();

            // Write lengths
            for &length in &lengths {
                file_write.write_all(&length.to_le_bytes()).unwrap();
            }

            // Write data
            for &value in &data {
                file_write.write_all(&value.to_le_bytes()).unwrap();
            }
        }

        let (load_data, load_num_pts, load_dims, load_num_points) =
            load_multivec_bin::<u64, VirtualStorageProvider<MemoryFS>>(
                &storage_provider,
                file_name,
            )
            .unwrap();
        assert_eq!(load_num_pts, num_pts);
        assert_eq!(load_dims, dims);
        assert_eq!(load_num_points, total_vecs);
        // check that loaded data matches expected structure and values
        assert_eq!(load_data.len(), num_pts);
        assert_eq!(load_data[0].nrows(), lengths[0] as usize);
        assert_eq!(load_data[1].nrows(), lengths[1] as usize);
        assert_eq!(load_data[0].ncols(), dims);
        assert_eq!(load_data[1].ncols(), dims);
        assert_eq!(load_data[0].as_slice(), &[0u64, 1u64]);
        assert_eq!(load_data[1].as_slice(), &[2u64]);
        storage_provider
            .filesystem()
            .remove_file(file_name)
            .unwrap();
    }

    #[test]
    fn load_multivec_bin_zero_vector_test() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::new());

        let file_name = "/load_multivec_bin_zero_vector_test";
        let data = vec![0u64, 1u64, 2u64];
        let num_pts = 2;
        let total_vecs = 3;
        let lengths = vec![0u32, 3u32]; // hand a zero-length test case to the loader
        let dims = 1;
        {
            let mut file_write = storage_provider.create_for_write(file_name).unwrap();

            // Write header
            file_write
                .write_all(&(num_pts as u32).to_le_bytes())
                .unwrap();
            file_write.write_all(&(dims as u32).to_le_bytes()).unwrap(); // dimension = 1
            file_write
                .write_all(&(total_vecs as u32).to_le_bytes())
                .unwrap();

            // Write lengths
            for &length in &lengths {
                file_write.write_all(&length.to_le_bytes()).unwrap();
            }

            // Write data
            for &value in &data {
                file_write.write_all(&value.to_le_bytes()).unwrap();
            }
        }

        let res = load_multivec_bin::<u64, VirtualStorageProvider<MemoryFS>>(
            &storage_provider,
            file_name,
        );

        assert!(res.is_err());

        storage_provider
            .filesystem()
            .remove_file(file_name)
            .unwrap();
    }
}
