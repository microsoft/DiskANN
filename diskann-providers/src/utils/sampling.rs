/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{
    io::{BufReader, Read, Seek},
    mem,
};

use crate::storage::StorageReadProvider;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use diskann::{ANNError, ANNResult, error::IntoANNResult, utils::VectorRepr};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

use super::READ_WRITE_BLOCK_SIZE;

/// A reader that efficiently handles vector sampling and retrieval from binary files.
/// Automatically selects between direct and buffered access based on sampling density
///
/// # Binary File Format
/// The reader expects files in the following binary format:
/// ```text
/// [Header: 8 bytes]
/// - num_points (u32): Number of vectors
/// - dimension (u32): Vector dimension
///
/// [Data: num_points * dimension * sizeof(T) bytes]
/// - Contiguous vector data
/// - Each element stored in little-endian format
/// ```
pub struct SampleVectorReader<T, Reader>
where
    T: Default + bytemuck::Pod,
    Reader: Read + Seek,
{
    reader: VectorDataReader<Reader>,
    npts: u32,
    dim: u32,
    cur_pos: u32,
    _phantom: std::marker::PhantomData<T>,
}

/// A reader type that provides either direct or buffered access to vector data
enum VectorDataReader<R: Read + Seek> {
    /// Direct file access for sparse sampling scenarios
    DirectAccess(R),
    /// Buffered access for dense sampling scenarios
    BufferedAccess(BufReader<R>),
}

impl<R: Read + Seek> VectorDataReader<R> {
    fn read_exact(&mut self, buffer: &mut [u8]) -> std::io::Result<()> {
        match self {
            VectorDataReader::DirectAccess(reader) => reader.read_exact(buffer),
            VectorDataReader::BufferedAccess(reader) => reader.read_exact(buffer),
        }
    }
    fn seek_relative(&mut self, offset: i64) -> std::io::Result<()> {
        match self {
            VectorDataReader::DirectAccess(reader) => reader.seek_relative(offset),
            VectorDataReader::BufferedAccess(reader) => reader.seek_relative(offset),
        }
    }

    fn read_u32<B: ByteOrder>(&mut self) -> std::io::Result<u32> {
        match self {
            VectorDataReader::DirectAccess(reader) => reader.read_u32::<B>(),
            VectorDataReader::BufferedAccess(reader) => reader.read_u32::<B>(),
        }
    }
}

/// Sampling density setting that determines whether to use direct or buffered file access
pub enum SamplingDensity {
    /// Direct file access for sparse sampling
    Sparse,
    /// Buffered access for dense sampling
    Dense,
}

impl SamplingDensity {
    /// Creates appropriate sampling density based on the sample rate
    pub fn from_sample_rate(sample_rate: f64) -> Self {
        const DENSE_SAMPLING_THRESHOLD: f64 = 0.5;
        if sample_rate < DENSE_SAMPLING_THRESHOLD {
            Self::Sparse
        } else {
            Self::Dense
        }
    }
}

impl<T, Reader> SampleVectorReader<T, Reader>
where
    T: Default + bytemuck::Pod,
    Reader: Read + Seek,
{
    /// Creates a new vector reader optimized for the given sampling density.
    ///
    /// # Arguments
    /// * `data_file` - Path to the binary vector file
    /// * `sampling_density` - Density setting that determines access pattern
    /// * `storage_provider` - Provider for file access operations
    ///
    /// # Returns
    /// Reader configured for the specified sampling density
    ///
    /// # Binary File Format
    /// ```text
    /// [Header: 8 bytes]
    /// - num_points (u32): Number of vectors
    /// - dimension (u32): Vector dimension
    ///
    /// [Data: num_points * dimension * sizeof(T) bytes]
    /// - Contiguous vector data
    /// - Each element stored in little-endian format
    /// ```
    pub fn new<P>(
        data_file: &str,
        sampling_density: SamplingDensity,
        storage_provider: &P,
    ) -> ANNResult<Self>
    where
        P: StorageReadProvider<Reader = Reader>,
    {
        let mut reader: VectorDataReader<P::Reader> = match sampling_density {
            SamplingDensity::Sparse => {
                // Direct reader for sparse sampling
                VectorDataReader::DirectAccess(storage_provider.open_reader(data_file)?)
            }
            SamplingDensity::Dense => {
                // Buffered reader for dense sampling
                VectorDataReader::BufferedAccess(BufReader::with_capacity(
                    READ_WRITE_BLOCK_SIZE as usize,
                    storage_provider.open_reader(data_file)?,
                ))
            }
        };

        let npts = reader.read_u32::<LittleEndian>()?;
        let dim = reader.read_u32::<LittleEndian>()?;

        let expected_size = 8 + (npts as u64 * dim as u64 * std::mem::size_of::<T>() as u64);
        let actual_size = storage_provider.get_length(data_file)?;
        if actual_size != expected_size {
            return Err(ANNError::log_invalid_file_format(format!(
                "Vector file '{}' has invalid format: size {} bytes doesn't match expected size of {} bytes based on header ({} vectors of dimension {})",
                data_file, actual_size, expected_size, npts, dim
            )));
        }

        Ok(Self {
            reader,
            npts,
            dim,
            cur_pos: 0,
            _phantom: std::marker::PhantomData::<T>,
        })
    }

    /// Reads vectors at given indices and processes them with the provided function.
    /// For optimal performance, indices should be provided in ascending order to maximize
    /// buffering efficiency and minimize disk seeks.
    ///
    /// # Arguments
    /// * `indices` - Iterator of vector indices. Should ideally be in ascending order
    /// * `process_fn` - Function to process each vector's raw bytes
    pub fn read_vectors<I, F>(&mut self, indices: I, mut process_fn: F) -> ANNResult<()>
    where
        I: Iterator<Item = u32>,
        F: FnMut(&[T]) -> ANNResult<()>,
    {
        let vector_len = self.dim as usize * mem::size_of::<T>();
        let mut vector_buf = vec![T::default(); self.dim as usize];

        for idx in indices {
            // Check if the index is within bounds
            if idx >= self.npts {
                return Err(ANNError::log_index_error(format!(
                    "Vector index {} is out of bounds (max: {})",
                    idx,
                    self.npts - 1
                )));
            }
            let offset = (idx as i64 - self.cur_pos as i64) * vector_len as i64;
            if offset != 0 {
                self.reader.seek_relative(offset)?;
            }

            let buf_u8: &mut [u8] = bytemuck::cast_slice_mut(&mut vector_buf);
            self.reader.read_exact(buf_u8)?;
            process_fn(&vector_buf)?;

            self.cur_pos = idx + 1;
        }

        Ok(())
    }

    /// Gets dataset metadata
    ///
    /// # Returns
    /// * `(u32, u32)` - Tuple of (number of points, dimension)
    pub fn get_dataset_headers(&self) -> (u32, u32) {
        (self.npts, self.dim)
    }
}

/// Streams data from the storage layer, samples each vector with probability p_val
/// and returns a matrix of vectors in floating point format after converting the
/// input vector into a `f32` slice.
///
/// # Arguments
/// * `data_file` - Path to the binary vector file
/// * `p_val` - Sampling probability (0.0 to 1.0)
/// * `storage_provider` - Provider for file access operations
/// * `generator` - Random number generator
///
/// # Returns
/// A tuple containing:
/// * `Vec<f32>` - Sampled vectors in flat array
/// * `usize` - Number of sampled vectors
/// * `usize` - Vector dimension in full-precision representation
pub fn gen_random_slice<T: VectorRepr, StorageProvider: StorageReadProvider>(
    data_file: &str,
    p_val: f64,
    storage_provider: &StorageProvider,
    generator: &mut impl Rng,
) -> ANNResult<(Vec<f32>, usize, usize)> {
    let p_val = p_val.min(1.0);

    let mut reader: SampleVectorReader<T, _> = SampleVectorReader::new(
        data_file,
        SamplingDensity::from_sample_rate(p_val),
        storage_provider,
    )?;
    let (npts, _) = reader.get_dataset_headers();
    let mut full_dim: Option<usize> = None;

    reader.read_vectors([0].into_iter(), |v| {
        //grab full-dimension from first vector
        full_dim = Some(T::full_dimension(v).into_ann_result()?);
        Ok(())
    })?;

    if let Some(full_dim) = full_dim {
        let distribution = StandardUniform;
        let iter = (0..npts).filter(|_| {
            let p: f64 = distribution.sample(generator);
            p < p_val
        });

        let mut sampled_vectors =
            Vec::with_capacity((npts as f64 * p_val).ceil() as usize * full_dim);

        let buffer = vec![f32::default(); full_dim];
        let mut len = 0;

        reader.read_vectors(iter, |vec_t| {
            sampled_vectors.extend_from_slice(&buffer);
            T::as_f32_into(vec_t, &mut sampled_vectors[len..len + full_dim]).into_ann_result()?;
            len += full_dim;
            Ok(())
        })?;

        let sampled_count = len / full_dim;

        Ok((sampled_vectors, sampled_count, full_dim))
    } else {
        Err(ANNError::log_index_error(
            "Could not read vectors to sample from.",
        ))
    }
}

#[cfg(test)]
mod sampling_test {
    use std::io::Write;

    use crate::storage::{StorageWriteProvider, VirtualStorageProvider};
    use byteorder::{LittleEndian, WriteBytesExt};
    use diskann::ANNErrorKind;
    use rstest::rstest;
    use vfs::MemoryFS;

    use super::*;

    #[test]
    fn test_sampling_density() {
        let density = SamplingDensity::from_sample_rate(0.1);
        //Expected Sparse for low sampling rate
        assert!(matches!(density, SamplingDensity::Sparse));

        let density = SamplingDensity::from_sample_rate(0.8);
        //Expected Dense for high sampling rate
        assert!(matches!(density, SamplingDensity::Dense));

        let density = SamplingDensity::from_sample_rate(0.5);
        //Expected Dense for boundary sampling rate
        assert!(matches!(density, SamplingDensity::Dense));
    }

    const TEST_BINARY_FILE: &str = "/test_binary_data.bin";
    const TEST_NUM_POINTS: u32 = 100;
    const TEST_DIM: u32 = 10;

    /// Create a binary file with the specified number of points and dimensions with f32 values
    fn create_test_binary_data<P: StorageWriteProvider>(
        storage_provider: &P,
        file_name: &str,
        num_pts: u32,
        dim: u32,
    ) {
        let mut writer = storage_provider.create_for_write(file_name).unwrap();
        writer.write_u32::<LittleEndian>(num_pts).unwrap();
        writer.write_u32::<LittleEndian>(dim).unwrap();
        for i in 0..num_pts {
            for j in 0..dim {
                let val = (i * dim + j) as f32;
                writer.write_f32::<LittleEndian>(val).unwrap();
            }
        }
        writer.flush().unwrap();
    }

    enum IndicesOrder {
        Ascending,
        Descending,
        Random,
    }

    #[rstest]
    #[case::dense_ascending(SamplingDensity::Dense, IndicesOrder::Ascending)]
    #[case::dense_descending(SamplingDensity::Dense, IndicesOrder::Descending)]
    #[case::dense_random(SamplingDensity::Dense, IndicesOrder::Random)]
    #[case::sparse_ascending(SamplingDensity::Sparse, IndicesOrder::Ascending)]
    #[case::sparse_descending(SamplingDensity::Sparse, IndicesOrder::Descending)]
    #[case::sparse_random(SamplingDensity::Sparse, IndicesOrder::Random)]
    fn test_sample_vector_reader_happy_path(
        #[case] sampling_density: SamplingDensity,
        #[case] indices_order: IndicesOrder,
    ) {
        use rand::seq::SliceRandom;

        let vfs = MemoryFS::new();
        let storage_provider = VirtualStorageProvider::new(vfs);
        create_test_binary_data(
            &storage_provider,
            TEST_BINARY_FILE,
            TEST_NUM_POINTS,
            TEST_DIM,
        );

        let mut reader = SampleVectorReader::<f32, _>::new(
            TEST_BINARY_FILE,
            sampling_density,
            &storage_provider,
        )
        .unwrap();

        assert_eq!(reader.get_dataset_headers(), (TEST_NUM_POINTS, TEST_DIM));

        // Generate indices based on order
        let indices = match indices_order {
            IndicesOrder::Ascending => (0..TEST_NUM_POINTS).collect(),
            IndicesOrder::Descending => (0..TEST_NUM_POINTS).rev().collect(),
            IndicesOrder::Random => {
                let mut rng = crate::utils::create_rnd_in_tests();
                let mut indices: Vec<u32> = (0..TEST_NUM_POINTS).collect();
                indices.shuffle(&mut rng);
                indices
            }
        };

        let mut cur_pos = 0u32;
        reader
            .read_vectors(indices.iter().copied(), |vec_t| {
                assert!(vec_t.len() == TEST_DIM as usize);
                for j in 0..TEST_DIM {
                    let expected = (indices[cur_pos as usize] * TEST_DIM + j) as f32;
                    assert_eq!(vec_t[j as usize], expected);
                }
                cur_pos += 1;
                Ok(())
            })
            .unwrap();
    }

    #[rstest]
    #[case(SamplingDensity::Dense)]
    #[case(SamplingDensity::Sparse)]
    fn test_sample_vector_reader_out_of_bounds(#[case] sampling_density: SamplingDensity) {
        let vfs = MemoryFS::new();
        let storage_provider = VirtualStorageProvider::new(vfs);
        create_test_binary_data(
            &storage_provider,
            TEST_BINARY_FILE,
            TEST_NUM_POINTS,
            TEST_DIM,
        );

        let mut reader = SampleVectorReader::<f32, _>::new(
            TEST_BINARY_FILE,
            sampling_density,
            &storage_provider,
        )
        .unwrap();

        // Try to read invalid indices
        let result = reader
            .read_vectors(vec![TEST_NUM_POINTS + 1].into_iter(), |_| Ok(()))
            .unwrap_err();
        assert!(matches!(result.kind(), ANNErrorKind::IndexError));
    }

    #[rstest]
    #[case(SamplingDensity::Dense)]
    #[case(SamplingDensity::Sparse)]
    fn test_sample_vector_reader_invalid_file(#[case] sampling_density: SamplingDensity) {
        let vfs = MemoryFS::new();
        let storage_provider = VirtualStorageProvider::new(vfs);

        {
            // Create file with wrong size
            let mut writer = storage_provider.create_for_write(TEST_BINARY_FILE).unwrap();
            writer.write_u32::<LittleEndian>(10).unwrap(); // num_points
            writer.write_u32::<LittleEndian>(5).unwrap(); // dim
            writer.write_f32::<LittleEndian>(1.0).unwrap(); // Too little data
            writer.flush().unwrap();
        }
        // Should fail validation check
        let err = match SampleVectorReader::<f32, _>::new(
            TEST_BINARY_FILE,
            sampling_density,
            &storage_provider,
        ) {
            Ok(_) => panic!("operations should not succeed"),
            Err(err) => err,
        };
        assert!(
            matches!(err.kind(), ANNErrorKind::InvalidFileFormatError),
            "Invalid file format error expected, got {:?}",
            err
        );
    }
}
