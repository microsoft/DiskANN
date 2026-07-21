/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    io::{Seek, SeekFrom, Write},
    marker::PhantomData,
    time::Instant,
};

use diskann::{error::IntoANNResult, utils::VectorRepr, ANNError, ANNResult};
use diskann_providers::{
    storage::{StorageReadProvider, StorageWriteProvider},
    utils::{load_metadata_from_file, BridgeErr, ParallelIteratorInPool, RayonThreadPoolRef},
};
use diskann_utils::{io::Metadata, matrix};
use rayon::iter::IndexedParallelIterator;
use tracing::info;

use crate::storage::quant::compressor::QuantCompressor;

/// [`QuantDataGenerator`] orchestrates the process of reading vector data, applying quantization,
/// and writing compressed results to storage in batches.
pub struct QuantDataGenerator<T, Q>
where
    T: Copy + VectorRepr,
    Q: QuantCompressor<T>,
{
    pub quantizer: Q,
    pub data_path: String,
    pub compressed_data_path: String,
    phantom: PhantomData<T>,
}

impl<T, Q> QuantDataGenerator<T, Q>
where
    T: Copy + VectorRepr,
    Q: QuantCompressor<T>,
{
    pub fn new(
        data_path: String,
        compressed_data_path: String,
        quantizer_context: &Q::CompressorContext,
    ) -> ANNResult<Self> {
        let quantizer = Q::new(quantizer_context)?;
        Ok(Self {
            data_path,
            compressed_data_path,
            quantizer,
            phantom: PhantomData,
        })
    }

    /// This method reads the source data file, processes vectors in batches, compresses them
    /// using the provided quantizer, and writes the results to the compressed data file.
    //
    /// The implementation is adapted from generate_quantized_data_internal in pq_construction.rs
    //
    /// # Processing Flow
    /// 1. Opens the source data file and validates its metadata.
    /// 2. Deletes any existing output.
    /// 3. Creates or opens output compressed file and writes metadata header - [num_points as i32, compressed_vector_size as i32]
    /// 4. Processes data in bounded blocks.
    /// 5. Compresses each block in small batch sizes in parallel to (potentially) take advantage of batch compression with quantizer
    /// 6. Writes compressed blocks to the output file.
    pub fn generate_data<Storage>(
        &self,
        storage_provider: &Storage, // Provider for reading source data and writing compressed results
        pool: RayonThreadPoolRef<'_>, // Thread pool for parallel processing
        max_block_size: usize,
    ) -> ANNResult<()>
    where
        Storage: StorageReadProvider + StorageWriteProvider,
    {
        let timer = Instant::now();

        let metadata = load_metadata_from_file(storage_provider, &self.data_path)?;
        let (num_points, dim) = metadata.into_dims();
        if max_block_size == 0 {
            return Err(ANNError::log_pq_error(
                "Data compression chunk vector count must be greater than zero",
            ));
        }
        if num_points == 0 {
            return Err(ANNError::log_pq_error(
                "Cannot generate compressed data for an empty dataset",
            ));
        }

        let compressed_path = self.compressed_data_path.as_str();

        if storage_provider.exists(compressed_path) {
            storage_provider.delete(compressed_path)?;
        }

        info!("Generating quantized data for {}", compressed_path);

        let data_reader = &mut storage_provider.open_reader(&self.data_path)?;
        data_reader.seek(SeekFrom::Start((std::mem::size_of::<i32>() * 2) as u64))?;

        let mut compressed_data_writer = storage_provider.create_for_write(compressed_path)?;
        Metadata::new(num_points, self.quantizer.compressed_bytes())?
            .write(&mut compressed_data_writer)?;

        let compressed_size = self.quantizer.compressed_bytes();
        let block_size = std::cmp::min(num_points, max_block_size);
        let num_blocks = num_points / block_size + !num_points.is_multiple_of(block_size) as usize;

        info!(
            "Compressing with block size {}, num_blocks {}, num_points {}",
            block_size, num_blocks, num_points
        );

        let mut compressed_buffer = vec![0_u8; block_size * compressed_size];

        //Every block has size exactly block_size, except for potentially the last one
        let mut action = |block_index| -> ANNResult<()> {
            let start_index: usize = block_index * block_size;
            let end_index: usize = std::cmp::min(start_index + block_size, num_points);
            let cur_block_size: usize = end_index - start_index;

            let block_compressed_base = &mut compressed_buffer[..cur_block_size * compressed_size];

            let raw_block: Vec<T> =
                diskann::utils::read_exact_into(data_reader, cur_block_size * dim)?;

            let full_dim = T::full_dimension(&raw_block[..dim]).into_ann_result()?; // read full-dimension from first vector

            let mut block_data: Vec<f32> = vec![f32::default(); cur_block_size * full_dim];
            for (v, dst) in raw_block
                .chunks_exact(dim)
                .zip(block_data.chunks_exact_mut(full_dim))
            {
                T::as_f32_into(v, dst).into_ann_result()?;
            }

            // We need some batch size of data to pass to `compress`. There is a balance
            // to achieve here. It must be:
            //
            // 1. Small enough to allow for parallelism across threads/tasks.
            // 2. Large enough to take advantage of cache locality in `compress`.
            //
            // A value of 128 is a somewhat arbitrary compromise, meaning each task will
            // process `BATCH_SIZE` many dataset vectors at a time.
            const BATCH_SIZE: usize = 128;

            // Wrap the data in `MatrixViews` so we do not need to manually construct view
            // in the compression loop.
            let mut compressed_block = matrix::MatrixViewMut::try_from(
                block_compressed_base,
                cur_block_size,
                compressed_size,
            )
            .bridge_err()?;
            let base_block =
                matrix::MatrixView::try_from(&block_data, cur_block_size, full_dim).bridge_err()?;
            base_block
                .par_window_iter(BATCH_SIZE)
                .zip_eq(compressed_block.par_window_iter_mut(BATCH_SIZE))
                .try_for_each_in_pool(pool, |(src, dst)| self.quantizer.compress(src, dst))?;

            let write_offset = start_index * compressed_size + std::mem::size_of::<i32>() * 2;
            compressed_data_writer.seek(SeekFrom::Start(write_offset as u64))?;
            compressed_data_writer.write_all(block_compressed_base)?;
            Ok(())
        };

        for block_index in 0..num_blocks {
            action(block_index)?;
        }
        compressed_data_writer.flush()?;

        info!(
            "Quant data generation took {} seconds",
            timer.elapsed().as_secs_f64()
        );

        Ok(())
    }
}

//////////////////
///// Tests /////
/////////////////

#[cfg(test)]
mod generator_tests {
    use std::io::BufReader;

    use diskann::utils::read_exact_into;
    use diskann_providers::storage::VirtualStorageProvider;
    use diskann_providers::utils::create_thread_pool_for_test;
    use diskann_utils::{
        io::{write_bin, Metadata},
        matrix::MatrixView,
    };
    use rstest::rstest;
    use vfs::{FileSystem, MemoryFS};

    use super::*;
    pub struct DummyCompressor {
        pub output_dim: u32,
        pub code: Vec<u8>,
    }
    impl DummyCompressor {
        pub fn new(output_dim: u32) -> Self {
            Self {
                output_dim,
                code: (0..output_dim).map(|x| (x % 256) as u8).collect(),
            }
        }
    }
    impl QuantCompressor<f32> for DummyCompressor {
        type CompressorContext = u32;

        fn new(context: &Self::CompressorContext) -> ANNResult<Self> {
            Ok(Self::new(*context))
        }

        fn compress(
            &self,
            _vector: matrix::MatrixView<f32>,
            mut output: matrix::MatrixViewMut<u8>,
        ) -> ANNResult<()> {
            output
                .row_iter_mut()
                .for_each(|r| r.copy_from_slice(&self.code));
            Ok(())
        }

        fn compressed_bytes(&self) -> usize {
            self.output_dim as usize
        }
    }

    fn create_test_data(num_points: usize, dim: usize) -> Vec<f32> {
        let mut data = Vec::new();

        // Generate some test vector data
        for i in 0..num_points {
            for j in 0..dim {
                data.push((i * dim + j) as f32);
            }
        }

        data
    }

    fn generate_data_files(
        num_points: usize,
        dim: usize,
    ) -> ANNResult<(VirtualStorageProvider<MemoryFS>, String, String)> {
        let storage_provider = VirtualStorageProvider::new_memory();
        storage_provider
            .filesystem()
            .create_dir("/test_data")
            .expect("Could not create test directory");

        let data_path = "/test_data/test_data.bin".to_string();
        let compressed_path = "/test_data/test_compressed.bin".to_string();

        // Setup test data
        let data = create_test_data(num_points, dim);
        let view = MatrixView::try_from(data.as_slice(), num_points, dim).unwrap();
        write_bin(
            view,
            &mut storage_provider.create_for_write(data_path.as_str())?,
        )?;

        Ok((storage_provider, data_path, compressed_path))
    }

    fn create_and_call_generator<F: vfs::FileSystem>(
        compressed_path: String,
        storage_provider: &VirtualStorageProvider<F>,
        data_path: String,
        output_dim: u32,
        max_block_size: usize,
    ) -> (QuantDataGenerator<f32, DummyCompressor>, ANNResult<()>) {
        let pool: diskann_providers::utils::RayonThreadPool = create_thread_pool_for_test();
        let generator = QuantDataGenerator::<f32, DummyCompressor>::new(
            data_path,
            compressed_path,
            &output_dim,
        )
        .unwrap();
        let result = generator.generate_data(storage_provider, pool.as_ref(), max_block_size);
        (generator, result)
    }

    #[rstest]
    #[case(100, 8, 4)]
    #[case(257, 4, 8)]
    #[case(60_000, 384, 192)]
    fn test_generate_data(
        #[case] num_points: usize,
        #[case] dim: usize,
        #[case] output_dim: u32,
    ) -> ANNResult<()> {
        let (storage_provider, data_path, compressed_path) = generate_data_files(num_points, dim)?;
        let (generator, result) = create_and_call_generator(
            compressed_path.clone(),
            &storage_provider,
            data_path,
            output_dim,
            10_000,
        );

        result?;
        assert!(storage_provider.exists(&compressed_path));

        let expected_size = num_points * output_dim as usize;
        let file_len = storage_provider.get_length(&compressed_path)? as usize;
        assert_eq!(file_len, expected_size + 2 * std::mem::size_of::<i32>());

        let mut r = storage_provider.open_reader(compressed_path.as_str())?;
        let mut reader = BufReader::new(&mut r);
        let metadata = Metadata::read(&mut reader)?;

        let data: Vec<u8> = read_exact_into(&mut reader, expected_size)?;

        assert_eq!(metadata.ndims_u32(), output_dim);
        assert_eq!(metadata.npoints(), num_points);

        data.chunks_exact(output_dim as usize)
            .for_each(|chunk| assert_eq!(chunk, generator.quantizer.code.as_slice()));

        Ok(())
    }

    #[test]
    fn generate_data_rejects_empty_dataset() -> ANNResult<()> {
        let storage_provider = VirtualStorageProvider::new_memory();
        storage_provider
            .filesystem()
            .create_dir("/test_data")
            .expect("Could not create test directory");

        let data_path = "/test_data/empty.bin".to_string();
        let compressed_path = "/test_data/empty_compressed.bin".to_string();
        Metadata::new(0, 8)?.write(&mut storage_provider.create_for_write(data_path.as_str())?)?;

        let (_, result) = create_and_call_generator(
            compressed_path.clone(),
            &storage_provider,
            data_path,
            4,
            10_000,
        );

        assert!(result.is_err());
        assert!(!storage_provider.exists(&compressed_path));
        Ok(())
    }

    #[test]
    fn generate_data_rejects_zero_chunk_size() -> ANNResult<()> {
        let (storage_provider, data_path, compressed_path) = generate_data_files(1, 8)?;

        let (_, result) =
            create_and_call_generator(compressed_path.clone(), &storage_provider, data_path, 4, 0);

        assert!(result.is_err());
        assert!(!storage_provider.exists(&compressed_path));
        Ok(())
    }
}
