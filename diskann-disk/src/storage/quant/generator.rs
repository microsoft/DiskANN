/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    io::{Seek, SeekFrom, Write},
    marker::PhantomData,
};

use diskann::{error::IntoANNResult, utils::VectorRepr, ANNError, ANNResult};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    forward_threadpool,
    utils::{
        load_metadata_from_file, write_metadata, AsThreadPool, BridgeErr, ParallelIteratorInPool,
        Timer,
    },
};
use diskann_utils::views::{self};
use rayon::iter::IndexedParallelIterator;
use tracing::info;

use crate::{
    build::chunking::{
        checkpoint::Progress,
        continuation::{process_while_resource_is_available, ChunkingConfig},
    },
    storage::quant::compressor::{CompressionStage, QuantCompressor},
};

/// [`GeneratorContext`] defines parameters for vector quantization checkpoint state
///
/// This struct holds offset position that allows resuming quantization from
/// a specific point in the dataset as well as the data path to store the
/// compressed vectors.
#[derive(Clone, Debug)]
pub struct GeneratorContext {
    /// * `offset`: The point index to start/resume quantization from (for checkpoint support)
    pub offset: usize,
    /// * `compressed_data_path`: The path to which to write compressed data to.
    pub compressed_data_path: String,
}

impl GeneratorContext {
    pub fn new(offset: usize, compressed_data_path: String) -> Self {
        Self {
            offset,
            compressed_data_path,
        }
    }
}

/// [`QuantDataGenerator`] orchestrates the process of reading vector data, applying quantization,
/// and writing compressed results to storage. It resumes data generation from the checkpoint manager
/// and processes data in batches.
pub struct QuantDataGenerator<T, Q>
where
    T: Copy + VectorRepr,
    Q: QuantCompressor<T>,
{
    pub quantizer: Q,
    pub data_path: String,         // Path to the source vector data
    pub context: GeneratorContext, // Overloadable context that contains metric and offset info
    phantom: PhantomData<T>,
}

impl<T, Q> QuantDataGenerator<T, Q>
where
    T: Copy + VectorRepr,
    Q: QuantCompressor<T>,
{
    pub fn new(
        data_path: String,
        context: GeneratorContext,
        quantizer_context: &Q::CompressorContext,
    ) -> ANNResult<Self> {
        let stage = match context.offset {
            0 => CompressionStage::Start,
            _ => CompressionStage::Resume,
        };
        let quantizer = Q::new_at_stage(stage, quantizer_context)?;
        Ok(Self {
            data_path,
            context,
            quantizer,
            phantom: PhantomData,
        })
    }

    /// This method reads the source data file, processes vectors in batches, compresses them
    /// using the provided quantizer, and writes the results to the compressed data file.
    /// It supports checkpointing through the chunking_config and resumes from previous
    /// interruptions using the offset stored in the context.
    //
    /// The implementation is adapted from generate_quantized_data_internal in pq_construction.rs
    //
    /// # Processing Flow
    /// 1. Checks if starting from beginning (offset=0) and deletes any existing output if needed
    /// 2. Opens source data file and reads metadata (num_points and dimension)
    /// 3. Creates or opens output compressed file and writes metadata header - [num_points as i32, compressed_vector_size as i32]
    /// 4. Processes data in blocks of size given by chunking_config.data_compression_chunk_vector_count = 50_000
    /// 5. Compresses each block in small batch sizes in parallel to (potentially) take advantage of batch compression with quantizer
    /// 6. Writes compressed blocks to the output file.
    pub fn generate_data<Storage, Pool>(
        &self,
        storage_provider: &Storage, // Provider for reading source data and writing compressed results
        pool: &Pool,                // Thread pool for parallel processing
        chunking_config: &ChunkingConfig, // Configuration for batching and checkpoint handling
    ) -> ANNResult<Progress>
    where
        Storage: StorageReadProvider + StorageWriteProvider,
        Pool: AsThreadPool,
    {
        let timer = Timer::new();

        let metadata = load_metadata_from_file(storage_provider, &self.data_path)?;
        let (num_points, dim) = (metadata.npoints, metadata.ndims);

        self.validate_params(num_points, storage_provider)?;

        let offset = self.context.offset;
        let compressed_path = self.context.compressed_data_path.as_str();

        if offset == 0 && storage_provider.exists(compressed_path) {
            storage_provider.delete(compressed_path)?;
        }

        info!("Generating quantized data for {}", compressed_path);

        let data_reader = &mut storage_provider.open_reader(&self.data_path)?;

        //open the writer for the compressed dataset if starting from the middle, else create a new one.
        let mut compressed_data_writer = if offset > 0 {
            storage_provider.open_writer(compressed_path)?
        } else {
            let mut sp = storage_provider.create_for_write(compressed_path)?;
            // write meatadata to header
            write_metadata(&mut sp, num_points, self.quantizer.compressed_bytes())?;
            sp
        };

        //seek to the offset after skipping metadata
        data_reader.seek(SeekFrom::Start(
            (size_of::<i32>() * 2 + offset * dim * size_of::<T>()) as u64,
        ))?;

        let compressed_size = self.quantizer.compressed_bytes();
        let max_block_size = chunking_config.data_compression_chunk_vector_count;
        let num_remaining = num_points - offset;

        let block_size = std::cmp::min(num_points, max_block_size);
        let num_blocks =
            num_remaining / block_size + !num_remaining.is_multiple_of(block_size) as usize;

        info!(
            "Compressing with block size {}, num_remaining {}, num_blocks {}, offset {}, num_points {}",
            block_size, num_remaining, num_blocks, offset, num_points
        );

        let mut compressed_buffer = vec![0_u8; block_size * compressed_size];

        forward_threadpool!(pool = pool: Pool);
        //Every block has size exactly block_size, except for potentially the last one
        let action = |block_index| -> ANNResult<()> {
            let start_index: usize = offset + block_index * block_size;
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
            let mut compressed_block = views::MutMatrixView::try_from(
                block_compressed_base,
                cur_block_size,
                compressed_size,
            )
            .bridge_err()?;
            let base_block =
                views::MatrixView::try_from(&block_data, cur_block_size, full_dim).bridge_err()?;
            base_block
                .par_window_iter(BATCH_SIZE)
                .zip_eq(compressed_block.par_window_iter_mut(BATCH_SIZE))
                .try_for_each_in_pool(pool, |(src, dst)| self.quantizer.compress(src, dst))?;

            let write_offset = start_index * compressed_size + std::mem::size_of::<i32>() * 2;
            compressed_data_writer.seek(SeekFrom::Start(write_offset as u64))?;
            compressed_data_writer.write_all(block_compressed_base)?;
            compressed_data_writer.flush()?;
            Ok(())
        };

        let progress = process_while_resource_is_available(
            action,
            0..num_blocks,
            chunking_config.continuation_checker.clone_box(),
        )?
        .map(|processed| processed * block_size + offset);

        info!(
            "Quant data generation took {} seconds",
            timer.elapsed().as_secs_f64()
        );

        Ok(progress)
    }

    fn validate_params<Storage: StorageReadProvider + StorageWriteProvider>(
        &self,
        num_points: usize,
        storage_provider: &Storage,
    ) -> ANNResult<()> {
        if self.context.offset > num_points {
            //check to make sure offset is within limits.
            return Err(ANNError::log_pq_error(
                "Error: offset for compression is more than number of points",
            ));
        }

        let compressed_path = &self.context.compressed_data_path;

        if self.context.offset > 0 {
            if !storage_provider.exists(compressed_path) {
                return Err(ANNError::log_file_not_found_error(format!(
                    "Error: Generator expected compressed file {compressed_path} but did not find it."
                )));
            }
            let expected_length = self.quantizer.compressed_bytes() * self.context.offset
                + std::mem::size_of::<i32>() * 2;
            let existing_length =
                storage_provider.get_length(&self.context.compressed_data_path)?;

            if existing_length != expected_length as u64 {
                //check to make sure compressed data file lengths is as expected based on offset.
                return Err(ANNError::log_pq_error(format_args!(
                    "Error: compressed data file length {existing_length} does not match expected length {expected_length}."
                )));
            }
        }

        Ok(())
    }
}

//////////////////
///// Tests /////
/////////////////

#[cfg(test)]
mod generator_tests {
    use std::{
        io::BufReader,
        sync::{Arc, RwLock},
    };

    use diskann::utils::read_exact_into;
    use diskann_providers::storage::VirtualStorageProvider;
    use diskann_providers::utils::{
        create_thread_pool_for_test, read_metadata, save_bin_f32, save_bytes,
    };
    use rstest::rstest;
    use vfs::{FileSystem, MemoryFS};

    use super::*;
    use crate::build::chunking::continuation::{
        ContinuationGrant, ContinuationTrackerTrait, NaiveContinuationTracker,
    };

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

        fn new_at_stage(
            _stage: CompressionStage,
            context: &Self::CompressorContext,
        ) -> ANNResult<Self> {
            Ok(Self::new(*context))
        }

        fn compress(
            &self,
            _vector: views::MatrixView<f32>,
            mut output: views::MutMatrixView<u8>,
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

    //Mock continuation checker that stops after stop_count - 1 iterations.
    struct MockStopContinuationChecker {
        count: Arc<RwLock<usize>>,
        stop_count: usize,
    }

    impl Clone for MockStopContinuationChecker {
        fn clone(&self) -> Self {
            MockStopContinuationChecker {
                count: self.count.clone(),
                stop_count: self.stop_count,
            }
        }
    }

    impl ContinuationTrackerTrait for MockStopContinuationChecker {
        fn get_continuation_grant(&self) -> ContinuationGrant {
            let mut count = self.count.write().unwrap();
            *count += 1;
            if !(*count).is_multiple_of(self.stop_count) {
                ContinuationGrant::Continue
            } else {
                ContinuationGrant::Stop
            }
        }
    }

    fn generate_data_and_compressed(
        num_points: usize,
        dim: usize,
        offset: usize,
        output_dim: u32,
    ) -> ANNResult<(VirtualStorageProvider<MemoryFS>, String, String)> {
        let storage_provider = VirtualStorageProvider::new_memory();
        storage_provider
            .filesystem()
            .create_dir("/test_data")
            .expect("Could not create test directory");

        let data_path = "/test_data/test_data.bin".to_string();
        let compressed_path = "/test_data/test_compressed.bin".to_string();

        // Setup test data
        let _ = save_bin_f32(
            &mut storage_provider.create_for_write(data_path.as_str())?,
            &create_test_data(num_points, dim),
            num_points,
            dim,
            0,
        )?;

        if offset > 0 {
            // write head of file
            let code = (0..output_dim).map(|x| (x % 256) as u8).collect::<Vec<_>>(); //this is the same code as in DummyQuantizer

            let mut buffer = vec![0_u8; offset * output_dim as usize];
            buffer
                .chunks_exact_mut(output_dim as usize)
                .for_each(|bf| bf.copy_from_slice(code.as_slice()));
            let _ = save_bytes(
                &mut storage_provider.create_for_write(compressed_path.as_str())?,
                buffer.as_slice(),
                num_points,
                output_dim as usize,
                0,
            )?;
        }

        Ok((storage_provider, data_path, compressed_path))
    }

    fn create_and_call_generator<F: vfs::FileSystem>(
        offset: usize,
        compressed_path: String,
        storage_provider: &VirtualStorageProvider<F>,
        data_path: String,
        output_dim: u32,
        chunking_config: &ChunkingConfig,
    ) -> (
        QuantDataGenerator<f32, DummyCompressor>,
        Result<Progress, ANNError>,
    ) {
        let pool: diskann_providers::utils::RayonThreadPool = create_thread_pool_for_test();
        // Create generator
        let context = GeneratorContext::new(offset, compressed_path.clone());
        let generator = QuantDataGenerator::<f32, DummyCompressor>::new(
            data_path.clone(),
            context,
            &output_dim,
        )
        .unwrap();
        // Run generator
        let result = generator.generate_data(storage_provider, &&pool, chunking_config);
        (generator, result)
    }

    #[rstest]
    #[case(100, 8, 4, 0, 10, 100 * 4)] //small test that fits in BATCH_SIZE
    #[case(100, 8, 4, 50, 10, 100 * 4)] //small test that fits in BATCH_SIZE with offset > 0
    #[case(257, 4, 8, 0, 10, 257 * 8)] //larger than BATCH_SIZE and not multiple of it
    #[case(60_000, 384, 192, 5_000, 10, 60_000 * 192)] //larger than chunk_vector_count = 10_000 with offset > 0
    #[case(60_000, 384, 192, 0, 10, 60_000 * 192)] //larger than chunk_vector_count = 10_000 with offset = 0
    #[case(60_000, 384, 192, 0, 2, 10_000 * 192)] //should stop after 1 action block
    #[case(60_000, 384, 192, 1000, 2, 11_000 * 192)] //same as above but with offset
    fn test_generate_data_from_offset(
        #[case] num_points: usize,
        #[case] dim: usize,
        #[case] output_dim: u32,
        #[case] offset: usize,
        #[case] config_stop_count: usize,
        #[case] expected_size: usize,
    ) -> ANNResult<()> {
        let (storage_provider, data_path, compressed_path) =
            generate_data_and_compressed(num_points, dim, offset, output_dim)?;

        let chunking_config = ChunkingConfig {
            continuation_checker: Box::new(MockStopContinuationChecker {
                count: Arc::new(RwLock::new(0)),
                stop_count: config_stop_count,
            }),
            data_compression_chunk_vector_count: 10_000,
            inmemory_build_chunk_vector_count: 10_000,
        };

        let (generator, result) = create_and_call_generator::<vfs::MemoryFS>(
            offset,
            compressed_path.clone(),
            &storage_provider,
            data_path,
            output_dim,
            &chunking_config,
        );

        assert!(result.is_ok(), "Result is not ok, got {:?}", result); //should have completed correctly
        assert!(storage_provider.exists(&compressed_path)); // Verify output file

        // Check compressed data size
        let file_len = storage_provider.get_length(&compressed_path)? as usize;
        assert_eq!(file_len, expected_size + 2 * std::mem::size_of::<i32>());

        let mut r = storage_provider.open_reader(compressed_path.as_str())?;
        let mut reader = BufReader::new(&mut r);
        let metadata = read_metadata(&mut reader)?;

        let data: Vec<u8> = read_exact_into(&mut reader, expected_size)?;

        // Check header
        assert_eq!(metadata.ndims as u32, output_dim);
        assert_eq!(metadata.npoints, num_points);

        // Check compressed data content
        data.chunks_exact(output_dim as usize)
            .for_each(|chunk| assert_eq!(chunk, generator.quantizer.code.as_slice()));

        Ok(())
    }

    #[test]
    fn test_stop_and_continue_chunking_config() -> ANNResult<()> {
        let (num_points, dim, output_dim) = (256, 128, 128);
        let chunking_config = ChunkingConfig {
            continuation_checker: Box::<NaiveContinuationTracker>::default(),
            data_compression_chunk_vector_count: 10,
            inmemory_build_chunk_vector_count: 10,
        };
        let (storage_provider, data_path, compressed_path) =
            generate_data_and_compressed(num_points, dim, 0, output_dim)?;
        let (mut generator, mut result) = create_and_call_generator::<vfs::MemoryFS>(
            0,
            compressed_path.clone(),
            &storage_provider,
            data_path.clone(),
            output_dim,
            &chunking_config,
        );
        loop {
            match result.as_ref().unwrap() {
                Progress::Completed => break,
                Progress::Processed(num_points) => {
                    (generator, result) = create_and_call_generator::<vfs::MemoryFS>(
                        *num_points,
                        compressed_path.clone(),
                        &storage_provider,
                        data_path.clone(),
                        output_dim,
                        &chunking_config,
                    );
                }
            }
        }

        assert!(result.is_ok(), "Result is not ok, got {:?}", result); //should have completed correctly
        assert!(storage_provider.exists(&compressed_path)); // Verify output file

        // Check compressed data size
        let file_len = storage_provider.get_length(&compressed_path)? as usize;
        let expected_size = (num_points * output_dim as usize) + 2 * std::mem::size_of::<i32>();
        assert_eq!(file_len, expected_size,);

        let mut r = storage_provider.open_reader(compressed_path.as_str())?;
        let mut reader = BufReader::new(&mut r);
        let metadata = read_metadata(&mut reader)?;

        let data: Vec<u8> =
            read_exact_into(&mut reader, expected_size - 2 * std::mem::size_of::<i32>())?;

        // Check header
        assert_eq!(metadata.ndims as u32, output_dim);
        assert_eq!(metadata.npoints, num_points);

        // Check compressed data content
        data.chunks_exact(output_dim as usize)
            .for_each(|chunk| assert_eq!(chunk, generator.quantizer.code.as_slice()));
        Ok(())
    }

    #[rstest]
    #[case(
        1_024,
        384,
        192,
        1_025,
        0,
        "offset for compression is more than number of points"
    )]
    #[case(
        1_1024,
        384,
        192,
        5,
        15,
        "compressed data file length 2888 does not match expected length 968."
    )]
    fn test_offset_error_case(
        #[case] num_points: usize,
        #[case] dim: usize,
        #[case] output_dim: u32,
        #[case] offset: usize,
        #[case] error_offset: usize,
        #[case] msg: String,
    ) -> ANNResult<()> {
        assert!(offset > 0);
        let (storage_provider, data_path, compressed_path) =
            generate_data_and_compressed(num_points, dim, error_offset, output_dim)?;

        let (_, result) = create_and_call_generator::<vfs::MemoryFS>(
            offset,
            compressed_path,
            &storage_provider,
            data_path,
            output_dim,
            &ChunkingConfig::default(),
        );

        assert!(result.is_err());
        if let Err(e) = result {
            let error_msg = format!("{:?}", e);
            assert!(error_msg.contains(&msg), "{}", &error_msg);
        }

        Ok(())
    }
}
