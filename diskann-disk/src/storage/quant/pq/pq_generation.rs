/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::marker::PhantomData;

use diskann::{utils::VectorRepr, ANNError};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    forward_threadpool,
    model::{
        pq::{accum_row_inplace, generate_pq_pivots},
        GeneratePivotArguments,
    },
    storage::PQStorage,
    utils::{AsThreadPool, BridgeErr, Timer},
};
use diskann_quantization::{product::TransposedTable, CompressInto};
use diskann_utils::views::MatrixBase;
use diskann_vector::distance::Metric;
use tracing::info;

use crate::storage::quant::compressor::{CompressionStage, QuantCompressor};

pub struct PQGenerationContext<'a, Storage, Pool>
where
    Storage: StorageReadProvider + StorageWriteProvider,
    Pool: AsThreadPool,
{
    pub pq_storage: PQStorage,
    pub num_chunks: usize,
    pub seed: Option<u64>,
    pub p_val: f64,
    pub storage_provider: &'a Storage,
    pub pool: Pool,
    pub metric: Metric,
    pub dim: usize,
    pub max_kmeans_reps: usize,
    pub num_centers: usize,
}

pub struct PQGeneration<'a, T, Storage, Pool>
where
    T: VectorRepr,
    Storage: StorageReadProvider + StorageWriteProvider + 'a,
    Pool: AsThreadPool,
{
    table: TransposedTable,
    num_chunks: usize,
    phantom_data: PhantomData<T>,
    phantom_storage: PhantomData<&'a Storage>,
    phantom_pool: PhantomData<Pool>,
}

impl<'a, T, Storage, Pool> QuantCompressor<T> for PQGeneration<'a, T, Storage, Pool>
where
    T: VectorRepr,
    Storage: StorageReadProvider + StorageWriteProvider + 'a,
    Pool: AsThreadPool,
{
    type CompressorContext = PQGenerationContext<'a, Storage, Pool>;

    fn new_at_stage(
        stage: CompressionStage,
        context: &Self::CompressorContext,
    ) -> diskann::ANNResult<Self> {
        // validate that the number of chunks is correct.
        if context.num_chunks > context.dim {
            return Err(ANNError::log_pq_error(
                "Error: number of chunks more than dimension.",
            ));
        }

        let pivots_exists = context
            .pq_storage
            .pivot_data_exist(context.storage_provider);

        let pool = &context.pool;
        forward_threadpool!(pool = pool: Pool);

        if !pivots_exists {
            if stage == CompressionStage::Resume {
                //checks for error case when stage is Resume and pivot data doesn't exist.
                return Err(ANNError::log_pq_error(
                    "Error: Pivot data does not exist when start_vertex_id is not 0.",
                ));
            }

            let timer = Timer::new();

            let rng =
                diskann_providers::utils::create_rnd_provider_from_optional_seed(context.seed);
            let (mut train_data, train_size, train_dim) = context
                .pq_storage
                .get_random_train_data_slice::<T, Storage>(
                    context.p_val,
                    context.storage_provider,
                    &mut rng.create_rnd(),
                )?;

            generate_pq_pivots(
                GeneratePivotArguments::new(
                    train_size,
                    train_dim,
                    context.num_centers,
                    context.num_chunks,
                    context.max_kmeans_reps,
                    context.metric == Metric::L2,
                )?,
                &mut train_data,
                &context.pq_storage,
                context.storage_provider,
                rng,
                pool,
            )?;

            info!(
                "PQ pivot generation took {} seconds",
                timer.elapsed().as_secs_f64()
            );
        }

        let (_, full_dim) = context
            .pq_storage
            .read_existing_pivot_metadata(context.storage_provider)?;

        //Load the pivots
        let num_chunks = context.num_chunks;
        let (mut full_pivot_data, centroid, chunk_offsets, _) =
            context.pq_storage.load_existing_pivot_data(
                &num_chunks,
                &context.num_centers,
                &full_dim,
                context.storage_provider,
                false,
            )?;

        let mut full_pivot_data_mat = diskann_utils::views::MutMatrixView::try_from(
            full_pivot_data.as_mut_slice(),
            context.num_centers,
            full_dim,
        )
        .bridge_err()?;

        accum_row_inplace(full_pivot_data_mat.as_mut_view(), centroid.as_slice());

        let table = TransposedTable::from_parts(
            full_pivot_data_mat.as_view(),
            diskann_quantization::views::ChunkOffsetsView::new(&chunk_offsets)
                .bridge_err()?
                .to_owned(),
        )
        .map_err(|err| ANNError::log_pq_error(diskann_quantization::error::format(&err)))?;

        Ok(Self {
            table,
            num_chunks,
            phantom_data: PhantomData,
            phantom_pool: PhantomData,
            phantom_storage: PhantomData,
        })
    }

    fn compress(
        &self,
        vector: MatrixBase<&[f32]>,
        output: MatrixBase<&mut [u8]>,
    ) -> Result<(), diskann::ANNError> {
        self.table
            .compress_into(vector, output)
            .map_err(|err| ANNError::log_pq_error(diskann_quantization::error::format(&err)))
    }

    fn compressed_bytes(&self) -> usize {
        self.num_chunks
    }
}

//////////////////
///// Tests /////
/////////////////

#[cfg(test)]
mod pq_generation_tests {
    use diskann::ANNError;
    use diskann_providers::model::pq::generate_pq_pivots;
    use diskann_providers::model::GeneratePivotArguments;
    use diskann_providers::storage::{PQStorage, StorageWriteProvider, VirtualStorageProvider};
    use diskann_providers::utils::{
        create_thread_pool_for_test, file_util::load_bin, save_bin_f32, AsThreadPool,
    };
    use diskann_utils::test_data_root;
    use diskann_utils::views::{MatrixView, MutMatrixView};
    use diskann_vector::distance::Metric;
    use rstest::rstest;
    use vfs::{FileSystem, MemoryFS, OverlayFS};

    use super::{CompressionStage, PQGeneration, PQGenerationContext};
    use crate::storage::quant::compressor::QuantCompressor;

    const TEST_PQ_DATA_PATH: &str = "/sift/siftsmall_learn.bin";
    const TEST_PQ_PIVOTS_PATH: &str = "/sift/siftsmall_learn_pq_pivots.bin";
    const TEST_PQ_COMPRESSED_PATH: &str = "/sift/siftsmall_learn_pq_compressed.bin";
    const VALIDATION_DATA: [f32; 40] = [
        //sample validation data: npoints=5, dim=8, 5 vectors [1.0;8] [2.0;8] [2.1;8] [2.2;8] [100.0;8]
        1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32,
        2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32, 2.1f32,
        2.1f32, 2.1f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 2.2f32, 100.0f32,
        100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32, 100.0f32,
    ];
    #[allow(clippy::too_many_arguments)]
    fn create_new_compressor<'a, R: AsThreadPool>(
        stage: CompressionStage,
        provider: &'a VirtualStorageProvider<OverlayFS>,
        dim: usize,
        num_chunks: usize,
        max_kmeans_reps: usize,
        num_centers: usize,
        p_val: f64,
        pool: R,
        pivots_path: String,
        compressed_path: String,
        data_path: Option<&str>,
    ) -> Result<PQGeneration<'a, f32, VirtualStorageProvider<OverlayFS>, R>, ANNError> {
        let pq_storage = PQStorage::new(&pivots_path, &compressed_path, data_path);
        let context = PQGenerationContext::<'_, _, _> {
            pq_storage,
            num_chunks,
            num_centers,
            seed: Some(42),
            p_val,
            max_kmeans_reps,
            storage_provider: provider,
            pool,
            metric: Metric::L2,
            dim,
        };
        PQGeneration::<_, _, _>::new_at_stage(stage, &context)
    }

    #[rstest]
    fn test_create_and_load_pivots_file() {
        let fs = OverlayFS::new(&[MemoryFS::default().into()]);
        fs.create_dir("/pq_generation_tests")
            .expect("Could not create test directory");
        let storage_provider = VirtualStorageProvider::new(fs);

        let pivot_file_name = "/pq_generation_tests/generate_pq_pivots_test.bin";
        let pivot_file_name_compressor = "/pq_generation_tests/compressor_pivots_test.bin";
        let compressed_file_name = "/pq_generation_tests/compressed_not_used.bin";
        let data_path = "/pq_generation_tests/data_path.bin";
        let pq_storage: PQStorage =
            PQStorage::new(pivot_file_name, compressed_file_name, Some(data_path));

        let (ndata, dim, num_centers, num_chunks, max_k_means_reps) = (5, 8, 2, 2, 5);
        let mut train_data: Vec<f32> = VALIDATION_DATA.to_vec();

        let _ = save_bin_f32(
            &mut storage_provider.create_for_write(data_path).unwrap(),
            &train_data,
            ndata,
            dim,
            0,
        );

        let pool = create_thread_pool_for_test();
        generate_pq_pivots(
            GeneratePivotArguments::new(
                ndata,
                dim,
                num_centers,
                num_chunks,
                max_k_means_reps,
                true,
            )
            .unwrap(),
            &mut train_data,
            &pq_storage,
            &storage_provider,
            diskann_providers::utils::create_rnd_provider_from_seed_in_tests(42),
            &pool,
        )
        .unwrap();

        let compressor = create_new_compressor(
            CompressionStage::Start,
            &storage_provider,
            dim,
            num_chunks,
            max_k_means_reps,
            num_centers,
            1.0, //take all the data to compute codebook
            &pool,
            pivot_file_name_compressor.to_string(),
            compressed_file_name.to_string(),
            Some(data_path),
        );

        assert!(compressor.is_ok());

        let compressor = compressor.unwrap();
        assert_eq!(compressor.num_chunks, num_chunks);
        assert_eq!(compressor.compressed_bytes(), num_chunks);

        assert_eq!(compressor.table.dim(), dim);
        assert_eq!(compressor.table.ncenters(), num_centers);
        assert_eq!(compressor.table.nchunks(), num_chunks);

        assert!(&storage_provider.exists(pivot_file_name_compressor));
        let (compressor_pivots, cn, cd) =
            load_bin::<u8, _>(&storage_provider, pivot_file_name_compressor, 0).unwrap();
        let (true_pivots, n, d) = load_bin::<u8, _>(&storage_provider, pivot_file_name, 0).unwrap();

        assert_eq!(cn, n);
        assert_eq!(cd, d);
        assert_eq!(compressor_pivots, true_pivots);
    }

    #[rstest]
    fn throw_error_for_resume_and_no_existing_file() {
        let fs = OverlayFS::new(&[
            MemoryFS::default().into(),
            // PhysicalFS::new("tests/data/").into(),
        ]);
        fs.create_dir("/pq_generation_tests")
            .expect("Could not create test directory");
        let storage_provider = VirtualStorageProvider::new(fs);

        let pivot_file_name = "/pq_generation_tests/generate_pq_pivots_test.bin";
        let compressed_file_name = "/pq_generation_tests/compressed_not_used.bin";
        let data_path = "/pq_generation_tests/data_path.bin";

        let (ndata, dim, num_centers, num_chunks, max_k_means_reps) = (5, 8, 2, 2, 5);

        let _ = save_bin_f32(
            &mut storage_provider.create_for_write(data_path).unwrap(),
            &VALIDATION_DATA,
            ndata,
            dim,
            0,
        );
        let pool = create_thread_pool_for_test();

        let compressor = create_new_compressor(
            CompressionStage::Resume,
            &storage_provider,
            dim,
            num_chunks,
            max_k_means_reps,
            num_centers,
            1.0,
            &pool,
            pivot_file_name.to_string(),
            compressed_file_name.to_string(),
            Some(data_path),
        );

        assert!(compressor.is_err());
    }

    #[rstest]
    fn test_pq_end_to_end_with_codebook() {
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());

        let pool = create_thread_pool_for_test();
        let dim = 128;
        let num_chunks = 1;
        let max_k_means_reps = 10;

        let compressor = create_new_compressor(
            CompressionStage::Resume,
            &storage_provider,
            dim,
            num_chunks,
            max_k_means_reps,
            256,
            1.0,
            &pool,
            TEST_PQ_PIVOTS_PATH.to_string(),
            "".to_string(),
            None,
        );

        if let Err(x) = compressor.as_ref() {
            println!("Error creating compressor: {x}");
        };

        assert!(compressor.is_ok());

        let (data, npts, dim) =
            load_bin::<f32, _>(&storage_provider, TEST_PQ_DATA_PATH, 0).unwrap();
        let mut compressed_mat = vec![0_u8; num_chunks * npts];
        let result = compressor.unwrap().compress(
            MatrixView::try_from(&data, npts, dim).unwrap(),
            MutMatrixView::try_from(&mut compressed_mat, npts, num_chunks).unwrap(),
        );
        assert!(result.is_ok());

        let (compressed_gt, _, _) =
            load_bin::<u8, _>(&storage_provider, TEST_PQ_COMPRESSED_PATH, 0).unwrap();
        assert_eq!(compressed_gt, compressed_mat);
    }

    #[rstest]
    #[case(129, 128, 256)] // num_chunks > dim
    #[case(128, 0, 256)] // num_chunks == 0
    #[case(128, 128, 0)] // num_centers == 0
    fn test_parameter_error_cases(
        #[case] dim: usize,
        #[case] num_chunks: usize,
        #[case] centers: usize,
    ) {
        //test the error cases for parameters: num_chunks > dim, num_chunks == 0, num_centers == 0
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let pool = create_thread_pool_for_test();
        let max_k_means_reps = 10;
        let compressor = create_new_compressor(
            CompressionStage::Start,
            &storage_provider,
            dim,
            num_chunks,
            max_k_means_reps,
            centers,
            1.0,
            &pool,
            TEST_PQ_PIVOTS_PATH.to_string(),
            "".to_string(),
            None,
        );
        assert!(compressor.is_err());
    }
}
