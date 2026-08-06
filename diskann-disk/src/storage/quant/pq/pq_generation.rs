/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, sync::OnceLock, time::Instant};

use diskann::utils::VectorRepr;
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::{
        pq::{accum_row_inplace, generate_pq_pivots},
        GeneratePivotArguments,
    },
    storage::PQStorage,
    utils::{BridgeErr, RayonThreadPoolRef},
};
use diskann_quantization::{error::Format, product::TransposedTable, CompressInto};
use diskann_utils::views::MatrixBase;
use diskann_vector::distance::Metric;
use tracing::info;

use crate::{
    error::{diskann_error, ErrorKind},
    storage::quant::compressor::QuantCompressor,
};

pub struct PQGenerationContext<'a, Storage>
where
    Storage: StorageReadProvider + StorageWriteProvider,
{
    pub pq_storage: PQStorage,
    pub num_chunks: usize,
    pub seed: Option<u64>,
    pub p_val: f64,
    pub storage_provider: &'a Storage,
    pub pool: RayonThreadPoolRef<'a>,
    pub metric: Metric,
    pub dim: usize,
    pub max_kmeans_reps: usize,
    pub num_centers: usize,
}

pub struct PQGeneration<'a, T, Storage>
where
    T: VectorRepr,
    Storage: StorageReadProvider + StorageWriteProvider + 'a,
{
    context: PQGenerationContext<'a, Storage>,
    table: OnceLock<TransposedTable>,
    num_chunks: usize,
    phantom_data: PhantomData<T>,
}

impl<'a, T, Storage> PQGeneration<'a, T, Storage>
where
    T: VectorRepr,
    Storage: StorageReadProvider + StorageWriteProvider + 'a,
{
    pub(crate) fn generate_pivots(
        context: &PQGenerationContext<'a, Storage>,
    ) -> diskann::ANNResult<()> {
        // validate that the number of chunks is correct.
        if context.num_chunks > context.dim {
            return Err(diskann_error!(
                ErrorKind::PQError,
                "Error: number of chunks more than dimension.",
            ));
        }

        let pivots_exists = context
            .pq_storage
            .pivot_data_exist(context.storage_provider);

        let pool = context.pool;

        if !pivots_exists {
            let timer = Instant::now();

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
                )?,
                context.metric == Metric::L2,
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

        Ok(())
    }
}

impl<'a, T, Storage> QuantCompressor<T> for PQGeneration<'a, T, Storage>
where
    T: VectorRepr,
    Storage: StorageReadProvider + StorageWriteProvider + 'a,
{
    type CompressorContext = PQGenerationContext<'a, Storage>;

    fn new(context: Self::CompressorContext) -> Self {
        let num_chunks = context.num_chunks;
        Self {
            context,
            table: OnceLock::new(),
            num_chunks,
            phantom_data: PhantomData,
        }
    }

    fn generate(&self) -> diskann::ANNResult<()> {
        if self.table.get().is_some() {
            return Ok(());
        }

        let context = &self.context;
        Self::generate_pivots(context)?;
        let (_, full_dim) = context
            .pq_storage
            .read_existing_pivot_metadata(context.storage_provider)?;

        //Load the pivots
        let num_chunks = context.num_chunks;
        let (mut full_pivot_data, centroid, chunk_offsets) =
            context.pq_storage.load_existing_pivot_data(
                &num_chunks,
                &context.num_centers,
                &full_dim,
                context.storage_provider,
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
        .map_err(|err| diskann_error!(ErrorKind::PQError, "{}", Format(err)))?;

        self.table.set(table).map_err(|_| {
            diskann_error!(
                ErrorKind::PQError,
                "PQ compressor was generated concurrently"
            )
        })
    }

    fn compress(
        &self,
        vector: MatrixBase<&[f32]>,
        output: MatrixBase<&mut [u8]>,
    ) -> Result<(), diskann::ANNError> {
        self.table
            .get()
            .ok_or_else(|| {
                diskann_error!(
                    ErrorKind::PQError,
                    "PQ compressor must be generated before compression"
                )
            })?
            .compress_into(vector, output)
            .map_err(|err| diskann_error!(ErrorKind::PQError, "{}", Format(err)))
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
    use diskann_providers::storage::{
        PQStorage, StorageReadProvider, StorageWriteProvider, VirtualStorageProvider,
    };
    use diskann_providers::utils::{create_thread_pool_for_test, RayonThreadPoolRef};
    use diskann_utils::{
        io::{read_bin, write_bin},
        test_data_root,
        views::{MatrixView, MutMatrixView},
    };
    use diskann_vector::distance::Metric;
    use rstest::rstest;
    use vfs::FileSystem;

    use super::{PQGeneration, PQGenerationContext};
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
    fn create_context<'a, F: vfs::FileSystem>(
        provider: &'a VirtualStorageProvider<F>,
        dim: usize,
        num_chunks: usize,
        max_kmeans_reps: usize,
        num_centers: usize,
        p_val: f64,
        pool: RayonThreadPoolRef<'a>,
        pivots_path: String,
        compressed_path: String,
        data_path: Option<&str>,
    ) -> PQGenerationContext<'a, VirtualStorageProvider<F>> {
        let pq_storage = PQStorage::new(&pivots_path, &compressed_path, data_path);
        PQGenerationContext::<'_, _> {
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
        }
    }

    #[rstest]
    fn explicit_generation_creates_pivots_file() {
        let storage_provider = VirtualStorageProvider::new_memory();
        storage_provider
            .filesystem()
            .create_dir("/pq_generation_tests")
            .expect("Could not create test directory");

        let pivot_file_name = "/pq_generation_tests/pivots_test.bin";
        let compressed_file_name = "/pq_generation_tests/compressed_not_used.bin";
        let data_path = "/pq_generation_tests/data_path.bin";

        let (ndata, dim, num_centers, num_chunks, max_k_means_reps) = (5, 8, 2, 2, 5);

        write_bin(
            MatrixView::try_from(VALIDATION_DATA.as_slice(), ndata, dim).unwrap(),
            &mut storage_provider.create_for_write(data_path).unwrap(),
        )
        .unwrap();

        let pool = create_thread_pool_for_test();
        let context = create_context(
            &storage_provider,
            dim,
            num_chunks,
            max_k_means_reps,
            num_centers,
            1.0, //take all the data to compute codebook
            pool.as_ref(),
            pivot_file_name.to_string(),
            compressed_file_name.to_string(),
            Some(data_path),
        );

        assert!(!storage_provider.exists(pivot_file_name));

        let compressor = PQGeneration::<f32, _>::new(context);
        assert!(!storage_provider.exists(pivot_file_name));

        let result = compressor.generate();
        assert!(result.is_ok());
        assert!(storage_provider.exists(pivot_file_name));

        assert_eq!(compressor.num_chunks, num_chunks);
        assert_eq!(compressor.compressed_bytes(), num_chunks);

        let table = compressor.table.get().unwrap();
        assert_eq!(table.dim(), dim);
        assert_eq!(table.ncenters(), num_centers);
        assert_eq!(table.nchunks(), num_chunks);
    }

    #[rstest]
    fn generate_creates_missing_pivots() {
        let storage_provider = VirtualStorageProvider::new_memory();
        storage_provider
            .filesystem()
            .create_dir("/pq_generation_tests")
            .expect("Could not create test directory");

        let pivot_file_name = "/pq_generation_tests/missing_pivots.bin";
        let compressed_file_name = "/pq_generation_tests/compressed_not_used.bin";
        let data_path = "/pq_generation_tests/data_path.bin";

        write_bin(
            MatrixView::try_from(VALIDATION_DATA.as_slice(), 5, 8).unwrap(),
            &mut storage_provider.create_for_write(data_path).unwrap(),
        )
        .unwrap();

        let pool = create_thread_pool_for_test();
        let context = create_context(
            &storage_provider,
            8,
            2,
            5,
            2,
            1.0,
            pool.as_ref(),
            pivot_file_name.to_string(),
            compressed_file_name.to_string(),
            Some(data_path),
        );

        let compressor = PQGeneration::<f32, _>::new(context);
        let result = compressor.generate();

        assert!(result.is_ok());
        assert!(storage_provider.exists(pivot_file_name));
    }

    #[rstest]
    fn test_pq_end_to_end_with_codebook() {
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());

        let pool = create_thread_pool_for_test();
        let dim = 128;
        let num_chunks = 1;
        let max_k_means_reps = 10;

        let context = create_context(
            &storage_provider,
            dim,
            num_chunks,
            max_k_means_reps,
            256,
            1.0,
            pool.as_ref(),
            TEST_PQ_PIVOTS_PATH.to_string(),
            "".to_string(),
            None,
        );
        let compressor = PQGeneration::<f32, _>::new(context);
        let result = compressor.generate();

        if let Err(x) = result.as_ref() {
            println!("Error creating compressor: {x}");
        };

        assert!(result.is_ok());

        let data_matrix =
            read_bin::<f32>(&mut storage_provider.open_reader(TEST_PQ_DATA_PATH).unwrap()).unwrap();
        let npts = data_matrix.nrows();
        let mut compressed_mat = vec![0_u8; num_chunks * npts];
        let result = compressor.compress(
            data_matrix.as_view(),
            MutMatrixView::try_from(&mut compressed_mat, npts, num_chunks).unwrap(),
        );
        assert!(result.is_ok());

        let compressed_gt = read_bin::<u8>(
            &mut storage_provider
                .open_reader(TEST_PQ_COMPRESSED_PATH)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(compressed_gt.as_slice(), &compressed_mat);
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
        let context = create_context(
            &storage_provider,
            dim,
            num_chunks,
            max_k_means_reps,
            centers,
            1.0,
            pool.as_ref(),
            TEST_PQ_PIVOTS_PATH.to_string(),
            "".to_string(),
            None,
        );
        let result = PQGeneration::<f32, _>::new(context).generate();
        assert!(result.is_err());
    }
}
