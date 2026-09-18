/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, time::Instant};

use diskann::utils::VectorRepr;
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::{pq::generate_pq_pivots, GeneratePivotArguments},
    storage::PQStorage,
    utils::RayonThreadPoolRef,
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

/// A freshly generated disk-search PQ codebook used for batch compression.
/// Construction trains and saves a new codebook even if a pivot file already exists.
pub struct PQGeneration<'a, T, Storage>
where
    T: VectorRepr,
    Storage: StorageReadProvider + StorageWriteProvider + 'a,
{
    table: TransposedTable,
    num_chunks: usize,
    phantom_data: PhantomData<T>,
    phantom_storage: PhantomData<&'a Storage>,
}

impl<'a, T, Storage> QuantCompressor<T> for PQGeneration<'a, T, Storage>
where
    T: VectorRepr,
    Storage: StorageReadProvider + StorageWriteProvider + 'a,
{
    type CompressorContext = PQGenerationContext<'a, Storage>;

    fn new(context: &Self::CompressorContext) -> diskann::ANNResult<Self> {
        if context.num_chunks == 0 || context.num_chunks > context.dim {
            return Err(diskann_error!(
                ErrorKind::PQError,
                "PQ chunks must be between 1 and {}, received {}",
                context.dim,
                context.num_chunks
            ));
        }
        if !(1..=diskann_providers::model::NUM_PQ_CENTROIDS).contains(&context.num_centers) {
            return Err(diskann_error!(
                ErrorKind::PQError,
                "PQ centers must be between 1 and {}, received {}",
                diskann_providers::model::NUM_PQ_CENTROIDS,
                context.num_centers
            ));
        }

        let timer = Instant::now();
        let rng = diskann_providers::utils::create_rnd_provider_from_optional_seed(context.seed);
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
            context.pool,
        )?;

        info!(
            "PQ pivot generation took {} seconds",
            timer.elapsed().as_secs_f64()
        );

        let num_chunks = context.num_chunks;
        let table = context.pq_storage.load_pivots(context.storage_provider)?;

        if table.nchunks() != num_chunks
            || table.ncenters() != context.num_centers
            || table.dim() != train_dim
        {
            return Err(diskann_error!(
                ErrorKind::PQError,
                "PQ pivot table mismatch: file has {} chunks, {} centers in {} dimensions but expected {} chunks, {} centers in {} dimensions.",
                table.nchunks(),
                table.ncenters(),
                table.dim(),
                num_chunks,
                context.num_centers,
                train_dim
            ));
        }

        let table =
            TransposedTable::from_parts(table.view_pivots(), table.view_offsets().to_owned())
                .map_err(|err| diskann_error!(ErrorKind::PQError, "{}", Format(err)))?;

        Ok(Self {
            table,
            num_chunks,
            phantom_data: PhantomData,
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
    use std::io::{Read, Write};

    use diskann::ANNError;
    use diskann_providers::model::pq::generate_pq_pivots;
    use diskann_providers::model::GeneratePivotArguments;
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
    use crate::storage::quant::{QuantCompressor, QuantDataGenerator};

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

    #[allow(clippy::too_many_arguments)]
    fn create_new_compressor<'a, F: vfs::FileSystem>(
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
    ) -> Result<PQGeneration<'a, f32, VirtualStorageProvider<F>>, ANNError> {
        let context = create_context(
            provider,
            dim,
            num_chunks,
            max_kmeans_reps,
            num_centers,
            p_val,
            pool,
            pivots_path,
            compressed_path,
            data_path,
        );
        PQGeneration::<f32, _>::new(&context)
    }

    #[rstest]
    fn construction_trains_fresh_codebook_and_compression_reuses_it() {
        let storage_provider = VirtualStorageProvider::new_memory();
        storage_provider
            .filesystem()
            .create_dir("/pq_generation_tests")
            .expect("Could not create test directory");

        let pivot_file_name = "/pq_generation_tests/construction_pivots.bin";
        let compressed_file_name = "/pq_generation_tests/construction_compressed.bin";
        let data_path = "/pq_generation_tests/construction_data.bin";

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

        let generator = QuantDataGenerator::<f32, PQGeneration<f32, _>>::new(
            data_path.into(),
            compressed_file_name.into(),
            &context,
        )
        .unwrap();
        assert!(storage_provider.exists(pivot_file_name));
        assert!(!storage_provider.exists(compressed_file_name));
        let first_pivots = read_file(&storage_provider, pivot_file_name);

        generator
            .generate_data(&storage_provider, pool.as_ref(), 2)
            .unwrap();
        assert_eq!(first_pivots, read_file(&storage_provider, pivot_file_name));

        let updated_data: Vec<f32> = VALIDATION_DATA.iter().map(|x| x + 10.0).collect();
        write_bin(
            MatrixView::try_from(updated_data.as_slice(), ndata, dim).unwrap(),
            &mut storage_provider.create_for_write(data_path).unwrap(),
        )
        .unwrap();

        let generator = QuantDataGenerator::<f32, PQGeneration<f32, _>>::new(
            data_path.into(),
            compressed_file_name.into(),
            &context,
        )
        .unwrap();
        assert_ne!(first_pivots, read_file(&storage_provider, pivot_file_name));
        generator
            .generate_data(&storage_provider, pool.as_ref(), 2)
            .unwrap();

        let fresh_context = create_context(
            &storage_provider,
            dim,
            num_chunks,
            max_k_means_reps,
            num_centers,
            1.0,
            pool.as_ref(),
            "/pq_generation_tests/fresh_pivots.bin".into(),
            "/pq_generation_tests/fresh_compressed.bin".into(),
            Some(data_path),
        );
        let compressor = PQGeneration::<f32, _>::new(&fresh_context).unwrap();
        assert_eq!(
            read_file(&storage_provider, pivot_file_name),
            read_file(&storage_provider, "/pq_generation_tests/fresh_pivots.bin")
        );

        let mut expected_codes = vec![0; ndata * num_chunks];
        compressor
            .compress(
                MatrixView::try_from(updated_data.as_slice(), ndata, dim).unwrap(),
                MutMatrixView::try_from(&mut expected_codes, ndata, num_chunks).unwrap(),
            )
            .unwrap();
        let codes =
            read_bin::<u8>(&mut storage_provider.open_reader(compressed_file_name).unwrap())
                .unwrap();
        assert_eq!(codes.as_slice(), expected_codes);
    }

    fn read_file<Storage: StorageReadProvider>(storage: &Storage, path: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        storage
            .open_reader(path)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        bytes
    }

    #[rstest]
    #[case(9, 2, "PQ chunks")]
    #[case(0, 2, "PQ chunks")]
    #[case(2, 0, "PQ centers")]
    #[case(2, 257, "PQ centers")]
    fn invalid_pq_parameters_preserve_existing_outputs(
        #[case] num_chunks: usize,
        #[case] num_centers: usize,
        #[case] expected_error: &str,
    ) {
        let storage_provider = VirtualStorageProvider::new_memory();
        let data_path = "/data.bin";
        let pivots_path = "/pivots.bin";
        let codes_path = "/codes.bin";
        write_bin(
            MatrixView::try_from(VALIDATION_DATA.as_slice(), 5, 8).unwrap(),
            &mut storage_provider.create_for_write(data_path).unwrap(),
        )
        .unwrap();
        let old_pivots = b"existing pivots";
        let old_codes = b"existing compressed data";
        storage_provider
            .create_for_write(pivots_path)
            .unwrap()
            .write_all(old_pivots)
            .unwrap();
        storage_provider
            .create_for_write(codes_path)
            .unwrap()
            .write_all(old_codes)
            .unwrap();
        let pool = create_thread_pool_for_test();
        let context = create_context(
            &storage_provider,
            8,
            num_chunks,
            5,
            num_centers,
            1.0,
            pool.as_ref(),
            pivots_path.into(),
            codes_path.into(),
            Some(data_path),
        );
        let error = QuantDataGenerator::<f32, PQGeneration<f32, _>>::new(
            data_path.into(),
            codes_path.into(),
            &context,
        )
        .err()
        .expect("invalid PQ parameters must be rejected before training");
        assert!(error.to_string().contains(expected_error), "{error}");
        assert_eq!(read_file(&storage_provider, pivots_path), old_pivots);
        assert_eq!(read_file(&storage_provider, codes_path), old_codes);
    }

    #[rstest]
    fn test_create_and_load_pivots_file() {
        let storage_provider = VirtualStorageProvider::new_memory();
        storage_provider
            .filesystem()
            .create_dir("/pq_generation_tests")
            .expect("Could not create test directory");

        let pivot_file_name = "/pq_generation_tests/generate_pq_pivots_test.bin";
        let pivot_file_name_compressor = "/pq_generation_tests/compressor_pivots_test.bin";
        let compressed_file_name = "/pq_generation_tests/compressed_not_used.bin";
        let data_path = "/pq_generation_tests/data_path.bin";
        let pq_storage: PQStorage =
            PQStorage::new(pivot_file_name, compressed_file_name, Some(data_path));

        let (ndata, dim, num_centers, num_chunks, max_k_means_reps) = (5, 8, 2, 2, 5);
        let mut train_data: Vec<f32> = VALIDATION_DATA.to_vec();

        write_bin(
            MatrixView::try_from(train_data.as_slice(), ndata, dim).unwrap(),
            &mut storage_provider.create_for_write(data_path).unwrap(),
        )
        .unwrap();

        let pool = create_thread_pool_for_test();
        generate_pq_pivots(
            GeneratePivotArguments::new(ndata, dim, num_centers, num_chunks, max_k_means_reps)
                .unwrap(),
            true,
            &mut train_data,
            &pq_storage,
            &storage_provider,
            diskann_providers::utils::create_rnd_provider_from_seed_in_tests(42),
            pool.as_ref(),
        )
        .unwrap();

        let compressor = create_new_compressor(
            &storage_provider,
            dim,
            num_chunks,
            max_k_means_reps,
            num_centers,
            1.0, //take all the data to compute codebook
            pool.as_ref(),
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
        let compressor_pivots = read_file(&storage_provider, pivot_file_name_compressor);
        let true_pivots = read_file(&storage_provider, pivot_file_name);
        assert_eq!(compressor_pivots, true_pivots);
    }

    #[rstest]
    fn test_pq_end_to_end_with_codebook() {
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());

        let dim = 128;
        let num_chunks = 1;

        // Keep the fixed-codebook compression regression independent of training.
        let pq_storage = PQStorage::new(TEST_PQ_PIVOTS_PATH, "", None);
        let pivots = pq_storage.load_pivots(&storage_provider).unwrap();
        let table = diskann_quantization::product::TransposedTable::from_parts(
            pivots.view_pivots(),
            pivots.view_offsets().to_owned(),
        )
        .unwrap();
        assert_eq!(table.dim(), dim);

        let data_matrix =
            read_bin::<f32>(&mut storage_provider.open_reader(TEST_PQ_DATA_PATH).unwrap()).unwrap();
        let npts = data_matrix.nrows();
        let mut compressed_mat = vec![0_u8; num_chunks * npts];
        use diskann_quantization::CompressInto;
        let result = table.compress_into(
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
}
