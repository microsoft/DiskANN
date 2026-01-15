/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{marker::PhantomData, sync::Arc};

use diskann::ANNResult;
use diskann_providers::storage::StorageReadProvider;
use diskann_providers::{model::pq::PQData, storage::PQStorage, utils::load_metadata_from_file};
use tracing::info;

/// This struct is used by the DiskIndexSearcher to read the index data from storage. Noted that the index data here is different from index graph,
/// It includes the PQ data, pivot table, and the warmup query data.
/// The Storage acts as a provider to read the data from storage system.
/// The storage provider should be provided as a generic type and be specified by the caller when it initializes the DiskIndexSearcher.
pub struct DiskIndexReader<VectorType> {
    phantom: PhantomData<VectorType>,

    pq_data: Arc<PQData>,

    num_points: usize,
}

impl<VectorType> DiskIndexReader<VectorType> {
    /// Create DiskIndexReader instance
    pub fn new<Storage: StorageReadProvider>(
        pq_pivot_path: String,
        pq_compressed_data_path: String,
        storage_provider: &Storage,
    ) -> ANNResult<Self> {
        let pq_storage = PQStorage::new(&pq_pivot_path, &pq_compressed_data_path, None);
        let pq_pivot_table = pq_storage.load_pq_pivots_bin::<Storage>(
            &pq_pivot_path,
            0, // Use 0 to infer num_pq_chunks from the file
            storage_provider,
        )?;

        // Auto-detect number of points from compressed PQ file metadata
        let metadata = load_metadata_from_file(storage_provider, &pq_compressed_data_path)?;

        let pq_compressed_data = PQStorage::load_pq_compressed_vectors_bin::<Storage>(
            &pq_compressed_data_path,
            metadata.npoints,
            pq_pivot_table.get_num_chunks(),
            storage_provider,
        )?;
        info!(
            "Loaded PQ centroids and in-memory compressed vectors. #points:{} #pq_chunks: {}",
            metadata.npoints,
            pq_pivot_table.get_num_chunks()
        );

        Ok(DiskIndexReader {
            phantom: PhantomData,
            pq_data: Arc::<PQData>::new(PQData::new(pq_pivot_table, pq_compressed_data)?),
            num_points: metadata.npoints,
        })
    }

    pub fn get_pq_data(&self) -> Arc<PQData> {
        Arc::clone(&self.pq_data)
    }

    pub fn get_num_points(&self) -> usize {
        self.num_points
    }
}

#[cfg(test)]
mod disk_index_storage_test {
    use diskann::ANNErrorKind;
    use diskann_providers::storage::VirtualStorageProvider;
    use diskann_utils::test_data_root;
    use vfs::OverlayFS;

    use super::*;

    #[test]
    fn load_pivot_test() {
        let pivot_file_prefix: &str = "/sift/siftsmall_learn";
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let storage = DiskIndexReader::<f32>::new::<VirtualStorageProvider<OverlayFS>>(
            pivot_file_prefix.to_string() + "_pq_pivots.bin",
            pivot_file_prefix.to_string() + "_pq_compressed.bin",
            &storage_provider,
        )
        .unwrap();

        // Creating the backend storage is sufficient to verify the constraints on the
        // PQ schema as both `FixedChunkPQTable` and the possible alternatives (such as
        // `quantization::TransposedTable`) check for the well-formedness of the schema.
        let _: Arc<PQData> = storage.get_pq_data();
    }

    #[test]
    fn load_pivot_file_not_exist_test() {
        let pivot_file_prefix: &str = "/sift/siftsmall_learn_file_not_exist";
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let err = match DiskIndexReader::<f32>::new::<VirtualStorageProvider<OverlayFS>>(
            pivot_file_prefix.to_string() + "_pq_pivots.bin",
            pivot_file_prefix.to_string() + "_pq_compressed.bin",
            &storage_provider,
        ) {
            Ok(_) => panic!("this function should not have succeeded"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), ANNErrorKind::PQError);
        assert!(err.to_string().contains("PQ k-means pivot file not found"));
    }

    #[test]
    fn test_get_num_points() {
        let pivot_file_prefix: &str = "/sift/siftsmall_learn";
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let storage = DiskIndexReader::<f32>::new::<VirtualStorageProvider<OverlayFS>>(
            pivot_file_prefix.to_string() + "_pq_pivots.bin",
            pivot_file_prefix.to_string() + "_pq_compressed.bin",
            &storage_provider,
        )
        .unwrap();

        let num_points = storage.get_num_points();
        assert_eq!(num_points, 25000);
    }
}
