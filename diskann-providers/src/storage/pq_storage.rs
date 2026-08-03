/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::io::{Seek, SeekFrom, Write};

use super::{StorageReadProvider, StorageWriteProvider};
use diskann::{
    ANNError, ANNResult,
    utils::{IntoUsize, VectorRepr},
};
use diskann_quantization::{product::BasicTable, views::ChunkOffsetsBase};
use diskann_utils::{
    io::{Metadata, read_bin, write_bin},
    views::{Matrix, MatrixView},
};
use rand::Rng;
use tracing::info;

use crate::{
    model::{
        NUM_PQ_CENTROIDS,
        pq::{METADATA_SIZE, accum_row_inplace},
    },
    utils::{gen_random_slice, read_bin_from, write_bin_from},
};

#[derive(Debug, Clone)]
pub struct PQStorage {
    /// Pivot table path
    pivot_data_path: String,

    /// Compressed pivot path
    compressed_data_path: String,

    /// Data stream used to construct PQ table and PQ compressed table.  If PQStorage is used
    /// for reading then this can be None
    data_path: Option<String>,
}

impl PQStorage {
    pub fn new(pivot_data_path: &str, compressed_data_path: &str, data_path: Option<&str>) -> Self {
        Self {
            pivot_data_path: pivot_data_path.to_string(),
            compressed_data_path: compressed_data_path.to_string(),
            data_path: data_path.map(|x| x.to_string()),
        }
    }

    pub fn write_compressed_pivot_metadata<Storage>(
        &self,
        npts: usize,
        pq_chunk: usize,
        writer: &mut Storage::Writer,
    ) -> ANNResult<()>
    where
        Storage: StorageWriteProvider,
    {
        Metadata::new(npts, pq_chunk)?.write(writer)?;
        Ok(())
    }

    /// Write the pivot table to file.
    ///
    /// # Arguments
    /// * `full_pivot_data` - the pivot table data
    /// * `centroid` - Optional per-dimension centroid. Pass `None` for the standard
    ///   (non-legacy) code path; a zero vector of length `dim` is written to preserve
    ///   the on-disk file format. Pass `Some(centroid)` only when legacy centroid
    ///   centering is enabled (see [`GeneratePivotArguments::with_legacy_centering`]).
    /// * `chunk_offsets` - the chunk offsets of the pivot table
    /// * `num_centers` - the number of centers
    /// * `dim` - the dimension of the pivot table
    /// * `storage_provider` - the storage provider
    ///
    /// # Return
    /// * `Result` - the result of writing the pivot table
    ///
    /// # Remarks
    /// * 4k bytes are reserved for metadata at the beginning of the file
    /// * the metadata is written in the following order:
    ///     * the size of the metadata
    ///     * the offset of the pivot table data
    ///     * the offset of the centroid of the pivot table
    ///     * the offset of the chunk offsets of the pivot table
    /// * the pivot table data: num_centers * dim
    /// * the centroid of the pivot table: dim*1
    /// * the chunk offsets of the pivot table: (num_pq_chunks) + 1 * 1
    pub fn write_pivot_data<Storage>(
        &self,
        full_pivot_data: &[f32],
        centroid: Option<&[f32]>,
        chunk_offsets: &[usize],
        num_centers: usize,
        dim: usize,
        storage_provider: &Storage,
    ) -> ANNResult<()>
    where
        Storage: StorageWriteProvider,
    {
        let mut cumul_bytes: Vec<usize> = vec![0; 4];
        cumul_bytes[0] = METADATA_SIZE;
        let writer = &mut storage_provider.create_for_write(&self.pivot_data_path)?;

        // Skip past the offset table — we'll write it last once we know all offsets.
        writer.seek(SeekFrom::Start(cumul_bytes[0] as u64))?;

        // Write PQ centroid vectors
        let pivot_view = MatrixView::try_from(full_pivot_data, num_centers, dim)?;
        cumul_bytes[1] = cumul_bytes[0] + write_bin(pivot_view, writer)?;

        // Write the centroid of PQ centroid vectors
        let centroid_bytes = match centroid {
            Some(centroid) => write_bin(MatrixView::column_vector(centroid), writer)?,
            None => write_bin(Matrix::<f32>::new(0.0, dim, 1).as_view(), writer)?,
        };
        cumul_bytes[2] = cumul_bytes[1] + centroid_bytes;

        // Write PQ chunk offsets
        let chunk_offsets_u32: Vec<u32> = chunk_offsets.iter().map(|&x| x as u32).collect();
        cumul_bytes[3] = cumul_bytes[2]
            + write_bin(
                MatrixView::column_vector(chunk_offsets_u32.as_slice()),
                writer,
            )?;

        // Seek back to offset 0 and write the offset table.
        let cumul_bytes_u64: Vec<u64> = cumul_bytes.iter().map(|&x| x as u64).collect();
        write_bin_from(
            MatrixView::column_vector(cumul_bytes_u64.as_slice()),
            writer,
            0,
        )?;

        writer.flush()?;
        Ok(())
    }

    pub fn pivot_data_exist<Storage>(&self, storage_provider: &Storage) -> bool
    where
        Storage: StorageReadProvider,
    {
        storage_provider.exists(&self.pivot_data_path)
    }

    pub fn read_existing_pivot_metadata<Storage>(
        &self,
        storage_provider: &Storage,
    ) -> std::io::Result<(usize, usize)>
    where
        Storage: StorageReadProvider,
    {
        let reader = &mut storage_provider.open_reader(&self.pivot_data_path)?;
        reader.seek(SeekFrom::Start(METADATA_SIZE as u64))?;
        Ok(Metadata::read(reader)?.into_dims())
    }

    /// Load the effective PQ pivot table from a pivot file.
    ///
    /// The loader validates internal consistency: file layout,
    /// centroid/pivot dimensions, offset monotonicity and bounds, and
    /// `BasicTable` invariants. Callers are responsible for validating that the
    /// resulting table is compatible with their build or search configuration.
    pub fn load_pivots<Storage: StorageReadProvider>(
        &self,
        storage_provider: &Storage,
    ) -> ANNResult<BasicTable> {
        let (mut pivots, centroid, chunk_offsets) =
            self.read_pivot_file(&self.pivot_data_path, None, None, storage_provider)?;

        if centroid.as_slice().iter().any(|c| *c != 0.0) {
            accum_row_inplace(pivots.as_mut_view(), centroid.as_slice())
        }

        Self::pivot_data_into_basic_table(&self.pivot_data_path, pivots, chunk_offsets)
    }

    /// Load the compressed pq dataset from file.
    ///
    /// Returns a `num_points × num_pq_chunks` matrix of u8 codes.
    pub fn load_pq_compressed_vectors_bin<Storage: StorageReadProvider>(
        pq_compressed_data: &str,
        num_points_to_load: usize,
        num_pq_chunks: usize,
        storage_provider: &Storage,
    ) -> ANNResult<Matrix<u8>> {
        info!(
            "Loading compressed from pq compressed data file {}...",
            pq_compressed_data,
        );

        info!(
            "# of Points: {} , # PQ chunks: {} ",
            num_points_to_load, num_pq_chunks
        );

        let data = read_bin::<u8>(&mut storage_provider.open_reader(pq_compressed_data)?)?;

        if data.nrows() != num_points_to_load || data.ncols() != num_pq_chunks {
            return Err(ANNError::message(format!(
                "PQ compressed data mismatch: file has {}x{} but expected {}x{}",
                data.nrows(),
                data.ncols(),
                num_points_to_load,
                num_pq_chunks
            )));
        }

        info!("PQ compressed dataset loaded.");
        Ok(data)
    }

    fn read_pivot_file<Storage: StorageReadProvider>(
        &self,
        pq_pivots: &str,
        expected_num_centers: Option<usize>,
        expected_dim: Option<usize>,
        storage_provider: &Storage,
    ) -> ANNResult<(Matrix<f32>, Matrix<f32>, Matrix<usize>)> {
        if !storage_provider.exists(pq_pivots) {
            return Err(ANNError::message(format!(
                "ERROR: PQ k-means pivot file not found: {pq_pivots}."
            )));
        }

        info!("Loading PQ pivots from {}...", pq_pivots);

        let mut reader = storage_provider.open_reader(pq_pivots)?;

        // File layout: offset table(4*1) -> pivot data(num_centers*dim) ->
        // centroid(dim*1) -> chunk offsets(num_chunks+1*1).
        //
        // The expected values here are file-format checks only. Structural PQ table
        // invariants, such as chunk-offset monotonicity and pivot/offset dimension
        // agreement, are validated by `ChunkOffsetsBase` and `BasicTable`.
        let offsets = read_bin_from::<u64>(&mut reader, 0)?;
        if offsets.nrows() != 4 || offsets.ncols() != 1 {
            return Err(ANNError::message(format!(
                "Error reading pq_pivots file {}. Offsets don't contain correct metadata, \
                 file has nr={}, nc={}, but expecting nr=4 and nc=1.",
                pq_pivots,
                offsets.nrows(),
                offsets.ncols()
            )));
        }
        let file_offset_data = offsets.map(|x| x.into_usize());

        info!(" Offset data: {:?}", file_offset_data.as_slice());

        let pivots = read_bin_from::<f32>(&mut reader, file_offset_data[(0, 0)])?;
        if let Some(num_centers) = expected_num_centers {
            if pivots.nrows() != num_centers {
                return Err(ANNError::message(format!(
                    "Error reading pq_pivots file {}. file_num_centers = {}, but expecting {} centers.",
                    pq_pivots,
                    pivots.nrows(),
                    num_centers
                )));
            }
        } else if pivots.nrows() > NUM_PQ_CENTROIDS {
            return Err(ANNError::message(format!(
                "Error reading pq_pivots file {}. file_num_centers = {}, but expecting {} centers.",
                pq_pivots,
                pivots.nrows(),
                NUM_PQ_CENTROIDS
            )));
        }

        if let Some(dim) = expected_dim
            && pivots.ncols() != dim
        {
            return Err(ANNError::message(format!(
                "Error reading pq_pivots file {}. file_dim = {} but expecting {} dimensions.",
                pq_pivots,
                pivots.ncols(),
                dim
            )));
        }

        let centroid = read_bin_from::<f32>(&mut reader, file_offset_data[(1, 0)])?;
        if centroid.nrows() != pivots.ncols() || centroid.ncols() != 1 {
            return Err(ANNError::message(format!(
                "Error reading pq_pivots file {}. file_dim = {}, file_cols = {} \
                 but expecting {} entries in 1 dimension.",
                pq_pivots,
                centroid.nrows(),
                centroid.ncols(),
                pivots.ncols()
            )));
        }

        let chunk_offsets_m = read_bin_from::<u32>(&mut reader, file_offset_data[(2, 0)])?;
        if chunk_offsets_m.ncols() != 1 {
            return Err(ANNError::message(format!(
                "Error reading pq_pivots file at chunk offsets; file has nc={}, but expecting nc=1.",
                chunk_offsets_m.ncols()
            )));
        }
        let chunk_offsets = chunk_offsets_m.map(|x| x.into_usize());

        Ok((pivots, centroid, chunk_offsets))
    }

    fn pivot_data_into_basic_table(
        pq_pivots: &str,
        pivots: Matrix<f32>,
        chunk_offsets: Matrix<usize>,
    ) -> ANNResult<BasicTable> {
        let offsets = ChunkOffsetsBase::new(chunk_offsets.into_inner()).map_err(|err| {
            ANNError::message(format!(
                "Error constructing chunk offsets from pq_pivots file {}: {}",
                pq_pivots,
                diskann_quantization::error::format(&err)
            ))
        })?;

        BasicTable::new(pivots, offsets).map_err(|err| {
            ANNError::message(format!(
                "Error constructing PQ table from pq_pivots file {}: {}",
                pq_pivots,
                diskann_quantization::error::format(&err)
            ))
        })
    }

    /// streams data from the file, and samples each vector with probability p_val
    /// and returns a matrix of size slice_size* ndims as floating point type.
    /// the slice_size and ndims are set inside the function.
    /// # Arguments
    /// * `file_name` - filename where the data is
    /// * `p_val` - possibility to sample data
    /// * `sampled_vectors` - sampled vector chose by p_val possibility
    /// * `slice_size` - how many sampled data return
    /// * `dim` - each sample data dimension
    pub fn get_random_train_data_slice<T: VectorRepr, Storage>(
        &self,
        p_val: f64,
        storage_provider: &Storage,
        generator: &mut impl Rng,
    ) -> ANNResult<(Vec<f32>, usize, usize)>
    where
        Storage: StorageReadProvider,
    {
        gen_random_slice::<T, _>(self.get_data_path()?, p_val, storage_provider, generator)
    }

    pub fn get_data_path(&self) -> ANNResult<&str> {
        self.data_path
            .as_ref()
            .ok_or_else(|| ANNError::message("expected \"data_path\" to be configured"))
            .map(|s| s.as_str())
    }

    pub fn get_compressed_data_path(&self) -> &str {
        &self.compressed_data_path
    }
}

#[cfg(test)]
mod pq_storage_tests {

    use crate::storage::VirtualStorageProvider;
    use diskann_utils::test_data_root;
    use vfs::MemoryFS;

    use super::*;
    use crate::utils::gen_random_slice;

    const DATA_FILE: &str = "/sift/siftsmall_learn.bin";
    const PQ_PIVOT_PATH: &str = "/sift/siftsmall_learn_pq_pivots.bin";
    const PQ_COMPRESSED_PATH: &str = "/sift/empty_pq_compressed.bin";

    fn write_test_pivots(
        storage_provider: &VirtualStorageProvider<MemoryFS>,
        pivot_path: &str,
        num_centers: usize,
        dim: usize,
        centroid: Option<&[f32]>,
        chunk_offsets: &[usize],
    ) {
        let pivots: Vec<f32> = (0..num_centers * dim).map(|i| i as f32).collect();
        PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None)
            .write_pivot_data(
                &pivots,
                centroid,
                chunk_offsets,
                num_centers,
                dim,
                storage_provider,
            )
            .unwrap();
    }

    #[test]
    fn new_test() {
        PQStorage::new(PQ_PIVOT_PATH, PQ_COMPRESSED_PATH, Some(DATA_FILE));
    }

    #[test]
    fn write_compressed_pivot_metadata_test() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let compress_pivot_path = "/write_compressed_pivot_metadata_test.bin";
        let result = PQStorage::new(PQ_PIVOT_PATH, compress_pivot_path, Some(DATA_FILE));
        {
            let mut writer = storage_provider
                .create_for_write(compress_pivot_path)
                .unwrap();

            result
                .write_compressed_pivot_metadata::<VirtualStorageProvider<MemoryFS>>(
                    100,
                    20,
                    &mut writer,
                )
                .unwrap();
        }

        let mut result_reader = storage_provider.open_reader(compress_pivot_path).unwrap();
        let metadata = Metadata::read(&mut result_reader).unwrap();

        assert_eq!(metadata.npoints(), 100);
        assert_eq!(metadata.ndims(), 20);

        storage_provider.delete(compress_pivot_path).unwrap();
    }

    #[test]
    fn pivot_data_exist_test() {
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let result = PQStorage::new(PQ_PIVOT_PATH, PQ_COMPRESSED_PATH, Some(DATA_FILE));
        assert!(result.pivot_data_exist(&storage_provider));

        let pivot_path = "not_exist_pivot_path.bin";
        let result = PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, Some(DATA_FILE));
        assert!(!result.pivot_data_exist(&storage_provider));
    }

    #[test]
    fn read_pivot_metadata_test() {
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let result = PQStorage::new(PQ_PIVOT_PATH, PQ_COMPRESSED_PATH, Some(DATA_FILE));
        let (npt, dim) = result
            .read_existing_pivot_metadata(&storage_provider)
            .unwrap();

        assert_eq!(npt, 256);
        assert_eq!(dim, 128);
    }

    #[test]
    fn load_pivot_data_test() {
        let storage_provider = VirtualStorageProvider::new_overlay(test_data_root());
        let result = PQStorage::new(PQ_PIVOT_PATH, PQ_COMPRESSED_PATH, Some(DATA_FILE));
        let table = result.load_pivots(&storage_provider).unwrap();

        assert_eq!(table.view_pivots().as_slice().len(), 256 * 128);
        assert_eq!(table.dim(), 128);
        assert_eq!(table.nchunks(), 1);
    }

    /// Write pivot data with `centroid = None`, read it back, and verify the
    /// effective pivots are unchanged.
    #[test]
    fn write_read_roundtrip_no_centroid() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let pivot_path = "/roundtrip_no_centroid_pivots.bin";

        let num_centers = 3;
        let dim = 4;
        let num_pq_chunks = 2;
        let pivots: Vec<f32> = (0..num_centers * dim).map(|i| i as f32).collect();
        let chunk_offsets = vec![0, 2, dim];

        let pq_storage = PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None);
        pq_storage
            .write_pivot_data(
                &pivots,
                None,
                &chunk_offsets,
                num_centers,
                dim,
                &storage_provider,
            )
            .unwrap();

        let table = pq_storage.load_pivots(&storage_provider).unwrap();

        assert_eq!(table.view_pivots().as_slice(), pivots);
        assert_eq!(table.view_offsets().as_slice(), chunk_offsets);
        assert_eq!(table.nchunks(), num_pq_chunks);
    }

    #[test]
    fn load_pq_pivots_infer_chunks_loads_without_expected_count() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let pivot_path = "/infer_chunk_count_pivots.bin";

        let num_centers = 3;
        let dim = 4;
        let pivots: Vec<f32> = (0..num_centers * dim).map(|i| i as f32).collect();
        let chunk_offsets = vec![0, 2, dim];

        let pq_storage = PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None);
        pq_storage
            .write_pivot_data(
                &pivots,
                None,
                &chunk_offsets,
                num_centers,
                dim,
                &storage_provider,
            )
            .unwrap();

        let table = pq_storage.load_pivots(&storage_provider).unwrap();

        assert_eq!(table.view_pivots().as_slice(), pivots);
    }

    #[test]
    fn load_pq_pivots_zero_chunk_count_infers_from_file() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let pivot_path = "/zero_chunk_count_pivots.bin";
        let pivots: Vec<f32> = (0..12).map(|i| i as f32).collect();

        PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None)
            .write_pivot_data(&pivots, None, &[0, 2, 4], 3, 4, &storage_provider)
            .unwrap();

        let table = PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None)
            .load_pivots(&storage_provider)
            .unwrap();

        assert_eq!(table.view_pivots().as_slice(), pivots);
    }

    #[test]
    fn load_pq_pivots_rejects_invalid_centroid() {
        let storage_provider = VirtualStorageProvider::new_memory();

        let wrong_centroid_path = "/wrong_centroid_pivots.bin";
        write_test_pivots(
            &storage_provider,
            wrong_centroid_path,
            3,
            4,
            Some(&[1.0, 2.0, 3.0]),
            &[0, 2, 4],
        );
        assert!(
            PQStorage::new(wrong_centroid_path, PQ_COMPRESSED_PATH, None)
                .load_pivots(&storage_provider)
                .is_err()
        );
    }

    #[test]
    fn load_pivot_data_rejects_invalid_chunk_bounds() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let pivot_path = "/legacy_chunk_bounds_pivots.bin";

        write_test_pivots(&storage_provider, pivot_path, 3, 4, None, &[1, 4]);

        let err = PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None)
            .load_pivots(&storage_provider)
            .unwrap_err();

        assert!(err.to_string().contains("offsets must begin at 0"));
    }

    #[test]
    fn load_pivot_data_rejects_chunk_offsets_dim_mismatch() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let pivot_path = "/chunk_offsets_dim_mismatch_pivots.bin";

        write_test_pivots(&storage_provider, pivot_path, 3, 4, None, &[0, 2, 3]);

        let err = PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None)
            .load_pivots(&storage_provider)
            .unwrap_err();

        assert!(err.to_string().contains("offsets expect 3"));
    }

    #[test]
    fn load_pq_pivots_infer_rejects_invalid_chunk_bounds() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let pivot_path = "/infer_wrong_chunk_bounds_pivots.bin";

        write_test_pivots(&storage_provider, pivot_path, 3, 4, None, &[1, 4]);

        let err = PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None)
            .load_pivots(&storage_provider)
            .unwrap_err();

        assert!(err.to_string().contains("offsets must begin at 0"));
    }

    #[test]
    fn load_pq_pivots_reports_chunk_offset_column_mismatch() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let pivot_path = "/wrong_chunk_offset_columns_pivots.bin";

        {
            let mut writer = storage_provider.create_for_write(pivot_path).unwrap();
            let mut cumul_bytes = [0usize; 4];
            cumul_bytes[0] = METADATA_SIZE;

            writer.seek(SeekFrom::Start(cumul_bytes[0] as u64)).unwrap();

            let pivots = [0.0, 1.0, 2.0, 3.0];
            cumul_bytes[1] = cumul_bytes[0]
                + write_bin(
                    MatrixView::try_from(pivots.as_slice(), 2, 2).unwrap(),
                    &mut writer,
                )
                .unwrap();

            let centroid = [0.0, 0.0];
            cumul_bytes[2] = cumul_bytes[1]
                + write_bin(MatrixView::column_vector(centroid.as_slice()), &mut writer).unwrap();

            let chunk_offsets = [0_u32, 2_u32];
            cumul_bytes[3] = cumul_bytes[2]
                + write_bin(
                    MatrixView::try_from(chunk_offsets.as_slice(), 1, 2).unwrap(),
                    &mut writer,
                )
                .unwrap();

            let offsets: Vec<u64> = cumul_bytes.iter().map(|&offset| offset as u64).collect();
            write_bin_from(
                MatrixView::column_vector(offsets.as_slice()),
                &mut writer,
                0,
            )
            .unwrap();
        }

        let err = PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None)
            .load_pivots(&storage_provider)
            .unwrap_err();
        let message = err.to_string();

        assert!(message.contains("file has nc=2, but expecting nc=1"));
        assert!(!message.contains("expecting nr=0"));
    }

    #[test]
    fn load_pq_pivots_rejects_too_many_centers() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let pivot_path = "/too_many_centers_pivots.bin";

        write_test_pivots(
            &storage_provider,
            pivot_path,
            NUM_PQ_CENTROIDS + 1,
            1,
            None,
            &[0, 1],
        );

        assert!(
            PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None)
                .load_pivots(&storage_provider)
                .is_err()
        );
    }

    #[test]
    fn load_pivot_data_rejects_malformed_offset_table() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let pivot_path = "/malformed_offsets_pivots.bin";

        {
            let mut writer = storage_provider.create_for_write(pivot_path).unwrap();
            let offsets = [METADATA_SIZE as u64, 0, 0];
            write_bin(MatrixView::column_vector(offsets.as_slice()), &mut writer).unwrap();
        }

        assert!(
            PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None)
                .load_pivots(&storage_provider)
                .is_err()
        );
    }

    /// Write pivot data with a non-zero centroid, read it back, and verify that
    /// loading folds the centroid into the effective pivots.
    #[test]
    fn write_read_roundtrip_with_legacy_centroid() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let pivot_path = "/roundtrip_legacy_centroid_pivots.bin";

        let num_centers = 3;
        let dim = 4;
        let num_pq_chunks = 2;
        let pivots: Vec<f32> = (0..num_centers * dim).map(|i| i as f32).collect();
        let centroid: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let chunk_offsets = vec![0, 2, dim];

        let pq_storage = PQStorage::new(pivot_path, PQ_COMPRESSED_PATH, None);
        pq_storage
            .write_pivot_data(
                &pivots,
                Some(&centroid),
                &chunk_offsets,
                num_centers,
                dim,
                &storage_provider,
            )
            .unwrap();

        let table = pq_storage.load_pivots(&storage_provider).unwrap();
        let pivot_view = table.view_pivots();
        let loaded_pivots = pivot_view.as_slice();

        // Each pivot row should have the centroid added element-wise.
        for (idx, (&pivot, &orig)) in loaded_pivots.iter().zip(pivots.iter()).enumerate() {
            let d = idx % dim;
            let expected = orig + centroid[d];
            assert_eq!(
                pivot, expected,
                "pivot[{}]: expected {expected}, got {pivot}",
                idx
            );
        }
        assert_eq!(table.view_offsets().as_slice(), chunk_offsets);
        assert_eq!(table.nchunks(), num_pq_chunks);
    }

    #[test]
    fn gen_random_slice_test() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let file_name = "/gen_random_slice_test.bin";
        //npoints=2, dim=8
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

        let (sampled_vectors, slice_size, ndims) =
            gen_random_slice::<f32, VirtualStorageProvider<MemoryFS>>(
                file_name,
                1f64,
                &storage_provider,
                &mut crate::utils::create_rnd_in_tests(),
            )
            .unwrap();
        let mut start = 8;
        (0..sampled_vectors.len()).for_each(|i| {
            assert_eq!(sampled_vectors[i].to_le_bytes(), data[start..start + 4]);
            start += 4;
        });
        assert_eq!(sampled_vectors.len(), 16);
        assert_eq!(slice_size, 2);
        assert_eq!(ndims, 8);

        let (sampled_vectors, slice_size, ndims) =
            gen_random_slice::<f32, VirtualStorageProvider<MemoryFS>>(
                file_name,
                0f64,
                &storage_provider,
                &mut crate::utils::create_rnd_in_tests(),
            )
            .unwrap();
        assert_eq!(sampled_vectors.len(), 0);
        assert_eq!(slice_size, 0);
        assert_eq!(ndims, 8);

        storage_provider
            .delete(file_name)
            .expect("Failed to delete file");
    }
}
