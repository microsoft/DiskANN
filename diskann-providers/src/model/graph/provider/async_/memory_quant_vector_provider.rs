/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Basic in-memory quant vector provider.
//!
//! This implementation stores all vectors in memory. It uses ArcSwap to synchronize reads and writes, and the vectors are not stored
//! cache aligned. The backing store is `Vec<ArcSwap<Vec<Data::VectorDataType>>>`.

use std::sync::Arc;

use crate::storage::{StorageReadProvider, StorageWriteProvider};
use arc_swap::{ArcSwap, Guard};
#[cfg(test)]
use diskann::utils::VectorRepr;
use diskann::{ANNError, ANNResult, utils::object_pool::ObjectPool};
#[cfg(test)]
use diskann_quantization::CompressInto;
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};

use super::{VectorGuard, common::TestCallCount};
#[cfg(test)]
use crate::utils::BridgeErr;
use crate::{
    model::{
        distance::common::distance_table_pool,
        pq::{self, FixedChunkPQTable},
    },
    storage::{self, AsyncIndexMetadata, AsyncQuantLoadContext, LoadWith, SaveWith, bin},
    utils::PQPathNames,
};

pub struct MemoryQuantVectorProviderAsync {
    max_vectors: usize,
    quant_vectors: Vec<ArcSwap<Vec<u8>>>,
    pq_chunk_table: Arc<FixedChunkPQTable>,

    /// The metric used.
    metric: Metric,

    num_get_calls: TestCallCount,
    vec_pool: Arc<ObjectPool<Vec<f32>>>,
}

type DistanceComputer = pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>;
type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;

impl MemoryQuantVectorProviderAsync {
    pub fn new(dist_metric: Metric, max_vectors: usize, pq_chunk_table: FixedChunkPQTable) -> Self {
        let quant_vectors = (0..max_vectors)
            .map(|_| {
                ArcSwap::new(Arc::new(vec![
                    u8::default();
                    pq_chunk_table.get_num_chunks()
                ]))
            })
            .collect();
        let vec_pool = Arc::new(distance_table_pool(&pq_chunk_table));

        Self {
            max_vectors,
            quant_vectors,
            pq_chunk_table: Arc::new(pq_chunk_table),
            metric: dist_metric,
            num_get_calls: TestCallCount::default(),
            vec_pool,
        }
    }

    /// Return the total number of points (including frozen points) included in `self.
    #[inline(always)]
    pub fn total(&self) -> usize {
        self.max_vectors
    }

    /// Return the dimension of the full-precision data associated with this provider.
    pub fn full_dim(&self) -> usize {
        self.pq_chunk_table.get_dim()
    }

    /// Return the number of PQ chunks in the underlying PQ schema.
    pub fn pq_chunks(&self) -> usize {
        self.pq_chunk_table.get_num_chunks()
    }

    /// Create a query computer for the provided query vector.
    pub fn query_computer<T>(&self, query: &[T]) -> ANNResult<QueryComputer>
    where
        T: Copy + Into<f32>,
    {
        QueryComputer::new(
            self.pq_chunk_table.clone(),
            self.metric,
            query,
            Some(self.vec_pool.clone()),
        )
    }

    /// Create a distance computer for the underlying schema.
    pub fn distance_computer(
        &self,
    ) -> Result<DistanceComputer, pq::distance::dynamic::DistanceComputerConstructionError> {
        DistanceComputer::new(self.pq_chunk_table.clone(), self.metric)
    }

    /// Return an immutable, reference counted guard over the data as position `i`.
    ///
    /// Errors if `i >= self.total()`.
    pub(crate) fn get_vector_sync(&self, i: usize) -> ANNResult<Guard<Arc<Vec<u8>>>> {
        self.num_get_calls.increment();
        match self.quant_vectors.get(i) {
            Some(vector) => Ok(vector.load()),
            None => Err(ANNError::log_index_error(
                "Vector id is out of boundary in the dataset.",
            )),
        }
    }

    /// Compress and store the data in `v` into the internal data at position `i`.
    ///
    /// Errors if:
    ///
    /// * `i > self.total()`: `i` must be inbounds.
    /// * `v.dim() != self.full_dim()`: The slice must have the proper length.
    /// * PQ compression encounters an error (such as the presence of `NaN`s).
    #[cfg(test)]
    pub(crate) fn set_vector_sync<T>(&self, i: usize, v: &[T]) -> ANNResult<()>
    where
        T: Copy + VectorRepr,
    {
        let v_f32 = T::as_f32(v).map_err(|x| x.into())?;
        if v_f32.len() != self.full_dim() {
            return Err(ANNError::log_index_error(
                "Vector f32 dimension is not equal to the expected dimension.",
            ));
        }

        let mut buffer = vec![0u8; self.pq_chunks()];
        self.pq_chunk_table
            .compress_into(v_f32.as_ref(), &mut *buffer)
            .bridge_err()?;
        self.set_quant_vector(i, Arc::new(buffer))
    }

    /// Store the compressed PQ vector directly at position `i`.
    ///
    /// Errors if:
    ///
    /// * `i >= self.total()`: `i` must be inbounds.
    /// * `v.len() != self.pq_chunks()`: `v` must have the right length.
    pub(crate) fn set_quant_vector(&self, i: usize, v: Arc<Vec<u8>>) -> ANNResult<()> {
        let slot = match self.quant_vectors.get(i) {
            Some(slot) => slot,
            None => {
                return Err(ANNError::log_index_error(
                    "Vector id is out of boundary in the dataset.",
                ));
            }
        };
        slot.swap(v);
        Ok(())
    }

    /// Load `self` from a pivots file and data file.
    ///
    /// The pivots file follows the format in [`storage::PQStorage::load_pq_pivots_bin`] and
    /// the compressed code is saved in a canonical `.bin` format.
    ///
    /// See also: [`storage::bin::load_from_bin`].
    ///
    /// Because the number of start points and distance metric are not saved as part of the
    /// `.bin` file format, they must be provided externally.
    pub fn load_direct<P>(provider: &P, pivots: &str, data: &str, metric: Metric) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        bin::load_from_bin(provider, data, |num_points, pq_bytes| {
            // The number of PQ bytes is extracted from the data file.
            // We can use that information to load the pivots, then finish the rest
            // of initialization.
            let pq_storage = storage::PQStorage::new(pivots, data, None);
            let table = pq_storage.load_pq_pivots_bin(pivots, pq_bytes, provider)?;
            Ok(Self::new(metric, num_points, table))
        })
    }

    /// Save the pivot table in binary form to `path`.
    ///
    /// See also: [`storage::PQStorage::write_pivot_data`].
    pub fn save_pivots<P>(&self, provider: &P, path: &str) -> ANNResult<()>
    where
        P: StorageWriteProvider,
    {
        let pq_storage = storage::PQStorage::new(path, "", None);
        let table = &self.pq_chunk_table;
        pq_storage.write_pivot_data(
            table.get_pq_table(),
            table.get_centroids(),
            table.get_chunk_offsets(),
            table.get_num_centers(),
            table.get_dim(),
            provider,
        )
    }

    /// Save `self` to disk with the pivot table stored at path `pivots` and the compressed
    /// data store in `.bin` form to file path `data`.
    ///
    /// See also:
    /// * [`storage::PQStorage::write_pivot_data`]
    /// * [`storage::bin::save_to_bin`]
    pub fn save_direct<P>(&self, provider: &P, pivots: &str, data: &str) -> ANNResult<usize>
    where
        P: StorageWriteProvider,
    {
        self.save_pivots(provider, pivots)?;
        storage::bin::save_to_bin(self, provider, data)
    }
}

/// This is an adaptor for compatibility with the async index serialization.
impl SaveWith<AsyncIndexMetadata> for MemoryQuantVectorProviderAsync {
    type Ok = usize;
    type Error = ANNError;

    /// Save the quant vector provider using `prefix` as a prefix to [`PQPathNames`].
    ///
    /// This will generate two files. See
    /// [`MemoryQuantVectorProviderAsync::save_direct`].
    ///
    /// Returns the number of bytes written while saving the compressed data.
    async fn save_with<P>(&self, provider: &P, metadata: &AsyncIndexMetadata) -> ANNResult<usize>
    where
        P: StorageWriteProvider,
    {
        let names = PQPathNames::new(metadata.prefix());
        self.save_direct(provider, &names.pivots, &names.compressed_data)
    }
}

impl LoadWith<AsyncQuantLoadContext> for MemoryQuantVectorProviderAsync {
    type Error = ANNError;

    /// Load the quant vector provider using the `prefix` in `ctx~ as a prefix to
    /// [`PQPathNames`] along with the number of PQ bytes.
    async fn load_with<P>(provider: &P, ctx: &AsyncQuantLoadContext) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        let names = match ctx.is_disk_index {
            true => PQPathNames::for_disk_index(ctx.metadata.prefix()),
            false => PQPathNames::new(ctx.metadata.prefix()),
        };
        Self::load_direct(provider, &names.pivots, &names.compressed_data, ctx.metric)
    }
}

/// Hook into [`storage::bin::load_from_bin`] by implementing [`storage::bin::SetData`].
impl storage::bin::SetData for MemoryQuantVectorProviderAsync {
    type Item = u8;

    fn set_data(&mut self, i: usize, element: &[Self::Item]) -> ANNResult<()> {
        self.set_quant_vector(i, Arc::new(element.to_owned()))
    }
}

/// Hook into [`storage::bin::save_to_bin`] by implementing [`storage::bin::GetData`].
impl storage::bin::GetData for MemoryQuantVectorProviderAsync {
    type Element = u8;
    type Item<'a> = VectorGuard<u8>;

    fn get_data(&self, i: usize) -> ANNResult<Self::Item<'_>> {
        Ok(VectorGuard::from_guard(self.get_vector_sync(i)?))
    }

    /// Return the total number of points, including frozen points.
    fn total(&self) -> usize {
        MemoryQuantVectorProviderAsync::total(self)
    }

    fn dim(&self) -> usize {
        self.pq_chunks()
    }
}

/// Overload `DistanceFunction` for `Guard<Arc<Vec<u8>>>` by dereferencing the
/// guard to a slice.
impl DistanceFunction<&[f32], &Guard<Arc<Vec<u8>>>, f32> for DistanceComputer {
    #[inline(always)]
    fn evaluate_similarity(&self, left: &[f32], right: &Guard<Arc<Vec<u8>>>) -> f32 {
        let right: &[u8] = right;
        self.evaluate_similarity(left, right)
    }
}

/// Overload `DistanceFunction` for `Guard<Arc<Vec<u8>>>` by dereferencing the
/// guard to a slice.
impl DistanceFunction<&Guard<Arc<Vec<u8>>>, &Guard<Arc<Vec<u8>>>, f32> for DistanceComputer {
    #[inline(always)]
    fn evaluate_similarity(&self, left: &Guard<Arc<Vec<u8>>>, right: &Guard<Arc<Vec<u8>>>) -> f32 {
        let left: &[u8] = left;
        let right: &[u8] = right;
        self.evaluate_similarity(left, right)
    }
}

/// Overload `PreprocessedDistanceFunction` for `Guard<Arc<Vec<u8>>>` by dereferencing the
/// guard to a slice.
impl PreprocessedDistanceFunction<&Guard<Arc<Vec<u8>>>, f32> for QueryComputer {
    #[inline(always)]
    fn evaluate_similarity(&self, changing: &Guard<Arc<Vec<u8>>>) -> f32 {
        let changing: &[u8] = changing;
        self.evaluate_similarity(changing)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::storage::VirtualStorageProvider;
    use vfs::MemoryFS;

    use super::*;

    fn create_test_provider() -> MemoryQuantVectorProviderAsync {
        let num_points = 5;
        let dim = 2;

        // We can create a really simple, 1 chunk PQ table with known entries to allow
        // us to easily verify results.
        let table = FixedChunkPQTable::new(
            dim,
            Box::new([0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
            Box::new([0.0, 0.0]),
            Box::new([0, dim]),
            None,
        )
        .unwrap();

        let provider = MemoryQuantVectorProviderAsync::new(Metric::L2, num_points, table);

        // Set Vector.
        provider.set_vector_sync(0, &[-1.5, -1.5]).unwrap();
        provider.set_vector_sync(1, &[-0.5, -0.5]).unwrap();
        provider.set_vector_sync(2, &[0.5, 0.5]).unwrap();
        provider.set_vector_sync(3, &[1.5, 1.5]).unwrap();
        provider.set_vector_sync(4, &[2.5, 2.5]).unwrap();

        provider
    }

    #[test]
    fn test_provider() {
        let provider = create_test_provider();

        // Get Vector.
        assert_eq!(&***provider.get_vector_sync(0).unwrap(), &[0]);
        assert_eq!(&***provider.get_vector_sync(1).unwrap(), &[0]);
        assert_eq!(&***provider.get_vector_sync(2).unwrap(), &[0]);
        assert_eq!(&***provider.get_vector_sync(3).unwrap(), &[1]);
        assert_eq!(&***provider.get_vector_sync(4).unwrap(), &[2]);

        // Error checking.
        assert!(provider.get_vector_sync(5).is_err());
        assert!(provider.set_vector_sync(5, &[0.0, 0.0]).is_err());
        assert!(provider.set_vector_sync(2, &[0.0]).is_err());

        // Query Computer.
        let c = provider.query_computer(&[-0.5, -0.5]).unwrap();
        let expected: f32 = 1.5 * 1.5 * 2.0;
        assert_eq!(
            c.evaluate_similarity(&provider.get_vector_sync(3).unwrap()),
            expected
        );

        // Distance Computer.
        let d = provider.distance_computer().unwrap();
        assert_eq!(
            d.evaluate_similarity(
                &provider.get_vector_sync(0).unwrap(),
                &provider.get_vector_sync(3).unwrap()
            ),
            2.0
        );

        let slice: &[f32] = &[-0.5, -0.5];
        assert_eq!(
            d.evaluate_similarity(slice, &provider.get_vector_sync(3).unwrap()),
            expected,
        );
    }

    fn check_providers_equal(
        original: &MemoryQuantVectorProviderAsync,
        reloaded: &MemoryQuantVectorProviderAsync,
    ) {
        assert_eq!(original.total(), reloaded.total());
        assert_eq!(original.full_dim(), reloaded.full_dim());
        assert_eq!(original.pq_chunks(), reloaded.pq_chunks());

        // Make sure the underlying tables are the same.
        assert_eq!(
            original.pq_chunk_table.get_pq_table(),
            reloaded.pq_chunk_table.get_pq_table()
        );
        assert_eq!(
            original.pq_chunk_table.get_chunk_offsets(),
            reloaded.pq_chunk_table.get_chunk_offsets()
        );
        assert_eq!(
            original.pq_chunk_table.get_dim(),
            reloaded.pq_chunk_table.get_dim()
        );
        assert_eq!(
            original.pq_chunk_table.get_num_centers(),
            reloaded.pq_chunk_table.get_num_centers()
        );

        for i in 0..original.total() {
            // SAFETY: We are the only users of both these providers. As long as we do not
            // keep the extracted slices around, we cannot alias.
            let a = &***original.get_vector_sync(i).unwrap();
            let b = &***reloaded.get_vector_sync(i).unwrap();
            assert_eq!(
                a, b,
                "reloaded vector does not match original at position {}",
                i
            );
        }
    }

    // Test Saving and Loading.
    #[test]
    fn test_direct_save_load() {
        type Provider = MemoryQuantVectorProviderAsync;

        let storage = VirtualStorageProvider::new(MemoryFS::new());
        let provider = create_test_provider();

        // Save to disk.
        let pivots = "/pivots.bin";
        let data = "/data.bin";
        assert!(!storage.exists(pivots));
        assert!(!storage.exists(data));

        provider.save_direct(&storage, pivots, data).unwrap();
        assert!(storage.exists(pivots), "expected a new file to be created");
        assert!(storage.exists(data), "expected a new file to be created");

        // Load from disk.
        let reloaded = Provider::load_direct(&storage, pivots, data, provider.metric).unwrap();

        check_providers_equal(&provider, &reloaded);
    }

    // Test Saving and Loading.
    #[tokio::test(flavor = "current_thread")]
    async fn test_async_save_load() {
        type Provider = MemoryQuantVectorProviderAsync;

        let storage = VirtualStorageProvider::new(MemoryFS::new());
        let provider = create_test_provider();

        let prefix = "/data.bin";
        let pq_names = PQPathNames::new(prefix);
        assert!(!storage.exists(&pq_names.pivots));
        assert!(!storage.exists(&pq_names.compressed_data));

        // Save to disk.
        let async_prefix = AsyncIndexMetadata::new(prefix.to_owned());
        provider.save_with(&storage, &async_prefix).await.unwrap();

        assert!(
            storage.exists(&pq_names.pivots),
            "expected a new file to be created"
        );
        assert!(
            storage.exists(&pq_names.compressed_data),
            "expected a new file to be created"
        );

        // Test that reloading finds the correct file paths.
        let ctx = AsyncQuantLoadContext {
            metadata: async_prefix,
            num_frozen_points: NonZeroUsize::new(5).unwrap(), // Don't care for this vector store.
            metric: provider.metric,
            prefetch_lookahead: None,
            is_disk_index: false,
            prefetch_cache_line_level: None,
        };

        let reloaded = Provider::load_with(&storage, &ctx).await.unwrap();

        check_providers_equal(&provider, &reloaded);
    }
}
