/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Fast in-memory quant vector provider.
//!
//! This implementation stores all vectors in memory, has no synchronization on
//! the read path, and serializes writes to a given vector. The backing store
//! is a special buffer which aligns every vector data start point with a cache
//! line (64 bytes); this means that a vector of dimension D may actually be
//! size_of(element) + 63 bytes long in the worst case.

use std::sync::{Arc, Mutex};

use crate::storage::{StorageReadProvider, StorageWriteProvider};
use diskann::{
    ANNError, ANNResult,
    error::IntoANNResult,
    utils::{VectorRepr, object_pool::ObjectPool},
};
use diskann_quantization::CompressInto;
use diskann_vector::distance::Metric;

use super::common::{AlignedMemoryVectorStore, TestCallCount};
use crate::{
    common::IgnoreLockPoison,
    model::{
        distance::common::distance_table_pool,
        pq::{self, FixedChunkPQTable},
    },
    storage::{self, AsyncIndexMetadata, AsyncQuantLoadContext, LoadWith, SaveWith, bin},
    utils::{BridgeErr, PQPathNames},
};

/// This controls how many vectors share a write lock.
///
/// With the default value of 16, vector IDs 0-15 will share write lock 0,
/// 16-31 will share write lock 1, etc.
const WRITE_LOCK_GRANULARITY: usize = 16;

pub struct FastMemoryQuantVectorProviderAsync {
    quant_vectors: AlignedMemoryVectorStore<u8>,
    max_vectors: usize,

    // We keep only write locks as reads are unsynchronized. Since there are
    // only writers, we use Mutex here. Note that sync::Mutex is ok here
    // because the Mutex is never held across an await.
    write_locks: Vec<Mutex<()>>,

    pub pq_chunk_table: Arc<FixedChunkPQTable>,

    /// The metric used.
    metric: Metric,

    pub num_get_calls: TestCallCount,

    vec_pool: Arc<ObjectPool<Vec<f32>>>,
}

type DistanceComputer = pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>;
type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;

impl FastMemoryQuantVectorProviderAsync {
    pub fn new(dist_metric: Metric, max_vectors: usize, pq_chunk_table: FixedChunkPQTable) -> Self {
        let dim = pq_chunk_table.get_num_chunks();
        let quant_vectors = AlignedMemoryVectorStore::with_capacity(max_vectors, dim);
        let write_locks = (0..max_vectors.div_ceil(WRITE_LOCK_GRANULARITY))
            .map(|_| Mutex::new(()))
            .collect::<Vec<_>>();
        let vec_pool = Arc::new(distance_table_pool(&pq_chunk_table));

        Self {
            max_vectors,
            quant_vectors,
            write_locks,
            pq_chunk_table: Arc::new(pq_chunk_table),
            metric: dist_metric,
            num_get_calls: TestCallCount::default(),
            vec_pool,
        }
    }

    /// Return the metric associated with this provider.
    pub(crate) fn metric(&self) -> Metric {
        self.metric
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
        T: Copy + VectorRepr,
    {
        QueryComputer::new(
            self.pq_chunk_table.clone(),
            self.metric,
            &T::as_f32(query).into_ann_result()?,
            Some(self.vec_pool.clone()),
        )
    }

    /// Create a distance computer for the underlying schema.
    pub fn distance_computer(
        &self,
    ) -> Result<DistanceComputer, pq::distance::dynamic::DistanceComputerConstructionError> {
        DistanceComputer::new(self.pq_chunk_table.clone(), self.metric)
    }

    /// Return an "immutable" slice over the PQ data at index `i`.
    ///
    /// # Safety
    ///
    /// This function is unsafe because this provider does not give a guarantee on
    /// exclusivity of the returned data. That is, the returned data *can* be modified
    /// concurrently using unsynchronized accesses.
    ///
    /// It's the caller's responsibility to either:
    ///
    /// 1. Use this method in a way that ensures mutual exclusion with mutable references to
    ///    the same ID.
    ///
    /// 2. Be okay with racey data.
    #[inline(always)]
    pub(crate) unsafe fn get_vector_sync(&self, i: usize) -> &[u8] {
        self.num_get_calls.increment();
        // SAFETY: The function called here has the same pre and post-conditions as the caller.
        unsafe { self.quant_vectors.get_slice(i) }
    }

    /// Compress and store the data in `v` into the internal data at position `i`.
    ///
    /// Errors if:
    ///
    /// * `i > self.total()`: `i` must be inbounds.
    /// * `v.dim() != self.full_dim()`: The slice must have the proper length.
    /// * PQ compression encounters an error (such as the presence of `NaN`s).
    ///
    /// # Safety
    ///
    /// This function guarantees mutual exclusion of **writers** to the underlying data,
    /// but does not guarantee the mutual exclusion of aliased readers to the same data.
    ///
    /// It is the caller's responsibility to either:
    ///
    /// 1. Use this method in a way that ensures mutual exclusion with mutable references to
    ///    the same ID.
    ///
    /// 2. Be okay with racey data.
    pub(crate) unsafe fn set_vector_sync<T>(&self, i: usize, v: &[T]) -> ANNResult<()>
    where
        T: VectorRepr,
    {
        if i >= self.total() {
            return Err(ANNError::log_index_error(
                "Vector id is out of boundary in the dataset.",
            ));
        }

        let vf32: &[f32] = &T::as_f32(v).into_ann_result()?;

        if vf32.len() != self.full_dim() {
            return Err(ANNError::log_index_error(
                "Vector f32 dimension is not equal to the expected dimension.",
            ));
        }

        // Acquire the write lock for mutual write exclusion.
        let lock_id = i / WRITE_LOCK_GRANULARITY;
        let _guard = self.write_locks[lock_id].lock_or_panic();

        // SAFETY: `get_mut_slice` guarantees it is safe to access the memory,
        // and but it may be a torn read. As we are trading off synchronization
        // for speed, this is okay.
        let slice = unsafe { self.quant_vectors.get_mut_slice(i) };
        Ok(self
            .pq_chunk_table
            .compress_into(vf32, slice)
            .bridge_err()?)
    }

    /// Store the compressed PQ vector directly at position `i`.
    ///
    /// Errors if:
    ///
    /// * `i >= self.total()`: `i` must be inbounds.
    /// * `v.len() != self.pq_chunks()`: `v` must have the right length.
    ///
    /// # Safety
    ///
    /// This function guarantees mutual exclusion of **writers** to the underlying data,
    /// but does not guarantee the mutual exclusion of aliased readers to the same data.
    ///
    /// It is the caller's responsibility to either:
    ///
    /// 1. Use this method in a way that ensures mutual exclusion with mutable references to
    ///    the same ID.
    ///
    /// 2. Be okay with racey data.
    pub(crate) unsafe fn set_quant_vector(&self, i: usize, v: &[u8]) -> ANNResult<()> {
        if i >= self.total() {
            return Err(ANNError::log_index_error(
                "Vector id is out of boundary in the dataset.",
            ));
        }
        if v.len() != self.pq_chunks() {
            return Err(ANNError::log_index_error(
                "Vector dimension is not equal to the expected dimension.",
            ));
        }

        let lock_id = i / WRITE_LOCK_GRANULARITY;
        let _guard = self.write_locks[lock_id].lock_or_panic();
        // SAFETY: `get_mut_slice` guarantees it is safe to access the memory,
        // and but it may be a torn read. As we are trading off synchronization
        // for speed, this is okay.
        unsafe { self.quant_vectors.get_mut_slice(i) }.copy_from_slice(v);
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
impl SaveWith<AsyncIndexMetadata> for FastMemoryQuantVectorProviderAsync {
    type Ok = usize;
    type Error = ANNError;

    /// Save the quant vector provider using `prefix` as a prefix to [`PQPathNames`].
    ///
    /// This will generate two files. See
    /// [`FastMemoryQuantVectorProviderAsync::save_direct`].
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

impl LoadWith<AsyncQuantLoadContext> for FastMemoryQuantVectorProviderAsync {
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
impl storage::bin::SetData for FastMemoryQuantVectorProviderAsync {
    type Item = u8;

    fn set_data(&mut self, i: usize, element: &[Self::Item]) -> ANNResult<()> {
        // SAFETY: No race can happen because we have a mutable reference to `self`.
        unsafe { self.set_quant_vector(i, element) }
    }
}

/// Hook into [`storage::bin::save_to_bin`] by implementing [`storage::bin::GetData`].
impl storage::bin::GetData for FastMemoryQuantVectorProviderAsync {
    type Element = u8;
    type Item<'a> = &'a [u8];

    fn get_data(&self, i: usize) -> ANNResult<Self::Item<'_>> {
        // SAFETY: We aren't full protected against races on the underlying data, but at
        // least `&self` will keep the data alive.
        Ok(unsafe { self.get_vector_sync(i) })
    }

    /// Return the total number of points, including frozen points.
    fn total(&self) -> usize {
        FastMemoryQuantVectorProviderAsync::total(self)
    }

    fn dim(&self) -> usize {
        self.pq_chunks()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use crate::storage::VirtualStorageProvider;
    use diskann::{ANNErrorKind, utils::ONE};
    use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
    use vfs::MemoryFS;

    use super::*;

    #[tokio::test]
    async fn common_errors() {
        let dim = 5;
        let centroid = vec![0.0; dim];
        let offsets = vec![0, dim];
        let full_pivot_data = vec![0.0; 256 * dim];

        let pq_chunk_table = FixedChunkPQTable::new(
            dim,
            full_pivot_data.into(),
            centroid.into(),
            offsets.into(),
            None,
        )
        .unwrap();
        let provider = FastMemoryQuantVectorProviderAsync::new(Metric::L2, 10, pq_chunk_table);

        // try to set an out of bounds vector
        // SAFETY: We have exclusive ownership of `provider`.
        let result = unsafe { provider.set_quant_vector(20, &[]) }.unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::IndexError);

        // SAFETY: We have exclusive ownership of `provider`.
        let result = unsafe { provider.set_vector_sync::<f32>(20, &[]) }.unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::IndexError);

        // try to set a vector with the wrong dimension
        // SAFETY: We have exclusive ownership of `provider`.
        let result = unsafe { provider.set_quant_vector(0, &[]) }.unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::IndexError);
    }

    fn create_test_provider() -> FastMemoryQuantVectorProviderAsync {
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

        let provider = FastMemoryQuantVectorProviderAsync::new(Metric::L2, num_points, table);

        assert_eq!(provider.total(), num_points);
        assert_eq!(provider.full_dim(), dim);
        // SAFETY: We have exclusive access over the fast provider and are careful to
        // not hold onto obtained slices.
        unsafe {
            // Set Vector.
            provider.set_vector_sync(0, &[-1.5, -1.5]).unwrap();
            provider.set_vector_sync(1, &[-0.5, -0.5]).unwrap();
            provider.set_vector_sync(2, &[0.5, 0.5]).unwrap();
            provider.set_vector_sync(3, &[1.5, 1.5]).unwrap();
            provider.set_vector_sync(4, &[2.5, 2.5]).unwrap();
        }

        provider
    }

    #[test]
    fn test_provider() {
        let provider = create_test_provider();

        // SAFETY: We have exclusive access over the fast provider and are careful to
        // not hold onto obtained slices.
        unsafe {
            // Get Vector.
            assert_eq!(provider.get_vector_sync(0), &[0]);
            assert_eq!(provider.get_vector_sync(1), &[0]);
            assert_eq!(provider.get_vector_sync(2), &[0]);
            assert_eq!(provider.get_vector_sync(3), &[1]);
            assert_eq!(provider.get_vector_sync(4), &[2]);

            // Error checking.
            assert!(provider.set_vector_sync(5, &[0.0, 0.0]).is_err());
            assert!(provider.set_vector_sync(2, &[0.0]).is_err());

            // Query Computer.
            let c = provider.query_computer(&[-0.5, -0.5]).unwrap();
            let expected: f32 = 1.5 * 1.5 * 2.0;
            assert_eq!(
                c.evaluate_similarity(&provider.get_vector_sync(3)),
                expected
            );

            // Distance Computer.
            let d = provider.distance_computer().unwrap();
            assert_eq!(
                d.evaluate_similarity(&provider.get_vector_sync(0), &provider.get_vector_sync(3)),
                2.0
            );

            let slice: &[f32] = &[-0.5, -0.5];
            assert_eq!(
                d.evaluate_similarity(slice, &provider.get_vector_sync(3)),
                expected,
            );
        }
    }

    fn check_providers_equal(
        original: &FastMemoryQuantVectorProviderAsync,
        reloaded: &FastMemoryQuantVectorProviderAsync,
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
            unsafe {
                let a = original.get_vector_sync(i);
                let b = reloaded.get_vector_sync(i);
                assert_eq!(
                    a, b,
                    "reloaded vector does not match original at position {}",
                    i
                );
            }
        }
    }

    // Test Saving and Loading.
    #[test]
    fn test_direct_save_load() {
        type Provider = FastMemoryQuantVectorProviderAsync;

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
        type Provider = FastMemoryQuantVectorProviderAsync;

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
            num_frozen_points: ONE, // Don't care for this vector store.
            metric: provider.metric,
            prefetch_lookahead: None,
            is_disk_index: false,
            prefetch_cache_line_level: None,
        };

        let reloaded = Provider::load_with(&storage, &ctx).await.unwrap();

        check_providers_equal(&provider, &reloaded);
    }
}
