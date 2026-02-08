/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Fast in-memory vector provider.
//!
//! This implementation stores all vectors in memory, has no synchronization on
//! the read path, and serializes writes to a given vector. The backing store
//! is a special buffer which aligns every vector data start point with a cache
//! line (64 bytes); this means that a vector of dimension D may actually be
//! size_of(element) + 63 bytes long in the worst case.

use std::sync::Mutex;

use crate::storage::{StorageReadProvider, StorageWriteProvider};
use diskann::{ANNError, ANNResult, utils::{IntoUsize, VectorRepr}};
use diskann_vector::distance::Metric;

use super::common::{AlignedMemoryVectorStore, PrefetchCacheLineLevel, TestCallCount};
use crate::{
    common::IgnoreLockPoison,
    model::graph::{provider::async_::common::VectorStore, traits::GraphDataType},
    storage::{self, AsyncIndexMetadata, AsyncQuantLoadContext, LoadWith, SaveWith},
};

/// This controls how many vectors share a write lock.
///
/// With the default value of 16, vector IDs 0-15 will share write lock 0,
/// 16-31 will share write lock 1, etc.
const WRITE_LOCK_GRANULARITY: usize = 16;

/// The default prefetch lookahead to use if not configured externally.
///
/// This is an empirically determined amount that attempts to provide the most benefit across
/// the widest range of datasets.
const PREFETCH_DEFAULT: usize = 8;

pub struct FastMemoryVectorProviderAsync<Data: GraphDataType> {
    dim: usize,
    max_vectors: usize,
    vectors: AlignedMemoryVectorStore<Data::VectorDataType>,

    // We keep only write locks as reads are unsynchronized. Since there are
    // only writers, we use Mutex here. Note that sync::Mutex is ok here
    // because the Mutex is never held across an await.
    write_locks: Vec<Mutex<()>>,

    // The distance object used to compare two vector representations.
    distance: <Data::VectorDataType as VectorRepr>::Distance,

    prefetch_cache_line_level: PrefetchCacheLineLevel,

    /// Prefetching for full-precision bulk operations.
    prefetch_lookahead: usize,

    num_get_calls: TestCallCount,
}

impl<Data: GraphDataType> FastMemoryVectorProviderAsync<Data> {
    pub fn new(
        max_vectors: usize,
        dim: usize,
        metric: Metric,
        prefetch_cache_line_level: Option<PrefetchCacheLineLevel>,
        prefetch_lookahead: Option<usize>,
    ) -> Self {
        let vectors = AlignedMemoryVectorStore::with_capacity(max_vectors, dim);

        let write_locks = (0..max_vectors.div_ceil(WRITE_LOCK_GRANULARITY))
            .map(|_| Mutex::new(()))
            .collect::<Vec<_>>();

        Self {
            dim,
            max_vectors,
            vectors,
            write_locks,
            distance: Data::VectorDataType::distance(metric, Some(dim)),
            num_get_calls: TestCallCount::default(),
            prefetch_cache_line_level: prefetch_cache_line_level.unwrap_or_default(),
            prefetch_lookahead: prefetch_lookahead.unwrap_or(PREFETCH_DEFAULT),
        }
    }

    /// Return the total number of points (including frozen points) included in `self.
    #[inline(always)]
    pub fn total(&self) -> usize {
        self.max_vectors
    }

    /// Return the dimension of the vectors in this memory store.
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return a [`diskann_vector::DistanceFunction`] capable of computing distances on elements
    /// yielded by this provider.
    pub fn distance(&self) -> &<Data::VectorDataType as VectorRepr>::Distance {
        &self.distance
    }

    /// Return an "immutable" slice over the data at index `i`.
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
    pub unsafe fn get_vector_sync(&self, i: usize) -> &[Data::VectorDataType] {
        self.num_get_calls.increment();
        // SAFETY: The caller must ensure that `i < self.total()` and that there is no
        // concurrent mutable access to the vector at index `i`.
        unsafe { self.vectors.get_slice(i) }
    }

    /// Store the data in `v` into the internal data at position `i`.
    ///
    /// Errors if:
    ///
    /// * `i > self.total()`: `i` must be inbounds.
    /// * `v.dim() != self.dim()`: The slice must have the proper length.
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
    #[inline(always)]
    pub unsafe fn set_vector_sync(
        &self,
        i: usize,
        v: &[Data::VectorDataType],
    ) -> ANNResult<()> {
        if v.len() != self.dim {
            return Err(ANNError::log_index_error(
                "Vector dimension is not equal to the expected dimension.",
            ));
        }
        if i >= self.total() {
            return Err(ANNError::log_index_error(
                "Vector id is out of boundary in the dataset.",
            ));
        }

        let lock_id = i / WRITE_LOCK_GRANULARITY;
        let _guard = self.write_locks[lock_id].lock_or_panic();
        // SAFETY: `get_mut_slice` guarantees it is safe to access the memory,
        // and but it may be a torn read. As we are trading off synchronization
        // for speed, this is okay.
        unsafe {
            self.vectors.get_mut_slice(i).copy_from_slice(v);
        }

        Ok(())
    }

    /// Load `self` directly from a `.bin` file at `path`.
    ///
    /// Because the number of start points are not saved as part of the `.bin` file format,
    /// this must be provided externally.
    ///
    /// See also: [`storage::bin::load_from_bin`].
    pub fn load_from_bin<P>(
        provider: &P,
        path: &str,
        metric: Metric,
        prefetch_cache_line_level: Option<PrefetchCacheLineLevel>,
        prefetch_lookahead: Option<usize>,
    ) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        storage::bin::load_from_bin(provider, path, |num_points, dim| -> ANNResult<Self> {
            Ok(Self::new(
                num_points,
                dim,
                metric,
                prefetch_cache_line_level,
                prefetch_lookahead,
            ))
        })
    }

    /// Save `self` directly to a `.bin` file at `path`.
    ///
    /// See also: [`storage::bin::save_to_bin`].
    pub fn save_to_bin<P>(&self, provider: &P, path: &str) -> ANNResult<usize>
    where
        P: StorageWriteProvider,
    {
        storage::bin::save_to_bin(self, provider, path)
    }

    /// Prefetch the first few cache lines of the data for vector `i` with a specific cache line level.
    /// the cache line level determines how many cache lines to prefetch
    #[inline(always)]
    pub fn prefetch_hint(&self, i: usize) {
        // SAFETY: Racing on the underlying data is okay because we are dispatching to
        // an architectural primitive for prefetching that doesn't care about the data
        // itself, just its address.
        let vector = unsafe { self.vectors.get_slice(i) };
        match self.prefetch_cache_line_level {
            PrefetchCacheLineLevel::CacheLine4 => {
                diskann_vector::prefetch_hint_max::<4, _>(vector);
            }
            PrefetchCacheLineLevel::CacheLine8 => {
                diskann_vector::prefetch_hint_max::<8, _>(vector);
            }
            PrefetchCacheLineLevel::CacheLine16 => {
                diskann_vector::prefetch_hint_max::<16, _>(vector);
            }
            PrefetchCacheLineLevel::All => {
                diskann_vector::prefetch_hint_all(vector);
            }
        }
    }

    /// Returns the prefetch lookahead for full-precision bulk operations.
    #[inline(always)]
    pub fn prefetch_lookahead(&self) -> usize {
        self.prefetch_lookahead
    }
}

impl<Data> VectorStore for FastMemoryVectorProviderAsync<Data>
where
    Data: GraphDataType,
{
    fn total(&self) -> usize {
        self.total()
    }

    fn count_for_get_vector(&self) -> usize {
        self.num_get_calls.get()
    }
}

impl<T> super::common::SetElementHelper<T> for FastMemoryVectorProviderAsync<crate::model::graph::traits::AdHoc<T>>
where
    T: VectorRepr,
{
    fn set_element(&self, id: &u32, element: &[T]) -> Result<(), ANNError> {
        unsafe { self.set_vector_sync(id.into_usize(), element) }
    }
}

/// This is an adaptor for compatibility with the async index serialization.
impl<Data> SaveWith<AsyncIndexMetadata> for FastMemoryVectorProviderAsync<Data>
where
    Data: GraphDataType,
{
    type Ok = usize;
    type Error = ANNError;

    /// Save the dataset to a file beginning with the prefix and ending in `.data`.
    ///
    /// The format of the serialized will be the traditional `.bin` format.
    ///
    /// Returns the number of bytes written.
    async fn save_with<P>(&self, provider: &P, prefix: &AsyncIndexMetadata) -> ANNResult<usize>
    where
        P: StorageWriteProvider,
    {
        self.save_to_bin(provider, &prefix.data_path())
    }
}

impl<Data> LoadWith<AsyncQuantLoadContext> for FastMemoryVectorProviderAsync<Data>
where
    Data: GraphDataType,
{
    type Error = ANNError;

    /// Load the dataset to a file beginning with the prefix and ending in `.data`.
    ///
    /// The format of the serialized should be in the traditional `.bin` format.
    async fn load_with<P>(provider: &P, ctx: &AsyncQuantLoadContext) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        Self::load_from_bin(
            provider,
            &ctx.metadata.data_path(),
            ctx.metric,
            ctx.prefetch_cache_line_level,
            ctx.prefetch_lookahead,
        )
    }
}

/// Hook into [`storage::bin::load_from_bin`] by implementing [`storage::bin::SetData`].
impl<Data: GraphDataType> storage::bin::SetData for FastMemoryVectorProviderAsync<Data> {
    type Item = Data::VectorDataType;

    fn set_data(&mut self, i: usize, element: &[Self::Item]) -> ANNResult<()> {
        // SAFETY: No race can happen because we have a mutable reference to `self`.
        unsafe { self.set_vector_sync(i, element) }
    }
}

/// Hook into [`storage::bin::save_to_bin`] by implementing [`storage::bin::GetData`].
impl<Data: GraphDataType> storage::bin::GetData for FastMemoryVectorProviderAsync<Data> {
    type Element = Data::VectorDataType;
    type Item<'a> = &'a [Self::Element];

    fn get_data(&self, i: usize) -> ANNResult<Self::Item<'_>> {
        // SAFETY: We aren't full protected against races on the underlying data, but at
        // least `&self` will keep the data alive.
        Ok(unsafe { self.get_vector_sync(i) })
    }

    /// Return the total number of points, including frozen points.
    fn total(&self) -> usize {
        FastMemoryVectorProviderAsync::<Data>::total(self)
    }

    /// Return the dimension of each vector.
    fn dim(&self) -> usize {
        FastMemoryVectorProviderAsync::<Data>::dim(self)
    }
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroUsize, sync::Arc};

    use crate::storage::VirtualStorageProvider;
    use diskann::utils::vecid_from_usize;
    use diskann_vector::distance::Metric;
    use vfs::MemoryFS;

    use super::*;
    use crate::test_utils::graph_data_type_utils::GraphDataF32VectorUnitData;

    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_vector_provider() {
        let num_points = 100;
        let vector_provider = Arc::new(
            FastMemoryVectorProviderAsync::<GraphDataF32VectorUnitData>::new(
                num_points,
                3,
                Metric::L2,
                None,
                None,
            ),
        );
        let mut handles = Vec::new();
        for i in 0..num_points {
            let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
            let vector_provider_clone = Arc::clone(&vector_provider);
            handles.push(tokio::spawn(async move {
                // SAFETY: We're ensuring accesses are disjoint
                unsafe {
                    vector_provider_clone.set_vector_sync(vecid_from_usize(i).unwrap(), &vector)
                }
                .unwrap()
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }
        for i in 0..num_points {
            // SAFETY: We're only accessing one at a time.
            let vector = unsafe { vector_provider.get_vector_sync(vecid_from_usize(i).unwrap()) };
            assert_eq!(vector, &vec![(i as f32), (i + 1) as f32, (i + 2) as f32]);
        }
        assert_eq!(vector_provider.num_get_calls.get(), num_points);
    }

    fn create_test_provider() -> FastMemoryVectorProviderAsync<GraphDataF32VectorUnitData> {
        let num_points = 5;
        let dim = 3;

        let provider = FastMemoryVectorProviderAsync::<GraphDataF32VectorUnitData>::new(
            num_points,
            dim,
            Metric::L2,
            None,
            None,
        );

        assert_eq!(provider.total(), num_points);
        assert_eq!(provider.dim(), dim);

        // SAFETY: We have exclusive access over the fast provider and are careful to
        // not hold onto obtained slices.
        unsafe {
            // Set vectors.
            provider.set_vector_sync(0, &[0.0, 0.0, 0.0]).unwrap();
            provider.set_vector_sync(1, &[1.0, 1.0, 1.0]).unwrap();
            provider.set_vector_sync(2, &[2.0, 2.0, 2.0]).unwrap();
            provider.set_vector_sync(3, &[3.0, 3.0, 3.0]).unwrap();
            provider.set_vector_sync(4, &[4.0, 4.0, 4.0]).unwrap();
        }

        provider
    }

    #[test]
    fn test_provider() {
        let provider = create_test_provider();

        // SAFETY: We have exclusive access over the fast provider and are careful to
        // not hold onto obtained slices.
        unsafe {
            // Get vectors
            assert_eq!(provider.get_vector_sync(4), &[4.0, 4.0, 4.0]);
            assert_eq!(provider.get_vector_sync(3), &[3.0, 3.0, 3.0]);
            assert_eq!(provider.get_vector_sync(2), &[2.0, 2.0, 2.0]);
            assert_eq!(provider.get_vector_sync(1), &[1.0, 1.0, 1.0]);
            assert_eq!(provider.get_vector_sync(0), &[0.0, 0.0, 0.0]);

            // Error checking.
            assert!(provider.set_vector_sync(5, &[0.0, 0.0, 0.0]).is_err());
            assert!(provider.set_vector_sync(2, &[0.0, 0.0]).is_err());
        }
    }

    fn check_providers_equal(
        original: &FastMemoryVectorProviderAsync<GraphDataF32VectorUnitData>,
        reloaded: &FastMemoryVectorProviderAsync<GraphDataF32VectorUnitData>,
    ) {
        assert_eq!(original.total(), reloaded.total());
        assert_eq!(original.dim(), reloaded.dim());

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
        type Provider = FastMemoryVectorProviderAsync<GraphDataF32VectorUnitData>;
        let storage = VirtualStorageProvider::new(MemoryFS::new());
        let provider = create_test_provider();

        // Save to disk.
        let filepath = "/data.bin";
        assert!(!storage.exists(filepath));

        provider.save_to_bin(&storage, filepath).unwrap();
        assert!(
            storage.exists(filepath),
            "expected a new file to be created"
        );

        // Load from disk.
        let reloaded = Provider::load_from_bin(&storage, filepath, Metric::L2, None, None).unwrap();

        check_providers_equal(&provider, &reloaded);
    }

    #[tokio::test]
    async fn test_async_save() {
        type Provider = FastMemoryVectorProviderAsync<GraphDataF32VectorUnitData>;

        let storage = VirtualStorageProvider::new(MemoryFS::new());

        let provider = create_test_provider();

        let prefix = "/myindex";
        let expected_path = format!("{}.data", prefix);
        assert!(!storage.exists(prefix));
        assert!(!storage.exists(&expected_path));
        provider
            .save_with(&storage, &AsyncIndexMetadata::new(prefix.to_owned()))
            .await
            .unwrap();

        assert!(!storage.exists(prefix));
        assert!(
            storage.exists(&expected_path),
            "expected the async index prefix to append `.data` to the file path"
        );

        // Test that reloading finds the correct file paths.
        let ctx = AsyncQuantLoadContext {
            metadata: AsyncIndexMetadata::new(prefix.to_owned()),
            num_frozen_points: NonZeroUsize::new(5).unwrap(), // Don't care for this vector store.
            metric: Metric::L2,
            prefetch_lookahead: None,
            is_disk_index: false,
            prefetch_cache_line_level: None,
        };

        let reloaded = Provider::load_with(&storage, &ctx).await.unwrap();

        check_providers_equal(&provider, &reloaded);
    }
}
