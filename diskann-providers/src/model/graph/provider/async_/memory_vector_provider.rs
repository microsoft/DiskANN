/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Basic in-memory vector provider.
//!
//! This implementation stores all vectors in memory. It uses ArcSwap to synchronize reads and writes, and the vectors are not stored
//! cache aligned. The backing store is `Vec<ArcSwap<Vec<Data::VectorDataType>>>`.

use std::sync::Arc;

use crate::storage::{StorageReadProvider, StorageWriteProvider};
use arc_swap::ArcSwap;
use diskann::{ANNError, ANNResult};

use super::{VectorGuard, common::TestCallCount};
use crate::{
    model::graph::traits::GraphDataType,
    storage::{self, AsyncIndexMetadata, AsyncQuantLoadContext, LoadWith, SaveWith},
};

pub struct MemoryVectorProviderAsync<Data: GraphDataType> {
    dim: usize,
    max_vectors: usize,
    vectors: Vec<ArcSwap<Vec<Data::VectorDataType>>>,

    pub num_get_calls: TestCallCount,
}

impl<Data: GraphDataType> MemoryVectorProviderAsync<Data> {
    pub fn new(max_vectors: usize, dim: usize) -> Self {
        let vectors = (0..max_vectors)
            .map(|_| ArcSwap::new(Arc::new(vec![Data::VectorDataType::default(); dim])))
            .collect();

        Self {
            dim,
            max_vectors,
            vectors,
            num_get_calls: TestCallCount::default(),
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

    /// Return a protected slice over the data at index `i`.
    ///
    /// Errors if `i >= self.total()`.
    pub(crate) fn get_vector_sync(&self, i: usize) -> ANNResult<VectorGuard<Data::VectorDataType>> {
        self.num_get_calls.increment();
        match self.vectors.get(i) {
            Some(vector) => Ok(VectorGuard::from_guard(vector.load())),
            None => Err(ANNError::log_index_error(
                "Vector id is out of boundary in the dataset.",
            )),
        }
    }

    /// Store the data in `v` into the internal data at position `i`.
    ///
    /// Errors if:
    ///
    /// * `i >= self.total()`: `i` must be inbounds.
    /// * `v.dim() != self.dim()`: The slice must have the proper length.
    pub(crate) fn set_vector_sync(&self, i: usize, v: &[Data::VectorDataType]) -> ANNResult<()> {
        if v.len() != self.dim {
            return Err(ANNError::log_index_error(
                "Vector dimension is not equal to the expected dimension.",
            ));
        }
        let slot = match self.vectors.get(i) {
            Some(slot) => slot,
            None => {
                return Err(ANNError::log_index_error(
                    "Vector id is out of boundary in the dataset.",
                ));
            }
        };

        slot.swap(Arc::new(v.to_vec()));
        Ok(())
    }

    /// Load `self` directly from a `.bin` file at `path`.
    ///
    /// Because the number of start points are not saved as part of the `.bin` file format,
    /// this must be provided externally.
    ///
    /// See also: [`storage::bin::load_from_bin`].
    pub fn load_from_bin<P>(provider: &P, path: &str) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        storage::bin::load_from_bin(provider, path, |num_points, dim| -> ANNResult<Self> {
            Ok(Self::new(num_points, dim))
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
}

/// This is an adaptor for compatibility with the async index serialization.
impl<Data> SaveWith<AsyncIndexMetadata> for MemoryVectorProviderAsync<Data>
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

impl<Data> LoadWith<AsyncQuantLoadContext> for MemoryVectorProviderAsync<Data>
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
        Self::load_from_bin(provider, &ctx.metadata.data_path())
    }
}

/// Hook into [`storage::bin::load_from_bin`] by implementing [`storage::bin::SetData`].
impl<Data: GraphDataType> storage::bin::SetData for MemoryVectorProviderAsync<Data> {
    type Item = Data::VectorDataType;

    fn set_data(&mut self, i: usize, element: &[Self::Item]) -> ANNResult<()> {
        self.set_vector_sync(i, element)
    }
}

/// Hook into [`storage::bin::save_to_bin`] by implementing [`storage::bin::GetData`].
impl<Data: GraphDataType> storage::bin::GetData for MemoryVectorProviderAsync<Data> {
    type Element = Data::VectorDataType;
    type Item<'a> = VectorGuard<Self::Element>;

    fn get_data(&self, i: usize) -> ANNResult<Self::Item<'_>> {
        self.get_vector_sync(i)
    }

    /// Return the total number of points, including frozen points.
    fn total(&self) -> usize {
        MemoryVectorProviderAsync::<Data>::total(self)
    }

    /// Return the dimension of each vector.
    fn dim(&self) -> usize {
        MemoryVectorProviderAsync::<Data>::dim(self)
    }
}

///////////
// Tests //
///////////

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
        let vector_provider =
            Arc::new(MemoryVectorProviderAsync::<GraphDataF32VectorUnitData>::new(num_points, 3));
        let mut handles = Vec::new();
        for i in 0..num_points {
            let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
            let vector_provider_clone = Arc::clone(&vector_provider);
            handles.push(tokio::spawn(async move {
                vector_provider_clone
                    .set_vector_sync(vecid_from_usize(i).unwrap(), &vector)
                    .unwrap()
            }));
        }
        for handle in handles {
            handle.await.unwrap();
        }
        for i in 0..num_points {
            let vector = vector_provider
                .get_vector_sync(vecid_from_usize(i).unwrap())
                .unwrap();
            assert_eq!(&*vector, &vec![(i as f32), (i + 1) as f32, (i + 2) as f32]);
        }
        assert_eq!(vector_provider.num_get_calls.get(), num_points);
    }

    fn create_test_provider() -> MemoryVectorProviderAsync<GraphDataF32VectorUnitData> {
        let num_points = 3;
        let frozen_points = 2;
        let dim = 3;

        let provider = MemoryVectorProviderAsync::<GraphDataF32VectorUnitData>::new(
            num_points + frozen_points,
            dim,
        );

        assert_eq!(provider.total(), num_points + frozen_points);
        assert_eq!(provider.dim(), dim);

        // Set vectors.
        provider.set_vector_sync(0, &[0.0, 0.0, 0.0]).unwrap();
        provider.set_vector_sync(1, &[1.0, 1.0, 1.0]).unwrap();
        provider.set_vector_sync(2, &[2.0, 2.0, 2.0]).unwrap();
        provider.set_vector_sync(3, &[3.0, 3.0, 3.0]).unwrap();
        provider.set_vector_sync(4, &[4.0, 4.0, 4.0]).unwrap();

        provider
    }

    #[test]
    fn test_provider() {
        let provider = create_test_provider();

        // Get vectors
        assert_eq!(&*provider.get_vector_sync(4).unwrap(), &[4.0, 4.0, 4.0]);
        assert_eq!(&*provider.get_vector_sync(3).unwrap(), &[3.0, 3.0, 3.0]);
        assert_eq!(&*provider.get_vector_sync(2).unwrap(), &[2.0, 2.0, 2.0]);
        assert_eq!(&*provider.get_vector_sync(1).unwrap(), &[1.0, 1.0, 1.0]);
        assert_eq!(&*provider.get_vector_sync(0).unwrap(), &[0.0, 0.0, 0.0]);

        // Error checking.
        assert!(provider.get_vector_sync(5).is_err());
        assert!(provider.set_vector_sync(5, &[0.0, 0.0, 0.0]).is_err());
        assert!(provider.set_vector_sync(2, &[0.0, 0.0]).is_err());
    }

    fn check_providers_equal(
        original: &MemoryVectorProviderAsync<GraphDataF32VectorUnitData>,
        reloaded: &MemoryVectorProviderAsync<GraphDataF32VectorUnitData>,
    ) {
        assert_eq!(original.total(), reloaded.total());
        assert_eq!(original.dim(), reloaded.dim());

        for i in 0..original.total() {
            let a = &*original.get_vector_sync(i).unwrap();
            let b = &*reloaded.get_vector_sync(i).unwrap();
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
        type Provider = MemoryVectorProviderAsync<GraphDataF32VectorUnitData>;
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
        let reloaded = Provider::load_from_bin(&storage, filepath).unwrap();

        check_providers_equal(&provider, &reloaded);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_async_save() {
        type Provider = MemoryVectorProviderAsync<GraphDataF32VectorUnitData>;

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
