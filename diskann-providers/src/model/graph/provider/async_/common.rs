/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{cell::UnsafeCell, mem, ops::Deref, slice, sync::Arc};

use crate::storage::{StorageReadProvider, StorageWriteProvider};
use arc_swap::Guard;
use diskann::{ANNError, ANNResult};
use diskann_utils::future::AsyncFriendly;
use diskann_vector::distance::Metric;

pub use diskann_provider_core::{
    CreateDeleteProvider, FullPrecision, Hybrid, NoDeletes, NoStore, Panics,
    PrefetchCacheLineLevel, Quantized, StartPoints, TableBasedDeletes, Unseeded,
};

use crate::storage::{AsyncIndexMetadata, AsyncQuantLoadContext, LoadWith, SaveWith};

pub struct VectorGuard<T> {
    inner: Guard<Arc<Vec<T>>>,
}

impl<T> VectorGuard<T> {
    pub(crate) fn from_guard(guard: Guard<Arc<Vec<T>>>) -> Self {
        Self { inner: guard }
    }
}

impl<T> Deref for VectorGuard<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

/// Memory-backed vector storage that aligns vectors to cache lines.
///
/// This stores vectors in a one giant allocation and guarantees that
/// each vector starts on a cache-aligned boundary (64 byte aligned).
/// To achieve this, vectors may not be densely packed into the underlying
/// buffer.
pub struct AlignedMemoryVectorStore<T: bytemuck::Pod> {
    store: UnsafeCell<Vec<T>>,
    max_vectors: usize,
    start_index: usize,
    dim: usize,
    padded_vector_dim: usize,
}

// SAFETY: It's not really, but the `bytemuck::Pod` bound helps mitigate the fallout.
unsafe impl<T: bytemuck::Pod + Sync> Sync for AlignedMemoryVectorStore<T> {}

// SAFETY: It's not really, but the `bytemuck::Pod` bound helps mitigate the fallout.
unsafe impl<T: bytemuck::Pod + Send> Send for AlignedMemoryVectorStore<T> {}

impl<T: bytemuck::Pod> AlignedMemoryVectorStore<T> {
    pub fn with_capacity(max_vectors: usize, dim: usize) -> Self {
        let elem_size = mem::size_of::<T>();
        assert!(64 % elem_size == 0);
        let vector_size = elem_size * dim;
        let extra_size = vector_size % 64;
        let padded_vector_dim = if extra_size == 0 {
            // vectors will be naturally aligned when packed
            dim
        } else {
            let padding_needed_size = 64 - extra_size;
            assert!(padding_needed_size.is_multiple_of(elem_size));
            let extra_elems = padding_needed_size / elem_size;
            dim + extra_elems
        };

        assert!((padded_vector_dim * elem_size).is_multiple_of(64));

        // Our allocation may start unaligned, so we will offset the first vector to ensure
        // correct alignment. This means we need some extra elements at the end to compensate.
        let last_elems: usize = 64 / elem_size - 1;

        let count = max_vectors * padded_vector_dim + last_elems;
        let mut store: UnsafeCell<Vec<T>> =
            UnsafeCell::new(vec![<T as bytemuck::Zeroable>::zeroed(); count]);

        let start_index = store.get_mut().as_ptr().align_offset(64);

        Self {
            store,
            max_vectors,
            start_index,
            dim,
            padded_vector_dim,
        }
    }

    pub fn max_vectors(&self) -> usize {
        self.max_vectors
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return a vector as a slice.
    ///
    /// # Safety
    ///
    /// This function will not read out of bounds, but it may observe a torn read if reading a vector at the same time
    /// as it is being written. It is up to the caller to deal with torn reads, but it is expected that clients wanting maximum
    /// performance will be okay with that tradeoff.
    ///
    /// Note that as vector elements are plain data, the impact of memory races is limited.
    #[inline(always)]
    pub unsafe fn get_slice(&self, index: usize) -> &[T] {
        assert!(
            index < self.max_vectors,
            "index ({}) exceeded max_vectors ({})",
            index,
            self.max_vectors
        );
        let index = index * self.padded_vector_dim + self.start_index;

        // SAFETY: Constructing a slice to the inside of the allocation. We know this is
        // valid memory because the allocation is sized so that all vectors fit, and we know
        // that the `index < max_vectors`.
        let buf = unsafe { (*self.store.get()).as_ptr() };

        // SAFETY: See comment above.
        unsafe { slice::from_raw_parts(buf.add(index), self.dim) }
    }

    /// Return a vector as a mutable slice.
    ///
    /// # Safety
    ///
    /// This function will not synchronize access, but the memory is guaranteed to be valid. Callers must synchronize
    /// themselves if they require consistency.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut_slice(&self, index: usize) -> &mut [T] {
        assert!(index < self.max_vectors);
        let index = index * self.padded_vector_dim + self.start_index;

        // SAFETY: Constructing a mutable slice to the inside of the allocation. We know
        // this is valid memory because the allocation is sized so that all vectors fit,
        // and we know that the `index < max_vectors`.
        unsafe {
            let buf = (*self.store.get()).as_mut_ptr();
            slice::from_raw_parts_mut(buf.add(index), self.dim)
        }
    }
}

//////////////////////////////////////////////////////////
// Common data structure and traits for async providers //
//////////////////////////////////////////////////////////

impl LoadWith<AsyncQuantLoadContext> for NoStore {
    type Error = ANNError;
    async fn load_with<P>(_: &P, _: &AsyncQuantLoadContext) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        Ok(NoStore)
    }
}

impl SaveWith<AsyncIndexMetadata> for NoStore {
    type Ok = usize;
    type Error = ANNError;
    async fn save_with<P>(&self, _provider: &P, _auxiliary: &AsyncIndexMetadata) -> ANNResult<usize>
    where
        P: StorageWriteProvider,
    {
        Ok(0)
    }
}

#[cfg(test)]
pub struct TestCallCount {
    count: std::sync::atomic::AtomicUsize,
}

#[cfg(test)]
impl TestCallCount {
    pub fn new() -> Self {
        TestCallCount {
            count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn enabled() -> bool {
        true
    }

    pub fn increment(&self) {
        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get(&self) -> usize {
        self.count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(not(test))]
pub struct TestCallCount {}

#[cfg(not(test))]
impl TestCallCount {
    pub fn new() -> Self {
        TestCallCount {}
    }

    pub fn enabled() -> bool {
        false
    }

    pub fn increment(&self) {}

    pub fn get(&self) -> usize {
        0
    }
}

impl Default for TestCallCount {
    fn default() -> Self {
        Self::new()
    }
}

/// A helper trait to select the quant vector store.
///
/// This is also implemented for [`NoStore`], which explicitly disables deletion
/// related functionality.
pub trait CreateVectorStore {
    /// The type of the created vector store.
    type Target: VectorStore;

    /// Create a quant provider capable of tracking `max_points`.
    fn create(
        self,
        max_points: usize,
        metric: Metric,
        prefetch_lookahead: Option<usize>,
    ) -> Self::Target;
}

/// A helper trait to set element in the Async index.
pub trait SetElementHelper<T> {
    /// Set an element in the index.
    fn set_element(&self, index: &u32, element: &[T]) -> ANNResult<()>;
}

pub trait VectorStore: AsyncFriendly {
    /// Total number of vectors in the store.
    fn total(&self) -> usize;

    /// Return the number of vector reads for a vector store.
    fn count_for_get_vector(&self) -> usize;
}

impl CreateVectorStore for NoStore {
    type Target = NoStore;
    fn create(
        self,
        _max_points: usize,
        _metric: diskann_vector::distance::Metric,
        _prefetch_lookahead: Option<usize>,
    ) -> Self::Target {
        self
    }
}

impl VectorStore for NoStore {
    fn total(&self) -> usize {
        0
    }

    fn count_for_get_vector(&self) -> usize {
        0
    }
}

impl<T> SetElementHelper<T> for NoStore {
    fn set_element(&self, _index: &u32, _element: &[T]) -> ANNResult<()> {
        Ok(())
    }
}
