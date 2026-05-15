/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{cell::UnsafeCell, mem, num::NonZeroUsize, ops::Deref, slice, sync::Arc};

use crate::storage::{StorageReadProvider, StorageWriteProvider};
use arc_swap::Guard;
use diskann::{ANNError, ANNResult, always_escalate, utils::IntoUsize};
use diskann_utils::future::AsyncFriendly;
use diskann_vector::distance::Metric;

use crate::{
    model::graph::provider::async_::{TableDeleteProviderAsync, postprocess},
    storage::{AsyncIndexMetadata, AsyncQuantLoadContext, LoadWith, SaveWith},
};

/// Represents a range of start points for an index.
/// The range includes `start` and excludes `end`.
/// `start` is the first valid point, and `end - 1` is the last valid point.
pub struct StartPoints {
    start: u32,
    end: u32,
}

impl StartPoints {
    pub fn new(valid_points: u32, frozen_points: NonZeroUsize) -> ANNResult<Self> {
        Ok(Self {
            start: valid_points,
            end: match valid_points.checked_add(frozen_points.get() as u32) {
                Some(end) => end,
                None => {
                    return Err(ANNError::log_index_error(
                        "Sum of valid points and frozen points exceeds u32::MAX",
                    ));
                }
            },
        })
    }

    pub fn range(&self) -> std::ops::Range<u32> {
        self.start..self.end
    }

    pub fn len(&self) -> usize {
        (self.end - self.start).into_usize()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn start(&self) -> u32 {
        self.start
    }

    pub fn end(&self) -> u32 {
        self.end
    }
}

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

/// Prefetch cache line level.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub enum PrefetchCacheLineLevel {
    /// 4 cache lines
    CacheLine4,
    /// 8 cache lines
    CacheLine8,
    /// 16 cache lines
    #[default]
    CacheLine16,
    /// prefetch all cache lines
    All,
}

//////////////////////////////////////////////////////////
// Common data structure and traits for async providers //
//////////////////////////////////////////////////////////

/// A ZST for [`MultiInsertStrategy::seed`](diskann::graph::glue::MultiInsertStrategy::Seed)
/// indicating no use of the input batch.
///
/// Inmem providers typically don't use a working set at all, instead passing through accesses
/// directly to the underlying provider. As such, no seeding is needed.
#[derive(Debug, Clone, Copy)]
pub struct Unseeded;

/// A helper trait to set element in the Async index.
pub trait SetElementHelper<T> {
    /// Set an element in the index.
    fn set_element(&self, index: &u32, element: &[T]) -> ANNResult<()>;
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

/// A helper trait to select the delete provider.
///
/// This is also implemented for [`NoDeletes`], which explicitly disables deletion
/// related functionality.
pub trait CreateDeleteProvider {
    /// The type of the created delete provider.
    type Target;

    /// Create a delete provider capable of tracking `total_points` number of deletes
    /// (or disabling deletion check all together).
    ///
    /// NOTE: The value `total_points` consists of the sum of `max_points` and
    /// `frozen_points`.
    fn create(self, total_points: usize) -> Self::Target;
}

pub trait VectorStore: AsyncFriendly {
    /// Total number of vectors in the store.
    fn total(&self) -> usize;

    /// Return the number of vector reads for a vector store.
    fn count_for_get_vector(&self) -> usize;
}

/// A tag type indicating that a method fails via panic instead of returning an error.
///
/// This is an enum with no alternatives, so is impossible to construct. Therefore, we know
/// that there can never be an actual value with this type.
///
#[derive(Debug)]
pub enum Panics {}

impl std::fmt::Display for Panics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "panics")
    }
}

impl std::error::Error for Panics {}
impl From<Panics> for ANNError {
    #[cold]
    fn from(_: Panics) -> ANNError {
        ANNError::log_async_error("unreachable")
    }
}

always_escalate!(Panics);

/// A tag type used to indicate that no store should be used.
///
/// Typically this would be for full precision only or quant only setups.
#[derive(Debug, Clone, Copy)]
pub struct NoStore;

impl CreateVectorStore for NoStore {
    type Target = NoStore;
    fn create(
        self,
        _max_points: usize,
        _metric: Metric,
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

impl LoadWith<AsyncQuantLoadContext> for NoStore {
    type Error = ANNError;
    async fn load_with<P>(_: &P, _: &AsyncQuantLoadContext) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        Ok(Self)
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

impl<T> SetElementHelper<T> for NoStore {
    fn set_element(&self, _index: &u32, _element: &[T]) -> ANNResult<()> {
        Ok(())
    }
}

/// A tag type to indicate that no deletes are allowed for this provider.
///
/// This effectively disables deletion support at compile-time.
#[derive(Debug, Clone, Copy)]
pub struct NoDeletes;

impl postprocess::DeletionCheck for NoDeletes {
    /// Always mark IDs as not deleted.
    ///
    /// We rely on constant propagation and dead-code elimination to optimize call-sites
    /// accordingly.
    #[inline(always)]
    fn deletion_check(&self, _: u32) -> bool {
        false
    }
}

impl CreateDeleteProvider for NoDeletes {
    type Target = Self;
    fn create(self, _: usize) -> Self {
        Self
    }
}

/// A tag type used to indicate that the `TableDeleteProviderAsync` should be used.
#[derive(Debug, Clone, Copy)]
pub struct TableBasedDeletes;

impl CreateDeleteProvider for TableBasedDeletes {
    type Target = TableDeleteProviderAsync;
    fn create(self, total_points: usize) -> Self::Target {
        TableDeleteProviderAsync::new(total_points)
    }
}

/// Operates entirely in full precision.
///
/// All indexing and search operations use the uncompressed full-precision vectors.
#[derive(Debug, Clone, Copy)]
pub struct FullPrecision;

/// Operates entirely in the quantized space.
///
/// All indexing and search operations use quantized vectors.
/// If full-precision vectors are available, they are only used for the final reranking step.
#[derive(Debug, Clone, Copy)]
pub struct Quantized;

/// Operates primarily in the quantized space with selective use of full precision.
///
/// # Search
/// Search is performed in the quantized space. Full-precision vectors are used only
/// to rerank the final candidate set.
///
/// # Insert and Prune
/// During insert operations, the search step uses quantized vectors.
/// Pruning then combines quantized vectors with a limited number of full-precision vectors.
///
/// The number of full-precision vectors used in pruning can be configured with
/// the `max_fp_vecs_per_prune` option when constructing a `BfTreeProvider`.
#[derive(Debug, Clone, Copy)]
pub struct Hybrid {
    /// Maximum number of full-precision vectors to use during pruning.
    /// This field is ignored during search, where full-precision vectors are never used.
    /// `None` defaults to use all full-precision vectors.
    pub max_fp_vecs_per_prune: Option<usize>,
}

impl Hybrid {
    /// Create a new `Hybrid` strategy with the specified maximum number of full-precision vectors
    /// to use during pruning.
    ///
    /// If `max_fp_vecs_per_prune` is `None`, use all full-precision vectors.
    pub fn new(max_fp_vecs_per_prune: Option<usize>) -> Self {
        Self {
            max_fp_vecs_per_prune,
        }
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

//////////////////////////////////
// diskann-record Save/Load     //
//////////////////////////////////

impl diskann_record::save::Save for StartPoints {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);

    fn save(
        &self,
        context: diskann_record::save::Context<'_>,
    ) -> diskann_record::save::Result<diskann_record::save::Record<'_>> {
        Ok(diskann_record::save_fields!(self, context, [start, end]))
    }
}

impl diskann_record::load::Load<'_> for StartPoints {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);

    fn load(
        object: diskann_record::load::Object<'_>,
    ) -> diskann_record::load::Result<Self> {
        diskann_record::load_fields!(object, [start: u32, end: u32]);
        Ok(Self { start, end })
    }

    fn load_legacy(
        _object: diskann_record::load::Object<'_>,
    ) -> diskann_record::load::Result<Self> {
        Err(diskann_record::load::error::Kind::UnknownVersion.into())
    }
}

impl diskann_record::save::Save for NoStore {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);

    fn save(
        &self,
        _context: diskann_record::save::Context<'_>,
    ) -> diskann_record::save::Result<diskann_record::save::Record<'_>> {
        Ok(diskann_record::save::Record::empty())
    }
}

impl diskann_record::load::Load<'_> for NoStore {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);

    fn load(
        _object: diskann_record::load::Object<'_>,
    ) -> diskann_record::load::Result<Self> {
        Ok(Self)
    }

    fn load_legacy(
        _object: diskann_record::load::Object<'_>,
    ) -> diskann_record::load::Result<Self> {
        Err(diskann_record::load::error::Kind::UnknownVersion.into())
    }
}

impl diskann_record::save::Save for NoDeletes {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);

    fn save(
        &self,
        _context: diskann_record::save::Context<'_>,
    ) -> diskann_record::save::Result<diskann_record::save::Record<'_>> {
        Ok(diskann_record::save::Record::empty())
    }
}

impl diskann_record::load::Load<'_> for NoDeletes {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);

    fn load(
        _object: diskann_record::load::Object<'_>,
    ) -> diskann_record::load::Result<Self> {
        Ok(Self)
    }

    fn load_legacy(
        _object: diskann_record::load::Object<'_>,
    ) -> diskann_record::load::Result<Self> {
        Err(diskann_record::load::error::Kind::UnknownVersion.into())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;

    #[test]
    fn new_creates_correct_range() {
        // valid_points of ten with five frozen points gives range 10..15
        let sp = StartPoints::new(10, NonZeroUsize::new(5).unwrap())
            .expect("should construct without overflow");
        let r = sp.range().collect::<Vec<_>>();
        assert_eq!(r, vec![10, 11, 12, 13, 14]);
        assert_eq!(sp.end(), 15);
    }

    #[test]
    fn new_returns_error_on_overflow() {
        // valid_points at u32::MAX plus one frozen point must overflow
        let max = u32::MAX;
        let res = StartPoints::new(max, NonZeroUsize::new(1).unwrap());
        assert!(res.is_err(), "expected an error when sum exceeds u32::MAX");
        if let Err(err) = res {
            let msg = err.to_string();
            assert!(
                msg.contains("Sum of valid points and frozen points exceeds u32::MAX"),
                "unexpected error message: {}",
                msg
            );
        }
    }

    /////////////////////////////////
    // diskann-record round-trips //
    /////////////////////////////////

    fn round_trip_helper<T>(value: &T) -> T
    where
        T: diskann_record::save::Saveable + for<'a> diskann_record::load::Loadable<'a>,
    {
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest = dir.path().join("manifest.json");
        diskann_record::save::save_to_disk(value, dir.path(), &manifest)
            .expect("save_to_disk");
        diskann_record::load::load_from_disk::<T>(&manifest, dir.path())
            .expect("load_from_disk")
    }

    #[test]
    fn start_points_round_trips_through_record() {
        let original = StartPoints::new(10, NonZeroUsize::new(5).unwrap()).unwrap();
        let restored = round_trip_helper(&original);
        assert_eq!(restored.start(), original.start());
        assert_eq!(restored.end(), original.end());
    }

    #[test]
    fn start_points_round_trips_at_boundary() {
        // Boundary: valid_points = 0, single frozen point.
        let original = StartPoints::new(0, NonZeroUsize::new(1).unwrap()).unwrap();
        let restored = round_trip_helper(&original);
        assert_eq!(restored.start(), 0);
        assert_eq!(restored.end(), 1);
    }

    #[test]
    fn no_store_round_trips_through_record() {
        let _restored = round_trip_helper(&NoStore);
    }

    #[test]
    fn no_deletes_round_trips_through_record() {
        let _restored = round_trip_helper(&NoDeletes);
    }
}
