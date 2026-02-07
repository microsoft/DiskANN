/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{future::Future, sync::Mutex};

use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann::{
    ANNError, ANNResult,
    graph::glue::{
        self, ExpandBeam, FillSet, FilterStartPoints, InsertStrategy, PruneStrategy, SearchExt,
        SearchStrategy,
    },
    provider::{
        Accessor, BuildDistanceComputer, BuildQueryComputer, DelegateNeighbor, ExecutionContext,
        HasId,
    },
    utils::{IntoUsize, VectorRepr},
};
use diskann_quantization::{
    AsFunctor, CompressInto,
    bits::{Representation, Unsigned},
    meta::NotCanonical,
    scalar::{
        CompensatedCosineNormalized, CompensatedIP, CompensatedSquaredL2, CompensatedVector,
        CompensatedVectorRef, InputContainsNaN, MeanNormMissing, MutCompensatedVectorRef,
        ScalarQuantizer,
    },
};
use diskann_utils::{Reborrow, ReborrowMut, future::AsyncFriendly};
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
use thiserror::Error;

use super::{DefaultProvider, GetFullPrecision, Rerank};
use crate::CreateVectorStore;
use diskann_providers::{
    common::IgnoreLockPoison,
    model::graph::{
        provider::async_::{
            FastMemoryVectorProviderAsync, SimpleNeighborProviderAsync,
            common::{
                AlignedMemoryVectorStore, NoStore, Quantized, SetElementHelper,
                TestCallCount, VectorStore,
            },
            postprocess::{AsDeletionCheck, DeletionCheck, RemoveDeletedIdsAndCopy},
        },
        traits::AdHoc,
    },
    storage::{self, AsyncIndexMetadata, AsyncQuantLoadContext, LoadWith, SaveWith},
};
use crate::{FullPrecisionProvider, FullPrecisionStore};

type CVRef<'a, const NBITS: usize> = CompensatedVectorRef<'a, NBITS>;

/// A thin wrapper around [`ScalarQuantizer`] that encodes the number of bits desired for
/// the [`SQStore`] derived from the quantizer.
///
/// This is meant to be used in conjunction with [`CreateQuantProvider`] to serve as a
/// precursor for [`DefaultProvider::new_empty`].
#[derive(Clone)]
pub struct WithBits<const NBITS: usize> {
    quantizer: ScalarQuantizer,
}

impl<const NBITS: usize> WithBits<NBITS> {
    pub fn new(quantizer: ScalarQuantizer) -> Self {
        Self { quantizer }
    }
}

//////////////
// Provider //
//////////////

/// This controls how many vectors share a write lock.
///
/// With the default value of 16, vector IDs 0-15 will share write lock 0,
/// 16-31 will share write lock 1, etc.
const WRITE_LOCK_GRANULARITY: usize = 16;

/// The default prefetch lookahead to use if not configured externally.
const PREFETCH_DEFAULT: usize = 8;

pub struct SQStore<const NBITS: usize> {
    data: AlignedMemoryVectorStore<u8>,
    quantizer: ScalarQuantizer,
    metric: Metric,

    // We keep only write locks as reads are unsynchronized. Since there are
    // only writers, we use Mutex here. Note that sync::Mutex is ok here
    // because the Mutex is never held across an await.
    write_locks: Vec<Mutex<()>>,

    /// Prefetching for scalar bulk operations.
    prefetch_lookahead: usize,

    num_get_calls: TestCallCount,
}

impl<const NBITS: usize> SQStore<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    pub(super) fn new(
        quantizer: ScalarQuantizer,
        num_vectors: usize,
        metric: Metric,
        prefetch_lookahead: Option<usize>,
    ) -> Self {
        let write_locks = (0..num_vectors.div_ceil(WRITE_LOCK_GRANULARITY))
            .map(|_| Mutex::new(()))
            .collect::<Vec<_>>();
        // Compute the number of bytes needed to hold the data and the compensation
        // coefficient.
        let bytes = CVRef::<NBITS>::canonical_bytes(quantizer.dim());
        Self {
            data: AlignedMemoryVectorStore::with_capacity(num_vectors, bytes),
            quantizer,
            metric,
            write_locks,
            num_get_calls: TestCallCount::default(),
            prefetch_lookahead: prefetch_lookahead.unwrap_or(PREFETCH_DEFAULT),
        }
    }

    /// Prefetch the first few cache lines of the data for vector `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.size()`.
    pub(crate) fn prefetch_hint(&self, i: usize) {
        // SAFETY: Racing on the underlying data is okay because we are dispatching to
        // an architectural primitive for prefetching that doesn't care about the data
        // itself, just its address.
        let data = unsafe { self.data.get_slice(i) };
        diskann_vector::prefetch_hint_max::<4, _>(data);
    }

    pub(super) fn dim(&self) -> usize {
        self.quantizer.dim()
    }

    pub(super) fn get_vector(&self, i: usize) -> Result<CVRef<'_, NBITS>, SQError> {
        self.num_get_calls.increment();
        Ok(CVRef::from_canonical_front(
            unsafe { self.data.get_slice(i) },
            self.dim(),
        )?)
    }

    pub(super) fn set_vector<T>(&self, i: usize, v: &[T]) -> Result<(), SQError>
    where
        T: VectorRepr,
    {
        let vf32: &[f32] =
            &T::as_f32(v).map_err(|e| SQError::FullPrecisionConversionErr(format!("{:?}", e)))?;

        debug_assert!(
            vf32.len() == self.dim(),
            "vector f32 dimension {} does not match dimension {}",
            vf32.len(),
            self.dim()
        );

        let lock_id = i / WRITE_LOCK_GRANULARITY;
        let _guard = self.write_locks[lock_id].lock_or_panic();

        self.quantizer.compress_into(
            vf32,
            MutCompensatedVectorRef::<NBITS>::from_canonical_front_mut(
                unsafe { self.data.get_mut_slice(i) },
                self.dim(),
            )?,
        )?;
        Ok(())
    }

    /// Store the compressed SQ vector directly at position `i`.
    ///
    /// Panic if:
    ///
    /// * `i >= self.total()`: `i` must be inbounds.
    /// * `v.len() != self.dim()`: `v` must have the right length.
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
        let expected_quant_len = CVRef::<NBITS>::canonical_bytes(self.dim());
        debug_assert!(
            v.len() == expected_quant_len,
            "vector length {} does not match dimension {}",
            v.len(),
            expected_quant_len
        );

        let lock_id = i / WRITE_LOCK_GRANULARITY;
        let _guard = self.write_locks[lock_id].lock_or_panic();
        // SAFETY: `get_mut_slice` guarantees it is safe to access the memory,
        // and but it may be a torn read. As we are trading off synchronization
        // for speed, this is okay.
        unsafe { self.data.get_mut_slice(i) }.copy_from_slice(v);
        Ok(())
    }

    // Return a distance computer.
    pub(super) fn distance_computer(&self) -> Result<DistanceComputer, SQError> {
        Ok(match self.metric {
            Metric::L2 => DistanceComputer::SquaredL2(self.quantizer.as_functor()),
            Metric::InnerProduct => DistanceComputer::InnerProduct(self.quantizer.as_functor()),
            Metric::CosineNormalized => {
                DistanceComputer::CosineNormalized(self.quantizer.as_functor())
            }
            unsupported_metric => {
                return Err(SQError::UnsupportedDistanceMetric(unsupported_metric));
            }
        })
    }

    pub(super) fn query_computer<T>(
        &self,
        query: &[T],
        allow_rescale: bool,
    ) -> Result<QueryComputer<NBITS>, SQError>
    where
        T: VectorRepr,
    {
        let mut boxed = CompensatedVector::new_boxed(self.dim());
        let q = T::as_f32(query)
            .map_err(|e| SQError::FullPrecisionConversionErr(format!("{:?}", e)))?;

        if allow_rescale && !matches!(self.metric, Metric::L2 | Metric::CosineNormalized) {
            let mut query: Box<[f32]> = q.as_ref().into();
            self.quantizer.rescale(&mut query)?;
            self.quantizer
                .compress_into(&*query, boxed.reborrow_mut())?;
        } else {
            self.quantizer
                .compress_into(q.as_ref(), boxed.reborrow_mut())?;
        }

        Ok(QueryComputer {
            inner: self.distance_computer()?,
            query: boxed,
        })
    }

    pub fn prefetch_lookahead(&self) -> usize {
        self.prefetch_lookahead
    }
}

#[derive(Debug)]
pub enum DistanceComputer {
    SquaredL2(CompensatedSquaredL2),
    InnerProduct(CompensatedIP),
    CosineNormalized(CompensatedCosineNormalized),
}

impl<const NBITS: usize> DistanceFunction<CVRef<'_, NBITS>, CVRef<'_, NBITS>, f32>
    for DistanceComputer
where
    Unsigned: Representation<NBITS>,
    CompensatedSquaredL2: for<'a, 'b> DistanceFunction<
            CVRef<'a, NBITS>,
            CVRef<'b, NBITS>,
            diskann_quantization::distances::Result<f32>,
        >,
    CompensatedIP: for<'a, 'b> DistanceFunction<
            CVRef<'a, NBITS>,
            CVRef<'b, NBITS>,
            diskann_quantization::distances::Result<f32>,
        >,
    CompensatedCosineNormalized: for<'a, 'b> DistanceFunction<
            CVRef<'a, NBITS>,
            CVRef<'b, NBITS>,
            diskann_quantization::distances::Result<f32>,
        >,
{
    #[inline(always)]
    fn evaluate_similarity(&self, left: CVRef<'_, NBITS>, right: CVRef<'_, NBITS>) -> f32 {
        let r = match self {
            DistanceComputer::SquaredL2(f) => f.evaluate_similarity(left, right),
            DistanceComputer::InnerProduct(f) => f.evaluate_similarity(left, right),
            DistanceComputer::CosineNormalized(f) => f.evaluate_similarity(left, right),
        };

        r.map_err(|err| err.panic(left.len(), right.len())).unwrap()
    }
}

pub struct QueryComputer<const NBITS: usize>
where
    Unsigned: Representation<NBITS>,
{
    inner: DistanceComputer,
    query: CompensatedVector<NBITS>,
}

impl<const NBITS: usize> PreprocessedDistanceFunction<CVRef<'_, NBITS>, f32>
    for QueryComputer<NBITS>
where
    Unsigned: Representation<NBITS>,
    DistanceComputer: for<'a, 'b> DistanceFunction<CVRef<'a, NBITS>, CVRef<'b, NBITS>, f32>,
{
    fn evaluate_similarity(&self, changing: CVRef<'_, NBITS>) -> f32 {
        self.inner
            .evaluate_similarity(self.query.reborrow(), changing)
    }
}

///////////////////
// Data Provider //
///////////////////

impl<const NBITS: usize> CreateVectorStore for WithBits<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    type Target = SQStore<NBITS>;

    /// Create a quant provider capable of tracking `max_points`.
    fn create(
        self,
        max_points: usize,
        metric: Metric,
        prefetch_lookahead: Option<usize>,
    ) -> Self::Target {
        SQStore::new(self.quantizer, max_points, metric, prefetch_lookahead)
    }
}

impl<const NBITS: usize> VectorStore for SQStore<NBITS> {
    fn total(&self) -> usize {
        self.data.max_vectors()
    }

    fn count_for_get_vector(&self) -> usize {
        self.num_get_calls.get()
    }
}

////////////////
// SetElement //
////////////////

/// Assign to SQ vector store.
impl<T, const NBITS: usize> SetElementHelper<T> for SQStore<NBITS>
where
    T: VectorRepr,
    Unsigned: Representation<NBITS>,
{
    fn set_element(&self, id: &u32, element: &[T]) -> ANNResult<()> {
        self.set_vector(id.into_usize(), element)?;
        Ok(())
    }
}

//////////////
// Accessor //
//////////////

/// The accessor for SQ.
pub struct QuantAccessor<'a, const NBITS: usize, V, D, Ctx> {
    provider: &'a DefaultProvider<V, SQStore<NBITS>, D, Ctx>,
    id_buffer: Vec<u32>,
    is_search: bool,
}

impl<T, const NBITS: usize, D, Ctx> GetFullPrecision
    for QuantAccessor<'_, NBITS, FullPrecisionStore<T>, D, Ctx>
where
    T: VectorRepr,
{
    type Repr = T;
    fn as_full_precision(&self) -> &FastMemoryVectorProviderAsync<AdHoc<T>> {
        &self.provider.base_vectors
    }
}

impl<const NBITS: usize, V, D, Ctx> HasId for QuantAccessor<'_, NBITS, V, D, Ctx> {
    type Id = u32;
}

impl<const NBITS: usize, V, D, Ctx> SearchExt for QuantAccessor<'_, NBITS, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
    Unsigned: Representation<NBITS>,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }
}

impl<'a, const NBITS: usize, V, D, Ctx> QuantAccessor<'a, NBITS, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    pub(crate) fn new(
        provider: &'a DefaultProvider<V, SQStore<NBITS>, D, Ctx>,
        is_search: bool,
    ) -> Self {
        Self {
            provider,
            id_buffer: Vec::with_capacity(32),
            is_search,
        }
    }
}

impl<'a, const NBITS: usize, V, D, Ctx> Accessor for QuantAccessor<'a, NBITS, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
    Unsigned: Representation<NBITS>,
{
    /// The extended element inherets the lifetime of the Accessor.
    type Extended = CVRef<'a, NBITS>;

    /// This accessor returns raw slices. There *is* a chance of racing when the fast
    /// providers are used. We just have to live with it.
    ///
    /// NOTE: We intentionally don't use `'b` here since our implementation borrows
    /// the inner `CVRef` from the underlying provider.
    type Element<'b>
        = CVRef<'a, NBITS>
    where
        Self: 'b;

    /// `ElementRef` has an arbitrarily short lifetime.
    type ElementRef<'b> = CVRef<'b, NBITS>;

    /// Choose to panic on an out-of-bounds access rather than propagate an error.
    type GetError = ANNError;

    /// Return the quantized vector stored at index `i`.
    ///
    /// This function always completes synchronously.
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        // SAFETY: We've decided to live with UB that can result from potentially mixing
        // unsynchronized reads and writes on the underlying memory.
        std::future::ready(
            match self.provider.aux_vectors.get_vector(id.into_usize()) {
                Ok(v) => Ok(v),
                Err(err) => Err(err.into()),
            },
        )
    }

    /// Perform a bulk operation.
    ///
    /// This implementation uses prefetching.
    fn on_elements_unordered<Itr, F>(
        &mut self,
        itr: Itr,
        mut f: F,
    ) -> impl Future<Output = Result<(), Self::GetError>> + Send
    where
        Self: Sync,
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + for<'b> FnMut(Self::ElementRef<'b>, Self::Id),
    {
        // Reuse the internal buffer to collect the results and give us random access
        // capabilities.
        let id_buffer = &mut self.id_buffer;
        id_buffer.clear();
        id_buffer.extend(itr);

        let len = id_buffer.len();
        let lookahead = self.provider.aux_vectors.prefetch_lookahead();

        // Prefetch the first few vectors.
        for id in id_buffer.iter().take(lookahead) {
            self.provider.aux_vectors.prefetch_hint(id.into_usize());
        }

        for (i, id) in id_buffer.iter().enumerate() {
            // Prefetch `lookahead` iterations ahead as long as it is safe.
            if lookahead > 0 && i + lookahead < len {
                self.provider
                    .aux_vectors
                    .prefetch_hint(id_buffer[i + lookahead].into_usize());
            }

            let vector = match self.provider.aux_vectors.get_vector(id.into_usize()) {
                Ok(v) => v,
                Err(e) => return std::future::ready(Err(e.into())),
            };

            // Invoke the passed closure on the vector.
            //
            // SAFETY: We're accepting the consequences of potential unsynchronized,
            // concurrent mutation.
            f(vector, *id)
        }

        std::future::ready(Ok(()))
    }
}

impl<'a, const NBITS: usize, V, D, Ctx> DelegateNeighbor<'a> for QuantAccessor<'_, NBITS, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type Delegate = &'a SimpleNeighborProviderAsync<u32>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<const NBITS: usize, V, D, Ctx, T> BuildQueryComputer<[T]>
    for QuantAccessor<'_, NBITS, V, D, Ctx>
where
    T: VectorRepr,
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
    Unsigned: Representation<NBITS>,
    QueryComputer<NBITS>: for<'a> PreprocessedDistanceFunction<CVRef<'a, NBITS>, f32>,
{
    type QueryComputerError = ANNError;
    type QueryComputer = QueryComputer<NBITS>;

    fn build_query_computer(
        &self,
        from: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        // Allow rescaling if this is search.
        Ok(self
            .provider
            .aux_vectors
            .query_computer(from, self.is_search)?)
    }
}

impl<const NBITS: usize, V, D, Ctx, T> ExpandBeam<[T]> for QuantAccessor<'_, NBITS, V, D, Ctx>
where
    T: VectorRepr,
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
    Unsigned: Representation<NBITS>,
    QueryComputer<NBITS>: for<'a> PreprocessedDistanceFunction<CVRef<'a, NBITS>, f32>,
{
}

impl<const NBITS: usize, V, D, Ctx> BuildDistanceComputer for QuantAccessor<'_, NBITS, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
    Unsigned: Representation<NBITS>,
    DistanceComputer: for<'a, 'b> DistanceFunction<CVRef<'a, NBITS>, CVRef<'b, NBITS>, f32>,
{
    type DistanceComputerError = ANNError;
    type DistanceComputer = DistanceComputer;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(self.provider.aux_vectors.distance_computer()?)
    }
}

impl<const NBITS: usize, V, D, Ctx> AsDeletionCheck for QuantAccessor<'_, NBITS, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
    Unsigned: Representation<NBITS>,
{
    type Checker = D;
    fn as_deletion_check(&self) -> &D {
        &self.provider.deleted
    }
}

////////////////
// Strategies //
////////////////

/// SearchStrategy for quantized search when a full-precision store exists alongside
/// the quantized store. This allows reranking using original vectors after
/// approximate search, so the post-processing step includes a [`Rerank`] stage.
impl<const NBITS: usize, D, Ctx, T>
    SearchStrategy<FullPrecisionProvider<T, SQStore<NBITS>, D, Ctx>, [T]> for Quantized
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
    Unsigned: Representation<NBITS>,
    QueryComputer<NBITS>: for<'a> PreprocessedDistanceFunction<CVRef<'a, NBITS>, f32>,
{
    type QueryComputer = QueryComputer<NBITS>;
    type SearchAccessor<'a> = QuantAccessor<'a, NBITS, FullPrecisionStore<T>, D, Ctx>;
    type SearchAccessorError = ANNError;
    type PostProcessor = glue::Pipeline<FilterStartPoints, Rerank>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, SQStore<NBITS>, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider, true))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

/// SearchStrategy for quantized search when only the quantized store is present.
/// Since no full-precision vectors exist, reranking is not possible and the
/// post-processing step just copies candidate IDs forward via [`RemoveDeletedIdsAndCopy`].
impl<const NBITS: usize, D, Ctx, T>
    SearchStrategy<DefaultProvider<NoStore, SQStore<NBITS>, D, Ctx>, [T]> for Quantized
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
    Unsigned: Representation<NBITS>,
    QueryComputer<NBITS>: for<'a> PreprocessedDistanceFunction<CVRef<'a, NBITS>, f32>,
{
    type QueryComputer = QueryComputer<NBITS>;
    type SearchAccessor<'a> = QuantAccessor<'a, NBITS, NoStore, D, Ctx>;
    type SearchAccessorError = ANNError;
    type PostProcessor = glue::Pipeline<FilterStartPoints, RemoveDeletedIdsAndCopy>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DefaultProvider<NoStore, SQStore<NBITS>, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider, true))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<const NBITS: usize, V, D, Ctx> PruneStrategy<DefaultProvider<V, SQStore<NBITS>, D, Ctx>>
    for Quantized
where
    V: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
    Unsigned: Representation<NBITS>,
    DistanceComputer: for<'a, 'b> DistanceFunction<CVRef<'a, NBITS>, CVRef<'b, NBITS>, f32>,
{
    type DistanceComputer = DistanceComputer;
    type PruneAccessor<'a> = QuantAccessor<'a, NBITS, V, D, Ctx>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DefaultProvider<V, SQStore<NBITS>, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(QuantAccessor::new(provider, false))
    }
}

impl<const NBITS: usize, V, D, Ctx> FillSet for QuantAccessor<'_, NBITS, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
    Unsigned: Representation<NBITS>,
{
}

impl<const NBITS: usize, V, D, Ctx, T>
    InsertStrategy<DefaultProvider<V, SQStore<NBITS>, D, Ctx>, [T]> for Quantized
where
    T: VectorRepr,
    V: AsyncFriendly,
    D: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
    Unsigned: Representation<NBITS>,
    QueryComputer<NBITS>: for<'a> PreprocessedDistanceFunction<CVRef<'a, NBITS>, f32>,
    DistanceComputer: for<'a, 'b> DistanceFunction<CVRef<'a, NBITS>, CVRef<'b, NBITS>, f32>,
    Quantized: SearchStrategy<DefaultProvider<V, SQStore<NBITS>, D, Ctx>, [T]>,
{
    type PruneStrategy = Self;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

////////////////////////////
// SaveWith  and LoadWith //
////////////////////////////

impl<const NBITS: usize> SaveWith<AsyncIndexMetadata> for SQStore<NBITS> {
    type Ok = usize;
    type Error = ANNError;

    async fn save_with<P>(
        &self,
        write_provider: &P,
        metadata: &AsyncIndexMetadata,
    ) -> Result<Self::Ok, Self::Error>
    where
        P: StorageWriteProvider,
    {
        let sq_storage = storage::SQStorage::new(metadata.prefix());
        let bytes_written =
            storage::bin::save_to_bin(self, write_provider, sq_storage.compressed_data_path())?;
        let quantizer_bytes_written = sq_storage.save_quantizer(&self.quantizer, write_provider)?;
        Ok(bytes_written + quantizer_bytes_written)
    }
}

impl<const NBITS: usize> LoadWith<AsyncQuantLoadContext> for SQStore<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    type Error = ANNError;

    async fn load_with<P>(read_provider: &P, ctx: &AsyncQuantLoadContext) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        let sq_storage = storage::SQStorage::new(ctx.metadata.prefix());
        let quantizer = sq_storage.load_quantizer(read_provider)?;

        storage::bin::load_from_bin(
            read_provider,
            sq_storage.compressed_data_path(),
            |num_points, _pq_bytes| {
                Ok(SQStore::<NBITS>::new(
                    quantizer,
                    num_points,
                    ctx.metric,
                    ctx.prefetch_lookahead,
                ))
            },
        )
    }
}

/// Hook into [`storage::bin::load_from_bin`] by implementing [`storage::bin::SetData`].
impl<const NBITS: usize> storage::bin::SetData for SQStore<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    type Item = u8;

    fn set_data(&mut self, i: usize, element: &[Self::Item]) -> ANNResult<()> {
        // SAFETY: No race can happen because we have a mutable reference to `self`.
        unsafe { self.set_quant_vector(i, element) }
    }
}

/// Hook into [`storage::bin::save_to_bin`] by implementing [`storage::bin::GetData`].
impl<const NBITS: usize> storage::bin::GetData for SQStore<NBITS> {
    type Element = u8;
    type Item<'a> = &'a [u8];

    fn get_data(&self, i: usize) -> ANNResult<Self::Item<'_>> {
        // SAFETY: We aren't full protected against races on the underlying data, but at
        // least `&self` will keep the data alive.
        Ok(unsafe { self.data.get_slice(i) })
    }

    /// Return the total number of points, including frozen points.
    fn total(&self) -> usize {
        self.data.max_vectors()
    }

    fn dim(&self) -> usize {
        self.data.dim()
    }
}

// SQError is defined in diskann-providers::storage::sq_storage and re-exported
// from diskann-inmem for backward compatibility.
pub use diskann_providers::storage::SQError;

#[cfg(test)]
mod tests {
    use diskann_providers::storage::VirtualStorageProvider;
    use diskann::utils::ONE;
    use diskann_quantization::scalar::train::ScalarQuantizationParameters;
    use diskann_utils::views::MatrixView;
    use diskann_vector::distance::Metric;
    use rstest::rstest;
    use vfs::MemoryFS;

    use super::*;

    const NBITS: usize = 1;
    const DIM: usize = 4;
    const NPTS: usize = 5;
    const DATA: [f32; 20] = [
        0.286541, -0.079761, 0.373634, 0.878595, -0.131049, -0.131040, 0.883841, 0.429512,
        -0.482576, 0.557701, -0.476350, -0.478727, 0.091383, -0.722600, -0.651460, -0.212363,
        -0.510018, 0.158241, -0.457242, -0.711176,
    ];
    const V: [f32; DIM] = [DATA[0], DATA[1], DATA[2], DATA[3]];

    fn make_store(metric: Metric) -> SQStore<NBITS> {
        let quantizer = ScalarQuantizationParameters::default()
            .train(MatrixView::try_from(&DATA, NPTS, DIM).unwrap());
        SQStore::new(quantizer, /* capacity */ 5, metric, None)
    }

    #[test]
    fn test_dim() {
        let store = make_store(Metric::L2);
        assert_eq!(store.dim(), DIM);
    }

    #[test]
    fn test_set_and_get_vector() {
        let store = make_store(Metric::L2);
        store.set_vector(0, &V).unwrap();
        store.get_vector(0).unwrap();
        // `set` and `get` should not panic
    }

    #[test]
    #[should_panic]
    fn test_set_vector_wrong_dim_panic_in_debug() {
        let store = make_store(Metric::L2);
        let _: Result<_, SQError> = store.set_vector(0, &[1.0f32; DIM + 1]);
    }

    #[test]
    #[should_panic]
    fn test_get_vector_oob() {
        let store = make_store(Metric::L2);
        let _: Result<_, SQError> = store.get_vector(NPTS);
    }

    #[test]
    fn test_prefetch_hint_ok() {
        let store = make_store(Metric::L2);
        store.prefetch_hint(NPTS - 1);
    }

    #[test]
    #[should_panic]
    fn test_prefetch_hint_oob() {
        let store = make_store(Metric::L2);
        store.prefetch_hint(NPTS);
    }

    #[test]
    fn test_distance_computer_variants() {
        let dc_l2 = make_store(Metric::L2).distance_computer().unwrap();
        match dc_l2 {
            DistanceComputer::SquaredL2(_) => {}
            _ => panic!("expected SquaredL2 variant"),
        }

        let dc_ip = make_store(Metric::InnerProduct)
            .distance_computer()
            .unwrap();
        match dc_ip {
            DistanceComputer::InnerProduct(_) => {}
            _ => panic!("expected InnerProduct variant"),
        }

        let dc_cosine_normalized = make_store(Metric::CosineNormalized)
            .distance_computer()
            .unwrap();
        match dc_cosine_normalized {
            DistanceComputer::CosineNormalized(_) => {}
            _ => panic!("expected CosineNormalized variant"),
        }

        let dc_unsupported = make_store(Metric::Cosine).distance_computer().unwrap_err();
        match dc_unsupported {
            SQError::UnsupportedDistanceMetric(Metric::Cosine) => {}
            _ => panic!("expected UnsupportedDistanceMetric error"),
        }
    }

    #[rstest]
    fn test_query_computer(
        #[values(Metric::L2, Metric::InnerProduct, Metric::CosineNormalized)] metric: Metric,
        #[values(false, true)] allow_rescale: bool,
    ) {
        let store = make_store(metric);
        let q = [1.0_f32; DIM];
        let result = store.query_computer(&q, allow_rescale);
        assert!(
            result.is_ok(),
            "query_computer() failed for metric {:?} with allow_rescale={}",
            metric,
            allow_rescale
        );
    }

    #[test]
    fn test_set_quant_vector() {
        let store = make_store(Metric::L2);
        let compressed_vec_len = CVRef::<NBITS>::canonical_bytes(DIM);
        let raw = vec![1u8; compressed_vec_len];

        unsafe {
            store.set_quant_vector(0, &raw).unwrap();
        }

        // read back the same bytes
        let slice = unsafe { store.data.get_slice(0) };
        assert_eq!(slice, raw.as_slice());
    }

    #[test]
    #[should_panic]
    fn test_set_quant_vector_with_wrong_dim_panics() {
        let store = make_store(Metric::L2);
        let wrong_compressed_vec_len = CVRef::<NBITS>::canonical_bytes(DIM) + 1;
        let raw = vec![1u8; wrong_compressed_vec_len];

        unsafe {
            store.set_quant_vector(0, &raw).unwrap();
        }
    }

    #[rstest]
    fn test_distance_computer_cosine_normalized(
        #[values(Metric::L2, Metric::InnerProduct, Metric::CosineNormalized)] metric: Metric,
    ) {
        let store = make_store(metric);
        // Set two vectors
        let v1 = [0.1, 0.2, 0.3, 0.4];
        let v2 = [0.4, 0.3, 0.2, 0.1];
        store.set_vector(0, &v1).unwrap();
        store.set_vector(1, &v2).unwrap();

        let dc = store.distance_computer().unwrap();
        let x = store.get_vector(0).unwrap();
        let y = store.get_vector(1).unwrap();

        // This will exercise the CosineNormalized match arm
        let _ = dc.evaluate_similarity(x, y);
    }

    #[tokio::test]
    async fn test_save_with_and_load_with() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::default());
        let store = make_store(Metric::InnerProduct);

        // Save to our memory provider
        let prefix = "/test";
        let metadata = AsyncIndexMetadata::new(prefix.to_string());
        let bytes_written = store.save_with(&storage_provider, &metadata).await.unwrap();
        let sq_storage = storage::SQStorage::new(prefix);
        assert!(bytes_written > 0);
        assert!(storage_provider.exists(sq_storage.compressed_data_path()),);
        assert!(storage_provider.exists(sq_storage.quantizer_path()));

        // Load back from the same provider
        let ctx = AsyncQuantLoadContext {
            metadata,
            num_frozen_points: ONE,
            metric: Metric::InnerProduct,
            prefetch_lookahead: None,
            is_disk_index: false,
            prefetch_cache_line_level: None,
        };

        let loaded = SQStore::<NBITS>::load_with(&storage_provider, &ctx)
            .await
            .unwrap();

        // verify dimension is preserved
        assert_eq!(loaded.dim(), store.dim());
        // verify the raw bytes round-trip correctly
        for i in 0..NPTS {
            let original = unsafe { store.data.get_slice(i) };
            let loaded = unsafe { loaded.data.get_slice(i) };
            assert_eq!(original, loaded);
        }
    }
}
