/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A provider/accessor interface for spherical quantization.

use std::{future::Future, sync::Mutex};

use diskann::{
    ANNError, ANNErrorKind, ANNResult,
    error::IntoANNResult,
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
    alloc::{GlobalAllocator, ScopedAllocator},
    meta::NotCanonical,
    spherical,
};
use diskann_utils::future::AsyncFriendly;
use diskann_vector::distance::Metric;
use thiserror::Error;

use super::{GetFullPrecision, Rerank};
use diskann_providers::{
    common::IgnoreLockPoison,
    model::graph::{
        provider::async_::{
            FastMemoryVectorProviderAsync, SimpleNeighborProviderAsync,
            common::{
                AlignedMemoryVectorStore, CreateVectorStore, NoStore, SetElementHelper,
                TestCallCount, VectorStore,
            },
            distances::UnwrapErr,
            postprocess::{AsDeletionCheck, DeletionCheck, RemoveDeletedIdsAndCopy},
        },
        traits::AdHoc,
    },
    utils::{Bridge, BridgeErr},
};

use crate::{DefaultProvider, FullPrecisionProvider, FullPrecisionStore};

/////////////////////
// Error Promotion //
/////////////////////

impl From<Bridge<QueryComputerError>> for ANNError {
    #[track_caller]
    fn from(err: Bridge<QueryComputerError>) -> Self {
        ANNError::new(ANNErrorKind::SQError, err)
    }
}

impl From<Bridge<diskann_quantization::spherical::CompressionError>> for ANNError {
    #[track_caller]
    fn from(err: Bridge<diskann_quantization::spherical::CompressionError>) -> Self {
        ANNError::new(ANNErrorKind::SQError, err)
    }
}

impl From<Bridge<spherical::UnsupportedMetric>> for ANNError {
    #[track_caller]
    fn from(err: Bridge<spherical::UnsupportedMetric>) -> Self {
        ANNError::new(ANNErrorKind::SQError, err)
    }
}

/// An allocator error scoped to the spherical store.
#[derive(Debug, Clone, Copy, Error)]
#[error(transparent)]
pub struct AllocatorError(#[from] diskann_quantization::alloc::AllocatorError);

impl From<AllocatorError> for ANNError {
    #[track_caller]
    fn from(err: AllocatorError) -> Self {
        ANNError::new(ANNErrorKind::SQError, err)
    }
}

///////////
// Error //
//////////

#[derive(Debug, Error)]
pub enum QueryComputerError {
    #[error("Quantizer computer error : {0}")]
    QuantizerComputerError(#[from] spherical::iface::QueryComputerError),
    #[error("Error in converting query to full precision : {0}")]
    FullPrecisionConversionErr(#[from] Box<dyn std::error::Error + Send + Sync>),
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

pub struct SphericalStore {
    data: AlignedMemoryVectorStore<u8>,
    plan: Box<dyn spherical::iface::Quantizer + Send + Sync>,

    // We keep only write locks as reads are unsynchronized. Since there are
    // only writers, we use Mutex here. Note that sync::Mutex is ok here
    // because the Mutex is never held across an await.
    write_locks: Vec<Mutex<()>>,

    /// Prefetching for spherical bulk operations.
    prefetch_lookahead: usize,

    num_get_calls: TestCallCount,
}

impl SphericalStore {
    pub(super) fn new<P>(plan: P, num_vectors: usize, prefetch_lookahead: Option<usize>) -> Self
    where
        P: spherical::iface::Quantizer + AsyncFriendly,
    {
        let write_locks = (0..num_vectors.div_ceil(WRITE_LOCK_GRANULARITY))
            .map(|_| Mutex::new(()))
            .collect::<Vec<_>>();
        // Compute the number of bytes needed to hold the data and the compensation
        // coefficient.
        let bytes = plan.bytes();
        Self {
            data: AlignedMemoryVectorStore::with_capacity(num_vectors, bytes),
            plan: Box::new(plan),
            write_locks,
            prefetch_lookahead: prefetch_lookahead.unwrap_or(PREFETCH_DEFAULT),
            num_get_calls: TestCallCount::default(),
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

    #[cfg(test)]
    pub(super) fn input_dim(&self) -> usize {
        self.plan.dim()
    }

    pub fn bytes(&self) -> usize {
        self.plan.bytes()
    }

    pub fn output_dim(&self) -> usize {
        self.plan.dim()
    }

    pub(super) fn get_vector(&self, i: usize) -> Result<spherical::iface::Opaque<'_>, RQError> {
        self.num_get_calls.increment();
        // SAFETY: We can tolerate some racing behavior on the data behind this slice.
        let data = unsafe { self.data.get_slice(i) };
        Ok(spherical::iface::Opaque::new(data))
    }

    pub(super) fn set_vector<T>(&self, i: usize, v: &[T]) -> Result<(), RQError>
    where
        T: VectorRepr,
    {
        let _guard = self.write_locks[i / WRITE_LOCK_GRANULARITY].lock_or_panic();

        // SAFETY: We have exclusive write access to this slice.
        // Readers of this slice (since it's composed of just plain old data), can tolerate
        // some amount of racing.
        let data_mut = unsafe { self.data.get_mut_slice(i) };
        let vf32 = T::as_f32(v).map_err(|e| RQError::FullPrecisionConversionErr(Box::new(e)))?;
        self.plan.compress(
            &vf32,
            spherical::iface::OpaqueMut::new(data_mut),
            ScopedAllocator::global(),
        )?;
        Ok(())
    }

    pub(super) fn distance_computer(
        &self,
    ) -> Result<spherical::iface::DistanceComputer, AllocatorError> {
        Ok(self.plan.distance_computer(GlobalAllocator)?)
    }

    pub(super) fn query_computer<T>(
        &self,
        query: &[T],
        layout: spherical::iface::QueryLayout,
        allow_rescale: bool,
    ) -> Result<spherical::iface::QueryComputer, QueryComputerError>
    where
        T: VectorRepr,
    {
        let qf32 = T::as_f32(query)
            .map_err(|e| QueryComputerError::FullPrecisionConversionErr(Box::new(e)))?;

        let computer = self.plan.fused_query_computer(
            qf32.as_ref(),
            layout,
            allow_rescale,
            GlobalAllocator,
            ScopedAllocator::global(),
        )?;

        Ok(computer)
    }

    pub fn prefetch_lookahead(&self) -> usize {
        self.prefetch_lookahead
    }
}

impl VectorStore for SphericalStore {
    fn total(&self) -> usize {
        self.data.max_vectors()
    }

    fn count_for_get_vector(&self) -> usize {
        self.num_get_calls.get()
    }
}

///////////////////
// Data Provider //
///////////////////

macro_rules! create_vector_store {
    ($N:literal) => {
        impl CreateVectorStore for spherical::iface::Impl<$N> {
            type Target = SphericalStore;

            fn create(
                self,
                max_points: usize,
                metric: Metric,
                prefetch_lookahead: Option<usize>,
            ) -> Self::Target {
                assert_eq!(self.quantizer().metric(), metric, "mismatched metrics!");
                SphericalStore::new(self, max_points, prefetch_lookahead)
            }
        }
    };
    ($N:literal, $($Ns:literal),*) => {
        create_vector_store!($N);
        $(create_vector_store!($Ns);)*
    };
}

create_vector_store!(1, 2, 4);

////////////////
// SetElement //
////////////////

/// Assign to SQ vector store.
impl<T> SetElementHelper<T> for SphericalStore
where
    T: VectorRepr,
{
    fn set_element(&self, id: &u32, element: &[T]) -> ANNResult<()> {
        self.set_vector(id.into_usize(), element)?;
        Ok(())
    }
}

//////////////
// Accessor //
//////////////

pub struct QuantAccessor<'a, V, D, Ctx> {
    provider: &'a DefaultProvider<V, SphericalStore, D, Ctx>,
    id_buffer: Vec<u32>,
    layout: spherical::iface::QueryLayout,
    is_search: bool,
}

impl<'a, V, D, Ctx> QuantAccessor<'a, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    pub(crate) fn new(
        provider: &'a DefaultProvider<V, SphericalStore, D, Ctx>,
        layout: spherical::iface::QueryLayout,
        is_search: bool,
    ) -> Self {
        Self {
            provider,
            id_buffer: Vec::with_capacity(32),
            layout,
            is_search,
        }
    }
}

impl<T, D, Ctx> GetFullPrecision for QuantAccessor<'_, FullPrecisionStore<T>, D, Ctx>
where
    T: VectorRepr,
{
    type Repr = T;
    fn as_full_precision(&self) -> &FastMemoryVectorProviderAsync<AdHoc<T>> {
        &self.provider.base_vectors
    }
}

impl<V, D, Ctx> HasId for QuantAccessor<'_, V, D, Ctx> {
    type Id = u32;
}

impl<V, D, Ctx> SearchExt for QuantAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }
}

impl<'a, V, D, Ctx> Accessor for QuantAccessor<'a, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    /// The extended element inherets the lifetime of the Accessor.
    type Extended = spherical::iface::Opaque<'a>;

    /// This accessor returns raw slices. There *is* a chance of racing when the fast
    /// providers are used. We just have to live with it.
    ///
    /// NOTE: We intentionally don't use `'b` here since our implementation borrows
    /// the inner `Opaque` from the underlying provider.
    type Element<'b>
        = spherical::iface::Opaque<'a>
    where
        Self: 'b;

    /// `ElementRef` has an arbitrarily short lifetime.
    type ElementRef<'b> = spherical::iface::Opaque<'b>;

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
            self.provider
                .aux_vectors
                .get_vector(id.into_usize())
                .into_ann_result(),
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

            f(vector, *id)
        }

        std::future::ready(Ok(()))
    }
}

impl<'a, V, D, Ctx> DelegateNeighbor<'a> for QuantAccessor<'_, V, D, Ctx>
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

impl<V, D, Ctx, T> BuildQueryComputer<[T]> for QuantAccessor<'_, V, D, Ctx>
where
    T: VectorRepr,
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type QueryComputerError = Bridge<QueryComputerError>;
    type QueryComputer =
        UnwrapErr<spherical::iface::QueryComputer, spherical::iface::QueryDistanceError>;

    fn build_query_computer(
        &self,
        query: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.provider
            .aux_vectors
            .query_computer(query, self.layout, self.is_search)
            .bridge_err()
            .map(UnwrapErr::new)
    }
}

impl<V, D, Ctx, T> ExpandBeam<[T]> for QuantAccessor<'_, V, D, Ctx>
where
    T: VectorRepr,
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
}

#[derive(Debug, Error)]
#[error("unconstructible")]
pub enum Infallible {}

impl From<Infallible> for ANNError {
    fn from(_: Infallible) -> Self {
        unreachable!("Infallible is an unconstructible enum")
    }
}

impl<V, D, Ctx> BuildDistanceComputer for QuantAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type DistanceComputerError = AllocatorError;
    type DistanceComputer =
        UnwrapErr<spherical::iface::DistanceComputer, spherical::iface::DistanceError>;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        self.provider
            .aux_vectors
            .distance_computer()
            .map(UnwrapErr::new)
    }
}

impl<V, D, Ctx> AsDeletionCheck for QuantAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type Checker = D;
    fn as_deletion_check(&self) -> &D {
        &self.provider.deleted
    }
}

////////////////
// Strategies //
////////////////

/// Unlike [`super::Quantized`], searches over a [`SphericalStore`] support different
/// [`spherical::iface::QueryLayout`]s. This strategy type allows the user to specify the
/// desired layout explicitly.
#[derive(Debug, Clone, Copy)]
pub struct Quantized {
    layout: spherical::iface::QueryLayout,
    is_search: bool,
}

impl Quantized {
    /// Construct a new [`Quantized`] strategy for index construction.
    ///
    /// The layout used by strategy is [`spherical::iface::QueryLayout::SameAsData`].
    pub fn build() -> Self {
        Self {
            layout: spherical::iface::QueryLayout::SameAsData,
            is_search: false,
        }
    }

    /// Construct a new [`QuantizedStrategy`] for index search using the specified layout.
    pub fn search(layout: spherical::iface::QueryLayout) -> Self {
        Self {
            layout,
            is_search: true,
        }
    }
}

/// SearchStrategy for quantized search when a full-precision store exists alongside
/// the quantized store. This allows reranking using original vectors after
/// approximate search, so the post-processing step includes a [`Rerank`] stage.
impl<D, Ctx, T> SearchStrategy<FullPrecisionProvider<T, SphericalStore, D, Ctx>, [T]> for Quantized
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type QueryComputer =
        UnwrapErr<spherical::iface::QueryComputer, spherical::iface::QueryDistanceError>;
    type SearchAccessor<'a> = QuantAccessor<'a, FullPrecisionStore<T>, D, Ctx>;
    type SearchAccessorError = ANNError;
    type PostProcessor = glue::Pipeline<FilterStartPoints, Rerank>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, SphericalStore, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider, self.layout, self.is_search))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

/// SearchStrategy for quantized search when only the quantized store is present.
/// Since no full-precision vectors exist, reranking is not possible and the
/// post-processing step just copies candidate IDs forward via [`RemoveDeletedIdsAndCopy`].
impl<D, Ctx, T> SearchStrategy<DefaultProvider<NoStore, SphericalStore, D, Ctx>, [T]> for Quantized
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type QueryComputer =
        UnwrapErr<spherical::iface::QueryComputer, spherical::iface::QueryDistanceError>;
    type SearchAccessor<'a> = QuantAccessor<'a, NoStore, D, Ctx>;
    type SearchAccessorError = ANNError;
    type PostProcessor = glue::Pipeline<FilterStartPoints, RemoveDeletedIdsAndCopy>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DefaultProvider<NoStore, SphericalStore, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider, self.layout, self.is_search))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<V, D, Ctx> PruneStrategy<DefaultProvider<V, SphericalStore, D, Ctx>> for Quantized
where
    V: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type DistanceComputer =
        UnwrapErr<spherical::iface::DistanceComputer, spherical::iface::DistanceError>;
    type PruneAccessor<'a> = QuantAccessor<'a, V, D, Ctx>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DefaultProvider<V, SphericalStore, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        let build = Self::build();
        Ok(QuantAccessor::new(provider, build.layout, build.is_search))
    }
}

impl<V, D, Ctx> FillSet for QuantAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
}

impl<V, D, Ctx, T> InsertStrategy<DefaultProvider<V, SphericalStore, D, Ctx>, [T]> for Quantized
where
    V: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
    Quantized: SearchStrategy<DefaultProvider<V, SphericalStore, D, Ctx>, [T]>,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

#[derive(Debug, Error)]
pub enum RQError {
    #[error("Issue with canonical layout of data: {0:?}")]
    CanonicalLayoutError(#[from] NotCanonical),

    #[error("error during data compression")]
    CompressionError(#[from] spherical::iface::CompressionError),

    #[error("Error during full-precision conversion: {0}")]
    FullPrecisionConversionErr(Box<dyn std::error::Error + Send + Sync>),
}

impl From<RQError> for ANNError {
    #[cold]
    #[track_caller]
    fn from(err: RQError) -> Self {
        ANNError::log_sq_error(err)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_quantization::{
        alloc::GlobalAllocator,
        spherical::{SphericalQuantizer, SupportedMetric},
    };
    use diskann_utils::views::{Matrix, MatrixView};
    use diskann_vector::{
        DistanceFunction, PreprocessedDistanceFunction, PureDistanceFunction,
        distance::{InnerProduct, Metric, SquaredL2},
    };
    use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    use rand_distr::StandardNormal;

    use super::*;

    ////////////////
    // Test Store //
    ////////////////

    fn make_store<const NBITS: usize>(
        data: MatrixView<f32>,
        metric: SupportedMetric,
        rng: &mut StdRng,
    ) -> SphericalStore
    where
        spherical::iface::Impl<NBITS>:
            spherical::iface::Quantizer + spherical::iface::Constructible,
    {
        let quantizer = SphericalQuantizer::train(
            data,
            diskann_quantization::algorithms::transforms::TransformKind::PaddingHadamard {
                target_dim: diskann_quantization::algorithms::transforms::TargetDim::Natural,
            },
            metric,
            diskann_quantization::spherical::PreScale::None,
            rng,
            GlobalAllocator,
        )
        .unwrap();

        SphericalStore::new(
            spherical::iface::Impl::<NBITS>::new(quantizer).unwrap(),
            data.nrows(),
            None,
        )
    }

    fn dataset(nrows: usize, ncols: usize, rng: &mut StdRng) -> Matrix<f32> {
        Matrix::new(
            diskann_utils::views::Init(|| StandardNormal {}.sample(rng)),
            nrows,
            ncols,
        )
    }

    #[test]
    fn test_dim() {
        let mut rng = StdRng::seed_from_u64(0x721e3de995bc908c);
        let data = test_dataset();
        let store = make_store::<1>(data.as_view(), Metric::L2.try_into().unwrap(), &mut rng);

        assert_eq!(store.input_dim(), data.ncols());
    }

    #[test]
    fn test_set_and_get_vector() {
        let input_dim = 30;
        let mut rng = StdRng::seed_from_u64(0x721e3de995bc908c);
        let data = dataset(5, input_dim, &mut rng);
        let store = make_store::<1>(data.as_view(), SupportedMetric::SquaredL2, &mut rng);

        // Initial vectors should be all zero.
        {
            let v = store.get_vector(0).unwrap().into_inner();
            assert_eq!(v.len(), store.plan.bytes());
            for i in v.iter() {
                assert_eq!(*i, 0)
            }
        }

        // Compress and store the new values.
        store.set_vector(0, data.row(0)).unwrap();

        {
            let v = store.get_vector(0).unwrap().into_inner();
            assert_eq!(v.len(), store.plan.bytes());
            let mut nonzero_count = 0;
            for i in v.iter() {
                if *i != 0 {
                    nonzero_count += 1;
                }
            }
            assert_eq!(
                nonzero_count,
                9 * v.len() / 10,
                "Expected roughly 90% of the vectors to be nonzero. Instead, on {} of {} wer.",
                nonzero_count,
                v.len(),
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_get_vector_oob() {
        let input_dim = 5;
        let mut rng = StdRng::seed_from_u64(0x5460899c60762fed);
        let data = dataset(5, input_dim, &mut rng);
        let store = make_store::<1>(data.as_view(), SupportedMetric::SquaredL2, &mut rng);

        let _ = store.get_vector(data.nrows());
    }

    #[test]
    fn test_prefetch_hint_ok() {
        let input_dim = 5;
        let mut rng = StdRng::seed_from_u64(0x40f06c4034892796);
        let data = dataset(5, input_dim, &mut rng);
        let store = make_store::<1>(data.as_view(), SupportedMetric::InnerProduct, &mut rng);

        for i in 0..data.nrows() {
            store.prefetch_hint(i)
        }
    }

    #[test]
    #[should_panic]
    fn test_prefetch_hint_oob() {
        let input_dim = 5;
        let mut rng = StdRng::seed_from_u64(0x5abccaa184dfdb93);
        let data = dataset(5, input_dim, &mut rng);
        let store = make_store::<1>(data.as_view(), SupportedMetric::InnerProduct, &mut rng);

        store.prefetch_hint(data.nrows());
    }

    fn relative_error(got: f32, expected: f32) -> f32 {
        if expected == 0.0 {
            return if got == 0.0 { 0.0 } else { f32::INFINITY };
        }

        (got - expected).abs() / expected.abs()
    }

    #[test]
    fn test_distance_computer_variants() {
        let input_dim = 128;
        let mut rng = StdRng::seed_from_u64(0x5abccaa184dfdb93);
        let data = dataset(10, input_dim, &mut rng);

        // Squared L2
        //
        // This is just a sanity check on the computers.
        {
            let store = make_store::<1>(data.as_view(), SupportedMetric::SquaredL2, &mut rng);
            let computer = store.distance_computer().unwrap();

            let max_relative_error = 0.25;

            for (i, r) in data.row_iter().enumerate() {
                store.set_vector(i, r).unwrap();
            }

            for (i, a) in data.row_iter().enumerate() {
                for (j, b) in data.row_iter().enumerate().skip(i + 1) {
                    let expected: f32 = SquaredL2::evaluate(a, b);
                    let got: f32 = computer
                        .evaluate_similarity(
                            store.get_vector(i).unwrap(),
                            store.get_vector(j).unwrap(),
                        )
                        .unwrap();
                    let err = relative_error(got, expected);
                    assert!(
                        err < max_relative_error,
                        "expected a relative error less than {}, instead found {} for\
                        expected = {}, got = {}",
                        max_relative_error,
                        err,
                        expected,
                        got
                    );
                }
            }
        }

        // Inner Product
        //
        // This is just a sanity check on the computers. Since the bit-level dot product
        // is pretty inaccurate, we just check for matching signs at this level.
        {
            let store = make_store::<1>(data.as_view(), SupportedMetric::InnerProduct, &mut rng);
            let computer = store.distance_computer().unwrap();

            for (i, r) in data.row_iter().enumerate() {
                store.set_vector(i, r).unwrap();
            }

            let mut signs_match = 0;
            let mut total = 0;

            for (i, a) in data.row_iter().enumerate() {
                for (j, b) in data.row_iter().enumerate().skip(i + 1) {
                    total += 1;
                    let expected: f32 = InnerProduct::evaluate(a, b);
                    let got: f32 = computer
                        .evaluate_similarity(
                            store.get_vector(i).unwrap(),
                            store.get_vector(j).unwrap(),
                        )
                        .unwrap();

                    if expected.is_sign_negative() == got.is_sign_negative() {
                        signs_match += 1;
                    }
                }
            }
            assert!(
                signs_match > 8 * total / 10,
                "expected 80% of the inner-product signs to match. Instead got {} of {}.",
                signs_match,
                total,
            );
        }
    }

    #[test]
    fn test_query_computer_variants() {
        let input_dim = 128;
        let mut rng = StdRng::seed_from_u64(0x5abccaa184dfdb93);
        let data = dataset(10, input_dim, &mut rng);

        // Squared L2
        //
        // This is just a sanity check on the computers.
        {
            let store = make_store::<1>(data.as_view(), SupportedMetric::SquaredL2, &mut rng);
            let max_relative_error = 0.2;

            for (i, r) in data.row_iter().enumerate() {
                store.set_vector(i, r).unwrap();
            }

            for (i, a) in data.row_iter().enumerate() {
                let computer = store
                    .query_computer(a, spherical::iface::QueryLayout::FourBitTransposed, false)
                    .unwrap();
                for (j, b) in data.row_iter().enumerate() {
                    if i == j {
                        continue;
                    }

                    let expected: f32 = SquaredL2::evaluate(a, b);
                    let got: f32 = computer
                        .evaluate_similarity(store.get_vector(j).unwrap())
                        .unwrap();

                    let err = relative_error(got, expected);
                    assert!(
                        err < max_relative_error,
                        "expected a relative error less than {}, instead found {}.\
                        expected = {}, got = {} for pair (i, j) = ({}, {})",
                        max_relative_error,
                        err,
                        expected,
                        got,
                        i,
                        j,
                    );
                }
            }
        }

        // Inner Product
        //
        // This is just a sanity check on the computers. Since the bit-level dot product
        // is pretty inaccurate, we just check for matching signs at this level.
        {
            let store = make_store::<1>(data.as_view(), SupportedMetric::InnerProduct, &mut rng);

            for (i, r) in data.row_iter().enumerate() {
                store.set_vector(i, r).unwrap();
            }

            let mut signs_match = 0;
            let mut total = 0;

            for (i, a) in data.row_iter().enumerate() {
                let computer = store
                    .query_computer(a, spherical::iface::QueryLayout::FourBitTransposed, true)
                    .unwrap();
                for (j, b) in data.row_iter().enumerate() {
                    if i == j {
                        continue;
                    }

                    total += 1;
                    let expected: f32 = InnerProduct::evaluate(a, b);
                    let got: f32 = computer
                        .evaluate_similarity(store.get_vector(j).unwrap())
                        .unwrap();

                    if expected.is_sign_negative() == got.is_sign_negative() {
                        signs_match += 1;
                    }
                }
            }
            assert!(
                signs_match > 85 * total / 100,
                "expected 85% of the inner-product signs to match. Instead got {} of {}.",
                signs_match,
                total,
            );
        }
    }

    ////////////
    // Errors //
    ////////////

    #[test]
    fn test_compression_errors() {
        let input_dim = 10;
        let mut rng = StdRng::seed_from_u64(0x721e3de995bc908c);
        let data = dataset(5, input_dim, &mut rng);
        let store = make_store::<1>(data.as_view(), SupportedMetric::SquaredL2, &mut rng);

        let err = store
            .set_vector(0, &(data.row(0)[..input_dim - 10]))
            .unwrap_err();
        assert!(matches!(err, RQError::CompressionError(..)));
    }

    fn test_dataset() -> Matrix<f32> {
        let data = vec![
            0.28657,
            -0.0318168,
            0.0666847,
            0.0329265,
            -0.00829283,
            0.168735,
            -0.000846311,
            -0.360779, // row 0
            -0.0968938,
            0.161921,
            -0.0979579,
            0.102228,
            -0.259928,
            -0.139634,
            0.165384,
            -0.293443, // row 1
            0.130205,
            0.265737,
            0.401816,
            -0.407552,
            0.13012,
            -0.0475244,
            0.511723,
            -0.4372, // row 2
            -0.0979126,
            0.135861,
            -0.0154144,
            -0.14047,
            -0.0250029,
            -0.190279,
            0.407283,
            -0.389184, // row 3
            -0.264153,
            0.0696822,
            -0.145585,
            0.370284,
            0.186825,
            -0.140736,
            0.274703,
            -0.334563, // row 4
            0.247613,
            0.513165,
            -0.0845867,
            0.0532264,
            -0.00480601,
            -0.122408,
            0.47227,
            -0.268301, // row 5
            0.103198,
            0.30756,
            -0.316293,
            -0.0686877,
            -0.330729,
            -0.461997,
            0.550857,
            -0.240851, // row 6
            0.128258,
            0.786291,
            -0.0268103,
            0.111763,
            -0.308962,
            -0.17407,
            0.437154,
            -0.159879, // row 7
            0.00374063,
            0.490301,
            0.0327826,
            -0.0340962,
            -0.118605,
            0.163879,
            0.2737,
            -0.299942, // row 8
            -0.284077,
            0.249377,
            -0.0307734,
            -0.0661631,
            0.233854,
            0.427987,
            0.614132,
            -0.288649, // row 9
            -0.109492,
            0.203939,
            -0.73956,
            -0.130748,
            0.22072,
            0.0647836,
            0.328726,
            -0.374602, // row 10
            -0.223114,
            0.0243489,
            0.109195,
            -0.416914,
            0.0201052,
            -0.0190542,
            0.947078,
            -0.333229, // row 11
            -0.165869,
            -0.00296729,
            -0.414378,
            0.231321,
            0.205365,
            0.161761,
            0.148608,
            -0.395063, // row 12
            -0.0498255,
            0.193279,
            -0.110946,
            -0.181174,
            -0.274578,
            -0.227511,
            0.190208,
            -0.256174, // row 13
            -0.188106,
            -0.0292958,
            0.0930939,
            0.0558456,
            0.257437,
            0.685481,
            0.307922,
            -0.320006, // row 14
            0.250035,
            0.275942,
            -0.0856306,
            -0.352027,
            -0.103509,
            -0.00890859,
            0.276121,
            -0.324718, // row 15
        ];

        Matrix::try_from(data.into(), 16, 8).unwrap()
    }
}
