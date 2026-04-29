/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, future::Future};

use diskann::default_post_processor;
use diskann::{
    ANNError, ANNResult,
    error::Infallible,
    graph::{
        SearchOutputBuffer,
        glue::{
            self, DefaultPostProcessor, ExpandBeam, InplaceDeleteStrategy, InsertStrategy,
            PruneStrategy, SearchExt, SearchStrategy,
        },
        workingset,
    },
    neighbor::Neighbor,
    provider::{
        Accessor, BuildDistanceComputer, BuildQueryComputer, DefaultContext, DelegateNeighbor,
        ExecutionContext, HasId,
    },
    utils::{IntoUsize, VectorRepr},
};

use diskann_utils::{arbiter, future::AsyncFriendly};
use diskann_vector::{DistanceFunction, distance::Metric};

use crate::model::graph::{
    provider::async_::{
        FastMemoryVectorProviderAsync, SimpleNeighborProviderAsync,
        common::{
            CreateVectorStore, FullPrecision, NoDeletes, NoStore, Panics, PrefetchCacheLineLevel,
            SetElementHelper,
        },
        inmem::{DefaultProvider, PassThrough},
        postprocess::{AsDeletionCheck, DeletionCheck, RemoveDeletedIdsAndCopy},
    },
    traits::AdHoc,
};

/// A type alias for the DefaultProvider with full-precision as the primary vector store.
pub type FullPrecisionProvider<T, Q = NoStore, D = NoDeletes, Ctx = DefaultContext> =
    DefaultProvider<FullPrecisionStore<T>, Q, D, Ctx>;

/// The default full-precision vector store.
pub type FullPrecisionStore<T> = FastMemoryVectorProviderAsync<AdHoc<T>>;

/// A default full-precision vector store provider.
#[derive(Clone)]
pub struct CreateFullPrecision<T: VectorRepr> {
    dim: usize,
    prefetch_cache_line_level: Option<PrefetchCacheLineLevel>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> CreateFullPrecision<T>
where
    T: VectorRepr,
{
    /// Create a new full-precision vector store provider.
    pub fn new(dim: usize, prefetch_cache_line_level: Option<PrefetchCacheLineLevel>) -> Self {
        Self {
            dim,
            prefetch_cache_line_level,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> CreateVectorStore for CreateFullPrecision<T>
where
    T: VectorRepr,
{
    type Target = FullPrecisionStore<T>;
    fn create(
        self,
        max_points: usize,
        metric: Metric,
        prefetch_lookahead: Option<usize>,
    ) -> Self::Target {
        FullPrecisionStore::new(
            max_points,
            self.dim,
            metric,
            self.prefetch_cache_line_level,
            prefetch_lookahead,
        )
    }
}

////////////////
// SetElement //
////////////////

impl<T> SetElementHelper<T> for FullPrecisionStore<T>
where
    T: VectorRepr,
{
    /// Set the element at the given index.
    fn set_element(&self, id: &u32, element: &[T]) -> Result<(), ANNError> {
        unsafe { self.set_vector_sync(id.into_usize(), element) }
    }
}

//////////////////
// FullAccessor //
//////////////////

#[derive(Debug)]
struct Reader<'a, T> {
    inner: arbiter::store::Reader<'a>,
    bytes: usize,
    _type: std::marker::PhantomData<T>,
}

impl<'a, T> Reader<'a, T>
where
    T: bytemuck::Pod,
{
    #[inline(always)]
    fn read(&self, i: usize) -> &[T] {
        let slice = self.inner.read(i).unwrap();

        // SAFETY: The buffer is 128-byte aligned (satisfies any primitive T alignment),
        // self.bytes is always dim * size_of::<T>() and therefore T-aligned and
        // evenly divisible, and self.bytes <= stride (the slice length) by construction.
        let count = self.bytes / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const T, count) }
    }
}

/// An accessor for retrieving full-precision vectors from the `DefaultProvider`.
///
/// This type implements the following traits:
///
/// * [`Accessor`] for the [`DefaultProvider`].
/// * [`ComputerAccessor`] for comparing full-precision distances.
/// * [`BuildQueryComputer`].
pub struct FullAccessor<'a, T, Q, D, Ctx>
where
    T: VectorRepr,
{
    /// The host provider.
    provider: &'a FullPrecisionProvider<T, Q, D, Ctx>,
    reader: Reader<'a, T>,

    /// A buffer for resolving iterators given during bulk operations.
    ///
    /// The accessor reuses this allocation to amortize allocation cost over multiple bulk
    /// operations.
    id_buffer: Vec<u32>,
}

impl<T, Q, D, Ctx> GetFullPrecision for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
{
    type Repr = T;
    fn as_full_precision(&self) -> &FullPrecisionStore<T> {
        &self.provider.base_vectors
    }
}

impl<T, Q, D, Ctx> HasId for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
{
    type Id = u32;
}

impl<T, Q, D, Ctx> SearchExt for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }
}

impl<'a, T, Q, D, Ctx> FullAccessor<'a, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    pub fn new(provider: &'a FullPrecisionProvider<T, Q, D, Ctx>) -> Self {
        Self {
            provider,
            reader: Reader {
                inner: provider.base_vectors.store.reader(),
                bytes: std::mem::size_of::<T>() * provider.base_vectors.dim(),
                _type: std::marker::PhantomData,
            },
            id_buffer: Vec::new(),
        }
    }
}

impl<'a, T, Q, D, Ctx> DelegateNeighbor<'a> for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type Delegate = &'a SimpleNeighborProviderAsync<u32>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<T, Q, D, Ctx> Accessor for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    /// This accessor returns raw slices. There *is* a chance of racing when the fast
    /// providers are used. We just have to live with it.
    type Element<'a>
        = &'a [T]
    where
        Self: 'a;

    /// `ElementRef` has an arbitrarily short lifetime.
    type ElementRef<'a> = &'a [T];

    /// Choose to panic on an out-of-bounds access rather than propagate an error.
    type GetError = Panics;

    /// Return the full-precision vector stored at index `i`.
    ///
    /// This function always completes synchronously.
    #[inline(always)]
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        std::future::ready(Ok(self.reader.read(id.into_usize())))
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
        let lookahead = self.provider.base_vectors.prefetch_lookahead();

        // Prefetch the first few vectors.
        for id in id_buffer.iter().take(lookahead) {
            self.reader.inner.prefetch(id.into_usize())
            // self.provider.base_vectors.prefetch_hint(id.into_usize());
        }

        for (i, id) in id_buffer.iter().enumerate() {
            // Prefetch `lookahead` iterations ahead as long as it is safe.
            if lookahead > 0 && i + lookahead < len {
                self.reader.inner.prefetch(id_buffer[i + lookahead].into_usize());
            }
            f(self.reader.read(id.into_usize()), *id)
        }

        std::future::ready(Ok(()))
    }
}

impl<T, Q, D, Ctx> BuildDistanceComputer for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type DistanceComputerError = Panics;
    type DistanceComputer = T::Distance;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(T::distance(
            self.provider.metric,
            Some(self.provider.base_vectors.dim()),
        ))
    }
}

impl<T, Q, D, Ctx> BuildQueryComputer<&[T]> for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type QueryComputerError = Panics;
    type QueryComputer = T::QueryDistance;

    fn build_query_computer(
        &self,
        from: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(T::query_distance(from, self.provider.metric))
    }
}

impl<T, Q, D, Ctx> ExpandBeam<&[T]> for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
}

//-------------------//
// In-mem Extensions //
//-------------------//

impl<'a, T, Q, D, Ctx> AsDeletionCheck for FullAccessor<'a, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type Checker = D;
    fn as_deletion_check(&self) -> &D {
        &self.provider.deleted
    }
}

//////////////////
// Post Process //
//////////////////

pub trait GetFullPrecision {
    type Repr: VectorRepr;
    fn as_full_precision(&self) -> &FastMemoryVectorProviderAsync<AdHoc<Self::Repr>>;
}

/// A [`SearchPostProcess`]or that:
///
/// 1. Filters out deleted ids from being returned.
/// 2. Reranks a candidate stream using full-precision distances.
/// 3. Copies back the results to the output buffer.
#[derive(Debug, Default, Clone, Copy)]
pub struct Rerank;

impl<'a, A, T> glue::SearchPostProcess<A, &'a [T]> for Rerank
where
    T: VectorRepr,
    A: BuildQueryComputer<&'a [T], Id = u32> + GetFullPrecision<Repr = T> + AsDeletionCheck,
{
    type Error = Panics;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: &'a [T],
        _computer: &A::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>>,
        B: SearchOutputBuffer<u32> + ?Sized,
    {
        let full = accessor.as_full_precision();
        let checker = accessor.as_deletion_check();
        let f = full.distance();

        // Filter before computing the full precision distances.
        let mut reranked: Vec<(u32, f32)> = candidates
            .filter_map(|n| {
                if checker.deletion_check(n.id) {
                    None
                } else {
                    todo!();
                    // Some((
                    //     n.id,
                    //     f.evaluate_similarity(query, accessor.reader.read(n.id.into_usize()))
                    //     ))
                }
            })
            .collect();

        // Sort the full precision distances.
        reranked
            .sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Store the reranked results.
        std::future::ready(Ok(output.extend(reranked)))
    }
}

////////////////
// Strategies //
////////////////

/// Perform a search entirely in the full-precision space.
impl<T, Q, D, Ctx> SearchStrategy<FullPrecisionProvider<T, Q, D, Ctx>, &[T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type QueryComputer = T::QueryDistance;
    type SearchAccessor<'a> = FullAccessor<'a, T, Q, D, Ctx>;
    type SearchAccessorError = Panics;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, Q, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider))
    }
}

impl<T, Q, D, Ctx> DefaultPostProcessor<FullPrecisionProvider<T, Q, D, Ctx>, &[T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    default_post_processor!(glue::Pipeline<glue::FilterStartPoints, RemoveDeletedIdsAndCopy>);
}

// Pruning
impl<T, Q, D, Ctx> PruneStrategy<FullPrecisionProvider<T, Q, D, Ctx>> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type DistanceComputer = T::Distance;
    type PruneAccessor<'a> = FullAccessor<'a, T, Q, D, Ctx>;
    type PruneAccessorError = diskann::error::Infallible;
    type WorkingSet = PassThrough;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, Q, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(FullAccessor::new(provider))
    }

    fn create_working_set(&self, _capacity: usize) -> Self::WorkingSet {
        PassThrough
    }
}

// All this does is return a `&Self` - which directly accesses the underlying provider.
impl<T, Q, D, Ctx> workingset::Fill<PassThrough> for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type Error = Infallible;

    type View<'a>
        = &'a Self
    where
        Self: 'a;

    async fn fill<'a, Itr>(
        &'a mut self,
        _state: &'a mut PassThrough,
        _itr: Itr,
    ) -> Result<Self::View<'a>, Self::Error>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
        Self: 'a,
    {
        Ok(self)
    }
}

// Pass-through view.
impl<T, Q, D, Ctx> workingset::View<u32> for &FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type ElementRef<'a> = &'a [T];
    type Element<'a>
        = &'a [T]
    where
        Self: 'a;

    fn get(&self, id: u32) -> Option<&[T]> {
        Some(self.reader.read(id.into_usize()))
    }
}

impl<T, Q, D, Ctx> InsertStrategy<FullPrecisionProvider<T, Q, D, Ctx>, &[T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

impl<T, Q, D, Ctx, B> glue::MultiInsertStrategy<FullPrecisionProvider<T, Q, D, Ctx>, B>
    for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
    B: glue::Batch,
    Self: for<'a> InsertStrategy<
            FullPrecisionProvider<T, Q, D, Ctx>,
            B::Element<'a>,
            PruneStrategy = Self,
        >,
{
    type Seed = PassThrough;
    type WorkingSet = PassThrough;
    type FinishError = diskann::error::Infallible;
    type InsertStrategy = Self;

    fn insert_strategy(&self) -> Self::InsertStrategy {
        *self
    }

    fn finish<Itr>(
        &self,
        _provider: &FullPrecisionProvider<T, Q, D, Ctx>,
        _ctx: &Ctx,
        _batch: &std::sync::Arc<B>,
        _ids: Itr,
    ) -> impl std::future::Future<Output = Result<Self::Seed, Self::FinishError>> + Send
    where
        Itr: ExactSizeIterator<Item = u32> + Send,
    {
        std::future::ready(Ok(PassThrough))
    }
}

// Inplace Delete //
impl<T, Q, D, Ctx> InplaceDeleteStrategy<FullPrecisionProvider<T, Q, D, Ctx>> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type DeleteElementError = Panics;
    type DeleteElement<'a> = &'a [T];
    type DeleteElementGuard = Box<[T]>;
    type PruneStrategy = Self;
    type DeleteSearchAccessor<'a> = FullAccessor<'a, T, Q, D, Ctx>;
    type SearchPostProcessor = RemoveDeletedIdsAndCopy;
    type SearchStrategy = Self;
    fn search_strategy(&self) -> Self::SearchStrategy {
        *self
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        Self
    }

    fn search_post_processor(&self) -> Self::SearchPostProcessor {
        RemoveDeletedIdsAndCopy
    }

    async fn get_delete_element<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, Q, D, Ctx>,
        _context: &'a Ctx,
        id: u32,
    ) -> Result<Self::DeleteElementGuard, Self::DeleteElementError> {
        todo!()
        // Ok(unsafe { provider.base_vectors.get_vector_sync(id.into_usize()) }.into())
    }
}
