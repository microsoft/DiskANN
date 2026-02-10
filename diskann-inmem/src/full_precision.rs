/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashMap, fmt::Debug, future::Future};

use diskann::{
    ANNResult,
    graph::{
        SearchOutputBuffer,
        glue::{
            self, ExpandBeam, FillSet, FilterStartPoints, InplaceDeleteStrategy, InsertStrategy,
            PruneStrategy, SearchExt, SearchStrategy,
        },
    },
    neighbor::Neighbor,
    provider::{
        Accessor, BuildDistanceComputer, BuildQueryComputer, DefaultContext, DelegateNeighbor,
        ExecutionContext, HasId,
    },
    utils::{IntoUsize, VectorRepr},
};
use diskann_utils::future::AsyncFriendly;
use diskann_vector::{DistanceFunction, distance::Metric};

use crate::CreateVectorStore;
use diskann_providers::model::graph::{
    provider::async_::{
        FastMemoryVectorProviderAsync, SimpleNeighborProviderAsync,
        common::{
            FullPrecision, Internal, NoDeletes, NoStore, Panics,
            PrefetchCacheLineLevel,
        },
        postprocess::{AsDeletionCheck, DeletionCheck, RemoveDeletedIdsAndCopy},
    },
    traits::AdHoc,
};
use crate::DefaultProvider;

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

//////////////////
// FullAccessor //
//////////////////

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

impl<'a, T, Q, D, Ctx> Accessor for FullAccessor<'a, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    /// The extended element inherets the lifetime of the Accessor.
    type Extended = &'a [T];

    /// This accessor returns raw slices. There *is* a chance of racing when the fast
    /// providers are used. We just have to live with it.
    ///
    /// NOTE: We intentionally don't use `'b` here since our implementation borrows
    /// the inner `Opaque` from the underlying provider.
    type Element<'b>
        = &'a [T]
    where
        Self: 'b;

    /// `ElementRef` has an arbitrarily short lifetime.
    type ElementRef<'b> = &'b [T];

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
        // SAFETY: We've decided to live with UB (undefined behavior) that can result from
        // potentially mixing unsynchronized reads and writes on the underlying memory.
        std::future::ready(Ok(unsafe {
            self.provider.base_vectors.get_vector_sync(id.into_usize())
        }))
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
            self.provider.base_vectors.prefetch_hint(id.into_usize());
        }

        for (i, id) in id_buffer.iter().enumerate() {
            // Prefetch `lookahead` iterations ahead as long as it is safe.
            if lookahead > 0 && i + lookahead < len {
                self.provider
                    .base_vectors
                    .prefetch_hint(id_buffer[i + lookahead].into_usize());
            }

            // Invoke the passed closure on the full-precision vector.
            //
            // SAFETY: We're accepting the consequences of potential unsynchronized,
            // concurrent mutation.
            f(
                unsafe { self.provider.base_vectors.get_vector_sync(id.into_usize()) },
                *id,
            )
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

impl<T, Q, D, Ctx> BuildQueryComputer<[T]> for FullAccessor<'_, T, Q, D, Ctx>
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

impl<T, Q, D, Ctx> ExpandBeam<[T]> for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
}

impl<T, Q, D, Ctx> FillSet for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    async fn fill_set<Itr>(
        &mut self,
        set: &mut HashMap<Self::Id, Self::Extended>,
        itr: Itr,
    ) -> Result<(), Self::GetError>
    where
        Itr: Iterator<Item = Self::Id> + Send + Sync,
    {
        for i in itr {
            set.entry(i).or_insert_with(|| unsafe {
                self.provider.base_vectors.get_vector_sync(i.into_usize())
            });
        }
        Ok(())
    }
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

impl<A, T> glue::SearchPostProcess<A, [T]> for Rerank
where
    T: VectorRepr,
    A: BuildQueryComputer<[T], Id = u32> + GetFullPrecision<Repr = T> + AsDeletionCheck,
{
    type Error = Panics;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: &[T],
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
                    Some((
                        n.id,
                        f.evaluate_similarity(query, unsafe {
                            full.get_vector_sync(n.id.into_usize())
                        }),
                    ))
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

// A layered approach is used for search strategies. The `Internal` version does the heavy
// lifting in terms of establishing accessors and post processing.
//
// However, during post-processing, the `Internal` versions of strategies will not filter
// out the start points. The publicly exposed types *will* filter out the start points.
//
// This layered approach allows algorithms like `InplaceDeleteStrategy` that need to adjust
// the adjacency list for the start point to reuse the `Internal` strategies.

/// Perform a search entirely in the full-precision space.
///
/// Starting points are not filtered out of the final results.
impl<T, Q, D, Ctx> SearchStrategy<FullPrecisionProvider<T, Q, D, Ctx>, [T]>
    for Internal<FullPrecision>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type QueryComputer = T::QueryDistance;
    type SearchAccessor<'a> = FullAccessor<'a, T, Q, D, Ctx>;
    type SearchAccessorError = Panics;
    type PostProcessor = RemoveDeletedIdsAndCopy;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, Q, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

/// Perform a search entirely in the full-precision space.
///
/// Starting points are not filtered out of the final results.
impl<T, Q, D, Ctx> SearchStrategy<FullPrecisionProvider<T, Q, D, Ctx>, [T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type QueryComputer = T::QueryDistance;
    type SearchAccessor<'a> = FullAccessor<'a, T, Q, D, Ctx>;
    type SearchAccessorError = Panics;
    type PostProcessor = glue::Pipeline<FilterStartPoints, RemoveDeletedIdsAndCopy>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, Q, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
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

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, Q, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(FullAccessor::new(provider))
    }
}

/// Implementing this trait allows `FullPrecision` to be used for multi-insert.
impl<'a, T, Q, D, Ctx> glue::AsElement<&'a [T]> for FullAccessor<'a, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type Error = diskann::error::Infallible;
    fn as_element(
        &mut self,
        vector: &'a [T],
        _id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'a>, Self::Error>> + Send {
        std::future::ready(Ok(vector))
    }
}

impl<T, Q, D, Ctx> InsertStrategy<FullPrecisionProvider<T, Q, D, Ctx>, [T]> for FullPrecision
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

// Inplace Delete //
impl<T, Q, D, Ctx> InplaceDeleteStrategy<FullPrecisionProvider<T, Q, D, Ctx>> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type DeleteElementError = Panics;
    type DeleteElement<'a> = [T];
    type DeleteElementGuard = Box<[T]>;
    type PruneStrategy = Self;
    type SearchStrategy = Internal<Self>;
    fn search_strategy(&self) -> Self::SearchStrategy {
        Internal(Self)
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        Self
    }

    async fn get_delete_element<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, Q, D, Ctx>,
        _context: &'a Ctx,
        id: u32,
    ) -> Result<Self::DeleteElementGuard, Self::DeleteElementError> {
        Ok(unsafe { provider.base_vectors.get_vector_sync(id.into_usize()) }.into())
    }
}
