/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashMap, future::Future, sync::Arc};

use diskann::{
    ANNError, ANNResult,
    graph::glue::{
        self, ExpandBeam, FillSet, FilterStartPoints, InplaceDeleteStrategy, InsertStrategy,
        PruneStrategy, SearchExt, SearchStrategy,
    },
    provider::{
        Accessor, BuildDistanceComputer, BuildQueryComputer, DelegateNeighbor, ExecutionContext,
        HasId,
    },
    utils::{IntoUsize, VectorRepr},
};
use diskann_utils::future::AsyncFriendly;
use diskann_vector::distance::Metric;

use crate::CreateVectorStore;
use diskann_providers::model::{
    graph::{
        provider::async_::{
            FastMemoryQuantVectorProviderAsync, FastMemoryVectorProviderAsync,
            SimpleNeighborProviderAsync,
            common::{
                Hybrid, Internal, NoStore, Panics, Quantized, SetElementHelper,
                VectorStore,
            },
            distances,
            postprocess::{AsDeletionCheck, DeletionCheck, RemoveDeletedIdsAndCopy},
        },
        traits::AdHoc,
    },
    pq::{self, FixedChunkPQTable},
};
use crate::{
    DefaultProvider, FullPrecisionProvider, FullPrecisionStore, GetFullPrecision,
    Rerank,
};

/// The default quant provider.
pub type DefaultQuant = FastMemoryQuantVectorProviderAsync;

impl CreateVectorStore for FixedChunkPQTable {
    type Target = DefaultQuant;
    fn create(
        self,
        max_points: usize,
        metric: Metric,
        _prefetch_lookahead: Option<usize>,
    ) -> Self::Target {
        DefaultQuant::new(metric, max_points, self)
    }
}

impl VectorStore for DefaultQuant {
    fn total(&self) -> usize {
        self.total()
    }

    fn count_for_get_vector(&self) -> usize {
        self.num_get_calls.get()
    }
}

////////////////
// SetElement //
////////////////

/// Assign to PQ vector store.
impl<T> SetElementHelper<T> for DefaultQuant
where
    T: VectorRepr,
{
    fn set_element(&self, id: &u32, element: &[T]) -> ANNResult<()> {
        unsafe { self.set_vector_sync(id.into_usize(), element) }
    }
}

///////////////////
// QuantAccessor //
///////////////////

/// An accessor that retrieves the quantized portion of the [`DefaultProvider`].
///
/// This type implements the following traits:
///
/// * [`Accessor`] for the `DefaultProvider`.
/// * [`BuildQueryComputer`].
pub struct QuantAccessor<'a, V, D, Ctx> {
    provider: &'a DefaultProvider<V, DefaultQuant, D, Ctx>,
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

impl<'a, V, D, Ctx> QuantAccessor<'a, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    pub(crate) fn new(provider: &'a DefaultProvider<V, DefaultQuant, D, Ctx>) -> Self {
        Self { provider }
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

impl<'a, V, D, Ctx> Accessor for QuantAccessor<'a, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    /// The extended element inherets the lifetime of the Accessor.
    type Extended = &'a [u8];

    /// This accessor returns raw slices. There *is* a chance of racing when the fast
    /// providers are used. We just have to live with it.
    ///
    /// NOTE: We intentionally don't use `'b` here since our implementation borrows
    /// the inner from the underlying provider.
    type Element<'b>
        = &'a [u8]
    where
        Self: 'b;

    /// `ElementRef` has an arbitrarily short lifetime.
    type ElementRef<'b> = &'b [u8];

    /// Choose to panic on an out-of-bounds access rather than propagate an error.
    type GetError = Panics;

    /// Return the quantized vector stored at index `i`.
    ///
    /// This function always completes synchronously.
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        // SAFETY: We've decided to live with UB that can result from potentially mixing
        // unsynchronized reads and writes on the underlying memory.
        std::future::ready(Ok(unsafe {
            self.provider.aux_vectors.get_vector_sync(id.into_usize())
        }))
    }

    /// Perform a bulk operation.
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
        for i in itr {
            // SAFETY: We're accepting the consequences of potential unsynchronized,
            // concurrent mutation.
            f(
                unsafe { self.provider.aux_vectors.get_vector_sync(i.into_usize()) },
                i,
            )
        }
        std::future::ready(Ok(()))
    }
}

impl<T, V, D, Ctx> BuildQueryComputer<[T]> for QuantAccessor<'_, V, D, Ctx>
where
    T: VectorRepr,
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type QueryComputerError = ANNError;
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;

    fn build_query_computer(
        &self,
        from: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.provider.aux_vectors.query_computer(from)
    }
}

impl<V, D, Ctx> BuildDistanceComputer for QuantAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type DistanceComputerError = ANNError;
    type DistanceComputer = pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(self.provider.aux_vectors.distance_computer()?)
    }
}

impl<T, V, D, Ctx> ExpandBeam<[T]> for QuantAccessor<'_, V, D, Ctx>
where
    T: VectorRepr,
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
}

impl<V, D, Ctx> FillSet for QuantAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
}

//-------------------//
// In-mem Extensions //
//-------------------//

impl<'a, V, D, Ctx> AsDeletionCheck for QuantAccessor<'a, V, D, Ctx>
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

/////////////////////
// Hybrid Accessor //
/////////////////////

/// A hybrid accessor that fetches a mixture of full-precision and quantized vectors during
/// pruning. This allows the application to trade full-precision fetches for accuracy.
///
/// This type implements the following traits:
///
/// * [`Accessor`] for the [`DefaultProvider`].
/// * [`BuildDistanceComputer`] for computing distances among [`distances::pq::Hybrid`]
///   element types.
/// * [`FillSet`] for populating a mixture of full-precision and quant vectors.
pub struct HybridAccessor<'a, T, D, Ctx>
where
    T: VectorRepr,
{
    provider: &'a FullPrecisionProvider<T, DefaultQuant, D, Ctx>,

    /// Maximum number of full-precision vectors to use during pruning.
    /// This field is ignored during search, where full-precision vectors are never used.
    max_fp_vecs_per_prune: usize,
}

impl<'a, T, D, Ctx> HybridAccessor<'a, T, D, Ctx>
where
    T: VectorRepr,
{
    fn new(
        provider: &'a FullPrecisionProvider<T, DefaultQuant, D, Ctx>,
        max_fp_vecs_per_prune: usize,
    ) -> Self {
        Self {
            provider,
            max_fp_vecs_per_prune,
        }
    }
}

impl<T, D, Ctx> HasId for HybridAccessor<'_, T, D, Ctx>
where
    T: VectorRepr,
{
    type Id = u32;
}

impl<'a, T, D, Ctx> DelegateNeighbor<'a> for HybridAccessor<'_, T, D, Ctx>
where
    T: VectorRepr,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type Delegate = &'a SimpleNeighborProviderAsync<u32>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<'a, T, D, Ctx> Accessor for HybridAccessor<'a, T, D, Ctx>
where
    T: VectorRepr,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    /// The extended element inherets the lifetime of the Accessor.
    type Extended = distances::pq::Hybrid<&'a [T], &'a [u8]>;

    /// The [`distances::pq::Hybrid`] is an enum consisting of either a full-precision
    /// vector or a quantized vector.
    ///
    /// This accessor can return either.
    ///
    /// NOTE: We intentionally don't use `'b` here since our implementation borrows
    /// the inner `CVRef` from the underlying provider.
    type Element<'b>
        = distances::pq::Hybrid<&'a [T], &'a [u8]>
    where
        Self: 'b;

    /// `ElementRef` has an arbitrarily short lifetime.
    type ElementRef<'b> = distances::pq::Hybrid<&'b [T], &'b [u8]>;

    /// Choose to panic on an out-of-bounds access rather than propagate an error.
    type GetError = Panics;

    /// The default behavior of `get_element` returns a full-precision vector. The
    /// implementation of [`FillSet`] is how the `max_fp_vecs_per_fill` is used.
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        // SAFETY: We've decided to live with UB that can result from potentially mixing
        // unsynchronized reads and writes on the underlying memory.
        std::future::ready(Ok(unsafe {
            distances::pq::Hybrid::Full(self.provider.base_vectors.get_vector_sync(id.into_usize()))
        }))
    }
}

impl<T, D, Ctx> BuildDistanceComputer for HybridAccessor<'_, T, D, Ctx>
where
    T: VectorRepr,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type DistanceComputerError = ANNError;
    type DistanceComputer = distances::pq::HybridComputer<T>;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        let metric = self.provider.aux_vectors.metric();
        Ok(distances::pq::HybridComputer::new(
            self.provider.aux_vectors.distance_computer()?,
            T::distance(metric, Some(self.provider.base_vectors.dim())),
        ))
    }
}

impl<T, D, Ctx> FillSet for HybridAccessor<'_, T, D, Ctx>
where
    T: VectorRepr,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    /// Fill up to `max_fp_per_prune` as full precision. The rest are quantized.
    ///
    /// if a full-precision vector already exists regardless of whether a full-precision
    /// vector or quant vector is needed, it is left as-is.
    async fn fill_set<Itr>(
        &mut self,
        set: &mut HashMap<Self::Id, Self::Extended>,
        itr: Itr,
    ) -> Result<(), Self::GetError>
    where
        Itr: Iterator<Item = Self::Id> + Send + Sync,
    {
        let threshold = self.max_fp_vecs_per_prune;
        itr.enumerate().for_each(|(i, id)| {
            let e = set.entry(id);
            // Below the threshold, we try to fetch full-precision vectors.
            if i < threshold {
                // If the item already exists but is not full precision, make it full
                // precision.
                e.and_modify(|v| {
                    if !v.is_full() {
                        // SAFETY: We've decided to live with UB (undefined behavior) that
                        // can result from potentially mixing unsynchronized reads and
                        // writes on the underlying memory.
                        *v = distances::pq::Hybrid::Full(unsafe {
                            self.provider.base_vectors.get_vector_sync(id.into_usize())
                        });
                    }
                })
                .or_insert_with(|| {
                    // Only invoke this method if the entry is not occupied.
                    //
                    // SAFETY: We've decided to live with UB (undefined behavior) that
                    // can result from potentially mixing unsynchronized reads and
                    // writes on the underlying memory.
                    distances::pq::Hybrid::Full(unsafe {
                        self.provider.base_vectors.get_vector_sync(id.into_usize())
                    })
                });
            } else {
                // Otherwise, only insert into the cache if the entry is not occupied.
                e.or_insert_with(|| {
                    // SAFETY: We've decided to live with UB (undefined behavior) that
                    // can result from potentially mixing unsynchronized reads and
                    // writes on the underlying memory.
                    distances::pq::Hybrid::Quant(unsafe {
                        self.provider.aux_vectors.get_vector_sync(id.into_usize())
                    })
                });
            }
        });
        Ok(())
    }
}

////////////////
// Strategies //
////////////////

////////////
// Hybrid //
////////////

/// Perform a search entirely in the quantized space.
///
/// Starting points are not filtered out of the final results but results are reranked using
/// the full-precision data.
impl<T, D, Ctx> SearchStrategy<FullPrecisionProvider<T, DefaultQuant, D, Ctx>, [T]>
    for Internal<Hybrid>
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;
    type SearchAccessor<'a> = QuantAccessor<'a, FullPrecisionStore<T>, D, Ctx>;
    type SearchAccessorError = Panics;
    type PostProcessor = Rerank;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

/// Perform a search entirely in the quantized space.
///
/// Starting points are filtered out of the final results and results are reranked using
/// the full-precision data.
impl<T, D, Ctx> SearchStrategy<FullPrecisionProvider<T, DefaultQuant, D, Ctx>, [T]> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;
    type SearchAccessor<'a> = QuantAccessor<'a, FullPrecisionStore<T>, D, Ctx>;
    type SearchAccessorError = Panics;
    type PostProcessor = glue::Pipeline<FilterStartPoints, Rerank>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<T, D, Ctx> PruneStrategy<FullPrecisionProvider<T, DefaultQuant, D, Ctx>> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type DistanceComputer = distances::pq::HybridComputer<T>;
    type PruneAccessor<'a> = HybridAccessor<'a, T, D, Ctx>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(HybridAccessor::new(
            provider,
            self.max_fp_vecs_per_prune.unwrap_or(usize::MAX),
        ))
    }
}

/// Implementing this trait allows `Quantized` to be used for multi-insert.
impl<'a, T, D, Ctx> glue::AsElement<&'a [T]> for HybridAccessor<'a, T, D, Ctx>
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type Error = diskann::error::Infallible;
    fn as_element(
        &mut self,
        vector: &'a [T],
        _id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'a>, Self::Error>> + Send {
        std::future::ready(Ok(distances::pq::Hybrid::Full(vector)))
    }
}

impl<T, D, Ctx> InsertStrategy<FullPrecisionProvider<T, DefaultQuant, D, Ctx>, [T]> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

/// Inplace Delete
impl<T, D, Ctx> InplaceDeleteStrategy<FullPrecisionProvider<T, DefaultQuant, D, Ctx>> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type DeleteElementError = Panics;
    type DeleteElement<'a> = [T];
    type DeleteElementGuard = Box<[T]>;
    type PruneStrategy = Self;
    type SearchStrategy = Internal<Self>;
    fn search_strategy(&self) -> Self::SearchStrategy {
        Internal(*self)
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    async fn get_delete_element<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
        id: u32,
    ) -> Result<Self::DeleteElementGuard, Self::DeleteElementError> {
        Ok(unsafe { provider.base_vectors.get_vector_sync(id.into_usize()) }.into())
    }
}

///////////////
// Quantized //
///////////////

/// Perform a search entirely in the quantized space.
///
/// Starting points are filtered out of the final results.
impl<T, D, Ctx> SearchStrategy<DefaultProvider<NoStore, DefaultQuant, D, Ctx>, [T]> for Quantized
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;
    type SearchAccessor<'a> = QuantAccessor<'a, NoStore, D, Ctx>;
    type SearchAccessorError = Panics;
    type PostProcessor = glue::Pipeline<FilterStartPoints, RemoveDeletedIdsAndCopy>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a DefaultProvider<NoStore, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

impl<D, Ctx> PruneStrategy<DefaultProvider<NoStore, DefaultQuant, D, Ctx>> for Quantized
where
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type DistanceComputer = pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>;
    type PruneAccessor<'a> = QuantAccessor<'a, NoStore, D, Ctx>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DefaultProvider<NoStore, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(QuantAccessor::new(provider))
    }
}

impl<T, D, Ctx> InsertStrategy<DefaultProvider<NoStore, DefaultQuant, D, Ctx>, [T]> for Quantized
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}
