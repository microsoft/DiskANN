/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{future::Future, sync::Arc};

use diskann::default_post_processor;
use diskann::{
    ANNError, ANNResult,
    error::IntoANNResult,
    graph::{
        AdjacencyList,
        glue::{
            self, DefaultPostProcessor, InplaceDeleteStrategy, InsertStrategy, PruneStrategy,
            SearchStrategy,
        },
        workingset,
    },
    provider::{ExecutionContext, HasId},
    utils::{IntoUsize, VectorRepr},
};
use diskann_utils::future::AsyncFriendly;
use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};

use crate::model::{
    graph::provider::async_::{
        FastMemoryQuantVectorProviderAsync, FastMemoryVectorProviderAsync,
        SimpleNeighborProviderAsync,
        common::{
            CreateVectorStore, Hybrid, NoStore, Panics, Quantized, SetElementHelper, Unseeded,
            VectorStore,
        },
        distances,
        inmem::{
            DefaultProvider, FullPrecisionProvider, FullPrecisionStore, GetFullPrecision,
            PassThrough, Rerank,
        },
        postprocess::{AsDeletionCheck, DeletionCheck, RemoveDeletedIdsAndCopy},
    },
    pq::{self, FixedChunkPQTable},
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
// PruneAccessor //
///////////////////

#[derive(Clone, Copy)]
pub struct PruneAccessor<'a> {
    provider: &'a FastMemoryQuantVectorProviderAsync,
    neighbors: &'a SimpleNeighborProviderAsync<u32>,
}

impl HasId for PruneAccessor<'_> {
    type Id = u32;
}

impl HasElementRef for PruneAccessor<'_> {
    type ElementRef<'a> = &'a [u8];
}

impl<'a> DelegateNeighbor<'a> for PruneAccessor<'_> {
    type Delegate = &'a SimpleNeighborProviderAsync<u32>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.neighbors
    }
}

impl BuildDistanceComputer for PruneAccessor<'_> {
    type DistanceComputerError = ANNError;
    type DistanceComputer = pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(self.provider.distance_computer())
    }
}

impl workingset::Fill<PassThrough> for PruneAccessor<'_> {
    type Error = std::convert::Infallible;
    type View<'a>
        = Self
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
        Ok(*self)
    }
}

// Pass-through view — reads PQ codes directly from the provider.
impl workingset::View<u32> for PruneAccessor<'_> {
    type ElementRef<'a> = &'a [u8];
    type Element<'a>
        = &'a [u8]
    where
        Self: 'a;

    fn get(&self, id: u32) -> Option<&[u8]> {
        // SAFETY: This is unsound. We assume no concurrent writes to this slot, but
        // this invariant is not enforced. See `get_vector_sync` for details.
        Some(unsafe { self.provider.get_vector_sync(id.into_usize()) })
    }
}

/////////////////////////
// HybridPruneAccessor //
/////////////////////////

#[derive(Clone)]
pub struct HybridPruneAccessor<'a, T>
where
    T: VectorRepr,
{
    full: &'a FastMemoryVectorProviderAsync<T>,
    quant: &'a FastMemoryQuantVectorProviderAsync,
    neighbors: &'a SimpleNeighborProviderAsync<u32>,
    max_fp_vecs_per_prune: usize,
}

impl<T> HasId for HybridPruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type Id = u32;
}

impl<T> HasElementRef for HybridPruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type ElementRef<'a> = distances::pq::Hybrid<&'a [T], &'a [u8]>;
}

impl<'a, T> DelegateNeighbor<'a> for HybridPruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type Delegate = &'a SimpleNeighborProviderAsync<u32>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.neighbors
    }
}

impl<T> BuildDistanceComputer for HybridPruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type DistanceComputerError = ANNError;
    type DistanceComputer = distances::pq::HybridComputer<T>;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        let metric = self.quant.metric();
        Ok(distances::pq::HybridComputer::new(
            self.quant.distance_computer(),
            T::distance(metric, Some(self.full.dim())),
        ))
    }
}

/// Tracks which IDs should use full-precision vectors during hybrid pruning.
///
/// IDs in the set get full-precision distance computations; all others fall back to
/// quantized vectors.
pub struct FullPrecisionTracker(hashbrown::HashSet<u32>);

impl workingset::AsWorkingSet<FullPrecisionTracker> for Unseeded {
    fn as_working_set(&self, capacity: usize) -> FullPrecisionTracker {
        FullPrecisionTracker(hashbrown::HashSet::with_capacity(capacity))
    }
}

// Selective fill — the first `max_fp_vecs_per_prune` candidates receive full-precision
// vectors; the remainder are accessed through the pass-through `MaybeFullPrecision` view.
impl<T> workingset::Fill<FullPrecisionTracker> for HybridPruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type Error = std::convert::Infallible;
    type View<'a>
        = MaybeFullPrecision<'a, T>
    where
        Self: 'a;

    async fn fill<'a, Itr>(
        &'a mut self,
        state: &'a mut FullPrecisionTracker,
        itr: Itr,
    ) -> Result<Self::View<'a>, Self::Error>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
        Self: 'a,
    {
        state.0.clear();
        state.0.extend(itr.take(self.max_fp_vecs_per_prune));
        Ok(MaybeFullPrecision {
            builder: self,
            full: state,
        })
    }
}

pub struct MaybeFullPrecision<'a, T>
where
    T: VectorRepr,
{
    builder: &'a HybridPruneAccessor<'a, T>,
    full: &'a FullPrecisionTracker,
}

impl<T> workingset::View<u32> for MaybeFullPrecision<'_, T>
where
    T: VectorRepr,
{
    type ElementRef<'a> = distances::pq::Hybrid<&'a [T], &'a [u8]>;
    type Element<'a>
        = distances::pq::Hybrid<&'a [T], &'a [u8]>
    where
        Self: 'a;

    fn get(&self, id: u32) -> Option<Self::Element<'_>> {
        let element = if self.full.0.contains(&id) {
            // SAFETY: This is unsound. We assume no concurrent writes to this slot, but
            // this invariant is not enforced. See `get_vector_sync` for details.
            unsafe {
                distances::pq::Hybrid::Full(self.builder.full.get_vector_sync(id.into_usize()))
            }
        } else {
            // SAFETY: This is unsound. We assume no concurrent writes to this slot, but
            // this invariant is not enforced. See `get_vector_sync` for details.
            unsafe {
                distances::pq::Hybrid::Quant(self.builder.quant.get_vector_sync(id.into_usize()))
            }
        };
        Some(element)
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
    computer: pq::distance::QueryComputer<Arc<FixedChunkPQTable>>,
}

impl<'a, V, D, Ctx> QuantAccessor<'a, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    pub(crate) fn new(
        provider: &'a DefaultProvider<V, DefaultQuant, D, Ctx>,
        query: &[f32],
    ) -> ANNResult<Self> {
        let computer = provider.aux_vectors.query_computer(query)?;
        Ok(Self { provider, computer })
    }
}

impl<T, D, Ctx> GetFullPrecision for QuantAccessor<'_, FullPrecisionStore<T>, D, Ctx>
where
    T: VectorRepr,
{
    type Repr = T;
    fn as_full_precision(&self) -> &FastMemoryVectorProviderAsync<T> {
        &self.provider.base_vectors
    }
}

impl<V, D, Ctx> HasId for QuantAccessor<'_, V, D, Ctx> {
    type Id = u32;
}

impl<V, D, Ctx> glue::SearchAccessor for QuantAccessor<'_, V, D, Ctx>
where
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }

    fn num_starting_points(&self) -> impl Future<Output = ANNResult<usize>> {
        std::future::ready(Ok(self.provider.num_start_points()))
    }

    fn start_point_distances<F>(
        &mut self,
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let mut f = move || -> ANNResult<()> {
            for i in self.provider.starting_points()? {
                // SAFETY: We're accepting the consequences of potential unsynchronized,
                // concurrent mutation.
                let distance = self.computer.evaluate_similarity(unsafe {
                    self.provider.aux_vectors.get_vector_sync(i.into_usize())
                });

                f(i, distance);
            }
            Ok(())
        };

        std::future::ready(f())
    }

    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        let f = move || -> ANNResult<()> {
            let mut neighbors = AdjacencyList::new();
            for n in ids {
                self.provider
                    .neighbor_provider
                    .get_neighbors_sync(n.into_usize(), &mut neighbors)?;
                for i in neighbors.iter().filter(|i| pred.eval_mut(i)) {
                    // SAFETY: We're accepting the consequences of potential unsynchronized,
                    // concurrent mutation.
                    let distance = self.computer.evaluate_similarity(unsafe {
                        self.provider.aux_vectors.get_vector_sync(i.into_usize())
                    });

                    on_neighbors(*i, distance);
                }
            }
            Ok(())
        };

        std::future::ready(f())
    }
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

////////////////
// Strategies //
////////////////

////////////
// Hybrid //
////////////

/// Perform a search entirely in the quantized space.
impl<'a, T, D, Ctx> SearchStrategy<'a, FullPrecisionProvider<T, DefaultQuant, D, Ctx>, &'a [T]>
    for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type SearchAccessor = QuantAccessor<'a, FullPrecisionStore<T>, D, Ctx>;
    type SearchAccessorError = ANNError;

    fn search_accessor(
        &'a self,
        provider: &'a FullPrecisionProvider<T, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
        query: &'a [T],
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        let as_f32 = T::as_f32(query).into_ann_result()?;
        QuantAccessor::new(provider, &as_f32)
    }
}

/// Starting points are filtered out of the final results and results are reranked using
/// the full-precision data.
impl<'a, T, D, Ctx>
    DefaultPostProcessor<'a, FullPrecisionProvider<T, DefaultQuant, D, Ctx>, &'a [T]> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    default_post_processor!(glue::Pipeline<glue::FilterStartPoints, Rerank>);
}

impl<T, D, Ctx> PruneStrategy<FullPrecisionProvider<T, DefaultQuant, D, Ctx>> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type DistanceComputer<'a> = distances::pq::HybridComputer<T>;
    type PruneAccessor<'a> = HybridPruneAccessor<'a, T>;
    type PruneAccessorError = diskann::error::Infallible;
    type WorkingSet = FullPrecisionTracker;

    fn create_working_set(&self, capacity: usize) -> Self::WorkingSet {
        FullPrecisionTracker(hashbrown::HashSet::with_capacity(capacity))
    }

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        let accessor = HybridPruneAccessor {
            full: &provider.base_vectors,
            quant: &provider.aux_vectors,
            neighbors: provider.neighbors(),
            max_fp_vecs_per_prune: self.max_fp_vecs_per_prune.unwrap_or(usize::MAX),
        };
        Ok(accessor)
    }
}

impl<'a, T, D, Ctx> InsertStrategy<'a, FullPrecisionProvider<T, DefaultQuant, D, Ctx>, &'a [T]>
    for Hybrid
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

impl<T, D, Ctx, B> glue::MultiInsertStrategy<FullPrecisionProvider<T, DefaultQuant, D, Ctx>, B>
    for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
    B: glue::Batch,
    Self: for<'a> InsertStrategy<
            'a,
            FullPrecisionProvider<T, DefaultQuant, D, Ctx>,
            B::Element<'a>,
            PruneStrategy = Self,
        >,
{
    type Seed = Unseeded;
    type WorkingSet = FullPrecisionTracker;
    type FinishError = diskann::error::Infallible;
    type InsertStrategy = Self;

    fn insert_strategy(&self) -> Self::InsertStrategy {
        *self
    }

    fn finish<Itr>(
        &self,
        _provider: &FullPrecisionProvider<T, DefaultQuant, D, Ctx>,
        _ctx: &Ctx,
        _batch: &std::sync::Arc<B>,
        _ids: Itr,
    ) -> impl std::future::Future<Output = Result<Self::Seed, Self::FinishError>> + Send
    where
        Itr: ExactSizeIterator<Item = u32> + Send,
    {
        std::future::ready(Ok(Unseeded))
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
    type DeleteElement<'a> = &'a [T];
    type DeleteElementGuard = Box<[T]>;
    type PruneStrategy = Self;
    type DeleteSearchAccessor<'a> = QuantAccessor<'a, FullPrecisionStore<T>, D, Ctx>;
    type SearchPostProcessor = Rerank;
    type SearchStrategy = Self;
    fn search_strategy(&self) -> Self::SearchStrategy {
        *self
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn search_post_processor(&self) -> Self::SearchPostProcessor {
        Rerank
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
impl<'a, T, D, Ctx> SearchStrategy<'a, DefaultProvider<NoStore, DefaultQuant, D, Ctx>, &'a [T]>
    for Quantized
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type SearchAccessor = QuantAccessor<'a, NoStore, D, Ctx>;
    type SearchAccessorError = ANNError;

    fn search_accessor(
        &'a self,
        provider: &'a DefaultProvider<NoStore, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
        query: &[T],
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        let as_f32 = T::as_f32(query).into_ann_result()?;
        QuantAccessor::new(provider, &as_f32)
    }
}

impl<'a, T, D, Ctx>
    DefaultPostProcessor<'a, DefaultProvider<NoStore, DefaultQuant, D, Ctx>, &'a [T]> for Quantized
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    default_post_processor!(glue::Pipeline<glue::FilterStartPoints, RemoveDeletedIdsAndCopy>);
}

impl<D, Ctx> PruneStrategy<DefaultProvider<NoStore, DefaultQuant, D, Ctx>> for Quantized
where
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type DistanceComputer<'a> = pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>;
    type PruneAccessor<'a> = PruneAccessor<'a>;
    type PruneAccessorError = diskann::error::Infallible;
    type WorkingSet = PassThrough;

    fn create_working_set(&self, _capacity: usize) -> Self::WorkingSet {
        PassThrough
    }

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DefaultProvider<NoStore, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        let accessor = PruneAccessor {
            provider: &provider.aux_vectors,
            neighbors: provider.neighbors(),
        };
        Ok(accessor)
    }
}

impl<'a, T, D, Ctx> InsertStrategy<'a, DefaultProvider<NoStore, DefaultQuant, D, Ctx>, &'a [T]>
    for Quantized
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

impl<D, Ctx, B> glue::MultiInsertStrategy<DefaultProvider<NoStore, DefaultQuant, D, Ctx>, B>
    for Quantized
where
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
    B: glue::Batch,
    Self: for<'a> InsertStrategy<
            'a,
            DefaultProvider<NoStore, DefaultQuant, D, Ctx>,
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
        _provider: &DefaultProvider<NoStore, DefaultQuant, D, Ctx>,
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
