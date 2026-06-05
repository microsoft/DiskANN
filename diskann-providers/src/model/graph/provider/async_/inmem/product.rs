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
            CreateVectorStore, Hybrid, NoStore, Panics, Quantized, SetElementHelper, VectorStore,
        },
        distances,
        inmem::{
            DefaultProvider, FullPrecisionProvider, FullPrecisionStore, GetFullPrecision, Rerank,
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

pub struct PruneAccessor<'a> {
    provider: &'a FastMemoryQuantVectorProviderAsync,
    neighbors: &'a SimpleNeighborProviderAsync<u32>,
    distance: pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>,
}

impl HasId for PruneAccessor<'_> {
    type Id = u32;
}

impl glue::PruneAccessor for PruneAccessor<'_> {
    type ElementRef<'a> = &'a [u8];

    type View<'a>
        = &'a Self
    where
        Self: 'a;

    type Distance<'a>
        = &'a pq::distance::DistanceComputer<Arc<FixedChunkPQTable>>
    where
        Self: 'a;

    type Neighbors<'a>
        = &'a SimpleNeighborProviderAsync<u32>
    where
        Self: 'a;

    async fn fill<'a, Itr>(
        &'a mut self,
        _itr: Itr,
    ) -> ANNResult<(Self::View<'a>, Self::Distance<'a>)>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        Ok((self, &self.distance))
    }

    fn neighbors(&mut self) -> Self::Neighbors<'_> {
        self.neighbors
    }
}

// Pass-through view — reads PQ codes directly from the provider.
impl workingset::View<u32> for &PruneAccessor<'_> {
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

pub struct HybridPruneAccessor<'a, T>
where
    T: VectorRepr,
{
    full: &'a FastMemoryVectorProviderAsync<T>,
    quant: &'a FastMemoryQuantVectorProviderAsync,
    neighbors: &'a SimpleNeighborProviderAsync<u32>,
    distance: distances::pq::HybridComputer<T>,

    // During pruning, we make the first `max_fp_vecs_per_prune` are full-precision with
    // the rest being quantized. This hash set records which IDs should be full-precision.
    full_precision_ids: hashbrown::HashSet<u32>,
    max_fp_vecs_per_prune: usize,
}

impl<T> HasId for HybridPruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type Id = u32;
}

impl<T> glue::PruneAccessor for HybridPruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type ElementRef<'a> = distances::pq::Hybrid<&'a [T], &'a [u8]>;
    type View<'a>
        = &'a Self
    where
        Self: 'a;
    type Distance<'a>
        = &'a distances::pq::HybridComputer<T>
    where
        Self: 'a;
    type Neighbors<'a>
        = &'a SimpleNeighborProviderAsync<u32>
    where
        Self: 'a;

    async fn fill<'a, Itr>(
        &'a mut self,
        itr: Itr,
    ) -> ANNResult<(Self::View<'a>, Self::Distance<'a>)>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        self.full_precision_ids.clear();
        self.full_precision_ids
            .extend(itr.take(self.max_fp_vecs_per_prune));
        Ok((self, &self.distance))
    }

    fn neighbors(&mut self) -> Self::Neighbors<'_> {
        self.neighbors
    }
}

impl<T> workingset::View<u32> for &HybridPruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type ElementRef<'a> = distances::pq::Hybrid<&'a [T], &'a [u8]>;
    type Element<'a>
        = distances::pq::Hybrid<&'a [T], &'a [u8]>
    where
        Self: 'a;

    fn get(&self, id: u32) -> Option<Self::Element<'_>> {
        let element = if self.full_precision_ids.contains(&id) {
            // SAFETY: This is unsound. We assume no concurrent writes to this slot, but
            // this invariant is not enforced. See `get_vector_sync` for details.
            unsafe { distances::pq::Hybrid::Full(self.full.get_vector_sync(id.into_usize())) }
        } else {
            // SAFETY: This is unsound. We assume no concurrent writes to this slot, but
            // this invariant is not enforced. See `get_vector_sync` for details.
            unsafe { distances::pq::Hybrid::Quant(self.quant.get_vector_sync(id.into_usize())) }
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
    type PruneAccessor<'a> = HybridPruneAccessor<'a, T>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
        capacity: usize,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        let full = &provider.base_vectors;
        let quant = &provider.aux_vectors;
        let metric = quant.metric();

        let distance = distances::pq::HybridComputer::new(
            quant.distance_computer(),
            T::distance(metric, Some(full.dim())),
        );

        let accessor = HybridPruneAccessor {
            full: &provider.base_vectors,
            quant: &provider.aux_vectors,
            neighbors: provider.neighbors(),
            distance,
            full_precision_ids: hashbrown::HashSet::with_capacity(capacity),
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
    type Seed = ();
    type FinishError = diskann::error::Infallible;
    type PruneStrategy = Self;
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
        std::future::ready(Ok(()))
    }

    fn seeded_prune_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, DefaultQuant, D, Ctx>,
        context: &'a Ctx,
        _seed: &'a (),
        capacity: usize,
    ) -> ANNResult<
        <Self as PruneStrategy<FullPrecisionProvider<T, DefaultQuant, D, Ctx>>>::PruneAccessor<'a>,
    > {
        Ok(self.prune_accessor(provider, context, capacity)?)
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
    type PruneAccessor<'a> = PruneAccessor<'a>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a DefaultProvider<NoStore, DefaultQuant, D, Ctx>,
        _context: &'a Ctx,
        _capacity: usize,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        let accessor = PruneAccessor {
            provider: &provider.aux_vectors,
            neighbors: provider.neighbors(),
            distance: provider.aux_vectors.distance_computer(),
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
    type Seed = ();
    type FinishError = diskann::error::Infallible;
    type PruneStrategy = Self;
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
        std::future::ready(Ok(()))
    }

    fn seeded_prune_accessor<'a>(
        &'a self,
        provider: &'a DefaultProvider<NoStore, DefaultQuant, D, Ctx>,
        context: &'a Ctx,
        _seed: &'a (),
        capacity: usize,
    ) -> ANNResult<PruneAccessor<'a>> {
        self.prune_accessor(provider, context, capacity)
            .into_ann_result()
    }
}
