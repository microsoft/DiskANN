/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, future::Future};

use diskann::default_post_processor;
use diskann::{
    ANNError, ANNResult,
    error::IntoANNResult,
    graph::{
        AdjacencyList, SearchOutputBuffer,
        glue::{
            self, DefaultPostProcessor, InplaceDeleteStrategy, InsertStrategy, PruneStrategy,
            SearchStrategy,
        },
        workingset,
    },
    neighbor::Neighbor,
    provider::{DefaultContext, ExecutionContext, HasId},
    utils::{IntoUsize, VectorRepr},
};

use diskann_utils::future::AsyncFriendly;
use diskann_utils::views::Matrix;
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};

use crate::model::graph::provider::async_::{
    FastMemoryVectorProviderAsync, SimpleNeighborProviderAsync,
    common::{
        CreateVectorStore, FlatVectorAccess, FullPrecision, NoDeletes, NoStore, Panics,
        PrefetchCacheLineLevel, SetElementHelper,
    },
    inmem::DefaultProvider,
    postprocess::{AsDeletionCheck, DeletionCheck, RemoveDeletedIdsAndCopy},
};
use crate::model::graph::provider::{DeterminantDiversityParams, determinant_diversity};

/// A type alias for the DefaultProvider with full-precision as the primary vector store.
pub type FullPrecisionProvider<T, Q = NoStore, D = NoDeletes, Ctx = DefaultContext> =
    DefaultProvider<FullPrecisionStore<T>, Q, D, Ctx>;

/// The default full-precision vector store.
pub type FullPrecisionStore<T> = FastMemoryVectorProviderAsync<T>;

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

impl<T> FlatVectorAccess<T> for FullPrecisionStore<T>
where
    T: VectorRepr,
{
    unsafe fn flat_prefix(&self, first_n: usize) -> &[T] {
        unsafe { FastMemoryVectorProviderAsync::<T>::flat_prefix(self, first_n) }
    }
}

///////////////////
// PruneAccessor //
///////////////////

pub struct PruneAccessor<'a, T>
where
    T: VectorRepr,
{
    store: &'a FastMemoryVectorProviderAsync<T>,
    neighbors: &'a SimpleNeighborProviderAsync,
    distance: <T as VectorRepr>::Distance,
}

impl<T> HasId for PruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type Id = u32;
}

impl<T> glue::PruneAccessor for PruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type ElementRef<'a> = &'a [T];

    type View<'a>
        = &'a Self
    where
        Self: 'a;

    type Distance<'a>
        = &'a <T as VectorRepr>::Distance
    where
        Self: 'a;

    type Neighbors<'a>
        = &'a SimpleNeighborProviderAsync
    where
        Self: 'a;

    async fn fill<Itr>(&mut self, _itr: Itr) -> ANNResult<(Self::View<'_>, Self::Distance<'_>)>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        Ok((self, &self.distance))
    }

    fn neighbors(&mut self) -> Self::Neighbors<'_> {
        self.neighbors
    }
}

// Pass-through view.
impl<T> workingset::View<u32> for &PruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type ElementRef<'a> = &'a [T];
    type Element<'a>
        = &'a [T]
    where
        Self: 'a;

    fn get(&self, id: u32) -> Option<&[T]> {
        // SAFETY: This is unsound. We assume no concurrent writes to this slot, but
        // this invariant is not enforced. See `get_vector_sync` for details
        Some(unsafe { self.store.get_vector_sync(id.into_usize()) })
    }
}

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

    /// The distance computer.
    computer: T::QueryDistance,

    /// A buffer for resolving iterators given during bulk operations.
    ///
    /// The accessor reuses this allocation to amortize allocation cost over multiple bulk
    /// operations.
    id_buffer: AdjacencyList<u32>,
}

impl<'a, T, Q, D, Ctx> FullAccessor<'a, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    pub fn new(provider: &'a FullPrecisionProvider<T, Q, D, Ctx>, query: &[T]) -> Self {
        Self {
            provider,
            computer: T::query_distance(query, provider.metric),
            id_buffer: AdjacencyList::new(),
        }
    }
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

impl<T, Q, D, Ctx> glue::SearchAccessor for FullAccessor<'_, T, Q, D, Ctx>
where
    T: VectorRepr,
    Q: AsyncFriendly,
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
                    self.provider.base_vectors.get_vector_sync(i.into_usize())
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
            let id_buffer = &mut self.id_buffer;
            for n in ids {
                self.provider
                    .neighbor_provider
                    .get_neighbors_sync(n.into_usize(), id_buffer)?;

                id_buffer.retain(|i| pred.eval_mut(i));
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
                    let v = unsafe { self.provider.base_vectors.get_vector_sync(id.into_usize()) };
                    let distance = self.computer.evaluate_similarity(v);
                    on_neighbors(*id, distance);
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
    fn as_full_precision(&self) -> &FastMemoryVectorProviderAsync<Self::Repr>;
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
    A: HasId<Id = u32> + GetFullPrecision<Repr = T> + AsDeletionCheck,
{
    type Error = Panics;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: &'a [T],
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

/// A [`SearchPostProcess`]or that reranks a full-precision candidate stream using the
/// Determinant-Diversity algorithm, reordering results to promote geometric diversity
/// while preserving relevance to the query.
#[derive(Debug, Clone, Copy)]
pub struct DeterminantDiversity {
    params: DeterminantDiversityParams,
}

impl DeterminantDiversity {
    /// Construct a new [`DeterminantDiversity`] post-processor with the given parameters.
    pub const fn new(params: DeterminantDiversityParams) -> Self {
        Self { params }
    }
}

impl<'a, A> glue::SearchPostProcess<A, &'a [f32], A::Id> for DeterminantDiversity
where
    A: HasId<Id = u32> + GetFullPrecision<Repr = f32> + Send + Sync,
{
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: &'a [f32],
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<A::Id> + Send + ?Sized,
    {
        let candidates: Vec<Neighbor<A::Id>> = candidates.collect();
        let candidate_count = candidates.len();
        let store: &FullPrecisionStore<f32> = accessor.as_full_precision();
        let mut vectors = Matrix::new(0.0f32, candidate_count, query.len());
        let mut ids = Vec::with_capacity(candidate_count);
        let mut distances = Vec::with_capacity(candidate_count);

        for (i, candidate) in candidates.into_iter().enumerate() {
            // SAFETY: We accept potential unsynchronized concurrent mutation, matching the
            // pattern used by `Rerank` above.
            let vector = unsafe { store.get_vector_sync(candidate.id.into_usize()) };
            ids.push(candidate.id);
            distances.push(candidate.distance);
            vectors.row_mut(i).copy_from_slice(vector);
        }

        let indices = match determinant_diversity(
            vectors.as_mut_view(),
            &distances,
            query,
            candidate_count,
            &self.params,
        ) {
            Ok(indices) => indices,
            Err(error) => return std::future::ready(Err(error.into())),
        };

        let reranked = indices.into_iter().map(|idx| (ids[idx], distances[idx]));

        std::future::ready(Ok(output.extend(reranked)))
    }
}

////////////////
// Strategies //
////////////////

/// Perform a search entirely in the full-precision space.
impl<'a, T, Q, D, Ctx> SearchStrategy<'a, FullPrecisionProvider<T, Q, D, Ctx>, &'a [T]>
    for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
    Ctx: ExecutionContext,
{
    type SearchAccessor = FullAccessor<'a, T, Q, D, Ctx>;
    type SearchAccessorError = Panics;

    fn search_accessor(
        &'a self,
        provider: &'a FullPrecisionProvider<T, Q, D, Ctx>,
        _context: &'a Ctx,
        query: &'a [T],
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider, query))
    }
}

impl<'a, T, Q, D, Ctx> DefaultPostProcessor<'a, FullPrecisionProvider<T, Q, D, Ctx>, &'a [T]>
    for FullPrecision
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
    type PruneAccessor<'a> = PruneAccessor<'a, T>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a FullPrecisionProvider<T, Q, D, Ctx>,
        _context: &'a Ctx,
        _capacity: usize,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        let accessor = PruneAccessor {
            store: &provider.base_vectors,
            neighbors: provider.neighbors(),
            distance: T::distance(provider.metric, Some(provider.base_vectors.dim())),
        };
        Ok(accessor)
    }
}

impl<'a, T, Q, D, Ctx> InsertStrategy<'a, FullPrecisionProvider<T, Q, D, Ctx>, &'a [T]>
    for FullPrecision
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
            'a,
            FullPrecisionProvider<T, Q, D, Ctx>,
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
        _provider: &FullPrecisionProvider<T, Q, D, Ctx>,
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
        provider: &'a FullPrecisionProvider<T, Q, D, Ctx>,
        context: &'a Ctx,
        _seed: &'a (),
        capacity: usize,
    ) -> ANNResult<PruneAccessor<'a, T>> {
        self.prune_accessor(provider, context, capacity)
            .into_ann_result()
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
        Ok(unsafe { provider.base_vectors.get_vector_sync(id.into_usize()) }.into())
    }
}
